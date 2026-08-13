#!/usr/bin/env python3
"""Train three cooperating agents (combat / card-select / path) by REINFORCE on
full Ascension-10 runs played through the CLI, and track COMBAT WIN RATE.

Rewards:
  combat agent : HP retained this combat = end_hp / start_hp  (>=1.0 = perfect win)
  card + path  : overall game victory (+ small act/floor progress shaping) so
                 their picks/routing set combat up to win the whole run.

Run (from the engine dir, venv python):
  python rl/train_agents.py --iters 200 --runs 10 --device cpu --out rl/az_ckpt
  python rl/train_agents.py --eval --runs 20 --resume rl/az_ckpt   # win-rate only

The engine must be built (dotnet build src/Sts2Headless/Sts2Headless.csproj).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import make_agents
from run_player import play_run
from rl.engine import Engine, EngineError  # type: ignore

CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]

# Per-character terminal colors (ANSI). os.system("") turns on ANSI processing
# in Windows PowerShell / Terminal; harmless elsewhere.
if os.name == "nt":
    os.system("")
_CHAR_COLORS = {"Ironclad": "\033[91m", "Silent": "\033[92m", "Defect": "\033[96m",
                "Regent": "\033[93m", "Necrobinder": "\033[95m"}
_RESET, _BOLD = "\033[0m", "\033[1m"


def col(char: str, text: str) -> str:
    return f"{_CHAR_COLORS.get(char, '')}{text}{_RESET}"


def run_reward(res: dict) -> float:
    """Card/path/event return. Overall victory is the true goal but it's sparse
    (≈0% early), so it gives those agents almost no gradient. We add a dense
    milestone for **defeating each act boss** (won combat tagged BOSS) plus small
    act/floor shaping, so drafting/pathing/events get a real signal long before
    full runs are won."""
    boss_wins = sum(1 for c in res["combats"] if c["tier"] == "BOSS" and c["won"])
    # HP-preservation: the meta heads (rest/card/path) get NO HP signal otherwise, so
    # they never learn to heal / route safely / draft for survivability — and runs die
    # of Act-1 attrition. Reward the average HP fraction the player carried into its
    # fights: a run kept healthy (heal at rest, avoid bleed) scores higher than one that
    # ground down to death at the same floor. Combat keeps its own per-fight HP reward.
    combats = res.get("combats") or []
    fracs = [c["start_hp"] / c["start_max_hp"] for c in combats if c.get("start_max_hp")]
    hp_health = (sum(fracs) / len(fracs)) if fracs else 0.0
    return (1.0 * float(res["victory"])
            + 0.5 * boss_wins                       # each act boss cleared
            + 0.3 * hp_health                       # stay healthy across the run
            + 0.15 * (float(res["act"]) - 1)
            + 0.01 * float(res["floor"]))


def hp_retained(c: dict) -> float:
    return c["end_hp"] / c["start_hp"] if c["start_hp"] > 0 else 0.0


def _log_failure(path, eng, char, reason, res=None):
    """Append a replayable failure record (seed + exact action sequence + trace)."""
    if not path:
        return
    try:
        rec = eng.repro()
        rec["char"] = char
        rec["reason"] = reason
        rec["last_decision"] = (res or {}).get("last_decision")
        rec["trace"] = (res or {}).get("trace")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"     ↳ logged failure to {path} (seed {rec.get('seed')}) — replay: python rl/replay.py")
    except Exception:
        pass


def _log_run(runlog, it, i, char, seed, res):
    """Append a compact record of one run (outcome + decision trace) so Ctrl-C leaves
    a readable trail of what happened. Not a full replay record (see faillog)."""
    if not runlog:
        return
    try:
        combats = res.get("combats") or []
        rec = {"it": it, "run": i, "char": char, "seed": seed,
               "win": bool(res.get("victory")), "act": res.get("act"), "floor": res.get("floor"),
               "combats": f"{sum(1 for c in combats if c.get('won'))}/{len(combats)}",
               "tiers": [c.get("tier") for c in combats],
               "end": res.get("end_reason"), "last": res.get("last_decision"),
               "events": len(res.get("event_samples") or []), "rest": res.get("rest_choices"),
               "trace": res.get("trace")}
        with open(runlog, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def _play_one(agents, char, seed, greedy):
    """Play a single run in its own engine process. Returns a result dict the main
    thread aggregates. Runs on a worker thread; only reads the shared agents (torch
    CPU forward + numpy sampling are safe to call concurrently — each run records the
    action it actually took, so its training tuples stay self-consistent)."""
    try:
        eng = Engine(character=char, seed=seed, ascension=10)
    except EngineError as e:
        return {"char": char, "seed": seed, "kind": "start_fail", "err": str(e)}
    try:
        res = play_run(eng, agents, greedy=greedy)
    except EngineError as ex:                # hang (read-timeout) or dead engine
        try: eng.close()
        except Exception: pass
        return {"char": char, "seed": seed, "kind": "engine", "err": str(ex), "eng": eng}
    try: eng.close()                         # normal path (eng.repro() still valid after close)
    except Exception: pass
    return {"char": char, "seed": seed, "kind": "ok", "res": res, "eng": eng}


def collect(agents, characters, n_runs, base_seed, greedy, faillog=None, runlog=None,
            it=0, workers=1):
    """Play n_runs full runs (across `workers` parallel engine processes) and return
    (samples-per-agent, combat/victory stats). Each run is an independent subprocess,
    so collection — the wall-clock bottleneck — scales ~linearly with cores. The learn
    step stays serial in the caller; aggregation here runs on the main thread in run
    order, so results are order-deterministic regardless of finish order."""
    # Rotate the run log every 5 iterations so it stays small but still covers the
    # recent past for post-Ctrl-C inspection.
    if runlog and it % 5 == 0:
        try:
            open(runlog, "w").close()
        except Exception:
            pass
    # each buffer: obs, mask, action, return
    buf = {k: [[], [], [], []] for k in
           ("card", "shop", "event", "rest", "upgrade", "path")}
    buf["combat"] = [[], [], [], [], []]      # dense, card_ids, mask, action, return
    buf["cselect"] = [[], [], [], [], [], []]  # glob, cand_dense, cand_ids, n, action, return
    stats = {"combats": [], "victories": 0, "runs": 0, "runs_detail": [], "events": 0,
             "rest": defaultdict(int)}

    # ---- collect runs in parallel ----
    results: list = [None] * n_runs
    seeds = [f"{base_seed}-{i}" for i in range(n_runs)]
    chars = [characters[i % len(characters)] for i in range(n_runs)]
    ex = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futs = {ex.submit(_play_one, agents, chars[i], seeds[i], greedy): i
                for i in range(n_runs)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    except KeyboardInterrupt:
        # Parallel mode can't cleanly dump every in-flight run; cancel what's pending
        # and re-raise. Hung runs still self-log via play_run's watchdog -> EngineError.
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        ex.shutdown(wait=True)

    # ---- aggregate on the main thread, in run order (deterministic) ----
    for i in range(n_runs):
        r = results[i]
        if r is None:
            continue
        char, seed = r["char"], r["seed"]
        if r["kind"] == "start_fail":
            print(f"  [skip {char}] engine start failed: {r['err']}")
            continue
        if r["kind"] == "engine":
            print(col(char, f"   ! {char:11s} engine hang/died @ run{i}: {r['err'][:50]}"))
            _log_failure(faillog, r["eng"], char, f"engine:{r['err'][:120]}")
            _log_run(runlog, it, i, char, seed,
                     {"end_reason": f"engine:{r['err'][:80]}", "combats": []})
            continue
        res, eng = r["res"], r["eng"]
        _log_run(runlog, it, i, char, seed, res)         # every run -> rolling log
        stats["runs"] += 1
        stats["victories"] += int(res["victory"])
        stats["runs_detail"].append((char, bool(res["victory"]), float(res["floor"]), float(res["act"])))
        rr = run_reward(res)
        for (dense, ids, mask, a, cid) in res["combat_samples"]:
            if cid < 0:
                continue
            r2 = hp_retained(res["combats"][cid])
            buf["combat"][0].append(dense); buf["combat"][1].append(ids)
            buf["combat"][2].append(mask); buf["combat"][3].append(a); buf["combat"][4].append(r2)
        for (glob, cd, ci, n, a, cid) in res["select_samples"]:   # in-combat card selects
            if cid < 0:
                continue
            r2 = hp_retained(res["combats"][cid])
            for j, val in enumerate((glob, cd, ci, n, a, r2)):
                buf["cselect"][j].append(val)
        for key in ("card", "shop", "event", "rest", "upgrade", "path"):  # rewarded by victory (rr)
            for (obs, mask, a) in res[f"{key}_samples"]:
                buf[key][0].append(obs); buf[key][1].append(mask)
                buf[key][2].append(a); buf[key][3].append(rr)
        for c in res["combats"]:
            stats["combats"].append((char, c["tier"], c["won"], hp_retained(c)))
        stats["events"] += len(res.get("event_samples") or [])
        for oid, cnt in (res.get("rest_choices") or {}).items():
            stats["rest"][oid] += cnt
        # Only surface ABNORMAL runs (no combats / error / stuck) — the per-iteration
        # summary covers everything else, keeping the log readable.
        er = res.get("end_reason", "")
        if len(res["combats"]) == 0 or er.startswith(("error", "stuck", "timeout", "max_steps")):
            print(col(char, f"   ! {char:11s} {er} @ {res.get('last_decision')}"))
            _log_failure(faillog, eng, char, er, res)
    return buf, stats


_ABBR = {"Ironclad": "Iron", "Silent": "Slnt", "Defect": "Dfct",
         "Regent": "Rgnt", "Necrobinder": "Necr"}


def report(stats, it=0, secs=0.0, best=None):
    """Compact one-block-per-iteration summary. `best` (dict) tracks bests across iters."""
    best = best if best is not None else {}
    combats = stats["combats"]
    n = len(combats)
    rd = stats["runs_detail"]
    R = len(rd) or 1
    cw = sum(1 for _, _, w, _ in combats if w)
    combat_wr = cw / n if n else 0.0
    hp = (sum(h for *_, h in combats) / n) if n else 0.0
    gw = sum(1 for r in rd if r[1])
    game_wr = gw / R
    avg_floor = sum(r[2] for r in rd) / R
    best["combat"] = max(best.get("combat", 0.0), combat_wr)
    best["game"] = max(best.get("game", 0.0), game_wr)
    best["floor"] = max(best.get("floor", 0.0), avg_floor)

    tier = defaultdict(lambda: [0, 0])
    for _, t, w, _ in combats:
        tier[t][0] += int(w); tier[t][1] += 1
    tc = lambda k: f"{tier[k][0]}/{tier[k][1]}"
    # per-character progression: average act + floor reached this iteration
    prog = defaultdict(lambda: [0.0, 0.0, 0])       # char -> [floor sum, act sum, runs]
    for c, _v, f, a in rd:
        prog[c][0] += f; prog[c][1] += a; prog[c][2] += 1
    chars = "  ".join(col(c, f"{_ABBR.get(c, c[:4])} a{p[1]/p[2]:.0f} f{p[0]/p[2]:.0f}")
                      for c, p in sorted(prog.items()) if p[2])

    print(f"\n{_BOLD}── iter {it} · {secs:.0f}s ─────────────────────────────────{_RESET}")
    print(f" combat {_BOLD}{combat_wr*100:3.0f}%{_RESET} (best {best['combat']*100:.0f}%)"
          f"      game {_BOLD}{game_wr*100:3.0f}%{_RESET} (best {best['game']*100:.0f}%)")
    print(f" floor {avg_floor:4.1f} (best {best['floor']:.0f})       HP kept {hp*100:.0f}%")
    print(f" fights  normal {tc('COMBAT')}   elite {tc('ELITE')}   boss {tc('BOSS')}"
          f"      events {stats.get('events', 0)}")
    rest = stats.get("rest") or {}
    rtot = sum(rest.values())
    other = rtot - rest.get("HEAL", 0) - rest.get("SMITH", 0)
    print(f" rests   {rtot}   heal {rest.get('HEAL', 0)}  smith(upgrade) {rest.get('SMITH', 0)}"
          + (f"  other {other}" if other else ""))
    if chars:
        print(f" reached {chars}")
    return combat_wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=0,
                    help="training iterations; 0 = run endlessly until Ctrl-C")
    ap.add_argument("--runs", type=int, default=10, help="runs collected per iteration")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 4),
                    help="parallel engine processes for run collection (set to physical "
                         "core count; each run is a single-threaded engine subprocess)")
    ap.add_argument("--characters", default=",".join(CHARACTERS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="rl/az_ckpt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval", action="store_true", help="greedy play + win-rate only (no training)")
    ap.add_argument("--seed", default="az")
    ap.add_argument("--faillog", default="rl/failures.jsonl",
                    help="append replayable records for hung/errored/stuck runs (rl/replay.py)")
    ap.add_argument("--runlog", default="rl/runs.jsonl",
                    help="rolling per-run log (all runs); rotated every 5 iterations")
    args = ap.parse_args()

    chars = [c.strip() for c in args.characters.split(",") if c.strip()]
    agents = make_agents(args.device)
    ckpt = args.resume or (args.out if os.path.exists(f"{args.out}.combat.pt") else None)
    if ckpt:
        for name in agents:
            try:
                agents[name].load(f"{ckpt}.{name}.pt"); print(f"loaded {name}")
            except Exception:
                pass

    if args.eval:
        _, stats = collect(agents, chars, args.runs, f"{args.seed}-eval", greedy=True,
                           faillog=args.faillog, runlog=args.runlog, it=0, workers=args.workers)
        report(stats, it=0, secs=0.0, best={})
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = {}                                       # bests across iterations
    it = 0
    while args.iters <= 0 or it < args.iters:      # --iters 0 => endless (Ctrl-C)
        t0 = time.time()
        buf, stats = collect(agents, chars, args.runs, f"{args.seed}-{it}", greedy=False,
                             faillog=args.faillog, runlog=args.runlog, it=it, workers=args.workers)
        agents["combat"].learn(*buf["combat"])
        agents["combat"].learn_select(*buf["cselect"])          # in-combat card selects
        agents["card"].learn(buf["card"], buf["shop"], buf["event"], buf["rest"], buf["upgrade"])
        agents["path"].learn(*buf["path"])
        wr = report(stats, it=it, secs=time.time() - t0, best=best)
        for name in agents:
            agents[name].save(f"{args.out}.{name}.pt")
        if wr >= 0.80:
            print(f" {_BOLD}*** reached {wr*100:.0f}% combat win rate ***{_RESET}")
        it += 1


if __name__ == "__main__":
    main()

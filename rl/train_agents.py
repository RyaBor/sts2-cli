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
import os
import sys
import time
from collections import defaultdict

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


def collect(agents, characters, n_runs, base_seed, greedy):
    """Play n_runs full runs; return (samples-per-agent, combat/victory stats)."""
    # each buffer: obs, mask, action, return
    buf = {k: [[], [], [], []] for k in
           ("card", "shop", "event", "rest", "upgrade", "path")}
    buf["combat"] = [[], [], [], [], []]      # dense, card_ids, mask, action, return
    buf["cselect"] = [[], [], [], [], [], []]  # glob, cand_dense, cand_ids, n, action, return
    stats = {"combats": [], "victories": 0, "runs": 0, "runs_detail": [], "events": 0}
    for i in range(n_runs):
        char = characters[i % len(characters)]
        try:
            eng = Engine(character=char, seed=f"{base_seed}-{i}", ascension=10)
        except EngineError as e:
            print(f"  [skip {char}] engine start failed: {e}")
            continue
        try:
            res = play_run(eng, agents, greedy=greedy)
        finally:
            eng.close()
        stats["runs"] += 1
        stats["victories"] += int(res["victory"])
        stats["runs_detail"].append((char, bool(res["victory"]), float(res["floor"])))
        rr = run_reward(res)
        for (dense, ids, mask, a, cid) in res["combat_samples"]:
            if cid < 0:
                continue
            r = hp_retained(res["combats"][cid])
            buf["combat"][0].append(dense); buf["combat"][1].append(ids)
            buf["combat"][2].append(mask); buf["combat"][3].append(a); buf["combat"][4].append(r)
        for (glob, cd, ci, n, a, cid) in res["select_samples"]:   # in-combat card selects
            if cid < 0:
                continue
            r = hp_retained(res["combats"][cid])
            for j, val in enumerate((glob, cd, ci, n, a, r)):
                buf["cselect"][j].append(val)
        for key in ("card", "shop", "event", "rest", "upgrade", "path"):  # rewarded by victory (rr)
            for (obs, mask, a) in res[f"{key}_samples"]:
                buf[key][0].append(obs); buf[key][1].append(mask)
                buf[key][2].append(a); buf[key][3].append(rr)
        for c in res["combats"]:
            stats["combats"].append((char, c["tier"], c["won"], hp_retained(c)))
        stats["events"] += len(res.get("event_samples") or [])
        # Only surface ABNORMAL runs (no combats / error / stuck) — the per-iteration
        # summary covers everything else, keeping the log readable.
        er = res.get("end_reason", "")
        if len(res["combats"]) == 0 or er.startswith(("error", "stuck")):
            print(col(char, f"   ! {char:11s} {er} @ {res.get('last_decision')}"))
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
    gw = sum(1 for _, v, _ in rd if v)
    game_wr = gw / R
    avg_floor = sum(f for *_, f in rd) / R
    best["combat"] = max(best.get("combat", 0.0), combat_wr)
    best["game"] = max(best.get("game", 0.0), game_wr)
    best["floor"] = max(best.get("floor", 0.0), avg_floor)

    tier = defaultdict(lambda: [0, 0])
    for _, t, w, _ in combats:
        tier[t][0] += int(w); tier[t][1] += 1
    tc = lambda k: f"{tier[k][0]}/{tier[k][1]}"
    cmb = defaultdict(lambda: [0, 0])
    for c, _, w, _ in combats:
        cmb[c][0] += int(w); cmb[c][1] += 1
    chars = "  ".join(col(c, f"{_ABBR.get(c, c[:4])} {(cmb[c][0]/cmb[c][1]*100 if cmb[c][1] else 0):.0f}%")
                      for c in sorted(cmb))

    print(f"\n{_BOLD}── iter {it} · {secs:.0f}s ─────────────────────────────────{_RESET}")
    print(f" combat {_BOLD}{combat_wr*100:3.0f}%{_RESET} (best {best['combat']*100:.0f}%)"
          f"      game {_BOLD}{game_wr*100:3.0f}%{_RESET} (best {best['game']*100:.0f}%)")
    print(f" floor {avg_floor:4.1f} (best {best['floor']:.0f})       HP kept {hp*100:.0f}%")
    print(f" fights  normal {tc('COMBAT')}   elite {tc('ELITE')}   boss {tc('BOSS')}"
          f"      events {stats.get('events', 0)}")
    if chars:
        print(f" chars   {chars}")
    return combat_wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=0,
                    help="training iterations; 0 = run endlessly until Ctrl-C")
    ap.add_argument("--runs", type=int, default=10, help="runs collected per iteration")
    ap.add_argument("--characters", default=",".join(CHARACTERS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="rl/az_ckpt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval", action="store_true", help="greedy play + win-rate only (no training)")
    ap.add_argument("--seed", default="az")
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
        _, stats = collect(agents, chars, args.runs, f"{args.seed}-eval", greedy=True)
        report(stats, it=0, secs=0.0, best={})
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = {}                                       # bests across iterations
    it = 0
    while args.iters <= 0 or it < args.iters:      # --iters 0 => endless (Ctrl-C)
        t0 = time.time()
        buf, stats = collect(agents, chars, args.runs, f"{args.seed}-{it}", greedy=False)
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

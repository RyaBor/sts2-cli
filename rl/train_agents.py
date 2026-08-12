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


def run_reward(res: dict) -> float:
    """Card/path return: overall victory, shaped by act/floor progress so there's
    a learning gradient before wins become common."""
    return 1.0 * float(res["victory"]) + 0.15 * (float(res["act"]) - 1) + 0.01 * float(res["floor"])


def hp_retained(c: dict) -> float:
    return c["end_hp"] / c["start_hp"] if c["start_hp"] > 0 else 0.0


def collect(agents, characters, n_runs, base_seed, greedy):
    """Play n_runs full runs; return (samples-per-agent, combat/victory stats)."""
    # combat: obs, mask, action, return   |   card/path: obs, mask, action, return
    buf = {"combat": [[], [], [], []], "card": [[], [], [], []], "path": [[], [], [], []]}
    stats = {"combats": [], "victories": 0, "runs": 0}
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
        rr = run_reward(res)
        for (obs, mask, a, cid) in res["combat_samples"]:
            if cid < 0:
                continue
            buf["combat"][0].append(obs); buf["combat"][1].append(mask)
            buf["combat"][2].append(a); buf["combat"][3].append(hp_retained(res["combats"][cid]))
        for (obs, mask, a) in res["card_samples"]:
            buf["card"][0].append(obs); buf["card"][1].append(mask)
            buf["card"][2].append(a); buf["card"][3].append(rr)
        for (obs, mask, a) in res["path_samples"]:
            buf["path"][0].append(obs); buf["path"][1].append(mask)
            buf["path"][2].append(a); buf["path"][3].append(rr)
        for c in res["combats"]:
            stats["combats"].append((char, c["tier"], c["won"], hp_retained(c)))
        print(f"  run {i:3d} {char:11s} A10  {'WIN' if res['victory'] else 'lose'} "
              f"act{res['act']} floor{res['floor']} combats={len(res['combats'])} "
              f"wins={sum(1 for c in res['combats'] if c['won'])}")
    return buf, stats


def report(stats):
    combats = stats["combats"]
    n = len(combats)
    if n == 0:
        print("  (no combats)")
        return 0.0
    wins = sum(1 for _, _, w, _ in combats if w)
    hp = sum(h for *_, h in combats) / n
    print(f"\n  COMBAT WIN RATE: {wins}/{n} = {wins/n*100:.1f}%   "
          f"avg HP retained {hp*100:.0f}%   game victories {stats['victories']}/{stats['runs']}")
    by_char = defaultdict(lambda: [0, 0])
    by_tier = defaultdict(lambda: [0, 0])
    for char, tier, w, _ in combats:
        by_char[char][0] += int(w); by_char[char][1] += 1
        by_tier[tier][0] += int(w); by_tier[tier][1] += 1
    print("   by character: " + "  ".join(
        f"{c}={w}/{t}({w/t*100:.0f}%)" for c, (w, t) in sorted(by_char.items())))
    print("   by opponent : " + "  ".join(
        f"{k}={w}/{t}({w/t*100:.0f}%)" for k, (w, t) in sorted(by_tier.items())))
    return wins / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
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
            p = f"{ckpt}.{name}.pt"
            if os.path.exists(p):
                agents[name].load(p); print(f"loaded {p}")

    if args.eval:
        _, stats = collect(agents, chars, args.runs, f"{args.seed}-eval", greedy=True)
        report(stats)
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for it in range(args.iters):
        t0 = time.time()
        print(f"\n=== iteration {it} ===")
        buf, stats = collect(agents, chars, args.runs, f"{args.seed}-{it}", greedy=False)
        losses = {}
        losses["combat"] = agents["combat"].learn(*buf["combat"])
        losses["card"] = agents["card"].learn(*buf["card"])
        losses["path"] = agents["path"].learn(*buf["path"])
        wr = report(stats)
        print(f"   losses {({k: round(v,3) for k,v in losses.items()})}  "
              f"iter {time.time()-t0:.1f}s")
        for name in agents:
            agents[name].save(f"{args.out}.{name}.pt")
        if wr >= 0.80:
            print(f"\n*** reached {wr*100:.0f}% combat win rate ***")


if __name__ == "__main__":
    main()

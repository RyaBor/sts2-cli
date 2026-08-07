"""Evaluate a checkpoint against a random-legal-move baseline.

    python rl/eval.py --model rl/checkpoints/combat_final.zip --episodes 200
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO

from rl.env import Sts2CombatEnv


def run(env: Sts2CombatEnv, episodes: int, model=None, rng=None):
    wins, hps, lens, rewards = 0, [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        total, steps, info = 0.0, 0, {}
        while True:
            mask = env.action_masks()
            if model is None:
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)
            obs, r, term, trunc, info = env.step(action)
            total += r
            steps += 1
            if term or trunc:
                break
        if info.get("outcome") == "win":
            wins += 1
            hps.append(info.get("final_hp", 0))
        lens.append(steps)
        rewards.append(total)
    return {
        "win_rate": wins / episodes,
        "avg_hp_on_win": statistics.mean(hps) if hps else 0.0,
        "avg_len": statistics.mean(lens),
        "avg_reward": statistics.mean(rewards),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--encounter", default="SHRINKER_BEETLE_WEAK",
                    help="comma-separated encounter ids, or 'default'")
    ap.add_argument("--hp", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    # Match train.py: accept a list so a policy can be evaluated on the same
    # mix it was trained on.
    encounters = (None if args.encounter == "default"
                  else [e.strip() for e in args.encounter.split(",") if e.strip()])
    cfg = dict(character=args.character, encounter=encounters,
               start_hp=args.hp, max_hp=args.hp)

    if not args.no_baseline:
        env = Sts2CombatEnv(seed="evalrand", **cfg)
        base = run(env, args.episodes, None, np.random.default_rng(0))
        env.close()
        print("random baseline:", {k: round(v, 3) for k, v in base.items()})

    model = MaskablePPO.load(args.model, device=args.device)
    env = Sts2CombatEnv(seed="evalpol", **cfg)
    res = run(env, args.episodes, model)
    env.close()
    print("policy:         ", {k: round(v, 3) for k, v in res.items()})


if __name__ == "__main__":
    main()

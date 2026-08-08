"""Train a combat policy with MaskablePPO across parallel engine processes.

    python rl/train.py --envs 12 --steps 2000000

Each env owns one headless engine process, so --envs is the real parallelism
knob. Watch progress with:  tensorboard --logdir rl/runs
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from rl.callbacks import CombatMetricsCallback
from rl.env import DEFAULT_REWARDS, Sts2CombatEnv
from rl.extractor import EntityAttentionExtractor
from rl import cardpool

HERE = os.path.dirname(os.path.abspath(__file__))


def _make(rank: int, cfg: dict):
    # No per-env Monitor: VecMonitor wraps the whole vec env below, and having
    # both makes the inner one's statistics silently discarded.
    def _init():
        return Sts2CombatEnv(seed=f"w{rank}", **cfg)
    return _init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=12, help="parallel engine processes")
    ap.add_argument("--steps", type=int, default=1_000_000, help="total timesteps")
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--encounter", default="SHRINKER_BEETLE_WEAK",
                    help="comma-separated encounter ids, or 'default'")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--hp", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=256, help="rollout per env")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--resume", default=None, help="path to a .zip checkpoint")
    ap.add_argument("--name", default="combat")
    ap.add_argument("--reward", action="append", default=[], metavar="KEY=VALUE",
                    help="override a reward weight, e.g. --reward win_hp=2.5 "
                         "(keys: win, win_hp, loss, timeout, dmg_dealt, hp_lost, step)")
    ap.add_argument("--snapshots", default=None,
                    help="jsonl of realistic loadouts from rl/runner.py")
    ap.add_argument("--deck-noise", type=float, default=0.0,
                    help="fraction of each snapshot deck replaced with random "
                         "cards, to cover builds the drafter never produces")
    args = ap.parse_args()

    snapshots = None
    if args.snapshots:
        with open(args.snapshots, encoding="utf-8") as f:
            snapshots = [json.loads(line) for line in f if line.strip()]
        acts = sorted({s.get("act") for s in snapshots})
        decks = [len(s.get("deck") or []) for s in snapshots]
        print(f"{len(snapshots)} snapshots | acts {acts} | "
              f"deck size {min(decks)}-{max(decks)} (mean {sum(decks)/len(decks):.1f})")

    pool = None
    if args.deck_noise > 0:
        pool = cardpool.valid_cards()
        print(f"deck noise {args.deck_noise:.0%} over {len(pool)} cards")

    rewards = {}
    for item in args.reward:
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in DEFAULT_REWARDS:
            ap.error(f"unknown reward key '{key}'; valid: {', '.join(DEFAULT_REWARDS)}")
        rewards[key] = float(value)
    if rewards:
        print(f"reward overrides: {rewards}")

    encounters = (None if args.encounter == "default"
                  else [e.strip() for e in args.encounter.split(",") if e.strip()])
    cfg = dict(character=args.character, encounter=encounters,
               ascension=args.ascension, start_hp=args.hp, max_hp=args.hp,
               rewards=rewards or None, snapshots=snapshots,
               deck_noise=args.deck_noise, card_pool=pool)

    venv = SubprocVecEnv([_make(i, cfg) for i in range(args.envs)], start_method="spawn")
    venv = VecMonitor(venv)

    run_dir = os.path.join(HERE, "runs")
    ckpt_dir = os.path.join(HERE, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.resume:
        model = MaskablePPO.load(args.resume, env=venv, device=args.device)
        print(f"resumed from {args.resume}")
    else:
        model = MaskablePPO(
            "MultiInputPolicy", venv,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            policy_kwargs=dict(
                features_extractor_class=EntityAttentionExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=[256, 256],
            ),
            tensorboard_log=run_dir,
            device=args.device,
            verbose=1,
        )

    ckpt = CheckpointCallback(
        save_freq=max(20_000 // args.envs, 1),
        save_path=ckpt_dir,
        name_prefix=args.name,
    )

    callbacks = [ckpt, CombatMetricsCallback()]

    try:
        model.learn(total_timesteps=args.steps, callback=callbacks,
                    tb_log_name=args.name, reset_num_timesteps=not args.resume)
    finally:
        final = os.path.join(ckpt_dir, f"{args.name}_final")
        model.save(final)
        print(f"saved {final}.zip")
        venv.close()


if __name__ == "__main__":
    main()

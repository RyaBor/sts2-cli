# Three-agent A10 trainer (`train_agents.py`)

Trains three cooperating policy agents by REINFORCE on **full Ascension-10 runs**
played through the headless CLI, for every character, and tracks **combat win rate**.

| Agent | Decides | Reward |
|---|---|---|
| `combat` | play card / **use potion** / end turn | HP retained this combat = `end_hp / start_hp` (**≥1.0 = perfect win**) |
| `card` (two heads) | card rewards **and the shop** (buy card/relic/potion, remove card, leave) | overall **game victory** (+ small act/floor progress shaping) |
| `path` | which map node to enter | overall **game victory** (+ progress shaping) |

Potions are part of the combat action space (`encoding.py`), so the combat agent
learns to drink them. The card agent has a second head for the shop, so drafting
*and* buying/removing are learned together (`agents.py: CardAgent`). Events, rest
sites (default: heal), and bundles use fixed defaults for now.

The card/path agents are rewarded by *whole-run* success, so they learn to pick
cards and routes that let the combat agent actually win the run — combat quality
isn't sabotaged by bad drafting/pathing.

## Prerequisites
1. Build the engine: `dotnet build src/Sts2Headless/Sts2Headless.csproj`
2. Install torch into the venv (CPU shown; use the CUDA wheel for a GPU):
   `pip install torch`  (or the `--index-url .../cu128` build per `requirements.txt`)

## Train
```bash
python rl/train_agents.py --iters 500 --runs 20 --device cpu --out rl/az_ckpt
# GPU:  --device cuda
```
Each iteration plays `--runs` full A10 runs (cycling all 5 characters), updates
the three agents, prints the combat win-rate report, and saves
`rl/az_ckpt.{combat,card,path}.pt`. Resumes automatically if those exist (or pass
`--resume PATH`). It prints `*** reached N% ***` when combat win rate hits the
80% target.

## Just measure win rate (no training)
```bash
python rl/train_agents.py --eval --runs 40 --resume rl/az_ckpt
```
Greedy play; reports win rate overall, per character, and per opponent tier
(COMBAT / ELITE / BOSS), plus average HP retained and game victories.

## Notes
- Self-play is pure-policy (no MCTS), so it's fast: a run is just engine
  round-trips. Untrained agents die early (short runs); runs lengthen as they
  improve. Scale `--runs`/`--iters` up on your machine.
- Non-agent decisions (events, rest sites, shops, in-combat card selects) use
  fixed sensible defaults so runs always progress.
- Applying a trained model as an MCTS heuristic (policy priors + value at leaves)
  is a separate, later step — this trainer produces the standalone policies first.
- Encoders: combat reuses `encoding.py`; card/path use compact encoders in
  `agents.py`. If a `map_select`/`card_reward` field name differs on your build,
  adjust `_room_type` / `encode_card_reward` there.

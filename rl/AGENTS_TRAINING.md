# Three-agent A10 trainer (`train_agents.py`)

Trains three cooperating policy agents by REINFORCE on **full Ascension-10 runs**
played through the headless CLI, for every character, and tracks **combat win rate**.

| Agent | Decides | Reward |
|---|---|---|
| `combat` | play card / use potion / **discard potion** / end turn / **in-combat card selects** (Armaments, Dual Wield, Exhume, discover, pile moves) | HP retained this combat = `end_hp / start_hp` (**≥1.0 = perfect win**) |
| `card` (three heads) | card rewards, **the shop** (buy card/relic/potion, remove card, leave), **and event/ancient choices** | overall **game victory** (+ small act/floor progress shaping) |
| `path` | which map node to enter | overall **game victory** (+ progress shaping) |

Potions are part of the combat action space (`encoding.py`): the combat agent can
use a potion (optionally targeted) **or discard one to free a slot** when the belt
is full. The card agent has a second head for the shop, so drafting *and*
buying/removing are learned together (`agents.py: CardAgent`). A shop holds up to
**7 cards** (5 colored + 2 colorless). The card and path agents observe **gold**,
so drafting/pathing can plan around affording shop buys and card removal.

**Events and ancient nodes** (including Neow) both arrive as the `event_choice`
decision — the engine resolves the name from the ancients table first, then events
— so a single third head on the card agent handles both. It encodes each option
generically (identity hash + signed resource magnitudes: gold / hp-cost / hp-heal
from the option's `vars`) and masks out locked options, rewarded by game victory
like the other card heads. Rest sites (default: heal) and bundles still use fixed
defaults.

The combat net (`agents.CombatNet`) uses a learned **card embedding**: `encode_combat`
returns `(dense, card_ids)`, and the net embeds each card id (compact, learns card
similarity, ~6× smaller input than a one-hot, faster). It also has a **selection head**
(`score_select`) that scores candidate cards for **in-combat `card_select`** prompts —
the single root mechanic behind Armaments/Dual Wield/Exhume/discover and all
draw↔hand↔discard↔exhaust pile moves — so the combat agent (not a dumb default) makes
those picks, rewarded by that combat's HP retained.

The combat observation (`encoding.py`) is built to be a **generalizing policy** — enough
that the extracted combat agent can serve as an MCTS prior in live games. It includes:
- **Card identity** via an embedding over a deterministic **vocabulary** (~606 ids from
  `cards.json`, sorted; unknown ids hash into a small tail). Upgrade/enchant behavior is
  captured by a **factored** representation (upgrade level, hashed effect profile,
  keywords, enchant markers) so Strike / Strike+ / enchanted-Strike are distinct.
- **Owned relics** (hashed multi-hot) so play can condition on relic effects
  (Burning Blood, Strength/energy relics, attack/block triggers).
- **Enemy intent type** (Attack/Defend/Buff/Debuff/…, hashed) alongside the
  **sim-resolved** incoming damage (all enemy Strength/Weak and player Vulnerable
  folded in; multi-hit uses `total_damage`).
- Card damage resolved via `damage_by_target`/`calculateddamage` (so computed cards
  like Unleash aren't seen as 0).
- **Character-specific mechanics**: Regent **stars** + per-card **star cost**,
  Defect **orbs** (passive/evoke/type), Necrobinder **Osty** (alive/HP/block).

Action legality (`can_play`) is the game's own native check and already handles
per-character costs. (Osty is **not** a legality gate in this build — a dead Osty
just lowers Unleash's damage, which the resolved-damage encoding already reflects.)
Remaining limitation: cards that must *target Osty* aren't a distinct action
(the space targets enemies or is untargeted); most Osty interactions are
untargeted so this rarely bites, but it's the next thing to add if Necrobinder
underperforms.

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

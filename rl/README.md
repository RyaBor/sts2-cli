# RL combat training

Trains a Slay the Spire 2 combat policy against the real game engine with
MaskablePPO. One episode = one fight. Each parallel env owns a headless engine
process, so `--envs` is the parallelism knob.

## Setup

The engine must be built first:

```bash
dotnet build src/Sts2Headless/Sts2Headless.csproj
```

Dependencies live in `.venv` at the repo root (see `requirements.txt` — note
the CUDA caveat for RTX 50xx cards).

## Train

```bash
.venv/Scripts/python.exe rl/train.py --envs 12 --steps 1000000 --name combat_v1
```

Useful flags: `--character`, `--encounter` (comma-separated ids, or `default`),
`--ascension`, `--hp`, `--device cpu|cuda`, `--resume <ckpt.zip>`.

Checkpoints land in `rl/checkpoints/`, TensorBoard logs in `rl/runs/`:

```bash
.venv/Scripts/python.exe -m tensorboard.main --logdir rl/runs
```

## Metrics and dashboard

Training logs the usual PPO diagnostics plus combat-specific metrics that SB3
does not provide on its own (`rl/callbacks.py`):

| Metric | Why it matters |
| --- | --- |
| `combat/win_rate` | The actual objective |
| `combat/hp_retained_on_win` | The real signal on easy encounters, where win rate saturates at 100% |
| `combat/invalid_actions` | Must stay **0** — a climbing count means the action mask and the engine disagree about legality |
| `combat/engine_errors` | Engine crashes swallowed as lost episodes |

**Live monitoring** during a run:

```bash
.venv/Scripts/python.exe -m tensorboard.main --logdir rl/runs
```

**Standalone report** — one self-contained HTML file, no external requests, safe
to share or open directly. Overlays multiple runs for comparison, and every
chart has a table view:

```bash
.venv/Scripts/python.exe rl/dashboard.py --open
```

`--run <name>` restricts it to one run, `--out <path>` changes the destination.
The generated `rl/dashboard.html` is gitignored — regenerate it after each run.

## Evaluate

Reports win rate, average HP remaining on wins, and episode length against a
random-legal-move baseline:

```bash
.venv/Scripts/python.exe rl/eval.py --model rl/checkpoints/combat_v1_final.zip --episodes 200
```

## Encounters

`SHRINKER_BEETLE_WEAK` (the default) is easy enough that random play wins every
time — judge progress by **HP retained**, not win rate. For a real challenge,
train against a mix:

```bash
.venv/Scripts/python.exe rl/encounters.py           # list ids by tier
.venv/Scripts/python.exe rl/encounters.py --check   # verify each one loads
```

79 of 81 load. Pass several to spread the agent across fights:

```bash
.venv/Scripts/python.exe rl/train.py --envs 12 --steps 2000000 --name mixed \
  --encounter CULTISTS_NORMAL,CHOMPERS_NORMAL,EXOSKELETONS_NORMAL,MYTES_NORMAL
```

## Layout

| File | Role |
| --- | --- |
| `engine.py` | One headless engine process; `reset_combat()` re-enters a fight in ~2ms |
| `encoding.py` | State → 879-dim observation; fixed 61-action space + legality masks |
| `env.py` | Gymnasium env, reward shaping, episode lifecycle |
| `train.py` | MaskablePPO over `SubprocVecEnv` |
| `eval.py` | Checkpoint vs random baseline |
| `encounters.py` | Encounter ids by tier, with a loadability check |
| `callbacks.py` | Logs win rate / HP retained / mask violations to TensorBoard |
| `dashboard.py` | Generates a standalone HTML metrics report |

## Design notes

**Action masking.** Hand size and enemy count change every turn, so the action
space is fixed (10 hand slots × 5 targets, plus untargeted plays, plus end
turn) and illegal entries are masked. Without masking most samples are wasted
on unplayable cards.

**Card identity** uses a crc32 hashing trick rather than a vocab file —
`hash()` is randomized per process, which would silently desync SubprocVecEnv
workers.

**HP is restored every episode.** Otherwise damage carries between fights until
the player dies, which ends the *run* and forces a ~0.9s engine restart. Pinning
the encounter also keeps the task stationary (the floor does not advance).

**stderr must be drained.** The engine logs heavily; if that pipe is never read
it fills, the engine blocks writing to it, and stops responding on stdout. This
deadlock is why `engine.py` runs a drain thread — the same bug previously
affected `python/play.py`.

## Reward

| Term | Value |
| --- | --- |
| Win | +1.0 |
| Loss | −1.0 |
| Enemy HP removed | +0.010 each |
| Player HP lost | −0.020 each |
| Per step | −0.001 |

Terminal outcome dominates; the HP terms give dense signal so the agent learns
to block and to kill quickly rather than merely survive.

## Scaling

Measured on a Ryzen 7 7800X3D / RTX 5080: ~814 env steps/sec single process;
end-to-end training ~1540 fps at 8 envs, ~1780 at 12, ~1800 at 16. Past 12 the
returns flatten — the engine processes, not the GPU, are the bottleneck.

## Extending beyond combat

`encoding.py` owns the action space and `env.py` the episode boundary. To add
map routing, drafting, or shops, extend the action layout and let episodes span
a whole run — `engine.py` already speaks every decision type the protocol
exposes (`map_select`, `card_reward`, `shop`, `event_choice`, `rest_site`).

# Fork notes

Fork of [wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli) (MIT).

Two things live here that upstream does not have:

1. **Compatibility fixes** for Slay the Spire 2 **v0.110.1** and for running on
   Windows. Upstream was written against an older `sts2.dll` and does not build
   or run against this game build.
2. **`rl/`** — a reinforcement-learning stack that trains combat policies
   against the real engine (see [rl/README.md](rl/README.md)).

---

## 1. Engine compatibility (v0.110.1)

Verified against game build `v0.110.1` (commit `db5d3552`, 2026-07-31).
Upstream fails to compile against it with four errors.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `RunManager` has no `SetUpSavedSinglePlayer` | Renamed, and now returns `Task` | Call `SetUpSavedSingleplayer(...)` (lowercase `p`) and await it — `src/Sts2Headless/RunSimulator.cs` |
| `CreatureCmd.Damage` argument type errors | Overload gained a trailing `CardPlay` parameter | Pass `play` as the final argument |
| `TypeInitializationException` on `start_run` | `Sentry.Godot.dll` missing from `lib/` — the game's crash reporter auto-inits at module load | Added it to the DLL list in `setup.sh` |
| `ModManager is not finished initializing!` | v0.110.1 gates `ReflectionHelper.ModTypes` behind mod-system init | `ModManager.Initialize(...)` with a no-op file IO (headless has no mods dir) |
| `ModelIdSerializationCache used before it was initialized!` — broke card rewards, shops, saves | `ContentSorter` requires `AssemblyInfo` first | Call `AssemblyInfo.Init()` before `ModelIdSerializationCache.Init()` |

The last one is easy to miss: `ModelIdSerializationCache.Init()` failed *silently*
(caught and logged as a warning), so the damage only showed up much later as
`skip_card_reward` throwing.

> **Version coupling.** The game version string passed to
> `ModManager.Initialize` is currently hardcoded to `0.110.1` in
> `RunSimulator.cs`. A future game patch may need that bumped.

## 2. Windows fixes in `python/play.py`

Both made the interactive CLI unusable on Windows:

- **stderr deadlock (the CLI appeared to freeze at startup).** The engine's
  stderr was piped but never read. The engine logs heavily while registering
  ~1658 models, so the OS pipe buffer filled, the engine blocked writing to
  stderr, and it never answered on stdout. Both processes stayed alive at
  ~0% CPU with no error — indistinguishable from a hang. Fixed with a drain
  thread. `python/play_full_run.py` was unaffected only because it does not
  pipe stderr by default.
- **`cp1252` crash.** Windows consoles default to cp1252, which cannot encode
  the box-drawing characters and em-dashes in the UI, so the first event screen
  raised `UnicodeEncodeError`. Fixed by forcing UTF-8 on the console streams.

## 3. `rl/` — combat RL

Full detail in [rl/README.md](rl/README.md). Summary: one episode = one fight,
MaskablePPO over N parallel engine processes.

Measured on a Ryzen 7 7800X3D / RTX 5080:

| | Random legal play | Trained (1M steps) |
| --- | --- | --- |
| Win rate | 99.5% | 100% |
| HP retained on win | 48.1 | **71.9** |
| Steps to win | 18.6 | **9.1** |

That baseline encounter (`SHRINKER_BEETLE_WEAK`) is trivial enough that the
policy plateaued by ~100k steps; judge progress by HP retained, not win rate,
and train on a harder encounter mix for meaningful results.

---

## Tools and environment

| Component | Version / note |
| --- | --- |
| Slay the Spire 2 | v0.110.1 (Steam) — supplies `lib/*.dll`, never committed |
| .NET SDK | 9.0.312 |
| Python | 3.13 |
| PyTorch | 2.11.0+**cu128** — required for Blackwell (RTX 50xx, sm_120); plain `pip install torch` gives a CPU-only build on Windows |
| stable-baselines3 / sb3-contrib | 2.9.0 — `MaskablePPO` |
| Gymnasium | 1.3.0 |
| Mono.Cecil | 0.11.6 — IL patching in `setup.sh`, and used to inspect changed engine APIs |
| GitHub CLI | 2.97.0 |

## Build and run

```bash
./setup.sh "/path/to/Steam/steamapps/common/Slay the Spire 2"
python3 python/play.py --lang en --character Ironclad
```

`setup.sh` copies the game DLLs from your own Steam install into `lib/`, which
is **gitignored** — no proprietary game binaries are distributed by this repo.

RL:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r rl/requirements.txt
.venv/Scripts/python.exe -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe rl/train.py --envs 12 --steps 1000000
```

## Upstream

The compatibility and Windows fixes are not specific to this fork and would
apply to any user on v0.110.1 / Windows. They are kept in separate commits so
they can be cherry-picked into a pull request upstream.

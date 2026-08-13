#!/usr/bin/env python3
"""Replay a logged failed/hung run to reproduce and inspect it.

Training appends a JSON record per failure to rl/failures.jsonl (see --faillog):
the run's seed + character + the EXACT command sequence. This script spins up a
fresh engine on that seed and replays the commands, so the failure reproduces
deterministically (engine RNG is seeded) — then you can see exactly where it broke.

Usage (from the engine dir, with the venv python + x64 dotnet on PATH):
  python rl/replay.py                 # replay the LAST logged failure
  python rl/replay.py -1              # same
  python rl/replay.py 3               # replay record on line 3 (0-indexed)
  python rl/replay.py path.jsonl 3    # from a specific log file
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.engine import Engine, EngineError  # type: ignore


def main():
    args = sys.argv[1:]
    path = "rl/failures.jsonl"
    idx = -1
    for a in args:
        if a.lstrip("-").isdigit():
            idx = int(a)
        else:
            path = a
    if not os.path.exists(path):
        print(f"no failure log at {path}"); return
    lines = [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln.strip()]
    if not lines:
        print(f"{path} is empty"); return
    rec = json.loads(lines[idx])
    acts = rec.get("actions") or []
    print(f"=== replay: char={rec.get('character')} seed={rec.get('seed')} "
          f"reason={rec.get('reason')}  ({len(acts)} actions) ===")

    e = Engine(rec["character"], seed=rec["seed"], ascension=rec.get("ascension", 10))
    last = e.last
    try:
        for k, cmd in enumerate(acts):
            st = e.send(cmd)
            last = st if isinstance(st, dict) else last
            if st.get("type") == "error":
                a = cmd.get("action", cmd.get("cmd"))
                print(f"\n>>> REPRODUCED at step {k}/{len(acts)}: cmd={a} args={cmd.get('args')}")
                print(f"    message: {st.get('message')}")
                dec_before = acts[k - 1] if k else None
                print(f"    previous cmd: {dec_before}")
                break
        else:
            print(f"\nreplay completed without an engine error — final decision: "
                  f"{last.get('decision') or last.get('type')}")
            print("(a threading race, e.g. a card_select timing bug, may not reproduce "
                  "every time — rerun, or it's non-deterministic)")
    except EngineError as ex:
        print(f"\n>>> REPRODUCED hang/death: {str(ex)[:200]}")
    finally:
        print("\n--- engine stderr tail ---")
        for ln in e._stderr_tail[-15:]:
            print("  ", ln)
        e.close()


if __name__ == "__main__":
    main()

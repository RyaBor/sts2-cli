"""Encounter ids, read from the shipped localization tables.

    python rl/encounters.py            # list all, grouped by tier
    python rl/encounters.py --check    # also verify each one actually loads
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOC = os.path.join(ROOT, "localization_eng", "encounters.json")

TIERS = ("WEAK", "NORMAL", "ELITE", "BOSS")


def all_ids() -> list[str]:
    with open(LOC, encoding="utf-8") as f:
        data = json.load(f)
    ids = sorted({k.split(".", 1)[0] for k in data})
    return [i for i in ids if not i.startswith("DEPRECATED")]


def by_tier() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {t: [] for t in TIERS}
    out["OTHER"] = []
    for i in all_ids():
        for t in TIERS:
            if i.endswith("_" + t):
                out[t].append(i)
                break
        else:
            out["OTHER"].append(i)
    return out


def _check(ids: list[str]) -> None:
    import sys
    sys.path.insert(0, ROOT)
    from rl.engine import Engine, EngineError

    eng = Engine(seed="enc")
    ok, bad = [], []
    for i in ids:
        try:
            st = eng.reset_combat(encounter=i, hp=80, max_hp=80)
            n = len(st.get("enemies") or [])
            hp = sum(e.get("hp") or 0 for e in st.get("enemies") or [])
            ok.append((i, n, hp))
        except EngineError as e:
            bad.append((i, str(e)[:60]))
    eng.close()
    print(f"\nloadable: {len(ok)}/{len(ids)}")
    for i, n, hp in ok:
        print(f"  {i:42s} enemies={n} total_hp={hp}")
    for i, err in bad:
        print(f"  FAIL {i}: {err}")


if __name__ == "__main__":
    import sys

    groups = by_tier()
    for tier, ids in groups.items():
        if ids:
            print(f"\n=== {tier} ({len(ids)}) ===")
            for i in ids:
                print(" ", i)
    if "--check" in sys.argv:
        _check([i for t in TIERS for i in groups[t]])

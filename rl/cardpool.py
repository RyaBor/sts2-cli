"""Card ids that can actually be put into a deck.

`cards.json` lists every localization entry, and a few of those do not construct
as real cards. Injecting one via --deck-noise kills the episode and costs an
engine restart, so the pool is validated once and cached.

    python rl/cardpool.py            # build/refresh the cache
    python rl/cardpool.py --show     # list what was rejected
"""
from __future__ import annotations

import json
import os

from . import vocab

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "snapshots", "valid_cards.json")
BATCH = 12  # cards tested per engine reset


def build(verbose: bool = True) -> dict:
    from .engine import Engine, EngineError

    ids = vocab.cards().ids
    good: list[str] = []
    bad: list[str] = []
    state = {"eng": Engine(seed="pool")}

    try:
        # Test in batches, then bisect any failing batch — one reset per card
        # would be 600 resets, and almost every card is fine.
        def test(chunk: list[str]) -> bool:
            try:
                state["eng"].reset_combat(
                    encounter="SHRINKER_BEETLE_WEAK", hp=80, max_hp=80,
                    deck=["STRIKE_IRONCLAD"] * 3 + list(chunk),
                    relics=["BURNING_BLOOD"])
                return True
            except EngineError:
                # A failed enter_room can leave the run unusable, which would
                # then fail every later card and reject them all as collateral.
                # Replace the engine so each verdict is independent.
                try:
                    state["eng"].close()
                except Exception:
                    pass
                state["eng"] = Engine(seed="pool")
                return False

        def bisect(chunk: list[str]) -> None:
            if not chunk:
                return
            if test(chunk):
                good.extend(chunk)
                return
            if len(chunk) == 1:
                bad.append(chunk[0])
                return
            mid = len(chunk) // 2
            bisect(chunk[:mid])
            bisect(chunk[mid:])

        for i in range(0, len(ids), BATCH):
            bisect(ids[i:i + BATCH])
            if verbose and (i // BATCH) % 10 == 0:
                print(f"  {i + BATCH}/{len(ids)} tested, {len(bad)} rejected")
    finally:
        state["eng"].close()

    data = {"valid": good, "rejected": bad}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    if verbose:
        print(f"{len(good)} usable, {len(bad)} rejected -> {CACHE}")
    return data


def valid_cards() -> list[str]:
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)["valid"]
    return build(verbose=False)["valid"]


if __name__ == "__main__":
    import sys

    if "--show" in sys.argv and os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            d = json.load(f)
        print(f"valid: {len(d['valid'])}\nrejected ({len(d['rejected'])}): {d['rejected']}")
    else:
        build()

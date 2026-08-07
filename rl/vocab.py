"""Stable id vocabularies, read from the game's own localization tables.

Replaces the crc32 hashing trick. Hashing collided (606 cards into 64 buckets),
which is fine for a 10-card starter deck and lossy for a real one — and it gave
the network no way to learn per-entity behaviour, since two unrelated cards
could share an input.

Ordering is alphabetical and derived from shipped data, so every process agrees
without a checked-in file. Index 0 is always <unk>: an id the tables do not
contain (a modded card, a renamed entry) degrades to a shared embedding rather
than crashing or silently aliasing onto something real.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOC = os.path.join(ROOT, "localization_eng")

UNK = 0


def _ids_from(filename: str) -> list[str]:
    path = os.path.join(LOC, filename)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return sorted({k.split(".", 1)[0] for k in data})


class Vocab:
    """id -> dense index, with 0 reserved for unknown."""

    def __init__(self, ids: list[str], name: str):
        self.name = name
        self.ids = ids
        self._index = {v: i + 1 for i, v in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.ids) + 1  # + <unk>

    def index(self, key: str | None) -> int:
        if not key:
            return UNK
        # Card ids arrive as "CARD.STRIKE_IRONCLAD" or "STRIKE_IRONCLAD+";
        # normalise both the category prefix and the upgrade marker.
        k = key.split(".", 1)[-1].rstrip("+")
        hit = self._index.get(k)
        if hit is not None:
            return hit
        # Intent types come off the engine in PascalCase ("DebuffStrong") while
        # the tables use SCREAMING_SNAKE ("DEBUFF_STRONG").
        return self._index.get(_CAMEL.sub("_", k).upper(), UNK)


@lru_cache(maxsize=1)
def cards() -> Vocab:
    return Vocab(_ids_from("cards.json"), "cards")


@lru_cache(maxsize=1)
def monsters() -> Vocab:
    return Vocab(_ids_from("monsters.json"), "monsters")


@lru_cache(maxsize=1)
def relics() -> Vocab:
    return Vocab(_ids_from("relics.json"), "relics")


@lru_cache(maxsize=1)
def potions() -> Vocab:
    return Vocab(_ids_from("potions.json"), "potions")


@lru_cache(maxsize=1)
def intents() -> Vocab:
    return Vocab(_ids_from("intents.json"), "intents")


@lru_cache(maxsize=1)
def encounters() -> Vocab:
    """Boss/encounter ids. `context.boss.id` is an encounter id (THE_KIN_BOSS),
    not a monster id, so it needs its own table."""
    return Vocab(_ids_from("encounters.json"), "encounters")


def sizes() -> dict[str, int]:
    return {
        "cards": len(cards()),
        "monsters": len(monsters()),
        "relics": len(relics()),
        "potions": len(potions()),
        "intents": len(intents()),
        "encounters": len(encounters()),
    }


if __name__ == "__main__":
    for k, v in sizes().items():
        print(f"{k:10s} {v:5d} (incl. <unk>)")

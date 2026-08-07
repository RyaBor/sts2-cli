"""State -> observation vector, and the fixed action space with legality masks.

The engine's action space is variable (hand size and enemy count change every
turn), so we expose a fixed-size space and mask out illegal entries. Card
identity uses the hashing trick (crc32) rather than a vocab file: it is stable
across processes, which matters because SubprocVecEnv workers must agree on
the encoding.
"""
from __future__ import annotations

import zlib
from typing import Any

import numpy as np

MAX_HAND = 10
MAX_ENEMIES = 5
CARD_HASH = 64
POWER_HASH = 16

# --- action layout ---
#   [0, MAX_HAND*MAX_ENEMIES)      play card i targeting enemy j
#   [.., + MAX_HAND)               play card i (self / all-enemies / untargeted)
#   last                           end turn
TARGETED = MAX_HAND * MAX_ENEMIES
UNTARGETED = TARGETED + MAX_HAND
END_TURN = UNTARGETED
N_ACTIONS = END_TURN + 1

GLOBAL_FEATS = 8
ENEMY_FEATS = 7 + POWER_HASH
CARD_FEATS = 10 + CARD_HASH
OBS_DIM = GLOBAL_FEATS + POWER_HASH + MAX_ENEMIES * ENEMY_FEATS + MAX_HAND * CARD_FEATS


def _bucket(text: str, size: int) -> int:
    return zlib.crc32(text.encode("utf-8")) % size


def _alive(enemies: list[dict]) -> list[dict]:
    return [e for e in enemies if (e.get("hp") or 0) > 0]


def _powers_vec(powers: Any) -> np.ndarray:
    v = np.zeros(POWER_HASH, dtype=np.float32)
    if isinstance(powers, list):
        for p in powers:
            name = str(p.get("name", p) if isinstance(p, dict) else p)
            amt = p.get("amount", 1) if isinstance(p, dict) else 1
            try:
                amt = float(amt)
            except (TypeError, ValueError):
                amt = 1.0
            v[_bucket(name, POWER_HASH)] += amt / 10.0
    return v


def encode_obs(st: dict) -> np.ndarray:
    """Flatten a combat_play state into a fixed-length float32 vector."""
    out = np.zeros(OBS_DIM, dtype=np.float32)
    p = st.get("player") or {}
    hp = float(p.get("hp") or 0)
    max_hp = float(p.get("max_hp") or 1)

    out[0] = float(st.get("energy") or 0) / 10.0
    out[1] = float(st.get("max_energy") or 0) / 10.0
    out[2] = float(st.get("round") or 0) / 20.0
    out[3] = float(st.get("draw_pile_count") or 0) / 30.0
    out[4] = float(st.get("discard_pile_count") or 0) / 30.0
    out[5] = hp / max(max_hp, 1.0)
    out[6] = float(p.get("block") or 0) / 30.0
    out[7] = max_hp / 100.0

    i = GLOBAL_FEATS
    out[i:i + POWER_HASH] = _powers_vec(st.get("player_powers"))
    i += POWER_HASH

    for slot, e in enumerate(_alive(st.get("enemies") or [])[:MAX_ENEMIES]):
        b = i + slot * ENEMY_FEATS
        ehp = float(e.get("hp") or 0)
        emax = float(e.get("max_hp") or 1)
        intents = e.get("intents") or []
        dmg = sum(float(x.get("damage") or 0) for x in intents if isinstance(x, dict))
        out[b + 0] = 1.0
        out[b + 1] = ehp / max(emax, 1.0)
        out[b + 2] = ehp / 60.0
        out[b + 3] = float(e.get("block") or 0) / 30.0
        out[b + 4] = dmg / 30.0
        out[b + 5] = 1.0 if e.get("intends_attack") else 0.0
        out[b + 6] = len(intents) / 3.0
        out[b + 7:b + 7 + POWER_HASH] = _powers_vec(e.get("powers"))
    i += MAX_ENEMIES * ENEMY_FEATS

    for slot, c in enumerate((st.get("hand") or [])[:MAX_HAND]):
        b = i + slot * CARD_FEATS
        stats = c.get("stats") or {}
        ctype = str(c.get("type") or "")
        ttype = str(c.get("target_type") or "")
        out[b + 0] = 1.0
        out[b + 1] = float(c.get("cost") or 0) / 3.0
        out[b + 2] = 1.0 if ctype == "Attack" else 0.0
        out[b + 3] = 1.0 if ctype == "Skill" else 0.0
        out[b + 4] = 1.0 if ctype == "Power" else 0.0
        out[b + 5] = 1.0 if ctype in ("Status", "Curse") else 0.0
        out[b + 6] = float(stats.get("damage") or 0) / 30.0
        out[b + 7] = float(stats.get("block") or 0) / 30.0
        out[b + 8] = 1.0 if c.get("can_play") else 0.0
        out[b + 9] = 1.0 if ttype == "AnyEnemy" else 0.0
        out[b + 10 + _bucket(str(c.get("id") or c.get("name") or ""), CARD_HASH)] = 1.0

    return out


def action_mask(st: dict) -> np.ndarray:
    """True where the action is currently legal."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[END_TURN] = True  # ending the turn is always allowed

    alive = _alive(st.get("enemies") or [])
    n_alive = min(len(alive), MAX_ENEMIES)

    for slot, c in enumerate((st.get("hand") or [])[:MAX_HAND]):
        if not c.get("can_play"):
            continue
        if str(c.get("target_type") or "") == "AnyEnemy":
            for j in range(n_alive):
                mask[slot * MAX_ENEMIES + j] = True
        else:
            mask[TARGETED + slot] = True
    return mask


def decode_action(action: int, st: dict) -> tuple[str, dict]:
    """Map an action index to an engine command."""
    if action >= END_TURN:
        return "end_turn", {}

    if action >= TARGETED:
        slot = action - TARGETED
        card = (st.get("hand") or [])[slot]
        return "play_card", {"card_index": card["index"]}

    slot, tgt = divmod(action, MAX_ENEMIES)
    card = (st.get("hand") or [])[slot]
    alive = _alive(st.get("enemies") or [])
    args = {"card_index": card["index"]}
    if tgt < len(alive):
        args["target_index"] = alive[tgt].get("index", tgt)
    return "play_card", args

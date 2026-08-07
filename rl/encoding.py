"""State -> observation, and the fixed action space with legality masks.

The observation is a Dict of entity *ids* plus numeric features, not one flat
vector. Ids index real vocabularies (see `vocab.py`) so the network can learn
per-card and per-monster behaviour through embeddings — a Cultist and a Jaw
Worm at the same HP are no longer identical inputs, which is what any
boss-aware or targeting-aware play depends on.

Padding convention: id 0 is <unk>/empty everywhere, and each entity group ships
a `*_mask` so the extractor can ignore padded slots instead of attending to
imaginary enemies.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from . import vocab

MAX_HAND = 10
MAX_ENEMIES = 5
MAX_INTENTS = 3
MAX_RELICS = 16
MAX_POTIONS = 5
MAX_PILE = 40      # cards summarised per pile (draw / discard / exhaust)

# --- action layout ---
#   [0, MAX_HAND*MAX_ENEMIES)       play card i targeting enemy j
#   [.., + MAX_HAND)                play card i (self / all-enemies / untargeted)
#   [.., + MAX_POTIONS*MAX_ENEMIES) drink potion p targeting enemy j
#   [.., + MAX_POTIONS)             drink potion p (untargeted)
#   last                            end turn
TARGETED = MAX_HAND * MAX_ENEMIES
UNTARGETED = TARGETED + MAX_HAND
POTION_TARGETED = UNTARGETED + MAX_POTIONS * MAX_ENEMIES
POTION_UNTARGETED = POTION_TARGETED + MAX_POTIONS
END_TURN = POTION_UNTARGETED
N_ACTIONS = END_TURN + 1

GLOBAL_FEATS = 10
HAND_FEATS = 9
ENEMY_FEATS = 8


def observation_space() -> spaces.Dict:
    f32 = np.float32
    return spaces.Dict({
        "global": spaces.Box(-10.0, 10.0, (GLOBAL_FEATS,), dtype=f32),
        "boss_id": spaces.Box(0, 1e6, (1,), dtype=f32),
        "hand_ids": spaces.Box(0, 1e6, (MAX_HAND,), dtype=f32),
        "hand_feats": spaces.Box(-10.0, 10.0, (MAX_HAND, HAND_FEATS), dtype=f32),
        "hand_mask": spaces.Box(0.0, 1.0, (MAX_HAND,), dtype=f32),
        "enemy_ids": spaces.Box(0, 1e6, (MAX_ENEMIES,), dtype=f32),
        "enemy_feats": spaces.Box(-10.0, 10.0, (MAX_ENEMIES, ENEMY_FEATS), dtype=f32),
        "enemy_intents": spaces.Box(0, 1e6, (MAX_ENEMIES, MAX_INTENTS), dtype=f32),
        "enemy_mask": spaces.Box(0.0, 1.0, (MAX_ENEMIES,), dtype=f32),
        "relic_ids": spaces.Box(0, 1e6, (MAX_RELICS,), dtype=f32),
        "potion_ids": spaces.Box(0, 1e6, (MAX_POTIONS,), dtype=f32),
        "draw_ids": spaces.Box(0, 1e6, (MAX_PILE,), dtype=f32),
        "discard_ids": spaces.Box(0, 1e6, (MAX_PILE,), dtype=f32),
        "exhaust_ids": spaces.Box(0, 1e6, (MAX_PILE,), dtype=f32),
    })


def _alive(enemies: Any) -> list[dict]:
    return [e for e in (enemies or []) if (e.get("hp") or 0) > 0]


def _pile_ids(entries: Any, cv: vocab.Vocab) -> np.ndarray:
    out = np.zeros(MAX_PILE, dtype=np.float32)
    if isinstance(entries, list):
        for i, e in enumerate(entries[:MAX_PILE]):
            out[i] = cv.index(str(e))
    return out


def encode_obs(st: dict) -> dict[str, np.ndarray]:
    cv, mv, rv, pv, iv = (vocab.cards(), vocab.monsters(), vocab.relics(),
                          vocab.potions(), vocab.intents())
    ev = vocab.encounters()
    p = st.get("player") or {}
    hp = float(p.get("hp") or 0)
    max_hp = float(p.get("max_hp") or 1)
    alive = _alive(st.get("enemies"))

    g = np.zeros(GLOBAL_FEATS, dtype=np.float32)
    g[0] = float(st.get("energy") or 0) / 10.0
    g[1] = float(st.get("max_energy") or 0) / 10.0
    g[2] = float(st.get("round") or 0) / 20.0
    g[3] = float(st.get("draw_pile_count") or 0) / 30.0
    g[4] = float(st.get("discard_pile_count") or 0) / 30.0
    g[5] = float(st.get("exhaust_pile_count") or 0) / 30.0
    g[6] = hp / max(max_hp, 1.0)
    g[7] = float(p.get("block") or 0) / 30.0
    g[8] = max_hp / 100.0
    g[9] = len(alive) / float(MAX_ENEMIES)

    hand_ids = np.zeros(MAX_HAND, dtype=np.float32)
    hand_feats = np.zeros((MAX_HAND, HAND_FEATS), dtype=np.float32)
    hand_mask = np.zeros(MAX_HAND, dtype=np.float32)
    for i, c in enumerate((st.get("hand") or [])[:MAX_HAND]):
        stats = c.get("stats") or {}
        ctype = str(c.get("type") or "")
        hand_ids[i] = cv.index(str(c.get("id") or c.get("name") or ""))
        hand_mask[i] = 1.0
        f = hand_feats[i]
        f[0] = float(c.get("cost") or 0) / 3.0
        f[1] = 1.0 if ctype == "Attack" else 0.0
        f[2] = 1.0 if ctype == "Skill" else 0.0
        f[3] = 1.0 if ctype == "Power" else 0.0
        f[4] = 1.0 if ctype in ("Status", "Curse") else 0.0
        f[5] = float(stats.get("damage") or 0) / 30.0
        f[6] = float(stats.get("block") or 0) / 30.0
        f[7] = 1.0 if c.get("can_play") else 0.0
        f[8] = 1.0 if str(c.get("target_type") or "") == "AnyEnemy" else 0.0

    enemy_ids = np.zeros(MAX_ENEMIES, dtype=np.float32)
    enemy_feats = np.zeros((MAX_ENEMIES, ENEMY_FEATS), dtype=np.float32)
    enemy_intents = np.zeros((MAX_ENEMIES, MAX_INTENTS), dtype=np.float32)
    enemy_mask = np.zeros(MAX_ENEMIES, dtype=np.float32)
    for i, e in enumerate(alive[:MAX_ENEMIES]):
        ehp = float(e.get("hp") or 0)
        emax = float(e.get("max_hp") or 1)
        intents = [x for x in (e.get("intents") or []) if isinstance(x, dict)]
        dmg = sum(float(x.get("damage") or 0) for x in intents)
        enemy_ids[i] = mv.index(str(e.get("id") or ""))
        enemy_mask[i] = 1.0
        f = enemy_feats[i]
        f[0] = ehp / max(emax, 1.0)
        f[1] = ehp / 60.0
        f[2] = emax / 100.0
        f[3] = float(e.get("block") or 0) / 30.0
        f[4] = dmg / 30.0
        f[5] = 1.0 if e.get("intends_attack") else 0.0
        f[6] = len(intents) / float(MAX_INTENTS)
        # Incoming damage as a fraction of what we have left to lose.
        f[7] = min(2.0, dmg / max(hp, 1.0))
        for j, x in enumerate(intents[:MAX_INTENTS]):
            enemy_intents[i, j] = iv.index(str(x.get("type") or ""))

    relic_ids = np.zeros(MAX_RELICS, dtype=np.float32)
    for i, r in enumerate((p.get("relics") or [])[:MAX_RELICS]):
        relic_ids[i] = rv.index(str(r.get("id") or r.get("name") or ""))

    potion_ids = np.zeros(MAX_POTIONS, dtype=np.float32)
    for i, po in enumerate((p.get("potions") or [])[:MAX_POTIONS]):
        potion_ids[i] = pv.index(str(po.get("id") or po.get("name") or ""))

    boss = (st.get("context") or {}).get("boss") or {}

    return {
        "global": g,
        "boss_id": np.array([ev.index(str(boss.get("id") or ""))], dtype=np.float32),
        "hand_ids": hand_ids,
        "hand_feats": hand_feats,
        "hand_mask": hand_mask,
        "enemy_ids": enemy_ids,
        "enemy_feats": enemy_feats,
        "enemy_intents": enemy_intents,
        "enemy_mask": enemy_mask,
        "relic_ids": relic_ids,
        "potion_ids": potion_ids,
        "draw_ids": _pile_ids(st.get("draw_pile"), cv),
        "discard_ids": _pile_ids(st.get("discard_pile"), cv),
        "exhaust_ids": _pile_ids(st.get("exhaust_pile"), cv),
    }


def action_mask(st: dict) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[END_TURN] = True  # ending the turn is always allowed

    n_alive = min(len(_alive(st.get("enemies"))), MAX_ENEMIES)

    for slot, c in enumerate((st.get("hand") or [])[:MAX_HAND]):
        if not c.get("can_play"):
            continue
        if str(c.get("target_type") or "") == "AnyEnemy":
            for j in range(n_alive):
                mask[slot * MAX_ENEMIES + j] = True
        else:
            mask[TARGETED + slot] = True

    player = st.get("player") or {}
    for slot, po in enumerate((player.get("potions") or [])[:MAX_POTIONS]):
        if po.get("can_use") is False:
            continue
        if str(po.get("target_type") or "") == "AnyEnemy":
            for j in range(n_alive):
                mask[UNTARGETED + slot * MAX_ENEMIES + j] = True
        else:
            mask[POTION_TARGETED + slot] = True
    return mask


def decode_action(action: int, st: dict) -> tuple[str, dict]:
    """Map an action index to an engine command."""
    if action >= END_TURN:
        return "end_turn", {}

    alive = _alive(st.get("enemies"))
    args: dict[str, Any]

    if action >= POTION_TARGETED:
        return "use_potion", {"potion_index": action - POTION_TARGETED}

    if action >= UNTARGETED:
        slot, tgt = divmod(action - UNTARGETED, MAX_ENEMIES)
        args = {"potion_index": slot}
        if tgt < len(alive):
            args["target_index"] = alive[tgt].get("index", tgt)
        return "use_potion", args

    if action >= TARGETED:
        slot = action - TARGETED
        card = (st.get("hand") or [])[slot]
        return "play_card", {"card_index": card["index"]}

    slot, tgt = divmod(action, MAX_ENEMIES)
    card = (st.get("hand") or [])[slot]
    args = {"card_index": card["index"]}
    if tgt < len(alive):
        args["target_index"] = alive[tgt].get("index", tgt)
    return "play_card", args

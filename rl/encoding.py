"""State -> observation vector, and the fixed action space with legality masks.

The engine's action space is variable (hand size and enemy count change every
turn), so we expose a fixed-size space and mask out illegal entries.

Card identity is a deterministic vocabulary built from the game's card loc table
(~606 ids, sorted → stable across processes); unknown ids hash into a small tail.
Enemy intent *type* (Attack/Defend/Buff/Debuff/...) and *owned relics* are hashed
multi-hots, so the combat agent can condition play on them — important for a
policy meant to generalize across relic sets (e.g. as an MCTS prior). Powers,
potions, orbs (Defect), stars (Regent) and Osty (Necrobinder) are also encoded.
Incoming/outgoing damage is the sim-resolved value (all powers folded in).
"""
from __future__ import annotations

import zlib
from typing import Any

import numpy as np

# Hash/cap sizes are set from the ACTUAL game entity counts (localization_eng/*.json):
# 606 cards, 284 powers, 309 relics, 64 potions, 24 enchantments + 10 afflictions,
# 14 intents, 7 orbs, 7 keywords. Identity hashes are sized comfortably above the count
# so distinct entities rarely collide.
MAX_HAND = 10
MAX_ENEMIES = 5
MAX_POTIONS = 5           # belt is 3 by default but relics (Potion Belt, etc.) expand it
POWER_HASH = 128          # 284 powers; few active at once so collisions among them are rare
POTION_HASH = 64          # 64 potions -> ~collision-free
INTENT_TYPE_HASH = 16     # 14 intent types -> collision-free
RELIC_HASH = 128          # 309 relics; you hold a handful at once
CARD_VOCAB_SIZE = 640     # fixed; ~606 real cards + an unknown tail (see _card_index)
# A card's *behavior* changes when upgraded or enchanted while its id stays the same.
# We capture that with a FACTORED representation: shared base identity (vocab) plus an
# upgrade level, an enchant marker, the card's full resolved effect profile (STATS_HASH:
# damage/block/vulnerable/weak/draw/... — this is what upgrade/enchant actually change),
# and keyword flags. So Strike / Strike+ / enchanted-Strike are distinct input vectors
# without exploding the vocab into a slot per (card × level × enchant).
CARD_STATS_HASH = 32      # hashed effect profile from the card's `stats` dict (~36 modifiers)
CARD_KW_HASH = 12         # 7 keywords (Exhaust/Ethereal/Innate/Retain/...) -> collision-free
CARD_ENCH_HASH = 48       # 24 enchantments + 10 afflictions -> collision-free

# --- action layout ---
#   [0, MAX_HAND*MAX_ENEMIES)      play card i targeting enemy j
#   [.., + MAX_HAND)               play card i (self / all-enemies / untargeted)
#   END_TURN                       end turn
#   [POTION_TGT, +MAX_POTIONS*MAX_ENEMIES)   use potion i targeting enemy j
#   [POTION_UNTGT, +MAX_POTIONS)             use potion i (untargeted)
TARGETED = MAX_HAND * MAX_ENEMIES
UNTARGETED = TARGETED + MAX_HAND
END_TURN = UNTARGETED
POTION_TGT = END_TURN + 1
POTION_UNTGT = POTION_TGT + MAX_POTIONS * MAX_ENEMIES
POTION_DISCARD = POTION_UNTGT + MAX_POTIONS          # ditch potion i to free a slot
N_ACTIONS = POTION_DISCARD + MAX_POTIONS

# Potions that never appear as a manual action (auto-trigger on death, etc.).
AUTO_ONLY_POTIONS = frozenset({"FAIRY_POTION", "FAIRY_IN_A_BOTTLE"})

MAX_ORBS = 10          # Defect: 3 slots default, grows with Focus/relics
ORB_HASH = 8           # 7 orb types -> collision-free
GLOBAL_FEATS = 12      # +stars (Regent) +encounter-tier one-hot (normal/elite/boss)
ENEMY_FEATS = 7 + POWER_HASH + INTENT_TYPE_HASH   # +intent-type multi-hot
POTION_FEATS = 2 + POTION_HASH
ORB_FEATS = 3 + ORB_HASH           # present, passive, evoke, type-hash
OSTY_FEATS = 3                     # alive, hp-fraction, block (Necrobinder)

# Card identity is learned via an nn.Embedding (see agents.CombatNet), NOT a one-hot,
# so the combat encoding is split into a DENSE feature vector + a per-slot card-id
# vector. Dense per-card features (everything except identity): the scalars, the
# hashed effect profile, keywords, enchant — these carry upgrade/enchant behavior.
CARD_SCALARS = 13   # 11 base + upgrade_level + enchant_amount
CARD_DENSE = CARD_SCALARS + CARD_STATS_HASH + CARD_KW_HASH + CARD_ENCH_HASH   # per-slot dense
EMBED_DIM = 32
PAD_CARD = CARD_VOCAB_SIZE          # embedding padding index for empty hand slots
DENSE_DIM = (GLOBAL_FEATS + POWER_HASH + MAX_ENEMIES * ENEMY_FEATS
             + MAX_HAND * CARD_DENSE + MAX_POTIONS * POTION_FEATS
             + MAX_ORBS * ORB_FEATS + OSTY_FEATS + RELIC_HASH)


def _potions(st: dict) -> list:
    return (st.get("player") or {}).get("potions") or []


def _bucket(text: str, size: int) -> int:
    return zlib.crc32(text.encode("utf-8")) % size


def _load_card_vocab() -> dict:
    """Deterministic card-id → index map, built from the game's own card loc table
    (localization_eng/cards.json). Keys there are the card Entry ids ("ABRASIVE",
    "STRIKE_IRONCLAD", ...), matching a combat card's id ("CARD.STRIKE_IRONCLAD")
    after stripping the "CARD." prefix. Sorted so the mapping is stable across
    processes and machines. Empty dict if the file is unavailable (→ everything
    falls into the unknown tail, still deterministic)."""
    import os
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "localization_eng", "cards.json")
    try:
        keys = json.load(open(path, encoding="utf-8")).keys()
        ids = sorted({k.rsplit(".", 1)[0] for k in keys if "." in k})
        return {cid: i for i, cid in enumerate(ids) if i < CARD_VOCAB_SIZE}
    except Exception:
        return {}


_CARD_VOCAB = _load_card_vocab()
# Cards not in the vocab (new/unknown ids) hash into a small reserved tail so
# identity stays collision-light without changing OBS_DIM.
_UNK_BASE = min(len(_CARD_VOCAB), CARD_VOCAB_SIZE - 32)
_UNK_SPAN = CARD_VOCAB_SIZE - _UNK_BASE


def _card_index(cid) -> int:
    """Vocab index for a card id. Strips the "CARD." prefix; unknown ids hash into
    the reserved tail [_UNK_BASE, CARD_VOCAB_SIZE)."""
    s = str(cid or "")
    base = s.rsplit(".", 1)[-1] if "." in s else s
    idx = _CARD_VOCAB.get(base)
    if idx is not None:
        return idx
    return _UNK_BASE + _bucket(base, _UNK_SPAN)


def _alive(enemies: list[dict]) -> list[dict]:
    return [e for e in enemies if (e.get("hp") or 0) > 0]


def _card_damage(c: dict) -> float:
    """Best available damage for a hand card. `stats['damage']` is 0 for cards whose
    damage is computed (Unleash → `calculateddamage` = base + Osty HP) or multi-hit;
    `damage_by_target` is the fully resolved value (Vulnerable, multi-hit totals). Use
    the most informative one so the agent actually sees a card's real damage."""
    dbt = c.get("damage_by_target") or []
    if dbt:
        return sum(float(x.get("total_damage") or x.get("damage") or 0)
                   for x in dbt if isinstance(x, dict))
    stats = c.get("stats") or {}
    for k in ("calculateddamage", "damage"):
        v = stats.get(k)
        if v:
            return float(v)
    return 0.0


def _stats_vec(stats, size: int) -> np.ndarray:
    """Hash a card's `stats` dict (its resolved effect profile: damage, block,
    vulnerablepower, weakpower, draw, summon, stars, ...) into a fixed vector. This
    is what upgrading/enchanting actually changes, so it makes upgraded/enchanted
    variants distinct even though they share a card id."""
    v = np.zeros(size, dtype=np.float32)
    if isinstance(stats, dict):
        for k, val in stats.items():
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            v[_bucket(str(k), size)] += fv / 10.0
    return v


def _intent_damage(intents: list) -> float:
    """Total incoming attack damage. Multi-hit intents carry `total_damage` (= per-hit
    * hits); plain `damage` alone undercounts them, so prefer total_damage."""
    return sum(float(x.get("total_damage") or x.get("damage") or 0)
               for x in intents if isinstance(x, dict))


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


def _write_card_dense(c: dict, out: np.ndarray, base: int) -> int:
    """Write a card's CARD_DENSE feature block into out[base:base+CARD_DENSE] and
    return its vocab id (for the embedding). Shared by the hand encoder and the
    in-combat selection encoder so their card features never drift apart."""
    stats = c.get("stats") or {}
    ctype = str(c.get("type") or "")
    ttype = str(c.get("target_type") or "")
    out[base + 0] = 1.0
    out[base + 1] = float(c.get("cost") or 0) / 3.0
    out[base + 2] = 1.0 if ctype == "Attack" else 0.0
    out[base + 3] = 1.0 if ctype == "Skill" else 0.0
    out[base + 4] = 1.0 if ctype == "Power" else 0.0
    out[base + 5] = 1.0 if ctype in ("Status", "Curse") else 0.0
    out[base + 6] = _card_damage(c) / 30.0
    out[base + 7] = float(stats.get("block") or 0) / 30.0
    out[base + 8] = 1.0 if c.get("can_play") else 0.0
    out[base + 9] = 1.0 if ttype == "AnyEnemy" else 0.0
    out[base + 10] = float(c.get("star_cost") or 0) / 3.0
    out[base + 11] = float(c.get("upgrade_level") or (1 if c.get("upgraded") else 0)) / 2.0
    out[base + 12] = (float(c.get("enchantment_amount") or 0)
                      + float(c.get("affliction_amount") or 0)) / 10.0
    sb = base + CARD_SCALARS
    out[sb:sb + CARD_STATS_HASH] += _stats_vec(stats, CARD_STATS_HASH)
    kb = sb + CARD_STATS_HASH
    for kw in (c.get("keywords") or []):
        out[kb + _bucket(str(kw), CARD_KW_HASH)] = 1.0
    eb = kb + CARD_KW_HASH
    for tag in (c.get("enchantment"), c.get("affliction")):
        if tag:
            out[eb + _bucket(str(tag), CARD_ENCH_HASH)] = 1.0
    return _card_index(c.get("id") or c.get("name"))


# in-combat card selection (Armaments/Dual Wield/Exhume/discover/...): the combat
# agent scores candidate cards. All variants (draw↔hand↔discard↔exhaust moves, play-
# from-pile, upgrade, etc.) arrive as this one `card_select` decision.
MAX_SEL = 10
SEL_GLOBAL = 4            # min_select, max_select, n_candidates, hp-fraction


def encode_select(st: dict):
    """Encode an in-combat card_select as (glob, cand_dense, cand_ids, n)."""
    glob = np.zeros(SEL_GLOBAL, dtype=np.float32)
    cand_dense = np.zeros((MAX_SEL, CARD_DENSE), dtype=np.float32)
    cand_ids = np.full(MAX_SEL, PAD_CARD, dtype=np.int64)
    cards = st.get("cards") or []
    p = st.get("player") or {}
    glob[0] = float(st.get("min_select") or 1) / 5.0
    glob[1] = float(st.get("max_select") or 1) / 5.0
    glob[2] = len(cards) / MAX_SEL
    glob[3] = float(p.get("hp") or 0) / max(float(p.get("max_hp") or 1), 1)
    for i, c in enumerate(cards[:MAX_SEL]):
        cand_ids[i] = _write_card_dense(c, cand_dense[i], 0)
    return glob, cand_dense, cand_ids, len(cards)


def encode_combat(st: dict):
    """Encode a combat_play state as (dense, card_ids):
      dense    : float32[DENSE_DIM] — everything except card identity.
      card_ids : int64[MAX_HAND]    — vocab index per hand slot (PAD_CARD = empty),
                 consumed by an nn.Embedding in the combat net.
    Card behavior (upgrade/enchant) still lands in `dense` via the per-card scalars,
    the hashed effect profile, keywords and enchant markers — only the *identity*
    moves to the embedding."""
    out = np.zeros(DENSE_DIM, dtype=np.float32)
    card_ids = np.full(MAX_HAND, PAD_CARD, dtype=np.int64)
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
    out[8] = float(st.get("stars") or 0) / 10.0        # Regent star economy
    # encounter tier one-hot (normal / elite / boss) from the authoritative room type
    rt = str((st.get("context") or {}).get("room_type") or "").upper()
    out[9] = 1.0 if ("BOSS" not in rt and "ELITE" not in rt) else 0.0   # normal
    out[10] = 1.0 if "ELITE" in rt else 0.0
    out[11] = 1.0 if "BOSS" in rt else 0.0

    i = GLOBAL_FEATS
    out[i:i + POWER_HASH] = _powers_vec(st.get("player_powers"))
    i += POWER_HASH

    for slot, e in enumerate(_alive(st.get("enemies") or [])[:MAX_ENEMIES]):
        b = i + slot * ENEMY_FEATS
        ehp = float(e.get("hp") or 0)
        emax = float(e.get("max_hp") or 1)
        intents = e.get("intents") or []
        dmg = _intent_damage(intents)
        out[b + 0] = 1.0
        out[b + 1] = ehp / max(emax, 1.0)
        out[b + 2] = ehp / 60.0
        out[b + 3] = float(e.get("block") or 0) / 30.0
        out[b + 4] = dmg / 30.0
        out[b + 5] = 1.0 if e.get("intends_attack") else 0.0
        out[b + 6] = len(intents) / 3.0
        out[b + 7:b + 7 + POWER_HASH] = _powers_vec(e.get("powers"))
        it_base = b + 7 + POWER_HASH        # intent-type multi-hot (Attack/Defend/Buff/...)
        for it in intents:
            t = str(it.get("type") or "") if isinstance(it, dict) else ""
            if t:
                out[it_base + _bucket(t, INTENT_TYPE_HASH)] = 1.0
    i += MAX_ENEMIES * ENEMY_FEATS

    for slot, c in enumerate((st.get("hand") or [])[:MAX_HAND]):
        card_ids[slot] = _write_card_dense(c, out, i + slot * CARD_DENSE)         # identity → embedding
    i += MAX_HAND * CARD_DENSE

    for slot, pt in enumerate(_potions(st)[:MAX_POTIONS]):
        b = i + slot * POTION_FEATS
        out[b + 0] = 1.0
        out[b + 1] = 1.0 if str(pt.get("target_type") or "") == "AnyEnemy" else 0.0
        out[b + 2 + _bucket(str(pt.get("id") or pt.get("name") or ""), POTION_HASH)] = 1.0
    i += MAX_POTIONS * POTION_FEATS

    for slot, orb in enumerate((st.get("orbs") or [])[:MAX_ORBS]):       # Defect
        b = i + slot * ORB_FEATS
        out[b + 0] = 1.0
        out[b + 1] = float(orb.get("passive") or 0) / 20.0
        out[b + 2] = float(orb.get("evoke") or 0) / 20.0
        out[b + 3 + _bucket(str(orb.get("type") or orb.get("name") or ""), ORB_HASH)] = 1.0
    i += MAX_ORBS * ORB_FEATS

    osty = st.get("osty") or {}                                          # Necrobinder
    if osty.get("alive"):
        out[i + 0] = 1.0
        out[i + 1] = float(osty.get("hp") or 0) / max(float(osty.get("max_hp") or 1), 1)
        out[i + 2] = float(osty.get("block") or 0) / 30.0
    i += OSTY_FEATS

    # owned relics — hashed multi-hot, so the combat agent can condition play on
    # relic effects (Burning Blood, Strength/energy relics, attack/block triggers).
    for r in (p.get("relics") or []):
        nm = (r.get("name") or r.get("id")) if isinstance(r, dict) else r
        if nm:
            out[i + _bucket(str(nm), RELIC_HASH)] = 1.0
    i += RELIC_HASH

    return out, card_ids


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

    for slot, pt in enumerate(_potions(st)[:MAX_POTIONS]):
        if not pt.get("can_use_in_combat") or str(pt.get("id") or "") in AUTO_ONLY_POTIONS:
            continue
        if str(pt.get("target_type") or "") == "AnyEnemy":
            for j in range(n_alive):
                mask[POTION_TGT + slot * MAX_ENEMIES + j] = True
        else:
            mask[POTION_UNTGT + slot] = True

    for slot in range(min(len(_potions(st)), MAX_POTIONS)):
        mask[POTION_DISCARD + slot] = True     # a held potion can always be ditched
    return mask


def decode_action(action: int, st: dict) -> tuple[str, dict]:
    """Map an action index to an engine command."""
    alive = _alive(st.get("enemies") or [])

    if action >= POTION_DISCARD:                     # discard potion i (free a slot)
        pt = _potions(st)[action - POTION_DISCARD]
        return "discard_potion", {"potion_index": pt.get("index", action - POTION_DISCARD)}
    if action >= POTION_UNTGT:                      # use potion i (untargeted)
        pt = _potions(st)[action - POTION_UNTGT]
        return "use_potion", {"potion_index": pt.get("index", action - POTION_UNTGT)}
    if action >= POTION_TGT:                         # use potion i @ enemy j
        slot, tgt = divmod(action - POTION_TGT, MAX_ENEMIES)
        pt = _potions(st)[slot]
        args = {"potion_index": pt.get("index", slot)}
        if tgt < len(alive):
            args["target_index"] = alive[tgt].get("index", tgt)
        return "use_potion", args
    if action == END_TURN:
        return "end_turn", {}
    if action >= TARGETED:                           # play card i (untargeted)
        card = (st.get("hand") or [])[action - TARGETED]
        return "play_card", {"card_index": card["index"]}

    slot, tgt = divmod(action, MAX_ENEMIES)          # play card i @ enemy j
    card = (st.get("hand") or [])[slot]
    args = {"card_index": card["index"]}
    if tgt < len(alive):
        args["target_index"] = alive[tgt].get("index", tgt)
    return "play_card", args

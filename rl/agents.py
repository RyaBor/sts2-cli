"""Three cooperating policy agents for full A10 runs, trained by REINFORCE.

  combat_agent : plays cards in combat        (reward = HP retained this combat)
  card_agent   : card rewards + shop + events/ancients   (reward = game victory)
  path_agent   : picks the next map node      (reward = overall game victory)

Combat reuses the fixed obs/action space in encoding.py. Card/path use the
compact encoders below. Each Agent samples during self-play (no grad) and does
a batched actor-critic update afterwards (value head is the REINFORCE baseline).
"""
from __future__ import annotations

import zlib
import numpy as np
import torch
import torch.nn as nn

from encoding import (N_ACTIONS, DENSE_DIM, EMBED_DIM, CARD_VOCAB_SIZE, PAD_CARD,
                      MAX_HAND, CARD_DENSE, CARD_ENCH_HASH, SEL_GLOBAL, MAX_SEL,
                      EVENT_VOCAB_SIZE, _card_index, _event_index, _write_card_dense,
                      encode_combat, encode_select, action_mask, decode_action)  # noqa: F401
import torch.nn.functional as F


def _bucket(s, n: int) -> int:
    return zlib.crc32(str(s).encode("utf-8")) % n


# ── card-reward encoding ───────────────────────────────────────────────────
# MAX_OFFER covers rewards enlarged by relics/effects (base 3, +cards from relics).
MAX_OFFER = 6
CARD_HASH = 256          # offered-card identity (606 cards); supplements the rich card features
# Each offered card is encoded EXACTLY like a combat hand card (encoding._write_card_dense):
# cost/type/resolved-damage/block + upgrade level + enchant hash&amount + keywords + effect
# profile — so drafting sees a card's real behavior (upgraded/enchanted) the way combat does.
# Plus an owned-count and a hashed identity.
CARD_FEATS = CARD_DENSE + 1 + CARD_HASH   # combat-style card features + owned-count + id-hash
# Reachable combat-tier counts from the current map position (fights this draft is for).
AHEAD_ROOMS = ("MONSTER", "ELITE", "BOSS", "SHOP", "RESTSITE", "TREASURE")
# Deck makeup: 4 type tallies + upgraded/enchanted/afflicted tallies, then TWO collision-free
# count vectors over the 606-card vocab — all copies, and upgraded copies (so upgraded vs base
# copies are distinguishable per card) — then an enchant-type hash (which enchants/afflictions
# the deck carries, and how many). Handles clones / duplicated decks.
DECK_FEATS = 7 + 2 * CARD_VOCAB_SIZE + CARD_ENCH_HASH
CS_GLOBAL = 6 + len(AHEAD_ROOMS)    # hp, deck_size, act, floor, gold, n_offered + ahead tiers
CS_OBS = CS_GLOBAL + MAX_OFFER * CARD_FEATS + DECK_FEATS
CS_ACTIONS = MAX_OFFER + 1          # pick offered card i, or SKIP (last)


def _ctx(state, key, default=0):
    return (state.get("context") or {}).get(key) or state.get(key) or default


def _card_key(c) -> str:
    return str(c.get("id") or c.get("name") or "")


def encode_card_reward(state) -> np.ndarray:
    out = np.zeros(CS_OBS, np.float32)
    p = state.get("player") or {}
    deck = p.get("deck") or []
    # count how many of each card id we already own (for duplicate awareness)
    owned = {}
    for dc in deck:
        owned[_card_key(dc)] = owned.get(_card_key(dc), 0) + 1
    out[0] = float(p.get("hp") or 0) / max(float(p.get("max_hp") or 1), 1)
    out[1] = float(p.get("deck_size") or 0) / 40.0
    out[2] = float(_ctx(state, "act", 1)) / 3.0
    out[3] = float(_ctx(state, "floor", 0)) / 50.0
    out[4] = float(p.get("gold") or 0) / 300.0       # gold-aware drafting
    cards = state.get("cards") or []
    out[5] = len(cards) / MAX_OFFER
    # reachable combat-tier counts this act (draft for the elites/bosses/normals ahead)
    ahead = {str(k).upper(): v for k, v in (state.get("ahead") or {}).items()}
    for j, room in enumerate(AHEAD_ROOMS):
        out[6 + j] = float(ahead.get(room, 0)) / 5.0        # not capped: 15 monsters ≠ 5
    for i, c in enumerate(cards[:MAX_OFFER]):
        b = CS_GLOBAL + i * CARD_FEATS
        _write_card_dense(c, out, b)                              # combat-style card features
        out[b + CARD_DENSE] = min(owned.get(_card_key(c), 0), 3) / 3.0      # already own N of this
        out[b + CARD_DENSE + 1 + _bucket(_card_key(c), CARD_HASH)] = 1.0    # hashed identity
    # ── current deck composition (what's already in the deck) ──
    db = CS_GLOBAL + MAX_OFFER * CARD_FEATS
    base_b = db + 7                          # all-copies count vector
    upg_b = base_b + CARD_VOCAB_SIZE         # upgraded-copies count vector
    ench_b = upg_b + CARD_VOCAB_SIZE         # enchant/affliction TYPE count hash
    na = ns = npw = ncs = nup = nen = naf = 0
    for dc in deck:
        dt = str(dc.get("type") or "")
        if dt == "Attack": na += 1
        elif dt == "Skill": ns += 1
        elif dt == "Power": npw += 1
        elif dt in ("Curse", "Status"): ncs += 1
        idx = _card_index(dc.get("id") or dc.get("name"))
        out[base_b + idx] += 1.0                              # every copy
        if dc.get("upgraded"):
            nup += 1
            out[upg_b + idx] += 1.0                           # upgraded copies (distinguishable)
        ench, aff = dc.get("enchantment"), dc.get("affliction")
        if ench: nen += 1
        if aff: naf += 1
        for tag in (ench, aff):                               # which enchant/affliction types
            if tag:
                out[ench_b + _bucket(str(tag), CARD_ENCH_HASH)] += 1.0
    out[db + 0] = na / 20.0
    out[db + 1] = ns / 20.0
    out[db + 2] = npw / 10.0
    out[db + 3] = ncs / 10.0
    out[db + 4] = nup / 20.0        # aggregate: cards upgraded
    out[db + 5] = nen / 10.0        # aggregate: cards enchanted
    out[db + 6] = naf / 10.0        # aggregate: cards afflicted
    # scale counts (NOT clipped, so 20 copies ≠ 10): a copy is ~1/10, an enchant type ~1/5
    out[base_b:base_b + CARD_VOCAB_SIZE] /= 10.0
    out[upg_b:upg_b + CARD_VOCAB_SIZE] /= 10.0
    out[ench_b:ench_b + CARD_ENCH_HASH] /= 5.0
    return out


def card_reward_mask(state) -> np.ndarray:
    m = np.zeros(CS_ACTIONS, bool)
    for i in range(min(len(state.get("cards") or []), MAX_OFFER)):
        m[i] = True
    m[MAX_OFFER] = True             # SKIP always legal
    return m


# ── map-node (path) encoding ───────────────────────────────────────────────
MAX_PATHS = 6
# Must match the engine's MapPointType.ToString() (see RunSimulator map choices):
# Monster, Elite, Boss, Shop, Treasure, RestSite, Ancient, Unknown/Unassigned.
ROOMS = ["MONSTER", "ELITE", "BOSS", "SHOP", "TREASURE", "RESTSITE", "ANCIENT", "UNKNOWN"]
# Per choice: present + immediate-type one-hot + downstream reachable-type counts
# (what taking this branch leads to, all the way to the boss).
PATH_FEATS = 1 + len(ROOMS) + len(ROOMS)
PS_GLOBAL = 4                      # hp, act, gold, n_choices
PS_OBS = PS_GLOBAL + MAX_PATHS * PATH_FEATS
PS_ACTIONS = MAX_PATHS


def _room_type(choice) -> str:
    for k in ("room_type", "type", "symbol", "node_type", "kind"):
        v = choice.get(k)
        if v:
            return str(v).upper()
    return "UNKNOWN"


def encode_map(state) -> np.ndarray:
    out = np.zeros(PS_OBS, np.float32)
    p = state.get("player") or {}
    out[0] = float(p.get("hp") or 0) / max(float(p.get("max_hp") or 1), 1)
    out[1] = float(_ctx(state, "act", 1)) / 3.0
    out[2] = float(p.get("gold") or 0) / 300.0       # gold-aware pathing (afford shop/removal)
    ch = state.get("choices") or []
    out[3] = len(ch) / MAX_PATHS
    for i, c in enumerate(ch[:MAX_PATHS]):
        b = PS_GLOBAL + i * PATH_FEATS
        out[b] = 1.0
        rt = _room_type(c)
        out[b + 1 + (ROOMS.index(rt) if rt in ROOMS else len(ROOMS) - 1)] = 1.0
        # downstream: how many of each room type this branch can reach (to the boss)
        rb = b + 1 + len(ROOMS)
        for k, v in (c.get("reach") or {}).items():
            ku = str(k).upper()
            idx = ROOMS.index(ku) if ku in ROOMS else len(ROOMS) - 1
            out[rb + idx] = min(float(v) / 5.0, 1.0)
    return out


def map_mask(state) -> np.ndarray:
    m = np.zeros(PS_ACTIONS, bool)
    for i in range(min(len(state.get("choices") or []), MAX_PATHS)):
        m[i] = True
    return m


# ── event / ancient encoding (handled by the card/economy agent) ───────────
# Events AND ancient nodes both arrive as decision "event_choice" (the engine
# resolves the name from the ancients table first, then events). Each option has
# {index, title, is_locked, vars:{Gold,HpLoss,Heal,...}}. We keep this generic:
# option identity via a hash + a few signed resource magnitudes (gold / hp-cost /
# hp-or-heal gain), rewarded by overall game victory like the rest of the agent.
MAX_OPTIONS = 6
OPT_HASH = 64                         # option identity (widened from 32 to reduce collisions)
OPT_VARS = 3                          # gold, hp-cost, hp/heal-gain (normalized)
OPT_FEATS = 2 + OPT_VARS + OPT_HASH   # present, is_locked, vars, id-hash
EV_GLOBAL = 6                         # hp, act, floor, gold, deck_size, n_options
# Event IDENTITY via a proper vocab (72 events+ancients + tail) instead of a 32-bucket
# hash — the old hash collided ~2-3 events per bucket, so the agent couldn't tell many
# events apart. Consequences (relic/card/curse/fight) are learned via the run reward.
EV_OBS = EV_GLOBAL + EVENT_VOCAB_SIZE + MAX_OPTIONS * OPT_FEATS
EV_ACTIONS = MAX_OPTIONS


def _opt_vars(opt) -> tuple[float, float, float]:
    """Pull signed resource magnitudes out of an option's `vars` dict, matched by
    key substring so it works across events without a per-event table."""
    gold = hp_cost = hp_gain = 0.0
    for k, v in (opt.get("vars") or {}).items():
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue                                  # e.g. RandomCard resolves to a name
        ku = str(k).upper()
        if "GOLD" in ku:
            gold += val
        elif any(s in ku for s in ("HEAL", "MAXHP", "MAX_HP")):
            hp_gain += val
        elif any(s in ku for s in ("HPLOSS", "HP_LOSS", "LOSE", "DAMAGE", "DMG", "COST")):
            hp_cost += val
    return gold, hp_cost, hp_gain


def encode_event(state) -> np.ndarray:
    out = np.zeros(EV_OBS, np.float32)
    p = state.get("player") or {}
    out[0] = float(p.get("hp") or 0) / max(float(p.get("max_hp") or 1), 1)
    out[1] = float(_ctx(state, "act", 1)) / 3.0
    out[2] = float(_ctx(state, "floor", 0)) / 50.0
    out[3] = float(p.get("gold") or 0) / 300.0
    out[4] = float(p.get("deck_size") or 0) / 40.0
    opts = state.get("options") or []
    out[5] = len(opts) / MAX_OPTIONS
    out[EV_GLOBAL + _event_index(state.get("event_name"))] = 1.0
    base0 = EV_GLOBAL + EVENT_VOCAB_SIZE
    for i, o in enumerate(opts[:MAX_OPTIONS]):
        b = base0 + i * OPT_FEATS
        gold, hp_cost, hp_gain = _opt_vars(o)
        out[b + 0] = 1.0
        out[b + 1] = 1.0 if o.get("is_locked") else 0.0
        out[b + 2] = gold / 100.0
        out[b + 3] = hp_cost / 30.0
        out[b + 4] = hp_gain / 30.0
        out[b + 5 + _bucket(o.get("text_key") or o.get("title") or "", OPT_HASH)] = 1.0
    return out


def event_mask(state) -> np.ndarray:
    m = np.zeros(EV_ACTIONS, bool)
    opts = state.get("options") or []
    for i, o in enumerate(opts[:MAX_OPTIONS]):
        if not o.get("is_locked"):
            m[i] = True
    if not m.any() and opts:          # all locked (rare): allow the first, best effort
        m[0] = True
    return m


# ── rest-site option choice (handled by the card/economy agent) ────────────
# Options carry {index, option_id (HEAL/SMITH/DIG/...), is_enabled}. Generic:
# option-id hash + enabled flag, so heal-vs-upgrade-vs-dig is a learned tradeoff.
MAX_REST = 6
REST_OPT_HASH = 16
REST_GLOBAL = 4                       # hp, act, floor, deck_size
REST_FEATS = 2 + REST_OPT_HASH        # present, enabled, id-hash
REST_OBS = REST_GLOBAL + MAX_REST * REST_FEATS
REST_ACTIONS = MAX_REST


def encode_rest(state) -> np.ndarray:
    out = np.zeros(REST_OBS, np.float32)
    p = state.get("player") or {}
    out[0] = float(p.get("hp") or 0) / max(float(p.get("max_hp") or 1), 1)
    out[1] = float(_ctx(state, "act", 1)) / 3.0
    out[2] = float(_ctx(state, "floor", 0)) / 50.0
    out[3] = float(p.get("deck_size") or 0) / 40.0
    for i, o in enumerate((state.get("options") or [])[:MAX_REST]):
        b = REST_GLOBAL + i * REST_FEATS
        out[b + 0] = 1.0
        out[b + 1] = 1.0 if o.get("is_enabled", True) else 0.0
        out[b + 2 + _bucket(o.get("option_id") or o.get("name") or "", REST_OPT_HASH)] = 1.0
    return out


def rest_mask(state) -> np.ndarray:
    m = np.zeros(REST_ACTIONS, bool)
    opts = state.get("options") or []
    for i, o in enumerate(opts[:MAX_REST]):
        if o.get("is_enabled", True):
            m[i] = True
    if not m.any() and opts:
        m[0] = True
    return m


# ── upgrade-target choice at a rest-site smith (card/economy agent) ─────────
# The smith opens a card_select over upgradeable deck cards; the card agent picks
# WHICH card to upgrade. Deck can be large, so this space is bigger than a reward.
MAX_UPG = 32
UPG_HASH = 48
UPG_GLOBAL = 4                        # hp, act, floor, n_cards
UPG_FEATS = 6 + UPG_HASH              # present, cost, atk, skill, power, dmg+block, id-hash
UPG_OBS = UPG_GLOBAL + MAX_UPG * UPG_FEATS
UPG_ACTIONS = MAX_UPG


def encode_upgrade(state) -> np.ndarray:
    out = np.zeros(UPG_OBS, np.float32)
    p = state.get("player") or {}
    out[0] = float(p.get("hp") or 0) / max(float(p.get("max_hp") or 1), 1)
    out[1] = float(_ctx(state, "act", 1)) / 3.0
    out[2] = float(_ctx(state, "floor", 0)) / 50.0
    cards = state.get("cards") or []
    out[3] = len(cards) / MAX_UPG
    for i, c in enumerate(cards[:MAX_UPG]):
        b = UPG_GLOBAL + i * UPG_FEATS
        stats = c.get("stats") or {}
        t = str(c.get("type") or "")
        out[b + 0] = 1.0
        out[b + 1] = float(c.get("cost") or 0) / 3.0
        out[b + 2] = 1.0 if t == "Attack" else 0.0
        out[b + 3] = 1.0 if t == "Skill" else 0.0
        out[b + 4] = 1.0 if t == "Power" else 0.0
        out[b + 5] = (float(stats.get("damage") or 0) + float(stats.get("block") or 0)) / 30.0
        out[b + 6 + _bucket(c.get("id") or c.get("name") or "", UPG_HASH)] = 1.0
    return out


def upgrade_mask(state) -> np.ndarray:
    m = np.zeros(UPG_ACTIONS, bool)
    for i in range(min(len(state.get("cards") or []), MAX_UPG)):
        m[i] = True                       # engine already filtered to upgradeable cards
    return m


# ── shop encoding (handled by the card/economy agent) ──────────────────────
MAX_SHOP_CARDS = 7          # 5 colored + 2 colorless (Courier can add more; rare)
MAX_SHOP_RELICS = 3
MAX_SHOP_POTIONS = 3
SHOP_HASH = 48
SI = 3 + SHOP_HASH                       # per shop item: stocked, cost, affordable, id-hash
SHOP_GLOBAL = 4
SHOP_OBS = SHOP_GLOBAL + (MAX_SHOP_CARDS + MAX_SHOP_RELICS + MAX_SHOP_POTIONS) * SI + 2
# action layout
SHOP_BUY_CARD = 0
SHOP_BUY_RELIC = SHOP_BUY_CARD + MAX_SHOP_CARDS
SHOP_BUY_POTION = SHOP_BUY_RELIC + MAX_SHOP_RELICS
SHOP_REMOVE = SHOP_BUY_POTION + MAX_SHOP_POTIONS
SHOP_LEAVE = SHOP_REMOVE + 1
SHOP_ACTIONS = SHOP_LEAVE + 1


def encode_shop(state) -> np.ndarray:
    out = np.zeros(SHOP_OBS, np.float32)
    p = state.get("player") or {}
    gold = float(p.get("gold") or 0)
    out[0] = gold / 300.0
    out[1] = float(p.get("deck_size") or 0) / 40.0
    out[2] = float(_ctx(state, "act", 1)) / 3.0
    out[3] = float(p.get("hp") or 0) / max(float(p.get("max_hp") or 1), 1)

    def put(items, base, n):
        for i, it in enumerate((items or [])[:n]):
            b = base + i * SI
            cost = float(it.get("cost") or 0)
            stocked = it.get("is_stocked", True)
            out[b + 0] = 1.0 if stocked else 0.0
            out[b + 1] = cost / 300.0
            out[b + 2] = 1.0 if (stocked and cost <= gold) else 0.0
            out[b + 3 + _bucket(it.get("name") or "", SHOP_HASH)] = 1.0

    off = SHOP_GLOBAL
    put(state.get("cards"), off, MAX_SHOP_CARDS); off += MAX_SHOP_CARDS * SI
    put(state.get("relics"), off, MAX_SHOP_RELICS); off += MAX_SHOP_RELICS * SI
    put(state.get("potions"), off, MAX_SHOP_POTIONS); off += MAX_SHOP_POTIONS * SI
    rc = state.get("card_removal_cost")
    out[off] = (float(rc) / 300.0) if rc else 0.0
    out[off + 1] = 1.0 if (rc and float(rc) <= gold) else 0.0
    return out


def shop_mask(state) -> np.ndarray:
    m = np.zeros(SHOP_ACTIONS, bool)
    p = state.get("player") or {}
    gold = float(p.get("gold") or 0)

    def can(items, base, n):
        for i, it in enumerate((items or [])[:n]):
            if it.get("is_stocked", True) and float(it.get("cost") or 0) <= gold:
                m[base + i] = True

    can(state.get("cards"), SHOP_BUY_CARD, MAX_SHOP_CARDS)
    can(state.get("relics"), SHOP_BUY_RELIC, MAX_SHOP_RELICS)
    can(state.get("potions"), SHOP_BUY_POTION, MAX_SHOP_POTIONS)
    rc = state.get("card_removal_cost")
    if rc and float(rc) <= gold and (p.get("deck_size") or 0) > 0:
        m[SHOP_REMOVE] = True
    m[SHOP_LEAVE] = True                  # leaving is always allowed
    return m


# ── network + agent ────────────────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class Agent:
    """One policy. act() samples with no grad; learn() does an actor-critic step."""

    def __init__(self, name: str, obs_dim: int, n_actions: int,
                 lr: float = 3e-4, device: str = "cpu"):
        self.name = name
        self.device = torch.device(device)
        self.net = PolicyNet(obs_dim, n_actions).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def act(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(obs, dtype=torch.float32,
                                                 device=self.device).unsqueeze(0))
        logits = logits.squeeze(0).cpu().numpy()
        logits[~mask] = -1e9
        if greedy:
            return int(np.argmax(logits))
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return int(np.random.choice(len(p), p=p))

    def learn(self, obs, masks, actions, returns, value_coef: float = 0.5,
              entropy_coef: float = 0.01) -> float:
        if len(obs) == 0:
            return 0.0
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        m = torch.as_tensor(np.asarray(masks), device=self.device)
        a = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        ret = torch.as_tensor(np.asarray(returns), dtype=torch.float32, device=self.device)
        logits, v = self.net(x)
        logits = logits.masked_fill(~m, -1e9)
        logp_all = torch.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
        adv = (ret - v).detach()
        p_loss = -(logp * adv).mean()
        v_loss = torch.nn.functional.mse_loss(v, ret)
        ent = -(logp_all.exp() * logp_all).sum(-1).mean()
        loss = p_loss + value_coef * v_loss - entropy_coef * ent
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))


class CombatNet(nn.Module):
    """Combat policy/value net with a learned card **embedding**. Card identity is an
    integer per hand slot (not a 640-wide one-hot); nn.Embedding maps it to a compact
    vector, so the net learns card similarity, uses far fewer params, and runs faster.
    The embeddings are concatenated with the dense combat features and fed to the MLP."""

    def __init__(self, dense_dim: int, n_actions: int, hidden: int = 512, depth: int = 3):
        super().__init__()
        self.embed = nn.Embedding(CARD_VOCAB_SIZE + 1, EMBED_DIM, padding_idx=PAD_CARD)
        layers, d = [], dense_dim + MAX_HAND * EMBED_DIM
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        # in-combat card-selection head: scores each candidate card (shares the card
        # embedding), for Armaments/Dual Wield/Exhume/discover/pile-move selects.
        self.sel = nn.Sequential(nn.Linear(SEL_GLOBAL + CARD_DENSE + EMBED_DIM, 128),
                                 nn.ReLU(), nn.Linear(128, 1))

    def forward(self, dense, ids):
        emb = self.embed(ids).reshape(ids.shape[0], -1)     # (B, MAX_HAND*EMBED_DIM)
        h = self.body(torch.cat([dense, emb], dim=-1))
        return self.pi(h), self.v(h).squeeze(-1)

    def score_select(self, glob, cand_dense, cand_ids):
        emb = self.embed(cand_ids)                          # (B, MAX_SEL, EMBED_DIM)
        g = glob.unsqueeze(1).expand(-1, cand_dense.shape[1], -1)
        x = torch.cat([g, cand_dense, emb], dim=-1)
        return self.sel(x).squeeze(-1)                      # (B, MAX_SEL)


class CombatAgent:
    """Combat policy over the (dense, card_ids) observation. Same actor-critic update
    as Agent, but the forward pass takes two tensors (dense features + card ids)."""

    def __init__(self, device: str = "cpu", lr: float = 3e-4, hidden: int = 512):
        self.name = "combat"
        self.device = torch.device(device)
        self.net = CombatNet(DENSE_DIM, N_ACTIONS, hidden=hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def act(self, dense, ids, mask, greedy: bool = False) -> int:
        with torch.no_grad():
            d = torch.as_tensor(dense, dtype=torch.float32, device=self.device).unsqueeze(0)
            ii = torch.as_tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
            logits, _ = self.net(d, ii)
        logits = logits.squeeze(0).cpu().numpy()
        logits[~mask] = -1e9
        if greedy:
            return int(np.argmax(logits))
        logits -= logits.max()
        pr = np.exp(logits)
        pr /= pr.sum()
        return int(np.random.choice(len(pr), p=pr))

    def learn(self, dense, ids, masks, actions, returns,
              value_coef: float = 0.5, entropy_coef: float = 0.01) -> float:
        if len(dense) == 0:
            return 0.0
        d = torch.as_tensor(np.asarray(dense), dtype=torch.float32, device=self.device)
        ii = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.device)
        m = torch.as_tensor(np.asarray(masks), device=self.device)
        a = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        ret = torch.as_tensor(np.asarray(returns), dtype=torch.float32, device=self.device)
        logits, v = self.net(d, ii)
        logits = logits.masked_fill(~m, -1e9)
        logp_all = torch.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
        adv = (ret - v).detach()
        p_loss = -(logp * adv).mean()
        v_loss = F.mse_loss(v, ret)
        ent = -(logp_all.exp() * logp_all).sum(-1).mean()
        loss = p_loss + value_coef * v_loss - entropy_coef * ent
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def act_select(self, glob, cand_dense, cand_ids, n, min_sel=1, greedy=False):
        """Pick which candidate card(s) to select. Returns a list of indices; the
        first is the 'primary' pick used for the gradient. min_sel>1 fills the rest
        greedily by score (approximate; the common case is a single pick)."""
        with torch.no_grad():
            g = torch.as_tensor(glob, dtype=torch.float32, device=self.device).unsqueeze(0)
            cd = torch.as_tensor(cand_dense, dtype=torch.float32, device=self.device).unsqueeze(0)
            ci = torch.as_tensor(cand_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            scores = self.net.score_select(g, cd, ci).squeeze(0).cpu().numpy()
        n = max(1, min(int(n), MAX_SEL))
        scores[n:] = -1e9
        k = max(1, min(int(min_sel or 1), n))
        if greedy:
            order = list(np.argsort(-scores))
            return order[:k]
        s = scores - scores.max()
        pr = np.exp(s)
        pr /= pr.sum()
        first = int(np.random.choice(len(pr), p=pr))
        rest = [j for j in np.argsort(-scores) if j != first][:k - 1]
        return [first] + rest

    def learn_select(self, globs, cand_denses, cand_ids, ns, actions, returns,
                     entropy_coef: float = 0.01) -> float:
        if len(globs) == 0:
            return 0.0
        g = torch.as_tensor(np.asarray(globs), dtype=torch.float32, device=self.device)
        cd = torch.as_tensor(np.asarray(cand_denses), dtype=torch.float32, device=self.device)
        ci = torch.as_tensor(np.asarray(cand_ids), dtype=torch.long, device=self.device)
        a = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        ret = torch.as_tensor(np.asarray(returns), dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(np.asarray(ns), dtype=torch.long, device=self.device)
        scores = self.net.score_select(g, cd, ci)                 # (B, MAX_SEL)
        valid = torch.arange(scores.shape[1], device=self.device).unsqueeze(0) < ns.unsqueeze(1)
        scores = scores.masked_fill(~valid, -1e9)
        logp_all = torch.log_softmax(scores, dim=-1)
        logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
        adv = (ret - ret.mean()).detach()                         # mean-baseline (no value head)
        ent = -(logp_all.exp() * logp_all).sum(-1).mean()
        loss = -(logp * adv).mean() - entropy_coef * ent
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))


class CardAgent:
    """The card/economy agent: one logical agent, five policy heads — card rewards,
    the shop (buy card/relic/potion, remove card, leave), event/ancient choices,
    rest-site option choice, and the rest-site smith upgrade-target choice. All
    rewarded by overall game victory. Saved/loaded as one agent (five .pt files)."""

    _HEADS = ("reward", "shop", "event", "rest", "upgrade")

    def __init__(self, device: str = "cpu"):
        self.reward = Agent("card_reward", CS_OBS, CS_ACTIONS, device=device)
        self.shop = Agent("card_shop", SHOP_OBS, SHOP_ACTIONS, device=device)
        self.event = Agent("card_event", EV_OBS, EV_ACTIONS, device=device)
        self.rest = Agent("card_rest", REST_OBS, REST_ACTIONS, device=device)
        self.upgrade = Agent("card_upgrade", UPG_OBS, UPG_ACTIONS, device=device)

    def act_reward(self, obs, mask, greedy=False):
        return self.reward.act(obs, mask, greedy)

    def act_shop(self, obs, mask, greedy=False):
        return self.shop.act(obs, mask, greedy)

    def act_event(self, obs, mask, greedy=False):
        return self.event.act(obs, mask, greedy)

    def act_rest(self, obs, mask, greedy=False):
        return self.rest.act(obs, mask, greedy)

    def act_upgrade(self, obs, mask, greedy=False):
        return self.upgrade.act(obs, mask, greedy)

    def learn(self, reward_batch, shop_batch, event_batch, rest_batch, upgrade_batch) -> float:
        losses = [self.reward.learn(*reward_batch), self.shop.learn(*shop_batch),
                  self.event.learn(*event_batch), self.rest.learn(*rest_batch),
                  self.upgrade.learn(*upgrade_batch)]
        return sum(losses) / len(losses)

    def _base(self, path):
        return path[:-3] if path.endswith(".pt") else path

    def save(self, path):
        for sub in self._HEADS:
            getattr(self, sub).save(self._base(path) + f".{sub}.pt")

    def load(self, path):
        import os
        # Per-head try/except: a shape mismatch on one head (e.g. an old shop.pt
        # from before MAX_SHOP_CARDS grew, or a run with no event/rest/upgrade.pt
        # yet) must not block the other, still-compatible heads from loading.
        for sub in self._HEADS:
            p = self._base(path) + f".{sub}.pt"
            if os.path.exists(p):
                try:
                    getattr(self, sub).load(p)
                except Exception as e:
                    print(f"  [card.{sub}] not loaded ({e.__class__.__name__}); starting fresh")


def make_agents(device: str = "cpu") -> dict:
    return {
        "combat": CombatAgent(device=device),      # embedding net over (dense, card_ids)
        "card": CardAgent(device=device),
        "path": Agent("path", PS_OBS, PS_ACTIONS, device=device),
    }

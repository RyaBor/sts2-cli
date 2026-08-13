"""Three cooperating policy agents for full A10 runs, trained by REINFORCE.

  combat_agent : plays cards in combat        (reward = HP retained this combat)
  card_agent   : picks card rewards           (reward = overall game victory)
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

from encoding import OBS_DIM, N_ACTIONS, encode_obs, action_mask, decode_action  # noqa: F401


def _bucket(s, n: int) -> int:
    return zlib.crc32(str(s).encode("utf-8")) % n


# ── card-reward encoding ───────────────────────────────────────────────────
MAX_OFFER = 4
CARD_HASH = 48
CARD_FEATS = 6 + CARD_HASH
CS_GLOBAL = 6                       # hp, deck_size, act, floor, gold, n_offered
CS_OBS = CS_GLOBAL + MAX_OFFER * CARD_FEATS
CS_ACTIONS = MAX_OFFER + 1          # pick offered card i, or SKIP (last)


def _ctx(state, key, default=0):
    return (state.get("context") or {}).get(key) or state.get(key) or default


def encode_card_reward(state) -> np.ndarray:
    out = np.zeros(CS_OBS, np.float32)
    p = state.get("player") or {}
    out[0] = float(p.get("hp") or 0) / max(float(p.get("max_hp") or 1), 1)
    out[1] = float(p.get("deck_size") or 0) / 40.0
    out[2] = float(_ctx(state, "act", 1)) / 3.0
    out[3] = float(_ctx(state, "floor", 0)) / 50.0
    out[4] = float(p.get("gold") or 0) / 300.0       # gold-aware drafting
    cards = state.get("cards") or []
    out[5] = len(cards) / MAX_OFFER
    for i, c in enumerate(cards[:MAX_OFFER]):
        b = CS_GLOBAL + i * CARD_FEATS
        stats = c.get("stats") or {}
        t = str(c.get("type") or "")
        out[b + 0] = 1.0
        out[b + 1] = float(c.get("cost") or 0) / 3.0
        out[b + 2] = 1.0 if t == "Attack" else 0.0
        out[b + 3] = 1.0 if t == "Skill" else 0.0
        out[b + 4] = 1.0 if t == "Power" else 0.0
        out[b + 5] = (float(stats.get("damage") or 0) + float(stats.get("block") or 0)) / 30.0
        out[b + 6 + _bucket(c.get("id") or c.get("name") or "", CARD_HASH)] = 1.0
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
PATH_FEATS = len(ROOMS) + 1
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
    return out


def map_mask(state) -> np.ndarray:
    m = np.zeros(PS_ACTIONS, bool)
    for i in range(min(len(state.get("choices") or []), MAX_PATHS)):
        m[i] = True
    return m


# ── shop encoding (handled by the card/economy agent) ──────────────────────
MAX_SHOP_CARDS = 5
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


class CardAgent:
    """The card/economy agent: one logical agent, two policy heads — card rewards
    and the shop (buy card/relic/potion, remove card, leave). Both rewarded by
    overall game victory. Saved/loaded as one agent (two .pt files)."""

    def __init__(self, device: str = "cpu"):
        self.reward = Agent("card_reward", CS_OBS, CS_ACTIONS, device=device)
        self.shop = Agent("card_shop", SHOP_OBS, SHOP_ACTIONS, device=device)

    def act_reward(self, obs, mask, greedy=False):
        return self.reward.act(obs, mask, greedy)

    def act_shop(self, obs, mask, greedy=False):
        return self.shop.act(obs, mask, greedy)

    def learn(self, reward_batch, shop_batch) -> float:
        a = self.reward.learn(*reward_batch)
        b = self.shop.learn(*shop_batch)
        return (a + b) / 2

    def _base(self, path):
        return path[:-3] if path.endswith(".pt") else path

    def save(self, path):
        self.reward.save(self._base(path) + ".reward.pt")
        self.shop.save(self._base(path) + ".shop.pt")

    def load(self, path):
        import os
        for sub, agent in (("reward", self.reward), ("shop", self.shop)):
            p = self._base(path) + f".{sub}.pt"
            if os.path.exists(p):
                agent.load(p)


def make_agents(device: str = "cpu") -> dict:
    return {
        "combat": Agent("combat", OBS_DIM, N_ACTIONS, device=device),
        "card": CardAgent(device=device),
        "path": Agent("path", PS_OBS, PS_ACTIONS, device=device),
    }

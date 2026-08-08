"""Attention feature extractor over game entities.

Two problems with a flat MLP over concatenated slots, both of which this fixes:

* **Identity.** Entities arrive as vocabulary ids and get real embeddings, so
  the network can learn that *this* monster is a Cultist rather than inferring
  it from HP. Boss-aware play is impossible without it.
* **Permutation.** An MLP treats "enemy in slot 0" and "enemy in slot 1" as
  unrelated inputs and has to learn each twice. Self-attention over a set of
  entity tokens is permutation-invariant, so targeting knowledge transfers
  across positions — the thing that matters most in multi-enemy fights.

Padded slots are masked out of attention rather than attended to as if they
were real enemies.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from . import vocab
from .encoding import ENEMY_FEATS, GLOBAL_FEATS, HAND_FEATS


class EntityAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__(observation_space, features_dim)
        sizes = vocab.sizes()
        d = d_model

        self.card_emb = nn.Embedding(sizes["cards"], d, padding_idx=0)
        self.monster_emb = nn.Embedding(sizes["monsters"], d, padding_idx=0)
        self.relic_emb = nn.Embedding(sizes["relics"], d, padding_idx=0)
        self.potion_emb = nn.Embedding(sizes["potions"], d, padding_idx=0)
        self.intent_emb = nn.Embedding(sizes["intents"], d, padding_idx=0)
        # Boss ids are encounter ids, not monster ids.
        self.boss_emb = nn.Embedding(sizes["encounters"], d, padding_idx=0)
        # Buffs/debuffs. Scaled by amount so 3 Vulnerable differs from 1.
        self.power_emb = nn.Embedding(sizes["powers"], d, padding_idx=0)

        # One learned marker per entity kind, so attention can tell a card
        # token from an enemy token after they are pooled into one sequence.
        self.kind_emb = nn.Embedding(5, d)  # 0 global, 1 hand, 2 enemy, 3 relic, 4 potion

        self.hand_proj = nn.Linear(HAND_FEATS, d)
        self.enemy_proj = nn.Linear(ENEMY_FEATS, d)
        # Global token also carries the three pile summaries and the boss id.
        self.global_proj = nn.Linear(GLOBAL_FEATS + 5 * d, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(2 * d, features_dim), nn.ReLU())

    def _bag(self, emb: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
        """Sum embeddings over a padded id list (padding_idx contributes 0)."""
        return emb(ids.long()).sum(dim=1)

    def _powers(self, ids: torch.Tensor, amts: torch.Tensor) -> torch.Tensor:
        """Amount-weighted sum of power embeddings over the last axis."""
        return (self.power_emb(ids.long()) * amts.unsqueeze(-1)).sum(dim=-2)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        b = obs["global"].shape[0]
        dev = obs["global"].device

        piles = torch.cat([
            self._bag(self.card_emb, obs["draw_ids"]),
            self._bag(self.card_emb, obs["discard_ids"]),
            self._bag(self.card_emb, obs["exhaust_ids"]),
            self.boss_emb(obs["boss_id"].long()).squeeze(1),
            # What the enemy has done to us: Weak, Vulnerable, Frail, ...
            self._powers(obs["player_power_ids"], obs["player_power_amts"]),
        ], dim=-1)
        g = self.global_proj(torch.cat([obs["global"], piles], dim=-1))
        g = g + self.kind_emb(torch.zeros(b, dtype=torch.long, device=dev))
        g = g.unsqueeze(1)

        hand = self.card_emb(obs["hand_ids"].long()) + self.hand_proj(obs["hand_feats"])
        hand = hand + self.kind_emb(torch.ones(b, 1, dtype=torch.long, device=dev))

        # An enemy token is its identity, its numbers, and what it is about to do.
        intents = self.intent_emb(obs["enemy_intents"].long()).sum(dim=2)
        enemy = (self.monster_emb(obs["enemy_ids"].long())
                 + self.enemy_proj(obs["enemy_feats"]) + intents
                 + self._powers(obs["enemy_power_ids"], obs["enemy_power_amts"]))
        enemy = enemy + self.kind_emb(
            torch.full((b, 1), 2, dtype=torch.long, device=dev))

        relic = self.relic_emb(obs["relic_ids"].long())
        relic = relic + self.kind_emb(torch.full((b, 1), 3, dtype=torch.long, device=dev))
        potion = self.potion_emb(obs["potion_ids"].long())
        potion = potion + self.kind_emb(torch.full((b, 1), 4, dtype=torch.long, device=dev))

        tokens = torch.cat([g, hand, enemy, relic, potion], dim=1)

        ones = torch.ones(b, 1, device=dev)
        valid = torch.cat([
            ones,
            obs["hand_mask"],
            obs["enemy_mask"],
            (obs["relic_ids"] > 0).float(),
            (obs["potion_ids"] > 0).float(),
        ], dim=1)
        pad = valid < 0.5

        out = self.encoder(tokens, src_key_padding_mask=pad)

        # Global token (what the state looks like) plus a masked mean over real
        # entities (what is on the table), so padding never dilutes the pool.
        keep = (~pad).float().unsqueeze(-1)
        pooled = (out * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.head(torch.cat([out[:, 0], pooled], dim=-1))

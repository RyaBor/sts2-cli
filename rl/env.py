"""Gymnasium env: one episode = one combat.

Exposes `action_masks()` so sb3-contrib's MaskablePPO can ignore illegal moves
(unplayable cards, dead targets, empty hand slots) instead of wasting samples
learning that they do nothing.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .encoding import (N_ACTIONS, OBS_DIM, action_mask, decode_action,
                       encode_obs)
from .engine import Engine, EngineError

# Reward shaping. Terminal outcome dominates; the per-HP terms give dense
# signal so the agent learns to block and to kill fast, not just to survive.
R_WIN = 1.0
R_LOSS = -1.0
R_DMG_DEALT = 0.010   # per enemy HP removed
R_HP_LOST = -0.020    # per player HP lost
R_STEP = -0.001       # mild pressure against stalling


class Sts2CombatEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, character: str = "Ironclad",
                 encounter: str | list[str] | None = "SHRINKER_BEETLE_WEAK",
                 ascension: int = 0, seed: str | None = None,
                 start_hp: int = 80, max_hp: int = 80, max_steps: int = 300):
        super().__init__()
        self.observation_space = spaces.Box(-10.0, 10.0, (OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.character = character
        self.encounters = ([encounter] if isinstance(encounter, str)
                           else list(encounter) if encounter else [None])
        self.ascension = ascension
        self.start_hp = start_hp
        self.max_hp = max_hp
        self.max_steps = max_steps
        self._seed = seed

        self.engine: Engine | None = None
        self.state: dict[str, Any] = {}
        self._steps = 0
        self._prev_player_hp = 0.0
        self._prev_enemy_hp = 0.0

    # ---------------- helpers ----------------

    def _ensure_engine(self) -> None:
        if self.engine is None:
            self.engine = Engine(character=self.character, seed=self._seed,
                                 ascension=self.ascension)

    @staticmethod
    def _enemy_hp_total(st: dict) -> float:
        return float(sum(max(0, e.get("hp") or 0) for e in (st.get("enemies") or [])))

    @staticmethod
    def _player_hp(st: dict) -> float:
        return float((st.get("player") or {}).get("hp") or 0)

    def _obs(self) -> np.ndarray:
        return encode_obs(self.state)

    def action_masks(self) -> np.ndarray:
        return action_mask(self.state)

    # ---------------- gym api ----------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._ensure_engine()
        assert self.engine is not None
        enc = self.encounters[self.np_random.integers(len(self.encounters))]
        # Restoring HP every episode keeps the task stationary and stops the
        # player carrying damage between fights until the run dies (a death
        # ends the run and forces a costly engine restart).
        try:
            self.state = self.engine.reset_combat(
                encounter=enc, hp=self.start_hp, max_hp=self.max_hp)
        except EngineError:
            # Run over (death) or engine wedged: rebuild and retry once.
            self.engine.close()
            self.engine = None
            self._ensure_engine()
            assert self.engine is not None
            self.state = self.engine.reset_combat(
                encounter=enc, hp=self.start_hp, max_hp=self.max_hp)

        self._steps = 0
        self._prev_player_hp = self._player_hp(self.state)
        self._prev_enemy_hp = self._enemy_hp_total(self.state)
        return self._obs(), {}

    def step(self, action: int):
        assert self.engine is not None, "call reset() first"
        name, args = decode_action(int(action), self.state)

        try:
            nxt = self.engine.act(name, **args)
        except EngineError:
            # Treat a dead engine as a lost episode rather than crashing training.
            return self._obs(), R_LOSS, True, False, {"engine_error": True}

        self._steps += 1

        if nxt.get("type") == "error":
            # Illegal action slipped through the mask: penalise lightly, keep state.
            return self._obs(), -0.05, False, False, {"invalid": nxt.get("message")}

        decision = nxt.get("decision")
        reward = R_STEP
        terminated = False
        info: dict[str, Any] = {}

        if decision == "combat_play":
            self.state = nxt
            player_hp = self._player_hp(nxt)
            enemy_hp = self._enemy_hp_total(nxt)
            reward += R_DMG_DEALT * max(0.0, self._prev_enemy_hp - enemy_hp)
            reward += R_HP_LOST * max(0.0, self._prev_player_hp - player_hp)
            self._prev_player_hp = player_hp
            self._prev_enemy_hp = enemy_hp
        else:
            # Left the combat screen: either we won, or the run ended in death.
            terminated = True
            died = decision == "game_over" or self._player_hp(nxt) <= 0
            reward += R_LOSS if died else R_WIN
            info["outcome"] = "loss" if died else "win"
            info["final_hp"] = self._player_hp(nxt)
            # Keep last combat obs; engine is left on the post-combat screen and
            # reset_combat() will clear it.

        truncated = self._steps >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, info

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None


def make_env(rank: int = 0, **kwargs):
    """Factory for SubprocVecEnv; distinct seeds keep workers decorrelated."""
    def _init():
        return Sts2CombatEnv(seed=f"rl{rank}", **kwargs)
    return _init

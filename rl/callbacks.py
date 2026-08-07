"""Log combat-specific metrics to TensorBoard.

SB3 logs reward and episode length; neither says whether the agent actually
won or how much HP it kept, which is what matters here. This scrapes the
episode-terminal `info` dicts and logs rolling averages.
"""
from __future__ import annotations

from collections import deque

from stable_baselines3.common.callbacks import BaseCallback


class CombatMetricsCallback(BaseCallback):
    def __init__(self, window: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.wins: deque[float] = deque(maxlen=window)
        self.hp: deque[float] = deque(maxlen=window)
        self.invalid = 0
        self.engine_errors = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not info:
                continue
            if info.get("invalid"):
                self.invalid += 1
            if info.get("engine_error"):
                self.engine_errors += 1
            outcome = info.get("outcome")
            if outcome is None:
                continue
            won = outcome == "win"
            self.wins.append(1.0 if won else 0.0)
            if won:
                self.hp.append(float(info.get("final_hp") or 0.0))

        if self.wins:
            self.logger.record("combat/win_rate", sum(self.wins) / len(self.wins))
        if self.hp:
            self.logger.record("combat/hp_retained_on_win", sum(self.hp) / len(self.hp))
        # Should stay flat at zero; a climbing count means the action mask and
        # the engine disagree about legality.
        self.logger.record("combat/invalid_actions", self.invalid)
        self.logger.record("combat/engine_errors", self.engine_errors)
        return True

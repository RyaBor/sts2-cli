"""Thin wrapper around one headless Sts2Headless process.

One process serves many combats: `reset_combat()` re-enters a fight without
paying the ~0.9s run-start cost again (a reset is ~50ms).
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLL = os.path.join(ROOT, "src", "Sts2Headless", "bin", "Debug", "net9.0", "Sts2Headless.dll")

# Screens that can appear after a fight ends and must be dismissed before the
# engine will accept a new enter_room.
_POST_COMBAT = {
    "card_reward": "skip_card_reward",
    "rest_site": "proceed",
    "shop": "proceed",
    "event_choice": "proceed",
}


class EngineError(RuntimeError):
    pass


class Engine:
    """Request/response JSON pipe to the C# game engine."""

    def __init__(self, character: str = "Ironclad", seed: str | None = None,
                 ascension: int = 0, dll: str = DLL):
        if not os.path.exists(dll):
            raise EngineError(
                f"Engine not built: {dll}\nRun: dotnet build src/Sts2Headless/Sts2Headless.csproj")
        self.character = character
        self.ascension = ascension
        self.seed = seed or "rl"
        self.proc = subprocess.Popen(
            ["dotnet", dll],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=ROOT, encoding="utf-8", errors="replace",
        )
        # The engine logs heavily to stderr. If nobody reads that pipe it fills,
        # the engine blocks writing to it, and stops answering on stdout.
        self._stderr_tail: list[str] = []
        self.last: dict[str, Any] = {}
        # Read stdout on a background thread into a queue so _read() can TIME OUT
        # instead of blocking forever if the engine deadlocks internally (hang guard).
        self.read_timeout = float(os.environ.get("STS2_READ_TIMEOUT", "60"))
        self._EOF = object()
        self._q: "queue.Queue[Any]" = queue.Queue()
        self.action_log: list[dict] = []       # every command sent this run (for replay)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        self._read()  # {"type":"ready"}
        st = self.send({"cmd": "start_run", "character": character,
                        "seed": self.seed, "ascension": ascension})
        if st.get("type") == "error":
            raise EngineError(f"start_run failed: {st.get('message')}")
        self.action_log.clear()                # keep only the run's actions (not start_run)

    def repro(self) -> dict[str, Any]:
        """A record that deterministically replays this run: same seed + the exact
        command sequence. See rl/replay.py."""
        return {"character": self.character, "seed": self.seed,
                "ascension": self.ascension, "actions": list(self.action_log)}

    # ---------------- plumbing ----------------

    def _drain_stderr(self) -> None:
        for line in iter(self.proc.stderr.readline, ""):
            # Keep a short tail purely for diagnostics on crash.
            self._stderr_tail.append(line.rstrip())
            if len(self._stderr_tail) > 40:
                self._stderr_tail.pop(0)

    def _drain_stdout(self) -> None:
        try:
            for line in iter(self.proc.stdout.readline, ""):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        self._q.put(json.loads(line))
                    except Exception:
                        pass
        finally:
            self._q.put(self._EOF)      # stream closed -> process exited

    def _read(self) -> dict[str, Any]:
        try:
            item = self._q.get(timeout=self.read_timeout)
        except queue.Empty:
            try:
                self.proc.kill()
            except Exception:
                pass
            tail = "\n".join(self._stderr_tail[-15:])
            raise EngineError(
                f"engine timed out after {self.read_timeout:.0f}s (hung); killed. "
                f"stderr tail:\n{tail}")
        if item is self._EOF:
            tail = "\n".join(self._stderr_tail[-15:])
            raise EngineError(f"engine exited unexpectedly. stderr tail:\n{tail}")
        return item

    def send(self, cmd: dict[str, Any]) -> dict[str, Any]:
        if self.proc.poll() is not None:
            raise EngineError("engine process is dead")
        self.action_log.append(cmd)            # record for failure replay
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        st = self._read()
        # Remember the last screen: there is no query command that reports it
        # (get_map returns no "decision"), so reset has to track it itself.
        if isinstance(st, dict) and st.get("decision"):
            self.last = st
        return st

    def act(self, action: str, **args: Any) -> dict[str, Any]:
        cmd: dict[str, Any] = {"cmd": "action", "action": action}
        if args:
            cmd["args"] = args
        return self.send(cmd)

    # ---------------- combat lifecycle ----------------

    def reset_combat(self, encounter: str | None = None, hp: int | None = None,
                     max_hp: int | None = None, deck: list[str] | None = None,
                     relics: list[str] | None = None,
                     potions: list[str] | None = None,
                     hand: list[str] | None = None,
                     discard: list[str] | None = None,
                     exhaust: list[str] | None = None,
                     enemy_hp: list[int] | None = None,
                     enemy_block: list[int] | None = None,
                     player_powers: list[dict] | None = None,
                     enemy_powers: list[list] | None = None,
                     rng_streams: dict | None = None,
                     draw_order: list[str] | None = None,
                     energy: int | None = None) -> dict[str, Any]:
        """Dismiss any leftover screen, apply loadout, and enter a fresh fight.

        If `hand` (and optionally `discard`/`exhaust`) is given, after entering
        combat the piles are forced to match: enter_room otherwise draws a random
        turn-1 hand from the whole deck, which won't match the player's real
        mid-combat piles. Each card lands in exactly one pile; anything not named
        in hand/discard/exhaust stays in the draw pile.
        """
        st = self._clear_to_neutral()

        loadout: dict[str, Any] = {"cmd": "set_player"}
        if hp is not None:
            loadout["hp"] = hp
        if max_hp is not None:
            loadout["max_hp"] = max_hp
        if deck is not None:
            loadout["deck"] = deck
        if relics is not None:
            loadout["relics"] = relics
        if potions is not None:
            loadout["potions"] = potions
        if len(loadout) > 1:
            r = self.send(loadout)
            if r.get("type") == "error":
                raise EngineError(f"set_player failed: {r.get('message')}\n"
                                  f"{r.get('stack_trace', '')}")

        room: dict[str, Any] = {"cmd": "enter_room", "type": "combat"}
        if encounter:
            room["encounter"] = encounter
        st = self.send(room)
        if st.get("type") == "error":
            raise EngineError(f"enter_room failed: {st.get('message')}\n"
                              f"{st.get('stack_trace', '')}")
        # A start-of-combat effect (relic / start-of-turn power) can open a
        # card_select before the first play. The live fight already resolved it,
        # so auto-resolve here (pick the first min_select cards) to reach the
        # combat_play position; set_hand below overrides the hand regardless.
        st = self._resolve_pre_combat_selects(st)
        if st.get("decision") != "combat_play":
            raise EngineError(f"expected combat_play, got {st.get('decision')}")

        # Force the real piles (enter_room drew a fresh random hand from the whole
        # deck). set_hand returns a fresh combat_play decision reflecting them.
        if hand is not None:
            cmd = {"cmd": "set_hand", "cards": hand}
            if discard is not None:
                cmd["discard"] = discard
            if exhaust is not None:
                cmd["exhaust"] = exhaust
            h = self.send(cmd)
            if h.get("type") == "error":
                raise EngineError(f"set_hand failed: {h.get('message')}\n"
                                  f"{h.get('stack_trace', '')}")
            if h.get("decision") == "combat_play":
                st = h

        # Pin the draw pile to the REAL order. set_hand left the leftover cards
        # in arbitrary order; set_draw_order reorders them top-first to match the
        # live pile, making the fight deterministic instead of re-sampling a
        # fresh shuffle every reconstruction (returns type=ok, no decision).
        if draw_order:
            d = self.send({"cmd": "set_draw_order", "cards": draw_order})
            if d.get("type") == "error":
                raise EngineError(f"set_draw_order failed: {d.get('message')}\n"
                                  f"{d.get('stack_trace', '')}")

        # Set enemy HP/block to the live values (enter_room spawns them at full
        # HP and full count, which hugely overestimates damage taken).
        if enemy_hp is not None:
            cmd = {"cmd": "set_enemies", "hps": enemy_hp}
            if enemy_block is not None:
                cmd["blocks"] = enemy_block
            e = self.send(cmd)
            if e.get("type") == "error":
                raise EngineError(f"set_enemies failed: {e.get('message')}\n"
                                  f"{e.get('stack_trace', '')}")
            if e.get("decision") == "combat_play":
                st = e

        # Apply powers (Frail/Weak/Strength/Vulnerable/...) to player + enemies,
        # else the sim ignores them and mis-simulates block/damage.
        if player_powers is not None or enemy_powers is not None:
            cmd = {"cmd": "set_powers",
                   "player": player_powers or [],
                   "enemies": enemy_powers or []}
            p = self.send(cmd)
            if p.get("type") == "error":
                raise EngineError(f"set_powers failed: {p.get('message')}\n"
                                  f"{p.get('stack_trace', '')}")
            if p.get("decision") == "combat_play":
                st = p

        # Set current energy to the live mid-turn value (enter_room reset it to a
        # fresh full turn). Without this the sim recommends cards you can't afford.
        if energy is not None:
            n = self.send({"cmd": "set_energy", "energy": energy})
            if n.get("type") == "error":
                raise EngineError(f"set_energy failed: {n.get('message')}\n"
                                  f"{n.get('stack_trace', '')}")
            if n.get("decision") == "combat_play":
                st = n

        # Restore RNG stream state so draw order + card creation replay
        # deterministically to the live run (else the sim reshuffles).
        if rng_streams:
            cmd = {"cmd": "set_rng",
                   "run": rng_streams.get("run") or {},
                   "player": rng_streams.get("player") or {}}
            g = self.send(cmd)
            if g.get("type") == "error":
                raise EngineError(f"set_rng failed: {g.get('message')}\n"
                                  f"{g.get('stack_trace', '')}")
            if g.get("decision") == "combat_play":
                st = g
        return st

    def _resolve_pre_combat_selects(self, st: dict[str, Any],
                                    max_steps: int = 6) -> dict[str, Any]:
        """Auto-resolve card_select prompts that appear before the first play.

        Picks the first `min_select` offered cards (min_select 0 => confirm with
        an empty selection). Bounded so a misbehaving prompt can't loop forever.
        """
        for _ in range(max_steps):
            if st.get("decision") != "card_select":
                return st
            cards = st.get("cards") or []
            need = int(st.get("min_select", 1) or 0)
            need = max(0, min(need, len(cards)))
            indices = ",".join(str(i) for i in range(need))
            nxt = self.act("select_cards", indices=indices)
            if nxt.get("type") == "error":
                raise EngineError(
                    f"resolving pre-combat card_select failed: "
                    f"{nxt.get('message')}\n{nxt.get('stack_trace', '')}")
            st = nxt
        return st

    def _clear_to_neutral(self, max_steps: int = 8) -> dict[str, Any] | None:
        """Advance past post-combat screens until the engine will accept a room.

        Driven off the remembered last screen; enter_room is silently ignored
        while a reward/event screen is still open.
        """
        for _ in range(max_steps):
            dec = (self.last or {}).get("decision")
            if dec in (None, "map_select", "combat_play"):
                return self.last
            action = _POST_COMBAT.get(dec)
            if action is None:
                return self.last
            self.act(action)
        return self.last

    def close(self) -> None:
        try:
            self.proc.stdin.write('{"cmd":"quit"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=2)
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

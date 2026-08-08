"""Thin wrapper around one headless Sts2Headless process.

One process serves many combats: `reset_combat()` re-enters a fight without
paying the ~0.9s run-start cost again (a reset is ~50ms).
"""
from __future__ import annotations

import json
import os
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
        self._runs = 1
        self.proc = subprocess.Popen(
            ["dotnet", dll],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=ROOT, encoding="utf-8", errors="replace",
        )
        # The engine logs heavily to stderr. If nobody reads that pipe it fills,
        # the engine blocks writing to it, and stops answering on stdout.
        self._stderr_tail: list[str] = []
        self.last: dict[str, Any] = {}
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._read()  # {"type":"ready"}
        st = self.send({"cmd": "start_run", "character": character,
                        "seed": seed or "rl", "ascension": ascension})
        if st.get("type") == "error":
            raise EngineError(f"start_run failed: {st.get('message')}")

    # ---------------- plumbing ----------------

    def _drain_stderr(self) -> None:
        for line in iter(self.proc.stderr.readline, ""):
            # Keep a short tail purely for diagnostics on crash.
            self._stderr_tail.append(line.rstrip())
            if len(self._stderr_tail) > 40:
                self._stderr_tail.pop(0)

    def _read(self) -> dict[str, Any]:
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                tail = "\n".join(self._stderr_tail[-15:])
                raise EngineError(f"engine exited unexpectedly. stderr tail:\n{tail}")
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

    def send(self, cmd: dict[str, Any]) -> dict[str, Any]:
        if self.proc.poll() is not None:
            raise EngineError("engine process is dead")
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

    def restart_run(self) -> dict[str, Any]:
        """Start a fresh run in this process.

        Model registration is cached after the first run, so this is far
        cheaper than tearing down the process and spawning a new engine.
        """
        st = self.send({"cmd": "start_run", "character": self.character,
                        "seed": f"{self.seed}_r{self._runs}", "ascension": self.ascension})
        self._runs += 1
        if st.get("type") == "error":
            raise EngineError(f"restart_run failed: {st.get('message')}")
        return st

    def reset_combat(self, encounter: str | None = None, hp: int | None = None,
                     max_hp: int | None = None, deck: list[str] | None = None,
                     relics: list[str] | None = None) -> dict[str, Any]:
        """Dismiss any leftover screen, apply loadout, and enter a fresh fight."""
        # A death ends the run, and a finished run silently refuses enter_room.
        if (self.last or {}).get("decision") == "game_over":
            self.restart_run()
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
        if len(loadout) > 1:
            r = self.send(loadout)
            if r.get("type") == "error":
                raise EngineError(f"set_player failed: {r.get('message')}")

        room: dict[str, Any] = {"cmd": "enter_room", "type": "combat"}
        if encounter:
            room["encounter"] = encounter
        st = self.send(room)
        if st.get("type") == "error":
            raise EngineError(f"enter_room failed: {st.get('message')}")

        # Some loadouts open a selection screen before the first turn (relics and
        # cards with "at combat start, choose ..." effects). Resolve those rather
        # than treating them as an error — raising here made the env throw away
        # the engine and spawn a new one, which is ~0.9s and silently dropped the
        # loadout we were trying to test.
        for _ in range(6):
            dec = st.get("decision")
            if dec == "combat_play":
                return st
            if dec == "card_select":
                cards = st.get("cards") or []
                st = (self.act("select_cards", indices="0") if cards
                      else self.act("skip_select"))
            elif dec == "bundle_select":
                st = self.act("select_bundle", bundle_index=0)
            else:
                break
            if st.get("type") == "error":
                raise EngineError(f"combat-start selection failed: {st.get('message')}")

        if st.get("decision") != "combat_play":
            # Usually means the run is no longer accepting rooms. One in-process
            # restart is ~100ms; letting this bubble up costs a process respawn.
            self.restart_run()
            if len(loadout) > 1:
                self.send(loadout)
            st = self.send(room)
            if st.get("decision") != "combat_play":
                raise EngineError(f"expected combat_play, got {st.get('decision')}")
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

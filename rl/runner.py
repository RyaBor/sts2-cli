"""Play complete runs and snapshot the state at the start of every combat.

Why this exists: training combat on the 10-card starter deck teaches the agent
to play a deck nobody actually has. Randomly *generating* decks is cheap but
produces incoherent piles — 20 rares, no synergy, relics that never co-occur.
Playing real runs produces builds that are correlated the way real ones are,
and relics arrive where the game actually hands them out (treasure rooms,
elites, bosses, shops, events).

The snapshots feed `--snapshots` in train.py, which restores one per episode
via set_player. That keeps combat episodes fast (~2ms reset) while making the
deck, relics, potions and HP realistic.

    python rl/runner.py --runs 40 --out rl/snapshots/ironclad.jsonl
    python rl/runner.py --runs 40 --policy rl/checkpoints/x.zip   # better decks
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.encoding import action_mask, decode_action, encode_obs
from rl.engine import Engine, EngineError

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------- non-combat heuristics ----------------
# Deliberately simple but not random: the point is to reach deep floors with a
# coherent deck, not to play optimally. A random drafter takes every card and
# bloats the deck, which is its own kind of unrealistic.

def _pick_map(state: dict, rng: random.Random) -> dict:
    choices = state.get("choices") or []
    if not choices:
        return {"cmd": "action", "action": "proceed"}
    # Prefer elites and rest sites: elites give relics, rests keep the run alive.
    def score(c: dict) -> float:
        t = str(c.get("room_type") or c.get("type") or "").lower()
        if "elite" in t:
            return 2.0
        if "rest" in t or "campfire" in t:
            return 1.5
        if "shop" in t or "merchant" in t:
            return 1.2
        if "treasure" in t:
            return 1.4
        return 1.0
    best = max(choices, key=lambda c: score(c) + rng.random() * 0.6)
    return {"cmd": "action", "action": "select_map_node",
            "args": {"col": best["col"], "row": best["row"]}}


def _pick_card_reward(state: dict, rng: random.Random, deck_size: int) -> dict:
    cards = state.get("cards") or []
    if not cards:
        return {"cmd": "action", "action": "skip_card_reward"}
    # Skip more often as the deck grows — taking every card is what makes
    # simulated decks unrealistic.
    skip_p = 0.15 + max(0.0, (deck_size - 15)) * 0.03
    if rng.random() < min(0.75, skip_p):
        return {"cmd": "action", "action": "skip_card_reward"}
    return {"cmd": "action", "action": "select_card_reward",
            "args": {"card_index": rng.randrange(len(cards))}}


def _pick_rest(state: dict, hp_frac: float) -> dict:
    options = [o for o in (state.get("options") or []) if o.get("is_enabled", True)]
    if not options:
        return {"cmd": "action", "action": "leave_room"}
    want = "HEAL" if hp_frac < 0.6 else "SMITH"
    choice = next((o for o in options if o.get("option_id") == want), options[0])
    return {"cmd": "action", "action": "choose_option",
            "args": {"option_index": choice["index"]}}


def _shop(state: dict, gold: int) -> dict:
    """Buy a relic if affordable — relics are the point of visiting."""
    for key, action in (("relics", "buy_relic"), ("cards", "buy_card")):
        for item in (state.get(key) or []):
            price = item.get("price")
            if price is not None and price <= gold:
                return {"cmd": "action", "action": action,
                        "args": {f"{key[:-1]}_index": item.get("index", 0)}}
    return {"cmd": "action", "action": "leave_room"}


def _combat_action(state: dict, rng: random.Random, model=None) -> dict:
    if model is not None:
        import numpy as np
        mask = action_mask(state)
        act, _ = model.predict(encode_obs(state), action_masks=mask, deterministic=False)
        name, args = decode_action(int(act), state)
        return {"cmd": "action", "action": name, **({"args": args} if args else {})}

    hand = state.get("hand") or []
    alive = [e for e in (state.get("enemies") or []) if (e.get("hp") or 0) > 0]
    playable = [c for c in hand if c.get("can_play")]
    if not playable:
        return {"cmd": "action", "action": "end_turn"}
    # Attack the lowest-HP enemy: kills reduce incoming damage fastest.
    card = max(playable, key=lambda c: (c.get("stats") or {}).get("damage", 0))
    args: dict[str, Any] = {"card_index": card["index"]}
    if str(card.get("target_type") or "") == "AnyEnemy" and alive:
        args["target_index"] = min(alive, key=lambda e: e.get("hp") or 0).get("index", 0)
    return {"cmd": "action", "action": "play_card", "args": args}


def _snapshot(state: dict) -> dict | None:
    """Record the loadout facing this combat."""
    p = state.get("player") or {}
    ctx = state.get("context") or {}
    # Combat states omit the full deck (it dominated the payload), so rebuild
    # it from the piles — during a fight those partition the whole deck.
    deck = [str(c.get("id") or "") for c in (state.get("hand") or [])]
    for pile in ("draw_pile", "discard_pile", "exhaust_pile"):
        deck += [str(x) for x in (state.get(pile) or [])]
    deck = [d for d in deck if d]
    if not deck:
        deck = [str(c.get("id") or "") for c in (p.get("deck") or []) if c.get("id")]
    if not deck:
        return None
    return {
        "act": ctx.get("act"),
        "floor": ctx.get("floor"),
        "encounter": ctx.get("encounter"),
        "room_type": ctx.get("room_type"),
        "hp": p.get("hp"),
        "max_hp": p.get("max_hp"),
        "gold": p.get("gold"),
        "deck": [d.split(".", 1)[-1] for d in deck],
        "relics": [r.get("id") for r in (p.get("relics") or []) if r.get("id")],
        "potions": [x.get("id") for x in (p.get("potions") or []) if x.get("id")],
    }


def play_run(engine: Engine, rng: random.Random, model=None,
             max_steps: int = 4000) -> tuple[list[dict], dict]:
    """Play one run to completion; return (snapshots, summary)."""
    snaps: list[dict] = []
    state = engine.last
    seen_combat = False
    steps = 0
    result = {"floors": 0, "act": 1, "outcome": "incomplete"}

    while steps < max_steps:
        steps += 1
        if not state or state.get("type") == "error":
            state = engine.send({"cmd": "action", "action": "proceed"})
            continue

        dec = state.get("decision")
        ctx = state.get("context") or {}
        result["act"] = ctx.get("act", result["act"])
        result["floors"] = max(result["floors"], ctx.get("floor") or 0)
        p = state.get("player") or {}

        if dec == "game_over":
            result["outcome"] = "win" if state.get("victory") else "loss"
            break

        if dec == "combat_play":
            if not seen_combat:
                snap = _snapshot(state)
                if snap:
                    snaps.append(snap)
                seen_combat = True
            state = engine.send(_combat_action(state, rng, model))
            continue

        seen_combat = False  # next combat gets a fresh snapshot

        if dec == "map_select":
            state = engine.send(_pick_map(state, rng))
        elif dec == "card_reward":
            state = engine.send(_pick_card_reward(state, rng, len(p.get("deck") or [])))
        elif dec == "rest_site":
            hp = float(p.get("hp") or 0); mx = float(p.get("max_hp") or 1)
            state = engine.send(_pick_rest(state, hp / max(mx, 1.0)))
        elif dec == "shop":
            state = engine.send(_shop(state, int(p.get("gold") or 0)))
        elif dec == "event_choice":
            opts = [o for o in (state.get("options") or []) if not o.get("is_locked")]
            state = (engine.send({"cmd": "action", "action": "choose_option",
                                  "args": {"option_index": rng.choice(opts)["index"]}})
                     if opts else engine.send({"cmd": "action", "action": "leave_room"}))
        elif dec == "bundle_select":
            state = engine.send({"cmd": "action", "action": "select_bundle",
                                 "args": {"bundle_index": 0}})
        elif dec == "card_select":
            cards = state.get("cards") or []
            state = (engine.send({"cmd": "action", "action": "select_cards",
                                  "args": {"indices": "0"}})
                     if cards else engine.send({"cmd": "action", "action": "skip_select"}))
        else:
            state = engine.send({"cmd": "action", "action": "proceed"})

    result["steps"] = steps
    return snaps, result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=25)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--policy", default=None, help="checkpoint to play combats with")
    ap.add_argument("--out", default=os.path.join(HERE, "snapshots", "ironclad.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = None
    if args.policy:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(args.policy, device="cpu")
        print(f"combats played by {args.policy}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(args.seed)
    total, outcomes = 0, {}

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.runs):
            eng = None
            try:
                eng = Engine(character=args.character, seed=f"run{args.seed}_{i}",
                             ascension=args.ascension)
                snaps, res = play_run(eng, rng, model)
                for s in snaps:
                    f.write(json.dumps(s) + "\n")
                total += len(snaps)
                outcomes[res["outcome"]] = outcomes.get(res["outcome"], 0) + 1
                print(f"run {i+1}/{args.runs}: act={res['act']} floor={res['floors']} "
                      f"{res['outcome']} snapshots={len(snaps)} (total {total})")
            except EngineError as e:
                print(f"run {i+1}: engine error: {str(e)[:90]}")
            finally:
                if eng is not None:
                    eng.close()

    print(f"\nwrote {total} snapshots to {args.out}")
    print("outcomes:", outcomes)


if __name__ == "__main__":
    main()

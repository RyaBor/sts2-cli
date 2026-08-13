"""Play one full A10 run through the CLI, routing decisions to the three agents.

combat_play -> combat agent   (sample records go to `combat_samples`, tagged by
                               which combat; return = that combat's HP retained)
card_reward -> card agent      (return = overall game victory)
map_select  -> path agent      (return = overall game victory)
everything else -> sensible fixed defaults (events/rest/shop/in-combat selects)
so the run always progresses.
"""
from __future__ import annotations

from typing import Any

from agents import (encode_obs, action_mask, decode_action,
                    encode_card_reward, card_reward_mask, MAX_OFFER,
                    encode_map, map_mask,
                    encode_shop, shop_mask, SHOP_LEAVE, SHOP_REMOVE,
                    SHOP_BUY_CARD, SHOP_BUY_RELIC, SHOP_BUY_POTION)


def _worst_deck_index(deck: list) -> int:
    """Which deck card to purge: a basic Strike/Defend or a curse/status, else 0."""
    for i, c in enumerate(deck):
        cid = str(c.get("id") or c.get("name") or "").upper()
        if "STRIKE" in cid or "DEFEND" in cid or c.get("type") in ("Curse", "Status"):
            return i
    return 0


def _shop_action(st, a: int) -> tuple[str, dict]:
    if a == SHOP_LEAVE:
        return "leave_room", {}
    if a == SHOP_REMOVE:
        return "remove_card", {"card_index": _worst_deck_index((st.get("player") or {}).get("deck") or [])}
    if a >= SHOP_BUY_POTION:
        return "buy_potion", {"potion_index": a - SHOP_BUY_POTION}
    if a >= SHOP_BUY_RELIC:
        return "buy_relic", {"relic_index": a - SHOP_BUY_RELIC}
    return "buy_card", {"card_index": a - SHOP_BUY_CARD}


def _tier_from_room(room: str | None, enemies: list) -> str:
    r = (room or "").upper()
    if "BOSS" in r:
        return "BOSS"
    if "ELITE" in r:
        return "ELITE"
    return "COMBAT"


# Decisions that mean a fight is genuinely OVER (reward/map/etc.). Note card_SELECT
# (Armaments, discover) is an IN-combat prompt and is deliberately NOT here — that
# was splitting one fight into several counted "combats".
POST_COMBAT = {"card_reward", "map_select", "rest_site", "event_choice",
               "shop", "bundle_select"}


def play_run(eng, agents: dict, greedy: bool = False, max_steps: int = 4000) -> dict:
    combats: list[dict] = []
    combat_samples: list[tuple] = []      # (obs, mask, action, combat_idx)
    card_samples: list[tuple] = []        # (obs, mask, action)   card rewards
    shop_samples: list[tuple] = []        # (obs, mask, action)   shop / economy
    path_samples: list[tuple] = []        # (obs, mask, action)

    st = eng.last
    prev_decision = None
    cur_combat = -1                       # index into combats, -1 = not in combat
    last_room = "COMBAT"
    last_key, stuck = None, 0

    def in_combat_start(state):
        nonlocal cur_combat
        combats.append({"tier": _tier_from_room(last_room, state.get("enemies") or []),
                        "start_hp": float((state.get("player") or {}).get("hp") or 0),
                        "end_hp": 0.0, "won": False})
        cur_combat = len(combats) - 1

    def close_combat(state, won):
        if cur_combat >= 0:
            combats[cur_combat]["end_hp"] = float((state.get("player") or {}).get("hp") or 0)
            combats[cur_combat]["won"] = won

    for _ in range(max_steps):
        if st.get("type") == "error":
            break
        dec = st.get("decision", "")

        # combat boundary bookkeeping: open a fight on the first combat_play, and
        # only close it (as won) when we reach a real post-combat screen. Staying
        # in combat across an in-combat card_select no longer ends the fight.
        if dec == "combat_play" and cur_combat < 0:
            in_combat_start(st)
        elif cur_combat >= 0 and dec in POST_COMBAT:
            close_combat(st, won=True)
            cur_combat = -1

        # stuck guard
        p = st.get("player") or {}
        key = f"{dec}:{st.get('round')}:{p.get('hp')}:{len(st.get('hand') or [])}"
        if key == last_key:
            stuck += 1
            if stuck > 30:
                break
        else:
            stuck, last_key = 0, key
        prev_decision = dec

        if dec == "game_over":
            if cur_combat >= 0:               # died mid-combat
                close_combat(st, won=bool(st.get("victory")))
            return {
                "victory": bool(st.get("victory")),
                "act": st.get("act") or 1, "floor": st.get("floor") or 0,
                "combats": combats, "combat_samples": combat_samples,
                "card_samples": card_samples, "shop_samples": shop_samples,
                "path_samples": path_samples,
            }

        if dec == "combat_play":
            obs, mask = encode_obs(st), action_mask(st)
            a = agents["combat"].act(obs, mask, greedy)
            combat_samples.append((obs, mask, a, cur_combat))
            name, args = decode_action(a, st)
            st = eng.act(name, **args)

        elif dec == "map_select":
            ch = st.get("choices") or []
            if not ch:
                st = eng.act("proceed"); continue
            obs, mask = encode_map(st), map_mask(st)
            a = agents["path"].act(obs, mask, greedy)
            a = min(a, len(ch) - 1)
            path_samples.append((obs, mask, a))
            last_room = None
            for k in ("room_type", "type", "symbol", "node_type", "kind"):
                if ch[a].get(k):
                    last_room = str(ch[a][k]); break
            st = eng.act("select_map_node", col=ch[a]["col"], row=ch[a]["row"])

        elif dec == "card_reward":
            cards = st.get("cards") or []
            obs, mask = encode_card_reward(st), card_reward_mask(st)
            a = agents["card"].act_reward(obs, mask, greedy)
            card_samples.append((obs, mask, a))
            if a >= MAX_OFFER or a >= len(cards):
                st = eng.act("skip_card_reward")
            else:
                st = eng.act("select_card_reward", card_index=a)

        elif dec == "shop":
            for _ in range(14):                       # buy a few things, then leave
                obs, mask = encode_shop(st), shop_mask(st)
                a = agents["card"].act_shop(obs, mask, greedy)
                shop_samples.append((obs, mask, a))
                name, args = _shop_action(st, a)
                if name == "leave_room":
                    st = eng.act("leave_room"); break
                nxt = eng.act(name, **args)
                if nxt is None or nxt.get("type") == "error" or nxt.get("decision") != "shop":
                    st = nxt if (nxt and nxt.get("type") != "error") else eng.act("leave_room")
                    break
                st = nxt

        elif dec == "event_choice":
            opts = st.get("options") or []
            ch = next((o for o in opts if not o.get("is_locked")), opts[0] if opts else None)
            st = eng.act("choose_option", option_index=ch["index"]) if ch else eng.act("leave_room")

        elif dec == "rest_site":
            opts = [o for o in (st.get("options") or []) if o.get("is_enabled", True)]
            ch = next((o for o in opts if o.get("option_id") == "HEAL"), opts[0] if opts else None)
            st = eng.act("choose_option", option_index=ch["index"]) if ch else eng.act("leave_room")

        elif dec == "card_select":
            st = (eng.act("select_cards", indices="0") if (st.get("cards"))
                  else eng.act("skip_select"))

        elif dec == "bundle_select":
            st = eng.act("select_bundle", bundle_index=0)

        else:
            st = eng.act("proceed")

    # ran out of steps
    return {"victory": False, "act": st.get("act") or 1, "floor": st.get("floor") or 0,
            "combats": combats, "combat_samples": combat_samples,
            "card_samples": card_samples, "shop_samples": shop_samples,
            "path_samples": path_samples}

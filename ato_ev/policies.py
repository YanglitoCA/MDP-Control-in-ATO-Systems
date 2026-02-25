from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SurfaceThresholds:
    b0: int
    bO: np.ndarray
    bS: np.ndarray


def hard_switch_action(state: np.ndarray, event_type: str, event_i: int | None, surfaces: SurfaceThresholds) -> Dict:
    I = int(state[0])
    B = state[1:]
    order_q = max(0, int(surfaces.b0) - I)
    action = {"order_q": order_q, "accept": None, "fulfill": None}
    if event_type == "ARRIVAL" and event_i is not None:
        i = int(event_i)
        action["accept"] = int(B[i] < surfaces.bO[i])
    if event_type == "DUE" and event_i is not None:
        i = int(event_i)
        action["fulfill"] = int(I > 0 and B[i] > 0 and I >= surfaces.bS[i])
    return action


def saac_action(scores: Dict[str, np.ndarray | float], state: np.ndarray, hard: bool = True) -> Dict:
    I = int(state[0])
    order_score = float(scores["order"])
    full_order = int(scores.get("full_order_level", 0) or 0)
    do_order = int(order_score > 0.5) if hard else float(order_score)
    order_q = do_order * max(0, full_order - I)
    accept = int(float(scores["accept"]) > 0.5) if hard else float(scores["accept"])
    fulfill = int(float(scores["fulfill"]) > 0.5) if hard else float(scores["fulfill"])
    return {"order_q": int(order_q), "accept": accept, "fulfill": fulfill}


def is_policy_action(state: np.ndarray, event_type: str, event_i: int | None, b0_star: int, b_star: np.ndarray) -> Dict:
    I = int(state[0])
    B = state[1:]
    action = {"order_q": max(0, b0_star - I), "accept": None, "fulfill": None}
    if event_type == "ARRIVAL" and event_i is not None:
        action["accept"] = int(B[event_i] < b_star[event_i])
    if event_type == "DUE" and event_i is not None:
        action["fulfill"] = int(I > 0 and B[event_i] > 0)
    return action

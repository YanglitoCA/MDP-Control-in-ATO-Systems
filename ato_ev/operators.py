from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from ato_ev.policies import SurfaceThresholds


def _clip_state(x: np.ndarray, params: Dict) -> np.ndarray:
    caps = np.concatenate([[params["B0"]], np.asarray(params["B_max"], dtype=np.int32)])
    return np.clip(x, 0, caps).astype(np.int32)


def hard_decisions(x: np.ndarray, i: int, surfaces: SurfaceThresholds) -> tuple[int, int, int]:
    I = int(x[0])
    Bi = int(x[i + 1])
    q = max(0, int(surfaces.b0) - I)
    accept = int(Bi < int(surfaces.bO[i]))
    fulfill = int(I > 0 and Bi > 0 and I >= int(surfaces.bS[i]))
    return q, accept, fulfill


def T_hard(V: Callable[[np.ndarray], float], x: np.ndarray, params: Dict, surfaces: SurfaceThresholds) -> float:
    n = int(params["n"])
    mu = float(params["mu"])
    lam = np.asarray(params["lam"], dtype=float)
    omega = np.asarray(params["omega"], dtype=float)
    theta = float(params["theta"])
    gamma = float(params["gamma"])
    h = float(params["h"])
    p_reject = np.asarray(params["p_reject"], dtype=float)
    d_delay = np.asarray(params["d_delay"], dtype=float)
    total_rate = mu + lam.sum() + omega.sum() + theta

    x = np.asarray(x, dtype=np.int32)
    hold = h * float(x[0])
    expected = 0.0

    x_supply = x.copy()
    x_supply[0] += 1
    expected += (mu / total_rate) * V(_clip_state(x_supply, params))

    for i in range(n):
        _, accept, _ = hard_decisions(x, i, surfaces)
        x_arr = x.copy()
        arr_cost = 0.0
        if accept == 1:
            x_arr[i + 1] += 1
        else:
            arr_cost += p_reject[i]
        expected += (lam[i] / total_rate) * (arr_cost + gamma * V(_clip_state(x_arr, params)))

    for i in range(n):
        _, _, fulfill = hard_decisions(x, i, surfaces)
        x_due = x.copy()
        due_cost = 0.0
        if fulfill == 1 and x_due[0] > 0 and x_due[i + 1] > 0:
            x_due[0] -= 1
            x_due[i + 1] -= 1
        elif x_due[i + 1] > 0:
            due_cost += d_delay[i]
        expected += (omega[i] / total_rate) * (due_cost + gamma * V(_clip_state(x_due, params)))

    expected += (theta / total_rate) * (gamma * V(x))
    return hold + expected

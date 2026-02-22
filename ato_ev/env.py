from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class Event:
    event_type: str
    event_i: Optional[int]


class ATOEVEnv:
    """Uniformized event-driven ATO-EV simulator."""

    def __init__(self, params: Dict):
        self.params = params
        self.n = int(params["n"])
        self.B0 = int(params["B0"])
        self.B_max = np.asarray(params["B_max"], dtype=np.int32)
        self.mu = float(params["mu"])
        self.lam = np.asarray(params["lam"], dtype=np.float64)
        self.omega = np.asarray(params["omega"], dtype=np.float64)
        self.theta = float(params["theta"])
        self.h = float(params["h"])
        self.p_reject = np.asarray(params["p_reject"], dtype=np.float64)
        self.d_delay = np.asarray(params["d_delay"], dtype=np.float64)
        self.gamma = float(params["gamma"])
        self.horizon = int(params.get("horizon", 100))
        self.normalize_obs = bool(params.get("normalize_obs", False))

        self._caps = np.concatenate([[self.B0], self.B_max]).astype(np.float32)
        self._lambda_u = float(self.mu + self.lam.sum() + self.omega.sum() + self.theta)
        self._event_probs = self._build_event_probs()

        self.rng = np.random.default_rng(params.get("seed", None))
        self.t = 0
        self.state = np.zeros(self.n + 1, dtype=np.int32)

    def _build_event_probs(self) -> np.ndarray:
        probs = [self.mu]
        probs.extend(self.lam.tolist())
        probs.extend(self.omega.tolist())
        probs.append(self.theta)
        probs = np.asarray(probs, dtype=np.float64) / self._lambda_u
        probs = probs / probs.sum()
        return probs

    def event_probabilities(self) -> np.ndarray:
        return self._event_probs.copy()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.state = np.zeros(self.n + 1, dtype=np.int32)
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = self.state.astype(np.float32)
        if self.normalize_obs:
            obs = obs / np.maximum(self._caps, 1.0)
        return obs

    def _sample_event(self) -> Event:
        idx = int(self.rng.choice(len(self._event_probs), p=self._event_probs))
        if idx == 0:
            return Event("SUPPLY", None)
        if 1 <= idx <= self.n:
            return Event("ARRIVAL", idx - 1)
        if self.n + 1 <= idx <= 2 * self.n:
            return Event("DUE", idx - self.n - 1)
        return Event("DUMMY", None)

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        self.t += 1
        I = int(self.state[0])
        forced = {"forced_reject": False, "forced_no_fulfill": False}

        order_q = int(action.get("order_q", 0))
        order_q = int(np.clip(order_q, 0, self.B0 - I))
        if order_q > 0:
            self.state[0] = np.int32(min(self.B0, I + order_q))
            I = int(self.state[0])

        event = self._sample_event()
        hold_cost = self.h * float(self.state[0])
        reject_cost = 0.0
        delay_cost = 0.0

        if event.event_type == "SUPPLY":
            self.state[0] = np.int32(min(self.B0, int(self.state[0]) + 1))
        elif event.event_type == "ARRIVAL":
            i = int(event.event_i)
            accept = int(action.get("accept", 0) or 0)
            if self.state[i + 1] >= self.B_max[i]:
                forced["forced_reject"] = True
                accept = 0
            if accept == 1:
                self.state[i + 1] = np.int32(min(self.B_max[i], int(self.state[i + 1]) + 1))
            else:
                reject_cost += float(self.p_reject[i])
        elif event.event_type == "DUE":
            i = int(event.event_i)
            fulfill = int(action.get("fulfill", 0) or 0)
            if self.state[0] <= 0 or self.state[i + 1] <= 0:
                forced["forced_no_fulfill"] = True
                fulfill = 0
            if fulfill == 1:
                self.state[0] = np.int32(max(0, int(self.state[0]) - 1))
                self.state[i + 1] = np.int32(max(0, int(self.state[i + 1]) - 1))
            else:
                if self.state[i + 1] > 0:
                    delay_cost += float(self.d_delay[i])

        cost = hold_cost + reject_cost + delay_cost
        reward = -cost
        done = self.t >= self.horizon
        info = {
            "event_type": event.event_type,
            "event_i": event.event_i,
            "state_int": self.state.copy(),
            "cost_breakdown": {"hold": hold_cost, "reject": reject_cost, "delay": delay_cost},
            "forced": forced,
        }
        return self._obs(), reward, done, info

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from ato_ev.env import ATOEVEnv
from ato_ev.policies import is_policy_action


def run_thresholds(params, b0, b, H, R, seed=0):
    vals = []
    for r in range(R):
        env = ATOEVEnv(params)
        obs = env.reset(seed=seed + r)
        g = 1.0
        total = 0.0
        for _ in range(H):
            evt = env._sample_event()
            a = is_policy_action(env.state.copy(), evt.event_type, evt.event_i, b0, b)
            _, rew, done, _ = env.step(a)
            total += g * (-rew)
            g *= params["gamma"]
            if done:
                break
        vals.append(total)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--H", type=int, default=200)
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--exp", type=str, default="is")
    args = ap.parse_args()

    params = {"n": 3, "B0": 20, "B_max": [20, 20, 20], "mu": 1.0, "lam": [0.8, 0.6, 0.5], "omega": [0.7, 0.4, 0.6], "theta": 1.0, "h": 0.2, "p_reject": [2.0, 2.5, 3.0], "d_delay": [1.0, 1.0, 1.5], "gamma": 0.99, "horizon": args.H}

    rng = np.random.default_rng(0)
    best = None
    for _ in range(args.M):
        b0 = int(rng.integers(0, params["B0"] + 1))
        b = rng.integers(0, np.asarray(params["B_max"]) + 1).astype(int)
        v = run_thresholds(params, b0, b, args.H, args.R)
        if best is None or v < best[0]:
            best = (v, b0, b)

    _, b0_best, b_best = best
    for _ in range(args.K):
        improved = False
        for j in range(params["n"] + 1):
            current = b0_best if j == 0 else b_best[j - 1]
            for cand in [current - 1, current, current + 1]:
                if j == 0 and not (0 <= cand <= params["B0"]):
                    continue
                if j > 0 and not (0 <= cand <= params["B_max"][j - 1]):
                    continue
                cb0, cb = b0_best, b_best.copy()
                if j == 0:
                    cb0 = cand
                else:
                    cb[j - 1] = cand
                v = run_thresholds(params, cb0, cb, args.H, args.R)
                if v < best[0]:
                    best = (v, cb0, cb)
                    b0_best, b_best = cb0, cb
                    improved = True
        if not improved:
            break

    out = Path("runs") / args.exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "is_thresholds.json").write_text(json.dumps({"best_cost": best[0], "b0": int(b0_best), "b": b_best.tolist()}, indent=2))


if __name__ == "__main__":
    main()

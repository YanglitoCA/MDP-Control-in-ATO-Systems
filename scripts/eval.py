#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ato_ev.env import ATOEVEnv
from ato_ev.policies import is_policy_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H_eval", type=int, default=200)
    ap.add_argument("--R_eval", type=int, default=20)
    ap.add_argument("--exp", type=str, default="eval")
    args = ap.parse_args()

    params = {"n": 3, "B0": 20, "B_max": [20, 20, 20], "mu": 1.0, "lam": [0.8, 0.6, 0.5], "omega": [0.7, 0.4, 0.6], "theta": 1.0, "h": 0.2, "p_reject": [2.0, 2.5, 3.0], "d_delay": [1.0, 1.0, 1.5], "gamma": 0.99, "horizon": args.H_eval}
    b0, b = 10, np.array([10, 10, 10])

    rows = []
    for r in range(args.R_eval):
        env = ATOEVEnv(params)
        env.reset(seed=r)
        disc = 1.0
        total = 0.0
        accepts = np.zeros(params["n"])
        arrivals = np.zeros(params["n"])
        fulfills = np.zeros(params["n"])
        dues = np.zeros(params["n"])
        for _ in range(args.H_eval):
            evt = env._sample_event()
            a = is_policy_action(env.state.copy(), evt.event_type, evt.event_i, b0, b)
            _, rew, done, info = env.step(a)
            total += disc * (-rew)
            disc *= params["gamma"]
            if info["event_type"] == "ARRIVAL":
                i = info["event_i"]
                arrivals[i] += 1
                accepts[i] += int((a["accept"] or 0) == 1 and not info["forced"]["forced_reject"])
            if info["event_type"] == "DUE":
                i = info["event_i"]
                dues[i] += 1
                fulfills[i] += int((a["fulfill"] or 0) == 1 and not info["forced"]["forced_no_fulfill"])
            if done:
                break
        rows.append({
            "discounted_cost": total,
            "accept_rate_overall": float(accepts.sum() / max(1, arrivals.sum())),
            "fulfill_rate_overall": float(fulfills.sum() / max(1, dues.sum())),
        })

    out = Path("runs") / args.exp
    out.mkdir(parents=True, exist_ok=True)
    with (out / "eval.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    costs = np.array([r["discounted_cost"] for r in rows])
    print({"cost_mean": float(costs.mean()), "cost_std": float(costs.std())})


if __name__ == "__main__":
    main()

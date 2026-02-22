#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ato_ev.env import ATOEVEnv
from ato_ev.networks import CriticV, PooledEncoder, SAACActorNet


def build_features(states, params):
    B = states.shape[0]
    n = params["n"]
    class_feats = []
    for i in range(n):
        class_feats.append(np.stack([states[:, i + 1], np.full(B, params["lam"][i]), np.full(B, params["omega"][i]), np.full(B, params["p_reject"][i]), np.full(B, params["d_delay"][i]), np.full(B, params["B_max"][i])], axis=-1))
    return torch.tensor(np.stack(class_feats, axis=1), dtype=torch.float32), torch.tensor(states[:, [0]], dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--exp", type=str, default="saac")
    args = ap.parse_args()

    params = {"n": 3, "B0": 20, "B_max": [20, 20, 20], "mu": 1.0, "lam": [0.8, 0.6, 0.5], "omega": [0.7, 0.4, 0.6], "theta": 1.0, "h": 0.2, "p_reject": [2.0, 2.5, 3.0], "d_delay": [1.0, 1.0, 1.5], "gamma": 0.99, "horizon": 100}
    env = ATOEVEnv(params)
    enc = PooledEncoder(6)
    actor = SAACActorNet(128)
    critic = CriticV(128)
    opt = torch.optim.Adam(list(enc.parameters()) + list(actor.parameters()) + list(critic.parameters()), lr=1e-3)

    logs = []
    obs = env.reset(seed=0)
    for it in range(args.iters):
        action = {"order_q": np.random.randint(0, params["B0"] + 1), "accept": np.random.randint(0, 2), "fulfill": np.random.randint(0, 2)}
        nxt, rew, done, _ = env.step(action)
        st = np.asarray([obs], dtype=np.float32)
        cf, inv = build_features(st, params)
        z = enc(cf, inv)
        pol = actor(z)
        v = critic(z)
        loss = ((v + torch.tensor([rew], dtype=torch.float32)) ** 2).mean() + sum((x - 0.5).pow(2).mean() for x in pol.values())
        opt.zero_grad(); loss.backward(); opt.step()
        logs.append({"iter": it, "loss": float(loss.item())})
        obs = env.reset(seed=it + 1) if done else nxt

    out = Path("runs") / args.exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_logs.json").write_text(json.dumps(logs, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ato_ev.buffer import ReplayBuffer
from ato_ev.env import ATOEVEnv
from ato_ev.networks import ActorSurfaces, CriticV, PooledEncoder
from ato_ev.sacr import compute_sacr


def build_features(states, params):
    # states: (B, n+1)
    B, d = states.shape
    n = params["n"]
    class_feats = []
    for i in range(n):
        feat = np.stack(
            [
                states[:, i + 1],
                np.full(B, params["lam"][i]),
                np.full(B, params["omega"][i]),
                np.full(B, params["p_reject"][i]),
                np.full(B, params["d_delay"][i]),
                np.full(B, params["B_max"][i]),
            ],
            axis=-1,
        )
        class_feats.append(feat)
    class_feats = np.stack(class_feats, axis=1)
    inv = states[:, [0]]
    return torch.tensor(class_feats, dtype=torch.float32), torch.tensor(inv, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", type=str, default="sarl")
    args = ap.parse_args()

    params = {
        "n": 3,
        "B0": 20,
        "B_max": [20, 20, 20],
        "mu": 1.0,
        "lam": [0.8, 0.6, 0.5],
        "omega": [0.7, 0.4, 0.6],
        "theta": 1.0,
        "h": 0.2,
        "p_reject": [2.0, 2.5, 3.0],
        "d_delay": [1.0, 1.0, 1.5],
        "gamma": 0.99,
        "horizon": 200,
    }
    if args.config:
        params.update(json.loads(Path(args.config).read_text()))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = ATOEVEnv(params)
    buf = ReplayBuffer(50000)

    encoder = PooledEncoder(class_feat_dim=6)
    actor = ActorSurfaces(128, params["n"], params["B0"], params["B_max"])
    critic = CriticV(128)
    target = CriticV(128)
    target.load_state_dict(critic.state_dict())

    opt_c = torch.optim.Adam(list(encoder.parameters()) + list(critic.parameters()), lr=1e-3)
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-3)

    obs = env.reset(seed=args.seed)
    logs = []
    caps = torch.tensor([params["B0"]] + params["B_max"], dtype=torch.float32)
    eta, tau, kappa = 0.1, 0.01, 5.0

    for it in range(args.iters):
        action = {"order_q": np.random.randint(0, params["B0"] + 1), "accept": np.random.randint(0, 2), "fulfill": np.random.randint(0, 2)}
        nxt, rew, done, info = env.step(action)
        buf.add(obs, info["event_type"], info["event_i"], action, -rew, nxt)
        obs = env.reset(seed=args.seed + it + 1) if done else nxt

        if len(buf) < args.batch:
            continue

        states, *_rest, costs, next_states = buf.sample(args.batch)
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        costs = torch.tensor(np.asarray(costs), dtype=torch.float32)

        cf, inv = build_features(states, params)
        z = encoder(cf, inv)
        v = critic(z)
        with torch.no_grad():
            ncf, ninv = build_features(next_states, params)
            nv = target(encoder(ncf, ninv))
            y = costs + params["gamma"] * nv

        def eval_v(x):
            x_np = x.detach().cpu().numpy()
            cff, invf = build_features(x_np, params)
            return critic(encoder(cff, invf))

        sac = compute_sacr(eval_v, torch.tensor(states), caps)
        lc = torch.mean((v - y) ** 2) + eta * sac
        opt_c.zero_grad(); lc.backward(); opt_c.step()

        surf = actor(z)
        b0, bO, bS = surf["b0"], surf["bO"], surf["bS"]
        I = torch.tensor(states[:, 0])
        B = torch.tensor(states[:, 1:])
        soft_a = torch.sigmoid(kappa * (bO - B)).mean(dim=1)
        soft_f = torch.sigmoid(kappa * (I[:, None] - bS)).mean(dim=1)
        q_soft = torch.nn.functional.softplus(b0 - I)
        t_soft = costs + 0.01 * (q_soft + soft_a + soft_f)
        la = torch.mean((v.detach() - t_soft) ** 2)
        opt_a.zero_grad(); la.backward(); opt_a.step()

        with torch.no_grad():
            for p, pt in zip(critic.parameters(), target.parameters()):
                pt.mul_(1 - tau).add_(tau * p)

        logs.append({"iter": it, "critic_loss": float(lc.item()), "actor_loss": float(la.item())})

    out = Path("runs") / args.exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_logs.json").write_text(json.dumps(logs, indent=2))


if __name__ == "__main__":
    main()

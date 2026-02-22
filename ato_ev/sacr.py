from __future__ import annotations

import torch


def _clip_states(x: torch.Tensor, caps: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.zeros_like(x), torch.minimum(x, caps))


def compute_sacr(critic_eval_fn, states: torch.Tensor, caps: torch.Tensor, class_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Compute nonnegative structural penalties from discrete differences."""
    device = states.device
    B, d = states.shape
    n = d - 1
    if class_idx is None:
        class_idx = torch.arange(n, device=device)

    e0 = torch.zeros((1, d), device=device)
    e0[:, 0] = 1.0

    V = critic_eval_fn(states)
    V_ip = critic_eval_fn(_clip_states(states + e0, caps))
    V_im = critic_eval_fn(_clip_states(states - e0, caps))
    second_I = V_ip - 2 * V + V_im
    p1 = torch.relu(-second_I).pow(2).mean()

    penalties = [p1]
    for i in class_idx.tolist():
        ei = torch.zeros((1, d), device=device)
        ei[:, i + 1] = 1.0

        V_bip = critic_eval_fn(_clip_states(states + ei, caps))
        V_bim = critic_eval_fn(_clip_states(states - ei, caps))
        V_ip_bip = critic_eval_fn(_clip_states(states + e0 + ei, caps))

        mixed = V_ip_bip - V_ip - V_bip + V
        # Condition 2 (submodularity): mixed <= 0
        p2 = torch.relu(mixed).pow(2).mean()
        # Condition 3 (supermodularity-like opposite direction): -mixed <= 0 -> mixed >= 0
        p3 = torch.relu(-mixed).pow(2).mean()

        first_B = V_bip - V
        # Condition 4 monotonicity in backlog (nondecreasing value/cost)
        p4 = torch.relu(-first_B).pow(2).mean()

        # Condition 5 joint monotonicity uses mixed curvature absolute violation around zero band
        p5 = mixed.pow(2).mean()
        penalties.extend([p2, p3, p4, p5])

    return torch.stack(penalties).sum()

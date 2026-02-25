import torch

from ato_ev.sacr import compute_sacr


def test_sacr_nonnegative_and_finite():
    caps = torch.tensor([20.0, 20.0, 20.0, 20.0])
    states = torch.randint(0, 20, (64, 4)).float()

    def critic_eval(x):
        return x.sum(dim=1) * 0.1

    val = compute_sacr(critic_eval, states, caps)
    assert torch.isfinite(val)
    assert val >= 0

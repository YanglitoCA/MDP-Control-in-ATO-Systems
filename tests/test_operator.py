import numpy as np

from ato_ev.operators import T_hard
from ato_ev.policies import SurfaceThresholds


def test_operator_returns_finite():
    params = {"n": 2, "B0": 10, "B_max": [10, 10], "mu": 1.0, "lam": [0.5, 0.4], "omega": [0.3, 0.2], "theta": 0.6, "h": 0.1, "p_reject": [2.0, 2.0], "d_delay": [1.0, 1.0], "gamma": 0.95}
    x = np.array([3, 2, 1], dtype=np.int32)
    s = SurfaceThresholds(b0=5, bO=np.array([4, 4]), bS=np.array([2, 2]))

    def V(xin):
        return float(np.sum(xin))

    v = T_hard(V, x, params, s)
    assert np.isfinite(v)

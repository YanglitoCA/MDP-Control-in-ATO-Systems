import numpy as np

from ato_ev.env import ATOEVEnv


def make_params():
    return {"n": 3, "B0": 20, "B_max": [20, 20, 20], "mu": 1.0, "lam": [0.8, 0.6, 0.5], "omega": [0.7, 0.4, 0.6], "theta": 1.0, "h": 0.2, "p_reject": [2.0, 2.5, 3.0], "d_delay": [1.0, 1.0, 1.5], "gamma": 0.99, "horizon": 100000}


def test_bounds_random_actions():
    env = ATOEVEnv(make_params())
    env.reset(seed=1)
    for _ in range(100000):
        _, _, done, info = env.step({"order_q": np.random.randint(0, 30), "accept": np.random.randint(0, 2), "fulfill": np.random.randint(0, 2)})
        x = info["state_int"]
        assert 0 <= x[0] <= env.B0
        assert np.all((x[1:] >= 0) & (x[1:] <= env.B_max))
        if done:
            break


def test_event_probabilities_sum_to_one():
    env = ATOEVEnv(make_params())
    assert abs(env.event_probabilities().sum() - 1.0) < 1e-10


def test_reproducibility_first_100_transitions():
    p = make_params()
    p["horizon"] = 200
    e1, e2 = ATOEVEnv(p), ATOEVEnv(p)
    e1.reset(seed=123)
    e2.reset(seed=123)
    traj1, traj2 = [], []
    for _ in range(100):
        a = {"order_q": 1, "accept": 1, "fulfill": 1}
        _, _, _, i1 = e1.step(a)
        _, _, _, i2 = e2.step(a)
        traj1.append((i1["event_type"], i1["event_i"], tuple(i1["state_int"].tolist())))
        traj2.append((i2["event_type"], i2["event_i"], tuple(i2["state_int"].tolist())))
    assert traj1 == traj2

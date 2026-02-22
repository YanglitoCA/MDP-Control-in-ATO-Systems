from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buf = deque(maxlen=capacity)

    def add(self, x, e_type, e_i, a, c, xp):
        self.buf.append((np.asarray(x), e_type, e_i, a, float(c), np.asarray(xp)))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        return list(zip(*batch))

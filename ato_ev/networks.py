from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class ClassEncoderMLP(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, emb_dim), nn.ReLU())

    def forward(self, x_i: torch.Tensor) -> torch.Tensor:
        return self.net(x_i)


class PooledEncoder(nn.Module):
    def __init__(self, class_feat_dim: int, emb_dim: int = 64, z_dim: int = 128):
        super().__init__()
        self.class_encoder = ClassEncoderMLP(class_feat_dim, emb_dim)
        self.post = nn.Sequential(nn.Linear(emb_dim + 1, z_dim), nn.ReLU(), nn.Linear(z_dim, z_dim), nn.ReLU())

    def forward(self, class_feats: torch.Tensor, inventory: torch.Tensor) -> torch.Tensor:
        # class_feats: (B, n, d), inventory: (B,1)
        B, n, d = class_feats.shape
        emb = self.class_encoder(class_feats.reshape(B * n, d)).reshape(B, n, -1)
        pooled = emb.mean(dim=1)
        return self.post(torch.cat([inventory, pooled], dim=-1))


class ActorSurfaces(nn.Module):
    def __init__(self, z_dim: int, n: int, B0: int, B_max: list[int]):
        super().__init__()
        self.n = n
        self.B0 = float(B0)
        self.register_buffer("B_max", torch.tensor(B_max, dtype=torch.float32))
        self.order_head = nn.Linear(z_dim, 1)
        self.accept_head = nn.Linear(z_dim, n)
        self.fulfill_head = nn.Linear(z_dim, n)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        b0 = torch.sigmoid(self.order_head(z)).squeeze(-1) * self.B0
        bO = torch.sigmoid(self.accept_head(z)) * self.B_max
        bS = torch.sigmoid(self.fulfill_head(z)) * self.B0
        return {"b0": b0, "bO": bO, "bS": bS}


class CriticV(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class SAACActorNet(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, 3))

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.net(z)
        return {"order": torch.sigmoid(x[:, 0]), "accept": torch.sigmoid(x[:, 1]), "fulfill": torch.sigmoid(x[:, 2])}

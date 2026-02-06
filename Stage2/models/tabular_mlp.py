from __future__ import annotations

import torch.nn as nn


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 128]

        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

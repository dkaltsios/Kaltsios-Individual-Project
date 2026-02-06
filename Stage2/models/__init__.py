"""
Model definitions for Stage2.
"""

from __future__ import annotations

from Stage2.models.tabular_mlp import TabularMLP


def build_model(name: str, input_dim: int, **kwargs):
    if name == "tabular_mlp":
        return TabularMLP(input_dim=input_dim, **kwargs)
    raise ValueError(f"Unknown model name: {name}")

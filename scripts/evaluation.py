"""Evaluation utilities for the genotype-alone effect model.

Mirrors the accuracy-vs-interpretability comparison pattern from
nb06_mlp_variant_comparison.ipynb (train/test Pearson r side by side, plus
a sparsity metric analogous to that notebook's effective-genes-per-protein).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr

from .genotype_models import GenotypeMLP


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float('nan')
    return float(pearsonr(y_true, y_pred)[0])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {'pearson_r': pearson_r(y_true, y_pred), 'rmse': rmse(y_true, y_pred)}


def effective_markers(model: GenotypeMLP, layer_idx: int = 0, threshold: float = 1e-3) -> int:
    """Counts input markers with non-negligible influence through the given
    hidden layer -- the per-marker L1 norm of that layer's weight column,
    thresholded. Only meaningful for layer_idx=0 (the input-facing layer),
    since sparsity in a deeper layer reflects hidden-unit usage, not marker
    selection.
    """
    linear_mods = model.linear_modules()
    with torch.no_grad():
        weight = linear_mods[layer_idx].weight.detach().cpu().numpy()  # (hidden_dim, input_dim)
    per_input_norm = np.abs(weight).sum(axis=0)
    return int((per_input_norm > threshold).sum())

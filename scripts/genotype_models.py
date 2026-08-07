"""Model definitions for the genotype-alone effect model.

GBLUP baseline: VanRaden-kinship kernel ridge regression.
Deep models: MLP-1 (one hidden layer, L1 sparsity on the input layer -- the
"lasso" variant) and MLP-2 (two hidden layers, L1 sparsity on both hidden
layers). LeakyReLU activation throughout, per project convention.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

LEAKY_RELU_SLOPE = 0.01


def compute_vanraden_kinship(dosage: np.ndarray) -> np.ndarray:
    """VanRaden (2008) genomic relationship matrix from a {0, 0.5, 1} dosage matrix,
    using allele frequencies estimated from this same matrix. Convenience
    wrapper for quick/standalone use; the CV pipeline uses
    `vanraden_allele_freq` / `vanraden_kernel` below instead, so that
    validation-fold hybrids are projected using train-fold-only allele
    frequencies rather than allele frequencies that include the val data.
    """
    allele_freq = vanraden_allele_freq(dosage)
    denom = vanraden_denominator(allele_freq)
    return vanraden_kernel(dosage, dosage, allele_freq, denom)


def vanraden_allele_freq(dosage_reference: np.ndarray) -> np.ndarray:
    """Per-marker allele frequency estimated from a reference (training) dosage matrix."""
    return dosage_reference.mean(axis=0)


def vanraden_denominator(allele_freq: np.ndarray) -> float:
    denom = 2.0 * np.sum(allele_freq * (1.0 - allele_freq))
    if denom <= 0:
        raise ValueError("Kinship denominator is zero or negative -- check for "
                          "zero-variance markers before computing kinship.")
    return float(denom)


def vanraden_kernel(dosage_a: np.ndarray, dosage_b: np.ndarray,
                     allele_freq: np.ndarray, denom: float) -> np.ndarray:
    """VanRaden kernel between two dosage matrices, centered by a shared
    reference allele frequency (fold-safe: allele_freq/denom should come
    from the training fold only, via vanraden_allele_freq/vanraden_denominator,
    even when dosage_b is the validation or test set).
    """
    Z_a = dosage_a - allele_freq[np.newaxis, :]
    Z_b = dosage_b - allele_freq[np.newaxis, :]
    return (Z_a @ Z_b.T) / denom


class GenotypeMLP(nn.Module):
    """Flat-vector MLP for genotype-alone prediction.

    hidden_dims of length 1 -> MLP-1 (one hidden layer). Length 2 -> MLP-2
    (two hidden layers). LeakyReLU activation after every hidden layer.
    l1_layer_indices marks which `hidden_layers[i]` Linear modules get L1
    sparsity applied (by the training loop, via `l1_penalty` below) --
    MLP-1 applies it to layer 0 (the "lasso" variant), MLP-2 to both.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        if not (1 <= len(hidden_dims) <= 2):
            raise ValueError(f"hidden_dims must have length 1 or 2, got {hidden_dims}")

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE))
            prev_dim = h
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.hidden_layers(x)).squeeze(-1)

    def linear_modules(self) -> list[nn.Linear]:
        """Returns the Linear sub-modules in order (excludes activations and output layer)."""
        return [m for m in self.hidden_layers if isinstance(m, nn.Linear)]


def build_variant(name: str, input_dim: int, hidden_dims_mlp1: list[int],
                   hidden_dims_mlp2: list[int]) -> tuple[nn.Module, list[int]]:
    """Builds a named genotype-model variant.

    Returns (model, layer_indices) where layer_indices are the indices into
    `model.linear_modules()` that should receive a sparsity/shrinkage
    penalty (empty list means no penalty).

    Variants: 'mlp1_lasso' (one hidden layer, group-lasso on the input
    layer), 'mlp2_sparse' (two hidden layers, group-lasso on both),
    'mlp2_l2' (two hidden layers, standard L2 on both -- same architecture
    as mlp2_sparse so the regularization TYPE is the only thing that
    differs), 'mlp2_no_reg' (two hidden layers, no penalty at all).
    """
    if name == 'mlp1_lasso':
        return GenotypeMLP(input_dim, hidden_dims_mlp1), [0]
    elif name == 'mlp2_sparse':
        return GenotypeMLP(input_dim, hidden_dims_mlp2), [0, 1]
    elif name == 'mlp2_l2':
        return GenotypeMLP(input_dim, hidden_dims_mlp2), [0, 1]
    elif name == 'mlp2_no_reg':
        return GenotypeMLP(input_dim, hidden_dims_mlp2), []
    raise ValueError(f"Unknown variant: {name!r}")


def l2_penalty(model: GenotypeMLP, layer_indices: list[int], lam: float) -> torch.Tensor:
    """Standard (smooth) L2 penalty: lam * sum of squared weights over the
    specified layers. Unlike group lasso, L2 doesn't need a proximal step --
    it's a smooth penalty that gradient descent (including Adam) optimizes
    directly and correctly; it shrinks weights toward zero without any
    intent to produce exact zeros, so there's no Adam-vs-penalty conflict
    the way there was with group lasso.
    """
    linear_mods = model.linear_modules()
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for idx in layer_indices:
        penalty = penalty + (linear_mods[idx].weight ** 2).sum()
    return lam * penalty


def apply_group_lasso_proximal(model: GenotypeMLP, l1_layer_indices: list[int], lam: float) -> None:
    """In-place proximal (block soft-thresholding) update for group lasso.
    Call ONCE PER EPOCH (not per batch) -- see rationale below.

    Gradient-based penalties (adding lam * sum(column L2 norms) to the loss
    and backpropagating) don't reliably zero anything out under Adam: Adam's
    per-parameter adaptive scaling actively resists pure gradient-based
    shrinkage-to-zero, so a column's weights get smaller on average without
    the column ever collapsing to exactly zero -- confirmed empirically
    (even lambda=1.0 as a gradient penalty badly hurt accuracy without
    producing a single zero column). The fix is decoupling the smooth loss
    (handled by Adam as usual) from the non-smooth group-lasso penalty
    (handled by an explicit proximal step): shrink each column's L2 norm by
    lam, zeroing it outright if that would go negative. This is the
    textbook proximal-gradient (ISTA-style) approach to group lasso.

    `lam` is the shrinkage applied directly (not scaled by a learning rate)
    and this function is called once per epoch, not once per batch -- an
    earlier version scaled by Adam's lr and ran every batch, which made the
    total shrinkage over training depend on batch count (dataset
    size / batch size) as well as lambda, and was calibrated far too small
    for typical column-norm scales (~0.1-1.0): lr=1e-3 * lam=1.0 per batch
    would need thousands of epochs to zero a column of norm ~0.5. Treating
    lam as "column-norm units removed per epoch" makes the grid directly
    interpretable and independent of batch size or optimizer step size.
    """
    if lam <= 0:
        return
    linear_mods = model.linear_modules()
    with torch.no_grad():
        for idx in l1_layer_indices:
            W = linear_mods[idx].weight  # (out_dim, in_dim)
            col_norms = W.norm(p=2, dim=0, keepdim=True)  # (1, in_dim)
            shrunk = torch.clamp(col_norms - lam, min=0.0)
            scale = torch.where(col_norms > 0, shrunk / col_norms, torch.zeros_like(col_norms))
            W.mul_(scale)  # broadcasts the per-column scale over all output rows

"""Building blocks for the effect-relationship / fusion layer (Phase 3):
small scalar-relationship models between the genotype-alone vote
(`genetic_value`, GBLUP's per-hybrid prediction from `03`) and the
environment-alone vote (`environment_value`, env_mlp_l2's per-environment
prediction from `04`), plus their relationship to observed phenotype.

Deliberately tiny: every model here is scalar-in/scalar-out (or 2-in for
the fusion regression, handled directly with sklearn in the notebook, not
here). The genotype and environment MLPs needed 24-256 hidden units because
they had 39-2,425 real input features; these relationship models have
exactly 1 input feature, so an 8-unit single hidden layer is already
generous capacity, not a compromise -- see the notebook markdown for the
reasoning against reaching for anything bigger here.
"""
from __future__ import annotations

import numpy as np


def scalar_feature_mean_std(x_train: np.ndarray) -> tuple[float, float]:
    """Train-only mean/std for standardizing a single scalar input feature
    (genetic_value or environment_value) before it goes into ScalarMLP.
    Not needed for the linear-regression fusion step -- that's fit directly
    on raw (yield-scale) values so its coefficients stay interpretable in
    yield units.
    """
    mean = float(x_train.mean())
    std = float(x_train.std())
    return mean, (std if std > 0 else 1.0)


class ScalarMLP:
    """Thin factory wrapper, same pattern as environment_models.EnvMLP --
    keeps the torch dependency contained to this module. Single hidden
    layer, deliberately narrow (default 8 units): the input is always a
    single standardized scalar, so this has far more capacity than a 1-in
    relationship needs already; the point is to catch visible curvature a
    straight line would miss, not to maximize fit.
    """

    def __new__(cls, n_in: int = 1, hidden_dim: int = 8, n_out: int = 1):
        import torch.nn as nn

        class _ScalarMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_in, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_out),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        return _ScalarMLP()


def scalar_l2_penalty(model, lam: float):
    """L2 penalty on the first (input) linear layer's weights only --
    same layer-scoping convention as environment_models.env_l2_penalty.
    Shape-agnostic (sums over all input columns), so this also covers the
    2-input interaction MLP below without change.
    """
    return lam * (model.net[0].weight ** 2).sum()


def vector_feature_mean_std(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise train-only mean/std, for standardizing `genetic_value`
    and `environment_value` together before either an explicit interaction
    term or GxEInteractionMLP. Generalizes scalar_feature_mean_std (which
    only ever handled one column) so both inputs share a single call rather
    than two independent scalar ones that could silently drift out of sync
    (e.g. one fit on train, one accidentally refit on train+test).
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return mean, std


def GxEInteractionMLP(hidden_dim: int = 8):
    """Two-input MLP (standardized genetic_value, standardized
    environment_value) -> pheno -- the Part B "full nonlinear surface"
    model. Identical architecture to ScalarMLP, just called under its own
    name here: ScalarMLP's docstring says the input is "always a single
    standardized scalar", which stops being literally true the moment
    n_in=2, so this thin wrapper keeps the name honest at the call site
    instead of leaving that docstring silently wrong for this use.
    """
    return ScalarMLP(n_in=2, hidden_dim=hidden_dim, n_out=1)

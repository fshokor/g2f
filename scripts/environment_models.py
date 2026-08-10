"""Environment-alone effect model building blocks: engineered weather
features, soil/location feature assembly, an environment relationship
kernel (EBLUP) that is the structural analog of genotype_models.py's
VanRaden kinship kernel (GBLUP), and a small MLP for the nonlinearity
check.

Weather/soil/location feature engineering here is ported (not
re-derived) from 01_effect_representations.ipynb Sections 2b/2c -- same
column-pattern matching, same GDD/heat-stress formulas, same soil
missingness threshold -- so 04_environment_model.ipynb builds on exactly
what was already explored and locked there, rather than a fresh
reimplementation that could quietly diverge.

Per-fold leakage note: `engineer_env_weather_features` and
`build_soil_features` are deterministic, Env-local aggregations (each
Env's engineered row depends only on that Env's own daily weather /
soil records) -- safe to compute once globally. What is NOT safe to
compute globally is the soil per-column median used to impute missing
soil records, or the feature standardization (mean/std) used by the
kernel and the MLP -- those are statistics over a set of environments
and must be refit inside each CV fold using only that fold's training
years, exactly like genotype_models.py's marker median/standardization.
That refitting happens in the model notebook's `prepare_fold`, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Weather feature engineering (ported verbatim from
# 01_effect_representations.ipynb, Section 2b)
# ---------------------------------------------------------------------------

WEATHER_PATTERNS = {
    'tmax': ['t2m_max', 'tmax'],
    'tmin': ['t2m_min', 'tmin'],
    'tmean': ['t2m_mean', 't2m'],
    'precip': ['prectot', 'precip'],
    'solar': ['allsky', 'srad', 'solar'],
    'humidity': ['rh2m', 'humidity'],
    'windspeed': ['ws2m', 'wind'],
    'soil_moisture': ['gwet', 'soil_moist', 'soil_wet'],
}


def match_column(columns: list[str], patterns: list[str]) -> str | None:
    """Exact-match first, substring fallback. Avoids e.g. pattern 't2m'
    grabbing 'T2M_MAX' when an exact 'T2M' column exists -- NASA POWER-style
    names often share a prefix (T2M/T2M_MAX/T2M_MIN).
    """
    lowered = {c: c.lower() for c in columns}
    for pat in patterns:
        for col, low in lowered.items():
            if low == pat:
                return col
    for pat in patterns:
        for col, low in lowered.items():
            if pat in low:
                return col
    return None


def match_weather_columns(weather_cols: list[str]) -> dict[str, str | None]:
    return {key: match_column(weather_cols, patterns) for key, patterns in WEATHER_PATTERNS.items()}


def engineer_env_weather_features(df: pd.DataFrame, matched: dict[str, str | None],
                                   gdd_base_c: float = 10.0, gdd_cap_c: float = 30.0,
                                   heat_stress_threshold_c: float = 35.0) -> pd.DataFrame:
    """Fixed-length engineered weather feature vector per Env. Locked in
    01_effect_representations.ipynb Section 2b:
    GDD_day = clip((Tmax + Tmin) / 2, gdd_base_c, gdd_cap_c) - gdd_base_c
    heat_stress_days = count of days where Tmax exceeds heat_stress_threshold_c
    Any feature whose source column wasn't matched is omitted rather than
    silently filled with zeros.
    """
    rows = []
    for env, g in df.groupby('Env'):
        row = {'Env': env, 'season_length_days': len(g)}

        tmax_col, tmin_col = matched.get('tmax'), matched.get('tmin')
        if tmax_col and tmin_col:
            tmax, tmin = g[tmax_col], g[tmin_col]
            tmean_daily = (tmax + tmin) / 2
            gdd = tmean_daily.clip(lower=gdd_base_c, upper=gdd_cap_c) - gdd_base_c
            row['gdd_sum'] = gdd.sum()
            row['tmax_mean'] = tmax.mean()
            row['tmax_max'] = tmax.max()
            row['tmin_mean'] = tmin.mean()
            row['heat_stress_days'] = (tmax > heat_stress_threshold_c).sum()

        precip_col = matched.get('precip')
        if precip_col:
            row['precip_sum'] = g[precip_col].sum()
            row['precip_max_daily'] = g[precip_col].max()

        solar_col = matched.get('solar')
        if solar_col:
            row['solar_mean'] = g[solar_col].mean()

        humidity_col = matched.get('humidity')
        if humidity_col:
            row['humidity_mean'] = g[humidity_col].mean()

        wind_col = matched.get('windspeed')
        if wind_col:
            row['windspeed_mean'] = g[wind_col].mean()

        soil_moisture_col = matched.get('soil_moisture')
        if soil_moisture_col:
            row['weather_soil_moisture_mean'] = g[soil_moisture_col].mean()

        rows.append(row)

    return pd.DataFrame(rows).set_index('Env')


# ---------------------------------------------------------------------------
# Soil features (ported verbatim from Section 2c). NOTE: returns the RAW
# per-Env soil matrix with NaNs still present for envs with no soil sample
# -- median imputation is fold-safe and happens in the notebook, not here.
# ---------------------------------------------------------------------------

def build_soil_features(df: pd.DataFrame, env_col: str = 'Env',
                         max_missing_frac: float = 0.5) -> pd.DataFrame:
    """Per-Env soil feature matrix: numeric columns only, duplicate Env rows
    averaged, columns >max_missing_frac missing across envs dropped (e.g.
    trace micronutrients ~98% missing per the Phase 1 audit).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != env_col]
    per_env = df.groupby(env_col)[numeric_cols].mean()
    missing_frac = per_env.isna().mean()
    kept_cols = missing_frac[missing_frac <= max_missing_frac].index.tolist()
    return per_env[kept_cols]


def match_meta_column(columns: list[str], patterns: list[str]) -> str | None:
    return match_column(columns, patterns)


# ---------------------------------------------------------------------------
# Environment relationship kernel -- structural analog to
# genotype_models.vanraden_kernel for the EBLUP variant. Linear kernel on
# standardized covariates: G = Z_a @ Z_b.T / n_features. There's no
# population-genetics allele-frequency structure to exploit here the way
# VanRaden centering does for markers, so ordinary train-fold standardization
# is the direct substitute -- the /n_features scaling keeps kernel magnitude
# comparable across feature-set sizes, mirroring VanRaden's denominator role.
# ---------------------------------------------------------------------------

def env_feature_mean_std(X_train_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-fold-only mean/std for standardizing environment covariates.
    Zero-variance columns get std=1 (constant columns become all-zero after
    centering rather than dividing by zero)."""
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def env_kernel(X_a: np.ndarray, X_b: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Linear kernel on standardized environment covariates. mean/std must
    come from env_feature_mean_std(X_train_raw) -- train-fold-only,
    reused identically for G_train and G_val, exactly like vanraden_kernel's
    allele_freq/denom being fit on the training fold alone.
    """
    Za = (X_a - mean) / std
    Zb = (X_b - mean) / std
    return (Za @ Zb.T) / X_a.shape[1]


# ---------------------------------------------------------------------------
# Small MLP for the environment-alone nonlinearity check. Deliberately much
# smaller than the genotype MLPs (256/64 units) -- ~269 training environments
# vs ~4,900+ hybrids means a wide two-hidden-layer network would be data-
# starved regardless of regularization. Single hidden layer, L2 only: with
# ~30-40 already-curated covariates there's no group-lasso sparsity story
# analogous to the 2,425-marker case, so this variant tests only "does any
# nonlinearity in weather/soil -> yield earn its keep over the linear kernel."
# ---------------------------------------------------------------------------

class EnvMLP:
    """Thin factory wrapper so the notebook doesn't need a bare `torch`
    import just to build the model -- keeps the torch dependency contained
    to this module, matching genotype_models.py's pattern.
    """

    def __new__(cls, n_features: int, hidden_dim: int = 24):
        import torch.nn as nn

        class _EnvMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        return _EnvMLP()


def env_l2_penalty(model, lam: float):
    """L2 penalty on the first (input) linear layer's weights only --
    matches genotype_models.l2_penalty's layer-scoping convention. Returns
    a scalar tensor to be added to the training loss by fit_mlp_adam.
    """
    return lam * (model.net[0].weight ** 2).sum()

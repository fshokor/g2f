# Notebooks

Sequential, numbered. Each is tested via `jupyter nbconvert --execute` on
synthetic data before being considered done.

| # | Notebook | Purpose |
|---|---|---|
| 00 | `00_data_setup_and_exploration` | Load and sanity-check genotype/phenotype/weather/soil data against the README, Understand structure, coverage, missingness |
| 02 | `02_effect_representations` | Build genotype and environment representations for the effect-alone models |
| 03 | `03_genotype_model` | Genotype-alone predictor + baseline |
| 04 | `04_environment_model` | Environment-alone predictor + baseline |
| 05 | `05_diagnostic_relationships` | Orthogonalize votes, study agreement/divergence |
| 06 | `06_fusion_model` | Fusion architecture informed by diagnostics |
| 07 | `07_evaluation` | Held-out evaluation vs. baselines |

Notebooks are not created until the analytical approach for that step has
been discussed and confirmed (see repo root README, "Conventions").

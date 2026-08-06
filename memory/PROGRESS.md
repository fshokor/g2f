# Progress

## Status

- [x] Project scoped, dataset selected (G2F)
- [x] Repo scaffolded
- [x] 2024/2025 competition dataset downloaded (train 2014-2023 + test 2024)
      and audited via `00_data_setup_and_exploration.ipynb`
- [x] Genotype representation decided -- see SESSION_MEMORY.md
- [x] Environment representation decided -- see SESSION_MEMORY.md
- [x] Train/test split scheme decided -- see SESSION_MEMORY.md
- [x] Genotype and environment representations built and explored via
      `01_effect_representations.ipynb` (real data run): per-hybrid and
      per-environment marginal targets, engineered weather features, soil
      (filtered + imputed + has_soil_data flag), latitude/longitude. City
      and EC excluded by design. TXH2_2015-2017 excluded from
      environment-alone modeling (no weather record).
- [ ] Genotype-alone baseline built
- [ ] Environment-alone baseline built
- [ ] Diagnostic/orthogonalization layer built
- [ ] Fusion model built and evaluated on held-out data

## Notebooks

- `00_data_setup_and_exploration.ipynb` -- Phase 1 data audit (integrity
  checks, profiling, env join-key audit)
- `01_effect_representations.ipynb` -- genotype/environment representation
  building + exploratory analysis of which environment features carry
  yield signal (Section 3: correlations, City/Year boxplots, joint PCA,
  K-means clustering). Run against real data; outputs saved to
  `data/processed/`.
- `03_genotype_model.ipynb` -- not yet started (next up)
- `04_environment_model.ipynb` -- not yet started

## Key real-data findings worth remembering

- Genotype reliability (n_envs_tested per hybrid) is heavily skewed:
  median 17, mean 21.6, max 259 -- loss-weighting is a real requirement,
  not optional polish.
- Weather correlations with env_mean_yield are agronomically sensible:
  heat/GDD-related features all negatively correlated with yield (~0.3-0.4
  magnitude), as expected.
- Latitude correlates with env_mean_yield almost as strongly as weather
  (+0.37) and isn't fully redundant with the engineered weather indices.
- Soil correlations are real but modest (~0.2-0.25); `has_soil_data` itself
  correlates with yield (+0.18), a possible site-quality confound to keep
  in mind for Phase 3.
- City shows the largest raw yield spread of any environment attribute
  examined (~2x between lowest- and highest-median cities) but was
  excluded from the feature vector as high-cardinality and
  downstream-of-other-features by design decision.

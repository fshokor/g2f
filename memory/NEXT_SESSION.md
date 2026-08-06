# Next session

## Immediate next steps

1. Design and confirm the genotype-alone model architecture before writing
   code:
   - Deep model: MLP on the raw dosage vector (2425 -> ... -> 1), dropout +
     weight decay given input dim is large relative to unique hybrids
   - Baseline: GBLUP via VanRaden kinship matrix, fit as kernel ridge
     regression
   - Loss-weighting strategy for the n_envs_tested reliability skew
     (median 17, max 259) -- explicit weighting vs. relying on GBLUP's
     shrinkage vs. both
   - Fold-safe target recomputation inside the leave-one-year-out CV loop
     (per-hybrid mean must use only that fold's training years)

2. Design and confirm the environment-alone model architecture:
   - Same flat-vector MLP + linear (ridge) baseline pairing, using
     `environment_combined_features.csv` (weather + soil + lat/long) built
     in `01_effect_representations.ipynb`
   - Same fold-safe target recomputation requirement (per-environment mean)
   - Exclude TXH2_2015/2016/2017 from training/eval (no weather record) --
     use `env_targets_modelable`, not the full `env_targets`

3. Only then: create `notebooks/03_genotype_model.ipynb` and
   `notebooks/04_environment_model.ipynb` (Phase 2 baselines)

## Deferred / documented follow-ups (not blocking Phase 2)

- EC (673 crop-model covariates): candidate richer environment
  representation once the framework is validated on the simpler version
  (e.g. PCA-reduced EC block)
- City as a learned embedding for the deep environment model, if
  weather+soil+lat/long don't capture enough of the City-level yield gap
  observed in exploration
- Parent inbred genotyping cohort heterogeneity (GBS/WGS/Exome/Assembly)
  as a potential confound -- not yet investigated

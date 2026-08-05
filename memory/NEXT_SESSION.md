# Next session

## Immediate next steps

1. Pin down the test_ec/test_observed environment count mismatch (23 vs.
   22) before writing any evaluation code -- see PROGRESS.md / SESSION_MEMORY.md
   "Known risks"
2. Design and confirm the genotype-alone model architecture (deep model on
   raw dosage matrix) and the GBLUP/kinship baseline before writing code
3. Design and confirm the environment-alone model: weather-primary feature
   set, plus the imputation/dropout strategy for soil and EC per environment
4. Only then: create `notebooks/01_genotype_model.ipynb` and
   `notebooks/02_environment_model.ipynb` (Phase 2 baselines)

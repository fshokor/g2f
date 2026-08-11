# Progress -- G2F Effect-Decomposition Framework

## Phase 1 -- Data audit
**Status: Done.** `00_data_setup_and_exploration.ipynb`.

## Phase 2 -- Effect representations
**Status: Done.** `01_effect_representations.ipynb`. Environment feature
vector locked (weather + soil + lat/long, city excluded, EC deferred).

## Phase 3a -- Genotype-alone model
**Status: Done.** `03_genotype_model.ipynb`. GBLUP selected as final
(alpha=0.1). Test pearson_r=0.229 (hybrid-level, true 2024 holdout).

## Phase 3b -- Environment-alone model
**Status: Done.** `04_environment_model.ipynb`. env_mlp_l2 selected as final
(lambda=1.0, hand-picked). Test pearson_r=0.470 (env-level, true 2024
holdout).

## Phase 4 -- Effect relationships / diagnostic layer
**Status: In progress -- first look complete, in-sample only.**
`05_effect_relationships.ipynb`.

Done:
- [x] Hybrid x Env cell-mean join table (genetic_value + environment_value + pheno)
- [x] Confound check (genetic_value vs environment_value): r=0.230, linear
- [x] Cross-prediction MLP pair (genetic_value <-> environment_value)
- [x] h() functional-form check (environment_value -> pheno): no curvature found
- [x] Linear fusion baseline + h-spliced variant, both evaluated in-sample and on true 2024 holdout

Not done:
- [ ] Rebuild `genetic_value`/`environment_value` as leave-one-year-out
      out-of-fold predictions (currently in-sample final-model refits --
      the single biggest caveat on every number above)
- [ ] Re-run confound check / h() / fusion with OOF values
- [ ] Resolve whether genotype's weak cell-level test showing (r=0.116) is
      real or an in-sample-overfitting artifact
- [ ] Decide whether to keep the h-spliced fusion path (added ~nothing
      in-sample) or drop it in favor of plain linear fusion

## Phase 5 -- Fusion model (final)
**Status: Not started.** Blocked on Phase 4's OOF rebuild -- current linear
fusion is a first-look baseline, not a validated final model. Current
signal: naive linear fusion is not yet clearly beating environment_value
alone on the true holdout (better pearson_r, worse RMSE) -- needs
investigation before any fusion architecture is treated as "the" model.

## Standing open items (not phase-blocking, not yet addressed)
- [ ] Parent inbred genotyping cohort heterogeneity (GBS/WGS/Exome/Assembly)
      as a potential genotype-model confound
- [ ] Formal write-up of the 23-vs-22 test environment count mismatch
      resolution (currently handled ad hoc, generically, in `04`/`05`)

## File inventory
- `notebooks/00_data_setup_and_exploration.ipynb` -- done
- `notebooks/01_effect_representations.ipynb` -- done
- `notebooks/03_genotype_model.ipynb` -- done
- `notebooks/04_environment_model.ipynb` -- done
- `notebooks/05_effect_relationships.ipynb` -- in progress (first look done, OOF rebuild pending)
- `scripts/genotype_models.py` -- VanRaden kinship kernel, genotype MLP variants
- `scripts/environment_models.py` -- weather/soil feature engineering, env kernel, env MLP
- `scripts/relationship_models.py` -- small scalar-relationship MLP (Phase 4)
- `scripts/training.py` -- effect-agnostic: `reliability_weights`, `make_loader`, `fit_mlp`, `fit_mlp_adam`, `fit_gblup`, `GBLUPModel`
- `scripts/evaluation.py` -- effect-agnostic: `evaluate_predictions`, `effective_markers`
- `results/genotype_model/` -- `03`'s CV fold results, comparison tables
- `results/environment_model/` -- `04`'s CV fold results, comparison tables
- `results/effect_relationships/` -- `05`'s cell-mean tables (train + test) and fusion comparison tables

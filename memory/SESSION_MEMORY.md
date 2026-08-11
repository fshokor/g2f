# Session Memory -- G2F Effect-Decomposition Framework

Narrative record of decisions and why they were made. See `PROGRESS.md` for
status and `NEXT_SESSION.md` for what to pick up next.

## Framework recap

Hypothesis: training separate genotype-alone and environment-alone models,
studying the relationship between their per-effect "votes" via a
diagnostic/orthogonalization layer, and only then fusing them, produces more
principled and interpretable maize grain yield predictions than naive data
integration. Motivated directly by a prior multiomics project (RNA-protein
fusion) where naive fusion of correlated layers consistently failed to beat
the stronger single modality, and apparent "coupling" often turned out to be
a confound (cell-type composition, tissue-of-origin) rather than genuine
complementary signal. Dataset: G2F 2024/2025 GxE Competition (CyVerse),
2014-2023 training environments, 2024 held-out test set.

## Phase 1 -- Data audit (`00_data_setup_and_exploration.ipynb`, finalized)

Genotype matrix: 5,899 hybrids x 2,425 markers, raw `{0, 0.5, 1}` dosage
format, `NA` for missing calls, not full rank, minimally filtered (README
warns of known errors). Weather (NASA POWER, daily): complete for core
variables, cleanest table in the dataset. Soil: absent for ~30% of
environments, and even where present, many chemistry fields (esp. trace
micronutrients) are ~90-98% missing. Metadata "Issue/comment" free-text
fields are heavily missing (>90%) but core identifying fields (Env, Year)
are complete. Target `Yield_Mg_ha` is ~5.2% missing in training (partly
intentional QC nulling per the README), 100% missing in the submission
template (to be predicted), fully present in the true 2024 holdout.

## Phase 2 -- Effect representations (`01_effect_representations.ipynb`, finalized)

**Genotype representation:** raw `{0, 0.5, 1}` dosage, no standardization at
this stage (standardization decision revisited per-model in `03`).

**Environment representation, locked:** engineered weather indices (GDD sum,
heat-stress days, temperature/precip/solar/humidity/wind/soil-moisture
summaries -- column-pattern-matched, not hardcoded to exact NASA POWER
column names) + soil (median-imputed, `has_soil_data` flag) + latitude/
longitude. City labels excluded (identified as a downstream location proxy
that could leak). EC (673 crop-model covariates) deliberately deferred --
overlaps substantially with the hand-engineered weather features and is a
dimensionality problem at ~270 environments.

**TXH2_2015/2016/2017 confirmed as a genuine three-year weather gap** --
1,223 trait rows requiring explicit exclusion from environment-alone
modeling. Detected generically (target-but-no-weather set difference) in
`04`/`05` rather than hardcoded, so it stays correct if a future data
release changes the gap. TXH1 treatment-suffixed Env names confirmed
consistent across files (no join-key surprises there).

## Phase 3a -- Genotype-alone model (`03_genotype_model.ipynb`, finalized)

Target: per-hybrid marginal mean yield, reliability-weighted by (capped)
`n_envs_tested`. Five variants compared via leave-one-year-out CV (5
evenly-spaced years): GBLUP (VanRaden kinship kernel ridge), MLP-1
group-lasso, MLP-2 group-lasso, MLP-2 L2, MLP-2 no-reg. **GBLUP selected as
final** (alpha=0.1) despite `mlp2_sparse` edging it out on median CV val_r --
GBLUP generalized best to the true 2024 holdout, which is what actually
matters. Final numbers: train pearson_r=0.848, **test pearson_r=0.229**
(hybrid-level, true 2024 holdout).

Key learnings: group lasso (column L2-norm proximal soft-thresholding, not
elementwise L1) needed for real sparsity; SGD+momentum+clipping required for
group-lasso variants (Adam's adaptive scaling fights proximal steps);
weighted MSE must be `sum(w*e^2)/sum(w)` not `.mean()`; one-standard-error
rule favors sparsity over pure accuracy in lambda selection; GBLUP needs an
explicit mean-offset wrapper since the kinship kernel is inherently
mean-centered.

## Phase 3b -- Environment-alone model (`04_environment_model.ipynb`, finalized)

Target: per-environment marginal mean yield, reliability-weighted by
`n_hybrids_tested`. Two variants (deliberately not `03`'s 5-variant
structure -- ~35-40 curated features and ~270 environments is a very
different scale than 2,425 markers/~4,900 hybrids, so group-lasso sparsity
and wide MLPs aren't well-motivated here): **EBLUP** (linear kernel on
standardized covariates, direct structural analog of GBLUP) vs
**env_mlp_l2** (single 24-unit hidden layer, L2 only). CV: full 10-fold
leave-one-year-out (cheap here, so no subsampling needed).

**env_mlp_l2 (lambda=1.0) selected as final** -- hand-picked over the
CV-auto-selected lambda=0.1 (median val_r 0.487 vs 0.491, within noise of
each other in CV; 1.0 won clearly on the true holdout). Final numbers:
train pearson_r=0.630, **test pearson_r=0.470** (env-level, true 2024
holdout) vs EBLUP's test pearson_r=0.255 -- confirms real, meaningful
nonlinearity in weather/soil -> yield.

Bug found+fixed during development: a fold's train-only soil median (or the
weather-to-meta Env join) can come out `NaN` -- either a sparsely-measured
soil field whose few real values all fall inside the held-out year, or an
Env present in the weather file but absent from `meta_df` (join-key
mismatches are an explicitly known risk per the project brief). Fixed with
`impute_remaining_nan` (global-median fallback, prints what it touched) plus
reindexing every feature piece to the weather-covered universe before
`pd.concat` (was silently pulling gap-year environments into the table via
outer-join and inflating the NaN diagnostic).

## Phase 4 -- Effect relationships (`05_effect_relationships.ipynb`, first look, in progress)

Strategy (as discussed): `pheno = a*genetic_value + b*environment_value +
bias` as a fusion baseline, where `genetic_value` = GBLUP's prediction and
`environment_value` = env_mlp_l2's prediction; a confound-audit pair of small
MLPs (`genetic_value <-> environment_value`, single hidden layer, ~8 units,
since both are already scalars -- interpretability matters more than
capacity for a diagnostic layer); and a third model `h(environment_value) ->
pheno` to find environment_value's functional-form relationship to true
phenotype, then splice `h()` into the fusion formula in place of the plain
linear term.

**Granularity: Hybrid x Env cell mean** (not raw plot-level) -- matches
classical G x E specification, avoids replicate-count pseudoreplication.

**Currently in-sample, no CV** (explicit choice, "first look" scope) --
`genetic_value`/`environment_value` are each model's final, fully-fit
prediction on its own training data, not out-of-fold. This is the single
biggest caveat on everything below.

### Results (in-sample train + true 2024 holdout test)

- **Confound check**: Pearson r(genetic_value, environment_value) = 0.230
  across 106,037 observed Hybrid x Env cells -- a real but modest,
  essentially **linear** relationship (cross-prediction MLPs barely beat
  plain correlation: 0.236 and 0.231 vs 0.230 baseline). Consistent with
  better-genetics hybrids being systematically promoted to
  higher-environment_value trial sites -- a real, mild design confound to
  keep in mind, not something to over-interpret as biological synergy.
- **h() functional-form check**: MLP r=0.515 vs linear baseline r=0.514 --
  no real curvature. env_mlp_l2's output is already well-calibrated in
  *shape* against true phenotype. Splicing `h()` into fusion added nothing
  measurable (baseline vs h-spliced fusion pearson_r 0.5677 vs 0.5678
  in-sample, functionally identical) -- the h-spliced path is likely
  droppable going forward in favor of the simpler plain-linear fusion.
- **True 2024 holdout, Hybrid x Env cell level:**

  | formula | pearson_r | rmse |
  |---|---|---|
  | genetic_value alone | 0.116 | 3.032 |
  | environment_value alone | 0.261 | **2.949** |
  | baseline fusion (linear) | **0.284** | 3.098 |
  | improved fusion (h-spliced) | 0.265 | 3.167 |

  **Naive linear fusion is not yet clearly beating environment_value alone**
  -- correlation ticks up slightly but RMSE gets worse, meaning fusion's
  cell ranking improved marginally while its predicted magnitudes drifted
  further from true yield (likely `genetic_value`'s large fitted weight
  (a=0.729) injecting miscalibrated signal, given its own test r is only
  0.116). This echoes the same "naive fusion underperforms the stronger
  single effect" failure mode the whole framework was built to diagnose --
  a real result worth taking seriously, not a formality to wave past.

  Also worth noting: both effects' test correlations drop substantially at
  the cell level vs. their own solo tests (genetic_value 0.229 -> 0.116;
  environment_value 0.470 -> 0.261) -- expected, not a bug, since each
  model's solo test benefited from averaging away the other effect's
  variation, while cell-level pheno still has G x E interaction and plot
  noise in it.

Bug found+fixed during development: `env_mlp_l2`'s refit was accidentally
trained with hyperparameters sized for the small scalar relationship models
(`batch_size=256` on a ~269-row dataset, few epochs) -- collapsed each epoch
to ~1 gradient update, producing a near-random `environment_value` (in-sample
train r went *negative*) that silently broke every downstream number in that
run. Fixed by strictly separating `ENV_*` (must match `04` exactly: batch
32, 300 epochs, patience 30, lr 1e-3) from `REL_*` (relationship-model-only)
hyperparameter namespaces, and added `check_refit_matches_source()`, which
warns loudly if any refit's in-sample fit diverges from the source
notebook's own reported number -- a general safeguard against this class of
bug recurring, not just a one-off patch.

### Open question for the next session

Is genotype's weak cell-level showing (test r=0.116) a real finding, or an
artifact of `genetic_value`/`environment_value` still being in-sample
(potentially overfit to 2014-2023, inflating both the confound correlation
and the fusion coefficients beyond what truly generalizes)? Not resolvable
without the out-of-fold rebuild -- see `NEXT_SESSION.md`.

## Standing open items (not yet addressed)

- Parent inbred genotyping cohort heterogeneity (GBS, WGS, Exome, Assembly
  across different year ranges, per `key_inbreds_G2F_2014-2025.txt`) as a
  potential confound for the genotype model -- flagged early, not yet
  investigated.
- Test-side environment count mismatch (submission template: 23
  environments; `7_Testing_Observed_Values.csv`: 22) -- handled ad hoc via
  generic intersection logic in `04` and `05` (prints whatever gets
  dropped), but not yet written up as a resolved/documented decision.

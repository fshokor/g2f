# Session Memory

## Project

G2F Effect-Decomposition Framework: train separate genotype-alone and
environment-alone models, study the relationship between their per-effect
"votes" via a diagnostic/orthogonalization layer (accounting for the risk
that genotype and environment are correlated -- certain hybrids
disproportionately tested in certain environments), then build a fusion
model evaluated on held-out data. Spun out of a prior multiomics project
where naive fusion of correlated layers rarely improved prediction due to
confounds rather than genuine complementary signal.

Dataset: G2F 2024/2025 GxE Prediction Competition (CyVerse Data Commons).
Maize grain yield, 2014-2023 train / 2024 test, genotype (VCF + numerical
dosage matrix), weather (NASA POWER), soil, environmental covariates (EC),
metadata.

## Confirmed decisions

### Genotype representation
Raw dosage matrix (5,899 hybrids x 2,425 markers, {0, 0.5, 1}) as direct
input to the genotype-alone deep model, with a GBLUP/kinship baseline built
alongside it.

**Target definition (confirmed after discussion):** per-hybrid marginal
mean yield across all environments that hybrid was tested in -- i.e. a
single fixed value per hybrid, not a per-plot or per-(Env,Hybrid) target.
This mirrors the classical quantitative-genetics G x E main-effects
decomposition (genotype main effect + environment main effect + G x E
interaction), keeps the genotype-alone and environment-alone models
symmetric, and deliberately does NOT pre-correct for the genotype x
environment confound -- that correction is Phase 3's job (orthogonalization
layer), not something to bake into the target upfront.

**Reliability:** per-hybrid n_envs_tested is highly skewed (real-data
result from `01_effect_representations.ipynb`: median 17, mean 21.6, max
259 -- one hybrid, almost certainly a repeatedly-used local check/reference,
tested in nearly every environment). This confirms loss-weighting (or
relying on GBLUP's kinship-based shrinkage) is a real requirement for the
genotype-alone model, not a nice-to-have. Must be decided explicitly in
`03_genotype_model.ipynb`.

**Fold-safety requirement:** the per-hybrid mean saved by
`01_effect_representations.ipynb` is computed from the full training set
and is exploratory-only. The model notebook must recompute it per CV fold
using only that fold's training-year environments, or held-out-year
information leaks into the "fixed" genotype value being predicted.

### Environment representation
**Weather** (engineered indices, Option B -- confirmed over raw daily
sequences to keep the genotype-alone and environment-alone models
architecturally symmetric, both flat-vector inputs): GDD sum, tmax
mean/max, tmin mean, heat-stress day count, precip sum/max daily, solar
mean, humidity mean, windspeed mean, a weather-derived soil-moisture proxy,
season length. Column names auto-detected by pattern from the real NASA
POWER-style file (`T2M_MAX`, `T2M_MIN`, `T2M`, `PRECTOTCORR`,
`ALLSKY_SFC_SW_DWN`, `RH2M`, `WS2M`, `GWETTOP` confirmed as the real matches
via `01_effect_representations.ipynb`).

**Soil**: included, imputed (median, exploratory/global in the
representation notebook -- must be refit per train fold in the model
notebook), with a `has_soil_data` binary flag. Real-data result: soil
present for 186/272 train envs; after column-level filtering (>50% missing
across envs dropped -- removes trace micronutrients Zn/Fe/Mn/Cu/B and BpH,
consistent with the Phase 1 audit), 24 of 30 numeric soil columns kept.

Individual soil correlations with env_mean_yield are real but modest
(strongest: `E Depth` -0.25, `Organic Matter LOI %` +0.24, `Magnesium`
+0.21) -- weaker than weather's top correlates (~0.3-0.4). **Caveat:**
`has_soil_data` itself correlates with yield (+0.18), suggesting a possible
confound where better-resourced/more-established trial sites both collect
more soil data and yield higher -- soil's correlation may be partly proxying
this rather than being purely causal soil chemistry. Worth remembering
during Phase 3 diagnostics.

**Location (latitude/longitude)**: included. Real column names are
`Weather_Station_Latitude (in decimal numbers NOT DMS)` and the longitude
equivalent, matched by pattern. Latitude correlates with env_mean_yield at
+0.37 -- nearly as strong as top weather features, and appears to carry
information not fully captured by the engineered weather indices (northern
vs. southern Corn Belt climatic gradient).

**City**: explicitly EXCLUDED from the environment feature vector. City
shows the largest raw yield spread of anything examined (city medians
range roughly 5-6 Mg/ha at the low end to 13-14 Mg/ha at the high end), but
per the user's judgment, City is treated as *defined by* the other
weather/soil/location features rather than an independent input -- and
raw City is high-cardinality (~40 levels across 272 environments) relative
to sample size, a real overfitting risk for the GBLUP-style linear
baseline in particular. If lat/long + weather + soil don't end up
capturing enough of the City-level gap, a learned City embedding is a
candidate follow-up for the deep model specifically (not the baseline).

**EC (673 crop-model covariates)**: deliberately deferred, not included in
Phase 2. Substantial conceptual overlap with the hand-engineered weather
features; 673 columns for ~270 environments is a dimensionality problem
better addressed as a documented follow-up (e.g. PCA-reduced EC block)
once the effect-decomposition framework is validated on the simpler
representation.

**Final environment feature vector:** weather (engineered) + soil (imputed
+ has_soil_data flag) + latitude/longitude. Saved as
`environment_combined_features.csv`. City and EC excluded from Phase 2.

### Train/test split scheme
CyVerse's native 2014-2023 (train) / 2024 (test) partition as the held-out
evaluation set (confirmed zero environment overlap, all 1,063 test hybrids
have genotype coverage). Leave-one-year-out CV within training years for
model selection.

## Resolved risks

- **Submission template / test_observed environment mismatch (23 vs. 22):**
  confirmed as a genuine single-environment gap, not a data artifact.
  `SCH1_2024` (385 rows) appears only in submission_template -- organizers
  are withholding its ground truth for their own scoring. Evaluation code
  must exclude SCH1_2024 from local metric computation while still
  producing predictions for it.

- **TXH2_2015/2016/2017 missing from weather file:** confirmed as a
  genuine structural gap (TXH2 has weather data for every other year,
  2014/2018-2023, but not these three) -- not a join-key or formatting
  issue. These 3 environments (1,223 trait rows, 0.75% of
  genotype-covered trait data) are EXCLUDED from environment-alone
  modeling. `env_targets_modelable` (269 environments) is the version to
  use going forward, not the full 272-row `env_targets`.

- **TXH1 treatment-suffixed environments (false alarm, resolved):** weather
  file has `TXH1-Dry_2017`, `TXH1-Early_2017`, `TXH1-Late_2017`, and 2018
  equivalents (6 total) instead of plain `TXH1_2017`/`TXH1_2018`. Checked
  whether trait/soil/meta silently used the plain form instead (which would
  mean a silent misjoin) -- they don't: trait and meta both already use the
  same suffixed names as weather, so the existing plain string-match join
  was already correct by construction. Soil has no suffixed entries for
  these 6 (soil not collected at that granularity), which is exactly what
  the `has_soil_data` flag is for -- no fix needed.

## Open risks (not yet investigated)

- Parent inbred genotyping cohort heterogeneity (GBS, WGS, Exome, Assembly
  across 2014-2025) as a potential confound in the genotype representation.

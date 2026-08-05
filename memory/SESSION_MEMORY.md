# Session memory

Persistent context that should carry into every session. Update as
decisions get made -- this is the file to read first.

## Working conventions

- Plan and confirm analytical approach before any code is written; report
  unexpected findings rather than improvising workarounds
- User runs code themselves and pastes results back -- interpretation
  happens on real outputs, not predicted ones
- Incremental iteration, not full rewrites
- WSL2 for local dev/git, Google Colab + Google Drive for GPU-heavy training
- Fit scalers/statistics on train fold only, never leak test statistics
- Train and test metrics always reported side by side
- No pitch/deadline framing -- priority is getting the science right first

## Open decisions (not yet made)

(none currently -- see Confirmed decisions below. New open decisions get
added here as they come up in Phase 2+.)

## Confirmed decisions

- **Genotype representation**: raw dosage matrix (5,899 hybrids x 2,425
  markers, {0, 0.5, 1, NA}) as direct input to the genotype-alone model,
  plus a GBLUP/kinship baseline built alongside it for comparison. Matrix
  is small enough (curated panel, not full-genome SNPs) that PCA
  compression isn't required as a first pass.
- **Environment representation**: weather as the primary signal (269/272
  train envs, 23/23 test envs covered; core variables like T2M, PRECTOTCORR,
  RH2M only ~1-1.5% missing within covered envs). Soil and EC are
  supplementary, imputed or dropped per environment rather than required --
  soil alone is missing for ~30% of environments in both train and test, so
  making it mandatory would silently drop a third of the data. Five NASA
  POWER weather variables (GWETTOP, GWETPROF, GWETROOT, ALLSKY_SFC_PAR_TOT,
  ALLSKY_SFC_SW_DNI) are missing 42-66% within covered envs and need
  separate handling from the reliable core weather variables.
- **Split scheme**: use CyVerse's provided 2014-2023 (train) / 2024 (test)
  split as the held-out evaluation set -- confirmed genuinely disjoint by
  environment (0 overlap, 272 vs. 23 envs) and every 2024 test hybrid has
  genotype coverage, so no custom holdout construction is needed. Also run
  leave-one-year-out CV within the 2014-2023 training years for model
  selection / hyperparameter tuning before touching the 2024 holdout.

## Known risks / open items to check later

- Parent inbreds were genotyped across 8 different cohorts/technologies
  (GBS 2014-2017, WGS 2018-2019, Exome 2020-2021, WGS 2024-2025, etc.).
  The hybrid genotype matrix looks standardized to the common 2,425-marker
  panel, so this is background provenance for now -- but worth checking if
  genotype-environment correlation shows up unexpectedly (e.g. genotyping
  cohort clustering with trial year).
- Test-side environment counts don't fully line up: submission_template /
  test_meta / test_weather all show 23 environments, but test_ec shows 22
  and test_observed (actual ground truth) shows 22. One environment in the
  prediction target doesn't have a matching observed value -- pin down
  which one before writing evaluation code that assumes a clean 1:1 join.

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

- Genotype representation: raw SNPs vs. PCs vs. kinship/GBLUP matrix
- Environment representation: raw daily weather vs. engineered indices
  (growing degree days, water-stress windows, etc.)
- Train/test split scheme: random split is invalid here (leaks G x E
  structure) -- likely leave-one-environment-out or leave-one-year-out
  (2024 competition release is pre-split train 2014-2023 / test 2024)

## Confirmed decisions

(none yet)

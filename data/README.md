# Data

Not committed to git (see `.gitignore`). Download from CyVerse Data Commons
and place under `data/raw/`.

**Always read the `readme.txt` shipped with each release before touching
data** -- environment codes (year + location) are easy to misjoin across
files.

## G2F (Genomes to Fields) releases

| Release | DOI | Notes |
|---|---|---|
| 2022 competition (start here) | `10.25739/tq5e-ak26` | Has a companion paper documenting every file |
| 2024 competition | `10.25739/78mn-4394` | Train 2014-2023, test 2024 |
| 2018 raw field season | `10.25739/anqq-sg86` | |
| 2019 raw field season | `10.25739/t651-yy97` | |

Resolve DOIs via `https://doi.org/<DOI>`.

Competition site / rules: https://www.maizegxeprediction.org/

## File types

- Genotype: VCF/HapMap -- use `scikit-allel` or `cyvcf2`, not pandas
- Phenotype, soil, weather, environmental covariates: plain CSV

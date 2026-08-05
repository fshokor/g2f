# Progress

## Background

Spun out of a prior multiomics project (RNA-protein fusion for drug
response, extended to COVID CITE-seq and NeurIPS multimodal single-cell
data). That project's finding across three datasets: naive or sophisticated
fusion of correlated omics layers rarely improved prediction, and apparent
"coupling" between modalities was usually a confound (cell-type
composition, tissue-of-origin, library size) rather than genuine
complementary biology. Diagnosis: the relationship being sought was never
rigorously defined -- what relationship, between what, for what trait.

Decision: fresh project, no pitch/deadline pressure, on a better-suited
dataset -- genomic prediction (genotype x environment x yield), where the
two effects are mechanistically distinct rather than correlated readouts of
the same biology, and the trait (grain yield) is physically measured.

## Key risk (carried into design from day one)

Two "effects" can still be correlated in the data (e.g. certain hybrids
disproportionately tested in certain environments). An effect-alone model
can then partially absorb the other effect's signal, so the measured
"relationship between effects" partly reflects leakage rather than true
complementarity. Orthogonalization/residualization between effects is
built into the diagnostic layer from the start, not added after the fact.
Correlation between per-effect votes and fusion improving held-out
prediction are two separate things -- both get checked.

## Status

- [x] Project scoped, dataset selected (G2F)
- [x] Repo scaffolded
- [x] 2024/2025 competition dataset downloaded (train 2014-2023 + test 2024)
      and audited via 00_data_setup_and_exploration.ipynb
- [x] Genotype representation decided -- see SESSION_MEMORY.md
- [x] Environment representation decided -- see SESSION_MEMORY.md
- [x] Train/test split scheme decided -- see SESSION_MEMORY.md
- [ ] Genotype-alone baseline built
- [ ] Environment-alone baseline built
- [ ] Diagnostic/orthogonalization layer built
- [ ] Fusion model built and evaluated on held-out data

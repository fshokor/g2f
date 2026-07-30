# G2F Effect-Decomposition Framework

Effect-decomposition approach to multi-effect trait prediction, applied to the
Genomes to Fields (G2F) maize genotype x environment x yield dataset.

## Core idea

1. Train a separate model to predict grain yield from **genotype alone** and
   another from **environment alone**. Each becomes that effect's "vote."
2. Study the **relationship between votes** (agreement, divergence, and what
   divergence correlates with) as a diagnostic layer explaining why
   integration helps or fails -- with orthogonalization built in from the
   start to guard against one effect absorbing the other's signal.
3. Feed that diagnostic understanding into a **fusion model**, evaluated on
   held-out data, to test whether integration genuinely improves prediction.

Full background and rationale: see `memory/PROGRESS.md`.

## Repo layout

```
notebooks/    sequential numbered analysis notebooks (00, 01, 02, ...)
scripts/      reusable Python modules imported by notebooks
data/         raw/ and processed/ G2F data (gitignored, see data/README.md)
outputs/      trained models and figures (gitignored)
memory/       session-tracking files: SESSION_MEMORY.md, NEXT_SESSION.md, PROGRESS.md
```

## Dataset

Genomes to Fields (G2F), 2022 competition release (recommended starting
point). See `data/README.md` for download instructions and DOI links.

## Environment

- Local dev / git: WSL2
- GPU training: Google Colab + Google Drive
- See `requirements.txt` for Python dependencies

## Conventions

- Plan and confirm analytical approach before writing code
- Fit scalers/statistics on train fold only, never leak test statistics
- Report train and test metrics side by side
- Notebooks: typed function signatures, minimal comments (why not what),
  no print-inside-functions, tested via `jupyter nbconvert --execute` on
  synthetic data before delivery
- Incremental iteration, not full rewrites

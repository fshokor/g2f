# Next Session -- G2F Effect-Decomposition Framework

## Primary next step: out-of-fold rebuild for Phase 4

Everything in `05_effect_relationships.ipynb` right now (`genetic_value`,
`environment_value`, the confound correlation, the cross-prediction MLPs,
`h()`, and both fusion formulas) is built on **in-sample, final-model**
predictions -- optimistic to an unknown degree, most for the two
relationship MLPs and the fusion coefficients.

The fix is cheap because `03` and `04` already use the *same*
leave-one-year-out fold structure: for any held-out year Y, `03`'s fold-Y
GBLUP model and `04`'s fold-Y env_mlp_l2 model both never saw year Y. So for
every observation in year Y, we can attach that year's held-out-fold GBLUP
prediction (by Hybrid) and that year's held-out-fold env_mlp_l2 prediction
(by Env) -- genuinely out-of-sample for both, no new modeling required, just
plumbing.

Concretely:
1. Confirm whether `03`/`04` already persist per-fold, per-hybrid /
   per-environment predictions anywhere, or only aggregate fold metrics
   (last known state: aggregate only -- would need adding).
2. Re-run (or extend) `03`'s and `04`'s fold loops to capture and save
   per-fold predictions for every hybrid/environment in the held-out year.
3. Rebuild the Hybrid x Env cell-mean join table using these OOF values
   instead of the final in-sample refits.
4. Re-run Section 4 (confound check), Section 5 (`h()`), and Section 6
   (fusion baseline + comparison) with the OOF table.
5. Specifically re-check: does genotype's weak cell-level test showing
   (r=0.116 in-sample) hold up, get worse, or improve under OOF? Does the
   fusion RMSE-worse-than-environment_value-alone pattern persist?

## Secondary items, lower priority

- Now that `h()` was found to add ~nothing over a plain linear term
  in-sample, consider whether to keep the h-spliced fusion path in the OOF
  rerun at all, or simplify to just the plain linear formula plus a note
  that the functional-form check was done and came back linear.
- If the OOF fusion numbers still show fusion trailing environment_value
  alone on RMSE, that's a real result worth designing around (not a bug to
  chase) -- worth thinking about whether a different fusion form
  (regularized regression, or weighting genetic_value down given its weaker
  standalone showing) would close the gap, once OOF numbers are in hand.
- Parent inbred genotyping cohort heterogeneity (GBS/WGS/Exome/Assembly) --
  still flagged, still not investigated. Worth a quick look at whether
  cohort membership correlates with `genetic_value` residuals.
- Formalize the 23-vs-22 test environment mismatch handling into a written
  decision rather than leaving it as inline generic-intersection logic.

## Reminders for whoever picks this up

- `genetic_value` = GBLUP's prediction (alpha=0.1, from `03`). `environment_value`
  = env_mlp_l2's prediction (lambda=1.0, hand-picked, from `04`). Both
  hardcoded as constants in `05` rather than re-selected there.
- If refitting either model anywhere, use `check_refit_matches_source()` (in
  `05`) or add an equivalent check -- this caught a real bug once already
  (env_mlp_l2 refit silently undertrained by a hyperparameter mixup) and
  should be treated as a standard guard, not a one-off.
- Fusion granularity is Hybrid x Env cell mean, not raw plot-level --
  keep that consistent if extending Phase 4/5 further.

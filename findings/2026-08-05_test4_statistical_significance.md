# Test 4 — Statistical Significance of the Corrected Human-vs-LLM Gap

**Date:** 2026-08-05
**Status:** PENDING — not yet run
**Parent decision doc:** `reports/2026-08-05_family_a_warmup_decision.md` (gap #4)
**Depends on:** step 4 of that doc (corrected `data/p0_corpus_level_SH.json` variant) must exist first — this test consumes its output, it doesn't recompute trajectories itself.

---

## Context

The parent decision doc's step 5 ("direction/sign check") asks whether corrected LLM `S`/`H` is still greater than Human `S`/`H`, per domain — a yes/no on the point estimate. That's necessary but not sufficient. A gap that goes from 0.095 to 0.012 is technically still "positive" but may not be distinguishable from zero given the corpus's own sampling variance — reporting it as "the claim survives" without a significance check would overstate what was actually shown.

The project already has an established bootstrap methodology for exactly this purpose: `reports/2026-07-26_S_estimator_fix.md` confirms `calc_S`/`calc_H` support an outer 10-fold bootstrap resample for confidence intervals (the "bootstrap-10" language in `Extension1_PhaseWise_Execution_Plan.md`, distinct from the inner `N_BOOT=2000` per-scale draws). Use that existing mechanism — don't design a new one.

## Objective

For each of the 5 domains, compute confidence intervals on the Human-vs-LLM `S` and `H` gap, both before and after the `continue_it`/`continue_pt` correction, and determine whether the corrected gap remains statistically distinguishable from zero.

## Method

1. For each (domain, label) cell in both the original and corrected `data/p0_corpus_level_SH.json`, run the existing 10-fold outer bootstrap (resample documents with replacement within the cell, recompute pooled `S`/`H` per resample) to get a CI on each cell's `S` and `H`.
2. Compute the gap CI directly: for each of the 10 bootstrap folds, compute `(LLM_S_fold − Human_S_fold)`, then take the 2.5th/97.5th percentiles of that distribution as the gap's 95% CI. Do this for `gemma_2b` and `mistral_7b` separately (matching the existing table's per-scorer rows) — don't pool scorers together.
3. Repeat identically for the corrected data.
4. Flag any domain/scorer/exponent cell where the **original** gap's CI excluded zero but the **corrected** gap's CI includes zero — this is the specific "claim no longer holds at this domain" outcome, distinct from "gap got smaller but is still real."

## Output format

| domain | scorer | exponent | original gap | original 95% CI | corrected gap | corrected 95% CI | CI still excludes 0? |
|---|---|---|---|---|---|---|---|

15 rows minimum (5 domains × gemma/mistral, `S` and `H` — matching the structure of the existing table in `reports/2026-07-26_S_estimator_fix.md`).

## Pass/fail criteria

- **Pass (claim survives with statistical backing):** corrected gap's CI excludes zero in the same direction as the original, for a given domain/scorer/exponent — the finding holds up, not just nominally.
- **Weakened (report explicitly, don't bury):** corrected gap's CI still excludes zero but is visibly narrower/closer to zero than the original — the claim survives but is quantifiably weaker than originally reported. State the exact before/after CI width, not just "still significant."
- **Fail (claim does not survive at this cell):** corrected gap's CI includes zero — cannot claim a reliable Human-vs-LLM difference for this domain/scorer/exponent after correction.

## How this feeds the overall conclusion

This is the test that turns "the sign is still positive" into an actual defensible claim. Any domain/scorer cell that fails here needs to be reported as a genuine limitation in the corrected write-up, not smoothed over by pointing at the pooled point estimate alone — consistent with how `reports/2026-07-26_S_estimator_fix.md` already disclosed that Fig. 10's narrow-human-band claim didn't replicate, rather than omitting it.

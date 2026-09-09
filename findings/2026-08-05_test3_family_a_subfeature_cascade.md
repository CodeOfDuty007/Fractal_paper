# Test 3 — Family A Sub-Feature Cascade Check (Beyond Point S/H)

**Date:** 2026-08-05
**Status:** PENDING — not yet run
**Parent decision doc:** `reports/2026-08-05_family_a_warmup_decision.md` (gap #3)
**Depends on:** the `continue_it`/`continue_pt` windowing fix (`skip = prefix_len + WARMUP_DROP`) being implemented per that doc's step 3

---

## Context

Every discussion of the `continue_it`/`continue_pt` defect so far has talked about "`S`/`H`" as if that's the whole of Family A. It isn't. Per `Extension1_PhaseWise_Execution_Plan.md` (Phase P1) and the design-decisions section of `reports/2026-07-31_P1_feature_store.md`, Family A includes, per document per scorer:

- point `Ŝ`, `Ĥ`
- bootstrap CI half-widths
- slope-fit residuals
- **windowed** mean/var/min/max/drift of `S` and `H` (200-token, 100-stride overlapping windows, `scale=[8,16,32,48,64,96]`)
- scorer-divergence terms (Gemma-2B vs Mistral-7B)

All of these are computed from the same underlying trajectory. If the trajectory is wrong for `continue_it`/`continue_pt` (prefix not properly skipped), **every one of these derived features is wrong too**, not just the two headline numbers — but they may not all be wrong by the same amount, and some may be far more sensitive to the windowing error than others (e.g., a windowed "drift" feature computed from four separate 200-token sub-windows could have its *early* windows almost entirely inside the old contaminated region, while a corpus-pooled point `S` dilutes that same contamination across the whole document).

**Also unaddressed so far: Mistral-7B.** Every finding in `reports/2026-08-03_prompt_type_reconstruction_probe.md` and `reports/2026-07-31_P1_feature_store.md`'s Finding 1 was established using **Gemma-2B** scores only. Nobody has checked whether Mistral-7B's stored score arrays for `continue_it`/`continue_pt` have the same prefix-inclusion structure, the same prefix length, or even the same defect at all. If Mistral-7B's arrays are structured differently, the scorer-divergence features (which explicitly compare Gemma vs Mistral) could be doubly distorted — one operand fixed, one not, or both wrong in different, non-cancelling ways.

## Objective

1. Confirm (or refute) that Mistral-7B has the same prefix-inclusion defect as Gemma-2B for `continue_it`/`continue_pt`.
2. Quantify how much each Family A sub-feature (not just point `S`/`H`) shifts under the corrected windowing, and flag which sub-features are most sensitive.

## Method

### Part A — Mistral-7B check
1. Repeat the length-triangulation check from the probe's §1 (`text_alone_delta`, `combined_delta`) for Mistral-7B's stored score arrays on the same `continue_it`/`continue_pt` sample already used for Gemma-2B, so results are directly comparable.
2. Repeat the boundary-crossover diagnostic (probe Part B, step 6 — piecewise shift-fit around the prefix/text seam) for Mistral-7B on 5-10 records.
3. Report: does Mistral-7B's prefix length distribution match Gemma-2B's (mean ~94 tokens, std ~9.5)? Does it need its own `skip` formula, or does `prefix_len_tokens` (computed once, tokenizer-dependent) already transfer correctly to both scorers?

### Part B — Sub-feature sensitivity
For a sample of n=200 `continue_it`/`continue_pt` documents already identified as affected:
1. Compute the full Family A feature vector (all sub-features listed above) under the **old** flat-64 windowing and the **corrected** `prefix_len+64` windowing.
2. For each sub-feature, report mean absolute change and % of documents with a change exceeding a feature-appropriate threshold (use each feature's own reported noise/std from prior validation as the threshold — e.g., windowed `S`'s already-known std of 0.35–0.48 from `reports/2026-07-31_P1_feature_store.md` means a 0.05 shift there is noise-level, while the same 0.05 shift in point `H` (std ~0.07 per that report) is not).
3. Explicitly flag any sub-feature whose early-window components (the first 200-token window, which overlaps most with the contaminated region) show materially larger shifts than later windows — this would indicate the windowed features are more exposed to this defect than the point estimates, not less.

## Output format

| sub-feature | mean |Δ| | % docs |Δ|>threshold | threshold used | verdict |
|---|---|---|---|---|

Plus the Mistral-7B section as a short pass/fail note: same defect confirmed / different structure, needs separate handling / not affected.

## Pass/fail criteria

- **Pass (per sub-feature):** mean |Δ| stays below that feature's own established noise floor — the correction doesn't meaningfully move this particular feature even though it moves the point estimate.
- **Fail (per sub-feature):** mean |Δ| exceeds the noise floor — this feature must be recomputed in the P1 feature store, not just the two headline columns.
- **Mistral-7B:** if the defect is confirmed there too, the fix (`skip = prefix_len + WARMUP_DROP`) must be applied to both scorers' pipelines before any scorer-divergence feature is trusted, and the recompute in the parent decision doc's step 3 must cover both scorers, not just Gemma-2B.

## How this feeds the overall conclusion

If some sub-features (e.g., windowed drift, early-window stats) turn out to be far more contaminated than the point `S`/`H` values already being corrected, the P1 feature store recompute needs to be scoped wider than currently planned — and if Mistral-7B has its own version of this bug, the fix needs a second, parallel implementation before scorer-divergence features (used in Family A's cross-scorer terms) can be trusted at all.

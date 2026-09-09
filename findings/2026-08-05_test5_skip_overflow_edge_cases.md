# Test 5 — Skip-Overflow Edge Cases in the Corrected Windowing

**Date:** 2026-08-05
**Status:** PENDING — not yet run
**Parent decision doc:** `reports/2026-08-05_family_a_warmup_decision.md` (gap #5)
**Depends on:** nothing new — reuses length data already computed in `reports/2026-08-03_prompt_type_reconstruction_probe.md`, §1

---

## Context

The proposed fix changes the skip from a flat constant to `skip = prefix_len_tokens + WARMUP_DROP` (64). The parent decision doc's step 3 plans a full recompute over ~57k rows unattended. Before that runs, it's worth checking whether `skip` can ever exceed what's actually available in the document, which would silently produce an empty or negative-length trajectory rather than a loud failure.

Evidence this isn't a purely theoretical concern: the probe's §1 `combined_delta` table already shows real variance in this direction — `continue_pt`'s `combined_delta` has mean 15.35 but **std 88.50 and max 935**, a much longer tail than `continue_it`'s (mean 7.23, std 8.72). That max-935 figure describes the gap between a *naive* prefix+text concatenation and GAGLE's array, not `prefix_len_tokens` directly, but it signals that `continue_pt` in particular has some documents whose length arithmetic is far from the typical case — exactly the kind of outlier that would break silently if not checked explicitly.

## Objective

Determine what fraction of `continue_it`/`continue_pt` documents would have `skip ≥ gagle_len` (the corrected window consumes the entire array or more, leaving zero or negative tokens for the actual fractal computation) under the corrected formula, and confirm the pipeline's existing length gate catches these rather than passing a corrupted trajectory through silently.

## Method

1. For the full `continue_it`/`continue_pt` population already flagged for recompute (~57k rows, or the n=500 sample already used in the probe if the full population isn't readily re-queryable), compute `skip = prefix_len_tokens + 64` and `remaining = gagle_len − skip` for every document.
2. Report the distribution of `remaining`: min, 1st percentile, 5th percentile, median.
3. Count and report:
   - documents where `remaining ≤ 0` (skip consumes the whole array — must be dropped, not processed)
   - documents where `0 < remaining < 320` (survives slicing but falls under this pipeline's own `MIN_TOKENS=320` floor, per `build_p0_local.py` — should already be excluded by the existing length gate, but confirm this explicitly rather than assuming)
   - documents where `remaining ≥ 400` (clean — full window available, matching the paper's clip-to-400 requirement)
4. Trace what the **current, unfixed** pipeline does with these same documents today — does it also drop them (if the flat-64 skip already put them under 400/320), or does the corrected `skip` newly push some previously-kept documents below threshold? This matters for the row-count reconciliation in the parent doc's step 3 ("57k rows in, 57k rows out") — if the corrected windowing legitimately drops more documents than before, that's expected and should be reflected in the row count, not treated as a bug.

## Output format

| prompt_type | n | % remaining≤0 | % 0<remaining<320 | % remaining≥400 | min remaining | p1 remaining | p5 remaining |
|---|---|---|---|---|---|---|---|

Plus one explicit statement: "of the ~57k rows planned for recompute, N are expected to newly drop below the length gate under corrected windowing, leaving M rows with valid corrected Family A values."

## Pass/fail criteria

- **Pass:** the existing `MIN_TOKENS=320` / clip-to-400 gate (already implemented in `build_p0_local.py` and reused elsewhere) naturally absorbs every `remaining≤0` and `remaining<320` case without code changes — these documents get correctly excluded the same way any other too-short document already is, and the recompute in the parent doc's step 3 should expect a row count of `57k − (dropped count)`, not exactly 57k.
- **Fail:** any document with `remaining≤0` or `remaining<320` is currently being processed anyway (e.g., a downstream function doesn't re-check length after the corrected slice, and computes `S`/`H` on a near-empty or empty array without erroring) — this is a real bug that must be fixed before the full recompute runs, since it would silently produce garbage `S`/`H` values rather than a clean exclusion.

## How this feeds the overall conclusion

This test doesn't change the scientific conclusion (survives/weakened/doesn't survive) — it protects the *mechanics* of getting there. Running the full 57k-row recompute without this check risks either a silent crash partway through, or worse, silently-wrong `S`/`H` values for edge-case documents that then feed into the corrected headline table in step 4, undermining the very number this whole effort is trying to get right.

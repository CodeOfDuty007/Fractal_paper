# Test 1 — Family A Contamination Risk in the 4 Unresolved Prompt Types

**Date:** 2026-08-05
**Status:** PENDING — not yet run
**Parent decision doc:** `reports/2026-08-05_family_a_warmup_decision.md` (gap #1)
**Depends on:** nothing new — reuses data already computed in `reports/2026-08-03_prompt_type_reconstruction_probe.md`

---

## Context

The confirmed Family A defect (`WARMUP_DROP=64` not clearing GAGLE's prefix-inclusive score arrays) was found and fixed for `continue_it`/`continue_pt` only. `excerpt`, `short_keywords`, `summary_keywords`, and `cot` still don't have a confirmed Family B reconstruction rule (see the probe report, Part A). That does **not** mean Family A is safe for them — it means nobody has checked. This test checks Family A risk directly, **without waiting for Family B's alignment problem to be solved**, by asking a narrower question: *does the fractal math conclusion (S/H) change materially depending on where the score-array window starts?* If S/H is insensitive to plausible starting-offset choices, Family A risk is low regardless of the Family B outcome. If it's highly sensitive, Family A is not trustworthy for that type yet, independent of anything else.

## Known facts to build on (already established, don't re-derive)

From `reports/2026-08-03_prompt_type_reconstruction_probe.md`, §1 and §6:

| prompt_type | mean prefix_len (tokens) | text_alone_delta mean | text_alone_delta std | direction |
|---|---|---|---|---|
| excerpt | 916.9 | 14.24 | 11.51 | array ≈ text length, small residual |
| short_keywords | 80.2 | 17.04 | 7.73 | array ≈ text length, tight residual |
| summary_keywords | 227.7 | 14.57 | 8.44 | array ≈ text length, small residual |
| cot | 122.5 | 141.60 | 87.93 | array **shorter** than raw text, highly variable |

Note the asymmetry: `excerpt`/`short_keywords`/`summary_keywords` look close to "text-alone" in length (small, fairly consistent residual) — this is a **different** failure mode from `continue_it`/`continue_pt`'s "array is way too short because a whole prefix is missing" problem, and a different one again from `cot`'s "array is way shorter than the raw text by a variable amount" pattern (consistent with a preamble being excluded, per the probe's §5 marker-detection finding: 78% of `cot` docs contain reasoning markers).

## Objective

For each of the 4 unresolved types, quantify how much the corpus-level pooled `S` and `H` point estimate moves as the score-array window start is swept across a plausible range — using the array as GAGLE actually provides it (no re-scoring, no assumption about the "correct" alignment). This is a sensitivity analysis, not an alignment fix.

## Method

1. For each type, sample n=300 documents (reuse the stratified sampler from `src/investigate_prompt_type_alignment.py` — same domain × model × temp strata already used).
2. For each sampled document, take GAGLE's stored score array as-is (length = `gagle_len`).
3. Sweep a window-start offset `k` from 0 to `min(100, gagle_len - 400)` in steps of 8 (matching the existing scale granularity). At each `k`, slice `array[k : k+400]` and compute per-document `Hurst_Exp` via the existing `calc_H` (skip `calc_S` here — already known to be per-document-unreliable regardless of windowing, per `reports/2026-07-26_S_estimator_fix.md`; this test is about **corpus-level pooled** `S` only, computed after pooling all documents' trajectories at a given `k`).
4. At each `k`, pool all 300 documents' trajectories (list-of-trajectories mode, matching the validated production pooling method) and compute corpus-level `S` and `H`.
5. Plot/tabulate corpus-level `S(k)` and `H(k)` as a function of `k`, per type.

## Output format

One table per type:

| offset k | pooled S | pooled H | ΔS vs k=0 | ΔH vs k=0 |
|---|---|---|---|---|

Plus a single summary line per type: `max|ΔS| across k∈[0,100]`, `max|ΔH| across k∈[0,100]`.

## Pass/fail criteria

- **Low risk (pass):** `max|ΔS| < 0.02` and `max|ΔH| < 0.02` across the swept range — the fractal point estimate is stable regardless of exactly where the window starts, so today's numbers for this type are trustworthy even without a fully solved Family B rule.
- **High risk (fail):** either exceeds 0.02 — the type's Family A numbers are sensitive to an unresolved alignment question and should be treated as **not yet trustworthy**, flagged the same way `Holder_Exp` per-document is already flagged (`Holder_PerDoc_Unreliable`-style convention), pending Family B resolution.
- Report per type, don't average across types — `cot`'s failure mode (array too short) and `excerpt`'s (array ≈ right length, small residual) are different enough that a shared verdict would hide which one is actually the problem.

## How this feeds the overall conclusion

If all 4 types pass, the confirmed defect is fully bounded to `continue_it`/`continue_pt` and the fix in `reports/2026-08-05_family_a_warmup_decision.md` is sufficient on its own. If any fail, that type's existing Family A values need the same "flag as unreliable, don't ship as a point feature" treatment already applied to per-document `S`, and it needs a red flag in the P1 feature manifest before any P2/P3 work touches it.

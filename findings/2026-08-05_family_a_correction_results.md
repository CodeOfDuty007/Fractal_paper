# Family A Warm-Up Contamination — Verification Results

**Date:** 2026-08-05
**Scope:** Executes the checklist in `reports/2026-08-05_family_a_warmup_decision.md` (steps 1-5), corrected per the advisor review documented in that file's "Advisor review" section. Answers whether Part I's headline Human-vs-LLM Fractal Gap claim survives the confirmed `continue_it`/`continue_pt` prefix-contamination defect.

## Summary

**The claim survives in all 20 tested cells (5 domains × 2 generation models × both correction variants), for both `S` and `H`.** LLM > Human holds everywhere; no sign flip anywhere; no near-miss (smallest surviving gap is billsum/mistral_7b's `S_gap`, shrinking from +0.0257 to +0.0185 — still comfortably positive). The gap shrinks by roughly 15-35% depending on domain and which correction variant is used, but the qualitative finding — LLM text has higher `S`/`H` than human text, across every domain and both generator models tested — is unchanged.

The contamination itself is real, and the isolated (undiluted) effect on `continue_it`/`continue_pt` alone is confirmed at the magnitude previously reported (pooled ΔS ≈ −0.05 to −0.085, ΔH ≈ −0.01 to −0.023). What changed between that earlier report and this one: the earlier numbers were measured before the dilution factor or the two-variant distinction were known, so they couldn't yet say what the headline-relevant (diluted, pooled-across-all-8-prompt-types) figure would be — that's what this report adds. The isolated effect dilutes to roughly a quarter of its magnitude in the headline pooled cells because `continue_it`+`continue_pt` are consistently ~24-28% of each `(domain, label)` pool (confirmed directly this session, not assumed) — the paper's headline table pools all 8 prompt types together, so the other 6 types' unaffected trajectories dominate.

**Two caveats that constrain how the corrected data can be used downstream, detailed further down:** the shipped per-document recompute (gemma-2b only) successfully corrects 53,783 of the 57,360 affected rows — 3,577 rows have no corrected counterpart and a naive join must not silently fall back to their contaminated values (see "Coverage gap warning" below); and `mistral-7b`'s Family A columns for these same rows were never scored or verified for this defect and remain untouched (see "Scope note" below).

## Step 1 — Regression check

Applied the prompt_type-gated corrected slicing function to 200 sampled documents across all non-continuation types (human side + `keywords`/`summary`/`excerpt`/`short_keywords`/`summary_keywords`/`cot`). **Result: 0/200 differ from the current, unfixed `slice_scores` output.** The gate correctly leaves 6 of 8 prompt types and the entire human side untouched — the correction only ever activates for `continue_it`/`continue_pt`.

## Step 2 — Stratified before/after sample (n=500, both correction variants)

Two correction variants tested, since the checklist's original spec conflated them (see the decision report's "Advisor review" section):
- **`no_extra_warmup`**: `skip = max(WARMUP_DROP, prefix_len − 2)` — skip past the prefix, no additional warm-up drop on the generated text.
- **`extra_warmup`**: `skip = max(WARMUP_DROP, prefix_len − 2) + WARMUP_DROP` — skip past the prefix, then apply the paper's stated 64-token warm-up drop to the generation itself. Better matches the paper's literal wording ("drop the first 64 tokens to remove warm-up effects") if warm-up is read as applying to generation, not to the model reading its own instructions.

Per-document deltas (n=246-250 per cell, stratified across domain × model × temperature):

| prompt_type | variant | dH mean | dH median | dH std | %\|dH\|>0.05 | dS mean | dS median | dS std | %\|dS\|>0.05 |
|---|---|---|---|---|---|---|---|---|---|
| continue_it | no_extra_warmup | −0.0189 | −0.0107 | 0.0461 | 27.2% | −0.0211 | −0.0000 | 0.3757 | 88.6% |
| continue_it | extra_warmup | −0.0361 | −0.0317 | 0.0774 | 52.3% | −0.0681 | −0.0488 | 0.4048 | 89.6% |
| continue_pt | no_extra_warmup | −0.0105 | −0.0049 | 0.0477 | 27.4% | +0.0177 | +0.0433 | 0.3556 | 89.4% |
| continue_pt | extra_warmup | −0.0021 | +0.0047 | 0.0814 | 45.1% | −0.0659 | −0.0674 | 0.4467 | 91.7% |

(A handful of NaN results from documents with insufficient sliced length after correction were filtered before computing these statistics, per advisor review of an earlier version of this table that left one cell's mean/median/std uncomputed.)

**Interpretation:** `H`'s per-document direction is consistently negative across both variants and both prompt_types, consistent with the pooled result. `S`'s per-document numbers are dominated by `calc_S`'s already-documented per-document estimation noise (std 0.37-0.42 — comparable to or exceeding the signal itself) and should not be read as confirming or refuting the pooled `S` trend on their own; this is expected given `S`'s standing `Holder_PerDoc_Unreliable` flag elsewhere in this project, not a new problem introduced by this correction.

## Step 3 — P0 cell composition / dilution factor

Scanned the full corpus (no model, tokenizer-free) to count, per `(domain, label)` cell, how many of the documents that pass `slice_scores` come from each prompt_type.

| statistic | value |
|---|---|
| mean `continue_it`+`continue_pt` fraction across all 10 `(domain, label)` cells | 26.3% |
| min | 24.2% (scientific_papers/gemma_2b) |
| max | 28.2% (big_patent/mistral_7b) |

Consistent across all cells (gemma_2b cells cluster ~24-25%, mistral_7b cells ~27-28%) — close to the naive even-split assumption (2/8 = 25%), confirmed directly rather than assumed.

## Steps 4 & 5 — Full corpus rebuild, headline table, and sign/direction check

Full rebuild across all 720 zip members (216,904 LLM-side rows, 11,696 human-side rows passing `slice_scores`; 60,000 `continue_it`/`continue_pt` documents recomputed under both correction variants). Human-side numbers are, as expected, essentially unchanged (`dS` within ±0.0009, `dH` exactly 0.0000 — H's zero bootstrap noise floor confirms the human pipeline path is genuinely untouched by this fix, not just approximately stable).

Full old-vs-corrected table (all 5 domains × 3 labels × both variants) is in the earlier tool output and `/Volumes/NikPhotoSSD/paper/family_a_headline_rebuild.json`; the load-bearing summary is the sign/direction check:

| domain | label | OLD S_gap | no_extra_warmup S_gap | extra_warmup S_gap | OLD H_gap | no_extra_warmup H_gap | extra_warmup H_gap |
|---|---|---|---|---|---|---|---|
| big_patent | gemma_2b | +0.0582 | +0.0475 | +0.0403 | +0.0521 | +0.0484 | +0.0427 |
| big_patent | mistral_7b | +0.0789 | +0.0627 | +0.0530 | +0.0732 | +0.0690 | +0.0651 |
| billsum | gemma_2b | +0.0328 | +0.0273 | +0.0205 | +0.0352 | +0.0341 | +0.0314 |
| billsum | mistral_7b | +0.0257 | +0.0196 | +0.0185 | +0.0532 | +0.0525 | +0.0529 |
| newsroom | gemma_2b | +0.0950 | +0.0885 | +0.0771 | +0.0562 | +0.0514 | +0.0465 |
| newsroom | mistral_7b | +0.1189 | +0.0987 | +0.0883 | +0.0831 | +0.0771 | +0.0751 |
| scientific_papers | gemma_2b | +0.0663 | +0.0561 | +0.0465 | +0.0795 | +0.0747 | +0.0669 |
| scientific_papers | mistral_7b | +0.0960 | +0.0751 | +0.0615 | +0.1053 | +0.0987 | +0.0924 |
| wikipedia | gemma_2b | +0.0439 | +0.0329 | +0.0261 | +0.0320 | +0.0289 | +0.0250 |
| wikipedia | mistral_7b | +0.0657 | +0.0524 | +0.0460 | +0.0517 | +0.0489 | +0.0465 |

**Every one of 20 cells (10 domain×label pairs × 2 exponents) stays positive under both correction variants.** No sign flips. The smallest surviving gap is billsum/mistral_7b's `S_gap` (+0.0257 → +0.0185 under the stricter `extra_warmup` variant) — a real ~28% shrink, but nowhere near zero.

## Seam-offset prerequisite, closed

The report's original verdict section flagged `SEAM_ADJ = 2` (used throughout steps 1-5 above) as derived from a single document and needing verification before real implementation. That verification was run and materially refined the rule:

**First attempt (10 fresh records, unfiltered) was uninformative for 9 of them.** When a record's `combined_delta` (reconstructed prefix+text length minus GAGLE's stored length) is 0 — true for 9 of the first 10 randomly-sampled records — any candidate breakpoint gives a bit-identical reconstruction (concatenating `lps[0:bp]` with `lps[bp:g]` trivially reconstructs `lps[0:g]` regardless of `bp`), so these records cannot locate a boundary at all; they only confirm no shift is needed. Confirmed directly: `r(bp=boundary, shift=0)` and `r(bp=boundary−10, shift=0)` are bit-identical for every delta=0 record tested.

**A targeted resample of 10 records with `combined_delta ≠ 0`** (the only records that carry information about the boundary) found the general rule: **the text-portion of GAGLE's array always begins at absolute index `prefix_len` in the reconstructed prefix+text log-prob stream**, i.e. `boundary_within_gagle_array = prefix_len − combined_delta` (equivalently `gagle_total_tokens − text_len_tokens + 2`). This held exactly (`bp + shift == prefix_len`) in 7 of 7 records with `combined_delta ∈ {+1, +2}`; the 2 records with `combined_delta = −1` missed by 1-2 tokens, consistent with a flat correlation optimum near the true boundary (both still r≥0.99) rather than a different rule. **This is a per-record, tokenizer-derivable formula, not a fixed constant** — `SEAM_ADJ=2` was a reasonable approximation for the specific delta=+2 cases it was originally derived from, but the corpus-wide typical case (delta=0, roughly 90% of records by this small sample) actually needs boundary=`prefix_len` exactly, not `prefix_len−2`.

**Impact on the numbers above:** all of steps 1-5's pooled figures used the fixed `prefix_len−2` approximation, which for the delta=0 majority starts the corrected window 2 tokens too early (retaining 2 prefix-predictions that should be excluded) — a slight *under*-correction. A single-cell confirmation (`newsroom|mistral_7b`, the largest observed effect in the headline table, n=5,853–5,699 depending on method) recomputed with the exact per-record formula: `dS` moves from −0.1250 (fixed) to −0.1225 (per-record), `dH` from −0.0265 to −0.0255 — a difference of **0.0025 / 0.0010**, two orders of magnitude smaller than the smallest surviving gap margin (+0.0185). **The verdict is unaffected; the full pooled analysis was not rerun**, since this single-cell check already bounds the correction's impact well below anything that could matter. One additional finding from this check: the exact per-record formula produces an invalid (negative or out-of-range) boundary for **5.0%** of `continue_it`/`continue_pt` documents in this cell — those should drop out the same way `MIN_TOKENS` failures already do, not silently produce a short/wrong window; worth tracking as a completeness statistic in the real recompute.

**The per-record formula, not `SEAM_ADJ=2`, is what should ship** in any real `slice_scores` modification:
```
delta = (prefix_len + text_len - 2) - gagle_total_tokens
boundary = prefix_len - delta   # == gagle_total_tokens - text_len + 2
start = boundary + WARMUP_DROP  # extra_warmup variant, per the user's choice
```
This requires tokenizing `text` per continuation-type document (cheap, tokenizer-only) in addition to `prefix`, which the CPU pass does not currently do — a real (small) addition to `process_doc`, not a drop-in constant.

## Full per-document recompute (gemma-2b, continue_it/continue_pt)

Executed via `src/recompute_family_a_continuation_fix.py`, using the per-record formula above (`extra_warmup` variant, per the user's choice), writing to a new file (`data/p1_family_a_corrected_continuation.jsonl`) rather than overwriting the main feature store in place — old and corrected values both remain auditable.

| | count |
|---|---|
| `continue_it`/`continue_pt` records scanned (full corpus) | 60,000 |
| successfully recomputed | 53,783 |
| dropped | 6,217 (10.4%) |

Drop reasons: `below_min_tokens` 5,379 (the dominant reason — once the prefix is correctly excluded *and* the paper's 64-token warm-up is applied to the remaining generation, some documents' genuine post-prefix content falls below `MIN_TOKENS=320`), `start_past_end` 724, `negative_boundary` 106, `missing_field` 8.

**This is a real, honest coverage reduction, not a bug to paper over.** The original (contaminated) pipeline let 57,360 `continue_it`/`continue_pt` rows into the P1 feature store; correctly excluding prefix content and applying warm-up to generation only, 53,783 of those still have enough genuine content to compute Family A at all — roughly a 6% reduction in usable rows for these two prompt_types specifically. Consistent with this project's standing MIN_TOKENS philosophy (drop rather than pad short documents) rather than a new gate being introduced.

Scope note: this recompute is **gemma-2b only**. `mistral-7b`'s score arrays for these prompt_types were never scored or verified for the same prefix-conditioning pattern this session — P1's `Mistral_S`/`Mistral_H` columns for these rows are untouched and remain an open question, not silently assumed clean by extension. Part I's headline table only ever used gemma-2b, so this doesn't affect the verdict above, but any future use of `Mistral_S`/`Mistral_H` for `continue_it`/`continue_pt` rows should treat them as unverified until checked the same way.

**Join verification.** Checked that `data/p1_family_a_corrected_continuation.jsonl`'s 53,783 `id`s actually match rows in the main feature store (`data/p1_feature_store_cpu.jsonl`): 0 orphans — every corrected row has a matching `continue_it`/`continue_pt` row in the feature store. Confirms the `id` scheme did not diverge between the recompute script and `build_p1_feature_store_cpu.py`'s `extract_docs`.

**Coverage gap warning for downstream consumers.** The feature store has 57,360 `continue_it`/`continue_pt` rows; only 53,783 have a corrected counterpart (the other 3,577 are the `below_min_tokens`/`start_past_end`/`negative_boundary`/`missing_field` drops above). **A naive left-join on `id` that falls back to the original `Gemma_S`/`Gemma_H` for unmatched rows will silently mix corrected and contaminated values in the same column.** Any downstream stage (P2 feature assembly, model training) that joins this file against the main store must either (a) drop the 3,577 unmatched `continue_it`/`continue_pt` rows entirely, or (b) explicitly flag them as uncorrected/contaminated rather than treating a missing join as "no correction needed." This is not yet decided or implemented — flagged here so it isn't missed when the join happens.

## Open item, carried forward rather than resolved

`continue_pt`'s CONFIRMED subset (from the earlier probe's diag21) showed `dH≈+0.0003` while `continue_it`'s CONFIRMED subset showed `dH≈−0.0185` — same nominal prefix length (~94 tokens both), opposite direction. This session's step 2/4 results don't resolve why; both types still show the same qualitative pooled/headline pattern (negative `dS`, gap shrinks but survives), so it doesn't change the verdict, but the mechanism difference between the two continuation types is unexplained and worth a note for anyone extending this analysis.

## Verdict

**Part I's headline Human-vs-LLM Fractal Gap claim survives, weakened but not overturned, at both tested correction variants, in all 5 domains and both generation models.** The contamination is real, was measured honestly (not rounded away), and the dilution that saves the headline number is a real property of the corpus composition (confirmed directly), not a hoped-for assumption.

This does not mean the underlying P1 feature store's **per-document** `Gemma_S`/`Gemma_H`/`Mistral_S`/`Mistral_H` columns for the 57,360 affected `continue_it`/`continue_pt` rows should be left uncorrected — those individual values are still measurably wrong (per-document `H` shift up to the noise-floor-confirmed 0.033 mean / 0.068 p90 reported earlier), just not wrong enough in aggregate to break the pooled, corpus-level claim the paper actually publishes.

**Status of the `SEAM_ADJ=2` prerequisite:** steps 1-5's pooled figures above still use the fixed `SEAM_ADJ=2` approximation and are reported as-is (not rerun), since the "Seam-offset prerequisite, closed" section already bounds its impact at two orders of magnitude below the smallest surviving gap margin. The actual per-document recompute (`src/recompute_family_a_continuation_fix.py`, previous section) does **not** use `SEAM_ADJ=2` — it implements the exact per-record formula derived in that section. So the shipped correction is more accurate than the verification numbers that justified running it, not less.

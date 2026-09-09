# P1 Feature Store — CPU Pass Progress Report

**Date:** 2026-07-31
**Scope:** Families A (fractal) + C (style/discourse) for the full GAGLE corpus, run locally on this machine. Family B (GLTR ranks, entropy, cross-perplexity) and Family C's embedding-dispersion sub-feature are deferred to a separate GPU-machine pass (`build_p1_feature_store_gpu.py`), per an explicit integrity-first design decision made before any implementation started (see "CPU/GPU split" below).

## Summary

`build_p1_feature_store_cpu.py` ran to completion against all 720 zip members (train+val+test), fully local, no network in the loop. Manifest confirms a gap-free run: `members_total: 720`, `members_done: 720`, `members_missing: []`, `complete: true`, `member_coverage_gap_free: true`, `json_errors_this_run: 0`.

- **Row count:** 230,530 documents written to `data/p1_feature_store_cpu.jsonl`, 62 columns, 519MB.
- **Zero duplicate ids** (verified directly: 230,530 rows, 230,530 unique ids).
- **Label breakdown:** gemma_2b 112,402 · mistral_7b 106,432 · Human 11,696.
- **Gemma-2b availability:** 228,600 of 230,530 rows have a valid (≥MIN_TOKENS) gemma-2b score array — this number matches P0's own final row count *exactly* (228,600), a strong independent consistency signal that P1's per-scorer gemma-2b gate reproduces P0's document set precisely.
- **Mistral-7b availability:** 230,530 of 230,530 (100% — every row that made it into the store had at least mistral-7b meet the threshold; 1,930 rows are missing gemma-2b specifically, 0 rows are missing both).
- **Family C word-budget fallback rate: 0%** — the per-document tokens-per-word ratio (used to length-match Family C's word budget to Family A's token window, avoiding a length confound between Human/LLM text — see design note below) was computable for every single document; no row fell back to full, untruncated text.

## Dropped/failed documents and why

From the run manifest (`data/p1_manifest.json`):

| Reason | Count |
|---|---|
| `raw_entries_seen_this_run` (total GAGLE records read) | 226,239 |
| `docs_written_this_run` (LLM-side + Human-side rows kept) | 230,530 |
| `skipped_dup_human_this_run` (Human doc repeats across prompt-type/temp files for the same domain — expected, not an error) | 214,543 |
| `skipped_no_scorer_meets_min_tokens_this_run` (neither gemma-2b nor mistral-7b array reached MIN_TOKENS=320 after slicing) | 7,397 |
| `json_errors_this_run` | 0 |

The `skipped_dup_human_this_run` count is large by design: each Human article recurs across every `prompt_type`/`temperature` file for its domain within a split, and is kept only once (deduped on `f"{split}_{domain}_{dataset_id}_human"` — the domain-scoped key fixed during P0 after finding 239 real documents were being silently dropped by a domain-unscoped key; that fix's key form is reused verbatim here, not re-derived).

## Reconciliation against P0 (exit-gate requirement)

Two different procedures, deliberately not one — averaging per-document point estimates is only a valid check for `H` (stable, unbiased at this token length); `S` is severely biased per-document (established this session) and can only be validly reconciled by re-pooling raw trajectories through the same list-of-trajectories `calc_S`, exactly as P0's own `compute_corpus_level_SH` does. Ran via `src/reconcile_p1_against_p0.py`, gemma-2b only (P0's only scorer):

- **H (via per-doc average from the feature store):** matches P0 exactly, `diff = 0.0000` in all 15 (domain, label) cells.
- **S (via a second, lightweight re-read of gagle.zip — scores only, no Family A/B/C recompute — re-pooling raw trajectories through `calc_S`'s list-of-trajectories mode):** matches P0 within `max |diff_S| = 0.0030` across all 15 cells — consistent with `calc_S`'s own bootstrap sampling randomness (`np.random.randint`, unseeded between runs), not a discrepancy.
- **Document counts matched P0 exactly in every one of the 15 cells** — strong evidence the extraction/dedup logic is identical between P0 and P1's gemma-2b path.

Full cell-by-cell numbers: `data/p1_reconciliation_pooled_SH.json`.

## Independent review

A fresh subagent, with no memory of this implementation, was tasked with (a) adversarially reviewing the extraction/scoring code for the specific bug classes already hit this project (unscoped dedup keys, off-by-one windowing, silently mismatched epsilon/scale, silent NaN drops) and (b) hand-recomputing several features for random real documents pulled directly from the zip (independent extraction code, not calling this pipeline's own functions) and diffing against the pipeline's actual output.

**The math is clean.** `calc_H`, the R/S algorithm, `compute_word_budget`, and `family_c_style` all reproduced *exactly* under independent reimplementation (Hurst exponent to 5 decimal places, `TTR`/`N_Words_Used`/`Sent_Len_Mean` bit-for-bit) across 15 hand-checked (document, feature) pairs, including 3 deliberately-targeted edge cases hitting the no-warmup-drop fallback branch. No stale-scale, mismatched-epsilon, or silent-NaN-drop bugs found anywhere in `build_p1_feature_store_cpu.py`, `fractal_math.py`, or the shared `build_p0_local.py` constants. The previously-fixed domain-scoped human dedup key (`f"{split}_{domain}_{dataset_id}_human"`) is applied consistently everywhere it's used.

**Two provenance findings, neither a math bug, both worth a decision before this store is used for classifier work:**

### Finding 1 — Human `original_text` scores are not interchangeable across `prompt_type` files, and the dedup key doesn't scope by it

Each `(model, temp, dataset, prompt_type)` zip member carries its own `prefix` field (e.g. `"(the following article contains a minimum of 15 paragraphs...)"` for `continue_it` vs. `"Using these keywords: ..."` for `keywords`). The `gemma-2b` score array for the SAME, byte-identical `original_text` string differs measurably depending on which member it's read from — confirmed directly on a real record (`scientific_papers`, `dataset_id=00005-of-00064__21`): identical 41,444-character text, but the `[64:464]`-windowed gemma-2b score array from the `excerpt` member vs. the `continue_it` member has **mean abs diff 2.97, max 16.6, Pearson r = 0.035** — essentially uncorrelated, not floating-point noise. (Exactly *why* — whether GAGLE re-scored under each prefix or something else — isn't determined from this repo; the consequence is what matters and is confirmed either way.)

The Human dedup key (`f"{split}_{domain}_{dataset_id}_human"`, used identically in P0 and P1) doesn't include `prompt_type` or `model`, so whichever member happens to sort alphabetically first for a given `(split, domain, dataset_id)` is the one silently kept — no flag records which one, and the choice isn't random: across a 200-key sample, `continue_it` wins 54% of the time, `cot`/`summary` 18% each, `summary_keywords`/`keywords` splitting the rest (whichever exists and sorts first for that record).

**Measured impact:**
- **Per-document:** across 50 sampled Human keys with ≥2 realizations (1,065 pairwise comparisons), `|ΔH|` between the kept realization and an alternate one has **mean 0.013, median 0.008, p90 0.029, max 0.055**. The tail is not negligible — comparable to some of the reported cross-domain H gaps.
- **Corpus-level (pooled):** for `newsroom|Human` (1,856 keys with ≥2 realizations), pooling from the alphabetically-*last* realization instead of the kept (alphabetically-*first*) one shifts the corpus-level estimate by only `ΔS=+0.0022, ΔH=+0.0008` — negligible next to the actual reported newsroom gemma_2b-vs-Human gap (`ΔS≈0.095, ΔH≈0.056`). **P0's pooled corpus-level table is not materially exposed** — pooling ~2,000+ documents washes out the per-document selection artifact. **P1's per-document Human `Gemma_H`/`Gemma_S` values are exposed** at the magnitude above, since nothing pools them before they're written to the feature store.

This is a design decision, not something fixed unilaterally here: whether Human documents should be deduplicated across `prompt_type` at all (vs. treating each `prompt_type`'s realization as its own valid, LLM-paired observation, the way LLM-side rows already are never deduplicated) is a question about the corpus's experimental design, not a bug in this code.

### Finding 2 — the no-warmup-drop windowing fallback is label-correlated

`slice_scores`'s fallback branch (used when a raw score array is shorter than `CLIP_TO=464` — keeps the whole array, warmup included, instead of dropping the first 64 tokens) is deliberate, consistently implemented across `build_p0_local.py`, `build_p1_feature_store_gpu.py`, and `build_p1_feature_store_cpu.py`'s word-budget logic — not a bug. Its real-world incidence wasn't previously measured; the independent reviewer estimated it from a partial pass (11,182 rows, 4.85%). **Backfilled exactly against the full corpus** (`src/backfill_short_window_flag.py`, `data/p1_short_window_flags.jsonl`, joined 1:1 against every one of the 230,530 feature-store ids, 0 missing) — the reviewer's *direction* was right, the *magnitude* undercounted:

| | Gemma short-window | Mistral short-window | Either scorer |
|---|---|---|---|
| Human | 0 | 0 | 0 |
| gemma_2b | 14,585 | 8,686 | 15,645 |
| mistral_7b | 4,839 | 3,992 | 5,708 |
| **Total** | **19,424** | **12,678** | **21,353 (9.26%)** |

100% of short-window rows are LLM-labeled, 0% Human — confirmed exactly, not just directionally, on the full corpus. `Gemma_Used_Short_Window`/`Mistral_Used_Short_Window` are now available per id in `data/p1_short_window_flags.jsonl` for downstream filtering/control, without needing to touch or re-run `p1_feature_store_cpu.jsonl`.

## Design decisions worth flagging explicitly

- **Windowed sub-document S/H (mean/var/min/max/drift across 200-token, 100-stride overlapping windows, `scale=[8,16,32,48,64,96]`):** validated against synthetic fBm ground truth before the full run. Windowed H's bias/variance is comparable to full-document H (bias +0.13 to -0.04 across H_true=0.3-0.8, std ~0.07). Windowed S is considerably noisier than even the already-unreliable per-document S (std 0.35-0.48, comparable to or exceeding the bias itself) — computed anyway per the P1 spec ("compute them anyway, P2 decides"), flagged via the existing `Holder_PerDoc_Unreliable=True` convention.
- **Family C word budget is per-document and ratio-derived, not full text.** A direct sample of real GAGLE records showed tokens-per-word ratio varies substantially by domain and, more importantly, systematically by side (LLM text is consistently shorter than matched Human text for the same record). Computing style features on full, untruncated text would make them a partial proxy for document length rather than a style signal — a length confound between the Human/LLM label and the features themselves. Fixed by converting Family A's token window into an equivalent word count using each document's own true tokens-per-word ratio (derived directly from data already available, no tokenizer dependency).
- **Two real bugs found and fixed during smoke-testing, before the full run:** (1) a naive `" ".join(text.split()[:budget])` truncation destroyed every newline, silently zeroing out paragraph-length variance for all 230,530 rows (`Para_Len_Std` was identically 0.0) and degrading sentence splitting on non-`.!?`-terminated paragraph breaks (common in `big_patent`/`billsum`) — fixed by truncating on the original string's character offsets, preserving structure. (2) discourse-marker frequency used substring counting (`"thus"` firing inside "enthusiasm") and a single lumped sum across 25 markers — fixed to word-boundary regex matching, split into 4 grouped categories (contrastive/additive/causal/exemplifying).
- **`calc_H` had a latent bug, fixed before any windowed computation was trusted:** it swept the module-level global `scale` regardless of any `scale_param` passed in, and built its regression's x-axis (`log_n`) from a front-truncated slice of that global list rather than the scales actually used — silently pairing R/S values against the wrong scale whenever a scale was dropped for a reason other than being the largest ones (only ever worked by luck on full 320-400 token documents). Fixed to accept an explicit `scale_param` and build `log_n` from the scales actually used, verified via a full fBm re-validation at both the original 11-scale and new 6-scale windowed configuration before proceeding.

## CPU/GPU split — what's deferred and why

Family B (GLTR rank buckets, exact next-token entropy, cross-perplexity) requires Gemma-2B/Mistral-7B forward passes with full logit capture — infeasible on this machine (no CUDA, no torch installed, limited unified memory). Confirmed with the user before any implementation started that splitting the work across two machines does not compromise data integrity, given three concrete safeguards: (1) the CPU pass emits a GPU-job manifest (`/Volumes/NikPhotoSSD/paper/gpu_manifest_shards/`, one parquet shard per zip member, 720 files + harmless macOS `._` metadata sidecars) carrying raw text + GAGLE's own sliced scores + both scorers' total token counts, so the GPU pass never re-derives its own tokenization/windowing decision; (2) `build_p1_feature_store_gpu.py` has a hard `--verify-alignment` gate (sweeping a residual index offset, since the exact HF checkpoint GAGLE used to produce its scores is not documented anywhere retrievable) that must pass before any full GPU run; (3) the join script (`build_p1_join_feature_store.py`) hard-asserts `set(cpu_ids) == set(gpu_ids)` before writing the final parquet, failing loudly rather than silently joining a partial set.

**The GPU script is untested** — written against a machine this session has no access to (no torch/CUDA here). `--verify-alignment N` is a mandatory first step on the GPU machine, not an optional nicety.

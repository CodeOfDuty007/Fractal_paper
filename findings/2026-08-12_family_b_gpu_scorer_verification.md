# Family B GPU-Scorer Verification: Bugs Found and Fixed Before the First Real Run

**Date:** 2026-08-12
**Scope:** First execution of `src/build_p1_feature_store_gpu.py` against real shard data (`--verify-alignment`), on the local Mac (16GB RAM, MPS), gemma-2b only. Task 3's condition from `reports/2026-08-06_headline_table_sensitivity_bound.md` (headline claim survives on sign in all 10 cells) cleared Family B to resume; this report covers what happened when it actually ran for the first time.

---

## Summary

The GPU-scorer script was written and syntax-validated last session but never run against real data. Every code path in it was therefore unverified. Running it — deliberately starting at N=5, not a full pass — surfaced **four real, independent defects**, three of them correctness bugs that would have silently produced wrong output on the actual GPU machine, and one a crash-causing memory bug. All four are fixed and verified. A stratified N=40 `--verify-alignment` run now reproduces the expected per-prompt_type pattern from `reports/2026-08-03_prompt_type_reconstruction_probe.md` closely enough to trust the alignment/candidate-selection/feature-window machinery.

**Mistral-7b is not covered by any of this.** Everything below is gemma-2b only — this Mac cannot load Mistral-7b (fp32 on non-CUDA devices, ~28GB, vs. 16GB total RAM). `--scorers gemma-2b` was added specifically so this machine could still validate the scorer-generic logic (which doesn't care which model runs it) without attempting to load a model that cannot fit. Mistral-7b's own `compute_boundary` derivation remains deferred exactly as reported in Task 2 (`reports/2026-08-06_headline_table_sensitivity_bound.md`) — not attempted here, not re-litigated.

---

## Bugs found, in the order they were caught

### 1. Mistral boundary-formula contamination (caught via advisor review, before running anything)

`compute_feature_window_prefix_text` called `compute_boundary` (Gemma's formula) unconditionally, regardless of scorer. Test 3A already established Gemma's formula does not transfer to Mistral (median `combined_delta` 35 tokens vs. Gemma's ~1, only 0.4% within ±3 tokens). Left as-is, a full run on the GPU machine would have silently written `Mistral_LogPPL_*` features for `continue_it`/`continue_pt` rows at the wrong span — reintroducing the exact prefix-contamination defect Family A was fixed for, on the Mistral side.

**Fix:** `BOUNDARY_CORRECTED_SCORERS = {"gemma-2b"}`, checked in `compute_doc_score_range`. Mistral `prefix_text` rows get no corrected feature window until Mistral's own formula exists (honest drop, not a guess).

### 2. `pick_best_candidate`'s blind `prefix_text` preference (caught via targeted diagnostic)

The original logic always kept `prefix_text` whenever both candidates cleared the alignment gate, reasoning that `text_alone` has no legitimate way to fit a genuinely prefix-contaminated `continue_it`/`continue_pt` document. Verified directly against real rows that this doesn't generalize: for Human `continue`-type rows whose winning dedup member came from a `continue_it` zip file, **both** candidates clear the gate, but `text_alone` is the genuine match (r=1.0000, mean_abs_diff≈0.01, offset landing exactly at `WARMUP_DROP+1`) while `prefix_text` is a spurious fit found by the ±150-token offset sweep on an autocorrelated series (r=0.998, mean_abs_diff≈0.07 — an order of magnitude worse, at a non-canonical offset ~153). The blind preference would have picked the wrong candidate and computed Family B features over the wrong span for this row class.

**Fix:** tiebreak on `mean_abs_diff` (the acceptance gate's own more discriminating metric — Pearson r is weak on autocorrelated surprisal series) instead of a label-based rule. Verified this doesn't regress the case the old rule was protecting: on genuine `continue_it`/`continue_pt` rows, `text_alone` doesn't narrowly lose the tiebreak, it fails the gate outright (r≈0.21, mean_abs_diff≈1.5 measured directly) — so the "both pass" branch this tiebreak governs never fires for the case it was originally worried about, and fires correctly for the Human case it missed.

*Note on process: a batch-composition discrepancy was initially suspected here (identical rows appeared to score differently across two runs) and treated as a possible severe batching bug. Bisected directly (solo/batch-of-3/batch-of-5, same row) — all three gave identical results. The apparent discrepancy was a misreading of the verify_alignment print order on my part (the "[note] BOTH candidates passed" line prints before its own row's id line, so in the transcript it visually attaches to the previous row). No batching bug exists. Recorded here so the false alarm doesn't get re-investigated later.*

### 3. Full-sequence logits computed instead of the scored window (caught via crash on stratified N=40)

`score_batch` called `model(input_ids=..., attention_mask=...)` without constraining `logits_to_keep`, so HF's default (`logits_to_keep=0`) projected **every** position in the padded sequence through `lm_head` (hidden_size → vocab_size≈256k), not just the ~700-1000 positions actually needed. With `MAX_CANDIDATE_TOKENS=2048` and a real batch, this is tens of GB. Crashed with `RuntimeError: Invalid buffer size: 10.80 GiB` on the first stratified N=40 attempt (N=5 had stayed small enough by luck — mostly short documents).

**Fix:** derive a single `logits_to_keep=K` per batch from the union of all documents' needed windows. Because every sequence is left-padded (real content right-aligned to the same edge), one shared K exactly covers every document's own `[start,end)` window regardless of individual document length — not an approximation. Regression-checked: the `continue_it` row scored identically (r=1.0000, mean_abs_diff=0.0098) before and after this change.

### 4. OOM-retry didn't recognize MPS's error string (caught by the same crash)

`score_documents_for_candidate` already had batch-size backoff logic for OOM, but it matched only `"out of memory"` (CUDA's phrasing). MPS raises `"Invalid buffer size: X GiB"` — a different string — so the retry never triggered and the exception propagated uncaught, crashing the whole run instead of halving the batch size as designed.

**Fix:** broadened the match to `MEMORY_ERROR_SUBSTRINGS = ("out of memory", "invalid buffer size")`. This is also relevant to the eventual CUDA run: if bug #3's fix ever under-covers on some document composition, the retry logic will now actually fire.

### 5. `verify_alignment`'s own sampler collapse (caught via advisor's explicit direction)

`verify_alignment(n_docs)` drained `n_docs` rows from `shard_files[0]` alone (sorted glob order). Since one shard = one (domain, prompt_type, temp, generating-model) combination, this can never sample more than one prompt_type — directly defeating the function's own documented purpose (comparing against the expected *per-prompt_type* pattern: human/continue_it/pt/keywords/summary pass, cot mostly fails, excerpt/short_keywords/summary_keywords never pass). This is the same sampler-collapse failure mode already caught once in Test 2 (`reports/2026-08-05_test1-5_anomaly_clearance_results.md`).

**Fix:** stratify across shards (≤1 row/shard, walking every shard until `n_docs` is reached) instead of draining the first one.

---

## Verification: stratified N=40, gemma-2b, after all fixes

40 documents, 1 per shard, walking shards in sorted order (covers 5 domains × 8 prompt_types at the earliest-sorted temp/split before repeating). No crash. Runtime dominated by model load (~10-20s); scoring itself completed in the same invocation.

| prompt_type | pass rate | expected (per 2026-08-03 probe) |
|---|---|---|
| continue_it | 3/5 | should pass |
| continue_pt | 5/5 | should pass |
| keywords | 5/5 | majority should pass |
| summary | 5/5 | majority should pass |
| cot | 0/5 | mostly fails |
| excerpt | 0/5 | essentially never passes |
| short_keywords | 0/5 | essentially never passes |
| summary_keywords | 0/5 | essentially never passes |

**Overall: 18/40 (45%).** Every category matches its documented expectation. `continue_pt`/`keywords`/`summary` pass at 100% (n=5 each); `cot`/`excerpt`/`short_keywords`/`summary_keywords` fail at 0% (stronger than "mostly"/"essentially never," consistent with those being small-n confirmations of an already-established near-total-failure pattern).

**`continue_it`'s 3/5 is a real, reproducible pattern, not noise -- checked directly, not assumed.** `billsum` (r=0.8909, mean_abs_diff=0.1533) and `wikipedia` (r=0.9689, mean_abs_diff=0.0418) fail at n=1/domain; both were re-examined with a same-prefix paired comparison against their `continue_pt` counterpart (identical underlying document position, confirmed byte-identical `prefix` string):

| domain | prompt_type | best_r | mean_abs_diff | best_offset |
|---|---|---|---|---|
| billsum | continue_it | 0.8909 | 0.1533 | 67 |
| billsum | continue_pt | 1.0000 | 0.0064 | 65 |
| wikipedia | continue_it | 0.9689 | 0.0418 | 65 |
| wikipedia | continue_pt | 0.9912 | 0.0443 | 65 |

Same prefix, same window logic, same everything except which model generated the continuation -- `continue_it` fits GAGLE's stored array measurably worse than `continue_pt` in both domains tested (24x worse `mean_abs_diff` on billsum; a real r-gap on wikipedia, 0.9689 vs 0.9912, enough to fail the gate where `continue_pt` doesn't). billsum's `continue_it` also lands at a non-canonical offset (67, not the usual 65). This is too consistent across 2/2 independently-tested domains to be sampling variance, though n=2 is not enough to fully characterize it. Plausible mechanism (not yet confirmed): instruction-tuned generations diverge structurally from a raw continuation (restated framing, formatting artifacts) even when scored against the identical prefix, whereas `continue_pt`'s more literal continuation doesn't. **This does not indicate a bug** -- the alignment gate is working exactly as designed: these two documents correctly get `Alignment_Passed=False` and no Family B features written, an honest drop, not silently-wrong output. But it means `continue_it`'s true corpus-wide pass rate may be measurably lower than `continue_pt`'s, not just "high with occasional noise" -- flagged as a specific, named item for the full-run's alignment-pass-rate audit, not a vague placeholder.

**The one visibly odd row (`newsroom`/`summary`, `best_offset=1` where every other passing row sits at 65) is benign, checked directly rather than assumed.** Its `align_window=(0, 458)` and `n_gagle=457` (vs. the usual 400) confirm it took the short-doc branch: `cand_len=458` is under `CLIP_TO`, so GAGLE never warmup-dropped/clipped this document in the first place -- its stored array starts at the very beginning, not at `WARMUP_DROP+1`. `best_abs_start=1` is the correct, expected result for that branch (`t_start = max(start,1) = max(0,1) = 1`), not drift or a bug.

**Read together, this confirms:** the gemma-2b checkpoint/tokenizer guess (`GEMMA_MODEL_ID = "google/gemma-2b"`) is correct -- a wrong checkpoint would have failed everything, not reproduced the expected per-type pattern this precisely, nor produced the internally-consistent it/pt paired gap above. The two-candidate/per-document-sweep/mean_abs_diff-tiebreak design is now verified against real data, not just self-reviewed. **This says nothing about `MISTRAL_MODEL_ID` = `"mistralai/Mistral-7B-v0.1"`, which is an equally unconfirmed best-guess and has not been checked at all** -- see "What this does and does not clear" below.

---

## What this does and does not clear

**Cleared:** the alignment/candidate-selection/feature-window logic, gemma-2b only, on this Mac, at N=40 stratified. Ready for a larger gemma-2b-only verification sample or a limited real `main()` run (`--limit-shards`) on this machine if useful before the Ubuntu GPU machine is available.

**Not cleared, and out of scope for this Mac:**
- **`MISTRAL_MODEL_ID`'s checkpoint identity itself is unconfirmed** -- it's an equally best-guess HF repo id (see the module docstring's "RUN --verify-alignment FIRST" warning) and, unlike `GEMMA_MODEL_ID`, has never been run against real data at all. If it's wrong (wrong repo, or a different revision/tokenizer version of the right one), the GPU machine will retokenize `text` into a different stream than GAGLE used and every Mistral alignment check will fail for a checkpoint reason, misdiagnosable as a prompt_type problem. This should be the FIRST thing checked on the GPU machine, before any real Mistral scoring.
- Mistral-7b's own alignment/candidate-selection behavior — untested here (cannot load the model). The scorer-generic code path is the same code Mistral will run through on the GPU machine, so this gemma-2b verification is evidence the *mechanism* is sound, but it is not evidence about Mistral-specific numbers.
- Mistral's `compute_boundary` derivation — still deferred, still blocked on hardware, unchanged from Task 2's status.
- `continue_it`'s true corpus-wide alignment pass rate relative to `continue_pt`'s — confirmed lower in the n=2-domain paired check above, not yet characterized at scale.
- A full-corpus run — this was a 40-document sample, not a scale test. Batch-size OOM-retry behavior under sustained load, checkpoint/resume correctness across many shards, and the true corpus-wide alignment pass rate per prompt_type all remain to be seen at scale, ideally on the GPU machine where memory pressure is a different regime than this Mac's 16GB.

---

## Verdict

Four real bugs (one correctness risk caught before running, three caught by actually running the code against real data) are fixed and verified, not just patched and assumed correct. The gemma-2b alignment mechanism now reproduces the established expected pattern closely enough to trust for a real run, including a genuine (checked, not assumed) `continue_it`-vs-`continue_pt` pass-rate gap that should be tracked, not smoothed over, in the full run's audit. Mistral-7b remains fully deferred, consistent with every prior report's treatment of that blocker — nothing here should be read as resolving it, and its checkpoint identity specifically has not been verified at all, which should be the first thing checked when the GPU machine is available, before any real Mistral scoring is trusted.

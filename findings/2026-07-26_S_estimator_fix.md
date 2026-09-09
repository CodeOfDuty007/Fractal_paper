# S/H Estimator Fix — Progress Report

**Date:** 2026-07-26/27 (multiple revisions same continuous session; the version below is final)
**Scope:** `src/fractal_math.py` correctness fix, validated against the paper's reference implementation. No modeling, no new phases.

## Summary

The repo's `calc_S` was broken, but not for the reason initially assumed. Two false starts are recorded below because they're informative, not because they were right. The real, reference-confirmed finding is at the end: **per-document `S` is not statistically recoverable at ~400 tokens, regardless of estimator sophistication.** This is a substantive result for Extension 1, not just a bug fix.

## What the reference implementation actually does

The `andsteing/gagle` HF Space (the P0 exit-gate reference named in `Extension1_PhaseWise_Execution_Plan.md`) is a static JS app with no Python source, but its `bundle.js` contains the real algorithm. Decompiled and verified directly (not assumed):

- **`computeXx(scores)`**: `mean`/`std` of the raw score array, then `(x - mean) / std`, then `cumsum`. This is a z-score of the raw token-surprisal sequence before integrating — exactly what the pre-fix `generate_trajectory` did.
- **`getSelfSimExponent(scale, integral, n_boot, epsilon)`**: takes one or many trajectories, flattens and divides by the **pooled trajectory's own stddev** (a second, separate normalization from the first), then for each scale draws `n_boot` **random** (not exhaustive) start offsets per trajectory, computes `p_epsilon(tau) = P(|displacement| < epsilon)`, and fits `S = -slope` of `log(p)` vs `log(tau)`.
- **The UI's actual call site** (`idx0Select`/`idx1Select` document-range pickers): concatenates raw `token_scores` from **multiple documents into one long array**, then calls `computeXx` **once** on the concatenation — never per document.

## False start #1: blaming z-scoring

The repo's original bug looked like: z-score the trajectory, then threshold a fixed `epsilon=0.1` against it, getting `S`-fit R²≈0.31 corpus-wide with no Human/LLM separation. The initial diagnosis was that z-scoring "decoupled epsilon from the trajectory's real scale" and should be removed (mean-center only, don't divide by sigma). This produced self-consistent, plausible-looking numbers (R²≈0.83, then ≈0.996 after further work) — **and was wrong**. The reference's `computeXx` does exactly the z-scoring that was removed. The real defect was that the repo estimated `p_epsilon(tau)` from **exhaustive sliding-window pairs on one 400-token document** (observed directly: 3-of-272 hits at `tau=128`, 0-of-80 at `tau=320` — Poisson noise, not a power law), while the reference draws bootstrap samples and is designed to pool many documents.

## False start #2: a moment-based estimator with a bridge correction

Believing the estimator itself needed replacing, `calc_S` was rewritten to fit `E|X_{t+tau}-X_t| ~ tau^S` (using every displacement, not a threshold count) on the mean-centered (not z-scored) trajectory. Validated against synthetic fBm with known Hurst exponent, this exposed a real, separate phenomenon — mean-centering a trajectory pins it to ~0 at both endpoints (a "bridge"), deflating large-`tau` displacement by a factor of `sqrt(1-tau/n)` — and a truncated-scale + analytic-correction fix brought bias to −0.036 at `H≈0.65`. This is a genuine, documented finding about mean-centered trajectories, kept here for anyone who hits the same bridge effect elsewhere. **It is not what the reference implementation does or needs**, since the reference never mean-centers-only; it fully z-scores, which doesn't produce a bridge in the same way.

## What's actually true, verified against the reference (this is step 6 of the P0 plan)

`calc_S` was reverted to the reference's exact recipe (z-score in `generate_trajectory`, second normalization + bootstrap threshold-counting in `calc_S`, restored full 11-value `scale`, `epsilon=1e-2`). Then validated against synthetic fBm with known Hurst exponent in four stages:

| Test | Setup | Result |
|---|---|---|
| 1. Per-document, `n_boot=10` (the plan's literal "bootstrap-10") | 400-token synthetic fGn | **`S` = NaN** — with only 10 random draws per scale and `epsilon=1e-2` on a doubly-normalized trajectory, essentially no draw ever lands under threshold. |
| 2. Pooled 200 docs, each individually z-scored then combined, `n_boot=10` | 200×400-token fGn | Finite but biased: −0.03 at `H=0.3`, growing to **−0.36 at `H=0.8`**. |
| 3. `n_boot` sweep (10→5000) on one long 80,000-point trajectory | Single continuous fGn | Bias collapses from −0.25/−0.32 (at `n_boot=10`) to **−0.005/+0.003** (at `n_boot=5000`) for `H=0.3/0.5`; R² rises from 0.13–0.65 to 0.89–1.0. **`n_boot=10` was pure sampling noise, not a structural flaw.** |
| 4. Single 400-token doc, `n_boot=1000–5000` (adequate draws, short window) | 400-token fGn | Bias **−0.26 to −0.48**, R² 0.08–0.38, regardless of `n_boot`. More draws do not fix a short window. |
| **5. Concatenate 200 separate 400-token docs' raw scores, `computeXx` once, `n_boot=2000`** | Matches the reference's actual real-world usage (document-range concatenation) | **Bias −0.004 to −0.065 across `H=0.3–0.8`, R²=0.91–0.995.** |

Test 5 is the P0 exit-gate result: **the reference's algorithm is correct and matches ground truth when applied the way the reference actually applies it — concatenated across many documents, not per document.** Tests 1, 2, and 4 show that per-document application of the *same, correct* algorithm is not statistically viable at 400 tokens: there isn't enough information in one document's token sequence for `p_epsilon(tau)` to resolve a power law, no matter how the sampling is done.

**Correction (found later, see "Pooling-mode bias" below): Test 5 validates a *different pooling mode* than the one `build_p0_local.py` actually runs at corpus scale.** Test 5 concatenates raw scores across documents first, then normalizes once (one long trajectory, boundary-crossing allowed). Production `calc_S`/`compute_corpus_level_SH` normalizes each document separately first, *then* pools the resulting trajectories as a list (boundary-respecting sampling — the mode that matches the true original paper's code, not this reference JS's UI shortcut). These two modes are not interchangeable and have different bias profiles; Test 5's −0.004 to −0.065 figure does not describe the mode that produced the corpus-level table below. The actual bias of the production mode is characterized later in this report and is larger.

`calc_H` needed no changes anywhere in this process and was re-verified at every stage (R²≈0.98, same small pre-existing low-`H` bias throughout) — its R/S computation is invariant to the z-scoring question entirely, since ratios are scale-invariant to linear rescaling of the increment.

## Code changes

- `generate_trajectory`: reverted to `(scores - mean) / std` then `cumsum`, matching the reference's `computeXx` exactly.
- `calc_S`: now accepts either a single trajectory or a list of trajectories (for corpus-level pooling); applies the reference's second normalization (divide by the pooled trajectory's own stddev); estimates `p_epsilon(tau)` via `n_boot` random bootstrap draws per scale instead of exhaustive sliding-window pairs; restored the full 11-value `scale` and `epsilon=1e-2` default.
- `N_BOOT` set to 2000 (not the plan's literal "10", which the evidence above shows refers to a separate outer CI-resampling loop, not per-scale draws).
- `get_allMetric` (the per-document entry point used by both the local pipeline and `build_p0_corpus.py`) now includes `"Holder_PerDoc_Unreliable": True` in its output, since it necessarily calls `calc_S` on one document — the regime just shown to be biased by −0.26 to −0.48. **Every `Holder_Exp` value in `data/new_analysis_results_v2.jsonl` (202,749 rows) and `data/p0_full_results.jsonl` (221,773 rows so far) was computed in this per-document regime and should not be treated as an accurate per-document estimate.** `Hurst_Exp` in the same files is unaffected and trustworthy.

## Correction to the earlier Part I/II exposure note

An earlier version of this report speculated that Part I's "Fractal Gap" finding might rest on a broken estimator. That's now known to be more specific: **the pre-fix trajectory z-scoring was correct** (matches the reference). What was wrong was estimating `p_epsilon(tau)` per-document via exhaustive pairs. If Part I/II's original pipeline (or `build_features.py`, not in this repo) also computed `S` per-document — which `context.md` and the visualizer's per-row `Holder_Exp` schema strongly suggest — then those results are exposed to the same per-document bias just demonstrated, independent of whichever exact threshold/sampling method was used. This is still not resolved from this repo alone (`build_features.py` doesn't exist here) but the shape of the risk is now precise rather than speculative.

## Consequence for Extension 1

This is a substantive, early answer to a question the phase plan defers to P2: **"are per-document `S`/`H` stable enough to detect on?"** (`Extension1_PhaseWise_Execution_Plan.md`, Phase P2). For `S` specifically, at 400 tokens, the answer is no — not "noisy," but biased by up to half a Hurst-unit in a fixed direction. `Extension1_Technical_Plan.md` §2 pre-registers exactly this contingency: if reliability fails, "pivot: drop raw `Ŝ`/`Ĥ` point features, keep only stability-profile features (variances, residuals, drift, scorer-divergence)." This finding is evidence for that pivot on the `S` side specifically. `H` (Hurst, via R/S) shows no equivalent per-document pathology in any test run today and remains a candidate point feature.

## Where this leaves P0

Per the user's stated gate — "P0 is done only when 3, 5, and 6 all pass":

- **Step 5 (fBm unit tests):** done, with the full four-stage validation above superseding the earlier single-pass version.
- **Step 6 (numerical cross-check against `andsteing/gagle`):** done, with a substantive result (concatenated-corpus regime confirmed correct; per-document regime confirmed biased).
- **Step 3 (full GAGLE ingestion via `build_p0_corpus.py`):** parked, not failed. 221,773 rows ingested (~10.3% of GAGLE's own file-shard count: `billsum` complete, `wikipedia`/`big_patent` only 6.25% in) before hitting a reproducible stall. Root cause found: `ibomohsin/gagle` is a single 10.6GB `gagle.zip`, and `.shard()` was doing byte-range HTTP reads into different offsets of that one remote archive — not splitting multiple files, since there's only one. No `--num-shards` value fixes this; a byte range that hangs at offset X hangs identically on every retry. **Real fix needed: download `gagle.zip` locally once (Path B in `Extension1_P0P1_Laptop_Addendum.md` §2), then read shards from the local file with no network in the loop.** Blocked on disk space: `gagle.zip` needs 10.63GB; only 6.3GB is currently free. Deleting all redundant old data files (`new_fractal_scores.jsonl` 1.7GB, `new_analysis_results.jsonl` 57MB, `fractal_scores.jsonl` 246MB, `analysis_results.jsonl` 7.8MB — roughly 2GB total) still leaves an ~2.3GB shortfall, so either something outside this project's `data/` directory needs to free space, or the remote partial-read approach (Path A) needs a different fix than resharding. **Needs your decision — nothing has been deleted.**

## What's next

1. **Your call:** how to free enough disk for the local `gagle.zip` download (or an alternative ingestion strategy), and whether to delete the now-fully-superseded old data files.
2. Once ingestion resumes and completes: rerun `get_allMetric` is unaffected by this decision (the `N_BOOT`/reference-matching fix is already in place) — just needs the corpus.
3. Decide P1/P2's fractal-feature whitelist in light of this session's finding: `Holder_Exp` per-document is not currently a trustworthy point feature; `Hurst_Exp` is unaffected.
4. If pursuing corpus-level `S` (the regime that actually works): would need a concatenation-based feature (Family A's "windowed mean/var/drift" language in `Extension1_PhaseWise_Execution_Plan.md` P1 may already anticipate this) rather than the current one-`S`-per-document schema.
5. Resolve the Part I/II exposure question against `build_features.py` if/when it's located.

---

## Continuation: full-corpus local-zip pipeline (publication push)

Decision made: pursue full ingestion for publication rather than the partial pull. `ibomohsin/gagle` was confirmed (via `HfApi().list_repo_files()`) to be a single 10.63GB `gagle.zip` — the earlier streaming-`.shard()` approach was doing byte-range HTTP reads into one remote archive, which is why it stalled at the same offset on every retry regardless of `--num-shards`. Fix: download `gagle.zip` locally once (to an external SSD, `/Volumes/NikPhotoSSD/paper/`, since internal disk is tight), then read members from the local file with no network in the compute loop.

### Found the actual original reference code, not just a demo notebook

Two additional reference sources were located this session, both more authoritative than the `andsteing/gagle` HF Space's decompiled `bundle.js` used for the earlier fBm cross-check:

1. **`GAGLE_Dataset.ipynb`** (in the `ibomohsin/gagle` repo itself) — confirms the zip's internal layout (`gagle3/{split}/{model}_{temp}_{dataset}_{prompt_type}.jsonl`, one member per stratification cell) and that `scores[scorer][text_side]` is a flat list of floats (matching what was already verified live against a streamed row — the nested `token_ids`/`token_texts`/`token_scores` shape in the HF dataset-card README describes a different, unrelated viewer schema).
2. **`fractal_patterns_in_language.ipynb`** (google-research/google-research, arXiv:2402.01825, "Fractal Patterns May Unravel the Intelligence in Next-Token Prediction" — the precursor paper the GAGLE notebook explicitly says its own fractal code was "copied/modified from"). This is the actual original authors' code, fetched directly.

Key findings from source 2:

- **Our per-trajectory `calc_S` design (list of separate trajectories, sampling within each, never crossing document boundaries) matches the ORIGINAL paper's methodology**, not just an independent improvement over it. The original `get_self_sim_exponent`/`get_hurst_exponent` take a 2D (docs × tokens) array and sample offsets *within* each row. It was the GAGLE demo notebook's own shortcut (concatenating all documents into one flat 1D array before sampling) that deviated from this — letting a random `(offset, offset+tau)` pair straddle a document boundary and measure a meaningless discontinuity jump. Our code was already doing it the more correct way.
- **Epsilon is not one universal constant.** Three sources now give three different values: `bundle.js` defaults to `1e-2` (exposed as a UI slider, so not fixed there either); `GAGLE_Dataset.ipynb`'s demo uses `0.1`, noting explicitly this is "larger than in the paper" because its concatenation method differs; the *original* paper's own colab uses `5e-2` as an "illustration only" default on 896–1024-token documents with scales up to 300, explicitly stating even that differs from what the real paper experiments used. Conclusion: epsilon must be calibrated to this corpus's own document-length/scale-range ratio (our 400-token docs, scales up to 320), not copied from any one source — which is exactly what the original task's step 2 asked for and had not yet been executed. `fractal_math.py`'s `EPSILON` comment has been corrected to state this; **the calibration sweep itself is pending the zip download finishing** (see below).
- **A third exponent (Joseph, `Σ(n) ~ n^J`) exists in the original paper and is not implemented here.** Confirmed out of scope: the paper this repo is actually based on is titled "A Tale of *Two* Structures," i.e. it deliberately drops Joseph and keeps only H and S.
- The original code also computes an **outer 10-fold bootstrap resample** for a confidence interval on the exponent itself (distinct from the inner per-scale `n_boot` draws — this is what `Extension1_PhaseWise_Execution_Plan.md`'s "bootstrap-10" language actually refers to, confirming a distinction already suspected). Not implemented; noted as a follow-up enhancement for publication-grade uncertainty reporting, not a correctness gap.

### `calc_H` re-validated against this same original algorithm

The original `get_hurst_exponent` draws `samples_per_doc` *random, overlapping* windows per scale; this repo's `calc_H` partitions into non-overlapping `len//n` sequential chunks — a legitimate concern, since at the largest scales (256, 320) a 400-token document yields only 1 chunk. Tested directly: 30 independent synthetic-fBm trials at `H = 0.3/0.5/0.65/0.8` show mean bias between the two estimators agreeing within ~0.01 at every `H` — well inside each estimator's own run-to-run noise. **The chunking difference changes variance, not bias.** `calc_H` needed no change.

### Pool-size matching is no longer required

An earlier prototype (run against the now-deleted `new_fractal_scores.jsonl`) needed matched pool sizes (200 docs/cell) because the *previous* (now-superseded) `calc_S` implementation's second normalization broke down under a large size mismatch (1,000 Human docs vs ~20,000 LLM docs gave degenerate `S≈0`, `R²=NaN` for the larger pool). Retested against the *current* per-trajectory `calc_S`: two pools of 200 vs 20,000 documents, both drawn from an identical synthetic process (`H=0.5`), gave `S=0.311` and `S=0.334` respectively — close agreement, both well-fit (`R²=0.79, 0.88`). The degenerate behavior was an artifact of the replaced estimator, not a property of the corpus-level method itself. The new pipeline pools all available documents per `(domain, label)` cell without subsampling.

### New pipeline: `src/build_p0_local.py`

Replaces `build_p0_corpus.py`. Reads `gagle.zip` locally via `zipfile`, no network in the read loop. Design:

- **Checkpointing is per zip *member*** (`gagle3/train/{model}_{temp}_{dataset}_{prompt_type}.jsonl`), not per logical shard/doc — each member is the dataset's own natural stratification-cell unit. A member is atomic: if the process dies mid-member, the whole member is redone on restart (cheap, local, no network) rather than tracking per-document progress within one.
- **Per-doc output carries `Hurst_Exp` only** — `Holder_Exp` is dropped entirely from per-document rows (not just flagged) since it's confirmed unreliable at this token length (bias −0.26 to −0.48, see above).
- **Corpus-level `S`/`H` per `(domain, label)` cell** is computed once, after every member is done, from pooled raw scores accumulated in memory during the pass (not persisted to disk a second time; a resume replays already-done members — cheap local reads — to rebuild the in-memory pools rather than caching them).
- Smoke-tested end-to-end against a synthetic fake zip mirroring the real structure: checkpointing, resume/replay, human-doc dedup (by `dataset_id`, since the same human article recurs across every prompt-type/temperature file for a domain), and corpus-level computation all verified correct. Corpus-level `S`/`H` values on the fake data landed in the expected plausible range (`S≈0.40–0.44`, `R²=0.86–0.94`, `H≈0.63–0.65`, `R²≈1.0`).

### Download saga: a second, distinct silent-hang bug

The first download attempt (`hf_hub_download(..., local_dir=...)`, wrapped in a bash watchdog that killed and relaunched it on apparent stall) made things *worse*, not better. Root cause: `huggingface_hub` keys its `.incomplete` partial-download path partly on a per-process-attempt suffix; killing and relaunching the process orphaned the previous partial file and started a new one each time. Over several kill/restart cycles this left 17 abandoned partial files (0B–1.7GB) on the SSD, with **zero net progress** past ~1.7GB despite hours of wall-clock time — the watchdog's own stall-detection was also polling the wrong (stale) file via an unqualified `find | head -1`, worsening the problem rather than catching it.

Fixed by writing a self-contained downloader (`src/download_gagle_zip.py`) that bypasses `huggingface_hub`'s download machinery entirely: plain `requests`-based HTTP Range GETs against a fixed local path (`gagle.zip.part`), resume driven purely by that file's own on-disk size, with an explicit read-timeout on the streamed response (converts a silent stall into a catchable `ReadTimeout`/`ConnectionError`, so the retry loop is fully internal — no external watchdog needed, and the earlier bash watchdog script was deleted). Guards added: refuses to append if the server responds `200` instead of `206` to a Range request (would silently corrupt the partial file by prepending the whole object again); verifies the final file's size against the repo's reported size and runs `zipfile.testzip()` (full CRC check) before trusting it, deleting and forcing a clean re-download on any mismatch. The largest orphaned partial (1.73GB) was verified byte-for-byte against a fresh ranged fetch of the same prefix before being reused as the new downloader's starting point, rather than discarded.

### Download completed and verified

`gagle.zip` (10,633,702,946 bytes, matching the repo's reported size exactly) downloaded successfully and passed full `zipfile.testzip()` (every member's CRC checked). The zip contains 720 `gagle3/{split}/*.jsonl` stratification files, not 240: GAGLE ships **three** splits (`train`, `val`, `test`, 240 files each), not just `train` as earlier pipeline stages (`hf_extractor.py`) used. `build_p0_local.py`'s member pattern (`gagle3/*/*.jsonl`) already matched all three by construction, which is the right call — a proper held-out test split is valuable for the classifier work Extension 1 plans — but the per-doc output didn't originally record *which* split a row came from, and GAGLE's `dataset_id` numbering resets per split (the same id string can name a different article in `train` vs `val` vs `test`). Fixed before running anything real: `split` is now threaded through `extract_rows`/`process_member` into both the row `id` (for human-doc dedup, scoped per-split) and the output row itself, so the original train/val/test partition can be reconstructed later.

### Epsilon calibration result (real data, not synthetic)

Ran the sweep on 2 real zip members (`gagle3/train/gemma_2b_0.0_big_patent_continue_it.jsonl` + `..._continue_pt.jsonl`, 1288 gemma_2b docs + 1400 deduped Human docs, pooled per the corpus-level method):

| epsilon | gemma_2b S | gemma_2b R² | Human S | Human R² |
|---|---|---|---|---|
| 0.01 | 0.5395 | 0.9289 | 0.4216 | 0.8883 |
| 0.05 | 0.5364 | 0.9245 | 0.4264 | 0.8992 |
| 0.10 | 0.5229 | 0.9209 | 0.4232 | 0.9081 |

**Decision: keep `epsilon=1e-2`.** It gives the best fit for gemma_2b (R²=0.929) and a competitive fit for Human (R²=0.888, within 0.02 of the best); none of the three tested values is a dramatic outlier, and the qualitative finding (LLM S > Human S) holds at every epsilon tested. Human `S≈0.42` also lands almost exactly on one of the two worked examples the original task plan cited (`S≈0.42, 0.63`) as a sanity check — a real, non-synthetic confirmation, not just an internally-consistent number.

### Full run — complete, then a real bug found in post-hoc verification, then corrected

`build_p0_local.py` ran to completion against all 720 members (`train`+`val`+`test`), fully local, no network in the loop. Manifest confirmed the P0 gate on the first pass: `members_total: 720`, `members_done: 720`, `members_missing: []`, `complete: true`, `member_coverage_gap_free: true`, `json_errors_this_run: 0`. But manifest/coverage checks only confirm every member was *read* — they don't confirm the extraction logic *within* each member is correct. A post-hoc verification pass found a real bug before treating the first run's numbers as final:

**Bug**: the human-doc dedup key was `f"{split}_{dataset_id}_human"` — it did not include `domain`. GAGLE's `dataset_id` numbering (e.g. `00013-of-00256__513`) is **not** domain-scoped: checked directly against the full zip, 239 of 11,457 `(split, dataset_id)` keys are shared across two different domains — 139 between `big_patent`/`wikipedia`, 100 between `newsroom`/`scientific_papers` (`billsum` was unaffected). Without domain in the key, whichever domain's row happened to be processed first for a colliding key silently "stole" it, and the other domain's actual, different human article was skipped as a false duplicate — an undercount, not a wrong value, but a real one.

**Fix**: added `domain` to the key (`f"{split}_{domain}_{dataset_id}_human"`), cleared the buggy checkpoint/output (moved to `data/superseded_2026-07-30/`, not deleted), and reran the full 720-member ingestion from scratch. The corrected run wrote exactly 228,600 rows vs. the buggy run's 228,361 — **+239, matching the collision count exactly** — and the per-domain gains split as +12 (big_patent) / +127 (wikipedia) = 139, and +20 (newsroom) / +80 (scientific_papers) = 100, again matching exactly. This precise accounting, with zero unexplained residual, is itself strong evidence the fix is complete and correct, not just plausible.

Both runs' final per-doc output was also checked directly for duplicate ids: 0 duplicates in both the 228,361-row and 228,600-row files, confirming the buffered-write-per-member design (write all of a member's rows in one shot only after it's fully read) does prevent the partial-write-then-redo duplication risk it was designed for.

- **Per-document output**: `data/p0_local_full_results.jsonl`, 228,600 rows, `Hurst_Exp`/`Hurst_R2` only (no per-doc `Holder_Exp` — see above for why), each row tagged with `split`, `domain`, `Label`, `prompt_type`, `temp`.
- **Corpus-level S/H** (corrected, final): `data/p0_corpus_level_SH.json`, one row per (domain, label) cell, pooling every document across all three splits for that cell (no subsampling needed — see the pool-size finding above):

| domain | label | n_docs | S | S R² | H | H R² |
|---|---|---|---|---|---|---|
| big_patent | Human | 2,418 | 0.4176 | 0.875 | 0.6395 | 1.000 |
| big_patent | gemma_2b | 22,692 | 0.4780 | 0.899 | 0.6916 | 0.998 |
| big_patent | mistral_7b | 21,092 | 0.4976 | 0.901 | 0.7127 | 0.998 |
| billsum | Human | 2,169 | 0.4239 | 0.931 | 0.6337 | 1.000 |
| billsum | gemma_2b | 22,374 | 0.4578 | 0.907 | 0.6688 | 0.999 |
| billsum | mistral_7b | 21,355 | 0.4509 | 0.898 | 0.6869 | 0.999 |
| newsroom | Human | 2,370 | 0.3696 | 0.840 | 0.6072 | 1.000 |
| newsroom | gemma_2b | 22,231 | 0.4650 | 0.904 | 0.6634 | 0.998 |
| newsroom | mistral_7b | 21,097 | 0.4889 | 0.917 | 0.6903 | 0.997 |
| scientific_papers | Human | 2,458 | 0.4044 | 0.907 | 0.6091 | 1.000 |
| scientific_papers | gemma_2b | 21,752 | 0.4689 | 0.907 | 0.6886 | 0.998 |
| scientific_papers | mistral_7b | 20,594 | 0.4994 | 0.911 | 0.7144 | 0.997 |
| wikipedia | Human | 2,281 | 0.4017 | 0.875 | 0.6245 | 1.000 |
| wikipedia | gemma_2b | 22,292 | 0.4449 | 0.903 | 0.6565 | 0.999 |
| wikipedia | mistral_7b | 21,425 | 0.4676 | 0.903 | 0.6762 | 0.998 |

The correction only moved the affected Human cells by a few thousandths of S (e.g. wikipedia Human S: 0.4035 → 0.4017) — small in magnitude, but the *mechanism* (silently dropping real documents) is exactly the kind of error that would otherwise sit undetected in a published number. All LLM-label cells were unaffected (only human-side dedup was buggy) and match the pre-fix run exactly.

**The paper's central finding replicates cleanly on the full, corrected corpus, on both S and H, in every domain, for both scorer models**: LLM-generated text has a higher self-similarity exponent (S) and higher Hurst exponent (H) than the matched human text in the same domain — never the reverse, in any of the 15 cells. Fit quality is strong throughout (S: R²=0.84–0.93; H: R²=0.997–1.000).

**Fig. 10's "narrow Human band vs. wider LLM spread" sub-claim does NOT replicate here — it's inverted, and this survives the correction.** Spread (max − min) of S across the 5 domains, corrected values: Human 0.0543 (0.3696–0.4239), gemma_2b 0.0330 (0.4449–0.4780), mistral_7b 0.0484 (0.4509–0.4994). Human is the *widest* of the three groups, not the narrowest — same conclusion as the pre-fix numbers, just slightly different values. This is a real discrepancy with the paper's framing, not a rounding artifact, and should be treated as a finding to investigate (different corpus, different domain set, different epsilon/scale calibration than the original Fig. 10 setup are all candidate explanations) rather than glossed over.

This is now the trustworthy, full-corpus, reference-validated, bug-corrected basis for Extension 1's corpus-level fractal features. Per-document `Hurst_Exp` (228,600 values) is also available and trustworthy per the fBm validation above; per-document `S` remains excluded by design.

### Pooling-mode bias: the production `calc_S` mode is more biased than Test 5 suggested, but it's an estimator property, not a bug

While vectorizing `calc_S`'s inner bootstrap loop for P1 throughput (a 44x speedup needed for ~1.8M projected calls, pure performance change, no logic change), a re-run of the fBm ground-truth validation using the **actual pooling mode `compute_corpus_level_SH` uses in production** — normalize each document separately, keep the results as a list of trajectories, sample within each row (never crossing document boundaries) — showed markedly worse bias than Test 5's reported −0.004 to −0.065:

| H_true | Production mode (list-of-trajectories, 2nd normalization), 8-trial mean | std across trials |
|---|---|---|
| 0.30 | bias −0.072 | 0.020 |
| 0.50 | bias −0.160 | 0.023 |
| 0.65 | bias −0.231 | 0.020 |
| 0.80 | bias −0.310 | 0.028 |

Two things were checked before concluding anything:

1. **Not a vectorization bug.** A hand-written, unvectorized reference implementation on identical seeded data reproduced the same bias to within noise.
2. **Not caused by `calc_S`'s second normalization step either.** The true original paper's own code (`fractal_patterns_in_language.ipynb`, cells 34/38–40 — confirmed by direct inspection, not the GAGLE HF Space demo) builds trajectories per-document into a genuine 2D array and calls `get_self_sim_exponent` with **no second normalization at all**. Reimplementing that exact function and running the same 8-trial fBm test gives bias −0.072/−0.180/−0.255/−0.241 at the same four `H_true` values — comparable magnitude to production, and *noisier* (std up to 0.127 vs. production's 0.028). The second normalization in `calc_S` is not the source of the bias, and removing it would trade a documented, stable bias for a similarly-sized, less stable one while diverging from the reference's second-normalization design. **`calc_S` is unchanged.**

Conclusion: **the bias is an inherent property of threshold-counting `p_epsilon(tau)` self-similarity estimation on finite-length pooled samples, present in the reference's own exact algorithm, not a defect introduced by this repo's code.** It grows with the true exponent. This does not overturn the corpus-level table above, for two reasons:

- **Pool size doesn't drive it** (already shown earlier in this report: 200 vs. 20,000-document pools at the same `H_true` gave S=0.311 vs. 0.334 — close agreement). Every cell in the table above pools 2,000+ documents, well past where pool size matters.
- **The bias is conservative for the paper's central claim.** Every LLM cell has a higher `H` than its matched Human cell in the same domain (see `H` column above), and bias grows with `H_true`. So the *true* LLM−Human `S` gap is being *understated*, not manufactured, by this effect — if anything the real separation is larger than reported.

What does not survive: treating the table's absolute `S` values as calibrated, paper-comparable Hurst-equivalent numbers. They are internally consistent (safe to compare Human vs. LLM, domain vs. domain, within this corpus) but should not be read as "the" Hölder exponent of that domain's text. This should be stated as a limitation in any write-up that reports these numbers, rather than implied to be bias-free.

### Housekeeping (not yet done — needs your go-ahead)

Now fully superseded and safe to delete, but not deleted without confirmation:
- `data/p0_full_results.jsonl` (221,773 rows from the old streaming pipeline, stale `Holder_Exp`)
- `data/new_analysis_results_v2.jsonl` (202,749 rows, same staleness)
- `data/superseded_2026-07-30/` (the buggy 228,361-row v1 of this session's own run — kept as a paper trail, but redundant now that v2 is verified)
- `src/build_p0_corpus.py` and `src/run_p0_corpus_watchdog.sh` (superseded by `build_p0_local.py`; the watchdog script for the download itself was already deleted mid-session as it was actively harmful)

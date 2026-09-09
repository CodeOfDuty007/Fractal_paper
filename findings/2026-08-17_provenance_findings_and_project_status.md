# Score-Array Provenance Findings + Project Status

**Date:** 2026-08-17
**Scope:** What today's GPU-side alignment testing established, what it means for
each standing claim, and where the project actually is. Numbers below are
measured this session on `ubuntu-S2600STB` (A100 80GB); nothing here is
restated from memory or from the bundle README.

---

## Bottom line

Today produced the **first positive provenance verification** this project has
ever had for GAGLE's precomputed token scores — and, in the same pass, evidence
that the verification **does not extend to three of the four
(scorer × generator) quadrants**.

- `gemma-2b` reproduces GAGLE's stored gemma scores on gemma-generated
  documents at **r = 1.0000, mean_abs_diff ≈ 0.007**. That is not "close" —
  that is the same numbers.
- No other combination reproduces. Not the mistral scorer anywhere, and not
  the gemma scorer on mistral-generated rows.

This is a **provenance** result, not a correctness result. Family A consumed the
stored arrays as-is; arrays we cannot reproduce are not thereby shown to be
wrong. But "verified reproducible" and "assumed correct" are different epistemic
standing, and the paper should not conflate them.

---

## 1. Where the project stands

Per `Extension1_PhaseWise_Execution_Plan.md`, mapped against what exists:

| Phase | Status | Note |
|---|---|---|
| **P0** — corpus-level S/H baseline | **Done** | `data/p0_corpus_level_SH.json` is the reconciliation ground truth |
| **P1 — CPU pass** (Family A + C) | **Done** | 720 parquet shards / 230,530 rows |
| **P1 — GPU pass** (Family B) | **Blocked at the verification gate** | Environment is fully working; blocked on a *scientific* finding, not an engineering one |
| **P1 — join + reconcile** | Not run | Needs GPU output first |
| **P2** — doc-level fractal reliability gate | Not started | The real fork in the road; gates which fractal columns P3+ may use |
| **P3** — binary detection (headline result) | Not started | |
| **P4** — adversarial robustness | Not started | |
| **P5** — generator-family / RAID transfer | Not started | |
| **P6** — release + writeup | Not started | |

P1 is roughly a fifth of the planned extension. The blocker moved today from
"can we run it" to "what does the input data actually mean."

**Part I** (the "Fractal Gap": LLM text shows higher S and higher H than human
text) and **Part II** (the negative result — deep 64-scale fractal residuals
*hurt* cross-domain generalization; 8 baseline features generalize better) are
both established prior work and are not overturned by anything below. Part I's
gemma half is strengthened.

---

## 2. What was measured today

### 2a. Environment (resolved, no longer interesting)

Offline GPU box brought fully up: 46 missing wheels sourced and installed
(`torch==2.7.1+cu118` plus the whole `nvidia-*-cu11` family and transitive
pure-Python deps); HuggingFace cache repaired after exFAT transport destroyed
every symlink into 1067-byte `XSym` stubs (`fix_hf_symlinks.py`); prefix
backfill JSONL placed. `torch.cuda.is_available() == True` on the A100.

### 2b. Stage A / Gate 2 — smoke test (gemma only, N=5)

Passed. Gemma and the MiniLM embedder both load from the repaired cache.
Stratified sampler confirmed working.

### 2c. Stage B / Gate 3 — N=40, default manifest

The default sampler takes 1 row from each of the first 40 shards in sorted
order. Given `gagle3__<split>__<model>_<temp>_<domain>_<prompt_type>`, that is
**exactly `test / gemma_2b / temp 0.0`, 5 domains × 8 prompt types**. All 40
documents are gemma-generated. This is worth stating because it was not obvious
and it confounded the first Mistral reading.

| scorer | passed | verdict |
|---|---|---|
| gemma-2b | **18/40** | exactly the documented expected pattern |
| mistral-7b | 1/40 | materially different — investigated below |

Gemma's 18 decomposes precisely as OVERVIEW §5 predicts:

| prompt_type | passed | expected |
|---|---|---|
| continue_pt | 5/5 | pass |
| keywords | 5/5 | pass |
| summary | 5/5 | pass |
| continue_it | 3/5 | "mostly passes, some genuine misses" |
| cot | 0/5 | "mostly fails even with the wide sweep" |
| excerpt / short_keywords / summary_keywords | 0/15 | "essentially never pass" |

3+5+5+5 = 18. Passing rows have `mean_abs_diff` ≈ 0.007–0.012; failing rows
1.2–2.1. Clean bimodal separation — not a threshold artifact.

### 2d. Repointing the manifest at mistral-generated shards (N=40)

`GAGLE_MANIFEST_DIR` was repointed at a symlink directory of
`test / mistral_7b / 0.0` shards. No code change.

| scorer | passed |
|---|---|
| gemma-2b | **0/40** |
| mistral-7b | **0/40** |

Both scorers fail on mistral-generated rows. Since gemma's checkpoint is proven
correct by 2c, this cannot be a gemma checkpoint problem.

> The script's own `[FAIL]` message asserts "GEMMA_MODEL_ID/MISTRAL_MODEL_ID are
> likely wrong … this is a checkpoint problem." **That diagnostic is wrong in
> this context** — it assumes the default sample and has no knowledge that the
> manifest was repointed. Worth fixing or annotating before someone trusts it.

### 2e. Cross-column probe (`probe_cross_column_alignment.py`, N=10 per shard)

Purpose-built for this question; reuses the pipeline's own windowing, sweep,
candidate-selection and gate code, so results are directly comparable.

**Control** — gemma-generated shard, `big_patent / continue_pt`:

| model × column | passed | median r | max r |
|---|---|---|---|
| gemma × gemma *(native)* | **9/10** | **1.0000** | 1.0000 |
| gemma × mistral *(crossed)* | 0/10 | 0.234 | 0.394 |
| mistral × gemma *(crossed)* | 0/10 | 0.203 | 0.326 |
| mistral × mistral *(native)* | 1/10 | 0.496 | 0.9993 |

**Failing case** — mistral-generated shard, same domain/prompt type:

| model × column | passed | median r | max r |
|---|---|---|---|
| gemma × gemma *(native)* | 0/10 | 0.431 | 0.936 |
| gemma × mistral *(crossed)* | 0/10 | 0.246 | 0.513 |
| mistral × gemma *(crossed)* | 0/10 | 0.242 | 0.419 |
| mistral × mistral *(native)* | 0/10 | 0.525 | 0.777 |

**Column swap is ruled out.** Both crossed cells sit at median r ≈ 0.20–0.25 in
both shards. Had the columns been transposed on mistral rows, a crossed cell
would have scored well. None does.

**Exactly one cell of the eight works.**

### 2f. Exhaustive sweep — the decisive negative

`gemma × gemma` on mistral-generated rows reached max r = 0.9357 under the
default ±150 offset sweep: well above the ~0.2 noise floor, below the 0.98 gate.
That is the signature of *nearly*-correct alignment, which usually means the
true offset lies outside the search window.

Re-run with `OFFSET_SWEEP_RADIUS` rebound to 10⁷ (searching every valid start
position in the sequence):

- **max r = 0.9357 — identical.** Best offsets still cluster at 66–72
  (median 66), unchanged.
- Control under the same full sweep: still 9/10, still median r = 1.0000,
  offsets 1–65. Widening the window did **not** manufacture spurious fits, so
  the method itself is sound.

Searching harder finds nothing. The arrays do not correspond to the text.

### 2g. The offset signature — right position, wrong values

`best_abs_start` is consistent, not scattered:

| cell | offsets | median r |
|---|---|---|
| mistral × mistral, mistral-gen | 66 ×8, 68 ×2 | 0.525 |
| mistral × mistral, gemma-gen | 66–71 | 0.496 |
| gemma × gemma, mistral-gen | 66–72 (one 27) | 0.431 |
| gemma × gemma, gemma-gen | 1–65 | **1.000** |

Locking onto position 66 on nearly every document and then returning r ≈ 0.5 is
diagnostic. Wrong text, wrong slice, or a bad join would **scatter** the
offsets. Consistent position with degraded correlation means: correct tokens,
scored by a model that is not the one that produced the stored numbers.

### 2h. Structural stats

Mostly clean, and one apparent anomaly turned out benign:

- `mis_tok` pinning at exactly 1025 across 8 mistral-generated rows, and
  `gem_tok` pinning at ~1025 on the longest gemma-generated rows, is a
  **generation cap** (`max_new_tokens` ≈ 1024, +BOS), appearing symmetrically
  for each generator. Not truncation.
- `<col>_tot − <col>_tok` holds a steady +70…+123 (the prefix) in both shards.
- **One genuine anomaly:** row 0 of the mistral shard has `gem_tot = 745 <
  gem_tok = 850` — a stated total smaller than the text alone, which is
  structurally impossible. That row's gemma metadata does not describe that
  row's text.

---

## 3. Diagnosis

Two **independent** faults. The evidence separates them.

**Fault 1 — `MISTRAL_MODEL_ID` is probably wrong, but close.**
r ≈ 0.5–0.78 *at the structurally correct offset* is what a sibling checkpoint
produces: `Mistral-7B-Instruct-v0.1` (a finetune of base) or `v0.2`/`v0.3`
(continued pretraining) would correlate partially with base `v0.1` and never
match it. An unrelated model returns ~0.2 flat, which is what the crossed cells
show. GAGLE never recorded which checkpoint produced its scores; `v0.1` was
always a documented guess (OVERVIEW §3).
*Status: strongly supported inference, not proven. Proving it requires
downloading candidate checkpoints (~14 GB each).*

**Fault 2 — mistral-generated rows have a text/score mismatch that also breaks
the gemma column there.**
Independent of Fault 1, because gemma's checkpoint is proven correct by 2c/2e.
Something about how that half of the corpus was assembled means the stored
arrays don't correspond to the stored text.
*Status: established empirically; root cause unknown.*

---

## 4. What this does and does not do to the claims

**Strengthened:**

- Part I's **`gemma_2b` cells** now rest on arrays *verified reproducible* from
  the stated checkpoint (r = 1.0000). That was previously an assumption. This
  is a genuine upgrade in evidential standing.
- The two-candidate + per-document offset sweep design is **vindicated**. The
  expected per-prompt-type pattern reproduced exactly on independent hardware,
  and the full-sweep control showed the gate isn't finding spurious fits.

**Unchanged:**

- The Fractal Gap itself. It held in all 30 LLM cells across 3 warmup variants
  (`family_a_headline_rebuild.json`), and nothing today touches those
  computations. Non-reproducibility ≠ incorrectness.
- Part II's negative result, which is downstream of features not yet rebuilt.

**Now requiring an explicit limitation:**

- Part I's **`mistral_7b` cells** consume arrays whose provenance is
  unverified — and which we currently cannot reproduce. State this. Do not
  retract it; there is no evidence they are wrong.
- **Family B coverage** is the live risk. On current evidence Family B is
  computable only for gemma-2b on gemma-generated rows — a fraction of the
  intended design, which assumed both scorers across the whole corpus.
  Cross-perplexity (gemma vs mistral log-PPL ratio), computed in
  `build_p1_join_feature_store.py`, **cannot be computed at all** if the
  mistral side never aligns.

---

## 5. Open questions, ranked

1. **Do human-side rows align?** *(untested — highest value, lowest cost)*
   Every document examined today is LLM-generated. The script's own expected
   pattern leads with "human side … should pass." Family A's per-domain
   normalization rests entirely on human rows. If human rows align cleanly in
   both shard families, Fault 2 is confined to Mistral's *generations* and the
   headline table's baseline is untouched.
2. **Is Fault 2 per-generator or per-shard?** Distinguishable by (1)'s result.
3. **Which Mistral checkpoint?** Requires downloading candidates. Not on the
   critical path — the gemma-only run is the long pole regardless.
4. **What is row 0's metadata inversion?** Small, but it is a concrete
   reproducible defect and may be the thread that unravels Fault 2.

---

## 6. Recommended next actions

1. **Run the human-side probe** (both shard families). One command each; decides
   the scope question.
2. **Then, if human rows pass:** start the full run as `--scorers gemma-2b` in
   tmux. It is resumable, checkpointed per shard, and valuable however Mistral
   resolves. Do not wait on Mistral to bank it.
3. **Annotate or fix the misleading `[FAIL]` diagnostic** in
   `verify_alignment` (§2d) so it doesn't assert a checkpoint problem when the
   manifest has been repointed.
4. **Defer the Mistral checkpoint hunt** until the gemma run is going.
5. **Carry a limitations paragraph** into the writeup covering §4.

---

## Artifacts produced today

| File | Purpose |
|---|---|
| `fix_hf_symlinks.py` | Repairs exFAT-destroyed HF cache symlinks. Tested against a synthetic fixture: dry run, refuse-on-missing-target, clean apply, idempotent re-run. |
| `probe_cross_column_alignment.py` | Cross-column 2×2 probe. `--full-sweep`, `--side`, `--verbose`. Reuses pipeline internals rather than reimplementing them. |
| `RUNBOOK_family_b_gpu.txt` | Operational runbook, reconciled against OVERVIEW.md. |
| `torch_cuda_deps_cu118/` | 46 wheels missing from the original offline bundle. |

No file in `src/` was modified at any point today.

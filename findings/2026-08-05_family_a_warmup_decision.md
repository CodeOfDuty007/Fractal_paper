# Family A Warm-Up Contamination — Verification Checklist & Decision

**Date:** 2026-08-05
**Scope:** Decision record for how to proceed after `reports/2026-08-03_prompt_type_reconstruction_probe.md` — an investigation that started as a Family B (perplexity/semantic) re-scoring fix — surfaced a confirmed, separate defect in Family A (the fractal `S`/`H` features that are the paper's core contribution).

---

## Background

For `continue_it`/`continue_pt` documents (~25% of the P1 corpus, ~57k rows), GAGLE's stored per-token `gemma-2b` score array covers the prompt **prefix** concatenated with the generated text, not the generated text alone. The pipeline's `WARMUP_DROP=64` — a flat, fixed skip applied to the front of every score array before computing fractal parameters — does not clear these prefixes, which average ~94–99 tokens. The result: for these two prompt types, ~30–35 tokens of "the model reading its own instructions" sit inside the trajectory that `calc_S`/`calc_H` treats as generation dynamics.

This uses the **same `[64:464]`-style windowing convention** that Part I's original "Fractal Gap" finding was built on. A preliminary pooled-corpus estimate already run shows:

- `S` drops ~0.05–0.085 across every split tested — roughly half of the reported ~0.095 domain gap.
- `H` drops 0–0.023, less consistently.

**Confirmed against the original paper** (`docs/fractal paper.pdf`, p.4): "We drop the first 64 tokens to remove any warm-up effects..." is specified verbatim in the paper's own methodology. The fix below does not remove, shorten, or deviate from that rule — it corrects *where counting starts*, since the paper's rule implicitly assumes position 0 of the scored array is where generation begins, and GAGLE's raw array doesn't satisfy that for these two prompt types. The current, unfixed pipeline is the one out of step with the paper's stated method, not the proposed fix.

---

## Advisor review — corrections to this checklist before execution

Per the user's explicit instruction, this checklist was reviewed by the advisor model before running any of it. Two real spec errors were caught and are fixed here (the checklist text above is left unmodified as the original record; these corrections govern what was actually executed):

1. **Two different corrections were being conflated.** Line 27's `skip = prefix_len_tokens + WARMUP_DROP` (start ~163, drop 64 tokens *of generation* as warm-up, on top of skipping the whole prefix) is NOT the same correction as `max(WARMUP_DROP, prefix_len)` (start ~99, where generation begins, no additional warm-up drop). The already-reported preliminary ΔS/ΔH estimates (diag19/20/21, −0.05 to −0.085) used `prefix_len − 2` — i.e. the *first* form (skip the prefix, no extra warmup on generation). The paper's own stated rule ("drop the first 64 tokens to remove warm-up effects") arguably means warm-up applies to *generation*, supporting the second form. **Both variants are run in step 2** below so the effect can be decomposed into "prefix removal" vs. "prefix removal plus warm-up-on-generation," rather than picking one and leaving the other unmeasured.
2. **Step 1's pass criterion was unachievable as written.** `prefix_len_tokens` is NOT 0 for the unaffected types — `keywords` averages 127 tokens, `summary` 130, and even the human side carries a prefix field (just doesn't use it for scoring). What's actually zero for those types is the score array's *prefix-conditioning*, not the prefix's length. The correction must therefore be gated on `prompt_type` (applied only to `continue_it`/`continue_pt`, whose `text_alone_delta ≈ −prefix_len` is the confirmed discriminator for prefix-conditioning), not on whether `prefix_len > 0`. Step 1 is rewritten below to test that the prompt_type-gated code path reproduces today's output bit-for-bit on the six non-continuation types plus the human side.

Two more things carried forward rather than silently resolved: line 36's "validated to r=0.9998" refers to a single document's piecewise fit, not a corpus-wide validation — the ±2 seam offset is assumed constant across the corpus, not independently confirmed at scale. And diag21's split showed `continue_pt` CONFIRMED at ΔH=+0.0003 vs. `continue_it` CONFIRMED at ΔH=−0.0185 — same nominal prefix length (~94 both), opposite H behavior, unexplained. Both are carried into the step 7 write-up rather than averaged away.

## Part 1 — Verification checklist

Run in this order. Each step has an explicit pass/fail criterion — don't proceed to the next step on a vague "looks fine."

### 1. Regression check on unaffected prompt types

- **What:** Apply the corrected windowing (`skip = prefix_len_tokens + WARMUP_DROP`, instead of a flat `WARMUP_DROP`) to a prompt type already confirmed to have zero prefix contamination — the Human side, or the majority of `keywords`/`summary`.
- **Why:** For these, `prefix_len_tokens = 0`, so the corrected code path must reduce to exactly today's behavior. Any difference here means the fix itself has a bug, not that there's new evidence about `continue_it`/`continue_pt`.
- **How:** Recompute Family A for a sample (n≈200) of unaffected rows through both the old and new windowing logic; diff row-by-row.
- **Pass:** 0 differing rows (bit-for-bit, or within float tolerance ~1e-9).

### 2. Small stratified before/after sample

- **What:** n=200–500 `continue_it`/`continue_pt` documents, stratified by domain × model × temperature (reuse the same sampling approach as the reconstruction probe), with `S`/`H` computed under both the old flat-64 skip and the corrected `prefix_len + 64` skip.
- **Why:** Confirms the already-reported ~0.05–0.085 pooled estimate isn't a one-off, and gives a full distribution instead of a single point estimate.
- **How:** Reuse `prefix_len_tokens` values already captured during the Family B investigation (already validated to r=0.9998 against the empirically-located seam boundary) — no new tokenization work needed.
- **Output:** mean / median / std of (`S_corrected − S_old`) and (`H_corrected − H_old`), plus % of documents with |Δ| > 0.05.

### 3. Full scoped recompute (~57k rows)

- **What:** Recompute Family A for exactly the affected `continue_it`/`continue_pt` P1 rows — not the full 230k-row corpus.
- **Why:** This is CPU-only and re-slices GAGLE score arrays already on disk. No GPU, no re-scoring — cheap relative to the paused Family B work.
- **How:** Batch over the existing feature store; write corrected `S`/`H` to a new, versioned column rather than overwriting in place, so old and corrected values are both auditable.
- **Pass:** 57k rows in, 57k rows out, 0 silently dropped — matches this project's existing manifest-driven completeness standard (`members_missing: []`, gap-free coverage).

### 4. Rebuild Part I's headline table, corrected

- **What:** Re-run the exact domain × label pooled `S`/`H` table method already validated in `reports/2026-07-26_S_estimator_fix.md` — same scale list `τ ∈ {8,16,32,48,64,96,128,160,192,256,320}`, same `epsilon=1e-2`, same list-of-trajectories pooling. Do not introduce a new pooling method here; that would make the comparison invalid.
- **Why:** This is the only number that's directly comparable to the already-published corpus-level table.
- **How:** Substitute corrected `continue_it`/`continue_pt` trajectories into the existing pooling code; leave all other prompt types untouched.
- **Output:** a corrected variant of `data/p0_corpus_level_SH.json`, side-by-side with the original, per (domain, label) cell.

### 5. Direction/sign check — the actual claim test

- **What:** For each of the 5 domains, is corrected LLM `S` still greater than Human `S`? Is corrected LLM `H` still greater than Human `H`?
- **Why:** This is what the paper's claim actually depends on — direction, not exact magnitude. A shrinking gap weakens the claim; a sign flip breaks it. These are different outcomes and must be reported separately.
- **Pass criterion:** explicit yes/no per domain per exponent (10 cells total), not an average across domains.

### 6. Diluted vs. undiluted reporting

- **What:** Report both (a) the pooled per-domain number (post-correction — matches the headline table, diluted by the ~75% of unaffected prompt types in the same cell) and (b) the effect isolated to just `continue_it`/`continue_pt` rows alone.
- **Why:** The pooled number is what's headline-relevant; the isolated number makes the underlying mechanism legible without dilution. Matches how every prior correction in this project (domain-scoped dedup key, per-document `S` bias) was documented — full magnitude disclosed, nothing rounded away.

### 7. Write-up

- **What:** A dated report (`reports/<date>_family_a_correction_results.md`) documenting steps 1–6's actual numbers, old vs. corrected, ending in an explicit verdict per domain: claim survives / survives but weaker / does not survive.
- **Why:** Consistent with this project's standing practice — corrections are always disclosed with their exact magnitude, never silently folded in.

---

## Part 2 — Decision

**Options considered:**

1. **Fix Family A first, Family B after.** Pause Family B/GPU-bundle work. Fix `slice_scores` for `continue_it`/`continue_pt`, recompute Family A for the ~57k affected rows, check impact on Part I's headline gap. Resume Family B once Family A is solid.
2. **Keep going on Family B, log this as a known issue.** Continue the Family B GPU scorer fix as planned; document the Family A finding as a flagged, unresolved provenance issue for later.
3. **Need more detail before deciding.**

**Chosen answer: Option 1 — Fix Family A first, Family B after.**

Reasoning:

- **This sits inside the paper's own headline claim, not a supporting feature.** Family A (`S`/`H`) is Part I's "Fractal Gap" finding — the scientific core of the whole project. Family B (log-perplexity, cross-perplexity) is a supporting detector feature; it doesn't touch Part I's claim at all. If the goal is protecting the paper's validity (as stated), Family A is the one actually at risk, and it's the one worth resolving first.
- **The fix is cheap, which lowers the cost of pausing.** It's CPU-only and re-slices score arrays already downloaded — no GPU dependency, unlike Family B's re-scoring work. Pausing Family B here costs a re-slice and a rerun of already-existing pooling code, not new weeks of work.
- **This project's own precedent already answers the question.** Every prior foundational-integrity issue in this project (the domain-unscoped Human dedup key, the per-document `S` estimator bias) was handled the same way: stop, fix, verify, and only then trust anything built downstream. Continuing Family B work on top of a feature store now *known* to have contaminated Family A values for 25% of it just relocates the rework to after P2/P3 are built on top of it — more expensive to unwind later than now.
- **Option 3 isn't buying much at this point.** The magnitude (0.05–0.085 of a 0.095 gap), the affected slice (~25%, `continue_it`/`continue_pt` only), and the fix's cost (cheap, CPU-only, no GPU) are already established. What remains unknown — whether the gap's *direction* survives per domain — is exactly what the checklist in Part 1 answers, and it's fast enough that further questions mostly delay getting the real answer rather than changing the decision.

**Next action:** run Part 1, steps 1–5, before writing any further Family A or Family B code. Step 7's write-up is the exit gate for this decision — once it exists, resume Family B (if the claim survives) or re-scope Part I's reporting (if it doesn't).

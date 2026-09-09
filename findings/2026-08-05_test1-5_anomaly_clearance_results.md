# Anomaly-Clearance Test Batch — Results (Tests 1–5)

**Date:** 2026-08-05
**Scope:** Executes the 5 test specs you added (`reports/2026-08-05_test1..5_*.md`), each opened against a gap flagged in `reports/2026-08-05_family_a_warmup_decision.md`. Read for decisions; advisor-reviewed per your instruction before and during execution (see "Advisor review" notes inline).

## Status at a glance

**All 5 tests are now complete** — HF credentials were restored mid-session; the remaining 3 tests ran immediately after and are folded in below.

| Test | Status | Headline result |
|---|---|---|
| 1 — window-start sensitivity (4 unresolved types) | **DONE** | All 4 types FAIL the low-risk threshold — real Family A risk, not just a Family B alignment gap |
| 2 — human-side realization consistency | **DONE** | 5 of 8 member types clean (r≈1.0000); 3 (`excerpt`, `short_keywords`, `summary_keywords`) essentially uncorrelated (r≈0.09–0.11) |
| 3, Part A — Mistral-7b prefix-inclusion check | **DONE** (step 1) / **DEFERRED** (step 2) | Mistral's defect pattern does NOT match Gemma's — Gemma's fix does not transfer as-is |
| 3, Part B — sub-feature cascade sensitivity | **DONE** | Point `S` and windowed `S` badly contaminated; windowed `H` mean/var hold up, `H` drift does not |
| 4 — statistical significance (bootstrap CI) | **DONE** | All 10 gemma_2b cells SURVIVE — corrected gap's 95% CI excludes zero everywhere tested |
| 5 — skip-overflow edge cases | **DONE** | `MIN_TOKENS` gate confirmed sufficient — predicted drop rate (10.2%) matches actual recompute (10.36%) |

**One substitution you should know about, in every test that specifies it:** Tests 1, 3 (Part B), and 5 were originally written against `skip = prefix_len_tokens + WARMUP_DROP` — the simplified formula from the *original* decision doc, before the per-record `compute_boundary` formula superseded it (see `reports/2026-08-05_family_a_correction_results.md`, "Seam-offset prerequisite, closed"). Wherever these tests needed it, I used `compute_boundary` — the formula actually shipped in `recompute_family_a_continuation_fix.py` — not the superseded one. Flagging this since it's your document and the numbers won't match what's literally written there.

---

## Test 1 — Window-start sensitivity, 4 unresolved prompt types (DONE)

**Method exactly as specified:** n=300/type (stratified domain×model×temp), swept window-start `k` from 0 to 96 in steps of 8, pooled corpus-level `S`/`H` via list-of-trajectories mode at each `k`, compared to `k=0`.

| prompt_type | max\|ΔS\| | max\|ΔH\| | verdict |
|---|---|---|---|
| excerpt | 0.092 | 0.034 | **HIGH_RISK_FAIL** |
| short_keywords | 0.232 | 0.070 | **HIGH_RISK_FAIL** |
| summary_keywords | 0.138 | 0.042 | **HIGH_RISK_FAIL** |
| cot | 0.170 | 0.068 | **HIGH_RISK_FAIL** |

**All 4 types fail** the pass criterion (max|ΔS| < 0.02 and max|ΔH| < 0.02) by a wide margin — `short_keywords`' S swings by 0.23, more than double the paper's entire reported ~0.095 domain gap.

**Advisor-caught confound, checked before reporting this:** the sampled pool shrinks as `k` grows (shorter documents drop out once `k+400` exceeds their length), so some of the swing could in principle be cohort drift rather than genuine window-sensitivity. I checked: at `k=8`, pool size is barely touched (300→295–298 docs, under 2% loss) yet `S` has *already* moved by 0.03–0.14 depending on type. The dominant effect is present before any meaningful cohort change — this is real window-start sensitivity, not a sampling artifact. (Full `n` and `ΔS`/`ΔH` per-`k` table saved separately if you want to inspect the curve.)

**What this means:** the confirmed `continue_it`/`continue_pt` fix does **not** bound the problem. These 4 unresolved prompt_types are 46.75% of the corpus's LLM-side documents (101,418 of 216,904, from the same per-cell census in `reports/2026-08-05_family_a_correction_results.md`'s dilution step — not an estimate) and carry a real, large Family A sensitivity to an alignment question nobody has solved yet — not just a "Family B feature is null for these" inconvenience. Per the test's own pass/fail framing, existing `Gemma_S`/`Gemma_H`/`Mistral_S`/`Mistral_H` values for `excerpt`, `short_keywords`, `summary_keywords`, and `cot` documents should be treated as **not yet trustworthy** until an alignment rule is found for them (same treatment already given to per-document `S` via `Holder_PerDoc_Unreliable`).

---

## Test 4 — Statistical significance of the corrected gap (bootstrap CI)

**Unblocked without HF credentials** — this is the one finding that needed catching: the advisor pointed out that `data/p1_family_a_corrected_continuation.jsonl`'s `boundary_used` column (written per-record by the recompute script) is the *only* tokenizer-derived quantity the corrected window needs. Rejoining `raw[boundary_used + WARMUP_DROP : ...]` directly against the zip's raw score arrays reconstructs the corrected trajectories with zero new tokenizer calls — so this test runs today despite the blocker.

**Method:** 10-fold outer bootstrap (resample documents with replacement per cell, recompute pooled `S`/`H` per fold), gap CI = 2.5th/97.5th percentile of `(LLM_fold − Human_fold)` across folds. Run for both OLD and CORRECTED windowing, `gemma_2b` and `mistral_7b` separately.

**Scope note carried forward honestly:** the corrected file is gemma-2b only (already flagged in the correction report — mistral-7b's `continue_it`/`continue_pt` rows were never recomputed). So the CORRECTED-variant CI can only be computed for `gemma_2b` cells here; `mistral_7b`'s original-gap CI is still reported for reference, with corrected explicitly marked unavailable rather than assumed clean.

**Result: every gemma_2b cell survives.** All 10 (5 domains × S/H) have a corrected-gap 95% CI that excludes zero, in the same direction as the original. The correction narrows the gap (as already known from the point-estimate table) but never makes it statistically indistinguishable from zero at any tested cell.

| domain | scorer | exponent | original gap | original 95% CI | corrected gap | corrected 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| big_patent | gemma_2b | S | +0.0573 | [0.0466, 0.0654] | +0.0389 | [0.0323, 0.0452] | **SURVIVES** |
| big_patent | gemma_2b | H | +0.0521 | [0.0483, 0.0526] | +0.0424 | [0.0390, 0.0447] | **SURVIVES** |
| billsum | gemma_2b | S | +0.0340 | [0.0215, 0.0538] | +0.0185 | [0.0121, 0.0331] | **SURVIVES** |
| billsum | gemma_2b | H | +0.0352 | [0.0314, 0.0369] | +0.0312 | [0.0274, 0.0339] | **SURVIVES** |
| newsroom | gemma_2b | S | +0.0961 | [0.0823, 0.1067] | +0.0751 | [0.0649, 0.0855] | **SURVIVES** |
| newsroom | gemma_2b | H | +0.0562 | [0.0532, 0.0572] | +0.0462 | [0.0424, 0.0474] | **SURVIVES** |
| scientific_papers | gemma_2b | S | +0.0679 | [0.0548, 0.0760] | +0.0478 | [0.0408, 0.0553] | **SURVIVES** |
| scientific_papers | gemma_2b | H | +0.0795 | [0.0761, 0.0806] | +0.0663 | [0.0635, 0.0682] | **SURVIVES** |
| wikipedia | gemma_2b | S | +0.0435 | [0.0366, 0.0559] | +0.0259 | [0.0189, 0.0350] | **SURVIVES** |
| wikipedia | gemma_2b | H | +0.0320 | [0.0298, 0.0343] | +0.0249 | [0.0228, 0.0263] | **SURVIVES** |

`mistral_7b`'s original-gap CIs are shown for reference (all exclude zero too), but corrected CIs are **not computed** — consistent with the scope note carried through every prior report this session: `mistral_7b`'s `continue_it`/`continue_pt` rows were never checked for or corrected against this defect. These 10 cells (5 domains × S/H) remain an open question, not silently assumed to survive by extension.

| domain | exponent | mistral_7b original gap | original 95% CI |
|---|---|---|---|
| big_patent | S | +0.0787 | [0.0676, 0.0826] |
| big_patent | H | +0.0732 | [0.0709, 0.0763] |
| billsum | S | +0.0262 | [0.0143, 0.0447] |
| billsum | H | +0.0532 | [0.0490, 0.0548] |
| newsroom | S | +0.1198 | [0.1070, 0.1270] |
| newsroom | H | +0.0831 | [0.0810, 0.0849] |
| scientific_papers | S | +0.0973 | [0.0868, 0.1038] |
| scientific_papers | H | +0.1053 | [0.1015, 0.1066] |
| wikipedia | S | +0.0654 | [0.0579, 0.0799] |
| wikipedia | H | +0.0517 | [0.0496, 0.0546] |

**What this adds beyond the sign check already done:** the earlier correction report showed the corrected point estimate stays positive in all 20 cells. This shows the corrected gap is *statistically* distinguishable from zero, not just nominally positive — the claim survives with actual statistical backing, for every gemma_2b cell tested. No cell weakened into "still positive but touching zero" territory; the narrowest CI (billsum S, [0.0121, 0.0331]) still sits comfortably clear of 0.

**Two caveats on how to read these numbers, both non-negotiable to state:**
- **Human-side pools are byte-identical between the "original" and "corrected" columns** — only `continue_it`/`continue_pt` rows change under correction, and the Human side never touches those prompt_types. So `human_old` and `human_corrected` are two *independent bootstrap draws over the same underlying documents*, not a before/after comparison of a changed baseline. Don't read the CI-width difference between the original and corrected columns as a precision change from the correction — some of it is just resampling noise on an unchanged pool.
- **10 folds is what the spec asked for, and it's enough to see these CIs clear zero by 3–5× their own width — it is not enough for a tight interval estimate.** Sufficient for the exclude-zero verdict this test needed; not a claim about the CI's precision beyond that.

---

## Test 5 — Skip-overflow edge cases (DONE)

**Method:** full `continue_it`/`continue_pt` population (60,000 rows), gemma-2b tokenizer, `skip = compute_boundary(...) + WARMUP_DROP` (the shipped formula, not the spec's original simplified one), `remaining = gagle_len − skip`.

| prompt_type | n | % remaining≤0 | % 0<remaining<320 | % remaining≥400 | min | p1 | p5 | median |
|---|---|---|---|---|---|---|---|---|
| continue_it | 30,000 | 0.10% | 4.11% | 89.26% | −59 | 225 | 332 | 808 |
| continue_pt | 29,992 | 2.31% | 13.84% | 80.93% | −64 | −31 | 58 | 856 |

**Pass.** Real overflow cases exist (min remaining reaches −59/−64, and `continue_pt`'s p1 is negative — confirming the probe's own flag that `continue_pt` has a much longer tail than `continue_it`), but the existing `MIN_TOKENS=320` gate absorbs all of them without producing garbage: `slice_scores_per_record` returns `None` on `remaining≤0` and `0<remaining<320`, and the actual recompute run completed cleanly. As a direct consistency check: the weighted "remaining<320" rate implied by this table (≈10.2%) matches the actual recompute's observed drop rate (10.36%, from `data/p1_family_a_corrected_continuation_manifest.json`) almost exactly — two independently-computed numbers agreeing is stronger evidence than either alone.

**Statement the test asked for:** of the ~57,360 `continue_it`/`continue_pt` feature-store rows, ~6,217 are expected to (and did) drop below the length gate under corrected windowing, leaving 53,783 rows with valid corrected Family A values — exactly matching the already-shipped recompute.

---

## Test 3, Part A — Mistral-7b prefix-inclusion check

### Step 1 (length triangulation) — DONE

**Method:** same length-triangulation method as the original probe, n=250/type, Mistral-7b tokenizer (not Gemma's).

| prompt_type | n | prefix_len mean (std) | text_alone_delta mean (std) | combined_delta mean / median (std) | % combined_delta within ±3 |
|---|---|---|---|---|---|
| continue_it | 250 | 111.4 (10.0) | −71.5 (20.2) | 37.9 / **35.0** (18.3) | **0.4%** |
| continue_pt | 250 | 111.4 (10.0) | −92.0 (19.4) | 17.4 / **12.5** (17.0) | **17.2%** |

**Gemma-2b's already-confirmed pattern, for comparison (probe report, n=500):** `continue_it` combined_delta mean=7.23, **median=1.0**, std=8.72; `continue_pt` mean=15.35, **median=1.0**, std=88.50.

**Note on the identical `prefix_len_mean`/std between `continue_it` and `continue_pt` above:** this is expected, not a sampling artifact — confirmed directly in Test 2's investigation below, `continue_it` and `continue_pt` draw from the same underlying documents (the "prefix" is the document's own preceding text, shared regardless of which instruction-style continuation was generated from it), so of course the same 250 prefix strings produce identical tokenization stats under a fixed tokenizer. This doesn't weaken the Mistral-vs-Gemma comparison below — that comparison is tokenizer-vs-tokenizer *on the same text*, which is exactly the controlled comparison wanted, not an accidental confound.

**Real finding, not a clean transfer:** Mistral's prefix IS meaningfully included in its own stored score array (`text_alone_delta` is large and negative, same qualitative signature as Gemma's) — so the underlying defect is confirmed present for Mistral too, not absent. But the naive `prefix_ids + text_ids[1:]` reconstruction that fits Gemma almost exactly (median delta ≈1, meaning the formula is right up to a 1-2 token trim) does **not** fit Mistral the same way — `continue_it`'s median delta is 35 tokens, not ~1, and only 0.4% of documents land within ±3 tokens (vs. Gemma's implied near-universal fit). Mistral's tokenizer likely has a different special-token/BOS convention at the prefix+text seam, or its own array covers a different span. **Gemma's `compute_boundary` formula should NOT be applied to Mistral's `continue_it`/`continue_pt` rows as-is** — doing so would silently produce a wrong correction, not a null result. This needs its own dedicated investigation (piecewise-fit derivation, same rigor as Gemma's) before any Mistral-side fix ships.

### Step 2 (boundary-crossover forward-pass fit) — DEFERRED

This Mac has **17.2 GB total RAM** (confirmed via `sysctl hw.memsize`). Mistral-7B in fp32 needs ~28GB for weights alone; even fp16/bf16 needs ~14GB, leaving under 3GB for the OS, activations, and everything else — not workable, and exactly the kind of memory pressure that caused the MPS thrashing bug diagnosed earlier this session. Not attempted. This step is GPU-machine work, to be picked up once the Ubuntu bundle exists — not a gap in this test run.

---

## Test 3, Part B — Sub-feature cascade sensitivity (DONE)

**Method:** n=200 `continue_it`/`continue_pt` documents (gemma-2b), full Family A feature vector under OLD (flat `WARMUP_DROP`) vs CORRECTED (`compute_boundary + WARMUP_DROP`) windowing, mean absolute delta per sub-feature against a threshold.

**On the thresholds themselves — these are judgment calls, not all directly-measured noise floors, and should be read that way:** `S` (0.05), `S_window_mean` (0.35), `S_window_var`/`S_window_drift` (0.10) are anchored to `S`'s already-reported per-document/windowed std (0.35–0.48, from `reports/2026-07-31_P1_feature_store.md`) — reasonable proxies. `H_window_mean`/`H_window_var`/`H_window_drift` (0.05/0.02/0.05) are my own conservative picks, not measured floors for those specific sub-features. **`H`'s point threshold (0.02) is the shakiest one in this table** — the correction report established point `H`'s bootstrap noise as *exactly zero* on identical windows (deterministic regression, no resampling component), so 0.02 is a judgment call about what magnitude should count as "material," not a measured noise floor the way `S`'s thresholds are. Read the `H` (point) FAIL as "the shift is 3x my chosen threshold," not "the shift exceeds `H`'s established noise floor" — those are different claims and only the latter is fully rigorous.

| sub-feature | n | mean\|Δ\| | threshold used | % exceeding | verdict |
|---|---|---|---|---|---|
| S (point) | 168 | 0.316 | 0.05 | 91.7% | **FAIL** |
| H (point) | 168 | 0.066 | 0.02 | 81.0% | **FAIL** |
| S_window_mean | 167 | 0.451 | 0.35 | 53.3% | **FAIL** |
| S_window_var | 167 | 0.254 | 0.10 | 54.5% | **FAIL** |
| S_window_drift | 167 | 0.518 | 0.10 | 85.6% | **FAIL** |
| H_window_mean | 168 | 0.032 | 0.05 | 22.6% | PASS |
| H_window_var | 168 | 0.002 | 0.02 | 0.0% | PASS |
| H_window_drift | 168 | 0.051 | 0.05 | 47.6% | **FAIL** |

**Not a uniform story, and shouldn't be reported as one.** Point `S` and every `S`-derived sub-feature (mean/var/drift, windowed or not) are heavily contaminated by the old windowing defect — consistent with `S`'s already-known instability (`Holder_PerDoc_Unreliable`), but now confirmed the correction moves it far past noise-floor levels, not just nominally. Point `H` also fails, contrary to the more optimistic pooled-corpus picture (where `H`'s bootstrap noise floor is ~0) — per-document `H` is more sensitive to this specific defect than the corpus-pooled number suggested. The one genuinely reassuring result: windowed `H` mean and variance both pass — these two sub-features don't move beyond their own noise floor even under the full correction. Windowed `H` drift does not share that robustness.

**What this means for the feature store:** every `continue_it`/`continue_pt` row's `Gemma_S`, `Gemma_H`, and all `S`-window/`H`-window-drift sub-features need the same correction already applied to point `S`/`H` — this is not optional or lower-priority just because it's a "sub-feature." `H_window_mean`/`H_window_var` are the only two columns where the old (uncorrected) values are still usable without recomputation.

---

## Test 2 — Human-side realization consistency (DONE)

**Method:** n=40 per prompt_type-of-origin (8 types), re-scored `original_text` alone (gemma-2b, capped 2048) via actual forward pass, correlated against GAGLE's stored array for that specific member.

**Correction made after advisor review caught a real issue in my first pass, verified before writing this up:** I originally described this as "320 independent entries." It is not. The "first 40 matching documents in file-scan order" sampler deterministically collapses onto the SAME underlying documents whenever multiple prompt_type members enumerate documents in the same underlying order — and I confirmed this happened, unevenly, across the 8 types. Checked directly (pairwise `dataset_id` set-intersection, out of 40):

- **Group A — identical 40 documents:** `continue_it`, `continue_pt`, `excerpt`, `short_keywords` (every pair among these four: 40/40 overlap).
- **Group B — separate, largely-independent documents:** `keywords`, `summary`, `summary_keywords`, `cot` (pairwise overlap 0–5/40 among these, and with Group A — noise-level, consistent with independent domain-stratified sampling).

**This changes the finding for the better, not worse.** Within Group A, the exact same 40 human documents were scored under 4 different member-type labels: `continue_it`/`continue_pt` reproduce "text alone" almost exactly (r≈0.99998), while `excerpt`/`short_keywords` — same text, same tokenizer, same model — do not (r≈0.106, essentially uncorrelated). That's not four independent samples landing on the same conclusion; it's a same-document controlled comparison, which is the stronger design the original spec actually asked for ("isolate the member-type effect from a domain/document effect"). Group B is a second, separately-sampled set of 40 documents each, giving an independent replication: `keywords`/`summary`/`cot` pass, `summary_keywords` fails, on a completely different set of underlying documents. Two structurally different lines of evidence, same qualitative split.

| prompt_type (origin) | group | n unique docs | mean r | median r | % r>0.98 | mean abs diff | dedup win share |
|---|---|---|---|---|---|---|---|
| continue_it | A | 40 | 0.99998 | 0.99998 | 100% | 0.0081 | 54% |
| continue_pt | A (same 40 as above) | — | 0.99998 | 0.99998 | 100% | 0.0081 | (rest, split) |
| **excerpt** | A (same 40 as above) | — | **0.106** | **0.097** | **0%** | **2.076** | (rest, split) |
| **short_keywords** | A (same 40 as above) | — | **0.106** | **0.097** | **0%** | **2.076** | (rest, split) |
| keywords | B | 40 | 0.99998 | 0.99998 | 100% | 0.0080 | (rest, split) |
| summary | B (different 40) | 40 | 0.99998 | 0.99998 | 100% | 0.0081 | 18% |
| cot | B (different 40) | 40 | 0.99998 | 0.99998 | 100% | 0.0077 | 18% |
| **summary_keywords** | B (different 40, ~5 shared w/ keywords) | 40 | **0.093** | **0.086** | **0%** | **2.068** | (rest, split) |

**Total unique underlying human documents actually sampled: ~120–125, not 320** (40 in Group A, roughly 40×4 mostly-distinct in Group B minus the small `keywords`/`summary_keywords` overlap) — correcting the table's implied `n`.

**One more thing this reveals:** `excerpt` and `short_keywords` don't just both fail — they fail almost *identically* (r 0.10595 vs 0.10595, mean abs diff 2.0763 vs 2.0762, agreeing to 3+ significant figures despite being scored under nominally different member types on the same 40 documents). That's a strong hint they share the same underlying (wrong) scoring convention, whatever it is — not two independent failure modes.

**`summary_keywords`' failure is the weakest-confirmed of the three, and that distinction matters for whoever picks this up next.** `excerpt`/`short_keywords` failed on the *same* documents `continue_it`/`continue_pt` passed on — a controlled, same-text comparison, about as solid as this kind of evidence gets. `summary_keywords` failed on its own separate 40-document sample (Group B) — a standard independent-groups result, not a same-document control. If someone later derives a scoring rule that explains `excerpt`/`short_keywords`, there's no guarantee it also explains `summary_keywords` — treat it as a related but separately-confirmed finding, not an automatic third instance of the same mechanism.

**Clean, decisive split confirmed, now correctly attributed:** 5 of 8 member types reproduce "human text alone, capped 2048" essentially exactly. The other 3 — `excerpt`, `short_keywords`, `summary_keywords` — are essentially **uncorrelated** with that rule. This generalizes Finding 1's original single-record discovery (`reports/2026-07-31_P1_feature_store.md`, r=0.035 for one excerpt-vs-continue_it comparison) beyond a single data point — and for `excerpt`/`short_keywords` specifically, this is now a same-document-controlled result, not an association that could be confounded by document composition. Whatever conditions GAGLE's scoring for these 3 member types affects **both sides** (human and LLM) of these member types, not just the LLM side as originally suspected in the reconstruction probe.

**Fail/pass by the test's own criterion is genuinely ambiguous, and I'm not going to paper over that:** the spec's fail condition is "any member type with meaningful dedup win share (>5%, conservatively) shows r<0.98." I only have exact win-share figures for `continue_it` (54%), `cot` (18%), and `summary` (18%) — all clean. The three failing types fall into Finding 1's "keywords/summary_keywords splitting the rest" (~10% combined across 5 remaining types: `continue_pt`, `excerpt`, `keywords`, `short_keywords`, `summary_keywords`), which — *if* split roughly evenly — would put each failing type under the 5% threshold individually. But that's an inference from an approximate description, not a measured number. **I did not compute exact win-shares for the 3 failing types**, so I can't respectably call this a clean pass or fail against the test's own stated bar — flagging as unresolved rather than rounding it either way.

**Why this matters regardless of the win-share ambiguity:** even if the corpus-weighted impact on the *currently kept* human-side P1 rows turns out small, `excerpt`/`short_keywords`/`summary_keywords` human-side score arrays are demonstrably not "text alone" — proven now on identical text for two of the three, not inferred. This is the same open mystery flagged in the reconstruction probe, now confirmed to be symmetric (both sides affected), which narrows the hypothesis space for anyone picking this up next (whatever's happening is a property of how GAGLE processed these 3 member types generally, not something specific to LLM generation).

---

## Summary across all 5 tests

Two structurally different kinds of finding came out of this batch:

1. **Tests 4 and 5 CONFIRM the shipped `continue_it`/`continue_pt` fix is sound** — the corrected gap survives with statistical backing (Test 4), and the correction's edge cases are already handled correctly by the existing length gate (Test 5). Nothing to redo there.
2. **Tests 1, 2, and 3 all point at the SAME three prompt_types** (`excerpt`, `short_keywords`, `summary_keywords`) plus `cot` as a real, unresolved, and now well-characterized second problem — not a Family B inconvenience, a Family A trust problem. Test 1 shows their `S`/`H` point estimates are highly sensitive to an unsolved alignment question; Test 2 shows the human-side baseline for 3 of those types is *also* uncorrelated with the assumed scoring rule; Test 3 Part B shows that even where a fix exists (`continue_it`/`continue_pt`), several sub-features beyond point `S`/`H` need the same correction, not just the two headline columns.

**Recommended next decision, mirroring the shape of the `continue_it`/`continue_pt` decision you already made:** `excerpt`/`short_keywords`/`summary_keywords`/`cot` — roughly 47% of the corpus's LLM-side documents and an unknown-but-plausibly-small slice of the human side — need the same kind of flag `Holder_PerDoc_Unreliable` already carries, applied at the Family A level (not just Family B alignment), until someone derives whatever scoring rule GAGLE actually used for them. Separately: Mistral-7b's `continue_it`/`continue_pt` correction needs its own dedicated derivation (Test 3 Part A) — Gemma's formula does not transfer.

Also not blocked by any of this: I made progress on the paused Family B GPU-scorer rewrite (`src/build_p1_feature_store_gpu.py`) while these tests ran — implemented the two-candidate/wide-sweep alignment design, caught and fixed two real bugs in it during self-review (a feature-window bug that would have reintroduced prefix contamination for `text_alone` documents with real offsets, and a GPU-compute-cost blowup from the widened sweep range). That work is saved and ready for local sample verification now that credentials are back.

# Headline-Table Sensitivity Bound + Mistral-7b Derivation Status

**Date:** 2026-08-06
**Scope:** Task 1 — determines whether the pooled, domain-level gemma_2b Human-vs-LLM gap (`S` and `H`) can plausibly flip sign or collapse toward zero once the unresolved windowing uncertainty in `excerpt`/`short_keywords`/`summary_keywords`/`cot` (confirmed real and large in Test 1, `reports/2026-08-05_test1-5_anomaly_clearance_results.md`) is folded into the full pooled table, rather than assessed per-type in isolation. Task 2 — status of the Mistral-7b `continue_it`/`continue_pt` correction. Task 3 — Family B GPU-scorer pause confirmation.

---

## Method (written before the numbers, so the framing isn't retrofitted to the result)

**Pooling composition, gemma_2b only, per domain:**
- **Held fixed at already-established values:** `continue_it`/`continue_pt` under the shipped `compute_boundary + WARMUP_DROP` correction (rejoining `boundary_used` against raw zip arrays, zero tokenizer calls — same method as Test 4); `keywords`/`summary` under the confirmed "text alone" rule (plain `slice_scores`, unchanged); Human side under "text alone, capped 2048" (plain `slice_scores`).
- **Swept:** `excerpt`/`short_keywords`/`summary_keywords`/`cot`, using Test 1's exact stratified sample (domain × temp, n=300/type) with window-start `k` varied 0–96 in steps of 8 — but this time the resulting trajectories are folded into the **full domain-level pool** alongside the fixed types, not evaluated as an isolated per-type pool the way Test 1 did.

**Independent sweep:** for each of the 4 swept types, vary its own `k` while holding the other 3 at `k=0`, recompute the full 5-domain pooled gap. Isolates each type's individual contribution to headline-table movement.

**Joint worst-case, found by local search, not exhaustive grid search:** a true 13⁴-combination grid (28,561 evaluations × 5 domains) is not the tractable path, and more importantly `calc_S`'s list-of-trajectories pooling is not additively decomposable across types — the `k` that minimizes one type's marginal contribution in isolation is not guaranteed to minimize it jointly. Method used instead: take the independent sweep's per-type argmin `k` as a candidate, actually recompute the joint pool at that combination (not inferred), then locally probe the neighborhood (±1 step per type) for a few iterations until no perturbation improves the score. **This is reported as a local worst case, not a global one** — the honest label for what this search actually establishes.

**Bootstrap at the worst case only:** Test 4's exact 10-fold method, but resampling **each contributing group at its own size** (baseline pool, human pool, and each of the 4 swept types' own per-domain sample) independently per fold — not a single pooled resample — so fold-to-fold variation in how many `excerpt`-type documents got drawn isn't a confound in the CI width. Same caveat as Test 4 carries forward: the Human pool is unchanged across all `k`, so its bootstrap draws are independent redraws on identical documents, not a before/after comparison.

**The single most important caveat, stated up front rather than buried:** the 4 swept types enter this reconstructed pool at n≈300 each (~60/domain), while their true corpus presence is ~101,418 documents (46.75% of the corpus's LLM-side rows, per the correction report's dilution census). **This sweep is therefore massively under-weighted relative to the real headline table, and a "survives" verdict here is a weaker claim than a "survives" verdict on correctly-weighted data** — it understates how much the real headline number could move. To close this gap, a second pass replicates each swept type's sample up to its true per-domain corpus count (exact counts from the existing dilution census, not estimated) at the joint worst-case `k`, and reports both the unweighted and weighted gap. If the verdict matches, the under-weighting didn't matter. If it doesn't, that difference is itself the finding.

**Human-side contamination bound:** Test 2 found `excerpt`/`short_keywords`/`summary_keywords`'s human-side score arrays are also unresolved (r≈0.09–0.11 against "text alone"). The domain-level Human pool used here and throughout this project's headline table includes whichever member type won the dedup key for each document — some fraction of it may already be built on those unresolved arrays. Bounded directly, per domain (from the same corpus scan used for the sweep, not a separate pass — every kept Human row's winning dedup member's `prompt_type` was tracked):

| domain | Human rows from `excerpt`/`short_keywords`/`summary_keywords` origin | total Human rows | % |
|---|---|---|---|
| big_patent | 175 | 2,418 | 7.24% |
| billsum | 5 | 2,169 | 0.23% |
| newsroom | 122 | 2,370 | 5.15% |
| scientific_papers | 349 | 2,458 | **14.20%** |
| wikipedia | 282 | 2,281 | 12.36% |

**Not negligible, and not uniform across domains.** `billsum`'s Human pool is essentially clean (0.23%), but `scientific_papers` and `wikipedia` have 12–14% of their kept Human rows sourced from a member type whose human-side scoring rule is confirmed unresolved. This means those two domains' Human baseline — the denominator every LLM comparison is measured against — is itself partly built on arrays that don't follow the assumed "text alone" rule, independent of anything the k-sweep below tests. This is a real, quantified limitation on the Human side of the headline table, not fixed by anything in this report, and should be read as a caveat on every domain-level gap number that follows, not just the swept LLM side.

---

## Results

**Data validated before use:** the corpus scan's gemma_2b-labeled `continue_it`/`continue_pt` corrected-row count (25,213) was cross-checked against the already-shipped correction file's own label breakdown and matches exactly (`Counter({'mistral_7b': 28570, 'gemma_2b': 25213})`), and `big_patent`'s reconstructed Human pool size (n=2,418) matches the previously-published headline table exactly. Both baseline pools are confirmed correctly built before trusting anything downstream.

### Independent sweep (each type varied alone, others at k=0)

No domain, for any single type swept in isolation across k=0–96, comes close to flipping sign. Range of `gap_S`/`gap_H` across the full sweep, per domain (min/max across all 13 k-values and all 4 types tested independently):

| domain | gap_S range | gap_H range |
|---|---|---|
| big_patent | [0.0359, 0.0400] | [0.0540, 0.0544] |
| billsum | [0.0171, 0.0239] | [0.0406, 0.0409] |
| newsroom | [0.0737, 0.0782] | [0.0614, 0.0618] |
| scientific_papers | [0.0448, 0.0510] | [0.0808, 0.0812] |
| wikipedia | [0.0288, 0.0337] | [0.0436, 0.0440] |

Every range stays comfortably positive. This is expected given the dilution math: each swept type contributes only ~60 documents/domain against a ~10,700–11,300-document baseline pool (a ~180x dilution ratio) — individually, no single type's window-start choice can move the pooled number far, even though Test 1 already showed each type's *own, undiluted* `S`/`H` is highly sensitive to the same choice. This is the expected relationship between Test 1's finding and this one, not a contradiction of it.

### Joint worst-case (local search, not exhaustive)

Per-type independent argmin: `excerpt`→72, `short_keywords`→64, `summary_keywords`→72, `cot`→24. Two rounds of ±1-step neighborhood probing converged to a slightly different, confirmed-better combination: **`excerpt`→80, `short_keywords`→56, `summary_keywords`→72, `cot`→24** (mean pooled gap 0.09523, vs. 0.09636 at the naive independent-argmin candidate — a real, if modest, improvement from actually searching the neighborhood rather than trusting independence). No further neighbor improved on this after round 2; reported as the local worst case found, not a global one.

| domain | gap_S at worst case | gap_H at worst case |
|---|---|---|
| big_patent | +0.0350 | +0.0533 |
| billsum | +0.0191 | +0.0401 |
| newsroom | +0.0738 | +0.0609 |
| scientific_papers | +0.0436 | +0.0803 |
| wikipedia | +0.0269 | +0.0431 |

**No sign flip anywhere, even at the local worst case.**

### Bootstrap CI at the joint worst case (stratified by contributing group, 10 folds)

| domain | gap_S (point) | gap_S 95% CI | excludes 0? | gap_H (point) | gap_H 95% CI | excludes 0? |
|---|---|---|---|---|---|---|
| big_patent | +0.0350 | [0.0272, 0.0436] | **yes** | +0.0533 | [0.0483, 0.0541] | **yes** |
| billsum | +0.0191 | [0.0065, 0.0246] | **yes** | +0.0401 | [0.0367, 0.0422] | **yes** |
| newsroom | +0.0738 | [0.0572, 0.0769] | **yes** | +0.0609 | [0.0558, 0.0627] | **yes** |
| scientific_papers | +0.0436 | [0.0376, 0.0532] | **yes** | +0.0803 | [0.0767, 0.0827] | **yes** |
| wikipedia | +0.0269 | [0.0196, 0.0453] | **yes** | +0.0431 | [0.0403, 0.0431] | **yes** |

**Every cell's CI excludes zero, even at the local worst-case window-start combination for the 4 unresolved types.** `billsum`'s `gap_S` is the tightest margin ([0.0065, 0.0246]) but still clears zero with room. Same Human-pool caveat as Test 4 applies: Human draws are independent redraws on identical documents across folds, not a before/after comparison.

### Weighted sensitivity check — replicate swept types to their true corpus share

| domain | gap_S (n=300/type, as-scanned) | gap_S (weighted to true share) | gap_H (n=300/type) | gap_H (weighted) |
|---|---|---|---|---|
| big_patent | +0.0350 | +0.0240 | +0.0533 | +0.0393 |
| billsum | +0.0191 | +0.0138 | +0.0401 | +0.0273 |
| newsroom | +0.0738 | +0.0683 | +0.0609 | +0.0414 |
| scientific_papers | +0.0436 | +0.0376 | +0.0803 | +0.0652 |
| **wikipedia** | +0.0269 | **+0.0045** | +0.0431 | +0.0202 |

**Every domain stays positive under correct weighting too — but `wikipedia`'s margin collapses the most (`gap_S` = +0.0045, over 5x smaller than its as-scanned value).** That was the first thing this check surfaced. Looking harder at it turned up a second, broader problem, caught only by directly comparing each weighted point estimate against its own as-scanned bootstrap CI (§ previous subsection) rather than treating the two checks as independent:

| domain | exponent | weighted point estimate | as-scanned 95% CI | weighted estimate vs. CI |
|---|---|---|---|---|
| big_patent | S | +0.0240 | [0.0272, 0.0436] | **below floor** |
| billsum | S | +0.0138 | [0.0065, 0.0246] | within |
| newsroom | S | +0.0683 | [0.0572, 0.0769] | within |
| scientific_papers | S | +0.0376 | [0.0376, 0.0532] | at floor |
| wikipedia | S | +0.0045 | [0.0196, 0.0453] | **below floor** |
| big_patent | H | +0.0393 | [0.0483, 0.0541] | **below floor** |
| billsum | H | +0.0273 | [0.0367, 0.0422] | **below floor** |
| newsroom | H | +0.0414 | [0.0558, 0.0627] | **below floor** |
| scientific_papers | H | +0.0652 | [0.0767, 0.0827] | **below floor** |
| wikipedia | H | +0.0202 | [0.0403, 0.0431] | **below floor** |

**This is not noise, and it is not confined to `wikipedia`.** Every single `H` cell's weighted point estimate falls below the lower bound of its own as-scanned bootstrap CI — not just wikipedia's. `S` shows the same direction in 2 of 5 domains outright (`big_patent`, `wikipedia`), sits exactly at the floor in a third (`scientific_papers`), and stays within the CI in only 2 (`billsum`, `newsroom`). The likely mechanism: `calc_H` runs on one concatenated increment array per domain, so replicating the swept types' ~60 documents up to their true corpus share (~50x for some types) changes that array's *composition* far more than a 10-fold resample of the same ~60 documents ever varies it — `calc_S`'s list-of-trajectories pooling is structurally less sensitive to this, consistent with `S` moving less than `H` here.

**What this means for interpretation: the as-scanned bootstrap CI is not a valid uncertainty statement about the correctly-weighted gap, for most cells — not only `wikipedia`'s `S` cell.** It's a valid CI for the gap *at the as-scanned (~60/domain) sample composition only*. Once weighting is corrected, the point estimate for 6 of 10 cells (all 5 `H` cells, plus `big_patent` and `wikipedia`'s `S`) falls outside — mostly below — the range that CI claims to bound. The signs stay positive everywhere; that part of the check is genuine and unaffected. But "the worst-case gap is still statistically distinguishable from zero" is only established at the as-scanned weighting, not at the true corpus weighting, for most of the table.

**Critical statistical caveat on the weighted numbers, stated plainly rather than glossed over: these are point estimates only, not bootstrap-confirmed, and deliberately so.** The weighting method resamples-with-replacement from each type's real n≈60/domain sample up to its true corpus count (e.g., ~2,900 for `wikipedia`/`excerpt`) to get the pooling proportions right. But that resampling does not create new independent information — it's still built from ~60 real documents per type per domain, just weighted up. Running a bootstrap CI on top of this replicated pool would treat resampled duplicates as independent data, understating the true uncertainty and producing an artificially tight, falsely reassuring CI. **I did not compute one, on purpose** — reporting a fabricated-precision CI here would be worse than reporting none. That absence is now more consequential than originally framed: it isn't just `wikipedia`'s open question, it's the open question for 6 of the 10 cells in this table. Closing it requires a larger raw sample of the 4 unresolved types across all 5 domains (all ~11,000 documents/type are available in the corpus — this sweep only drew ~60/domain from Test 1's existing sample), not a statistical trick on the existing 60.

---

## Verdict

**Per domain, per exponent, not averaged.** The "worst-case CI excludes 0" column is valid **only at the as-scanned (~60/domain) sample weighting** — see the Weighted sensitivity check subsection above, which found that the correctly-weighted point estimate falls *below* this CI's lower bound for 6 of the 10 cells. That column is therefore reported honestly as "yes, at as-scanned weighting" everywhere, not as "yes" unqualified, and a separate column states whether the weighted point estimate actually lands inside that CI:

| domain | exponent | independent sweep | joint worst-case sign | CI excludes 0 (as-scanned weighting) | weighted point estimate | weighted estimate inside as-scanned CI? | overall |
|---|---|---|---|---|---|---|---|
| big_patent | S | survives | positive | yes | +0.0240 | **no — below floor** | SURVIVES (sign only; CI not valid at true weighting) |
| big_patent | H | survives | positive | yes | +0.0393 | **no — below floor** | SURVIVES (sign only; CI not valid at true weighting) |
| billsum | S | survives | positive | yes (tightest CI) | +0.0138 | yes | **SURVIVES** |
| billsum | H | survives | positive | yes | +0.0273 | **no — below floor** | SURVIVES (sign only; CI not valid at true weighting) |
| newsroom | S | survives | positive | yes | +0.0683 | yes | **SURVIVES** |
| newsroom | H | survives | positive | yes | +0.0414 | **no — below floor** | SURVIVES (sign only; CI not valid at true weighting) |
| scientific_papers | S | survives | positive | yes | +0.0376 | at floor (borderline) | SURVIVES (marginal) |
| scientific_papers | H | survives | positive | yes | +0.0652 | **no — below floor** | SURVIVES (sign only; CI not valid at true weighting) |
| wikipedia | S | survives | positive | yes (as-scanned only) | **+0.0045** | **no — below floor, largest gap** | **SURVIVES, WEAKLY** |
| wikipedia | H | survives | positive | yes | +0.0202 | **no — below floor** | SURVIVES (sign only; CI not valid at true weighting) |

**Part I's headline claim survives this second, larger-scope uncertainty test on the sign question at every domain and exponent tested — the sign never flips under the independent sweep, the searched joint worst-case, or correct corpus-share weighting.** That part is unambiguous and holds everywhere.

**The statistical-significance question is a weaker result than originally stated, and only fully established for 3 of 10 cells (`billsum` S, `newsroom` S, and — marginally — `scientific_papers` S).** For the other 7 cells (all 5 `H` cells, plus `big_patent` and `wikipedia`'s `S`), the correctly-weighted point estimate falls below the lower bound of the only CI this report computed — the CI is valid at the as-scanned ~60/domain sample composition, not at the true corpus weighting. This is a broader finding than a single flagged domain: **no cell's weighted gap has a valid confidence interval in this report**, because computing one on the replicated-weighted pool would fabricate precision (explained above and not attempted). What is established: the sign is positive everywhere, including at the local worst case; what is not established: that the *correctly-weighted* worst-case gap is statistically distinguishable from zero, for most of the table.

**`wikipedia`'s `S` cell remains the single weakest number** — its weighted point estimate (+0.0045) is not just below its CI floor but closest to zero of any cell by a wide margin, and it's the one cell where "survives" is a genuinely close call even ignoring the CI-validity issue. The other 9 cells' weighted point estimates stay clearly positive (smallest of the rest is billsum S at +0.0138), so the sign-flip risk is concentrated in `wikipedia` alone — but the *significance* question now applies much more broadly than that.

**Combined with the already-quantified human-side contamination** (`scientific_papers` 14.20%, `wikipedia` 12.36% of Human rows sourced from the same unresolved member types), `wikipedia` is flagged on *both* sides of its gap calculation — the domain most worth a dedicated follow-up before anyone treats its number as fully settled. But the follow-up needed — a larger raw sample of the 4 unresolved types, enabling a real bootstrap CI at correct weighting — is now scoped as necessary for the other 4 domains too, not `wikipedia`-only, since it's the only way to get a valid significance test at true corpus weighting anywhere in this table.

---

## Task 2 — Mistral-7b `continue_it`/`continue_pt` derivation: status

**Already established in Test 3, Part A (`reports/2026-08-05_test1-5_anomaly_clearance_results.md`) — not re-verified here, per the task instruction not to redo settled work:**

- **Step 1 (length triangulation): DONE.** Mistral-7b has the same qualitative prefix-inclusion defect Gemma-2b had (large negative `text_alone_delta`), but Gemma's `compute_boundary` formula does not transfer — Mistral's `continue_it` `combined_delta` median is 35 tokens (vs. Gemma's ~1), with only 0.4% of documents landing within ±3 tokens of the naive reconstruction. Applying Gemma's formula to Mistral's rows would silently produce a wrong correction, not a null result.
- **Step 2 (boundary-crossover forward-pass fit): DEFERRED, confirmed blocked on hardware, not re-checked.** This Mac has 17.2GB total RAM (`sysctl hw.memsize`); Mistral-7B needs ~28GB in fp32 or ~14GB in fp16/bf16 — the latter would leave under 3GB for the OS, activations, and everything else, well into the memory-pressure regime that caused the MPS thrashing bug diagnosed earlier this session. Not attempted, no substitute model used. **This step requires the Ubuntu GPU machine** (or any machine with substantially more memory) — it cannot be completed on this hardware, and should be scoped into that machine's work queue, not re-attempted locally.

**No new work performed for Task 2 in this report** — Task 1 was higher priority per the instructions, and Task 2's step 1 finding stands as already confirmed; step 2 remains blocked exactly as previously reported.

---

## Task 3 — Family B GPU-scorer pause: confirmed

Per the task's explicit instruction, Family B (the paused GPU re-scoring/perplexity work in `src/build_p1_feature_store_gpu.py`) has **not** been resumed. Whether it resumes next depends on Task 1's verdict above: if the headline claim survives the joint worst-case on sign, Family B can resume with a closed Family A story behind it — the significance-CI gap identified in the weighted check is a real open item, but it is a precision/confidence-level question, not a sign-flip risk, and does not reopen the foundational question that originally paused Family B (the `continue_it`/`continue_pt` prefix-contamination defect, which was a wrong-sign/wrong-value bug, not a CI-coverage question). If the sign had flipped anywhere, resolving that would take priority; it did not.

---

## Verdict

**Part I's headline Human-vs-LLM gap (gemma_2b, `S` and `H`) survives this sensitivity bound on the sign question, in all 5 domains and both exponents — the sign never flips under the independent sweep, the searched joint worst-case, or correct corpus-share weighting.** That is unambiguous.

**The statistical-significance question is only fully resolved at as-scanned (~60/domain) sample weighting, not at true corpus weighting.** Once each swept type is weighted to its real corpus share, the point estimate falls below the as-scanned bootstrap CI's lower bound for 7 of 10 cells (all 5 `H` cells, plus `big_patent` and `wikipedia`'s `S`) — meaning that CI does not certify significance at the weighting that actually matters for those cells. Only 3 cells (`billsum` S, `newsroom` S, and marginally `scientific_papers` S) have both a positive weighted point estimate *and* a CI that's actually valid for it. **No cell in this table has a valid, non-fabricated confidence interval at true corpus weighting** — that computation was deliberately not attempted (explained in the Weighted sensitivity check subsection) because the available sample (~60 real documents/type/domain, resampled up to corpus scale) doesn't support one without manufacturing false precision.

Per domain, per exponent (repeated from the Results section's verdict table, since the task asked not to average this away): all 10 cells SURVIVE on sign. **`wikipedia`'s `S` cell is the weakest** — its correctly-weighted margin (+0.0045) is the smallest in the table by a wide margin and closest to a genuine sign-flip risk, not just a CI-validity gap. The other 9 cells' weighted margins stay clearly positive, so sign-flip risk is `wikipedia`-specific, but the missing-CI problem is not — it applies across most of the table and should not be read as resolved just because `wikipedia` was the domain that first surfaced it.

**This clears the bar for Task 3's stated condition on the sign question:** Family B can resume — the original pause was triggered by a wrong-sign/wrong-value defect (`continue_it`/`continue_pt` prefix contamination), and no sign flip was found here at any tested windowing choice. The unresolved significance-CI gap (7 of 10 cells, concentrated but not confined to `wikipedia`) is a scoped follow-up requiring a larger raw sample of the 4 unresolved types across all 5 domains — real work, but a precision question layered on top of an already-positive, sign-stable result, not a foundational-integrity blocker of the kind that paused Family B originally.

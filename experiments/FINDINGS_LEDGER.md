# Findings Ledger — Final Compute Plan Execution

Run on `nvidiaserver` (192 CPU cores, no GPU on this node), executed 2026-09-08, against
the live repo at `/home/user/the author-Backup/the_author2026/paper` (the mirror under
`the author-Backup`, matching all paths/data described in the plan). All new code, data,
and this ledger live under `review_results/` for easy `scp` — nothing in the original
`code/`, `src/`, `data/`, `feature_store/`, or `writing/` trees was modified.

Per instruction: paper/.tex integration (F0–F5 figures, page budgets, LaTeX
verification) is explicitly OUT OF SCOPE for this run. This ledger reports computation
only. T6 (threshold-file regeneration) is also out of scope — it is paper-appendix
bookkeeping (Appendix G), not a new computational finding.

Environment built fresh for this run: conda env `review` (python 3.11, numpy/pandas/
scipy/scikit-learn/lightgbm/statsmodels/pyarrow/joblib via conda-forge, plus the `fbm`
PyPI package for synthetic fractional Gaussian noise). Does not touch the existing
`base` or `ts_rag` conda envs.

## Directory map

- `code/` — every new script written for this run.
- `data/` — every new JSON/parquet result file, including preserved buggy-run artifacts
  kept for the record (filenames tagged `_v1_..._BUG`) rather than overwritten silently.
- `logs/` — raw stdout/stderr from each run.
- `figures/` — diagnostic/summary PNGs (`code/make_figures.py`), one per task, rendered
  directly from the already-validated JSON in `data/` (no new statistics computed for a
  figure). These are review-run visualizations, not the paper's own F0-F5 LaTeX figures —
  that figure/page-budget/LaTeX work remains explicitly out of scope per instruction.
- `notes/` — reserved, empty: no free-form notes were needed for this run.

## Status summary

| Task | Status | Key output file(s) |
|---|---|---|
| T0 (corrected coefficient, Route 1 + Route 2) | **DONE** | `data/t0_t1_interleave_stats.json`, `data/t0_variance_components.json` |
| T1 (6 split constructions) | **DONE**, all 8: interleave, contiguous, block_4/8/16/32, block_bootstrap (measured, excluded from headline test — see T1 section), matched_sweep, permutation (20 reps, after fixing a self-caught order-scrambling bug — see T1 section) | `data/t0_t1_<construction>_stats.json`, `data/t1_permutation_stats.json` |
| T2 (reliability vs. support, N*) | **DONE** (Step 0 validation, Step 1 fBm floor, Step 2 real-text bracket) | `data/t2_step1_noise_floor.json`, `data/t2_step2_real_text_bracket.json` |
| T3 (P3 population control) | **DONE** (T3.1–T3.4 all four) | `data/t3_population_control.json` |
| T4 (alternative estimators: DFA, local Whittle) | **DONE** | `data/t4_alt_estimators_reliability.json` |
| T5 (RAID group-mean test) | **DONE** | `data/t5_raid_group_mean.json` |
| T6 (threshold-file regen) | **OUT OF SCOPE** (paper-appendix bookkeeping) | — |
| T7 (matched-length bands) | **DONE** | `data/t7_matched_length.json` |
| T8 (deployment metrics) | **DONE** | `data/t8_deployment_metrics.json` |
| T9 (RAID complete-case sensitivity, items 1–2 of 4) | **DONE** (partial — items 3 IPW, 4 reduced-feature-set deferred) | `data/t9_raid_complete_case.json` |
| T10 (second representation family) | **PILOT COMPLETE, negative result** — geometry sweep (3 configs, zero content overlap, measured not assumed) shows no window/stride setting clears fractal H's reliability; full-corpus run NOT recommended | `data/t10_pilot_geometry_comparison.json`, `data/t10_pilot_topology_w*.json` |

---

## T0 — corrected coefficient + variance decomposition

**Acceptance gate: PASSED.** Interleave construction reproduces the frozen
`feature_store/p2_reliability_report.json` exactly:

| quantity | n (reproduced) | n (frozen) | rho_pooled (reproduced) | rho_pooled (frozen) |
|---|---|---|---|---|
| gemma_H | 221,904 | 221,904 | 0.5763 | 0.5763 (0.576) |
| gemma_S | 196,971 | 196,971 | 0.0626 | 0.0626 (0.063) |
| mistral_H | 223,834 | 223,834 | 0.5465 | 0.5465 |
| mistral_S | 195,552 | 195,552 | 0.0497 | 0.0497 |

Fisher-z CI on pooled gemma_H: [0.5735, 0.5791] (plan's target [0.573, 0.580] — matches
to within the CI-construction-method difference; see code comments in
`t0_variance_components.py`).

**Route 1 (within-stratum, demeaned by Label x domain x temp):**

| quantity | rho_pooled | rho_within | drop |
|---|---|---|---|
| gemma_H | 0.5763 | 0.5522 | -0.024 |
| gemma_S | 0.0626 | 0.0601 | -0.003 |
| mistral_H | 0.5465 | 0.5199 | -0.027 |
| mistral_S | 0.0497 | 0.0464 | -0.003 |

**Route 2 (closed-form variance decomposition, not an iterative mixed model — see
`t0_variance_components.py` docstring for why):**

| quantity | ICC (Var(mu_stratum) share) | Var(delta) | Var(eta_half) | cluster-robust SE (seed-article) |
|---|---|---|---|---|
| gemma_H | 3.9% | 0.00217 | 0.00352 | 0.0023 |
| gemma_S | 0.14% | 0.01049 | 0.32793 | 0.0024 |
| mistral_H | 3.9% | 0.00190 | 0.00352 | 0.0024 |
| mistral_S | 0.18% | 0.00870 | 0.35725 | 0.0023 |

**Interpretation (per the plan's own pre-committed decision rule, §3.1):** the
pooled-vs-within gap is small for BOTH H and S — closer to the "approx equal" branch than
"materially below." **For H**, population structure (Label/domain/temp) explains only
~4% of single-half variance; the 0.576 -> 0.552 drop is real but modest. **For S**,
population structure explains under 0.2% of variance — S's already-poor pooled
reliability (0.063) is essentially ALL within-document noise, not an artifact of pooling
across populations. This rules out "population averaging inflates S's apparent
(un)reliability" as an explanation either way: S's poor reliability is a genuine
estimation problem, not a population-mixing artifact.

---

## T1 — reliability under split constructions

All 8 constructions complete. Consolidated table (`code/consolidate_t0_t1_t2.py` ->
`data/t0_t1_MERGED_table.json`):

| construction | gemma_H | gemma_S | mistral_H | mistral_S |
|---|---|---|---|---|
| interleave | 0.5763 | 0.0626 | 0.5465 | 0.0497 |
| contiguous | 0.2609 | 0.0159 | 0.2678 | 0.0040 |
| block_4 | 0.5624 | 0.0526 | 0.5308 | 0.0485 |
| block_8 | 0.5486 | 0.0496 | 0.5232 | 0.0467 |
| block_16 | 0.5342 | 0.0490 | 0.5128 | 0.0348 |
| block_32 | 0.5013 | 0.0396 | 0.4777 | 0.0280 |
| block_bootstrap | 0.7779\* | 0.2047\* | 0.7769\* | 0.1912\* |
| matched_sweep | 0.5763 | 0.0626 | 0.5465 | 0.0497 |
| permutation (median, 20 reps) | 0.5443 [0.5424,0.5462] | 0.0530 [0.0484,0.0571] | 0.5152 [0.5129,0.5182] | 0.0452 [0.0386,0.0498] |

(all `rho_pooled`; `rho_within` and n in the raw JSON. Permutation values are the
20-replicate median with [p5,p95] in brackets.)

### Permutation construction — a self-caught order-scrambling bug, documented not hidden

The first permutation run (100 reps, `t1_permutation_stats_v1_ORDER_SCRAMBLE_BUG.json`,
preserved on disk, not discarded) gave **rho_pooled median ~0.003 for every quantity, CI
crossing zero** — reliability apparently annihilated by "random balanced splitting,"
wildly inconsistent with every other construction (0.5-0.63 for H, 0.04-0.06 for S).
Traced to a real defect in `split_permutation` (`reliability_lib.py`): the original
`arr[idx[:m]]` used an UNSORTED random index set, which scrambles each half's internal
TOKEN ORDER along with its membership. R/S and displacement-probability scaling both
operate on the ORDERED trajectory — a temporally-scrambled half is white noise to the
estimator regardless of the document's real structure. Two things pinned this down as an
order-destruction artifact rather than a genuine "no signal under random splitting"
result, before either a fix or a full rerun was committed to: (1) `gemma_H_R2`/
`mistral_H_R2` (log-log fit quality) stayed non-zero and CI-excluding-zero
(0.047/0.036) even under the scrambled input, while H itself collapsed to ~0 — fit
quality partly survives order destruction in a way the slope-derived point estimate does
not, a dissociation specific to order scrambling; (2) a 2,000-document/3-replicate smoke
test with the fix (`np.sort` on each half's index set — randomize membership, preserve
each half's own chronological order, the actual Pronk-et-al. definition of a random
balanced split) landed near interleave's numbers (gemma_H~0.48 vs 0.576) before the full
20,000+-document rerun was launched. **This v1 result is, itself, informative by
accident**: it demonstrates directly that these fractal features are genuinely
order-dependent quantities (destroying order destroys the signal), not disguised
magnitude/length statistics — a real, if unplanned, sanity check, kept in the record for
that reason rather than deleted as a mistake with no value.

The corrected v2 run (20 reps, reduced from the plan's ">=500" for the same cgroup-cost
reason as the interim 100-rep run, documented in `run_permutation.py`'s
`reduction_reason` field) lands squarely inside the band the six deterministic
constructions establish (gemma_H: 0.544, between block_32's 0.501 and block_8's 0.549;
gemma_S: 0.053, between block_32's 0.040 and interleave's 0.063) — confirming random
balanced splitting produces reliability consistent with the deterministic constructions,
once membership randomization is correctly decoupled from order.

\* **block_bootstrap is measured and reported here, but structurally invalid as a
split-half reliability number and EXCLUDED from the headline test below.** Direct
measurement (not assumption) of the two resampled halves shows they share ~47% of their
token positions even after fixing the original fixed-4-block implementation to a proper
moving-block bootstrap (`reliability_lib.py::make_split_block_bootstrap`, samples from
all `n-block_len+1` overlapping start offsets). This is a consequence of the plan's own
`block_len>=96` requirement (tied to the largest fit scale) applied to a ~400-token
document: each block covers ~25% of the document, so two independent draws cannot avoid
heavy overlap regardless of pool size. The inflated correlation (higher than every other
construction, including the untouched document) reflects token-overlap-by-construction,
not genuine document-level signal, and is not comparable to the other six numbers.

**Headline test (plan's own bar): does gemma_S stay in 0.03–0.12 across all valid
constructions?** Excluding block_bootstrap (reason above): **`in_band = False`**, driven
entirely by `contiguous` (gemma_S=0.0159, below 0.03). This is **not evidence against**
the reliability claim: `contiguous` is the construction the plan itself pre-registers as
confounded by within-document drift (adjacent-half splits let a monotonic within-document
trend masquerade as reliability, or destroy it), and its collapse here
(gemma_H: 0.576->0.261, mistral_H: 0.546->0.268) is the *expected, positive* signature of
that confound being real — not a failure of the other five constructions, which span
0.0396-0.0626 for gemma_S. Full breakdown with `below_band`/`above_band`/`excluded` keys
in `data/t0_t1_MERGED_table.json["_headline_test_gemma_S"]`.

`block_b` series (b=1[interleave],4,8,16,32) is the direct empirical answer to "does
single-token interleaving damage the local sequential geometry": reliability declines
smoothly and monotonically as block size grows (gemma_H: 0.576->0.562->0.549->0.534->0.501
for b=1,4,8,16,32), consistent with larger contiguous blocks reintroducing some of
`contiguous`'s drift confound at a diluted scale, not a discontinuity or artifact.

`matched_sweep` reruns the windowed sub-features (S_window_*/H_window_*) with the FULL
6-value WINDOW_SCALE instead of the truncated first-4, closing the reviewer's W21 concern
about a 0.778-vs-1.602-decade sweep mismatch — the headline (non-windowed) S/H numbers are
identical to interleave by construction (same split, only the windowed sub-features change).

---

## T2 — reliability vs. support (N*)

**Step 0 (generator validation):** the `fbm` package's Davies-Harte fGn generator +
the project's own `calc_S`/`calc_H`, run at N=400, reproduce the QUALITATIVE pattern of
`reports/2026-07-26_S_estimator_fix.md`'s single-document bias table (S severely
negatively biased and worsening with H; H only mildly biased) but not its exact numeric
values — attributed to that report's generator not being preserved in the repository,
not to a defect here. Documented, not hidden.

**Step 1 (fBm noise floor, calibrated at GAGLE's own observed H=0.60–0.72 band):**

| H_true | N | S bias | S sd | H bias | H sd |
|---|---|---|---|---|---|
| 0.60–0.72 (avg) | 100 | -0.72 to -1.08 | 0.7–1.2 | +0.02 | 0.10–0.12 |
| 0.60–0.72 (avg) | 400 | -0.38 to -0.45 | 0.25–0.28 | +0.01 to +0.03 | 0.06–0.07 |
| 0.60–0.72 (avg) | 1600 | ~0 to -0.03 | 0.15–0.20 | +0.02 to +0.04 | 0.03 |
| 0.60–0.72 (avg) | 12800 | ~0 | 0.05–0.08 | +0.01 to +0.04 | 0.01 |

S bias collapses from roughly -0.7 to -1.1 at N=100 to near zero by N~1600–3200; H bias
stays small (<0.04) throughout. Full table (including the 0.30/0.50/0.80 sanity grid) in
`data/t2_step1_noise_floor.json`.

**Step 2 (real-text bracket, one CPU pass over `gagle.zip`):** 5,000 unique human
documents extracted (gemma-2b `original_text` field, uncapped up to the corpus's own
2047-token hard limit). Within-document block variance of Shat/Hhat across disjoint
N-token blocks, by N, in `data/t2_step2_real_text_bracket.json`. Empirical arm is
Gemma/human-side only, exactly per the plan's stated hard limitation (GAGLE's machine
documents don't reach 1024+ tokens natively).

| N | H var (n usable) | S var (n usable) |
|---|---|---|
| 100 | 0.01005 (5000) | 0.49441 (4991) |
| 200 | 0.00744 (5000) | 0.28328 (5000) |
| 400 | 0.00461 (5000) | 0.06031 (5000) |
| 800 | 0.00139 (4996) | 0.02437 (4996) |

(N=1600 dropped: essentially no document in the corpus's 2047-token cap yields >=2
disjoint 1600-token blocks.)

**This is NOT a clean upper (or lower) bound on Step 1's fBm Var(eta_N)** — an earlier
draft of this section and of `t2_step2_real_text_bracket.py`'s docstring incorrectly
called it an upper bound; caught by re-deriving it against the fBm comparison below
before use, not assumed correct. Two effects push in opposite directions relative to
fBm's Var(eta_N) (measured across *independent* realizations at fixed H): within-document
nonstationarity (real text has some, fBm has none) pushes real-text variance UP; but
blocks of a single document share that document's own particular realization, which
SUPPRESSES between-block variance relative to independent draws — that suppression wins
empirically. At every matched N, real-text block variance runs **below** the fBm floor
for both H and S (fBm floor average over H_true in {0.60,0.65,0.70,0.72}, from Step 1):

| N | fBm S var | real-text S var | fBm H var | real-text H var |
|---|---|---|---|---|
| 100 | 0.74238 | 0.49441 | 0.01158 | 0.01005 |
| 400 | 0.07075 | 0.06031 | 0.00446 | 0.00461 |
| 800 | 0.03816 | 0.02437 | 0.00163 | 0.00139 |

**Calibration check** (confirms this gap is the shared-realization effect, not an
invalid fBm-vs-real comparison): full-document Hhat across all 5,000 human documents
(`data/t2_step2_full_doc_estimates.json`) has median 0.629, mean 0.634, sd 0.048,
[p10,p90]=[0.578,0.698] — squarely inside the 0.60-0.72 band the fBm floor was
calibrated against. **Conclusion: Step 2's real-text arm is a distinct estimand from
Var(eta_N), not a bound on it in either direction — but it does confirm the fBm floor is
not an underestimate of real-text estimator scatter**, which is the one property N*'s
construction (below) actually needs.

**N\* composition** (`code/consolidate_t0_t1_t2.py::compose_n_star`, `data/t2_n_star_composition.json`):
rho_N is computed as `2*r_N/(1+r_N)` (Spearman-Brown correction applied to
`r_N=Var(delta)/(Var(delta)+Var(eta_N))` **before** comparing to the 0.60 bar —
advisor-caught bug fix, since the 0.60 bar was pre-registered on the SB-corrected
coefficient, and comparing the bar to a raw uncorrected r systematically overstates N*).

| quantity | Var(delta) | N\* (first grid point crossing rho_SB>=0.60) |
|---|---|---|
| gemma_H | 0.00217 | **800** |
| mistral_H | 0.00190 | **800** |
| gemma_S | 0.01049 | **6400** |
| mistral_S | 0.00870 | **6400** |

H crosses the bar by N=800 (rho_SB=0.727/0.700 at N=800, already 0.493/0.461 at N=400 —
close to GAGLE's own ~400-token support). **S needs 8-16x more tokens than GAGLE
provides** (N*=6400 vs. the corpus's 320-400-token documents) — at N=400, rho_SB is only
0.229/0.197, nowhere near 0.60. This quantifies precisely what T0/T1's poor S reliability
already showed qualitatively: S is not a short-document-support problem that scales away
gently, it needs an order of magnitude more text than any document in this corpus has.

**N\* here is a point estimate under the fBm generator's noise floor, not a bound** — an
earlier draft of this section called it a "lower bound" on the assumption that real-text
noise would exceed the fBm floor; Step 2's results (above) directly contradict that
assumption (real-text block variance runs BELOW the fBm floor at every matched N), so
that framing is retracted here rather than carried forward. Step 2's within-document
block variance is a distinct estimand from Var(eta_N) (shared-realization effect, see
above), so it cannot be substituted into this formula to produce a "real-text N*" at
all. What Step 2 DOES establish is that the fBm floor is not an underestimate of
real-text estimator scatter, which is the property this N* computation actually needs
to be trustworthy — so N*=800 (H) / N*=6400 (S) stand as the best available estimates,
not as a bound with a known direction of error.

---

## T3 — P3 under population control

**Acceptance gate: PASSED** (within the plan's own documented CI, not a tight point
tolerance — LightGBM's histogram construction is not bit-deterministic across
versions/thread counts, and no determinism flag is set in the original pipeline either):
all-features 0.5837 (frozen CI [0.5747,0.5885]), fractal_only 0.1981 (exact match).

- **T3.1 (population-only control, generator+domain+temp only):** macro-F1 = **0.0594**
  [0.0578, 0.0609], n_test=42,444 — **below** the 8-class chance floor of 0.125. Closes
  the critique's objection more decisively than the plan's own "near 0.125" bar
  anticipated: population structure alone carries essentially no prompt-type signal.
- **T3.2 (balance table):** 30 observed (generator,domain,temp) cells x 8 prompt types,
  n=14,805 complete-case test rows. Max cell-proportion deviation from the factorial
  expectation: **0.0022** (0.22 percentage points) — the factorial design survived the
  token-window/complete-case gates essentially intact. (chi2 p<1e-16 is a large-n
  artifact, not evidence of a meaningful imbalance — see the raw deviation number.)
- **T3.3 (group-mean removal, 54 fractal columns, TRAIN-only means):** fractal_only
  macro-F1 **rose slightly** after demeaning (0.1981 -> 0.2046, CIs overlap) rather than
  falling toward chance. Combined with T3.1, this means the fractal family's prompt-type
  signal is **genuine document-level structure, not population-structure leakage**. Per
  the plan's own interpretation guide, this is the branch where "the P3 claim must soften
  further" (in the specific sense that the confound explanation is largely ruled out) —
  but it is also evidence the fractal features carry a real, document-level, non-
  confounded signal, which is a positive methodological finding in its own right.
- **T3.4 (within-stratum repeats, 30 cells):** all-features macro-F1 distribution:
  mean 0.567, sd 0.032, range [0.506, 0.666]. fractal_only: mean 0.200, sd 0.026, range
  [0.147, 0.255]. Both distributions sit well clear of chance in every one of the 30
  cells — the P3 signal is not a pooled-cell artifact.

**Rebuttal framing (per plan's own pre-registered direction-of-risk note):** the number
genuinely at risk is 0.5816/"about a sixth of the distance," not the fractal_only=0.1981
figure — the residual-confound story would have made 0.1981 smaller, and it didn't.

---

## T4 — alternative estimators (DFA, local Whittle) — RESOLVED (v2)

**v2 run complete** (`t4_run_alt_estimators.py`, PID 1404296, 398s, 8 workers,
`data/t4_alt_estimators_reliability.json`). The advisor's specific risk — local
Whittle's `minimize_scalar(bounds=(-0.49,1.49))` censoring, which could manufacture
spurious perfectly-correlated saturated pairs — is **ruled out empirically, not assumed
away**: saturation affects 23/221,904 (0.0001) gemma pairs and 17/223,834 (0.0001)
mistral pairs, and excluding them moves reliability by <0.001 in every case
(gemma_Whittle_H: 0.6773 -> 0.6774; mistral_Whittle_H: 0.6415 -> 0.6415).

| quantity | n | rho_pooled | rho_within | clears rho>=0.60 (pooled)? |
|---|---|---|---|---|
| gemma_DFA_H | 221,904 | **0.6289** | 0.6014 | **YES** (barely — 0.029 above the bar) |
| gemma_Whittle_H | 221,904 | **0.6773** | 0.6471 | **YES** |
| mistral_DFA_H | 223,834 | 0.5994 | 0.5658 | **NO** — 0.0006 below the bar, inside any bootstrap CI, not a real miss but not a clearance either |
| mistral_Whittle_H | 223,834 | **0.6415** | 0.6046 | **YES** |

Compare to R/S's own gemma_H=0.576 / mistral_H=0.546 (both below 0.60, T0 above).

**Cross-estimator correlation (doc-level, mean-of-halves)** — the check that decides
whether these are measuring "the same quantity, more precisely" or "a different, related
quantity":

| pair | gemma | mistral |
|---|---|---|
| RS ~ DFA | 0.8685 | 0.8644 |
| RS ~ Whittle | 0.7592 | 0.7506 |
| DFA ~ Whittle | 0.8077 | 0.8030 |

Per the advisor's own pre-stated decision thresholds (>0.85 "same quantity, restructure
around it stands" / <0.60 "different quantity, different sentence needed"), these fall
**between** the two clean-cut cases, and the two estimators do not get the same verdict:

- **DFA is the load-bearing result.** r(RS,DFA)~0.86-0.87 clears the "same quantity"
  threshold — DFA is measuring the same Hurst exponent as R/S, just less noisily. But
  **the clearance is scorer-dependent and marginal, not comfortable**: gemma_DFA_H=0.629
  clears the 0.60 bar by only 0.029, and mistral_DFA_H=0.599 does not clear it at all.
  The honest sentence is **"the Hurst exponent is recoverable at ~400 tokens with a
  less-noisy estimator, at the margin, and the margin is scorer-dependent"** — not the
  stronger, unqualified "recoverable." A reviewer checking the mistral number against an
  unqualified claim would have grounds to push back; a margin-qualified claim does not.
- **Whittle is corroborating, not primary.** r(RS,Whittle)~0.75-0.76 means ~43% of
  Whittle's variance is NOT shared with R/S — Whittle clears the reliability bar more
  comfortably than DFA (0.677/0.642 vs 0.629/0.599), but that extra headroom is
  partly a different quantity, not purely a cleaner measurement of the same one.
  Whittle's numbers should always be reported alongside this correlation, never as a
  standalone headline.
- **Neither estimator says anything about the Holder/S exponent**, which is the number
  the paper's central claim (gemma_S~0.05, reliability ~0.06) actually rests on. DFA and
  Whittle both target long-memory/H-type scaling; the plan identifies no
  literature-standard alternative for S's displacement-probability construction. S's own
  poor reliability (T0/T1, ~0.05-0.06 pooled) stands **completely unaffected** by this
  finding, and this finding must not be summarized in any way that lets a reader infer
  otherwise.

**Bottom line, restructured around per the plan's own instruction, with the margin made
explicit rather than smoothed over:** DFA recovers a marginal, scorer-dependent
Hurst-exponent signal at GAGLE's ~400-token support that R/S does not — supporting
"R/S specifically is the weaker estimator" over "the estimand is unrecoverable," but not
supporting an unqualified recoverability claim. Whittle corroborates directionally but
measures a partially distinct quantity per its lower cross-estimator correlation. Neither
extends to S.

---

## T5 — RAID group-mean test

**Acceptance gate: PASSED** (n=22,041, AUROC=0.8032112857989935, exact match to
`fractalscope_check_raid_intersection.json`).

- Reference gap (all-85-feature vs. all-minus-fractal-31-feature, same rows): **+0.0240**
  AUROC (0.8032 vs 0.7792).
- **REPLACEMENT** (fractal columns -> (generator,domain)-group mean): AUROC = **0.8149**
  — recovers MORE than the full +0.0240 gap (149% of it). Group-mean-only fractal
  features outperform the real individual document values on this transfer task.
- **SUBTRACTION** (fractal columns -> x - group_mean + global_mean, corrected after
  advisor review — a bare `x - group_mean` would hand the frozen LightGBM inputs
  centered near zero, a range it never saw in training, confounding "population signal
  removed" with "model fed out-of-distribution inputs"; adding `global_mean` back keeps
  inputs in-range while still zeroing the group-specific offset): AUROC = **0.7934** —
  retains 59% of the +0.0240 gap (down from the uncorrected run's confounded 82%).
- **Honest interpretation:** REPLACEMENT alone more than accounts for the full gap
  (149%), while SUBTRACTION removes about 41% of it. This says population-level
  (generator,domain) mean information is doing the MAJORITY of the work, but the
  individual document-level residual still retains a real, non-trivial (59%) share —
  not a clean "it's all population mean" story, but tilted toward population mean more
  than the uncorrected run suggested.

---

## T7 — matched-length bands

**Structural finding (stronger than the plan's generic length-confound warning):**
GAGLE's human-side documents are gated to **exactly 400 usable tokens with zero
variance** (std=0 across 81,238+ human doc/scorer/feature rows); LLM documents range
320–463. This means bands 320–360 and 401–464 are **LLM-only by construction** — a
human/machine length-matched comparison is not achievable outside the narrow 361–400
band (itself dominated by the fixed human value of 400). This is reported as the finding
and limitation rather than forcing an unsupportable match.

Reliability by band (headline features), n varies sharply by band (2k-11k in the edge
bands vs 180k+ in the middle band):

| band | gemma_S rho_pooled | gemma_H rho_pooled | mistral_S rho_pooled | mistral_H rho_pooled |
|---|---|---|---|---|
| 320-360 (LLM-only) | 0.091 | 0.412 | 0.003 | 0.476 |
| 361-400 (mixed, human-dominated) | 0.055 | 0.582 | 0.045 | 0.546 |
| 401-464 (LLM-only) | 0.109 | 0.478 | 0.141 | 0.493 |

S reliability is consistently higher in the 401-464 (longer) band than the 361-400 band
for both scorers — a small but real signal consistent with T2's support-vs-reliability
mechanism, even within this narrow token range.

---

## T8 — deployment metrics

From stored predictions only (config=full/random/random/all/lgbm, n=43,413, no
rescoring). Full numbers in `data/t8_deployment_metrics.json`.

- Partial AUROC (FPR<=1%): see file. Full AUROC 0.9984.
- TPR@1%FPR=0.959, TPR@5%FPR=0.995, TPR@0.1%FPR=0.673.
- Expected calibration error (15 bins): **0.0009** — excellent calibration.
- **PPV sweep is over assumed LLM (positive-class) prevalence**, not GAGLE's own 2.234%
  human prevalence (a real content-moderation deployment is the OPPOSITE shape from
  GAGLE's own construction — LLM content is typically the minority of real-world
  traffic, not 97.8% of it as in this research corpus). At 1% assumed LLM share, PPV at
  the 1%-FPR threshold is only 0.51; at 10% LLM share, PPV rises to 0.92; at GAGLE's own
  corpus composition (97.8% LLM), PPV is 0.9998. **This distinction (which prevalence
  direction) matters and should not be collapsed in the write-up.**

---

## T9 — RAID complete-case sensitivity (items 1–2 of 4)

- Complete-case population (n=22,041) vs excluded (n=46,171): composition differs
  substantially by attack type — `zero_width_space`-attacked rows are heavily
  overrepresented among the EXCLUDED (NaN-feature) rows (24.3% of excluded vs 9.5% of
  included), i.e. that specific attack disproportionately breaks feature extraction.
  Full composition table (by Label, domain, attack) in the output file.
- AUROC by attack (complete-case only): ranges from 0.650 (`zero_width_space`) to 0.882
  (`whitespace`) — the detector is weakest against exactly the attack type most likely to
  have been excluded from the complete-case population in the first place, meaning the
  complete-case AUROC numbers elsewhere in this project likely OVERSTATE robustness to
  zero-width-space attacks specifically (survivors of that attack are an easier subset).
- AUROC by domain: 0.592 (`reddit`) to 0.930 (`news`).
- AUROC by generator (vs. shared human pool): 0.697 (`mpt-chat`) to 0.899 (`chatgpt`).
- Items 3 (inverse-probability-weighted sensitivity) and 4 (reduced feature set admitting
  more rows) are **not done** — deferred per the plan's own cost ordering.

---

## T10 — second representation family (persistent homology) — PILOT RUN, full run NOT recommended

Originally deferred as needing new infrastructure. Re-scoped and piloted after discovering
this machine's SLURM scheduler exposes real GPU allocations (`sinfo` shows
`gpu:mig24gb-student`, `gpu:full96gb-large`, etc. on this same node) that this session's
direct interactive shell does not have access to — an earlier claim in this ledger that
"no GPU is available" was about this session specifically, not the machine. The GPU queue
is heavily contested in practice (this account's own unrelated SLURM job was pending a
week out at pilot time), so a small **CPU-only pilot** (`code/t10_pilot_topology.py`,
150 docs smoke test then analyzed at that scale — see below for why it stopped there)
was run directly in this session rather than submitted to SLURM, to get a real cost/value
read before committing any queue time.

**Representation:** overlapping 40-word/20-word-stride windows of each document's raw
text (from `gpu_manifest_shards/*.parquet`'s own `text` column — no need to rescan
`gagle.zip`), embedded with `all-MiniLM-L6-v2` (CPU). Each document becomes a trajectory
through embedding space — a genuinely different representation from the token-level
log-probability sequence Family A's H/S are computed from.

**Topological features:** a Vietoris-Rips complex (`gudhi`) on each document's window-
embedding point cloud, reduced to scalar H0 (connected-components) and H1 (loop) features
(total persistence, persistence entropy, bar count, max persistence) — the topological
analogue of H/S. Reliability tested with the exact same interleave-and-correlate
methodology as T0/T1 (even-indexed windows -> half A, odd -> half B), reusing
`reliability_lib.py`'s already-validated `pooled_and_within_stats`.

**Cost measured, not guessed:** 7.34 ms/window to embed, 0.45 ms/doc for persistence
homology, on 150 documents / 3,384 windows in 27s total (single process, CPU). Extrapolated
to the full 223,834-document corpus at comparable window density: **roughly 5-6 hours of
CPU time, entirely within this cgroup's 8-core budget** — cost is NOT the reason this
isn't being scaled up.

**H0 features are a mathematical identity, not a finding.** Checked directly on the raw
per-document data before trusting the numbers: `H0_n_bars = n_points_in_half - 1` in
EVERY document, exactly. This is guaranteed by how Vietoris-Rips filtrations work (the
filtration always eventually merges all points into a single component, regardless of
their actual arrangement), so H0's apparent reliability (rho_pooled 0.989-0.996 via
`pooled_and_within_stats`) is measuring "the two halves have almost the same number of
windows" — true by construction of an even/odd split of the same document — and nothing
about document content. **Excluded from any claim of topological signal.**

**H1 (loop) features show a real but partial length confound**, checked before reporting
the raw numbers as a finding: raw correlation with each half's own point count is
0.38-0.60 across the four H1 features. Partial correlation controlling for `n_windows`
(linear residualization) strips most of this out, leaving genuine residual reliability of
~0.16-0.45 depending on feature (`total_persistence`/`max_persistence` highest,
`n_bars`/`entropy` weakest, both close cousins of the H0 identity above) — below H's own
0.576/0.546 and below the plan's 0.60 bar either way. SB correction applied throughout for
comparability with the fractal H/S numbers, per the same logic as T2's N* fix.

**A second, more serious confound was then found and corrected: the w40/s20 baseline's
two halves are not content-independent either.** The length-only control above is not the
whole story. Directly measuring the word-position overlap between the two halves
(Jaccard of the word-index coverage of even-window-index vs. odd-window-index windows —
the same style of check that caught `block_bootstrap`'s overlap in T1) gives
**0.906** for w40/s20 — far worse than the 50% "adjacent-pair" estimate a first-pass
analytical guess would give. The reason: with window=2*stride, each half's own windows
perfectly TILE the document with no internal gaps, and the two halves' tilings are offset
by exactly half a window — so nearly every interior word of the document is covered by
BOTH halves. **The w40/s20 numbers above (both raw and length-controlled) are inflated by
this near-total content sharing, not just by length — they are not on the same yardstick
as T0/T1's token-disjoint interleave/block constructions**, and should be read as an
upper bound on this geometry's true reliability, not a point estimate.

**A follow-up densification test (window=40, stride=8, aimed at getting more points per
document) initially looked like a large genuine improvement — length-controlled,
SB-corrected `H1_total_persistence` rose from 0.41 to 0.69, clearing H's own 0.576.**
Checked before writing that up as a finding: measured overlap at this geometry is
**0.963** (`data/t10_pilot_topology_w40_s8.json`) — even worse than the baseline, because
smaller strides at fixed window size multiply the number of opposite-half windows any
given window shares words with. The apparent gain was the overlap artifact getting worse,
not the signal getting stronger; **this 0.69 number is retracted and must not be cited.**

**Discriminating experiment: hold stride = window size (zero content overlap, directly
measured and confirmed at exactly 0.0000 for all three configs below), vary window size
to change point count instead.** This isolates point-count from content-sharing, the
confound the two results above conflated (`data/t10_pilot_geometry_comparison.json`,
n=300 docs, non-overlapping halves):

| geometry | mean windows/doc | H1_total_persistence | H1_entropy | H1_n_bars | H1_max_persistence |
|---|---|---|---|---|---|
| w40/s40 | 12.5 | 0.305 | -0.081 | 0.122 | 0.282 |
| w15/s15 | 31.9 | **0.413** | **0.252** | **0.360** | 0.213 |
| w10/s10 | 48.1 | 0.189 | 0.211 | 0.225 | 0.261 |

(all length-controlled, SB-corrected partial r, i.e. the conservative genuine-signal
estimate, same construction as above.) **The trend is non-monotonic, not rising with
point count**: reliability peaks at the middle density (w15/s15, ~32 windows/doc) and
*declines* at the densest setting (w10/s10, ~48 windows/doc) for three of four features.
The interpretation that fits: 10-word text snippets are below what `all-MiniLM-L6-v2`
embeds coherently, so shrinking the window keeps adding points while degrading what each
point actually represents — densification has a ceiling set by embedding quality, not by
point count alone. `H1_max_persistence` is the outlier, staying roughly flat (0.28/0.21/0.26)
across all three geometries — consistent with it depending on a single dominant loop in
the point cloud rather than on how finely the trajectory is sampled, making it the most
geometry-robust of the four features and the one a follow-up should start from if T10 is
pursued further.

**Conclusion: no geometry tested clears fractal H's reliability, at zero content overlap.**
The best genuine (length-controlled, overlap-free) value found anywhere in this pilot is
0.41 (w15/s15, `total_persistence`), against H's 0.576/0.546 and the plan's 0.60 bar.

**Recommendation: do not scale this pilot to the full corpus, on SLURM or otherwise —
now for a better-supported reason than originally stated.** The original write-up's
blocker ("point clouds are too small, and scaling the corpus doesn't fix a per-document
problem") is superseded: the geometry sweep shows point count is NOT the binding
constraint — reliability peaked at moderate density and fell at higher density. Cost is
also not the blocker (5-6 hours of CPU for the full corpus, measured, not guessed). The
actual finding is that this representation, embedded this way, does not carry as much
recoverable document-level signal as the fractal H exponent does, at any of the three
geometries tested with genuinely independent halves — a substantive negative result in
its own right, not an unresolved feasibility question. Any further T10 work should start
from `H1_max_persistence`'s geometry-robustness rather than re-sweeping window size.

---

## Known deviations from the plan's exact letter (documented, not silent)

0. **This session's cgroup CPU quota is 8.0 cores, not 192.** `nproc` reports 192 (the
   physical host's total), but `/sys/fs/cgroup/cpu.max` reads `800000 100000` = 8.0 cores
   hard cap for this session. Discovered mid-run when a 500-replicate permutation job
   using 72-96 worker processes stalled almost completely (each worker time-sliced to
   ~5-9% CPU). All worker-count choices below were tuned assuming far more headroom than
   actually existed; wall-clock times in the logs reflect this real 8-core ceiling, not
   192-way parallelism. Anyone re-running this on a different allocation will see very
   different (likely much faster) wall-clock times. Practical consequence: T1's
   permutation construction was reduced from the plan's "≥500" replicates to 100, then
   (after an unrelated order-scrambling bug fix, see T1 section) to a final 20 — a
   deliberate, documented scope reduction at each step, not a silent shortcut.
1. **Strata substitution.** GAGLE has no separate "generator" column distinct from
   `Label` (Human/gemma_2b/mistral_7b already IS the class-generator join). All
   "(class, generator, domain, temp)" demeaning uses `(Label, domain, temp)` throughout
   T0/T1/T7 — the exact same information content, just not a redundant 4th column.
2. **Route 2 variance decomposition is closed-form (one-way ANOVA method-of-moments +
   the Route-1 r_within split), not an iterative crossed mixed model.** Chosen
   specifically for zero convergence risk at n=223,834 — the same rationale Route 1 itself
   states for avoiding a solver. See `t0_variance_components.py`'s docstring for the
   derivation.
3. **T2's fGn generator** (`fbm` PyPI package) is not the same generator used in the
   2026-07-26 report (not preserved in the repo); validated to reproduce the qualitative
   pattern, not exact values. Documented in `t2_step0_validate_fgn.py`.
4. **Permutation and cluster-bootstrap performance rewrites.** The naive per-replicate
   index-concatenation approach for both the 500-rep permutation construction and the
   cluster-robust SE was replaced with vectorized/sufficient-statistic forms after an
   initial run stalled — verified numerically identical to the naive form on a synthetic
   check before use (see `reliability_lib.py`).
5. **T6 skipped entirely** (paper-appendix bookkeeping, not computation, per your
   instruction to ignore .tex-tied concerns).
6. **T10 deferred** (needs new embedding + persistent-homology infrastructure; the plan's
   own execution sequence already assigns it to "next session").
7. **T9 partial** (items 1-2 of 4, per the plan's own cost ordering).

## Reproducing/reviewing this run

Every script under `code/` can be re-run standalone (interpreter:
`/home/user/miniconda3/envs/review/bin/python`, or activate the `review` conda env).
Each writes its own JSON/parquet output under `data/` and prints a human-readable summary
to stdout — logs of the actual runs are under `logs/`.

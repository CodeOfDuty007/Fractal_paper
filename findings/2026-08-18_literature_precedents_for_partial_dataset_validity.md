# Literature Precedents: Partial Dataset Non-Reproducibility and Label-Correlated Feature Missingness

**Date:** 2026-08-18
**Question:** Has anyone in this field hit our two problems — (a) a released benchmark
whose precomputed artifacts are only partly reproducible, and (b) feature availability
that correlates with the label — and how did they handle it?

**Short answer:** Yes to both, but in *separate* literatures. Nobody appears to have
published the specific GAGLE finding. The methodological playbooks for each half are
well established, and they point in opposite directions on one of our three options.

---

## 0. What I searched, and what I did not find

Searched for: benchmark provenance audits, precomputed-logit reproducibility,
MGT-detection benchmark leakage, MNAR/informative missingness with label correlation,
complete-case analysis in detection, and any follow-up or replication of
Alabdulmohsin & Steiner (ICML 2025).

**Not found:** any published audit of GAGLE, any paper reporting that a released
surprisal-array corpus fails to reproduce for one generator family, and any erratum
on the fractal-complexity paper. The original is
[ICML 2025 / PMLR v267](https://proceedings.mlr.press/v267/alabdulmohsin25a.html)
with an [OpenReview record](https://openreview.net/forum?id=p2smPMRQae) and the
[HF dataset card](https://huggingface.co/datasets/ibomohsin/gagle); none flag this.

That absence matters twice: our finding is novel, and there is no precedent
specifically excusing it either.

---

## 1. Closest precedent: forensic audit of a released benchmark artifact

**[Forensic Reproducibility Audit of a Radiology Vision-Language Model Benchmark:
From Intended Protocol to Released Artifact](https://arxiv.org/html/2607.25589v1)**

Structurally the same situation: the *released artifact* diverged from the *intended
protocol*. A corrected table reproduced numerically, but under conditions that differed
from the reported experiment.

**How they handled it:**
- Withdrew the specific performance and prompt-effect claims that depended on the
  divergent conditions.
- Declared the legacy benchmark invalid **for those claims** — not globally.
- Proposed a *benchmark contract*: cohort membership, rendered-image and prompt hashes,
  **provider-resolved model identity**, call status, annotation provenance, keyed
  analysis, derived-artifact checks.

**Read across to us.** "Provider-resolved model identity" is exactly our missing
`mistral-7b` checkpoint — bare label, no repo ID, no revision hash. Their contract
formalises the field's expectation that a released scored artifact must name the model
that produced it. Our email to Alabdulmohsin asks for precisely the contract item their
audit says should have been there.

The precedent for outcome is important: **scope and specify, do not retract wholesale.**

Related: **[Auditing the Audit: Five Failure Modes in Benchmark-Validity
Audits](https://arxiv.org/html/2607.02586)** distinguishes *legacy pipelines* (e.g.
first-token logits) from *canonical pipelines* following the benchmark's reference
protocol — the same class of divergence as "which scoring configuration produced these
arrays."

---

## 2. Leakage across splits: established, and openly disclosed

Our split-duplication finding (11,696 human rows = 5,000 distinct texts; 4,474 span
more than one split) is a known failure mode in this exact subfield.

**[GenAI Content Detection Task 3: Cross-Domain Machine-Generated Text Detection
Challenge](https://arxiv.org/pdf/2501.08913)** states the risk plainly: human documents
are sampled from publicly available datasets, so participants' models may have seen
test documents. They **disclose rather than conceal** it.

**Concrete, citable deduplication thresholds in current use:**
- Remove any training example sharing a **30-token n-gram** with a held-out test example.
- **5-gram** overlap thresholding to catch high-lexical-overlap near-duplicates.

Broader contamination work notes exact dedup alone is insufficient — paraphrases and
fragments survive it.

**Read across.** Our case is *easier* than the literature's hardest version: we have
`dataset_id` as an exact key (confirmed present in the raw GAGLE records), so we can do
exact document-level grouping rather than fuzzy n-gram matching. The fix is a
group-disjoint split on `dataset_id` — which the Extension1 plan already calls for
("group-disjoint splits"); the bug is that the dedup key wrongly included `split`.

---

## 3. Label-correlated missingness: this is MNAR, and it has consequences

Our Family B availability — human ~100%, gemma-generated ~50%, mistral-generated ~0% —
is textbook **MNAR with y-dependence**.

- **[Impact of missing data correlated with labels to be predicted in
  neurodegeneration classification](https://www.biorxiv.org/content/10.1101/2025.01.23.634117.full.pdf)**
  — the exact pathology: missingness correlated with the target.
- **[Informative missingness and its implications in semi-supervised
  learning](https://arxiv.org/pdf/2512.04392)** — under MAR/MNAR the mechanism is
  *non-ignorable*, and **the missingness indicators themselves carry information about
  decision boundaries and class proportions**. That is our leakage channel stated
  formally.

**Two consequences that bear directly on our options:**

1. **Complete-case analysis is only valid under MCAR.** Under MAR/MNAR it yields biased
   inference. Restricting to rows that happen to have Family B is complete-case
   analysis on an MNAR mechanism — not safe by default.
2. **Empirically, indicators are not a fix either.** An
   [EHR simulation study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11924964/)
   found AUROC similar with and without missingness indicators, and complete-case
   analysis showed the **largest variance** across runs.

**The standard diagnostic** (worth running regardless): train the model **with and
without** missingness flags and measure the lift. If the flags carry signal, the
missingness is doing work the features should be doing.

---

## 4. Mapping precedent onto our three options

| Option | What the literature says |
|---|---|
| **A — scope to human + gemma-generated** | Defensible *as a population redefinition*, which is what §1's audit did. **But** ~50% of gemma-generated rows still lack Family B (the four failing prompt types), and that residual gap is MNAR by `prompt_type`. So A does not escape §3 — it shrinks the problem without solving it. |
| **B — self-scored Family B, no alignment gate, 100% coverage** | **Eliminates the MNAR mechanism outright** rather than managing it. §3 says this is strictly preferable: no complete-case bias, no indicator leakage, no lift diagnostic needed. Cost is that Families A and B describe slightly different spans — a documented systematic offset, not a bias. |
| **C — ship 28% and caveat** | The weakest. It is complete-case on an MNAR mechanism, applied to the very feature family (perplexity baselines) that carries Part II's headline claim. No precedent found that would support this for a primary result. |

**The literature's verdict is clearer than I expected: B.** MNAR is a mechanism to be
*removed* where removal is affordable, and here it is affordable — one extra scoring
pass. A and C both leave the mechanism in place and then argue about it.

---

## 5. Recommended write-up posture, with precedent

1. **Report the gemma verification as a positive result.** r = 1.0000, mad ≈ 0.007. Per
   §1's benchmark-contract framing this is the artifact check the field is moving
   toward; almost nobody does it.
2. **Report the mistral-generated non-reproducibility as a scoped invalidity claim**,
   not a retraction — mirroring §1's "withdraw the dependent claims, keep the rest."
   State the quantitative signature: stored array shorter than the text tokenizes to,
   under both scorers, at bit-exact slice equality.
3. **Disclose the split duplication and fix it by `dataset_id` grouping**, citing the
   n-gram-threshold practice in §2 as the stricter alternative we did not need.
4. **Recompute the human-side CIs post-deduplication.** N inflated 2.34× means intervals
   ~1.53× too narrow; `2026-08-05_test1-5` Test 4 and the wikipedia S cell
   (+0.0045) in `2026-08-06` both need rechecking.
5. **Prefer full-coverage self-scored features** and say why, citing MNAR
   non-ignorability. If any complete-case result is reported, pair it with the
   with/without-flags lift diagnostic.

---

## 6. Honest limits of this search

- Search was English-language web/arXiv; no paywalled venue sweep (TACL, CL, TMLR).
- Several hits are recent arXiv preprints, not peer-reviewed.
- No paper was found matching our situation *exactly* — a precomputed per-token
  surprisal corpus reproducing for one generator family and not another. The precedents
  above are structural analogues, not identical cases.
- The absence of a GAGLE audit could mean nobody looked, not that nobody found anything.

---

## Sources

- [Forensic Reproducibility Audit of a Radiology VLM Benchmark](https://arxiv.org/html/2607.25589v1)
- [Auditing the Audit: Five Failure Modes in Benchmark-Validity Audits](https://arxiv.org/html/2607.02586)
- [GenAI Content Detection Task 3](https://arxiv.org/pdf/2501.08913)
- [Informative missingness and its implications in semi-supervised learning](https://arxiv.org/pdf/2512.04392)
- [Impact of missing data correlated with labels](https://www.biorxiv.org/content/10.1101/2025.01.23.634117.full.pdf)
- [Imputation and Missing Indicators for Handling Missing Longitudinal Data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11924964/)
- [Reproducibility in ML-based research: overview, barriers, drivers (AI Magazine)](https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002)
- [A large-scale audit of dataset licensing and attribution (Nature MI)](https://www.nature.com/articles/s42256-024-00878-8)
- [A Tale of Two Structures (PMLR v267)](https://proceedings.mlr.press/v267/alabdulmohsin25a.html)
- [OpenReview record](https://openreview.net/forum?id=p2smPMRQae) · [arXiv:2502.14924](https://arxiv.org/abs/2502.14924) · [GAGLE dataset card](https://huggingface.co/datasets/ibomohsin/gagle)

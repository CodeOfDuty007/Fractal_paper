# Scale Without Support

**Multi-scale fractal features for LLM-generated-text detection: what they capture at the
population level, and why that mostly doesn't survive to the per-document decision.**

[![ICLR 2027](https://img.shields.io/badge/ICLR-2027%20(under%20review)-blueviolet)](paper/ICLR2027_ScaleWithoutSupport.pdf)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> This repository is anonymized for double-blind review. No author-identifying information
> is present in the paper, commit metadata, or code comments.

## Table of Contents
- [Overview](#overview)
- [Key results](#key-results)
- [Repository structure](#repository-structure)
- [Environment setup](#environment-setup)
- [Reproducing the pipeline](#reproducing-the-pipeline)
- [Data availability](#data-availability)
- [Findings log](#findings-log)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Detectors of LLM-generated text mostly score a document by how *locally* predictable it is
under a reference model. This paper asks a different question: does a document's text carry
usable signal in *how its predictability structure scales* — the multifractal / long-range
character of the token-level surprisal trajectory, summarized by a Hölder exponent (Ĥ) and
a rescaled-range statistic (Ŝ) — beyond what existing detectors already extract?

We separate that question into two levels that the literature usually conflates:

1. **Population level** — does Ĥ/Ŝ separate human from machine text at all, averaged over
   many documents in a (domain, generator) cell?
2. **Document level** — does that separation survive as usable *per-document* signal once you
   estimate Ĥ/Ŝ from the token budget an actual detector has (one document, a few hundred to
   a couple thousand tokens), and does it move a real detector's operating point?

The paper's estimator decomposes a per-document estimate into a population term, a
document-specific term, and finite-sample estimator error (`\hat{H}_d = \mu(c_d, g_d, T_d)
+ \delta_d + \eta_d`, Eq. 2 in the paper), and uses Spearman–Brown-projected split-half
reliability to work out how much support (token count) each estimator needs before its
per-document estimate is trustworthy at all — the reliability crossing point `N*`.

## Key results

| Claim | Finding |
|---|---|
| Population-level separability | 20 of 20 (domain, generator, exponent) cells separate with non-overlapping 95% CIs; a second, differently biased estimator agrees on direction in 10 of 10 Hölder cells |
| Per-document utility | Adding all fractal features to an 85-feature detector leaves AUROC unchanged (0.9984) and TPR@1%FPR statistically unchanged (0.9587 → 0.9612); the family recovers only about a sixth of the achievable signal once the population is balanced by construction |
| Support requirement | `N*` — the support length at which split-half reliability crosses a preregistered 0.60 bar — is large enough that most real documents don't reach it; for the Ŝ estimator it is a **lower bound**, not a point estimate (the noise-floor calibration under-predicts real-text scatter by ~3.7–4.0×) |
| Robustness | Recall against same-lineage paraphrase (0.63–0.86) collapses to 1.2–4.5% under paraphrase from models outside the scorer's lineage, across a five-model capability ladder |
| Within-population signal | A weak but genuine document-level signal survives targeted confound checks (§4.2 in the paper): 0.198 vs. 0.125 chance, not population leakage |

Full numbers, confidence intervals, and the four falsifiable predictions (P1–P3 in the
paper's Table 1) are in [`paper/ICLR2027_ScaleWithoutSupport.pdf`](paper/ICLR2027_ScaleWithoutSupport.pdf).

## Repository structure

```
.
├── paper/                    # Full LaTeX source + compiled PDF (anonymized, ICLR2027 template)
│   ├── ICLR2027_ScaleWithoutSupport.tex
│   ├── ICLR2027_ScaleWithoutSupport.pdf
│   ├── references_090926.bib
│   ├── *.sty / *.bst / math_commands.tex  # ICLR2027 style files (flat, alongside main.tex)
│   ├── fig_*.tex              # Standalone TikZ figure sources
│   └── supplementary_figures/ # Figures used in the appendix
├── src/                      # Full analysis + pipeline code (see table below)
├── findings/                 # Dated research log: every verification / defect-fix pass
├── images/                   # Standalone result plots
├── review_results/           # T0-T10 statistical validation battery: code + stats + figures behind the paper's core numbers
├── data/                     # DATA.md: provenance + exact regeneration path (raw data not checked in)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

### Code map

| Script | Role |
|---|---|
| `fractal_math.py` | Core multi-scale fractal feature extraction (Ĥ, Ŝ) from a token surprisal trajectory |
| `scorer.py` | Token-level surprisal scoring against a reference LM |
| `build_p0_local.py` | Stage P0: corpus ingestion / local build |
| `build_p1_feature_store_cpu.py` / `build_p1_feature_store_gpu.py` | Stage P1: CPU and GPU (HPC, Family B) feature-store construction |
| `build_p1_join_feature_store.py` | Joins GPU shard outputs into one feature store |
| `reconcile_p1_against_p0.py` | Cross-checks P1 against P0 for coverage/drift |
| `backfill_gpu_shard_prefix.py` / `backfill_short_window_flag.py` | Post-hoc backfills applied to the feature store |
| `recompute_family_a_continuation_fix.py` | Recomputation after the Family A continuation-token fix |
| `investigate_prompt_type_alignment.py` / `probe_cross_column_alignment.py` | Diagnostic probes into prompt-type / column alignment |
| `download_gagle_zip.py` / `hf_extractor.py` | Corpus acquisition |
| `apply_per_row_cap.py` / `regress_candidate_cap.py` | Per-row capping and regression-candidate selection for the ablation battery |
| `tokenizer_checkpoint_screen.py` / `mac_tokenizer_dryrun.py` / `fix_hf_symlinks.py` | Tokenizer/environment sanity utilities |
| `visualizer.py` | Result plotting |

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Final paper runs used Python 3.12.13, torch 2.13.0, transformers 5.14.1. GPU-heavy stages
(Family B) were run on an institutional HPC cluster; cluster-specific access/runbook details
aren't included here since they're internal to that infrastructure, not needed to reproduce
the analysis itself (which only depends on the requirements above).

## Reproducing the pipeline

1. `python src/download_gagle_zip.py` — fetch the GAGLE benchmark bundle.
2. `python src/build_p0_local.py` — ingest and build the P0 corpus.
3. `python src/build_p1_feature_store_cpu.py` (and, on a GPU/HPC node, `build_p1_feature_store_gpu.py` + `build_p1_join_feature_store.py`) — build the P1 multi-scale feature store.
4. `python src/reconcile_p1_against_p0.py` — verify P1 coverage against P0.
5. `python src/fractal_math.py` — extract Ĥ/Ŝ fractal features.
6. Cross-check outputs against the corresponding dated report in `findings/`, which documents the expected shape and counts for that stage's output.

See [`data/DATA.md`](data/DATA.md) for the full stage → script → output mapping and why raw
intermediate files (hundreds of MB to ~500MB each) aren't checked into this repo.

## Data availability

Raw and intermediate data files are not included — see [`data/DATA.md`](data/DATA.md) for
provenance, sizing, and the exact commands to regenerate them from the public GAGLE/RAID
source corpora.

## Statistical validation battery

[`review_results/`](review_results/) holds the actual T0–T10 test code, computed statistics,
and the 10 figures they generated — the artifacts directly behind the paper's headline
numbers (reliability construction, population-leakage controls, N* composition, RAID
cross-checks, matched-length sensitivity, deployment metrics). See
[`review_results/README.md`](review_results/README.md).

## Findings log

[`findings/`](findings/) is the project's dated research log — every methodological
verification, statistical check, and defect investigation, in the order it happened, so a
number in the paper can be traced back to the check that produced it. See
[`findings/README.md`](findings/README.md) for an index.

## Citation

```bibtex
@inproceedings{anonymous2027scalewithoutsupport,
  title     = {Scale Without Support: Multi-Scale Fractal Features for {LLM}-Generated-Text Detection},
  author    = {Anonymous},
  booktitle = {Submitted to the International Conference on Learning Representations (ICLR)},
  year      = {2027},
  note      = {Under double-blind review}
}
```

## License

Code is released under the [MIT License](LICENSE). See the paper PDF for the paper's own
copyright terms.

## Acknowledgments

AI coding assistants (Claude, and OpenAI Codex) were used during development to help write
and refactor analysis code, draft documentation, and cross-check numeric consistency between
the paper and its underlying data — all under direct human review and direction throughout.
Consistent with how comparable ICLR paper repositories present authorship, they are credited
here in prose rather than listed as GitHub contributors, since GitHub's contributor graph is
derived from commit authorship and this project's commits reflect the responsible human
author(s). The paper's own AI-use disclosure statement is the authoritative record of what
AI assistance was used and how.

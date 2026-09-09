# Data Availability

## Why raw data isn't in this repo
The full pipeline processes on the order of hundreds of thousands of documents across eleven
generator models and eight domains (GAGLE + RAID-derived corpora). The intermediate feature
store and checkpoint files this produces are **hundreds of megabytes to half a gigabyte per
file** — well past GitHub's 100MB hard per-file limit — so raw and intermediate data files
are intentionally excluded (`data/` is gitignored) rather than checked in or pushed via LFS.

## What's here instead
This directory documents provenance and gives an exact regeneration path:

- **Source corpora**: GAGLE benchmark bundle (downloaded via `src/data_pipeline/download_gagle_zip.py`)
  and RAID-derived splits. Both are third-party benchmarks with their own distribution terms;
  this repo does not redistribute them.
- **Pipeline stage → script mapping**:
  | Stage | Script | Output |
  |---|---|---|
  | P0 — corpus ingestion | `src/data_pipeline/build_p0_local.py` | `data/p0_local_full_results.jsonl`, manifest |
  | P1 — CPU feature store | `src/data_pipeline/build_p1_feature_store_cpu.py` | `data/p1_feature_store_cpu.jsonl` |
  | P1 — GPU feature store (Family B, HPC) | `src/data_pipeline/build_p1_feature_store_gpu.py` | GPU shard outputs, joined via `src/data_pipeline/build_p1_join_feature_store.py` |
  | Reconciliation | `src/data_pipeline/reconcile_p1_against_p0.py` | `data/p1_reconciliation_pooled_SH.json` |
  | Short-window flagging | `src/data_pipeline/backfill_short_window_flag.py` | `data/p1_short_window_flags.jsonl` |
  | Family A continuation fix | `src/data_pipeline/recompute_family_a_continuation_fix.py` | `data/p1_family_a_corrected_continuation.jsonl` |

- **Manifests** (`p0_ingest_manifest.json`, `p0_local_manifest.json`, `p1_manifest.json`, and
  the two `*_checkpoint.txt` member lists) fully specify which documents entered each stage
  and in what order, so a rerun is checkable against the original run's coverage even without
  the underlying text.

## Reproducing
1. Set up the environment (`requirements.txt`).
2. Run `src/data_pipeline/download_gagle_zip.py` to fetch the GAGLE bundle.
3. Run the P0 → P1 pipeline stages in the order above.
4. Run `src/features/fractal_math.py` for the multi-scale fractal feature extraction itself.
5. Cross-check against `findings/` — each stage's dated report documents the expected
   shape/counts of its output, so a rerun's manifest can be diffed against it.

If you need the exact intermediate files rather than regenerating them, open an issue on
this repository (or, during anonymous review, a comment via the review platform) and the
authors will arrange access post-de-anonymization.

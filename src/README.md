# Pipeline Code

| Folder | Role |
|---|---|
| `data_pipeline/` | Corpus acquisition and the P0 -> P1 feature-store build (ingestion, CPU/GPU feature extraction, joining, reconciliation, backfills, the Family A continuation fix) |
| `features/` | The core multi-scale fractal feature extraction (`fractal_math.py`) and token-level surprisal scoring (`scorer.py`) |
| `diagnostics/` | Standalone probes and ablation-battery utilities (prompt-type/column alignment checks, per-row capping, regression-candidate selection) |
| `utils/` | Environment and tokenizer sanity utilities, plotting |

See the root [`README.md`](../README.md#code-map) for the full script-by-script table, and
[`../data/DATA.md`](../data/DATA.md) for how these scripts chain into the P0 -> P1 pipeline.

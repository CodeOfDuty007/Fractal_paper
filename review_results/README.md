# Statistical Validation Battery (T0–T10)

This is the actual code, computed statistics, and generated figures behind the paper's
core quantitative claims — the T0 through T10 test battery referenced throughout the paper
and its appendices (variance decomposition, permutation tests, reliability construction,
population-leakage controls, alternative-estimator agreement, RAID cross-checks, matched-length
sensitivity, deployment metrics, and the persistent-homology pilot topology sweep).

## What's here

| Dir | Contents |
|---|---|
| `code/` | The 18 scripts that ran each test and produced its figure (`t*_*.py`, `make_figures.py`, `reliability_lib.py`) |
| `data/` | Computed statistics as JSON (reliability tables, N* composition, noise-floor calibration, population-control results, etc.) — **not** the underlying raw feature parquet files (see below) |
| `figures/` | The 10 PNG figures these tests generated, several of which appear in the paper's main text and appendix |
| `logs/` | Raw stdout logs from the actual runs, including one (`t1_permutation_stats_v1_ORDER_SCRAMBLE_BUG.json` / `.log`) documenting a bug that was caught and fixed — kept for transparency rather than deleted |
| `FINDINGS_LEDGER.md` | The consolidated, dated ledger of every test's outcome and interpretation |

## What's not here

Nine `data/*.parquet` files (the per-document raw feature tables each test was computed
from) are excluded — they total **512MB**, well past what's sensible to check into git.
Each is regenerable by rerunning the corresponding `code/t*.py` script against the feature
store built via the top-level `src/` pipeline (see [`data/DATA.md`](../data/DATA.md)); the
`data/*_stats.json` files here already contain the aggregated statistics those parquet files
were computed down to, which is what the paper actually reports.

Paths in `code/` and `logs/` were written during local/HPC development and have been
redacted (`/data/user/...`, `[SRV:user]`) — the computed values and code logic are unedited.

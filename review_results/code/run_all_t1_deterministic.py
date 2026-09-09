#!/usr/bin/env python3
"""Runs the remaining deterministic T1 split constructions (contiguous,
block_4/8/16/32, block_bootstrap, matched_sweep) sequentially, loading the
corpus once and reusing it across all of them (run_construction.run() caches
into the module-level _G dict). Interleave was already validated separately
against the frozen P2 baseline before this was launched."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_construction import run  # noqa: E402
from reliability_lib import WINDOW_SCALE  # noqa: E402

CONSTRUCTIONS = ["block_16", "block_32", "block_bootstrap", "matched_sweep"]
# interleave/contiguous/block_4/block_8 already completed in an earlier pass
# (review_results/data/t0_t1_{construction}_stats.json) -- resuming from here.

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    for c in CONSTRUCTIONS:
        window_scale = WINDOW_SCALE if c == "matched_sweep" else None
        print(f"\n{'='*80}\n[driver] starting construction={c}\n{'='*80}", flush=True)
        run(c, workers, f"t0_t1_{c}", seed=20260908, window_scale=window_scale,
            save_long_parquet=True)
    print("\n[driver] ALL DETERMINISTIC T1 CONSTRUCTIONS DONE", flush=True)

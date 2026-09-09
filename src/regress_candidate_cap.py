#!/usr/bin/env python3
"""
Regression harness for the MAX_CANDIDATE_TOKENS -> per-row cap change.

WHY
  MAX_CANDIDATE_TOKENS = 2048 is not a resource limit, it is a RECONSTRUCTION
  parameter. Truncating to `cap` sets delta = n - gagle_total, and the sweep
  anchors at a_start + delta (~64 + delta) with radius 150. So:

      cap = gagle_total + m   ->   anchor = 64 + m   ->   need m <= ~151

  cap=2048 works on gemma human rows only because their gagle_total is 2047
  (m = 1, anchor = 65 -- the empirically observed winning offset). Raising it
  to 4096 gives m = 2049, anchor = 2113, ~2000 tokens outside the window --
  which is why 12/12 r=1.0000 collapsed to 0/12 r=0.15 in testing.

  Meanwhile gagle_mistral_total_tokens for human rows is 3071 > 2048, so the
  mistral scorer drops ~99.7% of human rows before any forward pass.

  Correct fix: cap = gagle_{scorer}_total_tokens + 1, per row, per scorer.

  This harness exists because that change SILENTLY alters which rows get
  features. A wrong fix produces a 6-10h run that looks fine and isn't.

USAGE
  # 1. BEFORE changing anything -- capture the baseline
  python3 regress_candidate_cap.py --baseline regress_before.json

  # 2. make the per-row cap change in src/build_p1_feature_store_gpu.py

  # 3. AFTER -- compare. Exits non-zero on any regression.
  python3 regress_candidate_cap.py --compare regress_before.json \
                                   --baseline regress_after.json

WHAT IT CHECKS
  MUST_PASS  cells: pass count must not fall, median r must not degrade
  MUST_FAIL  cells: pass count must not RISE (guards against spurious fits
                    from a wider effective search)
  MEASURE    cells: reported, never fatal -- these are the ones that may
                    legitimately improve
  ALL  cells: `dropped` (no usable candidate) must not rise. That is the
              headline number the fix is meant to move.

SIGNATURE-AGNOSTIC
  tokenize_candidates gains a gagle_total parameter as part of the fix. This
  harness introspects the signature and calls it correctly either way, so the
  SAME script runs before and after. Do not "update" it alongside the fix.
"""

import argparse
import collections
import glob
import inspect
import json
import os
import statistics
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_p1_feature_store_gpu as P  # noqa: E402

SHARD_DIR = os.environ.get("GAGLE_MANIFEST_DIR",
                           "/data/user/paper/gpu_manifest_shards")

# name, shard-name substring, side, prompt_type (None = any), scorer, expectation
CELLS = [
    ("gemma_human",           "gemma_2b_0.0",   "human", None,          "gemma-2b",   "MUST_PASS"),
    ("gemma_llm_continue_pt", "gemma_2b_0.0",   "llm",   "continue_pt", "gemma-2b",   "MUST_PASS"),
    ("gemma_llm_keywords",    "gemma_2b_0.0",   "llm",   "keywords",    "gemma-2b",   "MUST_PASS"),
    ("gemma_llm_summary",     "gemma_2b_0.0",   "llm",   "summary",     "gemma-2b",   "MUST_PASS"),
    ("gemma_llm_cot",         "gemma_2b_0.0",   "llm",   "cot",         "gemma-2b",   "MUST_FAIL"),
    ("gemma_llm_excerpt",     "gemma_2b_0.0",   "llm",   "excerpt",     "gemma-2b",   "MUST_FAIL"),
    ("gemma_on_mistralgen",   "mistral_7b_0.0", "llm",   "continue_pt", "gemma-2b",   "MEASURE"),
    ("mistral_human",         "gemma_2b_0.0",   "human", None,          "mistral-7b", "MEASURE"),
]

_TC_TAKES_TOTAL = len(inspect.signature(P.tokenize_candidates).parameters) > 3


def make_candidates(tok, row, total_key):
    """Call tokenize_candidates under either signature (pre- or post-fix)."""
    if _TC_TAKES_TOTAL:
        return P.tokenize_candidates(tok, row["text"], row.get("prefix"),
                                     row.get(total_key))
    return P.tokenize_candidates(tok, row["text"], row.get("prefix"))


def find_rows(substr, side, prompt_type, need):
    """Locate a shard with enough matching rows. Discovered, never hardcoded --
    hardcoded shard paths are exactly how the human-side probe was pointed at
    a shard containing zero human rows."""
    for f in sorted(glob.glob(os.path.join(SHARD_DIR, f"*{substr}*.parquet"))):
        base = os.path.basename(f)
        if prompt_type and not base.endswith(f"_{prompt_type}.jsonl.parquet"):
            continue  # endswith, else 'summary' also matches 'summary_keywords'
        rows = [r for r in pq.read_table(f).to_pylist()
                if side in str(r.get("side", "")).lower()]
        if len(rows) >= need:
            return f, rows[:need]
    return None, []


def run_cell(bundle, rows, scorer):
    torch_mod, device, _dtype, tokenizers, models, _emb = bundle
    tok, model = tokenizers[scorer], models[scorer]
    sliced_key, total_key = next((s, t) for n, s, t in P.SCORER_KEYS if n == scorer)

    cands = [make_candidates(tok, r, total_key) for r in rows]
    per_cand = {
        cn: P.score_documents_for_candidate(torch_mod, model, tok, device,
                                            rows, cands, cn, sliced_key,
                                            total_key, scorer)
        for cn in P.CANDIDATE_NAMES
    }

    passed = dropped = 0
    rs, offs, wins = [], [], collections.Counter()
    for i in range(len(rows)):
        pc = {cn: per_cand[cn].get(i) for cn in P.CANDIDATE_NAMES}
        name, best, _ = P.pick_best_candidate(pc)
        if best is None:
            dropped += 1
            continue
        r, mad = best["best_r"], best["best_mean_abs_diff"]
        rs.append(r)
        wins[name] += 1
        if best.get("best_abs_start") is not None:
            offs.append(best["best_abs_start"])
        if r > P.ALIGNMENT_PEARSON_MIN and mad < P.ALIGNMENT_MEAN_ABS_DIFF_MAX:
            passed += 1

    return {
        "n": len(rows), "passed": passed, "dropped": dropped,
        "median_r": round(statistics.median(rs), 4) if rs else None,
        "max_r": round(max(rs), 4) if rs else None,
        "offset_median": int(statistics.median(offs)) if offs else None,
        "candidate_wins": dict(wins),
    }


def collect(n_rows):
    scorers = sorted({c[4] for c in CELLS})
    n_shards = len(glob.glob(os.path.join(SHARD_DIR, "*.parquet")))
    print(f"[info] SHARD_DIR = {SHARD_DIR}")
    print(f"[info] shards visible: {n_shards}")
    if n_shards < 700:
        print(f"[WARN] expected 720 shards. GAGLE_MANIFEST_DIR is probably still")
        print(f"[WARN] pointed at a probe subset. Cells will SKIP. Fix with:")
        print(f"[WARN]   export GAGLE_MANIFEST_DIR=/data/user/paper/gpu_manifest_shards")
    print(f"[info] MAX_CANDIDATE_TOKENS = {P.MAX_CANDIDATE_TOKENS}")
    print(f"[info] tokenize_candidates takes gagle_total: {_TC_TAKES_TOTAL}")
    print(f"[info] loading scorers: {', '.join(scorers)}\n")
    bundle = P.load_models(scorers=tuple(scorers))
    prefix_map = P.load_prefix_map()

    out = {
        "max_candidate_tokens": P.MAX_CANDIDATE_TOKENS,
        "per_row_cap_active": _TC_TAKES_TOTAL,
        "n_rows_per_cell": n_rows,
        "cells": {},
    }
    for name, substr, side, pt, scorer, expect in CELLS:
        shard, rows = find_rows(substr, side, pt, n_rows)
        if not rows:
            print(f"  {name:<24} SKIP (no shard with {n_rows} '{side}' rows)")
            out["cells"][name] = {"skipped": True, "expect": expect}
            continue
        for r in rows:
            r["prefix"] = prefix_map.get(r["id"])
        res = run_cell(bundle, rows, scorer)
        res["expect"] = expect
        res["shard"] = os.path.basename(shard)
        out["cells"][name] = res
        print(f"  {name:<24} [{expect:<9}] passed {res['passed']}/{res['n']} "
              f"dropped {res['dropped']}  med_r={res['median_r']}  "
              f"off={res['offset_median']}")
    return out


def compare(before, after):
    print("\n" + "=" * 78)
    print("REGRESSION COMPARISON")
    print("=" * 78)
    print(f"  cap: {before.get('max_candidate_tokens')} -> {after.get('max_candidate_tokens')}"
          f"   per-row: {before.get('per_row_cap_active')} -> {after.get('per_row_cap_active')}\n")
    print(f"  {'cell':<24}{'expect':<11}{'passed':>14}{'dropped':>12}{'median r':>20}")
    print("  " + "-" * 76)

    failures = []
    for name, _s, _sd, _p, _sc, expect in CELLS:
        b, a = before["cells"].get(name, {}), after["cells"].get(name, {})
        if b.get("skipped") or a.get("skipped") or not b or not a:
            print(f"  {name:<24}{expect:<11}{'(skipped)':>14}")
            continue
        pb, pa = b["passed"], a["passed"]
        db, da = b["dropped"], a["dropped"]
        rb, ra = b["median_r"], a["median_r"]
        print(f"  {name:<24}{expect:<11}{f'{pb}->{pa}':>14}{f'{db}->{da}':>12}"
              f"{f'{rb}->{ra}':>20}")

        if da > db:
            failures.append(f"{name}: dropped ROSE {db}->{da} "
                            f"(the fix is supposed to reduce drops)")
        if expect == "MUST_PASS":
            if pa < pb:
                failures.append(f"{name}: passed FELL {pb}->{pa}")
            if rb is not None and ra is not None and ra < rb - 0.01:
                failures.append(f"{name}: median r degraded {rb}->{ra}")
        elif expect == "MUST_FAIL":
            if pa > pb:
                failures.append(f"{name}: passed ROSE {pb}->{pa} — likely "
                                f"SPURIOUS fits from a wider effective search")

    print("\n" + "=" * 78)
    if failures:
        print("REGRESSIONS DETECTED — do not start the full run")
        print("=" * 78)
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("NO REGRESSIONS")
    print("=" * 78)
    for name, _s, _sd, _p, _sc, expect in CELLS:
        b, a = before["cells"].get(name, {}), after["cells"].get(name, {})
        if expect == "MEASURE" and b and a and not b.get("skipped") and not a.get("skipped"):
            dp, dd = a["passed"] - b["passed"], a["dropped"] - b["dropped"]
            if dp or dd:
                print(f"  → {name}: passed {dp:+d}, dropped {dd:+d}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="path to WRITE results to")
    ap.add_argument("--compare", help="path of an earlier baseline to compare against")
    ap.add_argument("--n", type=int, default=12, help="rows per cell (default 12)")
    args = ap.parse_args()

    after = collect(args.n)
    with open(args.baseline, "w") as fh:
        json.dump(after, fh, indent=2)
    print(f"\n[info] wrote {args.baseline}")

    if not args.compare:
        print("[info] baseline captured. Make the cap change, then re-run with "
              "--compare " + args.baseline)
        return 0
    with open(args.compare) as fh:
        before = json.load(fh)
    return compare(before, after)


if __name__ == "__main__":
    sys.exit(main())

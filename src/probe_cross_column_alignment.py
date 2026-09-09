#!/usr/bin/env python3
"""
Cross-column alignment probe — diagnose why BOTH scorers fail on
mistral_7b-generated shards.

WHY THIS EXISTS
  verify_alignment only ever pairs each scorer with its OWN stored column:
      gemma-2b    <-> gagle_gemma_sliced_scores
      mistral-7b  <-> gagle_mistral_sliced_scores

  Observed 2x2 (2026-08-17, N=40 each, test split, temp 0.0):
      gemma scorer  on gemma_7b-generated docs .... 18/40   (r=1.0000, mad~0.007)
      mistral scorer on gemma_2b-generated docs ....  1/40
      gemma scorer  on mistral_7b-generated docs ...  0/40
      mistral scorer on mistral_7b-generated docs ..  0/40

  Gemma's checkpoint is PROVEN correct by the 18/40 cell. So a total failure
  of BOTH scorers on mistral-generated rows is not a checkpoint problem —
  it points at the stored arrays on those rows.

  The leading hypothesis this probe tests: the two score columns are SWAPPED
  (or otherwise mismatched) on mistral_7b-generated rows. That would produce
  exactly the observed signature, and verify_alignment structurally cannot
  see it.

WHAT IT DOES
  Runs the full 2x2 of {gemma-2b, mistral-7b} model x {gemma, mistral}
  stored column, on one shard, reusing the pipeline's own functions so the
  windowing/sweep/gate logic is identical. Also dumps raw structural stats
  (array lengths vs. tokenized lengths) which localize a mismatch even if
  no pairing aligns.

  Read-only. Writes no feature store, touches no checkpoint.

USAGE
  cd /data/user/paper/family_b_gpu_bundle/code
  export HF_HOME=/data/user/paper/hf_cache
  export HF_HUB_OFFLINE=1

  # the failing case
  python3 probe_cross_column_alignment.py \
      --shard /data/user/paper/gpu_manifest_shards/gagle3__test__mistral_7b_0.0_big_patent_continue_pt.jsonl.parquet

  # control: a shard already known to pass 5/5 for gemma
  python3 probe_cross_column_alignment.py \
      --shard /data/user/paper/gpu_manifest_shards/gagle3__test__gemma_2b_0.0_big_patent_continue_pt.jsonl.parquet

  (If those exact filenames differ, list them:
     ls /data/user/paper/gpu_manifest_shards/ | grep continue_pt | head)

HOW TO READ IT
  * gemma model x MISTRAL column scores well on mistral-generated rows
        -> COLUMN SWAP confirmed. Data-side fix, no re-scoring needed.
  * every pairing fails, but structural stats show array length matching a
    DIFFERENT column's total_tokens
        -> metadata mismatch; the structural table localizes it.
  * every pairing fails and lengths look self-consistent
        -> the stored arrays for these rows are not reproducible from the
           stored text at all. Escalate: those rows' provenance is unknown,
           which also bears on Family A, since Family A consumed these same
           arrays directly.
"""

import argparse
import collections
import os
import statistics
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_p1_feature_store_gpu as P  # noqa: E402


COLUMNS = [
    ("gemma_col", "gagle_gemma_sliced_scores", "gagle_gemma_total_tokens"),
    ("mistral_col", "gagle_mistral_sliced_scores", "gagle_mistral_total_tokens"),
]


def structural_stats(rows, tokenizers):
    """Length triangulation — runs without any forward pass."""
    print("\n" + "=" * 78)
    print("STRUCTURAL STATS (no model forward pass)")
    print("=" * 78)
    hdr = (f"{'#':<3} {'txt_chars':>9} "
           f"{'gem_arr':>8} {'gem_tot':>8} {'gem_tok':>8} "
           f"{'mis_arr':>8} {'mis_tot':>8} {'mis_tok':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, row in enumerate(rows):
        g_arr = row.get("gagle_gemma_sliced_scores")
        m_arr = row.get("gagle_mistral_sliced_scores")
        g_tok = len(tokenizers["gemma-2b"].encode(row["text"])) if "gemma-2b" in tokenizers else -1
        m_tok = len(tokenizers["mistral-7b"].encode(row["text"])) if "mistral-7b" in tokenizers else -1
        print(f"{i:<3} {len(row['text']):>9} "
              f"{(len(g_arr) if g_arr is not None else -1):>8} "
              f"{(row.get('gagle_gemma_total_tokens') or -1):>8} {g_tok:>8} "
              f"{(len(m_arr) if m_arr is not None else -1):>8} "
              f"{(row.get('gagle_mistral_total_tokens') or -1):>8} {m_tok:>8}")

    print("\nInterpretation hints:")
    print("  * <col>_arr should be <= <col>_tot, and <col>_tot should track")
    print("    <col>_tok closely (same tokenizer). If gem_tot tracks mis_tok")
    print("    (or vice versa), the metadata columns are crossed.")
    print("  * arr == -1 means the column is NULL on this row.")


def run_pairing(torch_mod, device, tokenizers, models, rows, row_candidates,
                model_name, col_label, sliced_key, total_key):
    """One cell of the 2x2. Mirrors verify_alignment's own selection logic."""
    tok, model = tokenizers[model_name], models[model_name]
    per_candidate_results = {
        cn: P.score_documents_for_candidate(
            torch_mod, model, tok, device, rows, row_candidates,
            cn, sliced_key, total_key, model_name)
        for cn in P.CANDIDATE_NAMES
    }

    passed, best_rs, detail = 0, [], []
    for idx in range(len(rows)):
        per_candidate = {cn: per_candidate_results[cn].get(idx) for cn in P.CANDIDATE_NAMES}
        best_name, best, _ = P.pick_best_candidate(per_candidate)
        if best is None:
            detail.append((idx, None, float("nan"), float("nan"), None, None))
            continue
        r, mad = best["best_r"], best["best_mean_abs_diff"]
        best_rs.append(r)
        ok = (r > P.ALIGNMENT_PEARSON_MIN and mad < P.ALIGNMENT_MEAN_ABS_DIFF_MAX)
        passed += int(ok)
        detail.append((idx, best_name, r, mad,
                       best.get("best_abs_start"), best.get("delta")))
    return passed, best_rs, detail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", required=True, help="path to ONE .parquet shard")
    ap.add_argument("--n", type=int, default=10, help="rows to probe (default 10)")
    ap.add_argument("--side", default=None,
                    help="filter rows by the `side` column (case-insensitive "
                         "substring, e.g. 'human' or 'llm'). Every probe so far "
                         "has silently tested LLM rows only; human rows are what "
                         "Family A's per-domain normalization rests on.")
    ap.add_argument("--scorers", nargs="+", default=["gemma-2b", "mistral-7b"])
    ap.add_argument("--verbose", action="store_true", help="print per-document rows")
    ap.add_argument("--full-sweep", action="store_true",
                    help="ignore OFFSET_SWEEP_RADIUS and search EVERY valid start "
                         "position. Decides whether a near-miss (r~0.93) is a "
                         "too-narrow search window (fixable) or genuinely "
                         "non-corresponding arrays (fatal). Slower.")
    args = ap.parse_args()

    if args.full_sweep:
        # sweep_best_alignment reads this module global at call time, so
        # rebinding it here widens the search for this process only. Nothing
        # on disk or in the pipeline is modified.
        old = P.OFFSET_SWEEP_RADIUS
        P.OFFSET_SWEEP_RADIUS = 10 ** 7
        print(f"[info] FULL SWEEP: OFFSET_SWEEP_RADIUS {old} -> {P.OFFSET_SWEEP_RADIUS} "
              f"(in-process only)")

    if not os.path.exists(args.shard):
        print(f"[abort] shard not found: {args.shard}")
        return 2

    print(f"[info] shard: {os.path.basename(args.shard)}")
    print(f"[info] scorers: {', '.join(args.scorers)}   rows: {args.n}")

    torch_mod, device, dtype, tokenizers, models, _ = P.load_models(scorers=tuple(args.scorers))
    prefix_map = P.load_prefix_map()

    table = pq.read_table(args.shard)
    all_rows = table.to_pylist()

    # What sides/labels actually exist in this shard? Print before filtering --
    # this is how you find out whether a probe has been silently one-sided.
    for col in ("side", "Label"):
        if all_rows and col in all_rows[0]:
            counts = collections.Counter(str(r.get(col)) for r in all_rows)
            print(f"[info] shard `{col}` value counts: {dict(counts)}")

    if args.side:
        want = args.side.lower()
        rows = [r for r in all_rows if want in str(r.get("side", "")).lower()]
        print(f"[info] --side {args.side!r}: {len(rows)}/{len(all_rows)} rows match")
        if not rows:
            print("[abort] no rows matched --side. Use the value counts above.")
            return 2
        rows = rows[:args.n]
    else:
        rows = all_rows[:args.n]

    for row in rows:
        row["prefix"] = prefix_map.get(row["id"])
    missing_prefix = sum(1 for r in rows if r.get("prefix") is None)
    print(f"[info] loaded {len(rows)} rows; {missing_prefix} missing a prefix")
    sides = collections.Counter(str(r.get("side")) for r in rows)
    print(f"[info] sides in this sample: {dict(sides)}")

    structural_stats(rows, tokenizers)

    row_candidates = [P.tokenize_candidates(tokenizers[args.scorers[0]],
                                            r["text"], r.get("prefix")) for r in rows]

    print("\n" + "=" * 78)
    print("CROSS-COLUMN 2x2  (pass = r > "
          f"{P.ALIGNMENT_PEARSON_MIN} AND mean_abs_diff < {P.ALIGNMENT_MEAN_ABS_DIFF_MAX})")
    print("=" * 78)

    grid = {}
    for model_name in args.scorers:
        # tokenization is scorer-specific — rebuild per model
        rc = [P.tokenize_candidates(tokenizers[model_name], r["text"], r.get("prefix"))
              for r in rows]
        for col_label, sliced_key, total_key in COLUMNS:
            passed, best_rs, detail = run_pairing(
                torch_mod, device, tokenizers, models, rows, rc,
                model_name, col_label, sliced_key, total_key)
            med = statistics.median(best_rs) if best_rs else float("nan")
            mx = max(best_rs) if best_rs else float("nan")
            grid[(model_name, col_label)] = (passed, med, mx)
            native = " (native pairing)" if (
                (model_name == "gemma-2b" and col_label == "gemma_col") or
                (model_name == "mistral-7b" and col_label == "mistral_col")) else " (CROSSED)"
            print(f"\n  {model_name:<12} x {col_label:<12}{native}")
            print(f"      passed {passed}/{len(rows)}   median r={med:.4f}   max r={mx:.4f}")
            offs = [d[4] for d in detail if d[4] is not None]
            if offs:
                print(f"      best_abs_start: min={min(offs)} max={max(offs)} "
                      f"median={int(statistics.median(offs))}")
            if args.verbose:
                for idx, cand, r, mad, abs_start, delta in detail:
                    print(f"        [{idx}] cand={cand} r={r:.4f} mad={mad:.4f} "
                          f"abs_start={abs_start} delta={delta}")

    print("\n" + "=" * 78)
    print("SUMMARY GRID  (passed / median r)")
    print("=" * 78)
    print(f"  {'':<14}" + "".join(f"{c[0]:>22}" for c in COLUMNS))
    for model_name in args.scorers:
        cells = []
        for col_label, _, _ in COLUMNS:
            p, med, _ = grid[(model_name, col_label)]
            cells.append(f"{p}/{len(rows)}  med r={med:.3f}")
        print(f"  {model_name:<14}" + "".join(f"{c:>22}" for c in cells))

    print("\nVERDICT GUIDE")
    print("  A CROSSED cell scoring well  -> column swap on these rows.")
    print("  All four cells failing       -> check STRUCTURAL STATS above for")
    print("                                  length/metadata mismatch; if those")
    print("                                  look consistent, these rows' stored")
    print("                                  arrays are unreproducible and their")
    print("                                  provenance must be escalated (this")
    print("                                  also bears on Family A, which")
    print("                                  consumed the same arrays).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

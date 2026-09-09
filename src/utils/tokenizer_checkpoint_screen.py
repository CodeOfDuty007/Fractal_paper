#!/usr/bin/env python3
"""
Stage 1 (CPU-only, no GPU, no torch model weights) of the Mistral checkpoint
hunt: screen candidate Mistral versions by RE-TOKENIZING sample human text
and comparing the resulting token count to GAGLE's own stored total-token
count for that row. This costs nothing but a small tokenizer download
(KB-MB, not the 14GB of model weights) and runs in seconds per candidate.

WHY HUMAN ROWS
  Human text is unmodified ground truth -- there is no "wrong generator"
  ambiguity the way there is for mistral-GENERATED documents. If a
  candidate's tokenizer doesn't even reproduce the right token COUNT on
  human text, it is not the checkpoint that scored this dataset, full stop.

WHAT THIS CANNOT DO (v2 -- this section was too confident in v1)
  GAGLE only stored a token COUNT per document, never the id sequence. So
  this test can only compare counts, never actual token ids. That is a
  real, structural limitation, not just a data-availability gap: two
  genuinely different vocabularies can produce the SAME count on a given
  piece of text while assigning completely different ids underneath
  (confirmed empirically on the companion Mac dry-run script -- v0.1 and
  v0.3, different 32000 vs 32768 vocabs, matched counts on plain English
  sentences despite totally different ids).

  Consequence: a count MISMATCH is strong evidence of elimination (real
  mismatch across n candidate rows, not a coincidence). A count MATCH is
  weaker evidence -- "not eliminated," not "confirmed." Do not skip
  Stage 2 for any survivor, even one that matches on every row here.

  Separately: v0.1 and v0.2 do NOT both exist as base checkpoints --
  mistralai/Mistral-7B-v0.2 was never released as a base model, only as
  Instruct-v0.2. Default candidates below reflect this.

DIAGNOSTIC MODE (v3 -- added after a real run on 2026-08-20)
  A real run against 20 GAGLE human rows eliminated all five candidates,
  but with IDENTICAL mean/min/max diff across every candidate -- i.e. every
  candidate agreed with every other candidate on n_ours per row (expected,
  see above -- shared vocab prefix), but ALL of them disagreed with
  gagle_mistral_total_tokens by a wildly inconsistent amount (19 to 12392
  tokens off, not a fixed offset). That pattern means the aggregate
  ELIMINATED/NOT-ELIMINATED verdict is not trustworthy on its own -- it may
  be comparing against a baseline field that doesn't mean "tokenize this
  row's text and count," rather than genuinely distinguishing candidates.

  --per-row and --dump-json exist to make that pattern visible instead of
  hiding it in an aggregate mean: per row, they report whether the loaded
  candidates actually agreed with EACH OTHER (the only rows this test can
  discriminate on at all), and a Pearson correlation between document
  length and diff -- a strong positive correlation points to something
  length-proportional missing from our retokenization (e.g. a prefix/
  prompt template baked into GAGLE's stored count); a weak/no correlation
  points instead to a per-row lookup or alignment problem (wrong field,
  wrong text for that id) rather than a tokenizer-identity problem at all.

USAGE
  SERVER HAS NO INTERNET ACCESS, so tokenizer files are downloaded on a
  machine that has internet (e.g. your Mac, via curl -- see the companion
  download commands) and moved over. Point this at the parent folder that
  contains one subfolder per candidate, named by repo id with "/" -> "_"
  (e.g. mistralai_Mistral-7B-v0.1/tokenizer.json, tokenizer_config.json, ...):

    python3 tokenizer_checkpoint_screen.py --n 20 \
        --tokenizer-dir /data/user/paper/mistral_tokenizers

    # add the per-row breakdown + correlation diagnostic:
    python3 tokenizer_checkpoint_screen.py --n 20 \
        --tokenizer-dir /data/user/paper/mistral_tokenizers \
        --per-row --dump-json mistral_screen_diag.json

  transformers must already be installed on the server (it is, if
  build_p1_feature_store_gpu.py already runs there). No download happens
  at run time -- from_pretrained(..., local_files_only=True) will error
  immediately and clearly if a needed file is missing, rather than hanging
  trying to reach a network that isn't there.

VERIFIED (no network / no real data available in the dev sandbox that wrote
this, so verified by simulation instead):
  - field auto-detection (gagle_mistral_total_tokens)
  - human-only row filtering, skipping rows with a missing total-tokens field
  - exact-match counting and NOT ELIMINATED/ELIMINATED summary, both branches
  - clean abort when transformers is missing, when no usable rows exist, and
    when a candidate's local tokenizer folder is missing/incomplete
  - v3 diagnostics: candidates_agree detection (all-agree and a forced
    disagreement case), pearson_r on a synthetic length-correlated series
    (recovers r > 0.99) and a synthetic uncorrelated/random series (|r| well
    under 0.6), pearson_r's None-return on a constant series, --per-row
    table rendering, --dump-json round-trips through json.load cleanly
  all exercised against a synthetic parquet shard + a stub tokenizer standing
  in for transformers.AutoTokenizer. Real tokenizer behaviour against real
  GAGLE data has NOT been re-verified after this edit -- that happens when
  you run this on the server.
"""

import argparse
import glob
import json
import os
import sys

CANDIDATES = [
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# Field names as used by build_p1_feature_store_gpu.py's SCORER_KEYS for the
# mistral-7b entry: (sliced_key, total_key). We only need total_key here.
TOTAL_TOKENS_FIELD_CANDIDATES = [
    "gagle_mistral_total_tokens",
    "gagle_mistral_7b_total_tokens",
]


def find_human_rows_with_total(manifest_dir, n_needed):
    """Scan shards for human rows carrying a mistral total-token count.
    Returns (field_name_used, list_of_rows). Auto-discovers rather than
    hardcoding a shard path -- hardcoded paths are exactly how earlier
    probes in this project got pointed at zero-human-row shards."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(manifest_dir, "*.parquet")))
    if not files:
        print(f"[abort] no .parquet files found under {manifest_dir}")
        return None, []

    field_used = None
    rows = []
    for f in files:
        try:
            table = pq.read_table(f)
        except Exception as exc:
            print(f"[warn] could not read {f}: {exc}")
            continue
        cols = set(table.column_names)
        if field_used is None:
            for cand in TOTAL_TOKENS_FIELD_CANDIDATES:
                if cand in cols and "text" in cols and "side" in cols:
                    field_used = cand
                    break
            if field_used is None:
                continue  # this shard's schema doesn't have what we need
        if field_used not in cols:
            continue
        for r in table.to_pylist():
            if "human" not in str(r.get("side", "")).lower():
                continue
            if not r.get("text") or r.get(field_used) is None:
                continue
            rows.append(r)
            if len(rows) >= n_needed:
                return field_used, rows
    return field_used, rows


def local_tokenizer_path(tokenizer_dir, repo_id):
    """repo id -> local folder name, matching the Mac-side download script's
    `tr '/' '_'` convention (mistralai/Mistral-7B-v0.1 -> mistralai_Mistral-7B-v0.1)."""
    return os.path.join(tokenizer_dir, repo_id.replace("/", "_"))


def pearson_r(xs, ys):
    """Manual Pearson correlation -- no numpy/scipy dependency for a script
    this small, and this needs to run on a CPU-only box that may not have
    them. Returns None when undefined (fewer than 2 points, or either series
    has zero variance -- e.g. every row the same length)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x ** 0.5 * var_y ** 0.5)


def build_row_diagnostics(rows, field_used, results, per_row_counts, candidates):
    """Per-row breakdown across whichever candidates actually loaded.

    For each row: does every loaded candidate report the SAME n_ours (the
    only rows this test can discriminate on at all), and if so, how does
    that shared n_ours differ from GAGLE's stored total. Kept separate from
    run() so it can be unit-tested without touching transformers/pyarrow."""
    loaded = [rid for rid in candidates if results.get(rid, {}).get("loaded")]
    diag = []
    for i, r in enumerate(rows):
        n_gagle = int(r[field_used])
        per_cand_n = {rid: per_row_counts[rid][i] for rid in loaded}
        values = list(per_cand_n.values())
        agree = (max(values) == min(values)) if values else None
        n_ours_repr = values[0] if (agree and values) else None
        diag.append({
            "row_index": i,
            "id": r.get("id", f"row{i}"),
            "text_chars": len(r.get("text", "") or ""),
            "n_gagle": n_gagle,
            "per_candidate_n_ours": per_cand_n,
            "candidates_agree": agree,
            "diff_repr": (n_ours_repr - n_gagle) if n_ours_repr is not None else None,
        })
    return diag


def print_row_diagnostics(row_diag, per_row):
    """Prints the agreement summary + length/diff correlation always;
    prints the full per-row table only if per_row=True (can be long)."""
    n_total = len(row_diag)
    agree_rows = [d for d in row_diag if d["candidates_agree"]]
    n_agree = len(agree_rows)

    print(f"\n[diag] candidates agree with EACH OTHER on n_ours "
          f"(identical count across every loaded candidate) on "
          f"{n_agree}/{n_total} rows.")
    if n_agree < n_total:
        print(f"[diag] {n_total - n_agree} row(s) show real disagreement between "
              f"candidates -- those are the only rows this test can actually "
              f"discriminate on, inspect them specifically.")

    r_val = None
    if n_agree >= 2:
        lengths = [d["text_chars"] for d in agree_rows]
        diffs = [d["diff_repr"] for d in agree_rows]
        r_val = pearson_r(lengths, diffs)
        if r_val is None:
            print("[diag] correlation(text length, diff) undefined (no variance "
                  "in length or diff across the agreeing rows).")
        else:
            print(f"[diag] correlation(text length in chars, diff = n_ours - n_gagle) "
                  f"over the {n_agree} agreeing rows: r = {r_val:.3f}")
            if r_val > 0.6:
                print("[diag] -> STRONG positive correlation with length: consistent "
                      "with gagle_mistral_total_tokens including something "
                      "PROPORTIONAL to document length that our retokenization does "
                      "not (e.g. a repeated prefix/prompt template per document).")
            elif r_val < -0.6:
                print("[diag] -> STRONG negative correlation with length: unexpected, "
                      "worth a closer look before assuming a prefix-inclusion cause.")
            else:
                print("[diag] -> weak/no correlation with length: the mismatch does "
                      "not look length-proportional. More consistent with a per-row "
                      "lookup/alignment problem (wrong field, wrong text for that id) "
                      "than with a tokenizer-identity problem.")

    if per_row:
        print(f"\n{'idx':<5}{'id':<24}{'chars':>8}{'n_gagle':>10}{'n_ours':>10}{'diff':>8}  agree")
        for d in sorted(row_diag, key=lambda x: x["text_chars"]):
            if d["diff_repr"] is not None:
                n_ours_disp = d["diff_repr"] + d["n_gagle"]
                diff_disp = d["diff_repr"]
            else:
                n_ours_disp = "DISAGREE"
                diff_disp = "--"
            print(f"{d['row_index']:<5}{str(d['id'])[:22]:<24}{d['text_chars']:>8}"
                  f"{d['n_gagle']:>10}{str(n_ours_disp):>10}{str(diff_disp):>8}  "
                  f"{d['candidates_agree']}")

    return r_val


def run(manifest_dir, n, candidates, tokenizer_dir, per_row=False, dump_json=None):
    try:
        import transformers
    except ImportError:
        print("[abort] transformers is not installed. It should already be present "
              "if build_p1_feature_store_gpu.py runs on this machine -- check the venv.")
        return 1

    field_used, rows = find_human_rows_with_total(manifest_dir, n)
    if not rows:
        print(f"[abort] found 0 usable human rows with a stored mistral "
              f"total-token count under {manifest_dir}")
        print("        Checked field names:", TOTAL_TOKENS_FIELD_CANDIDATES)
        print("        Check the manifest schema -- run:")
        print("        python3 -c \"import pyarrow.parquet as pq, glob; "
              "print(pq.read_table(sorted(glob.glob('%s/*.parquet'))[0]).column_names)\"" % manifest_dir)
        return 1

    print(f"[info] using stored-total field: {field_used}")
    print(f"[info] testing on {len(rows)} human rows")
    print(f"[info] candidates: {', '.join(candidates)}\n")

    results = {}
    per_row_counts = {}  # repo_id -> [n_ours per row], aligned with `rows` order
    for repo_id in candidates:
        print(f"--- {repo_id} ---")
        local_path = local_tokenizer_path(tokenizer_dir, repo_id)
        if not os.path.isdir(local_path):
            print(f"  [FAIL] no local folder at {local_path}\n"
                  f"         (download it on a machine with internet and copy it over)\n")
            results[repo_id] = {"loaded": False}
            continue
        try:
            tok = transformers.AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        except Exception as exc:
            print(f"  [FAIL] could not load tokenizer from {local_path}: {exc}\n")
            results[repo_id] = {"loaded": False}
            continue

        exact = 0
        diffs = []
        n_ours_list = []
        for r in rows:
            n_ours = len(tok(r["text"])["input_ids"])
            n_gagle = int(r[field_used])
            d = n_ours - n_gagle
            diffs.append(d)
            n_ours_list.append(n_ours)
            if d == 0:
                exact += 1
        per_row_counts[repo_id] = n_ours_list

        results[repo_id] = {
            "loaded": True,
            "exact_matches": exact,
            "n": len(rows),
            "mean_diff": sum(diffs) / len(diffs),
            "min_diff": min(diffs),
            "max_diff": max(diffs),
        }
        print(f"  exact token-count matches: {exact}/{len(rows)}")
        print(f"  diff (ours - gagle): mean={results[repo_id]['mean_diff']:.2f} "
              f"min={results[repo_id]['min_diff']} max={results[repo_id]['max_diff']}\n")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    not_eliminated = []
    for repo_id, res in results.items():
        if not res.get("loaded"):
            print(f"  {repo_id:<35} could not load")
            continue
        verdict = "NOT ELIMINATED (count matches every row)" if res["exact_matches"] == res["n"] else "ELIMINATED (count mismatch)"
        if res["exact_matches"] == res["n"]:
            not_eliminated.append(repo_id)
        print(f"  {repo_id:<35} {res['exact_matches']}/{res['n']} exact  ->  {verdict}")

    print()
    print("[caveat] 'NOT ELIMINATED' is a count match, not an id match -- GAGLE never "
          "stored ids, so a coincidental match cannot be ruled out here (see docstring). "
          "Elimination is the strong result of this test; survival is not.")
    if not_eliminated:
        print(f"\n[next] Stage 2 (needs GPU): run the real scoring test "
              f"(regress_candidate_cap.py-style, r vs GAGLE's stored scores) "
              f"on EVERY not-eliminated candidate, no shortcuts: {', '.join(not_eliminated)}")
    else:
        print("\n[next] every candidate eliminated by token count. Re-check the manifest "
              "field/values, widen --candidates, or wait for the author's reply.")

    row_diag = build_row_diagnostics(rows, field_used, results, per_row_counts, candidates)
    print_row_diagnostics(row_diag, per_row)

    if dump_json:
        with open(dump_json, "w") as fh:
            json.dump({
                "field_used": field_used,
                "candidates": candidates,
                "results": results,
                "row_diagnostics": row_diag,
            }, fh, indent=2)
        print(f"\n[info] wrote per-row diagnostics to {dump_json}")

    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="human rows to test (default 20)")
    ap.add_argument("--manifest-dir", default=os.environ.get(
        "GAGLE_MANIFEST_DIR", "/data/user/paper/gpu_manifest_shards"))
    ap.add_argument("--candidates", nargs="+", default=CANDIDATES)
    ap.add_argument("--tokenizer-dir", required=True,
                    help="parent folder holding one subfolder per candidate "
                         "(repo id with / -> _), downloaded on a machine with internet")
    ap.add_argument("--per-row", action="store_true",
                    help="print the full per-row n_ours/n_gagle/diff table, "
                         "sorted by document length")
    ap.add_argument("--dump-json", default=None,
                    help="write per-row diagnostics (agreement, diff, correlation "
                         "inputs) to this path for offline inspection")
    args = ap.parse_args()
    return run(args.manifest_dir, args.n, args.candidates, args.tokenizer_dir,
               per_row=args.per_row, dump_json=args.dump_json)


if __name__ == "__main__":
    sys.exit(main())

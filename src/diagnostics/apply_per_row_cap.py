#!/usr/bin/env python3
"""
Apply the per-row candidate-cap fix to src/build_p1_feature_store_gpu.py.

WHAT CHANGES AND WHY
  MAX_CANDIDATE_TOKENS is a RECONSTRUCTION parameter, not a resource limit.
  Truncating a candidate to `cap` sets

      delta  = n - gagle_total
      anchor = align_window.start + delta   (~64 + delta)

  and sweep_best_alignment searches anchor +/- OFFSET_SWEEP_RADIUS (150).
  With cap = gagle_total + m that is anchor ~= 64 + m, and the empirically
  observed true offset is 65, so the alignment is only reachable while

      |64 + m - 65| <= 150     ->     m <= ~151

  The flat 2048 works on gemma human rows ONLY because their gagle_total is
  2047 (m = 1 -> anchor 65). The same flat value silently drops mistral human
  rows (their cap is 3071 > 2048, so hi-lo < n_gagle and compute_doc_score_range
  returns None before any forward pass -- measured 99.7% NO_CANDIDATE). And
  raising the flat value to 4096 breaks the gemma rows (m = 2049 -> anchor
  2113, ~2000 tokens outside the window: 12/12 r=1.0000 -> 0/12 r=0.15).

  Hence: per row, per scorer, cap = gagle_<scorer>_total_tokens + 1.

BLAST RADIUS
  Truncation only binds on documents LONGER than the cap. Where it does not
  bind, `ids` is byte-identical to before, so short/medium rows -- which is
  every currently-passing LLM cell -- are provably unaffected. When
  gagle_total is None the function falls back to the OLD constant, so any
  caller that does not pass it (e.g. probe_cross_column_alignment.py) keeps
  its existing behaviour rather than silently jumping to the 4096 regime.

EDITS (5, each asserted to match exactly once)
  1. constants: add MAX_CANDIDATE_TOKENS_HARD + CAP_MARGIN
  2. tokenize_candidates signature: + gagle_total=None
  3. tokenize_candidates body: flat cap -> per-row cap
  4. call site in verify_alignment  (+ row.get(total_key))
  5. call site in main              (+ row.get(total_key))

USAGE
  python3 apply_per_row_cap.py --check     # verify targets, change nothing
  python3 apply_per_row_cap.py --apply     # writes .bak, patches, compiles

AFTER APPLYING
  python3 regress_candidate_cap.py --compare regress_before.json \
                                   --baseline regress_after.json
"""

import argparse
import os
import py_compile
import shutil
import sys

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "src", "build_p1_feature_store_gpu.py")

OLD_CONST = ('MAX_CANDIDATE_TOKENS = 2048  '
             '# matches the human-side cap the probe confirmed at r=1.0000')

NEW_CONST = '''MAX_CANDIDATE_TOKENS = 2048  # fallback ONLY, for callers that do not supply a
                             # per-row gagle_total. Not a resource limit -- see
                             # tokenize_candidates for why this is a
                             # reconstruction parameter.
MAX_CANDIDATE_TOKENS_HARD = 4096  # OOM guard. Must exceed the largest
                                  # gagle_*_total_tokens in the corpus
                                  # (GAGLE's own caps: gemma 2047,
                                  # mistral 3071).
CAP_MARGIN = 1  # `m` in the derivation; anchor lands at ~64 + m and the true
                # offset is 65, so this must stay well under
                # OFFSET_SWEEP_RADIUS.'''

OLD_SIG = 'def tokenize_candidates(tok, text, prefix):'
NEW_SIG = 'def tokenize_candidates(tok, text, prefix, gagle_total=None):'

OLD_BODY = '''    text_ids = tok(text)["input_ids"]
    text_len = len(text_ids)
    candidates = {
        "text_alone": {"ids": text_ids[:MAX_CANDIDATE_TOKENS], "prefix_len": None, "text_len": text_len},
    }
    if prefix:
        prefix_ids = tok(prefix)["input_ids"]
        combined = (prefix_ids + text_ids[1:])[:MAX_CANDIDATE_TOKENS]'''

NEW_BODY = '''    text_ids = tok(text)["input_ids"]
    text_len = len(text_ids)
    # Per-row, per-scorer cap. `gagle_total` is this scorer's
    # gagle_<scorer>_total_tokens for THIS row. Truncating to `cap` sets
    # delta = n - gagle_total, and sweep_best_alignment anchors at
    # align_window.start + delta (~64 + delta) with radius
    # OFFSET_SWEEP_RADIUS -- so cap = gagle_total + CAP_MARGIN puts the anchor
    # at ~65, which is the empirically observed winning offset. A flat cap
    # cannot serve both scorers: GAGLE capped gemma human arrays at 2047 and
    # mistral at 3071. None -> old constant, preserving legacy callers.
    if gagle_total:
        cap = min(MAX_CANDIDATE_TOKENS_HARD, int(gagle_total) + CAP_MARGIN)
    else:
        cap = MAX_CANDIDATE_TOKENS
    candidates = {
        "text_alone": {"ids": text_ids[:cap], "prefix_len": None, "text_len": text_len},
    }
    if prefix:
        prefix_ids = tok(prefix)["input_ids"]
        combined = (prefix_ids + text_ids[1:])[:cap]'''

OLD_CALL1 = ('        row_candidates = [tokenize_candidates(tok, row["text"], '
             'row.get("prefix")) for row in sample_rows]')
NEW_CALL1 = ('        row_candidates = [tokenize_candidates(tok, row["text"], '
             'row.get("prefix"), row.get(total_key)) for row in sample_rows]')

OLD_CALL2 = ('                row_candidates = [tokenize_candidates(tok, row["text"], '
             'row.get("prefix")) for row in rows]')
NEW_CALL2 = ('                row_candidates = [tokenize_candidates(tok, row["text"], '
             'row.get("prefix"), row.get(total_key)) for row in rows]')

EDITS = [
    ("1. constants",                 OLD_CONST, NEW_CONST),
    ("2. signature",                 OLD_SIG,   NEW_SIG),
    ("3. body (flat -> per-row)",    OLD_BODY,  NEW_BODY),
    ("4. call site: verify_alignment", OLD_CALL1, NEW_CALL1),
    ("5. call site: main",           OLD_CALL2, NEW_CALL2),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--target", default=TARGET)
    args = ap.parse_args()

    if not os.path.exists(args.target):
        print(f"[abort] not found: {args.target}")
        return 2
    src = open(args.target).read()

    if "MAX_CANDIDATE_TOKENS_HARD" in src:
        print("[abort] already patched (MAX_CANDIDATE_TOKENS_HARD present).")
        return 1

    print(f"target: {args.target}\n")
    ok = True
    for label, old, _new in EDITS:
        n = src.count(old)
        flag = "OK" if n == 1 else "FAIL"
        if n != 1:
            ok = False
        print(f"  [{flag:<4}] {label:<32} matches={n}")
    if not ok:
        print("\n[abort] every target must match EXACTLY once. File differs from "
              "what this patch was written against -- do not force it.")
        return 1
    print("\nAll 5 targets matched exactly once.")

    if not args.apply:
        print("Dry run. Re-run with --apply to write.")
        return 0

    bak = args.target + ".bak"
    shutil.copy2(args.target, bak)
    for _label, old, new in EDITS:
        src = src.replace(old, new, 1)
    open(args.target, "w").write(src)

    try:
        py_compile.compile(args.target, doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(bak, args.target)
        print(f"[abort] patched file does not compile; ROLLED BACK.\n{exc}")
        return 1

    print(f"APPLIED. Backup at {bak}")
    print("\nNext:")
    print("  python3 regress_candidate_cap.py --compare regress_before.json \\")
    print("                                   --baseline regress_after.json")
    print("\nRollback:  cp %s %s" % (bak, args.target))
    return 0


if __name__ == "__main__":
    sys.exit(main())

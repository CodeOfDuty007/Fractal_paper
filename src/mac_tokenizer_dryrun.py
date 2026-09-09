#!/usr/bin/env python3
"""
Mac-local dry run for candidate Mistral tokenizers -- NOT the real Stage 1
test (that needs gagle_mistral_total_tokens, which only exists in the full
manifest on the server, confirmed absent from every local copy on this Mac).
This script instead:

  1. confirms each downloaded tokenizer folder actually loads offline
     (local_files_only=True, identical call to the real server script)
  2. tokenizes a handful of ordinary sample sentences with each candidate
  3. compares FULL TOKEN ID SEQUENCES pairwise, not just counts

WHY ID SEQUENCES, NOT COUNTS (v2 -- this was wrong in v1)
  Two genuinely different vocabularies can produce the SAME token count on
  ordinary prose while assigning completely different ids -- confirmed
  empirically here: v0.1 (32000 vocab) and v0.3 (32768 vocab, +768
  control/reserved tokens) tokenized 5 plain sentences to identical counts
  but divergent ids. Comparing counts alone would have wrongly reported
  v0.1 and v0.3 as indistinguishable. Comparing ids catches this correctly.

  This does NOT fully carry over to the real server-side Stage 1 test
  (tokenizer_checkpoint_screen.py): GAGLE only stored a token COUNT per
  document, never the id sequence, so there is nothing to compare ids
  against there even in principle. That test's "SURVIVES" verdict is
  therefore weaker than a clean pass here -- it means "not eliminated by
  count," not "confirmed." Stage 2 (real weights, real scoring, real r
  against GAGLE's stored scores) is required for every survivor, not only
  for candidates that share a tokenizer.

CANDIDATES (v2 -- corrected)
  mistralai/Mistral-7B-v0.2 does NOT exist as a base checkpoint on the Hub
  -- v0.2 shipped Instruct-only. Removed the fictional base entry; added
  the three Instruct variants as legitimate candidates, since it is
  plausible GAGLE's scoring pipeline used (or was mistakenly pointed at)
  an instruct-tuned checkpoint rather than a base one.

  If a candidate's local folder contains a PROVENANCE.txt (e.g. the v0.2
  folder populated from Instruct-v0.2 as a stand-in), this script prints
  its contents so that substitution is never silently lost.

USAGE
  source .venv/bin/activate   # project venv -- has transformers already;
                               # do NOT use system python3/pip3 on this Mac,
                               # its pyexpat is broken at the OS link level
  python3 mac_tokenizer_dryrun.py --tokenizer-dir /path/to/fractal_analysis_paper/mistral_tokenizers
"""

import argparse
import os
import sys

SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1847, the treaty was signed by representatives of both nations.",
    "def compute(x, y):\n    return x + y * 2",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "She said, \"I don't think that's quite right,\" and walked away.",
]

CANDIDATES = [
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def local_tokenizer_path(tokenizer_dir, repo_id):
    return os.path.join(tokenizer_dir, repo_id.replace("/", "_"))


def run(tokenizer_dir, candidates):
    try:
        import transformers
    except ImportError:
        print("[abort] transformers not importable. Activate the project venv "
              "(source .venv/bin/activate) -- do not use system python3/pip3 on this Mac.")
        return 1

    loaded = {}  # repo_id -> list of id-lists, one per sample
    for repo_id in candidates:
        path = local_tokenizer_path(tokenizer_dir, repo_id)
        print(f"--- {repo_id} ({path}) ---")
        if not os.path.isdir(path):
            print(f"  [FAIL] folder not found\n")
            continue

        prov = os.path.join(path, "PROVENANCE.txt")
        if os.path.exists(prov):
            print(f"  [NOTE] PROVENANCE.txt present:")
            for line in open(prov):
                print(f"    {line.rstrip()}")

        try:
            tok = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
        except Exception as exc:
            print(f"  [FAIL] could not load: {exc}\n")
            continue

        ids_per_sample = [tok(s)["input_ids"] for s in SAMPLES]
        counts = [len(ids) for ids in ids_per_sample]
        loaded[repo_id] = ids_per_sample
        print(f"  loaded ok. token counts on {len(SAMPLES)} samples: {counts}\n")

    if len(loaded) < 2:
        print("[abort] fewer than 2 candidates loaded -- nothing to compare")
        return 1

    print("=" * 70)
    print("PAIRWISE COMPARISON (full id sequences, not just counts)")
    print("=" * 70)
    names = list(loaded.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ids_equal = loaded[a] == loaded[b]
            counts_a = [len(x) for x in loaded[a]]
            counts_b = [len(x) for x in loaded[b]]
            counts_equal = counts_a == counts_b
            if ids_equal:
                verdict = "IDENTICAL (same vocab, same ids)"
            elif counts_equal:
                verdict = "SAME COUNT BUT DIFFERENT IDS -- different vocab, coincidental count match"
            else:
                verdict = "DIFFERENT (different counts)"
            print(f"  {a}  vs  {b}")
            print(f"    {verdict}\n")

    print("[note] a clean pass here confirms the download is loadable and the id-level "
          "comparison logic works. It does NOT by itself confirm which candidate matches "
          "GAGLE's real mistral-7b scorer -- that still requires the server-side Stage 1 "
          "(count-only, weaker evidence -- eliminates mismatches, does not confirm matches) "
          "followed by Stage 2 (real weights, real scoring, real r-value).")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--candidates", nargs="+", default=CANDIDATES)
    args = ap.parse_args()
    return run(args.tokenizer_dir, args.candidates)


if __name__ == "__main__":
    sys.exit(main())

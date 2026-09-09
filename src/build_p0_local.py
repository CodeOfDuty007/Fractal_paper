#!/usr/bin/env python3
"""
Single-pass, resumable P0 corpus builder reading gagle.zip LOCALLY (no
network in the read loop). Supersedes build_p0_corpus.py, whose streaming
.shard() approach was reading byte-ranges into ibomohsin/gagle's single
remote gagle.zip and stalled unrecoverably at a fixed byte offset -- no
--num-shards value fixed it, because there was only ever one file.

SCHEMA (verified against the dataset's own GAGLE_Dataset.ipynb, which
outranks the andsteing/gagle HF Space's decompiled bundle.js as a
reference since it's the authors' own derivable code, not a minified UI
bundle):
  - Zip members live at gagle3/{split}/{model}_{temp}_{dataset}_{prompt_type}.jsonl
    -- each member IS one (model, temp, domain, prompt_type) stratification
    cell. This is why checkpointing is done per-member: it's the dataset's
    own natural unit of work, not an arbitrary reshard.
  - Each line is one JSON record with scores[SCORER]["llm_text"] /
    ["original_text"] as PLAIN LISTS of per-token floats (confirmed live,
    both via datasets.load_dataset streaming and via the notebook's own
    pretty_print_array usage -- NOT the nested token_ids/token_texts/
    token_scores dict shown in the HF dataset-card README, which appears to
    describe a different auto-generated viewer schema).

WHY Holder_Exp (S) IS NOT IN THE PER-DOC OUTPUT
  Validated against synthetic fBm with known Hurst exponent: per-document
  (~400 token) S is biased -0.26 to -0.48 regardless of bootstrap draw
  count -- there is not enough information in one document for
  p_epsilon(tau) to resolve a power law. S is only recoverable pooled
  across many documents (bias -0.004 to -0.065, R^2 0.91-0.995), which is
  also how the reference notebook itself applies it (concatenated across a
  document range, never per document). So per-doc rows carry Hurst_Exp
  only; corpus-level S/H per (domain, label) cell is computed once, after
  all members are processed, from pooled raw scores.

WHY calc_H's per-doc row IS kept
  calc_H (this repo's version: non-overlapping len//n chunks) was checked
  against the reference's get_hurst_exponent (samples_per_doc random
  overlapping windows) on synthetic fBm across 30 trials at H=0.3/0.5/
  0.65/0.8: mean bias between the two agreed within ~0.01 at every H,
  well inside each estimator's own run-to-run noise. The chunking
  difference changes variance, not bias -- cosmetic, not a defect.

CHECKPOINTING
  data/p0_local_member_checkpoint.txt: one line per FULLY processed zip
  member, fsynced after each member completes. A member is the atomic unit
  of work -- if the process dies mid-member, the whole member is redone on
  restart (cheap: it's a local read of at most a few thousand JSON lines),
  rather than tracking per-document progress within a member.

  Corpus-level pooling (raw score arrays grouped by (domain, label)) lives
  only in memory during a run. On resume, already-checkpointed members are
  replayed (read + accumulate only, no per-doc compute or output-file
  writes) to rebuild the in-memory pools before continuing with new
  members. This avoids persisting a second multi-GB raw-score cache to
  disk -- replay of local, already-downloaded members is fast.

USAGE
    python3 build_p0_local.py                 # full run (resumable)
    python3 build_p0_local.py --limit-members 2   # smoke test: only process
                                                    # the first N not-yet-done
                                                    # members, then stop
                                                    # (checkpoints normally,
                                                    # safe to continue after)
"""
import argparse
import collections
import fnmatch
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractal_math import generate_trajectory, calc_H, calc_S

ZIP_PATH = "/Volumes/NikPhotoSSD/paper/gagle.zip"
MEMBER_PATTERN = "gagle3/*/*.jsonl"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT_PATH = os.path.join(DATA_DIR, "p0_local_full_results.jsonl")
MEMBER_CKPT_PATH = os.path.join(DATA_DIR, "p0_local_member_checkpoint.txt")
MANIFEST_PATH = os.path.join(DATA_DIR, "p0_local_manifest.json")
CORPUS_SH_PATH = os.path.join(DATA_DIR, "p0_corpus_level_SH.json")

MIN_TOKENS = 320
WARMUP_DROP = 64
CLIP_TO = 464
SCORER = "gemma-2b"


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_line_set(path):
    s = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    s.add(line)
    return s


def append_line(path, value):
    with open(path, "a") as f:
        f.write(str(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def slice_scores(raw):
    if not raw:
        return None
    if len(raw) >= CLIP_TO:
        sliced = raw[WARMUP_DROP:CLIP_TO]
    else:
        sliced = raw
    if len(sliced) < MIN_TOKENS:
        return None
    return sliced


def extract_rows(entry, split):
    """Yield 0-2 row dicts (llm side, human side) from one raw GAGLE record.

    split (train/val/test) is threaded through and included in both the id
    and the output row: GAGLE's dataset_id numbering (e.g.
    "00013-of-00256__513") resets per split, so the SAME dataset_id string
    can refer to a DIFFERENT underlying article in train vs val vs test.
    Without split in the key, human-doc dedup (below) could wrongly treat
    two unrelated articles from different splits as the same repeat and
    silently drop one. split is also needed downstream to reconstruct
    GAGLE's own train/val/test partition (relevant for any later
    classifier work per Extension1's plan) -- there would be no way to
    recover it later if it isn't recorded now.
    """
    domain = entry.get("dataset_name")
    scores_block = entry.get("scores", {}).get(SCORER, {})

    try:
        llm_raw = scores_block["llm_text"]
    except (KeyError, TypeError):
        llm_raw = None
    llm_sliced = slice_scores(llm_raw)
    if llm_sliced is not None:
        yield {
            "id": str(entry.get("_id")),
            "split": split,
            "domain": domain,
            "label": entry.get("model", "Unknown_LLM"),
            "prompt_type": entry.get("prompt_type"),
            "temp": entry.get("temperature"),
            "scores": llm_sliced,
        }

    # Human side recurs across every prompt_type/temp file for the same
    # domain (within a split) -- key on (split, domain, dataset_id) so the
    # corpus-level Human pool and the per-doc output both dedup repeats
    # instead of counting the same article once per member it appears in.
    # domain MUST be part of the key: GAGLE's dataset_id numbering is not
    # domain-scoped -- verified directly against the real zip that 239 of
    # 11,457 (split, dataset_id) keys are shared between big_patent and
    # wikipedia (e.g. ('test', '00002-of-00256__21') names a different real
    # article in each). Without domain in the key, whichever domain's row
    # was processed first would silently steal the id and the other
    # domain's actual human doc would be wrongly skipped as a "duplicate".
    try:
        human_raw = scores_block["original_text"]
    except (KeyError, TypeError):
        human_raw = None
    human_sliced = slice_scores(human_raw)
    if human_sliced is not None:
        yield {
            "id": f"{split}_{domain}_{entry.get('dataset_id')}_human",
            "split": split,
            "domain": domain,
            "label": "Human",
            "prompt_type": "continue",
            "temp": "Human",
            "scores": human_sliced,
        }


def process_member(zf, member, groups, human_ids_seen, stats, write_output=True):
    """Returns a list of per-doc output rows (empty if write_output=False).

    Rows are buffered and returned rather than written directly so the
    caller can write+fsync+checkpoint a member atomically: if the process
    is killed partway through a member, nothing from that member has been
    written yet, so redoing it on restart can't produce duplicate rows in
    the output file.
    """
    split = member.split("/")[1]  # gagle3/{split}/{model}_{temp}_{dataset}_{prompt_type}.jsonl
    out_rows = []
    with zf.open(member) as fh:
        for line_bytes in fh:
            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue
            stats["raw_entries_seen"] += 1
            for row in extract_rows(entry, split):
                if row["label"] == "Human":
                    if row["id"] in human_ids_seen:
                        stats["skipped_dup_human"] += 1
                        continue
                    human_ids_seen.add(row["id"])

                # Store as float64 arrays, not python lists: a python list of
                # floats runs ~8x the memory of the equivalent numpy array
                # (per-object overhead + pointers), and these pools are kept
                # in memory for the entire run.
                groups[(row["domain"], row["label"])].append(np.asarray(row["scores"], dtype=np.float64))

                if write_output:
                    trajectory, increment = generate_trajectory(row["scores"])
                    H, H_r2 = calc_H(increment)
                    out_rows.append({
                        "id": row["id"],
                        "split": row["split"],
                        "domain": row["domain"],
                        "Label": row["label"],
                        "prompt_type": row["prompt_type"],
                        "temp": row["temp"],
                        "Hurst_Exp": float(H),
                        "Hurst_R2": float(H_r2),
                    })
                    stats["written"] += 1
    return out_rows


def compute_corpus_level_SH(groups):
    results = {}
    for (domain, label), score_lists in sorted(groups.items()):
        if len(score_lists) < 10:
            continue
        # Build increments first, concatenate, compute H, then free before
        # building trajectories for calc_S -- avoids holding both the
        # per-doc trajectory list and increment list in memory at once,
        # which matters most on the largest cells after hours of a run.
        increments = [generate_trajectory(s)[1] for s in score_lists]
        all_incr = np.concatenate(increments)
        H, H_r2 = calc_H(all_incr)
        del increments, all_incr
        trajectories = [generate_trajectory(s)[0] for s in score_lists]
        S, S_r2 = calc_S(trajectories)
        n_tokens = sum(len(s) for s in score_lists)
        results[f"{domain}|{label}"] = {
            "domain": domain,
            "label": label,
            "n_docs": len(score_lists),
            "n_tokens": n_tokens,
            "S": float(S),
            "S_R2": float(S_r2),
            "H": float(H),
            "H_R2": float(H_r2),
        }
        print(f"[corpus-level] {domain:20s} {label:12s} n_docs={len(score_lists):6d} "
              f"S={S:.4f} S_R2={S_r2:.3f} H={H:.4f} H_R2={H_r2:.3f}")
    return results


def main(limit_members=None, force_corpus_level=False):
    if not os.path.exists(ZIP_PATH):
        print(f"[abort] {ZIP_PATH} not found. Run download_gagle_zip.py first and let it "
              f"report [done] before running this.")
        sys.exit(1)

    _ensure_data_dir()
    done_members = load_line_set(MEMBER_CKPT_PATH)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        all_members = sorted(m for m in zf.namelist() if fnmatch.fnmatch(m, MEMBER_PATTERN))
        print(f"[info] {len(all_members)} members match {MEMBER_PATTERN} in the zip; "
              f"{len(done_members)} already checkpointed done.")

        groups = collections.defaultdict(list)
        human_ids_seen = set()
        stats = collections.Counter()

        # Replay already-done members to rebuild the in-memory corpus-level
        # pools (raw scores are never persisted to disk a second time --
        # see module docstring). No output rows are (re)written for these.
        already_done_sorted = [m for m in all_members if m in done_members]
        if already_done_sorted:
            print(f"[resume] replaying {len(already_done_sorted)} already-done members "
                  f"to rebuild in-memory pools (no output rewritten, no network involved)...")
            for i, member in enumerate(already_done_sorted, 1):
                process_member(zf, member, groups=groups,
                                human_ids_seen=human_ids_seen, stats=stats, write_output=False)
                if i % 20 == 0:
                    print(f"[resume] replayed {i}/{len(already_done_sorted)}")

        remaining = [m for m in all_members if m not in done_members]
        if limit_members:
            remaining = remaining[:limit_members]
            print(f"[smoke-test] limiting to first {limit_members} not-yet-done members")

        with open(OUT_PATH, "a") as fout:
            for i, member in enumerate(remaining, 1):
                # Rows are buffered by process_member and written here in one
                # shot, after the member is fully read -- if the process
                # dies mid-member, nothing for that member has hit disk yet,
                # so redoing it on restart can't duplicate output rows.
                rows = process_member(zf, member, groups=groups,
                                       human_ids_seen=human_ids_seen, stats=stats, write_output=True)
                for row in rows:
                    fout.write(json.dumps(row) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                append_line(MEMBER_CKPT_PATH, member)
                done_members.add(member)
                print(f"[member complete] {i}/{len(remaining)} ({member}) "
                      f"written={stats['written']} raw_seen={stats['raw_entries_seen']}")

        complete = (len(done_members) >= len(all_members)) and not limit_members
        coverage_ok = set(done_members) == set(all_members)

        manifest = {
            "members_total": len(all_members),
            "members_done": len(done_members),
            "members_missing": sorted(set(all_members) - set(done_members))[:20],
            "raw_entries_seen_this_run": stats["raw_entries_seen"],
            "docs_written_this_run": stats["written"],
            "skipped_dup_human_this_run": stats["skipped_dup_human"],
            "json_errors_this_run": stats["json_errors"],
            "complete": bool(complete),
            "member_coverage_gap_free": bool(coverage_ok),
        }
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)

        print("\n=== SUMMARY ===")
        for k, v in manifest.items():
            print(f"  {k}: {v}")

        if complete and coverage_ok:
            print("\n[complete] All members done. Computing corpus-level S/H per (domain, label)...")
            corpus_results = compute_corpus_level_SH(groups)
            with open(CORPUS_SH_PATH, "w") as f:
                json.dump(corpus_results, f, indent=2)
            print(f"[done] corpus-level S/H written to {CORPUS_SH_PATH}")
        elif force_corpus_level:
            print("\n[smoke-test] --force-corpus-level set: computing corpus-level S/H on PARTIAL "
                  "data (not all members done) purely to exercise the code path. NOT a real result.")
            corpus_results = compute_corpus_level_SH(groups)
            print(f"[smoke-test] {len(corpus_results)} (domain,label) cells computed on partial data "
                  f"-- not written to {CORPUS_SH_PATH} since the run is incomplete.")
        else:
            print("\n[incomplete] Rerun this exact command to continue -- already-done members "
                  "are skipped (after a quick in-memory replay). Corpus-level S/H is only "
                  "computed once every member is done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit-members", type=int, default=None,
                    help="Smoke-test: only process the first N not-yet-done members, then stop.")
    p.add_argument("--force-corpus-level", action="store_true",
                    help="Smoke-test: compute corpus-level S/H even if not all members are done. "
                         "Use with --limit-members to exercise that code path without a full run.")
    args = p.parse_args()
    main(limit_members=args.limit_members, force_corpus_level=args.force_corpus_level)

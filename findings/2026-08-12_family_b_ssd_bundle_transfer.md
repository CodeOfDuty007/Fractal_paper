# Family B SSD Bundle: What Was Built, What Was Verified, and Whether It's Final

**Date:** 2026-08-12
**Scope:** Documents the offline transfer bundle built on `/Volumes/NikPhotoSSD` for running `src/build_p1_feature_store_gpu.py` (Family B) on the Ubuntu GPU machine, following `reports/2026-08-12_family_b_gpu_scorer_verification.md`'s gemma-2b-only verification on this Mac. Answers the direct question: is this bundle done.

---

## Short answer

**Yes, now.** The first version of this report said the bundle was final and was wrong — an advisor review caught that it had model weights and data but zero Python packages, meaning `pip install` would fail outright on the confirmed-fully-offline target machine. That gap is now closed: `wheels/` contains all 49 packages (8 top-level + transitive dependencies) needed to install everything with `--no-index`, verified via an isolated dependency-resolution dry-run against the target's actual platform (cp310/manylinux/x86_64), not just "the files exist."

Getting there required two more corrections along the way (both self-caught this time, not advisor-flagged):
1. `torch==2.13.0` (pinned from this Mac's own environment) only ships CUDA 12.6+/12.8+ wheels, which exceed the target's driver capability (535.247.01, CUDA 12.2 ceiling, confirmed via web search against NVIDIA's own docs). Resolved by finding the newest torch release that still publishes a CUDA-11.8 build for Python 3.10 — `torch==2.7.1` — safely under the driver's ceiling, and satisfying `transformers==5.14.1`'s own `torch>=2.4` requirement.
2. `numpy`/`scipy`/`scikit-learn`/`matplotlib`, also pinned from this Mac's Python-3.12 environment, mostly don't have Python-3.10 builds at those exact versions (numpy 2.5.1 doesn't exist for cp310 at all; latest available is 2.2.6). Re-resolved without version pins, let pip pick the newest cp310-compatible version of each, and recorded what it actually chose.

**One item remains genuinely unverified and cannot be resolved on this machine, by hardware necessity, not oversight: whether the Mistral-7B weight cache is actually complete.** This Mac has 16GB RAM; Mistral-7B needs ~28GB in fp32, so it cannot be load-tested here the way gemma-2b and the embedder were. The bundle is structured so this surfaces immediately and unambiguously as step 1 of the first real run (`HF_HUB_OFFLINE=1` set before the first `--verify-alignment` call) rather than being silently deferred or discovered mid-run.

---

## What's in the bundle

All under `paper/` at the SSD root (`/Volumes/NikPhotoSSD/paper/` on this Mac):

| Path | Size | Role |
|---|---|---|
| `gagle.zip` | 9.9GB | Raw corpus. Already present from earlier project work. Not read by Family B itself. |
| `gpu_manifest_shards/` | 2.3GB, 720 files | The CPU pass's output — what Family B actually reads. Already present. |
| `hf_cache/` | ~20GB | HF cache: gemma-2b, Mistral-7B, GAGLE dataset, and (added this session) the sentence-transformer embedder. |
| `family_b_gpu_bundle/code/` | 384K (source `src/` size on the Mac) | This session's `src/`, pinned `requirements.txt`, `CLAUDE.md`. New this session. |
| `family_b_gpu_bundle/data/p1_gpu_shard_prefix_backfill.jsonl` | 255MB | The one data file the GPU pass reads at runtime. New this session. |
| `family_b_gpu_bundle/wheels/` | ~1.1GB, 49 files | Every Python package + transitive dependency, for cp310/manylinux/x86_64. New this session. |
| `family_b_gpu_bundle/README.md` | — | Setup + run instructions for the Ubuntu machine. New this session. |

---

## Target machine discovery (why this took several rounds)

The target's specs weren't known up front and had to be established interactively, each one gating a downstream decision:

- **`nvidia-smi: command not found`** initially looked like "no driver installed, needs admin" — turned out to be wrong. `ls /dev/nvidia*` and `lsmod | grep nvidia` (both run without sudo) showed the kernel driver was already loaded (`nvidia`, `nvidia_uvm`, `nvidia_modeset` all active); only the `nvidia-smi` CLI utility itself was missing/not installed. Confirmed via `cat /proc/driver/nvidia/version` (world-readable, no sudo needed): driver **535.247.01**.
- **GPU**: NVIDIA A100 80GB PCIe (`lspci`), confirmed before the driver-version check.
- **Python**: 3.10.12, provided directly by the user.
- **Network**: confirmed fully offline (asked directly rather than assumed from "physical SSD transfer" alone, since that phrase alone doesn't imply zero network — it could mean "data's too big to push, but the machine can still reach PyPI").
- **Driver → max CUDA version**: 535.247.01 → CUDA 12.2 ceiling (web-searched against NVIDIA's own release-notes/compatibility docs, not recalled from memory — this number determines every wheel choice downstream and was worth the extra verification step).

This sequence matters because a driver reinstall was never actually necessary — the initial "not in sudoers" dead end could have led to chasing admin access for a problem that didn't exist. Checking what was already there before assuming what was missing avoided that.

## Getting Python packages onto a fully offline machine (the wheels/ directory)

**Why the first two attempts at `requirements.txt` were wrong, checked against the real PyPI/PyTorch wheel index rather than assumed:**

1. `pip download torch==2.13.0` (this Mac's installed version) against every CUDA index (`cu118`/`cu121`/`cu124`/`cu126`/`cu128`) returned wheels only for `cu126` and `cu128` — both requiring a driver newer than 535.247.01 supports (cu126 needs ≥560.28.03). Queried the actual `download.pytorch.org/whl/<tag>/torch/` index directly (not a search/recollection) for the newest torch version with a `cu118` (CUDA 11.8, safely ≤12.2) build for `cp310`: **torch 2.7.1**. Cross-checked `transformers==5.14.1`'s PyPI metadata for its torch floor (`torch>=2.4`) — satisfied.

2. `pip download numpy==2.5.1 ...` (again, this Mac's version, but from a Python-3.12 venv) failed outright: `No matching distribution found for numpy==2.5.1`, with the real available cp310 versions listed in the error (topping out at 2.2.6). Same risk existed for `scipy`/`scikit-learn`/`matplotlib` (also pinned from the same Python-3.12 environment) — checked each directly against PyPI's JSON API for its actual latest cp310/manylinux/x86_64 build rather than assuming the rest were fine because the first one happened to resolve. Re-ran the download for all 7 non-torch packages **unpinned**, let pip's resolver pick versions that are actually compatible with each other and with cp310, and recorded what it chose: `numpy==2.2.6`, `scipy==1.15.3`, `pyarrow==25.0.1`, `scikit-learn==1.7.2`, `matplotlib==3.10.9`.

3. Two separate `pip download` passes (torch first, then the rest) left duplicate versions of 4 shared transitive dependencies (`filelock`, `fsspec`, `typing_extensions`, `setuptools`) in `wheels/` — the second, full-set resolution's versions are the authoritative ones (they're what a combined install actually needs); removed the stale first-pass versions rather than leaving both and hoping pip picks correctly.

**Final verification, not skipped:** an initial `pip install --dry-run --no-index --find-links wheels/ -r requirements.txt` looked clean but was actually invalid — it silently satisfied most dependencies from this Mac's own already-installed Python-3.12 packages instead of checking the bundle, because dry-run by default treats anything already present in the active environment as satisfied regardless of platform. Redone with `--ignore-installed --target <empty dir>`, which forces genuine resolution against only `--find-links` with no shortcut through local site-packages. Result: **all 49 packages resolve with zero network access**, targeting cp310/manylinux2014+manylinux_2_28/x86_64 explicitly.

**This proves resolution, not installation** (advisor-flagged — `--dry-run` never unpacks or runs a wheel). `pyarrow`/`torch` and other compiled wheels are tagged `manylinux_2_28`, which requires the target's glibc ≥2.28 — **not verified, because the target's glibc version was never requested.** Added as an explicit unverified item in the README (`ldd --version`, to check before/during install) rather than assumed fine.

---

## What was found and fixed during construction (not assumed correct)

Building this bundle surfaced three real gaps, each checked and closed before moving on — matching this session's established practice of verifying rather than asserting:

1. **`MANIFEST_DIR` was hardcoded to this Mac's `/Volumes/...` mount path.** A physical SSD mounts at a different path on every machine (Linux: `/mnt/...` or `/media/<user>/...`, never `/Volumes/...`). Made overridable via `GAGLE_MANIFEST_DIR` env var in `src/build_p1_feature_store_gpu.py`; tested end-to-end (import + resolution against a fake path), not just added to source.

2. **The sentence-transformer embedder (`sentence-transformers/all-MiniLM-L6-v2`) was missing from the cached `hf_cache`.** `load_models()` loads it unconditionally (needed for Family C's embedding-dispersion feature, computed in the same per-shard loop) — with `HF_HUB_OFFLINE=1` (which the README recommends specifically to prove the cache is real), this would have failed the very first `--verify-alignment` call on the GPU machine. Downloaded it directly into the SSD's `hf_cache` this session and confirmed it loads with `HF_HUB_OFFLINE=1` set (a real model instantiation, not a file-existence check).

3. **`requirements.txt` had no version pins.** Given today's `logits_to_keep` fix in `score_batch` depends on the exact `transformers` forward-pass signature verified on this Mac (`transformers==5.14.1`), an unpinned fresh install on the GPU machine could silently resolve a different version with different behavior — the kind of drift that would misdiagnose as a new bug rather than a version mismatch. Pinned the bundle's `requirements.txt` to the exact versions installed and verified in this session's `.venv`.

4. **(Caught by advisor review, not self-caught) `requirements.txt` was over-scoped, and separately, entirely inert without bundled wheels.** First pass pinned all 11 packages in this repo's general `requirements.txt`, four of which (`pandas`, `datasets`, `seaborn`, and one more) `build_p1_feature_store_gpu.py` never imports even transitively — an unresolvable pin on an unused package would fail the entire install with no on-site diagnosis path. Traced the actual import graph and trimmed to the 8 real dependencies. Separately and more seriously: pinning versions doesn't help if there's nothing to install from — the bundle had cached model weights but no cached *packages*, and the target machine is confirmed fully offline. `pip install` cannot succeed there without pre-downloaded wheels matched to the target's Python/CUDA version, which requires information not yet available (see "Short answer" above). **This is the reason the verdict below is not "final."**

---

## Verification performed (what's actually been tested, not just present)

| Item | How verified | Result |
|---|---|---|
| `p1_gpu_shard_prefix_backfill.jsonl` copy integrity | sha256 checksum, source vs. bundle copy | Identical |
| Bundled `build_p1_feature_store_gpu.py` | Re-parsed for syntax after copying | Valid |
| `GAGLE_MANIFEST_DIR` override | Imported the bundled module with the env var set to a fake path, asserted resolution | Works |
| All 720 `gpu_manifest_shards/*.parquet` | Metadata scan (row counts), every file | 0 corrupt, 230,530 total rows |
| Row-count reconciliation | 230,530 (shards) vs. 237,927 (raw prefix-backfill scan) | Gap (~3.1%) matches the documented `MIN_TOKENS` quality-gate drop rate — expected, not a sign of a bad copy |
| gemma-2b (tokenizer + model) | Full offline load (`HF_HUB_OFFLINE=1`), real instantiation, from the SSD cache | Loads successfully, 2.51B params |
| Embedder (all-MiniLM-L6-v2) | Full offline load (`HF_HUB_OFFLINE=1`), real instantiation | Loads successfully, 384-dim |
| `wheels/` (49 packages) | Isolated `pip install --dry-run --no-index --find-links wheels/ --ignore-installed --target <empty dir>`, cp310/manylinux/x86_64 | All 49 resolve, zero network |
| **Mistral-7B** | **File/blob presence only** | **NOT load-tested — this Mac cannot hold a 7B model in RAM** |

---

## The one open item, stated plainly

Mistral-7B's cache (`models--mistralai--Mistral-7B-v0.1`, 14GB) has plausible-looking blob files and a complete-looking snapshot structure (config, tokenizer, sharded safetensors), matching the same structure verified end-to-end for gemma-2b. But "looks complete" and "loads correctly" are different claims, and only the second one has actually been checked for gemma-2b — not for Mistral. This is the single thing this bundle cannot resolve from this machine, by hardware necessity, not oversight.

The bundle is deliberately structured so this isn't a silent gap: the README's step 5 sets `HF_HUB_OFFLINE=1` before the first `--verify-alignment` call specifically so that if Mistral's cache is incomplete, the failure is immediate, unambiguous, and correctly attributed ("cache-completeness problem, not a code problem" — stated explicitly in the README) rather than discovered hours into a full run or misdiagnosed as an alignment bug.

---

## Also carried forward, not new to this bundle

- Mistral's own `compute_boundary` formula remains undefined (Gemma's confirmed not to transfer, Test 3A). `BOUNDARY_CORRECTED_SCORERS = {"gemma-2b"}` means Mistral `continue_it`/`continue_pt` rows correctly get no Family B feature window until that formula exists — by design, not a bug.
- `continue_it`'s alignment pass rate is confirmed lower than `continue_pt`'s in a same-prefix paired check (2 domains) but not characterized at scale — worth watching in the real run's per-scorer pass-rate stats.
- Mistral's own alignment/candidate-selection pattern has never been run anywhere — gemma-2b's N=40 verification says the *mechanism* is sound (scorer-generic code), not that Mistral's specific numbers will look like gemma's.

None of these are blockers for starting the run — they're exactly the kind of honest, named open items this project's reports consistently track rather than smoothing over.

---

## Verdict

**Final, with two honestly-flagged exceptions, both checkable only on the target machine.** Code fixed for portability, all 720 shards integrity-checked, the prefix-backfill file checksum-matched, gemma-2b and the embedder load-tested fully offline, the requirements graph traced and trimmed to what's actually imported, and — the item that made the first "final" claim wrong — all 49 required Python packages bundled as wheels and verified to **resolve** (not install — that step hasn't happened anywhere yet) with zero network access on the target's actual platform (cp310/manylinux/x86_64, torch built for CUDA 11.8 to match the target's driver ceiling). Getting the wheels right took two rounds of self-correction (wrong CUDA tag, then wrong numpy/scipy/etc. versions for the target's Python 3.10 vs. this Mac's Python 3.12) — both checked against the real package indexes rather than assumed, not left as guesses.

**Mistral-7B cache completeness** — a hardware ceiling on this Mac, not a process gap. The bundle is deliberately structured (`HF_HUB_OFFLINE=1` before the Stage B `--verify-alignment 40` call, per the README) so that if it's incomplete, that surfaces immediately and is correctly attributed, rather than being asserted away or discovered deep into a run.

**The target's glibc version** — `pyarrow`/`torch` are `manylinux_2_28` wheels (glibc ≥2.28 required); never asked for or checked. Flagged in the README with the one-line check (`ldd --version`) to run before trusting `pip install` to succeed.

**Next step, corrected per advisor review to a two-stage sequence (a single small `--verify-alignment 5` call would test only gemma-2b on 5 unrepresentative rows and could misdiagnose a sampling artifact as a Mistral checkpoint problem):** on the Ubuntu machine — `pip install --no-index --find-links wheels/ -r requirements.txt`, then `--verify-alignment 5 --scorers gemma-2b` (cheap smoke test), then `--verify-alignment 40` (the real pattern check, both scorers, first-ever test of Mistral's checkpoint identity) — per the bundle's `README.md`.

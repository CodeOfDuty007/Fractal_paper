# What Remains After the Family B GPU Handoff

**Date:** 2026-08-12
**Scope:** Once the SSD bundle (`reports/2026-08-12_family_b_ssd_bundle_transfer.md`) is on the Ubuntu GPU machine and the run starts, what's left — both to close out Family B/P1, and the honest state of everything beyond it. Grounded directly against `Extension1_PhaseWise_Execution_Plan.md` and what's actually in `src/`, not a restated summary of the plan doc from memory.

---

## Immediate: finishing the run itself (on the GPU machine)

Already documented in the bundle's `README.md`, restated here for the full sequence:

1. `rsync` the SSD's `paper/` directory onto local disk (`/data/user/paper/` or similar) — or work directly off the mounted SSD if preferred, though local disk avoids repeated reads over the SSD's interface during the run and keeps checkpoint/output writes off removable media.
2. `pip install --no-index --find-links wheels/ -r requirements.txt` into a fresh venv.
3. `--verify-alignment 5 --scorers gemma-2b` (smoke test) → `--verify-alignment 40` (real pattern check, both scorers, first real test of Mistral's checkpoint identity and cache completeness).
4. Full run: `python3 src/build_p1_feature_store_gpu.py` — resumable, checkpointed per shard (720 shards, ~230K documents, both scorers). No time estimate exists yet; nothing in this session measured A100 throughput for this workload. Worth timing the first ~10 shards and extrapolating before assuming a total runtime.
5. Output: `data/p1_feature_store_gpu.jsonl` + `data/p1_gpu_manifest.json` on the GPU machine.

**Bring back to wherever the join step runs** (this Mac, unless you'd rather run the join on the GPU machine too — it's CPU-only and pandas-based, either works): `data/p1_feature_store_gpu.jsonl` and `data/p1_gpu_manifest.json`. Nothing else from the GPU machine is needed downstream.

---

## Closing out Phase P1 (Feature Store) — scripts already exist, not yet run

Per `Extension1_PhaseWise_Execution_Plan.md`'s own P1 exit gate ("Feature store complete for full GAGLE; no NaN/degenerate columns; per-doc aggregates reconcile with P0"), two more steps, **both already written in `src/`, neither yet run against real GPU output:**

1. **`python3 src/build_p1_join_feature_store.py`** — merges the CPU pass's output (Families A + C, `data/p1_feature_store_cpu.jsonl`) with the GPU pass's output (Family B + embedding dispersion, `data/p1_feature_store_gpu.jsonl`) into `feature_store/v1_gagle.parquet`. Enforces a **hard `set(cpu_ids) == set(gpu_ids)` integrity gate** — refuses to write anything if the two passes' id sets don't match exactly, rather than doing a silent partial join. Also computes cross-perplexity (Gemma vs. Mistral log-PPL ratio) here, since that's the first point where both scorers' rows for the same document are in scope together. This is the artifact every later phase (P2 onward) actually reads from.

2. **`python3 src/reconcile_p1_against_p0.py`** — the P1 spec's own exit-gate check: recomputes corpus-level `H`/`S` from the new feature store and diffs against `data/p0_corpus_level_SH.json` (P0's numbers, treated as ground truth). Two different reconciliation procedures for `H` vs. `S` (documented in the script's own docstring — per-doc `H` is trustworthy so it reconciles by simple averaging; per-doc `S` is known-biased so it must re-pool raw trajectories through `calc_S`'s list-mode, which means this script re-reads `gagle.zip` directly rather than the feature store). Only `gemma-2b` reconciles — P0 never scored Mistral, so there's no ground truth to check Mistral's Family A columns against.

**These can both run on this Mac** (or anywhere with `gagle.zip` and the CPU pass's output available) — neither is GPU-bound. `reconcile_p1_against_p0.py` specifically needs `gagle.zip`, which is why copying the full `paper/` directory (not just the GPU-relevant subset) to the GPU machine, or keeping a copy here, keeps this option open regardless of where you end up running it.

**If the join's integrity gate fails** (id-set mismatch): per the script's own design, this isn't something to route around — it means some document dropped differently between the CPU and GPU passes (e.g., a boundary case where the two tokenizers disagreed on `MIN_TOKENS`), and needs to be understood before treating the feature store as trustworthy, not silently inner-joined away.

---

## Beyond P1 — planned, not built

Per `CLAUDE.md`'s own explicit warning ("most of the pipeline stages they describe... are not yet in the repo") and confirmed directly against `src/`'s actual contents (no `splits.py`, no `eval/` directory, no `run_architecture_showdown.py`, no `models/` training code exist yet): everything from here forward is **design, not code**.

| Phase | Objective (from the plan doc) | Status |
|---|---|---|
| **P2 — Document-Level Fractal Validation Gate** | Split-half reliability of per-doc `S`/`H` (ρ ≥ 0.6 → trust point estimates; ρ < 0.6 → pivot to stability-profile features only). Produces `fractal_feature_whitelist.json`, gating everything downstream. | Not started. No script exists. This is the next real decision point — it determines which fractal columns P3+ is even allowed to use. |
| **P3 — Binary Detection (T1) + Baselines + Splits** | Freeze group-disjoint splits (in-distribution + LODO + LOGO), train the model ladder (logistic regression / LightGBM / MLP) on the ablation grid, run baselines (perplexity threshold, Fast-DetectGPT, Binoculars, supervised RoBERTa), report `All` vs. `All−fractal` with CIs. This is the headline result of the whole extension. | Not started. Needs `feature_store/v1_gagle.parquet` (from P1) and the P2 whitelist first. |
| **P4 — Edits, Robustness & T4** | Adversarial attack cascade (paraphrase, human-simulated revision, style transfer, etc.) on a GAGLE subset; GPU-heavy critical path. | Not started. |
| **P5 — Generator-Family (T2), Prompt-Type (T3) & RAID Transfer** | Multiclass generator/prompt-type ID; cross-corpus GAGLE→RAID transfer as the strongest external-validity check. | Not started. |
| **P6 — Release & Writeup** | Consolidate everything into the final release artifact. | Not started. |

**None of this is a surprise or a gap in this session's work** — it's the documented shape of the project. The point of flagging it explicitly here is so "what remains" isn't read as just "finish Family B" — Family B/P1 is maybe a fifth of the total planned work, and P2's fractal-reliability gate in particular is a real fork in the road (a ρ < 0.6 result would mean dropping the raw `Ŝ`/`Ĥ` features this entire session's Family A correction work was protecting the integrity of, in favor of stability-profile features only) that hasn't been touched yet.

---

## Practical next-decision point

Once the join + reconciliation both pass, the honest next question is **not** "start building P2's script" on autopilot — it's worth a deliberate checkpoint (plan review, or at minimum re-reading `Extension1_Technical_Plan.md`'s §2 on the three stabilization strategies P2 is supposed to compare) before committing to an implementation, since P2's outcome reshapes what P3 is even allowed to build on.

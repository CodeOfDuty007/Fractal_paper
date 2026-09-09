# Open Problems and Available Methods — 2026-08-19

Compact status board. Six independent open threads. None of them fix each
other — closing one does not move any of the others.

## Status board

| # | Problem | Blocking? | State | Method available |
|---|---|---|---|---|
| 1 | Mistral checkpoint identity unknown (Fault 1) | Blocks trusting mistral-7b scores at all | v0.1 tested, r≈0.51 (wrong-but-close). v0.2 doesn't exist as a base model — Instruct-only. Two-stage test written, not yet run on real server data. | `tokenizer_checkpoint_screen.py` (CPU, count-based screen) → survivors get `regress_candidate_cap.py`-style GPU scoring test (r vs GAGLE) |
| 2 | Mistral-*generated* docs' stored arrays too short (Fault 2) | Blocks Family A for those docs, both scorers | Proven: byte-identical text, array objectively shorter than tokenization, under both scorers. This is GAGLE's own defect. | No fix for Family A. Family B routes around it (Option B, below). No method exists yet for Family A on these docs — would require self-scoring, which changes the claim from reproduction to independent replication (undecided) |
| 3 | `MAX_CANDIDATE_TOKENS` reconstruction-parameter bug (Defect B) | Audit path only | v1 patch had a regression (lowered cap below 2048 for short docs). v2 fix given and applied on server. **Comparison result never confirmed clean in this chat.** | Re-run `regress_candidate_cap.py --compare regress_before.json --baseline regress_after.json`, paste the actual "NO REGRESSIONS" (or failure) output |
| 4 | Gemma human scores member-dependent, 12.4% deviant (Defect C) | Decision needed | No decision made. Diagnostic script queued, never written. | Write `human_variant_probe.py`, run it, then decide: exclude / investigate root cause / caveat in writeup |
| 5 | Human rows duplicated across splits, N inflated 2.34x (Defect D) | Hard blocker — invalidates every human-side CI/N so far | Root cause identified (dedup key wrongly included `split`). Fix not implemented. | One-line change: key by `dataset_id` only. Then recompute human-side CIs, specifically Test 4 and the wikipedia S cell (+0.0045) in `2026-08-06_headline_table_sensitivity_bound.md` |
| 6 | Option B (self-scored Family B, no GAGLE alignment gate) not yet built | Needed for full corpus coverage on Family B | Plan exists (6 steps: overlap validation → `--no-align-gate` code path → validate → run → join/reconcile → exit gate). Not started. | Build in a **copy** of `build_p1_feature_store_gpu.py`, not the live file — the gemma alignment-gated run is checkpointed and must not have its code change mid-run |

## Loose ends, not blocking but open

- Draft email to `ibomohsin@google.com` — written, not sent. Could be strengthened with the array-length proof (item 2) and today's checkpoint-screening results before sending.
- Whether a per-row S/H feature store file exists (`p1_feature_store_cpu.jsonl`) — asked, never confirmed. Needed before item 5's CI recompute can be scoped as a concrete script.
- Gemma-only full GPU run (~13h estimated) — last known status: started, in progress. No completion confirmation received since.

## Suggested order, if picking one thing next

Item 5 (duplication) has the cheapest fix and the widest blast radius — it invalidates numbers you may already be citing. Item 3 just needs you to paste output that may already exist. Item 1 has scripts sitting ready, waiting on tokenizer folders reaching the server. Items 2, 4, and 6 all require new work before they can move.

# Test 2 — Human-Side Realization Consistency Across Prompt-Type Files

**Date:** 2026-08-05
**Status:** PENDING — not yet run
**Parent decision doc:** `reports/2026-08-05_family_a_warmup_decision.md` (gap #2)
**Depends on:** `reports/2026-07-31_P1_feature_store.md` Finding 1 (already established, don't re-derive)

---

## Context

`reports/2026-08-03_prompt_type_reconstruction_probe.md` confirmed "Human side, text alone, capped at 2048 tokens" as the reconstruction rule, at **r=1.0000 on 6 records**. But `reports/2026-07-31_P1_feature_store.md` (Finding 1) already established something that six clean records could easily miss: the same byte-identical `original_text`, scored under two different `prompt_type` zip members, produces **measurably different** score arrays (confirmed case: `excerpt` member vs `continue_it` member, mean abs diff 2.97, Pearson r=0.035 — essentially uncorrelated). GAGLE's human-side scoring is conditioned on which member it came from, at least sometimes.

The human dedup key (`f"{split}_{domain}_{dataset_id}_human"`) doesn't scope by `prompt_type` — whichever member sorts alphabetically first wins, silently. The known win distribution (from Finding 1, 200-key sample): `continue_it` 54%, `cot`/`summary` 18% each, `summary_keywords`/`keywords` splitting the rest.

**The open question this test answers:** was the 6-record "r=1.0000" check run against a representative mix of these winning members, or did it happen to sample only from `continue_it`-sourced realizations (54% of the time, the most likely single source)? If the clean rule was validated on `continue_it`-sourced human text only, it's unconfirmed — not necessarily wrong, but unconfirmed — for the other ~46% of kept human documents sourced from `cot`, `summary`, `summary_keywords`, `keywords`, `excerpt`, `short_keywords`, or `continue_pt` members.

## Objective

Confirm "text alone, capped at 2048" holds for human-side score arrays **regardless of which prompt_type member they were scored under**, not just on average.

## Method

1. Pull a stratified sample of n=40 human documents **per prompt_type member type** (8 types × 40 = 320 total) — i.e., for each of the 8 `prompt_type` values, sample human documents whose GAGLE record for `original_text` came from a zip member of that type. This requires reading `original_text`'s score array from each of the 8 member types for the *same* underlying documents where possible (to isolate the member-type effect from a domain/document effect), similar to the direct single-record comparison already done in Finding 1.
2. For each sampled (document, member-type) pair: tokenize `original_text` alone, cap at 2048, and correlate against GAGLE's stored array for that member. Record Pearson r and mean abs diff.
3. Tabulate r by prompt_type-of-origin (8 rows), not pooled.
4. Cross-reference against the actual dedup key's win distribution (54% `continue_it`, 18%/18% `cot`/`summary`, rest split) to weight which member types matter most for what's actually **currently sitting in the P1 feature store** — a member type with poor r but only a 2% win share is a smaller problem than one with poor r and an 18% win share.

## Output format

| prompt_type (origin member) | n | mean r | median r | % r>0.98 | dedup win share (from Finding 1) | corpus-weighted risk |
|---|---|---|---|---|---|---|

"Corpus-weighted risk" = (1 − mean r) × win share — a simple proxy for how much of the *kept* human-side feature store is potentially affected, not just how bad the worst member type is in isolation.

## Pass/fail criteria

- **Pass:** all 8 member types show r≥0.98 for "text alone, capped 2048" — the rule is universal, Finding 1's earlier observed r=0.035 discrepancy was specific to that one `excerpt`-vs-`continue_it` comparison and not representative of the general human-side rule.
- **Fail:** any member type with meaningful dedup win share (>5%, conservatively) shows r<0.98 — the "solved" human-side rule needs a per-member-type correction, and the affected fraction of the ~11,696 human-labeled P1 rows needs to be identified and recomputed (cross-reference against `data/p1_manifest.json`'s Human label count).

## How this feeds the overall conclusion

This is independent of the `continue_it`/`continue_pt` Family A fix already in progress — it's a parallel question about whether the **baseline** (human side) that every LLM comparison is measured against is itself uniformly clean. If this fails, it doesn't just affect a subset of LLM documents — it affects the reference point every domain-level gap in `data/p0_corpus_level_SH.json` is computed against, which is a more fundamental risk than the `continue_it`/`continue_pt` fix and should be weighed accordingly once results come back.

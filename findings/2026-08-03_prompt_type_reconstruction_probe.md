# Prompt-Type Reconstruction Probe

**Date:** 2026-08-03
**Scope:** Why 4 of 8 GAGLE prompt_types (`excerpt`, `short_keywords`, `summary_keywords`, `cot`) do not reconstruct under either of the two `gemma-2b` scoring rules confirmed for the other 4 types + the human side, and how severe/general that gap is.

## Summary

The local alignment gate (`--verify-alignment`, `ALIGNMENT_PEARSON_MIN=0.98`) caught a real, previously-undocumented bug before any GPU manifest re-emission: `build_p1_feature_store_cpu.py`'s GPU-job manifest never captured `entry["prefix"]`, and GAGLE's own stored per-token `gemma-2b` scores are conditioned on that prefix for some prompt_types but not others, in a way that differs per prompt_type — not a single global rule.

Two rules are now confirmed, each independently reproducing GAGLE's stored scores at r≥0.98 (several at r=1.0000, exact) on small hand-checked samples:

- **Human side (`original_text`), all prompt_types:** scored **alone**, no prefix, capped at 2048 tokens (2047 scoreable positions). Confirmed on 6 records across different prompt_types, all r=1.0000.
- **`continue_it` / `continue_pt` (llm_text):** scored as `prefix_ids + text_ids[1:]` (prefix and text concatenated at the token level, avoiding a duplicate BOS), with a small length-matched trim (typically 0–2 tokens) taken from the end of the prefix when the naive concatenation is off by a token or two. 8/10 sampled docs cleared r>0.98 in the initial small-sample check; the 2 misses (r=0.9788, r=0.9560) were both still close, not catastrophic failures.

The other 4 prompt_types (`excerpt`, `short_keywords`, `summary_keywords`, `cot`) do **not** fit either rule. `keywords`/`summary` were initially believed solved by "text-alone, capped at 2048" based on a 3-record sample (r=1.0000 on all 3), but a much larger sample (n=500/type, this probe) shows that rule only holds for a **minority** of documents in those two types as well (see §1) — the earlier "solved" framing was based on a sample too small to see the real distribution.

**The model-dependent offset sweep (n=60/type, all 4 failing types) has completed** — much faster than the 3-4 hour estimate (37-90s/forward-pass was an artifact of a memory-thrashing incident during the first attempt at this script, not steady-state; once fixed, throughput settled to 2-13s/doc). Results split the 4 failing types into two clearly distinct groups:

- **`cot` is a genuine, mostly-recoverable rule, just noisier than the other confirmed rules.** Median best-r=0.805, 63.3% of documents clear r>0.70, and — the decisive number — the best-fitting offset correlates with the length-implied `text_alone_delta` at **r=0.996** (near-perfect). This means `cot`'s "offset" isn't an independent free parameter being fit per document; it *is* the length delta, to within noise. Combined with the marker-detection result (§5: 78% of `cot` documents contain numbered-list-style reasoning markers), the picture is coherent: GAGLE's `cot` scores most likely correspond to `llm_text` with a variable-length reasoning preamble excluded, and the preamble's length is exactly what drives both the token-count delta and the correct alignment offset. Still noisy — median r=0.805, not the 0.98+ seen for the fully-confirmed rules — and 36.7% of documents don't clear even r>0.70, so this is not yet a shippable rule, but it is a real, mechanistically-explained pattern, not noise.
- **`excerpt`, `short_keywords`, `summary_keywords` do not reconstruct under text-alone at any offset, for the large majority of documents.** Pass rates at r>0.98: excerpt 1.7% (1/60), short_keywords 0.0% (0/60), summary_keywords 5.0% (3/60). Mean best-r sits at 0.27-0.37 for all three — well below even a "partial match" bar — and, critically, the best-fitting offset does **not** track the length-implied delta (corr(offset, delta) = 0.16 for excerpt, 0.32 for short_keywords, 0.39 for summary_keywords, versus cot's 0.996). This rules out a length-driven trim (of the kind that works for `continue_it`/`continue_pt`/`cot`) as the explanation — whatever is wrong for these three types is not "the right tokens minus/plus a variable-length header," since the correlation doesn't track document length. The small non-zero pass rates (1-5%) are not a weaker version of the same rule; the histograms (§2 below) show these are concentrated at r≈0.2-0.4, with a handful of high-r outliers that look like a different population, not the tail of a single distribution.

**A follow-up hypothesis — that `excerpt`/`short_keywords`/`summary_keywords` differ from their working siblings `keywords`/`summary` only by how much leading title/header content the model prepends to its own output, and that stripping it would recover the fit — was tested directly (re-scored, not just re-measured for length) on 30 documents (10/type) and is empirically refuted.** Every one of 4 candidate strips (raw, strip-to-1st/2nd/3rd `\n\n`) was tried per document; stripping *never* improved correlation and typically made it sharply worse (mean r goes from ~0.22-0.30 at "raw" to as low as -1.0 after aggressive stripping, once the loss of real leading context outweighs whatever noise it was hypothesized to remove). "Raw" (no stripping at all) is the best of the 4 candidates for all three types, and even "raw" doesn't come close to 0.98 (mean r 0.22-0.30, 0% pass rate across all 30 documents at any candidate). This was a reasonable hypothesis given the shared prefix shape with `keywords`/`summary` (§6) — it just isn't the mechanism.

**Bundle status: still blocked.** `cot` needs a shippable trim rule derived from its preamble-length relationship (not yet built) before it can go in the manifest; `excerpt`/`short_keywords`/`summary_keywords` remain unexplained after two tested hypotheses (prefix+text, header-strip) — the working plan is to let these fail the alignment gate and ship with Family B null for the majority of these three types' documents, rather than continue searching for a third hypothesis. No SSD bundle assembly until code implementing this passes advisor review.

## Part A — Targeted divergence probe

### Sampling method and caveat

Stratified across (domain × model × temp), streaming each of the 720 zip members' `.jsonl` files line-by-line rather than loading each file fully into memory (`readlines()`) and shuffling — an earlier version of the sampler did the latter and, combined with the model's own memory footprint, pushed the process into macOS's "stuck" state (16GB used, near-zero CPU, on just 8 documents) during the first model-phase attempt. The line-by-line version takes the **first** k matching documents per stratum rather than a shuffled sample within each file — a real loss of within-file randomness, noted here rather than silently accepted. All checks below are run on n=500/prompt_type (cheap, tokenizer-only) or n=60/prompt_type (model-dependent, still running) sampled this way, stratified across 5 domains × 2 models × 3 temperatures (30 strata/type).

### 1. Length triangulation

For every sampled document: `gagle_len` = length of GAGLE's stored `gemma-2b` score array for `llm_text`; `text_alone_delta` = (tokenized `llm_text` alone, scoreable positions) − `gagle_len`; `combined_delta` = (tokenized `prefix_ids + text_ids[1:]`, scoreable positions) − `gagle_len`.

| prompt_type | n | text_alone_delta mean | median | std | min | max | % exactly 0 | % within ±3 |
|---|---|---|---|---|---|---|---|---|
| excerpt | 500 | 14.24 | 12.0 | 11.51 | 0 | 142 | 2.4% | 7.2% |
| short_keywords | 500 | 17.04 | 16.0 | 7.73 | 0 | 60 | 0.4% | 1.0% |
| summary_keywords | 500 | 14.57 | 13.0 | 8.44 | 0 | 63 | 4.4% | 5.8% |
| cot | 500 | 141.60 | 116.5 | 87.93 | 28 | 677 | 0.0% | 0.0% |
| continue_it | 500 | −85.69 | −86.0 | 13.27 | −127 | −21 | 0.0% | 0.0% |
| continue_pt | 500 | −77.27 | −88.0 | 88.06 | −127 | 935 | 0.0% | 0.0% |
| keywords | 500 | 9.33 | 6.0 | 10.42 | 0 | 62 | 46.8% | 49.4% |
| summary | 500 | 7.35 | 2.0 | 7.75 | 0 | 34 | 31.0% | 50.6% |

| prompt_type | n | combined_delta mean | median | std |
|---|---|---|---|---|
| excerpt | 500 | 930.16 | 928.5 | 149.22 |
| short_keywords | 500 | 96.29 | 92.0 | 28.63 |
| summary_keywords | 500 | 241.25 | 229.0 | 63.50 |
| cot | 500 | 263.09 | 238.0 | 90.69 |
| continue_it | 500 | 7.23 | 1.0 | 8.72 |
| continue_pt | 500 | 15.35 | 1.0 | 88.50 |
| keywords | 500 | 135.38 | 122.0 | 59.32 |
| summary | 500 | 136.20 | 133.0 | 28.20 |

**Reading this table:**

- `continue_it`/`continue_pt`: `text_alone_delta` is large and consistently *negative* (text-alone undercounts by ~77-86 tokens on average, matching prefix length), while `combined_delta` sits near 0 (median 1.0 for both). This confirms, at n=500 rather than the earlier n=5-10, that prefix+text is the right family for these two types, and the residual (mean 7-15, likely inflated by a long positive tail rather than a typical case, given the median is 1) is a small per-document trim, not a different rule.
- `excerpt`: `combined_delta` (mean 930) is enormous and clearly wrong — driven by `excerpt`'s prefix being unusually long (mean 917 tokens, see §6) — while `text_alone_delta` (mean 14, median 12) is an order of magnitude closer. Text-alone is the right family for `excerpt`; something small (~12-14 tokens) is still unaccounted for.
- `short_keywords`/`summary_keywords`: same pattern as `excerpt` — `combined_delta` is wrong by 90-240 tokens on average, `text_alone_delta` is small (13-17) and *tightly clustered* (std 7.7-8.4, versus e.g. `excerpt`'s wider std=11.5 or `cot`'s std=87.9). Tight clustering around a nonzero mean is the signature of a fairly consistent small correction (a header/artifact of roughly constant token length) rather than per-document noise.
- `cot`: worst-behaved of the four. `text_alone_delta` mean=141.6 but std=87.9, min 28, max 677 — no tight clustering, no exact-zero hits at all. Whatever text-alone is missing for `cot`, it is not a small fixed-length artifact; it scales with something document-specific (plausibly the length of the reasoning trace itself — see §5).
- `keywords`/`summary`: **this is the correction to the earlier "solved" claim.** At n=3-6 both looked like clean `text-alone, r=1.0000` matches. At n=500, only 46.8%/31.0% of documents hit an *exact* length match (delta=0), with a long tail extending to delta=62/34. These two types are **majority-correct** under text-alone but not **universally** correct — the earlier small samples happened to land in the aligned majority. The non-exact documents likely need the same kind of small-offset search as `short_keywords`/`summary_keywords`.

### 2. Offset sweep

For each of 60 sampled documents per failing prompt_type, one gemma-2b forward pass over `llm_text` alone (capped at 2048 tokens) is computed, then a window of length `gagle_len` is slid across the resulting log-prob array at every offset within ±100 of that document's own `text_alone_delta` (wide enough to cover the largest observed delta, 677 for `cot`, when centered per-document rather than globally), recording Pearson r and mean-abs-diff at each offset.

| prompt_type | n | best_r mean | median | std | min | max | %r>0.98 | %r>0.90 | %r>0.70 | corr(best_offset, text_alone_delta) |
|---|---|---|---|---|---|---|---|---|---|---|
| excerpt | 60 | 0.366 | 0.321 | 0.154 | 0.130 | 0.996 | 1.7% | 1.7% | 3.3% | 0.161 |
| short_keywords | 60 | 0.278 | 0.250 | 0.107 | 0.139 | 0.737 | 0.0% | 0.0% | 1.7% | 0.320 |
| summary_keywords | 60 | 0.335 | 0.280 | 0.199 | 0.148 | 1.000 | 5.0% | 6.7% | 6.7% | 0.393 |
| cot | 60 | 0.639 | 0.805 | 0.298 | 0.142 | 0.988 | 1.7% | 16.7% | 63.3% | **0.996** |

`best_r` histogram (bin edges 0/.2/.4/.6/.8/.9/.98/1.0):

| prompt_type | 0-.2 | .2-.4 | .4-.6 | .6-.8 | .8-.9 | .9-.98 | .98-1 |
|---|---|---|---|---|---|---|---|
| excerpt | 4 | 37 | 15 | 3 | 0 | 0 | 1 |
| short_keywords | 13 | 41 | 5 | 1 | 0 | 0 | 0 |
| summary_keywords | 12 | 35 | 8 | 1 | 0 | 1 | 3 |
| cot | 7 | 14 | 0 | 9 | 20 | 9 | 1 |

`cot`'s histogram is visibly bimodal-shifted-right compared to the other three: a real mass sits at 0.8-0.98 (29/60, 48%), a shape entirely absent from `excerpt`/`short_keywords`/`summary_keywords`, whose mass concentrates almost entirely at 0.2-0.4. Also note `best_offset`'s mean/median for `cot` (128.8/126.5, std 59.2) is an order of magnitude larger than for the other three (~4-5, std 4-6) — consistent with `cot`'s offset being driven by a genuinely long, variable reasoning preamble rather than a short fixed-length artifact.

**Interpretation:** `cot`'s near-perfect `corr(best_offset, text_alone_delta)=0.996` means its "correct alignment" is fully explained by the document's own length delta — there is no separate, independently-varying misalignment left over once length is accounted for. That is the signature of a real, mechanistic rule (skip a variable-length preamble, then score the rest), even though the resulting fit (median r=0.805) is noisier than the fully-confirmed rules' r≥0.98. The other three types show weak-to-moderate offset/delta correlation (0.16-0.39) *and* uniformly poor fits (mean r 0.27-0.37) — the offset search is not finding a real alignment for the bulk of these documents; it's finding whatever local optimum exists in a fundamentally mismatched comparison.

### 3. Substring/derivation check (excerpt)

Tested whether `llm_text` is a verbatim substring of `original_text` (or vice versa) for `excerpt` — i.e., whether the model was asked to *quote* rather than *generate*, which would mean GAGLE's score array reflects token positions within `original_text`'s own tokenization rather than a fresh 0-indexed scoring of `llm_text`.

**Result: ruled out, cleanly, at n=500.** `llm_text.strip() in original_text`: 0/500 (0.0%). `original_text.strip() in llm_text`: 0/500 (0.0%). Neither direction ever holds. `excerpt` is genuinely freshly-generated text, not an extracted quotation — the large prefix (mean 917 tokens, §6) is real context given to the model, not source material later reproduced verbatim in the output. This does not explain the ~12-14 token text-alone residual; whatever is missing lives inside the *generated* text, not by referencing `original_text`.

### 4. Length-gate false-negative check

Checked whether documents with small `gagle_len` (near `MIN_TOKENS=320`, this pipeline's own usability floor from `build_p0_local.py`) disproportionately drive the failing-r results, i.e. whether the earlier low correlations were a small-sample-size correlation-estimator artifact rather than a genuine reconstruction mismatch.

| prompt_type | % of docs with gagle_len < 370 (MIN_TOKENS+50) |
|---|---|
| excerpt | 19.0% |
| short_keywords | 1.4% |
| summary_keywords | 4.0% |
| cot | 5.4% |
| continue_it | 0.6% |
| continue_pt | 12.2% |
| keywords | 1.2% |
| summary | 6.2% |

`excerpt` has the highest near-threshold rate (19.0%) of any type, which is worth keeping in mind when interpreting any low-n subgroup results, but the other three failing types (`short_keywords` 1.4%, `summary_keywords` 4.0%, `cot` 5.4%) are not meaningfully different from the working types (`continue_it` 0.6%, `keywords` 1.2%) — **this rules out "near-threshold short documents" as the general explanation** for `short_keywords`/`summary_keywords`/`cot`'s failure to reconstruct; whatever's wrong for those three is not a sample-size artifact. `excerpt`'s elevated rate deserves a note in the eventual per-document r distribution (part of §2/§7, pending).

### 5. Structural-segment check (cot)

Checked for explicit reasoning-step delimiters in `cot`'s `llm_text` (numbered lists, "Step N", "in conclusion", "therefore", "let's think") to test whether GAGLE's stored score array corresponds to a sub-span (e.g., only a final-answer segment) rather than the full generated text.

| marker | docs matching (of 500) |
|---|---|
| `^\s*\d+\.\s` (numbered list item) | 372 (74.4%) |
| `\bin conclusion\b` | 38 (7.6%) |
| `\bfirst[,:]` | 24 (4.8%) |
| `\btherefore\b` | 13 (2.6%) |
| `\bstep\s*\d+\b` | 1 (0.2%) |
| **any of the above** | **390 (78.0%)** |

78% of sampled `cot` documents contain at least one structural marker, overwhelmingly numbered-list formatting (74.4%) rather than explicit "Step N" language (0.2%). This is consistent with — but does not by itself confirm — the hypothesis that GAGLE's `cot` scores correspond to a specific sub-segment of the generated text (e.g., excluding a numbered reasoning preamble) rather than the whole thing; `cot`'s `text_alone_delta` (mean 141.6, highly variable, std 87.9) is the kind of pattern a variable-length preamble would produce, but this needs the offset-sweep's per-document r's (§2/§7) to confirm rather than infer from marker presence alone.

### 6a. Header-strip hypothesis, tested with re-scoring (not just re-measurement)

A follow-up hypothesis after §6 below: `excerpt`/`short_keywords`/`summary_keywords`'s small text-alone deltas (§1) might be a leading title/header line the model prepends to its own output (many sampled `short_keywords`/`summary_keywords` texts visibly start with a bolded Markdown title, e.g. `"**Squararylium Dyes: A Revolutionary Advance...**\n\n"`) that GAGLE's scoring excludes. An earlier attempt at this (Part B, step 12) only *measured length* after a regex strip; this test actually re-scores.

For 10 documents per type, 4 candidates were tried: raw `llm_text`, and `llm_text` with everything up to and including the 1st/2nd/3rd `"\n\n"` stripped. Each candidate was tokenized, scored (one gemma-2b forward pass), and compared against GAGLE's stored array at whatever offset within the resulting length delta gave the best fit.

| prompt_type | candidate | n | mean best_r | pass_rate (r>0.98) |
|---|---|---|---|---|
| excerpt | raw | 10 | 0.272 | 0.0% |
| excerpt | strip 1st `\n\n` | 10 | 0.250 | 0.0% |
| excerpt | strip 2nd `\n\n` | 8* | −0.114 | 0.0% |
| excerpt | strip 3rd `\n\n` | 7* | −0.666 | 0.0% |
| short_keywords | raw | 10 | 0.217 | 0.0% |
| short_keywords | strip 1st `\n\n` | 10 | −0.328 | 0.0% |
| short_keywords | strip 2nd `\n\n` | 10 | −0.885 | 0.0% |
| short_keywords | strip 3rd `\n\n` | 5* | −1.000 | 0.0% |
| summary_keywords | raw | 10 | 0.295 | 0.0% |
| summary_keywords | strip 1st `\n\n` | 10 | −0.026 | 0.0% |
| summary_keywords | strip 2nd `\n\n` | 8* | −0.707 | 0.0% |
| summary_keywords | strip 3rd `\n\n` | 5* | −1.000 | 0.0% |

*n<10 where a document didn't contain that many `\n\n` occurrences to strip to.

**Result: refuted, cleanly.** Stripping never helps and typically hurts sharply (mean r goes negative — worse than chance — once 2+ `\n\n` groups are stripped, meaning real, load-bearing context is being thrown away, not noise). "Raw" (no stripping) is the best of the 4 candidates in all three types, and even raw's best-case mean r (0.22-0.30) is far from the 0.98 gate. This rules out "leading title/header" as the mechanism, despite the superficially plausible visual pattern (bolded titles at the start of many sampled documents) and despite `keywords`/`summary` sharing a similar prefix profile (§6) — whatever separates the working types from these three is not explained by trimming leading document structure.

### 6. Prefix structure diff

| prompt_type | mean prefix_len (tokens) | median | std |
|---|---|---|---|
| excerpt | 916.9 | 918.5 | 147.1 |
| short_keywords | 80.2 | 75.0 | 27.2 |
| summary_keywords | 227.7 | 218.0 | 62.7 |
| cot | 122.5 | 121.0 | 9.9 |
| continue_it | 93.9 | 93.0 | 9.5 |
| continue_pt | 93.6 | 93.0 | 9.1 |
| keywords | 127.0 | 114.0 | 58.6 |
| summary | 129.9 | 125.5 | 27.0 |

`excerpt`'s prefix is qualitatively different from every other type — nearly 10x longer than `continue_it`/`continue_pt`'s (~94 tokens), and with a comparably large standard deviation (147 tokens), meaning it is not just "long" but "long and highly variable in length," consistent with it embedding a substantial, document-specific chunk of context (plausibly a partial excerpt or a per-document generation instruction whose length varies with source-document properties) rather than a fixed boilerplate template. `cot`'s prefix, by contrast, is short (122 tokens) and unusually *tight* (std=9.9, the tightest of any type) — a near-fixed-length boilerplate, similar in shape to `continue_it`/`continue_pt`'s prefixes (std ~9.5) even though `cot` doesn't follow their prefix+text rule. `keywords` and `summary` have similar mean prefix lengths to `summary_keywords` (114-130 vs 218) but notably `keywords`/`summary` have much higher prefix-length variance (58.6/27.0) than `continue_it`/`continue_pt`'s tight ~9.5 — suggesting prefix structure alone doesn't cleanly separate "which rule applies"; it's a necessary clue, not a sufficient one.

### 7. Uniformity check

Answered by §2's per-document distribution:

- **`cot` is uniform in the relevant sense — not "one fixed offset for everyone" (std=59.2 on best_offset rules that out), but "one relationship" (offset is a near-deterministic function of length delta, r=0.996). This is the same category of rule as `continue_it`/`continue_pt` (a length-driven trim), just keyed off internal document structure rather than an external prefix field.**
- **`excerpt`, `short_keywords`, `summary_keywords` are uniform in a different, worse sense: uniformly poorly fit.** These are not bimodal populations of "some documents align well under text-alone, others don't" (which would show two separate clusters in the r histogram) — the histograms (§2) show one concentrated low mass (0.2-0.4) with a small number of scattered outliers (1-3 documents near r=0.98-1.0 per type) that look like coincidental fits or genuinely different documents (e.g., unusually short/simple ones), not evidence of a rule that works for a definable subpopulation. This is closer to "the text-alone hypothesis is wrong for these three types" than "the text-alone hypothesis is right but needs per-document tuning."

---

## Part B — Status of the existing investigation (methodology, not results)

### What was tried before this probe, and what each attempt produced

All of the following happened ad hoc this session, via one-off scripts in `/tmp` (not committed, not the shipped pipeline), driven by a chain of hypotheses each informed by the previous result:

1. **Standalone Gemma-only test harness** (`/private/tmp/.../scratchpad/local_gemma_only_test.py`, the one persistent throwaway script from this session): imported the real `compute_score_window`/`score_batch`/`family_b_summary` functions from the shipped `build_p1_feature_store_gpu.py` and ran them against real manifest-shard data, scoring `llm_text` alone with no prefix. Result: `best mean_pearson_r=0.0798` against the pipeline's own `ALIGNMENT_PEARSON_MIN=0.98` gate — catastrophic, and the trigger for this entire investigation.
2. **prefix+text vs text-alone, first single-record test:** tokenized both candidates for one `continue_it` record. `text alone`: 631 tokens (vs GAGLE's 735 — too short). `prefix+text`: 738 tokens (735+2, much closer). Correlation on the first 200 positions: text-alone r=0.10, prefix+text (text-portion slice) r=0.15 — both still bad, but the closer token count was the first real clue.
3. **Raw-value comparison** (not just correlation) on the same record, prefix+text: my recomputed log-probs matched GAGLE's stored values to 2-3 significant figures position-by-position (e.g. mine `-5.851, -8.061, -2.987...` vs GAGLE's `-5.854, -8.041, -2.978...`) — decisive confirmation that prefix+text is directionally correct for this type, with a remaining +2 token-count puzzle.
4. **Separator-choice test:** no-separator / space / newline concatenation all tokenized identically (738 tokens) — ruled out separator choice as the length-discrepancy source.
5. **Full-array correlation, aligned from start vs end:** first-735-of-737 gave r=0.557; last-735-of-737 gave r=0.642 — better from the end, suggesting the +2 discrepancy concentrates near the prefix/text seam rather than being a uniform shift.
6. **Boundary-crossover diagnostic:** per-position abs-diff between shift=0 and shift=2 around the prefix/text token boundary (~index 106-107) showed a clean crossover — diffs near-zero (shift=0) before the boundary, near-zero (shift=2) after. A piecewise reconstruction (shift=0 before, shift=2 after, breakpoint=106) gave r=0.9998 — this is what established prefix+text as the correct rule family for `continue_it`, with a small seam-adjustment needed.
7. **Reconstruction-method test** (tokenize prefix and text separately then concatenate IDs, vs. tokenize the joined string as one span): produced *identical* token sequences in this case — ruled out "tokenization method" as the seam-effect's mechanism.
8. **rstrip()-before-concat hypothesis:** tested stripping trailing whitespace from `prefix` before concatenation. Only closed part of the gap (one record's delta went from 10→5 tokens, others unchanged) — falsified as the general seam-fix mechanism; a length-driven trim search (rather than guessing the exact string GAGLE used) was adopted instead, per advisor guidance.
9. **Precision test (fp32 vs bf16 vs fp16):** identical results across all three (r=0.5568/0.5556/0.5569) — ruled out precision/checkpoint mismatch as an explanation for any of the residual misalignment seen at any point in this investigation.
10. **`original_text` (human side) length-triangulation:** discovered `gagle_orig_len` is *exactly* 2047 across every one of 50 sampled records regardless of true document length (up to 10,700+ raw tokens) — a hard truncation cap, not a construction difference. Confirmed as "text alone, capped at 2048" via 6 records at r=1.0000 (§ below).
11. **8-prompt_type census + per-type reconstruction test (n=5/type):** found `continue_it`/`continue_pt` fit prefix+text (r 0.96-1.0, 80% pass rate at n=5), `keywords`/`summary` fit text-alone (r=1.0000 at n=3 each — since revised by this probe's n=500 result, see §1), and `cot`/`excerpt`/`short_keywords`/`summary_keywords` fit neither (large length deltas under prefix+text).
12. **Header-strip heuristic test (excerpt):** tried regex-stripping a leading `"<N> words\n\n"`-style header from `llm_text` before tokenizing. Reduced the delta in 1 of 3 tested records (10→5), left the other two unchanged (20→19, 8→8) — falsified as a general rule.
13. **This probe** (Part A above): scaled the investigation to n=500 (cheap checks) / n=60 (model checks, in progress), corrected the "keywords/summary fully solved" claim from step 11 (majority-correct, not universal), ruled out the excerpt-substring hypothesis cleanly, and characterized `cot`'s reasoning-marker structure.

### Why no earlier script exists for this specific investigation

Steps 1-12 above were done as a chain of small, disposable Python scripts run directly via the Bash tool (heredocs or files under `/tmp`), each one testing a single hypothesis raised by the previous result, then discarded — a fast-iteration exploratory style appropriate for narrowing down an unknown mechanism one clue at a time, but not written with reproducibility or a fixed sample size in mind. Only one script from that chain (`local_gemma_only_test.py`, step 1) was kept, because it directly imports and exercises the real shipped pipeline code rather than reimplementing scoring logic ad hoc. This probe (`src/investigate_prompt_type_alignment.py`) is the first script in this specific investigation written for reuse: fixed sample sizes, stratified sampling, JSON output per prompt_type, and a documented rationale for every parameter choice (sample size, offset-sweep width, precision/device).

### Where the reconstruction-rule logic for the working types currently lives

Not yet in the shipped pipeline. The two confirmed rules (prefix+text for `continue_it`/`continue_pt`, text-alone-capped-2048 for the human side) exist only as the ad hoc `/tmp` scripts described in steps 1-11 above and as this probe's `cheap_checks`/`stratified_sample` functions (`src/investigate_prompt_type_alignment.py`) — neither `build_p1_feature_store_cpu.py` nor `build_p1_feature_store_gpu.py` has been modified yet. That is deliberately sequenced after this probe, per the user's explicit request and the standing "test with sample too to verify" / "run all code thru advisor" instructions: implementation work should follow full characterization of all 8 prompt_types' behavior, not precede it.

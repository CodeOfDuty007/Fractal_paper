# Defects C and D, settled — 2026-08-19

Runs of TODO #3 and TODO #4 from `diagnostics_2026-08-18/DIAGNOSTICS_2026-08-18.md` §9.
Both were "blocking — do before starting any full run." Both are now answered, and
both answers are actionable.

| | question | answer |
|---|---|---|
| **Defect C** (TODO #3) | are the 1,450 deviant human rows salvageable? | **The rows are not — but the articles all are.** Both deviant classes fail alignment at the cap valid for their own array length (not a cap artifact). All 1,450 nonetheless have a byte-identical clean twin, so **0 of 5,000 articles need excluding.** |
| **Defect D** (TODO #4) | does `(domain, dataset_id)` name the same article across splits? | **Yes, in 4,474/4,474 cases, byte-identical.** The comment justifying `split` in the human dedup key is empirically false. `domain` must stay. |

**The headline is that these two defects cancel.** The cross-split duplication that
makes Defect D a leakage problem is exactly what makes Defect C fixable without
dropping anything: dedup on `(domain, dataset_id)` with an explicit preference for
the `continue_it`-sourced triple, and the human corpus becomes 5,000 articles, every
one carrying a gemma array from the class verified at r = 1.0000. One change, both
defects closed, no rows lost to an exclusion list.

Everything below was run on this Mac (MPS, gemma-2b) and on `gagle.zip` directly.
Raw logs and scripts: `diagnostics_2026-08-18/`.

---

## 0. What was actually run

```
# Defect C — three human variant classes side by side, at BOTH caps
cd /Volumes/NikPhotoSSD/paper/diagnostics_2026-08-18
$VENV/bin/python human_variant_probe.py --n-per-cell 10 --cap 2048
$VENV/bin/python human_variant_probe.py --n-per-cell 10 --cap 4096   # = the script default

# which cap is a valid test for which cell (no model, seconds)
$VENV/bin/python cell_cap_validity_census.py

# Defect D — the canonical census, run to completion
cd /path/to/fractal_analysis_paper/diagnostics_2026-08-18
python3 defect_d_cross_split_text_census.py --zip /Volumes/NikPhotoSSD/paper/gagle.zip

# Defect D — independent cross-check, different code path
python3 defect_d_within_split_crosscheck.py --zip /Volumes/NikPhotoSSD/paper/gagle.zip

# C x D — how many ARTICLES actually have to be dropped (the result that matters)
$VENV/bin/python deviant_residual_after_dedup.py
```

`$VENV` = `/path/to/fractal_analysis_paper/.venv` (py3.12.13, torch 2.13.0,
transformers 5.14.1). **The bare `python3` on this Mac is now Homebrew 3.14 and has
no numpy** — the first attempt at the probe died on `ModuleNotFoundError: numpy`.
Anything importing the pipeline must use the venv interpreter explicitly.

Two notes on the commands as originally specified:

- `human_variant_probe.py` lives in `diagnostics_2026-08-18/`, not in
  `family_b_gpu_bundle/code/src/` (it *imports* the pipeline from there).
- The bare `--n-per-cell 10` uses the script's **default `--cap 4096`**, which §4.5
  of the 2026-08-18 report already showed destroys alignment on the 2047-length
  gemma arrays. Run alone it returns 0/10 in every cell and settles nothing. Both
  caps were run, and §1.1 below establishes which one is the valid test per cell.

---

## 1. Defect C — the deviant rows aren't salvageable, but no article is lost

### 1.1 First: which cap is a valid test for which cell

A row is dropped **pre-forward-pass** ("no usable candidate") when the cap truncates
below its stored `gagle_gemma_total_tokens`, and alignment breaks when the cap runs
far past it. So a 0/10 cell means nothing until you know the cell was evaluable at
all. Read straight off the manifest shards, human rows only, no model:

| cell | n | min | median | max | ≤2048 | 2049–4096 |
|---|---|---|---|---|---|---|
| C1 human from `continue_it` member (majority) | 5,000 | 971 | **2047** | 2047 | 5,000 | 0 |
| C2 human from `summary_keywords` member (deviant) | 933 | 1791 | **2047** | 2047 | 933 | 0 |
| C3 human from `mistral_7b_*` member (deviant) | 517 | 1821 | **2658** | 3071 | 23 | 494 |
| — human from all other gemma members | 5,246 | 1182 | 2047 | 2047 | 5,246 | 0 |

This reproduces §4.1's row counts exactly: majority = 5,000 + 5,246 = **10,246**;
deviant = 933 + 517 = **1,450**; total **11,696**. The classification is sound.

**So the valid test condition differs by cell**, and no single flat cap serves all three:

- **C1 and C2 → cap 2048.** Their arrays are 2047. Cap 4096 is invalid for them.
- **C3 → cap 4096.** 494/517 exceed 2048, so cap 2048 drops them before any forward pass.

### 1.2 Results, each cell read at its own valid cap

n = 10 per cell. Gate is the pipeline's own: `r > 0.98` and `mean_abs_diff < 0.10`.

| cell | cap | valid? | passed | no-usable-cand. | median r | median MAD | offsets | verdict |
|---|---|---|---|---|---|---|---|---|
| **C1 majority** | **2048** | ✅ | **10/10** | 0 | **1.0000** | 0.0091 | 65 | **PASS** |
| C1 majority | 4096 | ❌ cap artifact | 0/10 | 0 | 0.1512 | 2.1277 | 1003–2260 | (known §3 artifact) |
| **C2 `summary_keywords`** | **2048** | ✅ | **0/10** | **0** | **0.5292** | 1.1532 | 67, 68, 69 | **FAILS ON CORRELATION** |
| C2 `summary_keywords` | 4096 | ❌ cap artifact | 0/10 | 0 | 0.1593 | 2.0479 | 506–2255 | (cap artifact) |
| C3 `mistral_7b`-sourced | 2048 | ❌ not evaluable | 0/10 | **10** | — | — | — | (all dropped pre-forward-pass) |
| **C3 `mistral_7b`-sourced** | **4096** | ✅ | **0/10** | **0** | **0.1664** | 2.3597 | 66–1527 | **FAILS ON CORRELATION** |

**C1 reproduces §4.5 exactly** — 10/10 at r = 1.0000, `text_alone` winning at offset
65 (= `WARMUP_DROP + 1`). This is the control that licenses reading the other two
cells at face value, and it holds.

**C2 is the decisive new result.** At cap 2048 — *its own valid condition* — all 10
rows were scored (`no-usable-candidate 0`, so nothing was dropped) and all 10 failed
on correlation, median r = 0.5292, best r = 0.7449, nowhere near the 0.98 gate.
Median mean-abs-diff 1.1532 against a 0.10 ceiling. This is not a truncation
artifact and not a cap-tuning problem: **the stored gemma array for these 933 rows
is a different array than gemma-2b produces for that text.** They are unsalvageable.

The offsets are themselves informative: C2 aligns best at 67–69 rather than the
majority's clean 65. A consistent 2–4 token shift alongside a wrecked correlation is
the signature of a *different generation setup*, not of noise.

**C3 fails too, and for the same reason.** At cap 4096 — its own valid condition —
all 10 rows made it to a forward pass (`no-usable-candidate 0`, versus 10/10 dropped
at cap 2048) and all 10 failed, median r = 0.1664. Lifting the cap makes these rows
*evaluable*; it does not make them *align*.

**This is a re-run, not an independent replication — and only C2 is new.** Both
probes take the first N human rows from the first matching shard in sorted order, so
this C3 draw is a strict subset of the 2026-08-18 `human_probe_cap4096.log` mistral
cell: its offsets `[66, 71, 531, 1199, 1368, 1373, 1382, 1503, 1527]` are that run's
list minus two rows, and the shared max r = 0.6654 is the *same row*, not agreement
across draws. Same for C1 @ 4096. So C1 and C3 confirm the earlier run on the same
rows under the same code, which is worth having but is not new evidence. **C2 is the
genuinely new cell** — the 2026-08-18 probe never tested `summary_keywords`-sourced
human rows at all, and C2 at cap 2048 is the result this report adds.

One nuance worth recording: two of the ten found offsets at 66 and 71, near the
expected 65, while the rest scattered to 531–1527. The census shows 23 of the 517
C3 rows have arrays ≤ 2048. It is possible a small sub-population of C3 is ordinary
and the class is not perfectly homogeneous. This does not change the recommendation
— none of the 10 passed the gate — but if 517 rows are ever worth recovering, that
sub-population is where to look.

### 1.3 Answer to TODO #3 — and Defect D pays for Defect C entirely

**The 1,450 deviant rows are not salvageable as scored. But not one article has to
be dropped, because every deviant row has a clean twin.**

The alignment verdict first. Both deviant classes were tested at the cap their own
array length makes valid, both had **zero** rows dropped pre-forward-pass, and both
failed the 0.98 gate by a wide margin (median r 0.53 and 0.17). The majority variant,
run in the same session under the same code, passed 10/10 at r = 1.0000. The contrast
is the result: those stored arrays are not what gemma-2b produces.

The obvious remedy — "exclude 1,450 rows by id" — would have been **wrong**, and
Defect D is why. A row's variant class is a property of *which member the shard build
happened to harvest it from*, not of the article. The build keys human dedup on
`{split}_{domain}_{dataset_id}`, so one article yields up to three triples from three
different members, and §2 proves those triples are byte-identical. Merging them is
therefore safe, and the merge gets to **choose which stored gemma array survives**.

So the question is not how many rows are deviant, but how many *articles* have no
clean copy at all. `deviant_residual_after_dedup.py` counts it:

```
human triples (split, domain, dataset_id) : 11,696
distinct articles (domain, dataset_id)    :  5,000

  C2_summary_keywords    933 triples ->  933 have a majority twin,  0 do not
  C3_mistral_sourced     517 triples ->  517 have a majority twin,  0 do not

  articles with NO majority-variant triple : 0 / 5,000  (0.00%)
  surviving human N after dedup+exclusion  : 5,000 articles
```

**Zero.** All 1,450 deviant triples — 1,450 of 1,450 — belong to articles that also
appear, byte-identically, in a majority-variant triple. This is consistent with
§4.1's pair table, where `continue_it × summary_keywords` alone accounts for 932 of
the 933 C2 rows.

**Consequence: apply the Defect D dedup with majority-variant preference and Defect C
costs nothing.** The human corpus becomes 5,000 distinct articles, every one of them
carrying a gemma array from the class verified at r = 1.0000. There is no exclusion
list, no id blacklist, and no limitations paragraph about dropped human rows. The
two defects cancel: the duplication that made Defect D a leakage problem is exactly
what makes Defect C recoverable.

Stated plainly, because it inverts the expected outcome: the human side is not
merely *mostly* trustworthy at 87.6 %. After the dedup that has to happen anyway,
it is trustworthy at **100 %**, on a corpus of 5,000 rather than 11,696.

### 1.3.1 Two closures, and one thing that turned out not to be free

**The "verified at r = 1.0000" claim, made exact.** The probe forward-passed only
the `continue_it`-sourced class (C1, 5,000 triples); the "majority" class as defined
in §1.1 also includes `cot`/`keywords`/`summary`/`excerpt`/`continue_pt`/
`short_keywords`-sourced rows (ZZ, 5,246 triples), which were never forward-passed.
`cell_class_residual_full.py` isolates `C1_continue_it` as its own class (not merged
into "majority") and re-runs the residual test against it specifically:

```
RESIDUAL B: articles with NO C1_continue_it triple specifically
  articles lacking a continue_it-sourced (r=1.0000-verified) triple: 0 / 5,000 (0.00%)
```

**Every one of the 5,000 articles has a `continue_it`-sourced triple available, and
it is unique.** Counting `continue_it` triples per article directly
(`{1: 5000}` — every article has exactly one, none has zero or more than one)
confirms the merge can specifically prefer `continue_it` with no ambiguity about
which triple that means, and the r = 1.0000 claim covers 100% of survivors exactly
— since C1 is available for every article, the merge never needs to fall back to
the untested ZZ class at all, so ZZ's alignment status stays irrelevant to the
result rather than assumed. The chosen row's mistral array comes along with it
unaffected — §4 established the mistral column is consistent across every member,
so selecting on gemma provenance doesn't touch it.

**What is not free: relying on member-processing order instead of an explicit rule
— and this applies to both dedup sites.** `build_p0_local.py`'s dedup
(`process_member`, ~line 210) and `build_p1_feature_store_cpu.py`'s
(`process_member`, ~line 476–491, replayed on resume at 515–536) are each a plain
first-wins set check over `sorted(all_members)`, where `all_members` sorts by full
path `gagle3/{split}/{model}_{temp}_{domain}_{prompt_type}.jsonl`. Because
`continue_it` sorts alphabetically first among prompt types and `gemma_2b` before
`mistral_7b`, it's tempting to assume dropping `split` from the id would reach a
`continue_it` member first for every domain, for free. **It doesn't.**
`simulate_split_free_dedup.py` replays that exact sort order and first-wins logic
(against `build_p0_local.py`'s loop; `build_p1_feature_store_cpu.py`'s is
line-for-line the same pattern, so the result transfers) with only `split` removed
from the id string, no other change:

```
surviving human articles: 5,000
  ZZ_other_gemma_member                    2,326  (46.52%)
  C1_continue_it (verified r=1.0000)       1,997  (39.94%)
  C2_summary_keywords                        420  ( 8.40%)
  C3_mistral_sourced                         257  ( 5.14%)
which split's copy survived: {'test': 3932, 'train': 1068}
```

13.54% of survivors would still be deviant-sourced. **Not a `MIN_TOKENS` artifact**
— checked directly: `MIN_TOKENS=320`, `CLIP_TO=464`, and every stored human array
(majority 2047, deviant 1791–3071 per §1.1) is ≥ `CLIP_TO`, so `slice_scores`
always takes the `len(raw) >= CLIP_TO` branch and always yields exactly 400 tokens
≥ 320. The gate cannot fire on human rows, ever. The real mechanism is mundane:
**each member holds only a partial subset of its split-domain's articles** —
checked directly on `test/big_patent`, whose `continue_it` member contains 200 of
that split-domain's 820 distinct articles, not all of them. A domain's
`continue_it` member simply doesn't reach every article before some other,
later-sorted member does.

So §3's recommendation is not "delete `split` from the id" in either file — it is
"delete `split` from the id, in both `build_p0_local.py` and
`build_p1_feature_store_cpu.py`, **and** add an explicit preference for the
(measured unique, §1.3.1) `continue_it`-sourced triple." All three parts are
required; the preference rule is the one that actually eliminates the 1,450-row
problem, and skipping the second file would leave `gpu_manifest_shards` — the data
this whole report is built on — unfixed while `build_p0_local.py`'s output looked
correct.

### 1.4 What this says about TODO #2 (the per-row cap)

The two-cap protocol used here is a manual stand-in for the per-row cap. Note what
the cap-validity census proves: **no flat `MAX_CANDIDATE_TOKENS` can test all three
cells in one pass**, because C1/C2 need 2048 and C3 needs > 3071. That is the
argument for TODO #2, now backed by a length census rather than by inference.

`RUNBOOK_family_b_gpu.txt` already carries the patch, in the correct
*raise-only* form:

```python
cap = MAX_CANDIDATE_TOKENS
if gagle_total:
    cap = max(cap, min(MAX_CANDIDATE_TOKENS_HARD, int(gagle_total) + CAP_MARGIN))
```

Raise-only matters, and the results here show why: lowering the cap toward
`gagle_total` for the 2047-length rows would truncate rows that currently pass at
r = 1.0000. **The local `src/build_p1_feature_store_gpu.py` on the SSD is still
unpatched** (line 171, flat `MAX_CANDIDATE_TOKENS = 2048`, no `CAP_MARGIN` or
`MAX_CANDIDATE_TOKENS_HARD`). The runbook's patch targets the server copy.

One caution for when it is applied: the per-row cap changes behaviour **only** for
rows with `gagle_total > 2047` — which, on the human side, is exactly the 494 C3 rows
this report just showed do not align anyway. So the per-row cap does **not** rescue
the human side. Its value is on the mistral *scorer*, whose arrays are 3071
throughout. Run `regress_candidate_cap.py` before/after regardless.

---

## 2. Defect D — confirmed at source, and the code comment is false

`defect_d_cross_split_text_census.py` ran to completion over all 720 members of
`gagle.zip`. `defect_d_within_split_crosscheck.py` — written independently, using a
different hash (md5 vs sha256) and a different data structure — was run over the
same zip as a check. **They agree on every shared number.**

```
human-side records scanned                : 226,239
distinct (domain, dataset_id) keys        : 5,000
splits represented                        : ['test', 'train', 'val']

(domain, dataset_id) keys appearing in >1 split : 4,474 / 5,000

  IDENTICAL text across splits (real duplication)  : 4,474
  DIFFERENT text across splits (numbering collision): 0
```

**VERDICT (the script's own): CONFIRMED.** Every cross-split collision in the full
corpus is genuine text duplication. Zero numbering collisions.

This is the full-corpus confirmation TODO #4 asked for, and it reproduces §5's
shard-level finding exactly rather than merely being consistent with it:

| | §5 (shard-level) | this run (zip-level) | agree? |
|---|---|---|---|
| distinct human keys `(split, domain, dataset_id)` | 11,696 | **11,696** | ✅ |
| distinct texts / `(domain, dataset_id)` | 5,000 | **5,000** | ✅ |
| appearing in 1 / 2 / 3 splits | 526 / 2,252 / 2,222 | **526 / 2,252 / 2,222** | ✅ |
| multi-split groups | 4,474 | **4,474** | ✅ |
| ...byte-identical across splits | 4,474 | **4,474 (100.00%)** | ✅ |
| human N inflation | 2.34× | **2.3392×** | ✅ |

So the comment in `build_p0_local.py` / `build_p1_feature_store_cpu.py`:

> GAGLE's dataset_id numbering resets per split, so the SAME dataset_id string can
> refer to a DIFFERENT underlying article in train vs val vs test.

is **empirically false for this corpus** — 0 of 4,474 opportunities. The `split`
component of the human dedup key buys nothing and costs a 2.34× inflation of the
human N plus train/test leakage into P3.

### 2.1 But `domain` must stay in the key — the docstring's other warning is real

The census also asked whether `dataset_id` strings collide across domains, which is
the *other* half of the comment's concern and is a genuinely different question:

```
dataset_id strings used in >1 domain           : 131
  ...where the text is the SAME across domains :   0
  ...where the text DIFFERS across domains     : 131   (100%)
```

e.g. `00002-of-00256__21` names one article in `big_patent` and a different one in
`wikipedia`. All 131 are real collisions. This is consistent with §5's "0 duplicated
texts span more than one domain" — §5 counted *texts*, this counts *id strings* — and
it confirms the concern noted in the census script's own docstring (recorded there as
239/11,457 keys; that figure is shard-key-level, this one is zip id-string-level, so
the units differ — both say the same thing qualitatively).

**Conclusion: change the human dedup key from `{split}_{domain}_{dataset_id}_human`
to `{domain}_{dataset_id}_human`. Drop `split`. Keep `domain`.**

### 2.2 One thing the canonical script cannot see, and why it does not matter here

`defect_d_cross_split_text_census.py` stores `by_key[key][split] = (hash, snippet)`,
overwriting per split. If the same `(domain, dataset_id, split)` cell ever held two
*different* texts, only the last one scanned would be compared, and the
IDENTICAL/DIFFERENT partition would be computed on an arbitrary representative.

The cross-check tracks a **set** of hashes per split and reports that count directly:

```
(domain, dataset_id, split) cells holding >1 distinct text: 0
```

Zero. So the two scripts are provably equivalent on this input and the CONFIRMED
verdict stands without qualification. Worth keeping the cross-check for any re-run on
different data, where that assumption would need re-testing rather than assuming.

### 2.3 A loose end this surfaced, not on the critical path

The zip holds **226,239** records with non-empty `original_text`. The shard build
yields 218,834 LLM rows + 11,696 human rows. 226,239 − 218,834 = **7,405 zip records
whose LLM side did not become a shard row** (~3.3%). That is plausibly the documented
`MIN_TOKENS` drop (§6 notes a "~3 % MIN_TOKENS drop" on the prefix backfill), but it
has not been verified to be the same 7,405. Cheap to check; not blocking; noted so it
is not rediscovered later.

---

## 3. What to change, in order

Defects C and D turn out to have **one** fix between them, not two.

1. **Re-key human dedup on `{domain}_{dataset_id}_human` — drop `split`, keep
   `domain` — and make the merge EXPLICITLY prefer the majority variant. Two sites
   need this, not one.** `build_p0_local.py` and `build_p1_feature_store_cpu.py`
   each carry their own independent implementation of the identical first-wins
   dedup over `sorted(all_members)`, both keyed on the same
   `{split}_{domain}_{dataset_id}_human` string (`build_p1_feature_store_cpu.py`
   ~line 388, 476–491, 515–536). The second one is the more consequential of the
   two — it is what actually wrote `gpu_manifest_shards/*.parquet`, the exact data
   this report's probes and censuses read.

   Merely deleting `split` from the id string in either file is **not** enough.
   Simulating that exact change with no other logic added
   (`simulate_split_free_dedup.py`, using `build_p0_local.py`'s loop — identical
   sort order and dedup logic in `build_p1_feature_store_cpu.py`, so the result
   transfers directly) shows **13.5% of the 5,000 surviving articles would still be
   claimed by a deviant member** — 8.40% `summary_keywords`, 5.14%
   `mistral_7b`-sourced. The reason is mundane, not the `MIN_TOKENS` gate: each
   member holds only a **partial subset** of its split-domain's articles — the
   `test/big_patent` `continue_it` member has 200 of that split-domain's 820
   articles — so a domain's `continue_it` member frequently doesn't reach an
   article before some other, later-sorted member does. (`MIN_TOKENS=320` never
   actually gates human rows at all: every stored array, majority or deviant, is
   ≥ `CLIP_TO=464`, so `slice_scores` always yields exactly 400 tokens ≥ 320 —
   verified against both `build_p0_local.py`'s constants and the length census of
   §1.1.) **The preference has to be explicit, in both files**: when several
   triples collapse into one article, keep the gemma array from whichever triple's
   source member is `*_continue_it` (not merely "non-deviant" — §1.3.1 shows this
   is stronger and still always available). Counted directly, not inferred: every
   one of the 5,000 articles has **exactly one** `continue_it`-sourced triple
   (`{1: 5000}`, no article has zero or more than one) — the preferred source is
   unique, and the merge is deterministic, not merely "exists."

   Then recompute human N (11,696 → 5,000), Family A's per-domain normalization, and
   every bootstrap CI computed over human rows.

2. **Assert the invariant in code, don't trust this report.** After the merge, assert
   that no surviving human row's source member is `*_summary_keywords` or
   `mistral_7b_*`. It holds today at 0/5,000; an assertion is what keeps it holding
   if GAGLE is ever re-extracted or the shard set changes.

3. **Update the now-false code comment** in `build_p0_local.py` /
   `build_p1_feature_store_cpu.py` and cite this report. Keep the *domain* half of
   its concern — it is real (131 cases, §2.1) and is the reason `domain` stays in
   the key.

4. **Then TODO #2** (per-row cap), with `regress_candidate_cap.py` before/after. It
   is now well-motivated but, per §1.4, it does not change the human-side conclusion.

**No exclusion list is needed, and no human-side limitations paragraph.** That was
the expected outcome going in; §1.3 is the reason it is not the actual one.

Neither §1 nor §2 blocks starting `--scorers gemma-2b` — as the 2026-08-18 report
concluded, both change *which rows end up with features*, not whether the run should
start. Both do block P3, and #2 blocks any human-side confidence interval.

---

## 4. Status of the two TODOs

- [x] **TODO #3 — Are the 1,450 deviant human rows salvageable?** The *rows* are not
  — both classes fail at their own valid cap with zero pre-forward-pass drops. But
  the *articles* are: all 1,450 have a byte-identical majority-variant twin, so the
  Defect D dedup recovers 100 % of them and **0 of 5,000 articles need excluding**.
- [x] **TODO #4 — Settle the split-duplication question.** **Confirmed at full-corpus
  scale**, 4,474/4,474 byte-identical, 0 numbering collisions, by two independent
  scripts. Drop `split`; keep `domain`.

### Caveats, stated plainly

- **n = 10 per cell, and all three cells come from one corner of the corpus.**
  `human_variant_probe.py` takes the first 10 human rows from the first matching
  shard in sorted order, and sorted order starts at `test / *_0.0 / big_patent`. So
  every cell here is **one split × one domain × one temperature**. This cuts both
  ways: the C1-vs-C2 contrast is unusually well controlled — same split, domain and
  temperature, differing only in source member — but *no* cell is established across
  domains. n = 10 is adequate only because the separation is r = 1.0000 versus
  r ≈ 0.17–0.53, unmissable rather than marginal; it would not support a claim about
  the *rate* of failure within a class, and §1.2's C3 offset nuance hints C3 may not
  be homogeneous.
- **§1.3's zero-residual result does not depend on any of that.** It is a census over
  all 11,696 human triples and all 5,000 articles — exact, not sampled. The probe
  establishes *that* the deviant classes fail; the census establishes *that it does
  not matter*. Only the first of those two rests on n = 10.
- Defect C was probed with **gemma-2b only**. The mistral scorer is still blocked on
  Fault 1 (wrong checkpoint), untouched by this report.
- Defect D's census is **complete**, not sampled: all 720 members, all 226,239
  records. No caveat needed there.

---

## 5. Files

In `diagnostics_2026-08-18/` (mirrored to the paper repo, so this report is readable
without the SSD mounted):

| file | what it is |
|---|---|
| `human_variant_probe.py` | Defect C probe — three human variant classes, pre-existing |
| `human_variant_probe.log` | **new** — both cap runs, all six cells |
| `cell_cap_validity_census.py` + output | **new** — `gagle_gemma_total_tokens` per cell; establishes which cap tests what |
| `defect_d_cross_split_text_census.py` | the canonical Defect D census, pre-existing |
| `defect_d_census_official.log` | **new** — its full run to completion |
| `defect_d_within_split_crosscheck.py` | **new** — independent second implementation; adds the per-split hash-set check of §2.2 |
| `deviant_residual_after_dedup.py` + log | **new** — joins Defect C's classes onto Defect D's article groups; the 0/5,000 result of §1.3 |
| `defect_d_census.log` | **new** — the cross-check's output |
| `simulate_split_free_dedup.py` | **new** — replays `build_p0_local.py`'s actual dedup with `split` dropped from the id, no other change; the 13.5%-still-deviant result of §1.3.1 |
| `cell_class_residual_full.py` + log | **new** — `deviant_residual_after_dedup.py` with `C1_continue_it` isolated from ZZ; the 0/5,000 exactness result of §1.3.1 |
| `c1_triple_uniqueness_check.py` + log | **new** — counts `continue_it` triples per article directly; the `{1: 5000}` uniqueness result of §1.3.1 |


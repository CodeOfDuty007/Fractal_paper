# Verification and notes --- ICLR 2027 package

Everything below was re-run against the files in this folder, not carried forward
from an earlier draft.

## 1. Page-limit compliance

ICLR 2027 sets a **strict 9-page limit on the main text** at submission (10 at
rebuttal/camera-ready), with references, the AI-use / ethics / reproducibility
statements, and unlimited appendices all outside the limit.

- The paper's line numbers run 1--54 per page. The Conclusion's last line is **485**,
  the last line of page 9. The AI use statement begins at line **486**, the first
  line of page 10.
- **Main text = 9 pages exactly.** Total document = 22 pages (9 body, 1 statements,
  2 references, 10 appendices).

## 2. Formatting compliance

| Requirement | Status |
|---|---|
| Official ICLR 2027 style files, unmodified | yes --- `iclr2027_conference.sty/.bst` verbatim from the ICLR Master-Template |
| No change to margins, text block, or font sizes | yes --- no `geometry`, no font-size overrides |
| `\iclrfinalcopy` commented out | yes (line 76) |
| Double-blind: no author names, affiliations, acknowledgements | yes --- `\author{}`; style file prints "Anonymous authors" |
| Pages numbered | yes (style default) |
| US Letter, pdflatex | yes |
| AI use statement (**required**) | present, `\subsection*{AI use statement}` |
| Ethics statement (recommended) | present |
| Reproducibility statement (recommended) | present |
| Undefined references / citations | **0** |
| Overfull boxes | **0** |
| Bibliography entries cited | 28 / 28 |
| Bibliography entries undefined | 0 |

## 3. Data provenance

- **33 provenance comments** in `ICLR2027_ScaleWithoutSupport.tex`, each naming the artifact a number was
  read from (`% source: ...`). None was added speculatively.
- **Numeric parity with `main2.tex` (the pre-conversion master): every one of the 419
  distinct numeric values in `main2.tex` appears in this document.** Missing: 0.
  Where a number was cut from the body for length it was relocated to an appendix, not
  dropped --- specifically the harness-control and leak-injection series, the
  Holm--Bonferroni p-values, the MLP collapse, the per-column importance detail, the
  per-fold substitution numbers, and the non-lineage attacker range.
- Nothing was estimated, interpolated or inferred. Claims the project's artifacts
  cannot support are marked in the text rather than silently omitted --- see the
  operating-threshold audit (Appendix F) and the marked pre-attack row in Table 3.

## 4. What moved from `main2.tex` to this version

`main2.tex` had a 19-page body. The 9-page limit forced a restructure, not a deletion:

- **Body keeps** the argument spine and every headline number: the estimand/support
  formulation, the towns/thermometer analogy, both halves of the dissociation
  (Tables 1--2), P1--P4 with the ablation table, the perturbation partition with the
  Class I result (Table 3), the cross-corpus transfer (Table 4), and the condition.
- **Appendices carry** everything else: the full 20-row reliability table, the eight-fold
  grid and all diagnostic arms, the harness controls, the feature inventory, the full
  five-detector battery and arm decomposition, the operating-threshold audit, the
  attacker ladder, the auxiliary multiclass results, the full related work, what prior
  work says about its own estimator, and the extended limitations.
- **Nothing was cut without a home.** The numeric parity check above is the evidence.

## 5. Difference audit: `main1.tex` -> `main2.tex`

Asked separately: how much data was lost when `main2` added its analogies?

- Distinct numeric values: `main1` **416**, `main2` **419**.
- Total numeric occurrences: `main1` **991**, `main2` **1061** (+70).
- Values present in `main1` but absent from `main2`: **0**.
- Values present in `main2` but absent from `main1`: **3** --- 0.157, 1.65, 15 ---
  all from the operating-threshold audit added as Appendix B of `main2`.
- Tables: 12 in both. Figures: 1 in both. Equations: 2 in both. Captions: 13 in both.
  The `\label` inventory is byte-identical.

**Conclusion: `main2` removed no data.** It is a strict superset of `main1`. The
analogies were added on top of the existing evidence, and two paragraphs were
relocated to appendices rather than deleted. The +70 occurrences are the
threshold-provenance findings and the reworded passages that repeat a number in a
second place.

## 6. Known open items, carried forward deliberately

1. **The operating-threshold file** (`Appendix F`) still has no writer in the
   repository and reproduces under no convention tested. The paper marks the one row
   that depends on it and shows that every other number in that table reproduces
   bitwise. If you re-run the P3/P4 thresholds on the cluster, the pre-attack row is
   the number to replace.
2. **The leak-injection control** numbers (0%: 0.4166; 2%: 0.5080; 5%: 0.5534;
   20%: 0.8019) come from the machine-generated `results/p3_tables.md` table dated
   2026-08-29, not from the stale prose sentence beneath it. Worth one confirmation
   run if the cluster is available.
3. **The AI use statement** describes this project's actual practice as we understood
   it. Read it before submitting and adjust if it does not match.
4. `math_commands.tex` ships with the template but is not `\input` by this paper; it
   is kept so the folder matches the official template contents.

## 7. Reproducing these checks

```
cd submission_iclr2027
latexmk -pdf ICLR2027_ScaleWithoutSupport.tex
grep -cE "LaTeX Warning: (Reference|Citation).*undefined" ICLR2027_ScaleWithoutSupport.log   # expect 0
grep -c Overfull ICLR2027_ScaleWithoutSupport.log                                           # expect 0
pdftotext -layout ICLR2027_ScaleWithoutSupport.pdf - | grep -B1 "AI USE STATEMENT"          # expect line 486
```

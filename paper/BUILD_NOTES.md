# Scale Without Support --- ICLR 2027 submission package

Everything needed to compile and submit is in this folder. Nothing outside it is
required.

## Build

```
latexmk -pdf ICLR2027_ScaleWithoutSupport.tex
```

Produces `ICLR2027_ScaleWithoutSupport.pdf`. The build is self-contained: `iclr2027_conference.sty`,
`iclr2027_conference.bst`, `natbib.sty`, `fancyhdr.sty` and `math_commands.tex` are
the official ICLR 2027 style files (from
`https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`, mirrored at
`github.com/ICLR/Master-Template`, `iclr2027/`, commit of 1 Sep 2026). No system
packages beyond a standard TeX Live are needed; the figure is TikZ, compiled from
source, so there are no external image files.

## Contents

| File | Role |
|---|---|
| `ICLR2027_ScaleWithoutSupport.tex` | The paper. Body + statements + appendices, one file. |
| `references.bib` | 28 entries; all cited, all venues verified by hand. |
| `fig_dissociation.tex` | TikZ source for the one figure (`\input` by the paper). |
| `iclr2027_conference.sty` | Official ICLR 2027 style file --- do not modify. |
| `iclr2027_conference.bst` | Official ICLR 2027 bibliography style. |
| `natbib.sty`, `fancyhdr.sty` | Shipped with the ICLR template. |
| `math_commands.tex` | Shipped with the ICLR template (not `\input` by this paper). |
| `supplementary_figures/` | The project's standalone TikZ diagrams, for the supplementary upload. |
| `VERIFICATION_AND_NOTES.md` | What was checked, how, and what remains open. |
| `ICLR2027_ScaleWithoutSupport.pdf` | Compiled output. |

## Before you submit

1. **Do not uncomment `\iclrfinalcopy`.** It is on line 76 and must stay commented
   until the camera-ready.
2. `\author{}` is empty; the style file substitutes ``Anonymous authors / Paper
   under double-blind review''. Do not add names.
3. The AI use statement (required) is filled in with what this project actually did.
   **Read it and correct it if the description does not match your own practice** ---
   it is a declaration, not boilerplate.
4. Upload `supplementary_figures/` (and, if you wish, an anonymised code archive) as
   supplementary material.
5. Abstract deadline 18 Sep 2026, paper deadline 25 Sep 2026, both 11:59 PM AOE.
   No authors may be added after the abstract deadline.

## Compliance summary

- Main text: **9 pages exactly** (limit: 9 at submission, 10 at rebuild/camera-ready).
- References, the three statements and the appendices are outside the limit, as ICLR
  permits.
- Anonymous: no names, affiliations, acknowledgements, or non-anonymous links.
- AI use statement: present and required-section-compliant.
- Ethics statement and Reproducibility statement: present (both recommended).
- Zero undefined references, zero undefined citations, zero overfull boxes.
- 33 provenance comments in the source, each naming the artifact a number was
  read from.

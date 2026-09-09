# Plain-Language Briefing: What P1/P2 Found, and What's Blocking P3

**Date:** 2026-08-23
**Audience:** project owner, non-technical framing
**Covers:** the two findings from the X-02 and P2 runs, what's blocking P3, and the F-1 question

---

## Finding One: The average gap between human and AI text is real

**What it means.** Take a large pile of human-written articles and a large pile of AI-generated articles from the same domain (news, summaries, etc.). Measure a fractal "roughness" score (S) and a "memory" score (H) on each pile as a whole. The two piles score differently, and that difference is not random noise — it holds up even after two real bugs (a duplicate-document bug and a randomness bug in the math) were found and fixed. In every one of the five text domains tested, the human range and the AI range don't even overlap.

**Why this matters / how it affects us.** This is good news that costs us nothing extra. It means the original headline finding of this project — "human and AI writing differ in fractal structure" — is not an artifact of the two bugs that were just fixed. All the compute already spent reproducing that original finding stays valid; nothing needs to be rerun, and nothing needs to be walked back in the paper. Think of this as the foundation of the paper surviving an audit.

---

## Finding Two: You cannot trust this score on any single document

**What it means.** The average-across-thousands-of-documents result above is real. But when the test was narrowed to a single document — split one document in half, measure the score on each half separately, and check whether the two halves agree — they mostly don't. For the roughness score (S), the two halves of the same document agreed barely better than two totally unrelated documents would by chance. For the memory score (H), agreement was better but still fell just short of the bar set for "usable."

A simple way to picture this: imagine a thermometer that, on average across a thousand readings, correctly tells you Tuesdays are colder than Wednesdays — but hand it to someone and ask them to read one single Tuesday, and the number it gives is nearly a coin flip. That's the situation here. The group-level pattern is real; the individual-level reading is not trustworthy.

**Why this matters / how it affects us.** This is the one with real consequences for compute and plans:

- It means the plan to build detailed fractal features (looking at fractal structure at 64 different zoom levels) for the AI-detection model should not go forward as originally scoped — building those features would spend real GPU/CPU time on a signal that has already been shown, on this same data, to be too noisy to trust per document.
- It explains something that was previously just an observation without a clear cause: earlier work found that adding these deep fractal features to a detection model made it worse at generalizing to new domains, not better. Now there's a concrete reason why — the underlying measurement itself isn't reliable enough per document to have been a good input in the first place.
- It points the detection model back toward the simpler, already-validated approach (perplexity and semantic features), which does not depend on this unreliable per-document fractal score.
- Net effect on compute: this finding *saves* future compute, at the cost of one already-completed, relatively cheap test run.

---

## What's blocking P3, step by step

Nothing broken is blocking P3 — everything that was tested this round passed its checks. What's blocking P3 is three decisions that need to be made once, so the next phase is built on a settled foundation instead of a guess that might need to be undone later:

1. **Decide what to do with Finding Two's result.** Three options are on the table: report it plainly as "no fractal feature is reliable enough to use" (the straightforward, defensible choice), make a documented exception for the memory score (H) since it came very close to the bar, or re-run the test on longer documents to see if reliability improves. The one thing to avoid is quietly lowering the bar after already seeing the score fall just short — that would be shaping the rule to fit the result rather than the other way around.
2. **Decide how to handle a skew in the data.** Only about 2 in 100 documents in the test set are the kind the model needs to catch (the rest are the "easy" majority class). A simple accuracy-style score already showed weak performance under this skew, so before the model is built, a decision is needed on how to measure and train around this imbalance.
3. **Approve merging the finalized train/test split.** The official split rules were finished and locked, but they're sitting on a side branch, not yet merged into the main project. P3's code needs to build against the real, merged version.

## How serious is this?

Not serious in the sense of danger — nothing is broken, no data was lost, and every check that ran this round passed cleanly. It's serious only in the sense of being a genuine fork in the road: P3 needs an answer to each of the three questions above before it can proceed, because each one changes what gets built (which features, which evaluation metric, which exact set of documents). These are one-time decisions, not open bugs — once made, they're quick to act on.

## Why F-1 still matters, even though P1 is closed

F-1 is about a completely separate model — Mistral-7B — and one specific question: which exact checkpoint of that model did the original dataset's creators use when they pre-computed their scores? This project sidestepped that question a while ago by deciding to score all the text itself, with its own copies of the models, rather than trying to reproduce the dataset's original numbers. So F-1 has never been something P1 or P2 needed answered — the project's own scoring pipeline doesn't depend on matching someone else's exact model version.

It's being flagged again now for a narrower reason: the earlier note that closed this question turned out to be based on an invalid comparison (it compared two things that were measured differently, an apples-to-oranges mistake). Leaving an incorrect "settled" note in the project record risks someone trusting it later without re-checking — this project has already been burned by that exact pattern twice this month. So the record is being corrected, not because F-1 is standing in anyone's way, but because a wrong "closed" label is worse than an honest "still open, but not on the critical path."

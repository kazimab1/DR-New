# 5-minute presentation

`verify_dr_5min.pptx` — 9 slides, ~5 minutes. Speaker notes with timings are embedded
in each slide (View → Notes Page in PowerPoint).

## Run sheet

| # | Slide | Time | The one thing to land |
|---|---|---|---|
| 1 | VERIFY-DR | 0:00–0:20 | This is not about grading more accurately. It's about knowing when to trust the grade. |
| 2 | What is DR? | 0:20–1:00 | Painless until advanced — that's why screening exists. Grade 2 is the referral threshold. |
| 3 | Screening doesn't scale | 1:00–1:40 | The bottleneck is grader time, mostly spent on normal images. |
| 4 | Where AI falls short | 1:40–2:25 | You can't ask a model to audit itself with the same numbers that produced the error. |
| 5 | The idea | 2:25–3:15 | Two pathways, different supervision, no shared weights. They can genuinely disagree. |
| 6 | The case it gets right | 3:15–3:55 | **The persuasive slide.** Confident "normal" contradicted by found lesions. |
| 7 | Phases | 3:55–4:30 | Phase 4 is the freeze. Externals touched once. |
| 8 | Experiments | 4:30–4:50 | 26 experiments, 3 decide the thesis: F2, E2, C4. |
| 9 | Contributions | 4:50–5:00 | Pre-registered, so a negative result is still a result. |

## If you only get three minutes

Cut slides 3 and 8, and compress 7 to one sentence. Slides 4 → 5 → 6 are the argument;
everything else is scaffolding.

## Likely questions

**"Isn't this just an ensemble?"** No. An ensemble averages models to be more accurate.
This keeps two models deliberately separate and uses their *conflict* as a signal. The
pathways are trained on different label types — image grades versus pixel masks — so
they fail differently, which is the entire point.

**"What if the evidence pathway is just worse?"** It is worse, and it has to be —
experiment C4 measures exactly that. It needs to be informative but weaker. If it
matched the grader, the grader would be redundant; if it were noise, disagreement
would mean nothing.

**"Your accuracy won't beat the leaderboard."** Correct, and it's in the abstract.
A single EfficientNet-B0 with no ensemble won't. The contribution is the referral
decision, not the grade. Experiment H1 reports on the official split so the comparison
is at least honest.

**"How do you know there's no leakage?"** Patient-grouped splits with a hard assertion,
externals locked and evaluated once, and a pre-registration commit hash recorded before
the unblinding.

## Rebuilding

Source: `build_pptx.js` (in the session scratchpad — copy into `scripts/` if you want
it version-controlled). Requires `pptxgenjs`. Editing directly in PowerPoint is fine;
the file is not generated as part of any build.

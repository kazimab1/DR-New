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

---

## Final figures · 2026-09-30

`figures/` holds eight 16:9 figures, each 3200 × 1800 px. `make_figures.py` draws them from
numbers copied out of the committed readouts. Every number names the readout line it came
from, and the script checks the transcription against the published means before drawing.
Rebuild with:

    python docs/presentation/make_figures.py

The file order is a suggested talk order.

| # | File | What it shows | The point to make |
|---|---|---|---|
| 1 | `01_method.png` | The two pathways and the two gates | M1 and M2 never share weights, so their disagreement could carry information the grader's own confidence cannot |
| 2 | `02_timeline.png` | Freeze to unblinding, 16–29 September | Both analyses were fixed before a test label was read, and the order is on the record |
| 3 | `03_headline_auc.png` | Coverage–accuracy AUC: three rules, three test sets | The headline: confidence beat disagreement everywhere, in both analyses |
| 4 | `04_errors_deferred.png` | Share of the grader's errors each rule defers | The pre-registered disagreement deferred the grader's safest calls, catching fewer errors than chance |
| 5 | `05_referral_errors.png` | The same, for referral errors | The one place disagreement wins: false referrals under shift. It was found after the unblinding |
| 6 | `06_why_disagreement_lost.png` | Correct calls M3 disputes; errors above grade 2 | Two causes: M3 is not specific, and it cannot see grades 3 and 4 |
| 7 | `07_faithfulness.png` | Lesion removal against 19 random controls | Verification works: the grader does use the lesions M2 finds |
| 8 | `08_examples.png` | Figure 7.1, laid out for a slide | The intended case exists, but its evidence is often not the disease |

Colour means the same thing on every slide:
- **blue** is confidence;
- **orange** is disagreement and the evidence pathway;
- **grey** is a reference (no gate, or chance).

A hollow orange dot is the pre-registered analysis, and a filled one is the amended
analysis. The two colours pass colour-blind separation and contrast checks on white.

### What in the 5-minute deck predates the results

The deck and the run sheet above were written at the scaffold, before any result. These
parts no longer hold:
- **Slide 5.** The evidence pathway trained on DDR only; IDRiD was held out (D2). REACQUIRE
  was never built (D10).
- **Slide 6.** Its example is illustrative: "p = 0.91", "14 microaneurysms, 9
  haemorrhages" and "3 quadrants" were not measured, and quadrants never ran (D1). The real
  case is figure 8, which is rare (24 of 17,615) and often artefactual.
- **Slide 8.** The three deciding experiments now have outcomes:
  - F2 not supported (figure 3);
  - E2 holds (figure 7);
  - C4 informative but weaker (QWK 0.375 against 0.679).
- **Slide 9.** Of the three contributions, the calibration claim failed: EM worsened
  external ECE in 23 of 24 cells (H2). The contribution is now a pre-registered negative
  result with its mechanism.
- **Likely questions, the leaderboard answer.** No split is comparable with the
  leaderboard (D8), so the benchmark experiment H1 on the official split did not run.

---

## The new 5-minute deck · 2026-09-30

This deck replaces `verify_dr_5min.pptx` for talks: https://claude.ai/artifact/EHeVPgAsfEXz6Gkf67g1Jj.

It has nine slides and four hidden backup slides. Speaker notes with timings are on
every slide.

| # | Slide | Time | The one thing to land |
|---|---|---|---|
| 1 | Cover | 0:00–0:20 | The question is which gradings to trust, and the test was allowed to fail |
| 2 | The problem | 0:20–0:55 | Confidence is the usual safeguard, and a network can be confidently wrong |
| 3 | The idea (figure 1) | 0:55–1:35 | Two pathways that never share a weight |
| 4 | The case | 1:35–2:05 | A real image: the confident grader is wrong, and the evidence is right |
| 5 | The test (figure 2) | 2:05–2:30 | Fixed before any test label was read |
| 6 | The result (figure 3) | 2:30–3:15 | Confidence beat disagreement on every test set |
| 7 | Why (figure 6) | 3:15–3:55 | M3 is not specific, and it is blind above grade 2 |
| 8 | What survives | 3:55–4:30 | Verification works; false referrals under shift, found after the unblinding |
| 9 | Close | 4:30–5:00 | A pre-registered negative result with its mechanism |

The backup slides, hidden in the run-through, are figures 4, 5, 7 and 8.

# Chapter 6 — Results

> **Draft, 2026-09-29.** Built from `docs/04_experiment_register.md` and two verbatim
> readouts: the unblinding (`docs/unblinding/2026-09-28_readout.txt`), and the secondary
> outcomes that Phase 8 transcribed from `results.json` (`docs/phase8/2026-09-29_readout.txt`,
> section 3). Every outcome here was registered and computed at the one label join;
> the exploratory analyses are in Chapter 7. Target length about 3,500 words.

**How to read this chapter.** The locked test sets were evaluated once, under an
analysis plan fixed before any of their labels was read. Two analyses come from that
single label join:

- **The registered analysis** is the primary result. It decides every hypothesis.
- **The amended analysis** adds two dated deviations, D12 and D13. It is reported beside
  the registered one and never replaces it.

Each training variant was run with three seeds. A figure given as "mean (min–max)" is
over those seeds. A hypothesis counts as supported only if all three seeds show a
positive effect, every paired 95% bootstrap interval excludes zero, and the mean effect
exceeds the spread between seeds (`ANALYSIS_PLAN.md` §8). The in-domain test set is this
project's own patient-grouped split of EyePACS (D8), and is never compared with the
Kaggle leaderboard.

## 6.1 The grading pathway (B1–B5)

Stage B chose the grader's recipe on the validation split only. Each option was run
with one seed, and every effect was read against a per-metric noise floor. The floors
are the spread B1's three resolutions produced: QWK 0.0276, macro-F1 0.0144, grade-1 F1
0.0190, grade-1 recall 0.0220, MAE 0.0230.

| Lever | Largest effect | × noise floor | Decision |
|---|---|---|---|
| Sampler: weighted vs natural (B4) | macro-F1 +0.087 | 6.0 | weighted (stratified exposure) |
| Head: focal on or off (B3) | grade-1 recall +0.123 | 5.6 | focal-ordinal, γ = 2 |
| Eye-pair fusion (B5) | grade-1 recall +0.046 | 2.1 | rejected: confounds the signal, 2.2× the compute |
| Encoder: B0 vs ResNet50 (B2) | grade-1 F1 +0.038 | 2.0 | EfficientNet-B0: tied QWK at 0.70× the cost |
| Resolution: 384/512/768 (B1) | grade-1 F1 spread 0.019 | 1.0 | 512 px, the cache's native size |

The recipe that survived every ablation is the one designed before any run. Two
findings from Stage B are reportable in their own right:

- **Focal weighting does the job it was included for.** Grade-1 recall rose 83%, at a
  QWK cost inside the noise floor.
- **Fusion was the best grader Stage B produced** (QWK 0.713 against 0.679). It was
  rejected because it makes M1's grade depend on the fellow eye while M2 sees one
  image. Their disagreement would then partly measure asymmetric disease, not error.

B1 is reported as choosing an operating point under a fixed 512 px cache, not as a
study of resolution sensitivity (D4). The 768 px arm upsampled 512 px data.

The six final models (Phase 6a) reached validation QWK of 0.746–0.784. Both seed-43
runs stopped early, at best epochs 3 and 1, under the frozen patience rule. They were
not retrained, because retraining a seed for a better result would be choosing seeds by
outcome. Table 6.1 gives the grader's performance on the locked sets.

**Table 6.1 — M1 on the three test sets, mean (min–max) over three seeds.**

| Set | Variant | QWK | Exact accuracy |
|---|---|---|---|
| EyePACS test (custom, D8) | `eyepacs_full` | 0.777 (0.762–0.784) | 0.747 (0.715–0.770) |
| EyePACS test (custom, D8) | `eyepacs_ddr_full` | 0.771 (0.758–0.792) | 0.733 (0.709–0.765) |
| APTOS | `eyepacs_full` | 0.847 (0.818–0.873) | 0.615 (0.525–0.685) |
| APTOS | `eyepacs_ddr_full` | 0.869 (0.862–0.880) | 0.677 (0.632–0.709) |
| Messidor-2 | `eyepacs_full` | 0.784 (0.772–0.805) | 0.669 (0.642–0.698) |
| Messidor-2 | `eyepacs_ddr_full` | 0.717 (0.684–0.752) | 0.599 (0.559–0.661) |

QWK is higher on APTOS than in-domain while exact accuracy is lower. The two agree once
prevalence is taken into account. APTOS carries more severe disease (51% of images are
grade 1 or above, against 26% in EyePACS), and QWK rewards getting the order of a wide
spread of grades right. Part of the external gap is also label noise rather than model
failure: EyePACS grades come from a single grader, while Messidor-2's are adjudicated by
three specialists.

## 6.2 The protocol-matched benchmark (not run, D8)

The pre-registration planned one leaderboard-comparable figure: QWK on the official
EyePACS test partition. It was dropped before any locked data were read (D8). The
training variants regrouped every EyePACS image into patient-grouped splits, so about
80% of the official test partition sits in the models' training, validation or
calibration data. The mirror does not record the competition's own partition, so it
could not be recovered. No figure in this thesis is comparable with the leaderboard.

## 6.3 The evidence pathway (C1–C4, and M3 on the test sets)

The disc-and-fovea network failed its gate: 0.686 disc diameters against a 0.5 target
(C1). So the reasoner ran on lesion counts alone and could reach grades 0–2 only (D1).
The lesion segmenter reached a mean Dice of 0.505 on DDR, with microaneurysms at 0.344
(C2). Moved to IDRiD, which it had never seen, it kept a Dice of 0.425, a 16% relative
drop rather than a collapse (C3). Graded by M3 alone, the evidence pathway reached QWK
0.375 on 11,767 DDR images it had not been trained on, against the grader's 0.679 (C4).
That is informative but weaker, which is the condition under which disagreement is
worth testing (§8 of the pre-registration).

The internal pass exposed the pathway's weakness before any locked data were read. At
the frozen operating point, where one lesion component of 4 px or more counts, M3 found
evidence of disease in 74.5% of calibration images, against about 19% truly referable.
D12 therefore fitted a per-type minimum lesion area on the calibration split, choosing
by M3's own QWK and never consulting M1. Table 6.2 shows both operating points on the
locked sets.

**Table 6.2 — M3 against the truth on the locked sets, registered → amended (D12).**

| Set | True grades 0 / 1 / 2 / 3 / 4 | M3 QWK | Evidence on grade-0 images | No evidence on referable images |
|---|---|---|---|---|
| EyePACS test | 13,002 / 1,226 / 2,593 / 414 / 380 | 0.141 → 0.444 | 75.3% → 29.6% | 2.5% → 21.0% |
| APTOS | 1,805 / 370 / 999 / 193 / 295 | 0.366 → 0.755 | 62.4% → 10.5% | 0.0% → 0.9% |
| Messidor-2 | 1,017 / 270 / 347 / 75 / 35 | 0.152 → 0.505 | 85.3% → 35.7% | 0.0% → 6.1% |

The amended operating point was fitted on EyePACS alone and transfers: on APTOS, M3
reaches QWK 0.755. It trades specificity for sensitivity. In-domain, 21% of referable
images now get no evidence at all.

## 6.4 Calibration under shift (D1–D4, H2)

**D1: in-domain.** Stages 0+1 of the registered pipeline (the sampling-prior correction,
D11, then temperature scaling) take ECE from 0.24–0.27 to 0.039–0.062.
Bias-corrected temperature scaling (D13) reaches 0.016–0.020.

**D2: does it transfer?** Partly. The stage-0+1 ECE is 0.04–0.11 on APTOS and 0.09–0.13
on Messidor-2, against 0.18–0.25 uncalibrated.

**D3, and H2: does prior-shift correction help?** No. The Saerens–Decock EM step made
external calibration worse in all twelve model-by-set cells of the registered analysis,
by 0.04–0.25 ECE. In the amended analysis it was worse in eleven of twelve, and the
twelfth moved by +0.0003. The oracle, which corrects to the true prior, separates the
two external sets:

| | stages 0+1 | after EM | oracle prior |
|---|---|---|---|
| APTOS, `eyepacs_full` | 0.082 (0.056–0.113) | 0.168 (0.130–0.223) | 0.120 (0.090–0.162) |
| Messidor-2, `eyepacs_full` | 0.122 (0.112–0.133) | 0.184 (0.178–0.192) | 0.057 (0.048–0.073) |

On Messidor-2 the true prior roughly halves ECE, so it is EM's estimate of the prior
that fails. On APTOS even the true prior makes calibration worse, so the shift there is
not the label shift the method assumes. **H2 is not supported.**

**D4: how much unlabelled target data EM needs.** More data does not help. With
`eyepacs_full`, EM's ECE on APTOS is 0.164 from 50 unlabelled images and 0.168 from all
of them. On Messidor-2 it is 0.183 and 0.184. In the amended analysis, APTOS gets worse
as the sample grows (0.195 → 0.241). Messidor-2 improves (0.266 → 0.118) but stays
above BCTS alone (0.093). An estimate that does not improve with more data is biased,
not noisy, which fits a shift that is not pure label shift.

## 6.5 Faithfulness (E1–E3)

The randomisation test compares removing M2's lesion regions with removing 19
equal-area random regions. A lesion image counts as *faithful* when removing its lesions
lowers M1's expected grade more than every one of the 19 controls. If the lesion regions
were no different from random ones, that would happen about 5% of the time.

**Table 6.4 — Faithfulness (E1–E3), `eyepacs_full`, mean of three seeds.**

| Set | Lesion images decided | Beats the mean control (E2) | Faithful: beats all 19 | Faithful by true grade 0 / 1 / 2 / 3 / 4 (E3) |
|---|---|---|---|---|
| EyePACS test | 14,095 of 14,158 | 83.3% | 66.5% | 0.60 / 0.77 / 0.87 / 0.84 / 0.59 |
| APTOS | 2,900 of 2,984 | 85.5% | 70.0% | 0.47 / 0.96 / 0.89 / 0.75 / 0.59 |
| Messidor-2 | 1,561 of 1,574 | 90.8% | 77.8% | 0.67 / 0.85 / 0.96 / 0.95 / 0.74 |

**The grader uses the lesions the evidence pathway finds.** Removing them changes M1's
expected grade (E1: median 0.041 in-domain, 0.206 on APTOS, 0.145 on Messidor-2). More
to the point, it changes the grade more than random removal of the same area, far above
the rate chance allows (E2). Faithfulness is strongest where the lesions are real, in
grades 1–3 (0.75–0.96). It is weakest in grade 0, where M3's lesions are mostly false
positives, and in grade 4, perhaps because disease there is widespread, so a random
region often lands on disease too. This answers the question every reviewer asks of an explanation: yes, it
is lesion-specific.

## 6.6 Selective triage: H1 and H1′

Every gating arm ranks the cases it trusts least for deferral first. The
coverage–accuracy AUC is the mean accuracy over every possible number of accepted
cases, with tied cases taken in expectation (`ANALYSIS_PLAN.md` §6.2). No gating at all
scores exactly the overall accuracy.

**Table 6.3 — Coverage–accuracy AUC per arm, mean of three seeds.** Disagreement and
combined give registered → amended; the other arms are the same in both analyses.

| Set | Variant | none | confidence | OOD | disagreement | combined |
|---|---|---|---|---|---|---|
| EyePACS test | `eyepacs_full` | 0.747 | **0.917** | 0.858 | 0.746 → 0.801 | 0.815 → 0.890 |
| EyePACS test | `eyepacs_ddr_full` | 0.733 | **0.912** | 0.845 | 0.735 → 0.792 | 0.804 → 0.883 |
| APTOS | `eyepacs_full` | 0.615 | **0.860** | 0.816 | 0.549 → 0.672 | 0.775 → 0.860 |
| APTOS | `eyepacs_ddr_full` | 0.677 | **0.889** | 0.840 | 0.633 → 0.747 | 0.801 → 0.889 |
| Messidor-2 | `eyepacs_full` | 0.669 | **0.841** | 0.768 | 0.659 → 0.716 | 0.727 → 0.805 |
| Messidor-2 | `eyepacs_ddr_full` | 0.599 | **0.803** | 0.687 | 0.607 → 0.667 | 0.687 → 0.765 |

**H1 and H1′ are not supported.** The tested effect is disagreement minus confidence.
Per seed it runs from −0.156 to −0.191 in-domain, −0.228 to −0.367 on APTOS and −0.167
to −0.217 on Messidor-2 in the registered analysis. In the amended analysis it runs from
−0.104 to −0.134, −0.114 to −0.236 and −0.116 to −0.156. Every paired 95% interval lies
below −0.09. H1 and H1′ therefore also meet the pre-registration's own falsification
criterion (§1): confidence-gating beats disagreement-gating, in every seed of both
variants, on every set, in both analyses.

**Registered: disagreement carries no signal.** In-domain it ranks within 0.011 of no
gating in every model, and on Messidor-2 within 0.037. On APTOS it is worse than no
gating in all six models, by 0.030–0.076. The in-domain result had been predicted to
within 0.002 by the rehearsal on the validation split. There, disagreement's largest
level covered 47% of images, and M1 was right on 91% of them. Most of those were M3
reporting disease in healthy eyes that M1 correctly called grade 0.

**Amended: a real signal, still well short.** Once M3 is specific enough (D12),
disagreement beats no gating in every model on every set, by 0.037–0.084. It remains
0.10–0.24 behind confidence.

**The combined policy comes closest.** It sets an action level from the evidence,
faithfulness and OOD signals, and orders by confidence within each level. Amended, it
ties confidence on APTOS (mean 0.860 against 0.860) and is ahead in four of six models
there, by at most 0.016. It trails confidence by 0.02–0.05 on the other two sets. No
registered claim concerns the combined arm, so this is description, not support.

**Accuracy at fixed coverage** tells the same story as the AUC.

**Table 6.5 — Exact-grade accuracy among accepted cases, `eyepacs_full`.**

| Set | Coverage | none | confidence | OOD | disagreement | combined |
|---|---|---|---|---|---|---|
| EyePACS test | 80% | 0.747 | **0.883** | 0.794 | 0.710 → 0.764 | 0.787 → 0.808 |
| EyePACS test | 90% | 0.747 | **0.813** | 0.769 | 0.727 → 0.746 | 0.794 → 0.807 |
| APTOS | 80% | 0.615 | **0.740** | 0.665 | 0.569 → 0.663 | 0.732 → 0.734 |
| APTOS | 90% | 0.615 | 0.675 | 0.623 | 0.572 → 0.618 | 0.675 → **0.676** |
| Messidor-2 | 80% | 0.669 | **0.766** | 0.682 | 0.643 → 0.680 | 0.683 → 0.708 |
| Messidor-2 | 90% | 0.669 | **0.715** | 0.675 | 0.645 → 0.663 | 0.694 → 0.708 |

Registered disagreement leaves the accepted set *less* accurate than no gating at both
coverages on every set.

**F3: the clinical view.** Among accepted cases at 80% coverage, the table gives
sensitivity and specificity for referable DR (grade ≥ 2), and the share of truly
referable cases the arm deferred.

**Table 6.6 — F3 at 80% coverage, `eyepacs_full`.**

| Set | none | confidence | disagreement, registered | disagreement, amended |
|---|---|---|---|---|
| EyePACS test | 0.695 / 0.978; 20% | **0.835** / 0.982; 31% | 0.717 / 0.971; 3% | 0.671 / **0.987**; 28% |
| APTOS | 0.995 / 0.827; 20% | **1.000** / 0.825; 26% | 0.996 / 0.768; 9% | 0.997 / **0.867**; 21% |
| Messidor-2 | 0.834 / 0.932; 20% | **0.917** / 0.911; 10% | 0.843 / 0.912; 5% | 0.834 / **0.951**; 17% |

The two signals improve different things. Confidence raises the accepted set's
**sensitivity**: it defers the referable cases M1 would have missed. Amended
disagreement raises **specificity** instead: it defers false referrals, and leaves
sensitivity close to the ungated level (0.671 against 0.695 in-domain, 0.997 against
0.995 on APTOS, 0.834 against 0.834 on Messidor-2). For a screening programme, the missed
referral is the costlier error, so confidence's gain is the more valuable one. The
registered disagreement deferred almost no referable cases (3–9%, where chance is 20%),
because its largest disagreements were healthy eyes.

**F4: which form of the signal.** In the registered analysis the binary form, whether
there is any disagreement at all, beats the frozen magnitude form on every set: 0.773
against 0.746 in-domain, 0.554 against 0.549 on APTOS, 0.675 against 0.659 on
Messidor-2. Magnitude's top level is where M3's false evidence concentrates. In the
amended analysis the two forms agree to within 0.001. The plan reports F4 and does not
select on it (§6.3).

**F5: where the combined policy sends cases.** In the registered analysis the policy
defers 56% of in-domain images, and those deferred cases are 81% correct against 85% for
the accepted ones, so deferral carries almost no information. In the amended analysis
it defers 22% at 55% accuracy against 84% accepted. On APTOS the amended policy defers
39% of images, and those are 39% correct against 83.5% for the accepted ones. The
adjacent-grade set holds the truth within one grade in 95–99% of cases everywhere.

## 6.7 The training source: H3 (B7)

Adding all of DDR to training did not improve transfer. Per seed, external QWK for DDR
minus EyePACS-only is −0.091, −0.092 and −0.020 on Messidor-2, and −0.011, +0.047 and
+0.031 on APTOS. **H3 is not supported:** the effect is negative on Messidor-2 in every
seed, and inconsistent in sign on APTOS. In-domain the difference is within noise
(validation QWK −0.0095 on average). The DDR seed-43 run stopped at epoch 1, which
widens this variant's seed spread. B6, the balanced-training ablation listed in the
register, was not run. It tests no hypothesis, and the thesis should say so rather than
leave it silently absent.

## 6.8 Summary of the verdicts

| Hypothesis | Registered (primary) | Amended (D12, D13) |
|---|---|---|
| H1: disagreement beats confidence in-domain | not supported | not supported |
| H1′: …and on both external sets | not supported | not supported |
| H2: prior-shift EM improves external calibration | not supported | not supported |
| H3: DDR in training improves external QWK | not supported | not supported |

The evidence pathway is informative. After D12 it grades APTOS at QWK 0.755. But
disagreement with it is a weaker signal of the grader's errors than the grader's own
calibrated confidence, both in-domain and under shift. Chapter 7 examines why. Its
analyses are exploratory, declared before they ran, and labelled post hoc.

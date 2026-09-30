# How the results compare with published work

> **Working note for chapters 2 and 8, 2026-09-30.** Our numbers are the committed
> readouts (`docs/unblinding/`, `docs/phase8/`): means of the three `eyepacs_full` seeds,
> with the seed range in brackets.
>
> **Check the published numbers against the original papers before citing them.** They
> were gathered through web search, because this environment could not open the
> full-text sites (nature.com, arxiv.org, the NeurIPS proceedings). Each has a source
> below.

**Most published numbers are not directly comparable with ours.** The test images,
labels, metrics or training data differ. Each row says how far the comparison holds.
Rule 6 applies throughout: the in-domain test is a custom, patient-regrouped split (D8),
so no in-domain number of ours sits beside a leaderboard figure.

## 1. The grader (M1)

| Test | This project | Published | How far it compares |
|---|---|---|---|
| EyePACS, in-domain | *not compared* | Kaggle 2015 winner: κ ≈ 0.850 on the official split [7] | **Not at all.** Our split regroups every EyePACS image by patient (D8); rule 6 forbids the comparison |
| APTOS | QWK **0.85** (0.82–0.87), with no APTOS image in training | APTOS 2019 winner: QWK 0.936 on the competition's hidden test set, trained on APTOS plus 88,702 EyePACS images, as an ensemble [6] | **Loosely.** Theirs is in-domain on different images; ours is transfer from EyePACS alone. A gap of about 0.09 is what a single transferred model should expect |
| Messidor-2, adjudicated grades | QWK **0.78** (0.77–0.80), with no Messidor-2 image in training | Against an adjudicated reference: algorithm κ 0.84; ophthalmologists 0.80–0.84; retina specialists 0.82–0.91 [3] | **Loosely.** It is the same kind of reference and the same metric, but not the same images, and their model was trained at far larger scale with adjudicated tuning. Ours sits below individual ophthalmologists' range |
| Messidor-2, referable DR | AUC not computed | Gulshan 2016: AUC 0.99 (private data) [4]; Voets 2019 public-data reproduction: 0.853 [5] | **Context only.** Public-data models land well below the headline results. The reproduction is the relevant reference for a public-data thesis |

## 2. Referral: which gradings to trust

| Question | This project | Published | How far it compares |
|---|---|---|---|
| Does deferring the least confident 20% meet screening targets? | Referable-DR sensitivity / specificity among accepted cases, after the confidence gate defers 20%: EyePACS test **83.5% / 98.2%**, APTOS **100% / 82.5%**, Messidor-2 **91.7% / 91.1%** | Leibig 2017: above 85% sensitivity and 80% specificity (NHS targets) when referring 0–20% of the most uncertain decisions, using MC-dropout uncertainty on Kaggle data [1] | **Reasonably well.** Both use the same referral threshold (grade ≥ 2) and the same idea. A single calibrated network meets the targets on both external sets, and falls just short of 85% sensitivity in-domain |
| Does confidence still work under a change of country and camera? | Coverage–accuracy AUC **0.917** in-domain, **0.860** APTOS, **0.841** Messidor-2. It is the best or joint-best of five arms on every set (the amended combined policy ties it on APTOS) | Band 2021 (RETINA), EyePACS to APTOS: Bayesian methods (MC dropout, deep ensembles, FSVI) beat MAP, and method rankings depend on the task [2] | **Same shift, different models.** Our confidence arm is the simplest MAP baseline. The benchmark suggests a Bayesian baseline would widen confidence's lead, so disagreement lost to the weakest version of its competitor |
| Does disagreement with independent lesion evidence beat confidence? | **No.** Disagreement minus confidence is −0.156 to −0.367 registered and −0.104 to −0.236 amended; every 95% interval lies below −0.09 | No published test of this exact comparison was found | **New evidence,** and negative. It is the thesis's contribution |
| Out-of-distribution distance as a trust signal | Mahalanobis AUC 0.858 / 0.816 / 0.768: above no gate, below confidence | Standard classifiers are over-confident under shift, and an OOD detector helps flag shifted images [11] | **Partly consistent.** OOD distance helps, but here confidence degraded less than expected and stayed ahead |

## 3. The components

| Component | This project | Published | How far it compares |
|---|---|---|---|
| Lesion segmentation (M2) | Dice, lesion-present images: mean **0.505**, microaneurysms **0.344** on DDR; mean **0.425** on IDRiD, never seen in training | IDRiD 2018 challenge, best AUPR: microaneurysms 0.50, haemorrhages 0.68, soft exudates 0.70, hard exudates 0.885, trained on IDRiD [8] | **Not numerically** (Dice against AUPR). The ordering matches: microaneurysms are the hardest lesion in both |
| Grading from lesions | M3's rules on M2's lesions, amended: QWK **0.44** / **0.755** / **0.505**. It stops at grade 2 | Lesion-aware networks use lesions as *features* and report κ 0.88 on EyePACS and referable-DR AUC 0.98–0.99 on Messidor-1 [9] | **Different role.** Lesions help as an input to a learned grader; as an independent rule-based verdict they are weak, which is what C4 and the unblinding show |
| Calibration under prevalence shift (H2) | EM prior correction made external ECE **worse in 23 of 24 cells** | Alexandari 2020: EM with bias-corrected calibration is hard to beat when *only* the class prior shifts [10] | **Consistent** once the assumption is checked. Our shift changes cameras and populations, not just prevalence, so EM's premise does not hold. D13 used their BCTS |
| Explanations (E2) | Removing M2's lesions changes the grader more than all 19 equal-area random removals in **67–78%** of lesion images, against 5% by chance | Ayhan 2022: saliency-map quality varies greatly and often disagrees with clinicians' annotations [12] | **Complementary.** Theirs checks overlap with annotations; E2 intervenes on the image |
| Why evidence goes wrong | M3's false evidence: rim, debris, camera reflex, vessels, microaneurysm-sized specks (chapter 7) | Where ophthalmologists' majority grades differed from adjudicated ones: missed microaneurysms 36%, artifacts 20%, misclassified haemorrhages 16% [3] | **Same weak points** for humans and models: the smallest lesions and image artefacts |

## What this means for the thesis

- **The grader is ordinary, and that is fine.** Transferred to APTOS and Messidor-2, it
  lands where a single public-data model should: below big-data systems and competition
  ensembles, and a little below individual ophthalmologists on adjudicated grades.
- **The baseline behaves as the literature says it should.** Confidence-based referral
  meets the NHS targets that Leibig 2017 used, on both external sets, without Bayesian
  machinery. This makes the negative result stronger. Disagreement lost to the simplest
  form of its competitor, not to a tuned one.
- **The central comparison is new.** No published test was found of whether
  disagreement with an independently supervised lesion pathway beats confidence as a
  deferral signal. The answer here is no, pre-registered, with a mechanism.
- **The failures match known weak points.** The false evidence behind the negative result
  is microaneurysm-scale specks and artefacts, the same things human graders most often
  disagree on [3]. The EM failure is the known failure of the label-shift assumption [10].

## Sources

1. Leibig et al. 2017, *Scientific Reports* 7:17816, "Leveraging uncertainty information from deep neural networks for disease detection". https://www.nature.com/articles/s41598-017-17876-z
2. Band et al. 2021, NeurIPS Datasets and Benchmarks, "Benchmarking Bayesian Deep Learning on Diabetic Retinopathy Detection Tasks". https://arxiv.org/abs/2211.12717
3. Krause et al. 2018, *Ophthalmology* 125:1264–1272, "Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy". https://arxiv.org/pdf/1710.01711
4. Gulshan et al. 2016, *JAMA*, as summarised in [5].
5. Voets, Møllersen and Bongo 2019, *PLoS ONE* 14(6): e0217541, a public-data reproduction of [4]. https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0217541
6. APTOS 2019 Blindness Detection, winning solution, as summarised in https://arxiv.org/pdf/2301.04644
7. Kaggle 2015 Diabetic Retinopathy Detection, winner (B. Graham). https://warwick.ac.uk/fac/sci/wdsi/news/?newsItem=094d43f5505c9edb01506b2135853426
8. Porwal et al. 2020, *Medical Image Analysis*, "IDRiD: Diabetic Retinopathy – Segmentation and Grading Challenge". https://par.nsf.gov/servlets/purl/10189648
9. Sun et al. 2021, CVPR, "Lesion-Aware Transformers for Diabetic Retinopathy Grading". https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Lesion-Aware_Transformers_for_Diabetic_Retinopathy_Grading_CVPR_2021_paper.html
10. Alexandari, Kundaje and Shrikumar 2020, ICML, "Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation". https://proceedings.mlr.press/v119/alexandari20a.html
11. "Distributional Shifts in Automated Diabetic Retinopathy Screening", IEEE ICIP 2021. https://arxiv.org/abs/2107.11822
12. Ayhan et al. 2022, *Medical Image Analysis*, "Clinical validation of saliency maps for understanding deep neural networks in ophthalmology". https://www.sciencedirect.com/science/article/abs/pii/S1361841522000172

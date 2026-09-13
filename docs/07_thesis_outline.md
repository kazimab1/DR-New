# VERIFY-DR — Thesis Outline

Target ~15,000–20,000 words. Each section names the experiments that feed it, so you
never face a blank page — you are writing up results you already have.

---

## 1. Introduction (~1,500 words)

- Diabetic retinopathy: prevalence, why screening works, why it does not scale
- The automation opportunity and where current systems fall short
- **The gap:** systems that defer on their own confidence are asking the model to
  grade its own homework
- Research question and the three hypotheses
- Contributions, stated modestly and precisely
- Thesis structure

## 2. Background and related work (~3,000 words)

| Section | Content | Position yourself against |
|---|---|---|
| 2.1 | DR pathology and the ICDR scale | — |
| 2.2 | Deep learning for DR grading | Gulshan 2016; Krause 2018 (adjudication and label noise) |
| 2.3 | Ordinal regression | CORAL / CORN |
| 2.4 | Lesion-based and lesion-aware grading | Wei 2020; Sun 2021 — *they use lesions to boost accuracy; you use them to verify* |
| 2.5 | Uncertainty and selective prediction | **Leibig 2017** — the baseline this project exists to beat |
| 2.6 | Calibration and dataset shift | Guo 2017; Saerens–Decock EM |
| 2.7 | Explanation and faithfulness | Boreiko 2022; the case against unfalsifiable saliency maps |
| 2.8 | Summary of the gap | Nobody tests whether an *independent* pathway makes a better deferral signal than confidence |

## 3. Data (~2,000 words) — from `docs/05_dataset_card.md`

- Six Kaggle sources, roles, distributions
- The prevalence shift (EyePACS 73% grade-0 → APTOS 49%) and why it matters later
- Preprocessing and the 512 px cache
- Patient-grouped splitting, and why the naive image-stem proxy leaks
- Label provenance: single-grader EyePACS vs adjudicated Messidor-2
- Limitations: which lesions are annotated anywhere, and which are not

## 4. Methods (~3,500 words) — from `docs/03_model_architecture.md`

- 4.1 System overview
- 4.2 M0 preprocessing and quality (DDR grade-5 supervision)
- 4.3 M1 grading pathway, ordinal head, auxiliary heads
- 4.4 M2 evidence pathway, 4 channels, OD/fovea geometry
- 4.5 M3 reasoner — partial ICDR, and **what it declares unobservable**
- 4.6 M4 calibration, faithfulness test, disagreement gate
- 4.7 **Pre-registration and protocol** — write this from
  `preregistration/PREREGISTRATION.md`; it is already in the right order

## 5. Experimental setup (~1,500 words) — from `docs/04_experiment_register.md`

- The 26 experiments and what each decides
- Metrics: QWK, macro-F1, MAE, ECE, NLL, Brier, AUROC, Dice, coverage–accuracy AUC
- Statistical plan: seeds, bootstrap CIs, paired tests
- Compute environment

## 6. Results (~3,500 words)

| Section | Experiments |
|---|---|
| 6.1 Grading pathway selection | B1–B5 |
| 6.2 Protocol-matched benchmark | H1 — *the* comparable number |
| 6.3 Evidence pathway | C1–C4 |
| 6.4 Calibration under shift (H2) | D1–D4 |
| 6.5 Faithfulness | E1–E3 |
| 6.6 **Selective triage (H1, H1′)** | F1–F5 — the headline |
| 6.7 Training-source ablation (H3) | B6, B7 |

## 7. Error analysis (~2,000 words) — from G1–G3

The chapter that earns the marks. Lead with **figure 1**: a real case the grader calls
confidently normal while the evidence path finds microaneurysms — caught by
disagreement gating, missed by confidence gating.

## 8. Discussion (~2,000 words)

- What the hypotheses actually showed, including the negatives
- Why disagreement does or does not beat confidence — mechanism, not just outcome
- Label noise vs generalisation failure: how much of the external gap is which
- Clinical framing: what a screening programme would gain, honestly bounded
- Limitations: single-field quadrant approximation, unobservable lesions, ~840
  annotated images, three seeds, no prospective validation

## 9. Conclusion and future work (~800 words)

Future work worth naming: FGADR for the missing lesion channels (needs a signed
agreement); multi-field imaging for true 4-2-1; prospective evaluation; a reader study
comparing clinician trust in rule traces versus saliency maps.

---

## Writing order

Do **not** write chapter 1 first.

1. **Chapter 4 (Methods)** — from the pre-registration, before results arrive
2. **Chapter 3 (Data)** — from the dataset card
3. **Chapter 6 (Results)** — as each stage completes; never let results pile up
4. **Chapter 7 (Error analysis)** — while the failures are fresh
5. **Chapter 2 (Related work)** — once you know what you are positioning against
6. **Chapter 8 (Discussion)**
7. **Chapters 1 and 9** — last, when you know what the thesis actually says

## Three things a viva will probe

1. **"How do you know this isn't leakage?"** → patient-grouped splits with an
   assertion, locked externals, a committed pre-registration hash.
2. **"Your QWK is below the leaderboard."** → correct, and stated in the abstract.
   You are not competing on accuracy; H1 reports on the official split precisely so
   the comparison is honest.
3. **"Is the explanation real?"** → E2. The random-region control is why you can
   answer yes or no rather than assert.

# VERIFY-DR — Glossary

## Clinical

| Term | Meaning |
|---|---|
| **DR** | Diabetic retinopathy — damage to retinal blood vessels caused by diabetes; a leading cause of preventable blindness in working-age adults |
| **ICDR** | International Clinical Diabetic Retinopathy scale — the 5-point severity grading used throughout (0–4) |
| **NPDR** | Non-proliferative DR — grades 1–3 |
| **PDR** | Proliferative DR — grade 4; new vessel growth |
| **rDR** | Referable DR — grade ≥ 2; the threshold at which a patient needs an ophthalmologist |
| **VTDR** | Vision-threatening DR — grade ≥ 3 |
| **MA** | Microaneurysm — the earliest visible sign; 10–20 px at full resolution, which is why input resolution matters so much |
| **Haemorrhage** | Intraretinal bleeding; counts per quadrant drive the 4-2-1 rule |
| **Hard exudate** | Lipid deposits; bright yellow, well-defined |
| **Soft exudate / cotton wool spot** | Nerve-fibre infarct; pale, fuzzy-edged |
| **Venous beading** | Irregular vein calibre — a severe-NPDR sign. **Not annotated in any Kaggle dataset.** |
| **IRMA** | Intraretinal microvascular abnormality — a severe-NPDR sign. **Not annotated in any Kaggle dataset.** |
| **Neovascularisation** | New vessel growth; defines PDR. **Not annotated in any Kaggle dataset.** |
| **4-2-1 rule** | Severe NPDR if: ≥20 haemorrhages in each of 4 quadrants, **or** venous beading in ≥2 quadrants, **or** IRMA in ≥1 quadrant. This project implements the haemorrhage arm only and declares the rest unobservable. |
| **ETDRS** | Early Treatment Diabetic Retinopathy Study — defines the 7-field imaging protocol the 4-2-1 rule assumes. This project has one field. |
| **Optic disc** | Where the optic nerve enters; the anatomical origin for the quadrant frame |
| **Fovea** | Centre of sharp vision; with the disc, defines the temporal axis |
| **Disc diameter (DD)** | Standard retinal distance unit; C1's accuracy target is < 0.5 DD |
| **Gradable** | Image quality sufficient to assign a grade. DDR labels ungradable images as grade 5. |

## Metrics

| Term | Meaning |
|---|---|
| **QWK** | Quadratic weighted kappa — the standard DR grading metric. Penalises distant errors quadratically, so 0↔4 costs far more than 2↔3. Chance = 0, perfect = 1. |
| **Macro-F1** | F1 averaged equally over classes; unlike accuracy it does not let a model ignore rare grades |
| **MAE** | Mean absolute error in grade units; the natural ordinal metric |
| **AUROC** | Area under the ROC curve; used for the binary rDR and VTDR heads |
| **ECE** | Expected calibration error — gap between confidence and accuracy, binned. Low ECE means a stated 80% is actually right 80% of the time. |
| **NLL** | Negative log-likelihood; punishes confident errors hard |
| **Brier** | Mean squared error on probabilities |
| **Dice / IoU** | Segmentation overlap measures; used per lesion channel |
| **Coverage** | Fraction of cases the system decides automatically rather than deferring |
| **Coverage–accuracy curve** | Accuracy on accepted cases as coverage varies. The central figure of this thesis. |
| **Selective prediction** | Letting a model abstain. The question here is *what signal* it should abstain on. |

## Method

| Term | Meaning |
|---|---|
| **Ordinal regression** | Predicting an ordered label. Grades are ordered, so a plain softmax throws away structure. |
| **CORAL** | Ordinal method using rank-monotonic binary thresholds; the basis of M1's head |
| **Focal loss** | Down-weights easy examples; handles the ~36:1 grade imbalance without discarding data |
| **Temperature scaling** | One-parameter post-hoc calibration (Guo 2017) |
| **Prior shift / label shift** | When class prevalence differs between train and deployment but the class-conditional appearance does not. Exactly the EyePACS→APTOS situation. |
| **Saerens–Decock EM** | Estimates the target prior from *unlabelled* target data and reweights predictions. Uses no target labels, so it does not break the external lock. |
| **OOD** | Out-of-distribution; here an embedding-space z-score against the training manifold |
| **Counterfactual test** | Modify the input, see whether the prediction moves as the explanation implies |
| **Faithfulness** | Whether an explanation actually reflects the computation. Untested explanations are decoration. |
| **Disagreement gating** | **This project's contribution.** Deferring when two independently-supervised pathways disagree, rather than when one is unconfident. |
| **Pre-registration** | Committing hypotheses and protocol before seeing results — what makes a negative result publishable |
| **Patient-grouped split** | All images from one patient in one split. DR is bilateral; splitting a patient's two eyes leaks. |
| **Natural prevalence** | Test-set class distribution matching reality. Balancing a test set makes specificity and PPV meaningless. |

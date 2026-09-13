# VERIFY-DR — Pre-registration

> **Fill this in completely at the end of Phase 4, before any external dataset is
> evaluated. Commit it. Record the commit hash below. Show it to your supervisor and
> get written acknowledgement.**
>
> After this is committed, the architecture, hyperparameters and decision thresholds
> are frozen. Later changes are allowed but must be recorded as dated deviations at
> the bottom of this file.

---

## Administrative

| Field | Value |
|---|---|
| Project | VERIFY-DR |
| Author | _(name)_ |
| Supervisor | _(name)_ |
| Institution | _(institution)_ |
| Freeze date | _(YYYY-MM-DD)_ |
| Freeze commit hash | _(fill in after committing)_ |
| Supervisor acknowledgement | _(date + how — email is fine)_ |

---

## 1. Hypotheses

| | Hypothesis | Decided by | Falsified if |
|---|---|---|---|
| H1 | Disagreement-gating dominates confidence-gating on the coverage–accuracy curve | F1 | Confidence-gating matches or beats it at equal coverage |
| H2 | Prior-shift correction restores external calibration | D3 | External ECE unchanged after correction |
| H3 | Adding DDR to training improves external transfer | B7 | External QWK unchanged or worse |
| H1′ | The H1 ranking survives dataset shift | F2 | Disagreement wins internally but not externally |

## 2. Frozen model configuration

Copy the selected config to `preregistration/frozen_config.yaml`. Summarise here:

| Setting | Value |
|---|---|
| Input resolution | _(from B1)_ |
| Backbone | _(from B2)_ |
| Head | _(from B3)_ |
| Loss + focal γ | _(from B3)_ |
| Sampler | _(from B4)_ |
| Eye-pair fusion | _(from B5)_ |
| Epochs / early stopping | _(…)_ |
| Optimiser, LR, schedule | _(…)_ |
| Seeds | 42, 43, 44 |
| Evidence segmenter | ResNet18-UNet, 4 channels |
| Evidence training data | DDR-seg + IDRiD-seg |
| OD/fovea regressor | ResNet18 + coordinate head, IDRiD |

## 3. Models to be evaluated externally

Exactly these, and no others:

1. Frozen recipe trained on `eyepacs_full`, seeds 42/43/44
2. Frozen recipe trained on `eyepacs_ddr_full`, seeds 42/43/44

Declaring both in advance is what makes H3 a hypothesis test rather than fishing.

## 4. External evaluation plan

| Dataset | Use | Touched |
|---|---|---|
| EyePACS official test (53,576) | H1 protocol-matched benchmark | Once |
| APTOS 2019 train (3,662) | External generalisation | Once |
| Messidor-2 (~1,744 gradable) | External generalisation | Once |

**Permitted exception:** the Saerens–Decock EM step (D3) reads *unlabelled* target
images to estimate the target prior. No target labels are used. This is documented in
the methods so it is not mistaken for leakage.

## 5. Primary outcome

Area under the coverage–accuracy curve, disagreement-gating vs confidence-gating, on
APTOS and Messidor-2 (**F2**).

## 6. Secondary outcomes

- Accuracy at fixed 80% and 90% coverage (F1, F2)
- rDR sensitivity/specificity at fixed coverage (F3)
- External ECE before and after prior-shift correction (D2, D3)
- Δgrade for lesion vs random-region removal (E1, E2)
- QWK on the official EyePACS test split (H1)

## 7. Analysis plan

- Three seeds; paired comparison across seeds
- Bootstrap 95% CIs, 2,000 resamples over test images
- No difference claimed that is smaller than the seed-to-seed spread
- ECE with 15 bins

## 8. What would falsify the project's premise

If C4 shows the evidence pathway is uninformative (QWK near chance), then disagreement
between the pathways carries no signal and H1 cannot hold for the stated reason. This
outcome is reported as a negative result with analysis, **not** worked around by
tuning the evidence path against test data.

## 9. Stopping rule

External evaluation runs **once**. If a bug is found afterwards, the fix and the
re-run are both recorded as deviations, with the original numbers retained in this
file.

---

## Deviations

Append dated entries. Never edit the sections above after the freeze.

| Date | Section | Change | Reason |
|---|---|---|---|
| | | | |

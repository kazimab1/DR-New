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

### M1 — grading pathway (settled; Stage B closed)

| Setting | Value | From |
|---|---|---|
| Input resolution | **512 px** (the cache's native size) | B1 |
| Backbone | **EfficientNet-B0** | B2 |
| Head | **CORAL ordinal**, 4 cumulative logits | B3 |
| Loss + focal γ | **focal-ordinal, γ = 2.0** | B3 |
| Sampler | **stratified exposure** (inverse grade frequency) | B4 |
| Eye-pair fusion | **none** | B5 |
| Epochs / early stopping | 12 epochs, early stop on val QWK, patience 3 | default |
| Optimiser, LR, schedule | AdamW, lr 3e-4, weight decay 1e-4, 2 warm-up epochs then cosine | default |
| Batch size / dropout | 32 / 0.3 | default |
| Seeds | 42, 43, 44 | |

Every Stage B ablation either lost or tied against this configuration, which is also
the default specified in `docs/03_model_architecture.md` before any run happened.

### M2 — evidence pathway

| Setting | Value | Status |
|---|---|---|
| Evidence segmenter | ResNet18-UNet, 4 channels, 0.5·Dice + 0.5·BCE | settled (C2) |
| Evidence training data | **DDR-seg only; IDRiD held out** | **changed** — see deviation D2 |
| OD/fovea regressor | ResNet18, **head to be fixed** — coordinate vs heatmap | **OPEN** — see below |
| Quadrant reasoning (R4) | available only if the OD/fovea gate passes | depends on the above |

> **Two rows here are not yet frozen, and this section cannot be signed off until they
> are.**
>
> **Evidence training data** originally read "DDR-seg + IDRiD-seg". C2 trains on DDR
> alone with IDRiD held out, which makes IDRiD a genuine external domain for C3. That
> is a deliberate change of design, recorded as a deviation rather than silently
> edited.
>
> **The OD/fovea head is undecided.** C1 with a coordinate head failed its 0.5 DD gate
> at 0.686, and the per-image dump attributes 47% of the disc error to 12% of images
> that place the disc on the wrong side of the fovea. A heatmap head is implemented and
> is the hypothesised fix, but it has not been run on this data. **Whichever is frozen
> here determines whether M3's rule R4 (the 4-2-1 severe-NPDR criterion) can fire at
> all**, so it is not a detail that can be settled later.

### Not yet logged — Phase 4's exit condition is unmet

| | Needs | Cost |
|---|---|---|
| C3 cross-domain | run the eval-only cell | ~1 min GPU |
| C1 head choice | run the comparison cell | ~1.5 min GPU |
| **C4 evidence-only grading** | **M3 does not exist yet** | hours of Python, no GPU |

C4 is the falsification test named in section 8: if the evidence pathway is
uninformative, disagreement carries no signal and H1 cannot hold for the stated reason.
Freezing before C4 means committing 21.7 GPU-hours of Phase 6 training to a premise that
has not been tested.

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

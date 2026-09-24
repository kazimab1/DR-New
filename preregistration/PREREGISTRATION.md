# VERIFY-DR — Pre-registration

> **FROZEN 2026-09-16.** Filled in at the end of Phase 4, before any external dataset was
> evaluated. APTOS and Messidor-2 are untouched as of this commit; Phase 6a trains on
> internal data only, and Phase 6b — the single external evaluation — has not run.
>
> ### How the hash is recorded
>
> The freeze is **the commit that first contained this filled-in file**, and its hash
> is written into the *next* commit. It cannot be written into the freeze commit
> itself: a commit hash covers the file contents, so writing the hash into the file
> changes the hash. Amending would produce a file naming a commit that no longer
> exists. The follow-up commit names the freeze commit truthfully, and
> A local tag `preregistration-freeze` also points at it. That tag is not yet on
> the remote — push it with `git push origin refs/tags/preregistration-freeze`.
> The hash recorded above is the authoritative record regardless.
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
| Freeze date | **2026-09-16** |
| Freeze commit hash | [`17472a2`](../../commit/17472a2497054158c20588e9b71505b40b44a590) — full: `17472a2497054158c20588e9b71505b40b44a590` |
| Supervisor acknowledgement | **2026-09-16** — given directly by the supervisor in the project working session that produced this commit. |

> The three name fields are **outstanding at the freeze and deliberately left blank
> rather than guessed**. They identify the parties; they are not part of what is being
> frozen, and filling them in later changes no protocol, threshold or configuration.
> Complete them before submission.
>
> If the author and the supervisor are the same person, say so here. A supervisor
> acknowledgement is evidence that someone independent saw the protocol before the
> locked data was touched, and a reader cannot assess that from two blank fields.

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
| Evidence training data | **DDR-seg only; IDRiD held out** | settled — deviation D2 |
| OD/fovea regressor | ResNet18 + **coordinate head**, IDRiD Part B | settled — deviation D6 |
| Quadrant reasoning (R4) | **disabled** — the gate was not met | settled — deviation D1 |
| Reachable evidence grades | **0, 1, 2** only | consequence of R4 being disabled |

> **Both rows are now settled by Stage C.**
>
> **Evidence training data.** The template read "DDR-seg + IDRiD-seg". C2 trains on DDR
> alone with IDRiD held out, which gives C3 a genuine external domain — worth more to a
> thesis about generalisation than 81 extra training images. C3 then measured a 16%
> relative Dice drop rather than a collapse.
>
> **The OD/fovea head is the coordinate head.** C1 reached 0.686 DD against a 0.5 gate
> and **failed**. A heatmap head was built to address the laterality flips that carry
> 47% of that error; it gave 1.028 DD and 27/83 flips against 10/83, so it is worse on
> the very thing it targeted. The coordinate head is frozen as the better of two, with
> its failure recorded rather than revised.
>
> **What that costs, stated plainly:** R4 — the 4-2-1 severe-NPDR criterion — needs
> quadrants, quadrants need geometry that passed its gate, and it did not. R4 never
> fires, so **evidence grades 3 and 4 are unreachable** and every C4 figure is a floor.
> `docs/00_START_HERE.md` names this as the designed fallback, not a failure to hide.

### Stage C, logged

| | result | verdict |
|---|---|---|
| C1 | 0.686 DD against a 0.5 gate | **FAIL** — R4 disabled, M3 count-only |
| C2 | mean Dice 0.505; microaneurysm 0.344 | DDR only |
| C3 | 0.425 on held-out IDRiD, −16% relative | **transfers** |
| C4 | **QWK 0.375** against M1's 0.679, n = 11 767 | **informative but weaker** |

**Section 8's falsification test has been run and the premise survived.** C4 was
specified to be capable of ending the project: an evidence path as good as the grader
makes the grader redundant, one at chance makes disagreement meaningless. 0.375 against
0.679 is neither.

Two qualifications that belong beside every C4 figure: only 3 of 5 grades are
reachable, so the QWK is a floor; and `R3*` — haemorrhage or exudate with no
microaneurysm detected — fired on 35% of the corpus, which is M2a's MA channel missing
what its larger-lesion channels find.

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

Recorded at the freeze, before any external evaluation. Later entries are appended
with their date; the sections above are never edited.

| # | Section | Change | Reason |
|---|---|---|---|
| D1 | 2 · reasoner | R4 disabled; evidence grades limited to 0–2 | C1 measured 0.686 DD against a 0.5 gate. Quadrants need geometry that passed; a wrong frame makes R4 confidently wrong rather than cautious. The count-only fallback is named in `docs/00_START_HERE.md`. |
| D2 | 2 · evidence data | `[ddr_seg, idrid_seg]` → **DDR only**, IDRiD held out | Gives C3 a genuine external domain. A thesis claiming disagreement generalises needs an unseen source more than it needs 81 extra training images. C2's specified `+IDRiD` arm therefore did not run. |
| D3 | 7 · analysis plan | B4 tested two samplers, not the three specified | `class_balanced` and `stratified_exposure` are identical at default `epoch_samples` — same weights, same draw count, same index sequence, verified by comparing the emitted indices. `make_sampler` now warns and a test pins the equivalence. |
| D4 | 2 · input resolution | B1 reported as an operating-point choice, **not** a resolution-sensitivity study | All three arms read one 512 px cache, so the 768 arm upsamples and cannot test the claim that motivated the sweep. Testing it honestly needs a cache rebuilt at 768 from the originals. |
| D5 | — · data | EyePACS is 693 images (0.78%) short of the official 88 702 | Upstream of this project: `cache_report.json` shows found = cached = 88 009, failed = 0. The loss is non-random by grade (χ² = 105.6, 4 df; grade 4 lost at 3× grade 0's rate). Prevalence impact negligible (rDR 19.34% → 19.23%), but a severity-dependent selection effect belongs in a referral study's limitations. |
| D6 | 2 · OD/fovea head | A heatmap head was built, tested and **rejected** | Intended to fix the laterality flips carrying 47% of C1's error. It produced 27/83 flips against 10/83 and 1.028 DD against 0.686 — worse on the fovea too, where bimodality cannot apply, so it refutes the remedy rather than the diagnosis. The cause of the flips is not established. A weighted-loss retrain (~1.3 GPU-min) was declined: landmark accuracy is not this project's contribution. |
| D7 | 2 · training | `epochs: 12` → **10** in `frozen_config.yaml` | 12 was `train_grading.py`'s default leaking into the config, not a choice. Every Stage B arm that selected this recipe ran 10 epochs (`03_grading_sweeps.ipynb` sets `epochs = 10`; the register records "seed 42, 10 epochs"), so B1–B5 measured the 10-epoch recipe and 12 would freeze a configuration no experiment evaluated. Corrected before the freeze commit, not after. |
| D8 · 2026-09-22 | 4 · external evaluation; 6 · secondary outcomes | "EyePACS official test (53,576) — H1 protocol-matched benchmark" → **dropped**. The leaderboard comparison is not reported. In-domain results use our own patient-grouped EyePACS test split (17,615 images, 20%), labelled custom-split and never placed beside the 0.8496 leaderboard figure (Rule 6). | The variants were built with `--eyepacs-split regroup`: one patient-grouped split over all 88,009 images that ignores the competition's partition, and `dataset_plan.json` records `leaderboard_comparable: False`. About 80% of any independently chosen EyePACS subset — the official test set included — therefore sits in the six models' train, val or calibration splits. `frozen_config.yaml`'s `split: official_eyepacs  # 35,126 train / 53,576 test` misdescribes the split the models were trained on and is superseded by this row. The partition cannot be recovered from the cache: the mirror's `source_split` is its own 70/15/15 re-split (61,606 / 13,201 / 13,202), not the competition's. Rebuilding it from the competition's `trainLabels.csv` was considered and declined: it would still not be protocol-matched (693 images are missing non-randomly by grade, A0), and the leaderboard winners used eye-pair ensembles this project rejected on purpose (B5). The primary outcome (F2, APTOS and Messidor-2), H2 and H3 do not depend on the official split. Found and recorded **before any locked data was read**. |
| D9 · 2026-09-23 | 5–7 · outcomes and analysis | `ANALYSIS_PLAN.md` added and marked FINAL. It pins every implementation choice §§5–7 left open: the four gate signals, tie handling on the coverage–accuracy curve, the claim rule, the role of every split, and the order of execution. | Settled before any locked label was read, each choice resolved by the frozen text, then the existing code, then the reading that does not favour the project's own hypotheses. It changes no registered outcome; it fixes how each one is computed. |
| D10 · 2026-09-23 | `frozen_config` · `triage.actions` | **REACQUIRE removed**; G2 not run | REACQUIRE is reached only through M0's gradability check, and M0 — the quality head — was never built. Every image is treated as gradable, and the triage policy has three actions, not four. |
| D11 · 2026-09-23 | `frozen_config` · `calibration` | A parameter-free **sampling-prior correction** now precedes temperature scaling | The frozen sampler (`stratified_exposure`, no `--epoch-samples`) draws every grade equally often, so M1's posteriors are expressed under a uniform prior rather than EyePACS's. Temperature scaling cannot re-weight classes, and the Saerens–Decock step needs posteriors calibrated under a known prior. Nothing is fitted: the correction uses the training split's grade counts. D1 reports the temperature-only variant beside it. |
| D12 · 2026-09-24 | 2 · reasoner — M3's operating point | **Amended analysis only.** A lesion type counts as present for M3 when it has a component of ≥ 4 px (the frozen rule) *and* its total predicted area reaches a per-type minimum from {4, 16, 64, 256, 1,024} px, fitted on the calibration split by M3's QWK against the true grade. All-4 px is the frozen rule, which stays the registered primary. | The step-2 rehearsal on val (development data) found M3 reporting evidence of disease in 75.9% of grade-0 images, QWK 0.138. Its operating point was set for pixel segmentation (C2), never for image-level presence. So the largest disagreement marks M1's safest calls (M1 correct 91% where d_evidence = 2), and the disagreement arm ranked no better than no gate. The criterion consults only M3 and the grades, never M1, so it cannot be chosen for H1. It uses the per-type counts and areas the pass already stored; no image is read again. Decided before any locked image or label was read. |
| D13 · 2026-09-24 | `frozen_config` · `calibration`; H2 | **Amended analysis only.** Bias-corrected temperature scaling (Alexandari et al., 2020) replaces stages 0+1 for the calibration outcomes (D1–D4) and H2: one temperature plus one log-bias per grade, fitted by NLL on the calibration split. EM's reference prior becomes the calibration split's grade mix. d_conf, τ_conf and the confidence arm stay on stages 0+1, so H1's baseline is unchanged. | At the rehearsal, EM run on the calibration split itself, where nothing has shifted, drove grade 1 to ~0 in all six models: the mean stage-0+1 posterior for grade 0 was 0.79–0.82 against a true 0.744. Temperature scaling calibrates confidence, not the grade mix, and EM needs the grade mix. With a free bias per grade, the fitted mean posterior equals the calibration grade mix exactly (the bias score equations), which is EM's premise. Decided before any locked image or label was read. |

### The pre-unblinding amendment (D12, D13) · 2026-09-24

The step-2 rehearsal on val showed the registered analysis failing in-domain for two
reasons, each traced to a component rather than to the hypothesis: M3's image-level
specificity, and EM's premise. The author chose, from three options (proceed as frozen;
amend with the frozen analysis primary; amend with the amended analysis primary), to
**amend with the frozen analysis primary**:

- **The registered analysis is unchanged and stays primary** for H1, H1′, H2 and H3.
  The amended analysis (the registered one plus D12 and D13) is reported beside it,
  labelled as a deviation analysis, under the same claim rule (`ANALYSIS_PLAN.md` §8).
- **Both come from one locked pass and one label join.** Nothing else changes: M1, ŷ,
  M2's detections and the faithfulness regions, the OOD statistics, τ_ood, τ_conf, the
  confidence arm, the bootstrap and the data roles.
- **One round.** D12 and D13 are specified in `ANALYSIS_PLAN.md`'s addendum and committed
  before the amended rehearsal runs. After it, `fitted_params.json` is committed and the
  locked pass runs, whatever that rehearsal shows. No further amendment before unblinding.

> **Not a deviation, but decide it here:** Phase 6 at the frozen recipe projects to
> **21.7 GPU-hours** against a ~20 h plan, on 59 842 real training rows. Trim seeds or
> epochs in `frozen_config.yaml` *before* committing. Resolved: epochs were
> already 10 in every Stage B run (D7), and the six Phase 6 runs are executed
> across sessions by `notebooks/06_final_training.ipynb` rather than trimmed.
> Seeds stay at three: section 7's paired-seed analysis needs them.
> Discovering it mid-run is what this document exists to prevent.

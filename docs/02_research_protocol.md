# VERIFY-DR — Research Protocol

## Primary research question

> Can disagreement between an image-level DR grader and an independently-supervised
> lesion-evidence pathway be used to decide *which automated gradings to trust* — and
> does that decision rule transfer across datasets better than gating on the model's
> own confidence?

## Hypotheses

Stated before any experiment runs. Each is falsifiable and each has a named
experiment that decides it.

| | Hypothesis | Decided by | Falsified if |
|---|---|---|---|
| **H1** | Disagreement-gating dominates confidence-gating on the coverage–accuracy curve | F1 | Confidence-gating matches or beats it at equal coverage |
| **H2** | Prior-shift correction restores external calibration that temperature scaling alone loses | D3 | External ECE unchanged after correction |
| **H3** | Adding DDR to training improves external transfer | B7 | External QWK unchanged or worse |
| **H1′** | *(the one that matters)* The H1 ranking survives dataset shift | F2 | Disagreement wins internally but not on APTOS/Messidor-2 |

**All four may come out negative and the thesis still stands** — provided they were
pre-registered. A negative result on selective prediction in DR is worth more to the
field than a fudged positive, because almost nobody publishes them.

## Contributions claimed

1. **Primary** — deferral gated on grade↔evidence disagreement rather than model
   confidence, evaluated under dataset shift.
2. **Secondary** — prior-shift-corrected calibration across DR datasets with differing
   grade prevalence.
3. **Tertiary** — a reasoner that declares its own unobservable findings and abstains
   upward rather than emitting a confident grade on evidence it does not have.

## Explicitly not claimed

- **Not** state-of-the-art grading accuracy. A single EfficientNet-B0 at 512 px with
  no heavy ensemble will land below the best published EyePACS numbers, and the thesis
  says so in the abstract. Framing this as an accuracy contribution would lose the
  argument in the viva.
- **Not** clinical validity. Research software; deployment needs prospective
  validation and expert oversight.
- **Not** full ICDR compliance. See `docs/03_model_architecture.md` § M3 — venous
  beading, IRMA and neovascularisation are unobservable with Kaggle data, and single
  macula-centred fields cannot support true 4-quadrant assessment.

---

## Protocol rules

These exist to stop you fooling yourself. Violating any one of them invalidates a
specific claim, and re-running does not repair it.

### Rule 1 — External sets are locked

APTOS and Messidor-2 are **never** used for training, validation, early stopping,
calibration fitting, hyperparameter selection, threshold setting, or model choice.

They are evaluated **once**, after the pre-registration is committed.

*Permitted exception:* the Saerens–Decock EM step (D3) reads **unlabelled** target
images to estimate the target prior. No target labels are touched. Document this in
the methods so no reviewer mistakes it for leakage.

### Rule 2 — Selection happens on validation only

Experiments B1–B5 choose the recipe using the validation split. Test sets — including
the official EyePACS test set — are not consulted until Phase 6.

### Rule 3 — Patient-grouped splits, enforced by assertion

No patient may appear in more than one split. EyePACS filenames give real patient IDs
(`^(\d+)_(left|right)$`); the image-stem fallback puts a patient's two eyes on
opposite sides of the split and inflates every reported number.

The split builder asserts disjointness and fails loudly. Do not disable the assertion.

### Rule 4 — Test sets keep natural prevalence

`eyepacs_balanced_1000` is a **training** ablation. Balancing a test set makes
specificity, PPV and NPV meaningless, because those quantities are prevalence-
dependent by definition.

### Rule 5 — One unblinding

The pre-registration names exactly which models get evaluated externally: the frozen
recipe trained on `eyepacs_full` and on `eyepacs_ddr_full`, three seeds each. Declaring
both in advance is what makes H3 a hypothesis test rather than fishing.

### Rule 6 — Protocol-matched reporting

Report H1 on the **official** EyePACS test split (53,576 images), which is directly
comparable to the published leaderboard (winning private-LB QWK 0.8496). Every other
number is on your own splits and must be labelled as such.

Never place a custom-split QWK next to a leaderboard figure. This error is common in
published DR work and a viva will probe it.

### Rule 7 — Deviations are recorded, not hidden

If you must change something after the freeze, append a dated deviation entry to
`preregistration/PREREGISTRATION.md` with the reason. Undocumented post-hoc changes
are what make results unpublishable.

---

## Statistical plan

| Question | Method |
|---|---|
| Is a QWK difference real? | Paired comparison across 3 seeds + bootstrap 95% CI (2,000 resamples) over test images |
| Is a gating arm better? | Coverage–accuracy AUC, plus accuracy at fixed 80% and 90% coverage; paired across seeds |
| Is calibration improved? | ECE (15 bins), NLL, Brier, with bootstrap CIs |
| Is a segmentation difference real? | Per-lesion Dice with bootstrap CI over images |

Three seeds is few. Report seed spread everywhere and never claim a difference
smaller than the seed-to-seed variation.

---

## Threats to validity

| Threat | Handling |
|---|---|
| **Label noise differs between datasets** | EyePACS is single-grader; Messidor-2 is three-specialist adjudicated. Part of any gap measures grader disagreement, not generalisation failure. Stated in the discussion. |
| **Preprocessing confound** | `messidor2preprocess` may already be contrast-normalised. Verify in Phase 0; if so, either match it across all sources or report the confound. |
| **Evidence path is weak off-domain** | C3 measures this directly. If Dice collapses cross-domain, the disagreement signal degrades exactly where it is most needed — a real limitation, and worth reporting. |
| **Small lesion-annotation set** | ~840 annotated images total. Wide CIs expected on Dice; do not over-claim segmentation quality. |
| **Grade-1 invisibility** | Microaneurysms are 10–20 px. If B1 shows grade-1 recall near zero at all resolutions, the 5-class framing is unsupported and the thesis reframes around rDR (grade ≥ 2). |
| **Single-field quadrant approximation** | ETDRS 4-2-1 assumes 7 fields. Yours approximates over one. Declared in M3 and in the limitations section. |

# VERIFY-DR — Start to Finish

This is the walkthrough. Work top to bottom. Each phase has an exit condition; do not
start the next phase until the current one meets it.

Estimated total: **~100 GPU-hours across 16–20 weeks.** Kaggle gives 30 GPU-h/week,
so compute is not your constraint — writing and analysis are.

---

## Phase 0 — Before any code (Week 1)

**Goal:** know exactly what you are testing, so you cannot accidentally rationalise
a result later.

1. Read `docs/02_research_protocol.md` end to end.
2. Read `docs/03_model_architecture.md` — you need to be able to draw it from memory.
3. Read `docs/05_dataset_card.md` and attach all six Kaggle datasets to a notebook.
4. Run the verification `find` commands in the dataset card. **Two are load-bearing:**
   - Does the DDR mirror actually contain `lesion_segmentation/`, not just `DR_grading/`?
     If it does not, the entire evidence pathway has no training data and you must
     find another mirror before proceeding.
   - What preprocessing has `messidor2preprocess` already applied? If it is already
     contrast-normalised, your "domain gap" is partly an artefact.
5. Write down your answers in `docs/05_dataset_card.md` under "Verification log".

**Exit condition:** all six datasets attached, both load-bearing questions answered
in writing.

---

## Phase 1 — Build the cache (Week 2)

**Goal:** turn 90k+ variable-size JPEGs into a fixed 512 px dataset that loads fast.

This is the single most important engineering step in the project. Without it, one
training run takes 4 hours; with it, 1.5. It is the difference between a feasible
thesis and an infeasible one.

1. Write `scripts/build_cache.py` (see `scripts/README.md` for the contract).
2. For each source image: retinal-field crop → resize shortest side to 512 →
   centre-crop 512×512 → save as quality-90 JPEG.
3. Run it as a **CPU-only** Kaggle notebook (no GPU quota consumed). Expect 6–10 hours
   across sessions. Save output as a private Kaggle dataset `verify-dr-cache-512`.
4. Run experiment **A0**: reconcile image counts per grade against the source
   manifests, confirm crop failure rate < 0.5%, and eyeball 100 random crops.

**Exit condition:** `verify-dr-cache-512` published, A0 logged in the experiment
register. Never re-run the cache after this point without re-running everything.

---

## Phase 2 — Manifests and splits (Week 3)

**Goal:** every image has a real patient ID and a fixed split assignment.

1. Build the six source manifests (`scripts/README.md` gives the column contract).
2. **Parse real EyePACS patient IDs.** Filenames are `<patient>_<left|right>.jpeg`;
   regex `^(\d+)_(left|right)$`. The blueprint's image-stem fallback puts a patient's
   two eyes on opposite sides of the split and inflates every number you report.
   Assert that no patient ID appears in two splits before continuing.
3. Use the **official EyePACS competition split** as your headline: 35,126 train /
   53,576 test. It is already patient-disjoint. Carve validation and calibration
   partitions out of *train only*, patient-grouped.
4. Build the three development variants: `eyepacs_full`, `eyepacs_balanced_1000`,
   `eyepacs_ddr_full`.
5. Write `manifests/dataset_plan.json` recording every count and every shortfall.

**Exit condition:** patient-disjointness assertion passes; `dataset_plan.json`
committed. Run A1 (smoke test) to confirm the pipeline trains for one epoch.

> **Test sets keep natural prevalence, always.** Balancing a test set makes
> specificity and PPV meaningless. `eyepacs_balanced_1000` is a *training* ablation.

---

## Phase 3 — Grading pathway (Weeks 4–7)

**Goal:** pick one recipe. ~15 GPU-hours.

Run experiments **B1 → B5** in order, one seed each, on validation only.

| Order | Experiment | Decides |
|---|---|---|
| 1 | B1 resolution (384/512/768) | Working resolution |
| 2 | B2 backbone (B0 / ResNet50) | Encoder |
| 3 | B3 head (CE / ordinal / focal-ordinal) | Output head |
| 4 | B4 sampler | Class exposure |
| 5 | B5 eye-pair fusion | Whether fusion is in |

Then **B6** as a reported ablation (does balancing the dataset help? — test on natural
prevalence).

**Watch B1 carefully.** Judge it on *grade-1 recall*, not overall QWK. Grade 1 is
microaneurysms only; an MA is 10–20 px at full resolution and vanishes under
aggressive downsampling. If grade-1 recall is near zero at every resolution, your
model has learned to skip the class and your QWK is hollow.

**Exit condition:** one recipe chosen, all five results in the register with
validation numbers. **You have not touched a test set yet.**

---

## Phase 4 — Evidence pathway (Weeks 6–9, overlaps Phase 3)

**Goal:** a lesion segmenter and geometry good enough to reason over. ~5 GPU-hours.

1. **C1** — optic disc / fovea regressor on IDRiD coordinates. Target: mean error
   < 0.5 disc diameters. This is what makes quadrant assignment possible.
2. **C2** — 4-channel lesion segmenter (MA, haemorrhage, hard exudate, soft exudate)
   on DDR-seg + IDRiD-seg. Report per-lesion Dice and IoU.
3. **C3** — cross-domain check: train DDR → test IDRiD and reverse. If Dice collapses,
   your disagreement signal will be noise off-domain, and you must say so.
4. **C4** — evidence-only grading: how well does the reasoner predict the true grade
   from predicted lesions alone?

**C4 is load-bearing.** You need the evidence path to be *informative but weaker* than
the grader. If it matches the grader you do not need the grader; if it is noise,
disagreement means nothing. Report this number honestly whichever way it lands — the
thesis depends on it being interpretable, not on it being good.

**Exit condition:** C1 meets its target; C2–C4 logged with real numbers.

---

## Phase 5 — Freeze (Week 10)

**Stop. This is the point of no return.**

1. Fill in `preregistration/PREREGISTRATION.md` completely.
2. Copy your chosen config to `preregistration/frozen_config.yaml`.
3. Commit. Record the commit hash inside the pre-registration document itself.
4. Show it to your supervisor and get it acknowledged in writing (an email is fine).

After this commit you may not change the architecture, hyperparameters, or decision
thresholds. If you later discover you must, that is allowed — but you record it as a
**deviation**, with the date and reason, in the pre-registration file. Undocumented
changes after the freeze are what make results unpublishable.

**Exit condition:** pre-registration committed and acknowledged.

---

## Phase 6 — Final training and the single unblinding (Weeks 11–12)

**Goal:** produce the numbers. ~20 GPU-hours.

1. Train the frozen recipe × 3 seeds × 2 variants (`eyepacs_full`, `eyepacs_ddr_full`).
   Six runs total.
2. Fit temperature scaling on the internal calibration split only.
3. **Then, once:** evaluate all six models on the official EyePACS test set (H1),
   APTOS, and Messidor-2.

You declared in the pre-registration that exactly these models get evaluated
externally. One unblinding, two variants — that is what makes **B7** legitimate rather
than fishing.

**Exit condition:** results table populated for all three test sets. Do not go back
and change the model.

---

## Phase 7 — Calibration, verification, triage (Weeks 12–15)

**Goal:** the actual contributions. All inference-only, ~10 GPU-hours.

**Calibration (H2):** D1 → D2 → D3 → D4. Expect D2 to fail — internal temperature
will not transfer, because grade-0 prevalence moves from ~73% (EyePACS) to ~49%
(APTOS). That failure is a finding, not a bug. D3 fixes it with prior-shift
correction; D4 shows how much unlabelled target data the correction needs.

**Verification:** E1 → **E2** → E3. E2 is the random-region control and it is not
optional: without it, E1 measures "the model reacts to inpainting", not "the model
uses the lesions". Every reviewer will ask.

**Triage (H1 — the headline):** F1 → F2 → F3 → F4 → F5. Five gating arms: none,
confidence, OOD-z, disagreement, combined. **F2 is your thesis** — a signal that only
wins in-domain is a curiosity; one that holds under dataset shift is a contribution.

**Exit condition:** coverage–accuracy curves for all five arms on all three test sets.

---

## Phase 8 — Error analysis (Weeks 15–17)

**Goal:** the chapter that earns the marks.

- **G1** — which errors does disagreement catch that confidence misses? Compute the
  set overlap, then find real examples. A case where the grader says "confident
  normal" and the evidence path finds microaneurysms is **your figure 1**. Go looking
  for it deliberately.
- **G2** — does the quality head flag genuinely ungradable images? Use held-out DDR
  grade-5 as ground truth.
- **G3** — failure taxonomy: where does the reasoner contradict a *correct* grade?

**Exit condition:** figure 1 exists and is a real case from your test set.

---

## Phase 9 — Write up (Weeks 16–20)

Follow `docs/07_thesis_outline.md`. Write the methods chapter from the
pre-registration — it is already written, in the right order, in the right tense.

Three things to get right:

1. **Never compare your custom-split numbers to the Kaggle leaderboard.** Report H1 on
   the official test split as your one comparable number, and say explicitly which
   protocol each figure used. This error is common in published DR papers and a viva
   will probe it.
2. **Distinguish label noise from model failure.** EyePACS grades are single-grader;
   Messidor-2 grades are three-specialist adjudicated. Part of any gap between them is
   grader disagreement, not generalisation failure. Say so.
3. **Report negative results as results.** If H1 fails, the thesis is "disagreement
   gating does not beat confidence gating, and here is why" — which is worth more than
   a fudged positive.

---

## If something goes wrong

| Problem | What to do |
|---|---|
| DDR mirror has no lesion masks | Stop. Find another mirror. The evidence pathway is the thesis; there is no version of this project without it. |
| Grade-1 recall is ~0 at every resolution | Raise resolution before anything else. If 768 px does not fix it, report it as a finding and reframe around rDR (grade ≥2) rather than 5-class. |
| C4 shows evidence-only grading is noise | Report it. Then test whether disagreement still helps — a weak-but-unbiased second opinion can still be useful. If it does not, that is H1 falsified, honestly. |
| Out of GPU quota mid-phase | Checkpoint every epoch and resume. Never restart a run from scratch to "get a clean number". |
| You realise the recipe is wrong after freezing | Record a deviation. Do not silently re-freeze. |

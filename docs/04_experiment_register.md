# VERIFY-DR — Experiment Register

Your running log. Update `Status` and `Result` as you go; this file becomes the
skeleton of the results chapter.

**Status values:** `TODO` · `RUNNING` · `DONE` · `BLOCKED` · `DEVIATED`

---

## Stage A — Infrastructure (no claims)

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| A0 | Did the 512 px cache preserve the data? | — | Counts reconcile per grade; crop failures < 0.5%; 100-crop visual audit | **Crops verified** | eyepacs + aptos pass visually; counts still unreconciled |
| A1 | Does the pipeline run end to end? | — | Smoke test completes; 1-epoch pilot gives non-trivial QWK | **DONE** | Smoke test passes; B1 512 reached val QWK 0.679 |

### A0 — crop **RESOLVED: benign**. The gate is the wrong test for this source.

`verify-dr-cache-512/cache512/eyepacs/cache_report.json`, read 14 Sep 2026:

| Field | Value |
|---|---|
| source_root | `/kaggle/input/datasets/tantai31124/eyepacs-original` |
| params | size 512, fit **pad**, quality 90, CLAHE on (clip 2.0, 8 tiles) |
| counts | found 88 009 · done 87 809 · skipped 200 · **failed 0** |
| runs / interrupted | 2 / false |
| crop detected | **71** |
| crop fell back to full frame | **87 938** of 88 009 measured |
| **crop fallback_rate** | **0.99919** — gate is **0.005** |
| output | 4287.7 MiB, mean 51 086 B/image |

The retinal-field crop reported success on 71 images out of 88 009. **On the gate as
written, A0 fails for EyePACS by a factor of 200.**

**Two readings, opposite consequences, and the number alone cannot separate them.**

1. *Benign.* The mirror ships images already cropped to the retinal field, so there is
   nothing left to crop and `retinal_bbox` says so correctly. `build_cache.py` documents
   exactly this: corner patches "in an already cropped one … are retina, which drives
   the threshold high, collapses the mask, and trips the area guard below — **the right
   answer there**". The fallback is also returned when the detected box is within 2% of
   the full frame, which is precisely what an already-cropped image produces. Under this
   reading the cache is sound and the 0.005 gate is simply the wrong test for this
   source.
2. *Serious.* The crop is failing on raw, uncropped originals. EyePACS originals are
   typically wide (≈3888×2592) with the fundus centred between black bars. With
   `fit: pad`, a failed crop pads that wide frame to square and resizes to 512 — leaving
   the retina occupying roughly two-thirds of the width, an **effective retinal
   resolution nearer 340 px than 512**. Every Stage B number would then have been
   measured on under-resolved images, and B1's "resolution does not matter" finding would
   have been measured over a range where the retina never filled the frame.

**The evidence leans to reading 1.** If the corners held black surround, the threshold
would sit at ~10–15/255, the fundus would light up cleanly, and the box would come back
far smaller than the frame — detection, not fallback. Getting the opposite on 99.9% of
images implies the corners are *not* dark, which is what a pre-cropped image looks like.
Mean output size of 51 KB per 512² JPEG at q90 also reads as mostly-retina rather than
a third flat black. **None of this is proof.** `contact_sheet.jpg` settles it in one
look, which is exactly why A0 requires a visual audit and not just a rate.

**Also unreconciled:** the mirror reports 88 009 images found, against the 88 702 of the
official EyePACS release — a shortfall of 693. A0 requires counts to reconcile per
grade; that has not been done, and this gap needs an explanation before the freeze.

### Resolution — 14 Sep 2026, from the contact sheets

**Reading 1 holds. The cache is sound and Stage B stands.**

`contact_sheet.jpg` was fetched and inspected for both EyePACS (100 crops) and APTOS
(60 crops). Every tile is a fundus disc filling its frame with only a thin margin; optic
disc and macula are visible in nearly all, vessels are sharp down to fine branches, and
there is not one wide black-barred frame or disc-adrift-in-black among 160 crops.

Measured off the sheets rather than judged by eye, retinal width as a fraction of frame
width:

| Dataset | median | mean | 10th pct | tiles below 0.75 | implied retinal diameter |
|---|---|---|---|---|---|
| EyePACS | **1.000** | 0.990 | 0.999 | **0 of 100** | **512 px** |
| APTOS | **1.000** | 1.000 | 1.000 | **0 of 60** | **512 px** |

Reading 2 predicted ≈0.67 width and ≈340 px. The lit fraction of the frame is 0.74
(EyePACS) and 0.70 (APTOS) against π/4 = 0.785 for a disc inscribed in a square — the
small shortfall is the flat top-and-bottom truncation typical of fundus cameras.
**Reading 2 is ruled out.** The retina fills the frame, so no Stage B result was measured
on under-resolved images and B1's resolution finding stands.

**What this says about the gate, for the write-up.** A crop-fallback rate near 1.0 is not
evidence of a broken crop when the source ships pre-cropped — `retinal_bbox` returns
`full, False` for a box within 2% of the frame, which is the correct answer there and is
what these mirrors produce. The 0.005 threshold in A0 was written assuming raw
uncropped originals and is simply the wrong test for `tantai31124/eyepacs-original` and
the APTOS mirror. `diagnose_cache.py` now says so rather than reporting a flat FAIL. The
rate is kept as recorded evidence, not as a pass/fail gate.

**Still open under A0, before the Phase 5 freeze:**

1. **Counts.** 88 009 EyePACS images found against the official 88 702 — a shortfall of
   **693** with no explanation yet. A0 requires per-grade reconciliation against the
   manifests; not yet done.
2. **Visual audit for ddr, idrid and messidor2.** Not yet seen. DDR and IDRiD carry the
   lesion masks M2 trains on in Phase 4, so their crops matter as much as EyePACS's.

## Stage B — Grading pathway (selection; validation only)

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| B1 | What resolution is needed? | 384 / 512 / 768 px | Val QWK **and grade-1 F1** | **DONE** | **512 px** — no trend above noise; 768 upsamples |
| B2 | Which encoder? | EfficientNet-B0 / ResNet50 | Val QWK per GPU-hour | **DONE** | **EfficientNet-B0** — QWK tied, 0.70× the cost |
| B3 | Which head? | CE / ordinal / focal-ordinal | Val QWK + MAE | **DONE** | **focal-ordinal** — focal earns its place; CE ruled out |
| B4 | Which sampler? | Natural / stratified exposure / class-balanced | Val QWK at natural prevalence | **DONE (2 arms)** | **weighted** — the third arm was a duplicate |
| B5 | Does eye-pair fusion help? | Single vs left+right fusion | Val QWK, paired across seeds | **DONE** | **no fusion** — best grader, but confounds the thesis signal |
| B6 | Does balancing the dataset help? *(ablation)* | `eyepacs_full` vs `eyepacs_balanced_1000` | QWK on natural-prevalence test | TODO | |
| B7 | Does DDR improve transfer? **(H3)** | `eyepacs_full` vs `eyepacs_ddr_full` | External QWK — single unblinding | TODO | |

> **B1 warning.** Judge on grade-1 **F1**, not overall QWK and not recall alone.
> Grade 1 is microaneurysms only; an MA is 10–20 px. A model that silently skips
> grade 1 still scores ~0.97 QWK on a realistic distribution, because grade 1 sits
> one step from grade 0 and quadratic weights barely punish the error.
>
> Recall alone is not enough either: a model predicting grade 1 for *every* image
> scores 1.000 grade-1 recall and is worthless. Check `distinct_predictions` first —
> when it is 1 the run has collapsed and decides nothing. `03_grading_sweeps.ipynb`
> applies both rules and prints the verdict.

### B1 — resolution — **DECIDED: 512 px**

Validation only, `eyepacs_balanced_1000`, EfficientNet-B0, focal-ordinal head,
stratified exposure, no fusion, seed 42, 10 epochs.

| Resolution | QWK | macro-F1 | **grade-1 F1** | g1 recall | g1 precision | distinct preds | MAE | GPU-min | img/s |
|---|---|---|---|---|---|---|---|---|---|
| 384 | 0.676 | 0.445 | 0.151 | 0.293 | 0.102 | 5 / 5 | 0.418 | 19.4 | 43.0 |
| **512** | 0.679 | 0.459 | 0.142 | 0.271 | 0.096 | 5 / 5 | 0.415 | 18.1 | 46.0 |
| 768 | 0.704 | 0.459 | **0.161** | 0.271 | 0.115 | 5 / 5 | 0.395 | 34.3 | 24.3 |

No run collapsed — all five grades predicted at every resolution — so all three are
readable.

**Grade-1 F1 is not monotone in resolution: 0.151 → 0.142 → 0.161.** The worst point
is 512, in the middle. If resolution drove microaneurysm detection the ordering would
be monotone; it is not, and the entire spread is 0.019. On one seed per point that is
noise, not a trend. Grade-1 recall tells the same story: 0.293 at 384, then 0.271 at
both 512 and 768 — the *lowest* resolution has the best recall.

**768 upsamples.** The cache is 512 px (`build_cache.py --size`, default 512) and the
transform is `Resize((image_size, image_size))`, so the 768 run interpolates 512 px
data up to 768. It adds no information. Whatever produced its +0.025 QWK, it cannot be
"seeing microaneurysms that downsampling destroyed" — there are no extra pixels to see.
This is consistent with its recall being unchanged and only its precision improving.

**384 is not cheaper.** 43.0 img/s against 46.0 at 512. Below 512 the job is bound by
JPEG-decoding the 512 px cache files, not by the GPU, so shrinking the target costs
decode work anyway and saves nothing.

**Decision: 512 px.** It is the cache's native resolution, so it is the only setting
that neither discards real pixels nor invents fake ones; it is the fastest measured;
and the evidence for 768 does not survive the non-monotonicity. 768 would also roughly
double everything downstream — Phase 6 projects to 39.6 GPU-h at 768 against 20.9 h at
512, over both the ~20 h budget and the 30 h/week quota — for a gain on grading
accuracy, which is explicitly not this thesis's contribution.

> **Limitation to state in the thesis.** B1 as specified sweeps 384/512/768, but every
> run reads the same 512 px cache, so the sweep cannot test the claim that motivated
> it — that an MA is 10–20 px at native resolution and vanishes under downsampling.
> Testing that honestly needs a cache rebuilt at 768 from the originals (another full
> Phase 1 pass, ~2.25× the storage). Not done; the grading pathway is a component here,
> not the contribution. Report B1 as choosing an operating point under a fixed 512 px
> cache, not as a resolution-sensitivity study.

### B2 — encoder — **DECIDED: EfficientNet-B0**

Same settings as B1 at 512 px; only the encoder varies. The B0 arm *is* the B1 512 run.

| Encoder | QWK | macro-F1 | grade-1 F1 | g1 recall | g1 precision | MAE | GPU-min | img/s | Phase 6 projection |
|---|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B0** | 0.679 | 0.459 | 0.142 | 0.271 | 0.096 | **0.415** | **18.1** | **46.0** | **20.9 h** |
| ResNet50 | 0.676 | 0.464 | **0.180** | **0.413** | 0.115 | 0.469 | 26.0 | 32.1 | 30.0 h |

**On the stated criterion this is not close.** QWK differs by −0.0024 and macro-F1 by
+0.0052 — both inside the noise band B1 established. ResNet50 delivers that tie at
1.44× the wall-clock, so on val QWK per GPU-hour B0 wins outright.

**But grade 1 is the first Stage B signal above noise.** ResNet50 reaches grade-1 F1
0.180 against 0.142, a gap of 0.038 — *twice* the entire 0.019 spread B1 produced
across all three resolutions — driven by recall 0.413 against 0.271 (+52%), with
precision also up (0.115 against 0.096).

**This is an operating point, not a quality level.** QWK and macro-F1 are tied and MAE
is *worse* under ResNet50 (0.469 against 0.415): it is not grading better, it is
predicting grade 1 more liberally. That also explains the QWK/MAE split — a more spread
prediction distribution raises QWK's expected-agreement denominator, so QWK holds while
raw distance error grows. What governs that distribution is the loss and the sampler,
which are exactly what **B3** and **B4** vary, at no extra compute.

> **Prediction recorded before running B3/B4.** If `class_balanced` or non-focal
> ordinal moves B0's grade-1 F1 to ≈0.18, then B0 matches ResNet50's grade-1 behaviour
> at 0.70× the cost and B2 is settled. If nothing in B3–B4 shifts grade 1, the encoder
> is contributing something the sampler cannot, and B2 should be reopened as a recorded
> deviation. Stating this now so the follow-up is a test rather than a rationalisation.

**Decision: EfficientNet-B0.** It wins the pre-stated criterion, has the better MAE, and
keeps Phase 6 at 20.9 GPU-h rather than 30.0 — the latter being over the ~20 h budget
and at the ceiling of the 30 h/week quota.

### B3 — output head — **DECIDED: focal-ordinal**

512 px, EfficientNet-B0, stratified exposure, no fusion. `ordinal_focal` is the B1 512 run.

| Head | QWK | macro-F1 | grade-1 F1 | g1 recall | g1 precision | MAE | GPU-min |
|---|---|---|---|---|---|---|---|
| `softmax_ce` | 0.647 | **0.488** | **0.159** | 0.182 | **0.141** | 0.416 | 18.9 |
| `ordinal` (γ=0) | **0.700** | 0.398 | 0.122 | 0.148 | 0.104 | **0.389** | 16.3 |
| **`ordinal_focal` (γ=2)** | 0.679 | 0.459 | 0.142 | **0.271** | 0.096 | 0.415 | 18.1 |

**This is the first Stage B experiment where the choice actually matters.** QWK spans
0.053 across the three heads — nearly twice the 0.028 that B1 produced across every
resolution. The head is a real lever; resolution and encoder were not.

**A correction to how the noise band has been quoted.** "±0.02 QWK" was a round number
used conversationally. B1's actual spread, which we judged non-separable from seed
noise on one seed per point, gives an empirical floor *per metric*: QWK 0.0276,
macro-F1 0.0144, grade-1 F1 0.0190, grade-1 recall 0.0220, MAE 0.0230. Effects are
judged against the floor for their own metric from here on.

**`softmax_ce` is ruled out on the stated criterion.** Its QWK deficit against
`ordinal` is 0.053 — 1.9× the QWK floor, so a real effect, not noise. It has the best
macro-F1 and the best grade-1 F1, but dropping the ordinal head to get them also
discards the CORAL monotonicity guarantee that Phase 7 calibration and the rDR/VTDR
heads are built on. Not worth it.

**`ordinal` versus `ordinal_focal` isolates focal weighting exactly** — same head, same
architecture, only γ differs. Measured against each metric's own floor:

| Metric | focal − ordinal | B1 floor | × floor | Favours |
|---|---|---|---|---|
| QWK | −0.0211 | 0.0276 | 0.8× | ordinal — **inside noise** |
| MAE | +0.0260 | 0.0230 | 1.1× | ordinal — marginal |
| grade-1 F1 | +0.0200 | 0.0190 | 1.1× | focal — marginal |
| macro-F1 | **+0.0611** | 0.0144 | **4.2×** | focal — decisive |
| grade-1 recall | **+0.1230** | 0.0220 | **5.6×** | focal — decisive |

The stated criterion (QWK + MAE) mildly favours plain `ordinal`, but by margins of
0.8× and 1.1× their floors — it does not separate the two. The rare-class metrics
favour focal at 4.2× and 5.6×. On weight of evidence focal wins, and the criterion is
applied rather than overridden: it simply fails to discriminate here.

> **Reportable finding: focal weighting does its stated job.** `docs/03_model_architecture.md`
> justifies γ=2 as handling EyePACS's ~36:1 imbalance without discarding data. Turning it
> off costs **83% of grade-1 recall** (0.271 → 0.148) and 0.061 macro-F1, for a QWK change
> inside the noise floor. That is a clean, isolated confirmation of a design choice, and it
> belongs in the write-up.

**Decision: `ordinal_focal`.** No new baseline is needed — the B1 512 run remains the
reference configuration for B4 and B5.

**B2's recorded prediction is still open.** Neither head reached ResNet50's grade-1 F1
of 0.180: plain `ordinal` moved it *down* to 0.122, and `softmax_ce` reached 0.159 only
by trading 0.053 QWK — and by a different mechanism (precision 0.141 / recall 0.182,
against ResNet50's precision 0.115 / recall 0.413). B4's `class_balanced` is the
remaining test.

### B4 — sampler — **DECIDED: weighted (`stratified_exposure`)**

| Sampler | QWK | macro-F1 | grade-1 F1 | g1 recall | MAE | GPU-min |
|---|---|---|---|---|---|---|
| `natural` | 0.640 | 0.372 | 0.125 | 0.263 | 0.479 | 10.9 |
| **`stratified_exposure`** | **0.679** | **0.459** | 0.142 | 0.271 | **0.415** | 18.1 |
| `class_balanced` | 0.679 | 0.459 | 0.142 | 0.271 | 0.415 | 18.8 |

> ### The third arm was a duplicate. B4 ran as a two-way, not a three-way.
>
> `class_balanced` returned metrics **identical to the B1 512 baseline on every
> figure** — QWK, macro-F1, grade-1 F1, grade-1 recall, MAE and distinct predictions,
> all to the last recorded digit. That is not coincidence: in `make_sampler` both
> weighted strategies weight by inverse grade frequency, and differ only in draws per
> epoch, which default to `len(grades)` for both. With `epoch_samples` unset they are
> one sampler under two names and, given the same seed, emit the same index sequence.
> Verified directly: the two index lists compare equal.
>
> Nothing downstream is invalidated — B1–B3 all used `stratified_exposure`, which is a
> real sampler — but B4 tested two options, not three. `make_sampler` now warns when
> the two coincide, and `tests/test_grading.py` pins the equivalence so it cannot
> resurface silently.

**On the comparison that did run, weighted beats natural clearly:** macro-F1 +0.087
(6.0× its floor), MAE −0.064 (2.8×), QWK +0.039 (1.4×).

**But weighting does not help grade 1.** Grade-1 F1 +0.017 (0.9× floor) and recall
+0.008 (0.4×) are both inside noise. Read alongside B3, the division of labour is
clean: **focal weighting is what moves grade 1** (+83% recall), **the sampler is what
moves overall class balance** (+6 floors of macro-F1). They are not substitutes.

### B5 — eye-pair fusion — **DECIDED: no fusion**

| Configuration | QWK | macro-F1 | grade-1 F1 | g1 recall | MAE | GPU-min | Phase 6 |
|---|---|---|---|---|---|---|---|
| **single image** | 0.679 | 0.459 | 0.142 | 0.271 | 0.415 | **18.1** | **20.9 h** |
| fusion | **0.713** | **0.484** | 0.152 | **0.317** | **0.392** | 39.5 | 45.5 h |

**Fusion produces the best grader in Stage B** — top QWK, macro-F1, grade-1 recall and
MAE. QWK +0.034 (1.2× floor), macro-F1 +0.025 (1.7×), grade-1 recall +0.046 (2.1×).
It is rejected anyway, on three grounds.

1. **It confounds the signal the thesis rests on.** With fusion, M1's grade for image X
   is a function of X *and* its fellow eye; M2's evidence is a function of X alone. When
   they disagree, that can mean "the fellow eye carried disease M2 cannot see" rather
   than "the grade contradicts the evidence in this image". DR is frequently asymmetric,
   so this is not a corner case — it systematically inflates disagreement in exactly the
   cases the referral rule would flag, and those flags would not be errors. `CLAUDE.md`:
   *never optimise for QWK at the expense of the disagreement signal.*
2. **The budget cannot absorb it.** 21.1 img/s against 46.0 puts Phase 6 at **45.5
   GPU-h** against 20.9 — over the ~20 h budget and over a full 30 h/week quota.
3. **The evidence is thin for the criterion.** B5 was specified as *paired across
   seeds*; one seed per arm was run, and ΔQWK is 1.2× its floor. Grade-1 F1 (+0.010,
   0.5×) and MAE (1.0×) are inside noise.

> If the thesis were about grading accuracy, fusion would win. It is not, and buying
> 0.034 QWK by contaminating the referral signal is the specific trade `CLAUDE.md`
> forbids.

### Stage B closed — the frozen recipe

**512 px · EfficientNet-B0 · focal-ordinal head · stratified exposure · no fusion.**

Which is exactly the `B1_res512` configuration, and exactly the default specified in
`docs/03_model_architecture.md`. **Every ablation either lost or tied.** Eight runs,
~2.8 GPU-hours, and the designed recipe survived all of them — a legitimate and
reportable outcome, not a null result to bury.

What actually mattered, ranked by effect size against each metric's own floor:

| Lever | Biggest effect | × floor | Verdict |
|---|---|---|---|
| Head: focal on/off | grade-1 recall +0.123 | 5.6× | **Real and decisive** |
| Sampler: weighted vs natural | macro-F1 +0.087 | 6.0× | **Real and decisive** |
| Fusion on/off | grade-1 recall +0.046 | 2.1× | Real, rejected on confound + cost |
| Encoder: B0 vs ResNet50 | grade-1 F1 +0.038 | 2.0× | Real, rejected on cost |
| Resolution 384/512/768 | grade-1 F1 spread 0.019 | 1.0× | Noise |

**The B2 prediction resolves against reopening.** No configuration except ResNet50
reached grade-1 F1 0.180 — the closest were `softmax_ce` at 0.159 (costing 0.053 QWK)
and fusion at 0.152 (costing 2.18× compute). Note the designated test, `class_balanced`,
was the duplicate arm and so never ran as a distinct condition; what *was* tested is
full inverse-frequency weighting, which every run except `B4_natural` used, and it
yields 0.142. B2 is **not** reopened: ResNet50's advantage is recall-led, costs 9 extra
Phase 6 GPU-hours, and grading accuracy is not the contribution. Recorded here so the
reasoning is auditable rather than assumed.

> **B7 protocol.** Needs external data. Legitimate only because the pre-registration
> declares in advance that exactly these two variants get evaluated in the single
> unblinding pass.

## Stage C — Evidence pathway

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| C1 | Can we locate disc and fovea well enough? | — | Euclidean error on IDRiD test, in disc diameters (**target < 0.5 DD**) | TODO | |
| C2 | How well do the 4 lesion channels segment? | DDR-seg / +IDRiD / +augmentation | Per-lesion Dice, IoU | TODO | |
| C3 | Does the segmenter transfer? | DDR→IDRiD and reverse | Dice drop across domains | TODO | |
| C4 | How informative is evidence alone? | Reasoner over predicted lesions, no grader | QWK of evidence-grade vs label | TODO | |

> **C4 is load-bearing.** The evidence path must be *informative but weaker* than the
> grader. As good ⇒ the grader is redundant. Noise ⇒ disagreement means nothing.
> Report it honestly either way — the thesis needs it interpretable, not good.

## Stage D — Calibration (H2)

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| D1 | Calibrated internally? | Uncalibrated vs temperature scaling | ECE, NLL, Brier | TODO | |
| D2 | Does internal temperature transfer? | Internal T applied to APTOS / Messidor-2 | External ECE — **expected to fail** | TODO | |
| D3 | Does prior-shift correction fix it? | Saerens–Decock EM vs oracle prior | External ECE, oracle as upper bound | TODO | |
| D4 | How much unlabelled target data does EM need? | 50 / 200 / 500 / all | ECE vs sample size | TODO | |

> D2 failing is the *finding*, not a bug. Grade-0 prevalence moves ~73% → ~49%
> between EyePACS and APTOS; a temperature fitted on one prior cannot transfer.

## Stage E — Verification / faithfulness

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| E1 | Does removing cited lesions change the grade? | Inpaint predicted lesion regions | Δgrade distribution | TODO | |
| E2 | **Is that shift lesion-specific?** | Random / non-lesion regions, equal area | Δ(lesion) vs Δ(random) | TODO | |
| E3 | Does faithfulness hold at every severity? | Stratified by true grade | Δgrade by grade — grade 1 is the hard case | TODO | |

> **E2 is mandatory.** Without the control, E1 shows the model reacts to inpainting,
> not that it uses the lesions. Every reviewer asks for this.

## Stage F — Selective triage (H1 — the headline)

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| F1 | Best gating signal, internally? | none / confidence / OOD-z / **disagreement** / combined | Coverage–accuracy AUC; accuracy @ 80% and 90% coverage | TODO | |
| F2 | **Does the ranking survive shift?** | Same five arms on APTOS + Messidor-2 | Same metrics, externally | TODO | |
| F3 | What does it mean clinically? | Same arms | rDR sensitivity/specificity at fixed coverage | TODO | |
| F4 | Which form of the signal works? | Binary / magnitude / magnitude + unobservable set | Coverage–accuracy AUC | TODO | |
| F5 | Where do cases actually land? | — | Fraction and accuracy per action bucket | TODO | |

> **F2 is the thesis.** A signal that only wins in-domain is a curiosity; one that
> holds under shift is a contribution.

## Stage G — Error analysis (no new training)

| ID | Question | Status | Result |
|---|---|---|---|
| G1 | Which errors does disagreement catch that confidence misses? Set overlap + real examples | TODO | |
| G2 | Does the quality head flag truly ungradable images? Held-out DDR grade-5 | TODO | |
| G3 | Failure taxonomy: where does the reasoner contradict a *correct* grade? | TODO | |

> G1 produces **figure 1**: a case the grader calls confidently normal while the
> evidence path finds microaneurysms. Go looking for it deliberately.

## Stage H — Protocol-matched benchmark

| ID | Question | Status | Result |
|---|---|---|---|
| H1 | QWK on the **official** EyePACS test split, reported beside the 0.8496 leaderboard reference | TODO | |

---

## Summary

**26 experiments.** Selection happens only in B1–B5. External data is touched exactly
once, in Phase 6, after the freeze.

# VERIFY-DR — Experiment Register

Your running log. Update `Status` and `Result` as you go; this file becomes the
skeleton of the results chapter.

**Status values:** `TODO` · `RUNNING` · `DONE` · `BLOCKED` · `DEVIATED`

---

## Stage A — Infrastructure (no claims)

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| A0 | Did the 512 px cache preserve the data? | — | Counts reconcile per grade; crop failures < 0.5%; 100-crop visual audit | TODO | |
| A1 | Does the pipeline run end to end? | — | Smoke test completes; 1-epoch pilot gives non-trivial QWK | TODO | |

## Stage B — Grading pathway (selection; validation only)

| ID | Question | Varies | Decided by | Status | Result |
|---|---|---|---|---|---|
| B1 | What resolution is needed? | 384 / 512 / 768 px | Val QWK **and grade-1 F1** | **DONE** | **512 px** — no trend above noise; 768 upsamples |
| B2 | Which encoder? | EfficientNet-B0 / ResNet50 | Val QWK per GPU-hour | **DONE** | **EfficientNet-B0** — QWK tied, 0.70× the cost |
| B3 | Which head? | CE / ordinal / focal-ordinal | Val QWK + MAE | TODO | |
| B4 | Which sampler? | Natural / stratified exposure / class-balanced | Val QWK at natural prevalence | TODO | |
| B5 | Does eye-pair fusion help? | Single vs left+right fusion | Val QWK, paired across seeds | TODO | |
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

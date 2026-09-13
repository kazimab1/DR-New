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
| B1 | What resolution is needed? | 384 / 512 / 768 px | Val QWK **and grade-1 recall** | TODO | |
| B2 | Which encoder? | EfficientNet-B0 / ResNet50 | Val QWK per GPU-hour | TODO | |
| B3 | Which head? | CE / ordinal / focal-ordinal | Val QWK + MAE | TODO | |
| B4 | Which sampler? | Natural / stratified exposure / class-balanced | Val QWK at natural prevalence | TODO | |
| B5 | Does eye-pair fusion help? | Single vs left+right fusion | Val QWK, paired across seeds | TODO | |
| B6 | Does balancing the dataset help? *(ablation)* | `eyepacs_full` vs `eyepacs_balanced_1000` | QWK on natural-prevalence test | TODO | |
| B7 | Does DDR improve transfer? **(H3)** | `eyepacs_full` vs `eyepacs_ddr_full` | External QWK — single unblinding | TODO | |

> **B1 warning.** Judge on grade-1 recall, not overall QWK. Grade 1 is microaneurysms
> only; an MA is 10–20 px. A model that silently skips grade 1 can still post a
> respectable QWK.

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

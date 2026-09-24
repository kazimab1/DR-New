# VERIFY-DR — Analysis plan for Phase 6b and Phase 7

> **Status: FINAL — marked final by the author on 2026-09-23, before any locked data
> was read.** Binding from the commit that first contains this line. D9–D11 (§12) are
> recorded in `PREREGISTRATION.md`. From here, changes follow the pre-registration's
> stopping rule (§9 there): dated deviations, never silent edits.
>
> **Finalising commit:** `1f2f9c1` (full: `1f2f9c15c565d64024006ad45d75495713c2fe51`) — recorded one commit later,
> because a commit cannot contain its own hash; the freeze's was recorded the same way.
>
> Drafted 2026-09-22. Finalising changed no choice in it: only this status, and three
> sentences whose tense or wording would otherwise have been wrong (§5.4, §10, and the
> list at the end).

---

## 0. Why this exists

`PREREGISTRATION.md` froze the *procedures* — which arms, which outcomes, fit on the
internal calibration split, three seeds, bootstrap intervals. `docs/03` gives the
*formulas*. Neither settles every choice needed to turn a formula into a number:
what `|grade_M1 − evidence_grade_M3|` means when M3 cannot output grades 3–4, which
embeddings the OOD score is measured against, how a tie is broken on a
coverage–accuracy curve. Each of those, settled after the labels are seen, is a fork
that could be taken knowing the answer.

This document settles all of them now, while no locked label has been read. The code
for Phase 6b and Phase 7 is then written *to this plan*, tested on internal data, and
run on the locked sets once.

## 1. How every open choice was resolved

In this order, stopping at the first that applies:

1. **The literal text** of `PREREGISTRATION.md` or `frozen_config.yaml`, where it is
   unambiguous.
2. **Existing code and documentation**, where the frozen text is silent.
3. **Otherwise, the reading that does not favour the project's own hypotheses.**

And one rule without exceptions: nothing in this plan is fitted on, tuned against, or
chosen after seeing any label from the EyePACS test split, APTOS or Messidor-2.

---

## 2. Data roles

| Split | Images | Role | Labels read |
|---|---|---|---|
| EyePACS train | 59,842 (`_full`) / 72,364 (`_ddr_full`) | M1 training (done); OOD reference sample | yes — internal |
| EyePACS val | 7,040 | **development and debugging only; nothing from it is reported** | yes — internal |
| EyePACS calibration | 3,512 | fits T, the OOD scale and threshold, τ_conf, the gate ratio r | yes — internal |
| EyePACS test | 17,615 | in-domain evaluation (D1, E1–E3, F1, F3–F5) — custom split (D8) | **once**, at the unblinding |
| APTOS 2019 train | 3,662 | external evaluation | **once**, at the unblinding |
| Messidor-2 | ~1,744 gradable | external evaluation | **once**, at the unblinding |

APTOS and Messidor-2 *images* are also read, unlabelled, by the Saerens–Decock step
(D3, D4) — the exception `PREREGISTRATION.md` §4 already permits.

## 3. Models and the one grade that is scored

- **M1:** the six Phase 6a runs, each at its `best.pt`. **Primary variant
  `eyepacs_full`** (decides H1, H1′, H2); **replication variant
  `eyepacs_ddr_full`** (reported beside it with the same rules; decides nothing on
  its own except as H3's second arm). `eyepacs_full` is primary because it is the
  base recipe and `_ddr_full` the ablation.
- **M2:** the C2 checkpoint, threshold 0.5, components ≥ 4 px (`frozen_config`).
- **M3:** count-only (D1): evidence grades 0–2, `max_excludable_grade` = 2.

**M1's predicted grade ŷ is the CORAL threshold count on the raw logits — the same
decision rule every Stage B and Phase 6a number used — and it is identical in every
arm and at every calibration stage.** Calibration changes probabilities, never ŷ. So
every arm scores the same predictions at 100% coverage; arms differ only in which
cases they defer.

---

## 4. Calibration — M4a, D1–D4, H2

### 4.1 A problem the freeze did not see: the models were trained on a uniform prior

The frozen sampler (`stratified_exposure`, no `--epoch-samples`) draws **every grade
equally often** (`dataset.py`, `make_sampler`). So M1's posteriors estimate
p(grade | image) under a *uniform* class prior, not EyePACS's ~73% grade 0.

Temperature scaling only sharpens or flattens a distribution; it cannot re-weight
classes. Fitted alone on the natural-prevalence calibration split, it would be asked
to absorb a prior mismatch it cannot represent, and the Saerens–Decock step (which
needs posteriors calibrated under a *known* prior) would inherit the error.

### 4.2 The pipeline

| Stage | What | Parameters |
|---|---|---|
| 0 | **Sampling-prior correction:** p_src(y\|x) ∝ p(y\|x) · π_src(y) / u(y), with u uniform and π_src the grade distribution of the **EyePACS training rows** (identical in both variants) | none fitted — π_src is a count |
| 1 | **Temperature:** cumulative logits divided by T before stage 0; T minimises NLL of p_src on the calibration split, bounded search over [0.05, 20] | T, one per model |
| 2 | **Saerens–Decock EM** (externals only): start π = π_src; iterate until max \|Δπ\| < 1e-6 or 1,000 iterations; p_tgt ∝ p_src · π_tgt / π_src | none fitted — reads unlabelled target images |

Stage 0 is parameter-free and standard for a resampled training set (Saerens et al.,
2002, §2). **It is new relative to the frozen procedure and is recorded as a deviation
(§12).** D1 reports the stage-1-only variant beside it so the effect is visible.

### 4.3 Confidence

**Confidence is p(ŷ | x): the calibrated probability of the grade actually predicted,
not the maximum probability.** `metrics.py` already enforces this for ECE and says why:
under the ordinal head ŷ need not be the argmax, and max(p) would measure the
calibration of a prediction the model never made. `docs/03`'s `1 − max(p_calibrated)`
is read as `1 − p_cal(ŷ | x)`; the two coincide whenever ŷ is the argmax.

ECE: 15 equal-width bins over p(ŷ | x), as `expected_calibration_error` implements.

### 4.4 The D experiments

| | Where | Compared | Metric |
|---|---|---|---|
| D1 | EyePACS test | raw · stage 1 only · stages 0+1 | ECE, NLL, Brier |
| D2 | APTOS, Messidor-2 | stages 0+1 transferred unchanged | ECE — expected to fail |
| D3 | APTOS, Messidor-2 | stage 2 (EM) vs oracle prior (true target prevalence, used for this bound only) | ECE |
| D4 | APTOS, Messidor-2 | EM prior from 50 / 200 / 500 / all unlabelled images, 20 seeded draws each, applied to all | ECE vs sample size |

---

## 5. The four signals — M4c

All computed per image and per model.

### 5.1 d_evidence — and M3's blind spot above grade 2

M3 cannot emit grades 3–4, and `frozen_config` sets `abstain_upward: true`: it does
not claim anything above `max_excludable_grade` = 2. The literal formula
`|ŷ − e|` would therefore count **every** M1 grade-3 or grade-4 call as a disagreement
of at least 1 — not because the evidence contradicts it, but because M3 cannot reach
those grades. That would defer the hardest grades by construction and credit the
disagreement arm for M3's blind spot.

**Pinned (rule 3 — the reading that does not favour H1):**

```
d_evidence = |ŷ − e|   if ŷ ≤ max_excludable_grade
           = 0         otherwise           # M3 abstains; it cannot contradict ŷ
```

Consequence, stated plainly: the disagreement arm never questions an M1 grade-3 or
grade-4 call on evidence grounds. Those grades are referable, so a human sees them
anyway.

### 5.2 d_conf

`1 − p_cal(ŷ | x)`, stages 0+1 in-domain and on the externals. (§4.3)

### 5.3 d_ood — "z_score(embedding, train_manifold)"

- **Embedding:** M1's penultimate layer — the 512-d `neck` output feeding every head.
- **Manifold:** 5,000 images drawn uniformly (seed 0) from that model's own training
  rows; class-conditional means with one shared covariance, Ledoit–Wolf shrinkage
  (Lee et al., 2018).
- **Distance:** minimum Mahalanobis distance to a class mean.
- **z-score:** against the mean and SD of the same distance on the calibration split.
- **τ_ood:** the 95th percentile of calibration z — "fixed coverage", as frozen: 5% of
  the calibration split is flagged by construction, and about 5% of in-domain cases.

### 5.4 d_faith — faithfulness as a randomisation test

`docs/03`: faithful := Δ_lesion > Δ_random + margin. The margin is unspecified, and
with one random draw a model that ignores lesions entirely passes about half the time.
**Pinned: the margin is replaced by a test.**

- **Regions:** every M2 component (all four channels, threshold 0.5, ≥ 4 px) — under
  count-only M3 the fired rule always cites every channel present. Each dilated by
  3 px; inpainted together with OpenCV Telea, radius 5 (`frozen_config`).
- **Controls:** K = 19 draws. Each draw moves every dilated component, shape intact, to
  a uniformly random position fully inside the field of view (max RGB > 10), overlapping
  neither the original lesions nor components already placed; up to 200 attempts per
  component; a draw fails if any component cannot be placed. Seeded by image ID.
- **Δ:** on M1's continuous expected grade E[y] = Σₖ σ(zₖ), raw logits (faithfulness is
  a property of the network's computation, not its calibration). Δ = E(original) −
  E(inpainted).
- **faithful** iff Δ_lesion exceeds **every** successful control Δ, with at least 15 of
  19 draws succeeding. If the network reacts to inpainting but not to *what* was
  inpainted, removing the lesions is just one more draw, and it comes out on top with
  probability ≤ 1/20 (≤ 1/16 at 15 draws).
- `d_faith` = 1 if determined unfaithful; **0** if faithful, if no lesion was detected
  (nothing cited, nothing to test), or if undetermined. Undetermined cases are counted
  and reported.

Cost: ~1 GPU-hour across all splits. K = 9 would have halved it at α = 0.10; K = 19 was
kept when this plan was finalised.

### 5.5 The disagreement score and its one parameter

```
disagreement = d_evidence + r · d_faith
```

`docs/03` writes `w1·d_evidence + w2·d_faith`, but a ranking is unchanged by scaling,
so only r = w2/w1 matters. With d_evidence ∈ {0,1,2} and d_faith ∈ {0,1}, only four rankings exist:
r = 0 ignores faithfulness, and every r in (0, 1), (1, 2) or (2, ∞) reproduces the
ranking of **0.5** (faithfulness breaks ties), **1.5** (worth more than one grade of
disagreement, less than two) or **2.5** (dominates). r = 1 and r = 2 only create ties.
So the grid is **r ∈ {0, 0.5, 1.5, 2.5}**. **r is chosen on the calibration split by
coverage–accuracy AUC; ties go to the smaller r.**

---

## 6. The five arms and how they are scored

### 6.1 Arms — each is a score; higher is deferred first

| Arm | Score |
|---|---|
| none | constant — every case tied |
| confidence | d_conf |
| OOD | d_ood |
| **disagreement** | d_evidence + r · d_faith (**the arm under test**) |
| combined | the `docs/03` decision policy's level, then d_conf within a level (below) |

**Combined**, pinned from the `docs/03` policy table:

- **level 2, DEFER:** d_evidence ≥ 2, or d_faith = 1, or z_ood ≥ τ_ood, or
  (e = 0 and d_conf ≥ τ_conf). `unobservable` is the same three findings for every
  image, so "unobservable non-empty" is always true and drops out; "d_conf high" is
  τ_conf = the 90th percentile of d_conf on the calibration split.
- **level 1, ADJACENT_GRADE_SET:** d_evidence = 1.
- **level 0, ACCEPT:** otherwise.
- **REACQUIRE cannot fire** — M0 was never built (§11, §12).

### 6.2 The coverage–accuracy curve

Sort by score, most trusted first. For k = 1…n accepted cases, accuracy(k) is the
exact-grade accuracy of ŷ among them; coverage = k/n. **AUC = the mean of accuracy(k)
over k.** Exact-grade because the primary outcome says "accuracy", and referable-DR
performance is F3's separate question.

**Ties are resolved by expectation**, computed exactly: a tied block is accepted as if
in uniformly random order. Ties are *not* broken by confidence — that would fold the
baseline into the arm it is being compared with. The disagreement score has few
distinct values, so its curve has flat stretches; that coarseness is a property of the
method as specified, and the comparison reports it rather than hiding it. The "none"
arm's AUC is therefore exactly the overall accuracy.

Secondary points: accuracy at 80% and 90% coverage, same expectation.

### 6.3 F3, F4, F5

- **F3:** among accepted cases at 80% and 90% coverage, rDR sensitivity and specificity
  of ŷ ≥ 2 against true grade ≥ 2 (ratios of expected counts), plus the fraction of
  truly referable cases that were deferred to a human rather than accepted.
- **F4:** binary `[d_evidence > 0]` vs magnitude `d_evidence`, **reported, not
  selected**. The frozen primary form is magnitude (§5.5). The third form,
  "magnitude + unobservable set", is identical to magnitude here because the set never
  varies, and is reported as such.
- **F5:** fraction of cases and accuracy per action; for ADJACENT_GRADE_SET, "accurate"
  means the true grade lies in {ŷ − 1, ŷ, ŷ + 1}.

## 7. Faithfulness experiments — E1–E3

On the EyePACS test split and both externals, per model: the distribution of
Δ_lesion (E1), against the mean of the 19 control Δs per image (E2 — the mandatory
control), stratified by true grade (E3). Images with no detected lesion are excluded
and counted.

---

## 8. When a hypothesis counts as supported

`PREREGISTRATION.md` §7: three seeds, paired across seeds; 95% bootstrap intervals
from 2,000 resamples over test images; no difference claimed smaller than the
seed-to-seed spread. Made operational as **one claim rule, applied identically to
every hypothesis**. For an effect Δₛ measured on seed s:

> **(a) direction** — Δₛ > 0 for all three seeds;
> **(b) precision** — each seed's paired bootstrap 95% interval excludes 0;
> **(c) stability** — the mean of the Δₛ exceeds their spread (max − min).
>
> Supported only if all three hold. Otherwise *not supported* — which, for every
> hypothesis below, is what the protocol calls falsified.

Bootstrap: seed 42; one set of resample indices per dataset, shared by every arm and
model, so every comparison is paired.

| | Δₛ | Where | Decided on |
|---|---|---|---|
| **H1** (F1) | AUC(disagreement) − AUC(confidence) | EyePACS test | primary variant |
| **H1′** (F2) — the thesis | the same | **APTOS and Messidor-2; must hold on both** | primary variant |
| **H2** (D3) | ECE(stages 0+1) − ECE(after EM) | APTOS and Messidor-2; must hold on both | primary variant |
| **H3** (B7) | QWK(`_ddr_full`, seed s) − QWK(`_full`, seed s) | APTOS and Messidor-2; must hold on both | the seed pairing |

Each dataset is also reported on its own. The replication variant is run through H1,
H1′ and H2 with the same rule and reported beside the primary; it neither rescues nor sinks the primary result.

Seed 43 stopped early in both variants (register, Phase 6a), which widens every
spread. Criterion (c) therefore sets a higher bar than it would have. That is the
honest consequence of the frozen recipe and is not adjusted for.

---

## 9. What the single pass records

One script, run once over each locked set. **It writes no label.** It reads a manifest and
drops the grade column before anything else happens. A static test asserts that the
script touches the grade column only to drop it, and that no output column carries it.

| Per image | Per image × model |
|---|---|
| image ID, dataset, path | cumulative logits z₀…z₃, ŷ |
| M2: component count and area per channel | rDR and VTDR logits |
| M3: evidence grade, max_excludable, rule fired | Mahalanobis distance (§5.3) |
| faithfulness status, control draws that succeeded | E[y] original, lesion-inpainted, and each control |

Probabilities, calibration and every score are computed downstream from these columns
and the frozen fitted parameters (§10, step 3), so a calibration bug found later does
not require reading the locked images again.

## 10. Order of execution

1. **Internal pass** over the 5,000-image training references, the calibration split
   and the val split, recording everything in §9.
2. **Fit** on the calibration split: T, the OOD statistics and τ_ood, τ_conf, r — per
   model. **Rehearse every table and figure end to end on val**, labelled as rehearsal.
3. **Commit** the analysis code and `fitted_params.json`; record the commit hash in
   `PREREGISTRATION.md`. This is the last commit before the locked pass, so the code and
   every fitted value that produce the results are fixed before any locked image or label
   is read.
4. **Locked pass:** the same script over the EyePACS test split, APTOS and Messidor-2.
   Label-free. Published as a Kaggle dataset.
5. **Unblinding:** labels joined in one step; every outcome in §§4–8 computed and
   written. Once.

Estimated GPU: ~30 minutes for steps 1–2, ~1–1.5 hours for step 4, dominated by the
19 faithfulness controls.

---

## 11. What this analysis cannot do

- **No REACQUIRE, no G2.** M0, the quality head, was never built. Every image is treated
  as gradable.
- **No evidence above grade 2** (D1), so disagreement cannot question M1's grade-3/4
  calls (§5.1).
- **Bootstrap over images, as registered.** Messidor-2's eyes come in patient pairs, so
  its intervals are somewhat too narrow.
- **The in-domain test split is custom** (D8) and never compared with the leaderboard.
- **Wider seed spreads** from seed 43's early stops (§8).
- **The EyePACS cache is 693 images short**, non-randomly by grade (A0).

## 12. Deviations recorded with this plan

Recorded in `PREREGISTRATION.md` on 2026-09-23.

| | Change | Why |
|---|---|---|
| D9 | This analysis plan added | pins the implementation choices §§5–7 of the pre-registration left open; changes no registered outcome |
| D10 | REACQUIRE removed from the triage actions; G2 not run | M0 was never built |
| D11 | Stage-0 sampling-prior correction added before temperature scaling | the frozen sampler trains on a uniform prior (§4.1); without it EM's input is miscalibrated |

---

## The choices that matter most

Five choices here shape the headline result more than the rest. Each is argued above,
and each was settled when this plan was marked final.

1. **§5.1 — d_evidence is 0 for M1 grades above 2.** The literal formula would defer
   them all; this plan says M3's blind spot is not evidence.
2. **§4.1 — stage-0 prior correction.** A change to the frozen calibration procedure,
   recorded as D11.
3. **§5.4 — faithfulness as a 19-draw randomisation test**, replacing an unspecified
   margin. About one GPU-hour.
4. **§6.2 — ties resolved by expectation, never by confidence.** Keeps the arm under
   test pure; it is also why its curve will look coarse.
5. **§8 — the claim rule.** Strict by design: all three seeds, every interval, and
   larger than its own spread.

---

## Addendum · 2026-09-24 · the amended analysis (D12, D13)

Everything above is unchanged and defines the **registered, primary** analysis. This
addendum specifies a second analysis, recorded as D12 and D13 in `PREREGISTRATION.md`.
It was fixed after the step-2 rehearsal on val and before any locked image or label was
read. It is reported beside the primary, labelled as a deviation analysis, under the same
§8 claim rule. One round: after its rehearsal, the parameters are committed and the
locked pass runs, whatever that rehearsal shows.

### A.1 · D12 — M3's image-level operating point

- A lesion type (microaneurysm, haemorrhage, hard exudate, soft exudate) is **present**
  iff it has at least one component of ≥ 4 px (the frozen rule) **and** its total
  predicted area, the pass's `area_<type>` (pixels at probability ≥ 0.5), is at least
  a_type.
- a_type ∈ {4, 16, 64, 256, 1024} px for each type: 625 combinations. All four at 4 px is
  the frozen rule.
- Chosen on the **calibration split** by **M3's QWK against the true grade** (five
  grades, as C4). QWKs within 1e-12 are ties. A tie goes to the smallest sum of grid
  positions, then lexicographically in the order above. **M1 is not consulted.**
- M3's ladder (R1, R2, R3, R3\*) is unchanged; only "present" changes. The amended
  evidence grade e′ replaces e in d_evidence and in the combined arm's e = 0 clause.
- d_faith′ = d_faith where e′ ≥ 1, else 0: §5.4's rule that where nothing is cited,
  nothing is tested. The regions tested are M2's detections as frozen, so no image is
  read again.
- r′ is re-chosen on calibration by §5.5's rule for d_evidence′ + r′ · d_faith′.

### A.2 · D13 — bias-corrected temperature scaling (Alexandari et al., 2020)

- q(y | x) ∝ p_T(y | x) · exp(b_y), where p_T are the CORAL probabilities of z / T and
  b_0 = 0. T and b_1…b_4 minimise the NLL of the true grades on the **calibration
  split**. T is searched as in §4.2 (log grid on [0.05, 20], refined by golden section),
  and b is solved exactly at each T by Newton's method; the NLL is convex in b.
- At the optimum, the sum of q(y | x) over calibration equals the calibration count of
  every grade: EM's premise holds on calibration by construction.
- EM (§4.2, stage 2) runs on q with **reference prior π_cal**, the calibration split's
  grade mix, and starts there. Same tolerance and cap.
- The amended D1–D4 and H2 use q. H2′'s effect is ECE(q) − ECE(q after EM), with EM
  re-run inside every resample, as in the primary. **d_conf, τ_conf, and the confidence
  and combined arms keep stages 0+1**, so H1's baseline is the registered one.

### A.3 · What the amendment does not touch

M1 and ŷ; M2's detections, the faithfulness regions and the 19 controls; the OOD
statistics and τ_ood; τ_conf; the arms' definitions; ties by expectation; the bootstrap
(2,000, seed 42, indices shared by every arm, model and analysis); the claim rule; the
data roles; the order of execution in §10.


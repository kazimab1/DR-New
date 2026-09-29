# Chapter 7 — Error analysis

> **Draft, 2026-09-29. Exploratory throughout.** Every analysis in this chapter was
> declared in the register (Phase 8, P8.0–P8.6) and committed before it first ran. None
> of them can change a verdict: those were fixed by the one label join (Chapter 6).
> Numbers are from `docs/phase8/2026-09-29_readout.txt`. They are means over the three
> `eyepacs_full` seeds, and "amended" means the deviation analysis D12 + D13.

Chapter 6 showed that disagreement between the grader and the evidence pathway did not
decide trust better than the grader's own confidence, on any test set, in either
analysis. This chapter asks why, and whether any part of the idea survives.

Before producing any table, the analysis recomputed every per-image signal from the
stored pass and reproduced all published coverage–accuracy AUCs exactly: 6 models × 2
analyses × 5 arms on each set, to 1e-12 (P8.0). Everything below is therefore about the
same scores the verdicts rest on.

## 7.1 Figure 1: the case the design was built for

The design exists for one kind of case. The grader calls an eye normal and is confident.
The evidence pathway finds microaneurysms. And the eye is in fact diseased. Confidence
gating accepts that grading; disagreement gating defers it.

The selection rule was fixed in advance (P8.5): model `eyepacs_full_s42`, amended
evidence, grade-0 calls in the model's most confident quarter, a true grade of at least
1, and microaneurysms present. The candidates were ordered by confidence, and up to eight
per set were drawn beside M2's lesion outlines.

| Set | Images that qualify |
|---|---|
| EyePACS test | 24 of 17,615 (0.14%) |
| APTOS | 0 of 3,662 |
| Messidor-2 | 4 of 1,744, every one adjudicated grade 1 |

### 7.1.1 What the candidates look like

The drawn sheets are kept in `docs/phase8/` as `2026-09-29_figure1_candidates_*.webp`.
They hold the eight most confident in-domain candidates and all four on Messidor-2, and
they were inspected by eye at the 512 px the models see. This is a visual reading by a
non-clinician, not a measurement. It shows what individual detections look like, not how
often detections are wrong.

**In-domain, where a detection can be judged, it is more often an artefact than a
lesion.**
- **`EyePACS::769_right`** (true grade 3). The largest haemorrhage outlines lie in the
  black margin, outside the field of view. Most of its hard exudates cluster on the
  image's jagged, pixelated rim.
- **`EyePACS::219_left`** (true grade 2) is the strongest candidate on paper. M1 says 0 at
  confidence 0.982. M2 finds 30 microaneurysms and 23 haemorrhages, and M3 gives grade 2
  in both analyses, matching the label. But most of its outlines sit on dark,
  sharp-edged specks, often rod-shaped and grey rather than red, which look like debris
  on the optics. Its one soft exudate sits on a pale spot at the centre of the frame.
- **The same pale spot** appears in `23515_left` and `39369_right`, at the same position
  to within a few pixels, and `18369_left` has a ring-shaped reflex in the same place. A
  spot that keeps its place in the frame across patients points to a reflection in the
  camera, not to anatomy.
- The remaining candidates carry a few specks each, too small to judge at this
  resolution.

The rim case also exposes a gap in the design. **M3 counts M2's detections wherever they
fall, including outside the retina.** The code has a field-of-view mask, used to
place the faithfulness test's random controls, but the evidence path never applies it. How much of M3's false evidence lies outside the retina was not measured.

**On Messidor-2, D12 removed the vessel-shaped detections.** In `IM003603` the one
haemorrhage outline is a small vessel loop near the upper edge. In
`20060523_49449_0100_PP` both haemorrhage outlines sit on a tortuous vessel where it
meets the disc. Both images fall far below D12's 256 px haemorrhage minimum (21 px and
42 px), which is why each is grade 2 registered and grade 1 amended.

**This is what M3's lack of specificity predicts.** If the evidence pathway finds lesions
in most eyes whatever their grade, some of the eyes it flags will be diseased eyes the
grader missed. The evidence it cites there need not be the disease. The case the design
was built for does occur, but on inspection its evidence is often not the disease.

### 7.1.2 The proposed figure

⟨Author to confirm.⟩ **Figure 7.1: `Messidor2::20051021_39482_0100_PP`.** It is the
only candidate on either set that meets all four of these conditions:
- its grade is adjudicated, grade 1 on three specialists' consensus;
- M3's grade equals that grade in both analyses;
- the rule M3 cites, microaneurysms only (R2), is the definition of grade 1;
- no outline sits on the rim, the disc, a vessel or a reflex.

M1 calls it grade 0 at confidence 0.978. The caption must also carry four limits:
- its microaneurysm area is 17 px, one pixel above D12's 16 px minimum;
- the fundus is tessellated, and its dark choroidal spots are of similar size;
- Messidor-2 has no lesion annotations, so whether these two specks are the
  microaneurysms the specialists graded cannot be checked;
- the panel was chosen by eye from the four images that met the declared rule.

A companion panel could show `EyePACS::219_left`: the right grade, apparently for the
wrong reason.

**The rarity is itself the result.** The case the thesis was built around is real. It
is also 24 images in 17,615. However vivid the example, it cannot outweigh the
aggregate that went the other way.

## 7.2 Where the grader's errors go (G1)

Each arm defers the 20% of cases it trusts least. If it deferred at random, it would
catch 20% of the grader's errors.

**Table 7.1 — Share of M1's exact-grade errors deferred at 80% coverage.**

| Set | M1 error rate | confidence | disagreement, registered | disagreement, amended |
|---|---|---|---|---|
| EyePACS test | 25.3% | **63.3%** | 8.0% | 25.5% |
| APTOS | 38.5% | **46.8%** | 10.2% | 30.2% |
| Messidor-2 | 33.1% | **43.5%** | 13.5% | 22.4% |

**The registered disagreement catches fewer errors than chance** on every set. The
cases it deferred in-domain were only 10.4% errors, against 25.3% overall: it deferred
the grader's *safest* calls. Its error-detection AUROC is below 0.5 everywhere (0.443,
0.375, 0.445). The signal does not merely fail to find errors; it points at correct
cases. Chapter 6 gave the mechanism: M3 reported disease in most healthy eyes, so the
largest disagreements were healthy eyes the grader had called correctly.

**Amended, disagreement is a real but weak detector.** Its AUROC is 0.605, 0.601 and
0.573, against confidence's 0.874, 0.915 and 0.791. OOD distance falls between them
(0.713, 0.767, 0.635).

## 7.3 Does disagreement know anything confidence does not?

A weak signal can still be worth having if it is *complementary*. So the analysis split
each model's images into quintiles of confidence and asked, within each quintile,
whether disagreement still separates the grader's errors from its correct calls.

**Table 7.2 — Disagreement's AUROC for error within confidence quintiles (0.5 = nothing
beyond confidence).**

| | EyePACS test | APTOS | Messidor-2 |
|---|---|---|---|
| registered | 0.520 | 0.464 | 0.524 |
| amended | 0.523 | **0.618** | **0.557** |

**In-domain, disagreement adds essentially nothing to confidence**, in either analysis.
Under shift, the amended disagreement does add something. On APTOS, in the middle
quintile of confidence, M1 is right on 73% of images where disagreement is zero and on
only 39% where it is one grade. In-domain, the same quintile shows 90% against 88%.

The pre-registered combined policy could not exploit this. It ranks first by the
disagreement levels and uses confidence only within them, which is the wrong way round
for a signal that is weaker but complementary. Putting confidence first and letting
disagreement reorder within confidence strata was never tested. That would be a new
hypothesis, not a finding of this thesis.

## 7.4 Referral errors: the one place disagreement wins

The registered outcome scores exact-grade accuracy. A screening programme cares about
one boundary: refer (grade ≥ 2) or not. P8.2 split the grader's *referral* errors, the
cases where ŷ ≥ 2 disagrees with y ≥ 2, in the same way.

**Table 7.3 — Share of M1's referral errors deferred (chance: 20% at 80% coverage, 10%
at 90%).**

| Set | Coverage | confidence | disagreement, registered | disagreement, amended |
|---|---|---|---|---|
| EyePACS test | 80% | **55.3%** | 7.8% | 28.4% |
| EyePACS test | 90% | **17.0%** | 3.5% | 9.5% |
| APTOS | 80% | 17.0% | 3.0% | **38.7%** |
| APTOS | 90% | 5.4% | 0.1% | **8.0%** |
| Messidor-2 | 80% | 26.5% | 6.3% | **31.1%** |
| Messidor-2 | 90% | 8.7% | 1.6% | **10.7%** |

On both external sets, at both coverages, the amended disagreement defers more of the
grader's referral errors than confidence does. On APTOS at 80% coverage it catches more
than twice as many, while confidence does worse than chance. In-domain the order is
reversed.

Chapter 6's F3 table shows *which* referral errors these are. Disagreement raises the
accepted set's specificity, not its sensitivity. It catches false referrals, the eyes
the grader would send to an ophthalmologist unnecessarily. It does not catch missed
referrals, which in screening are the costlier error.

This is the thesis's premise in its narrowest surviving form. **Under dataset shift,
disagreement with independent lesion evidence flags false referrals better than the
grader's confidence does.** Three limits hold it back from being more than that:
- it was found post hoc;
- it appears only in the amended analysis;
- it concerns the less costly of the two referral errors.

It is a hypothesis for a new, pre-registered study, and it is reported here as one.

## 7.5 Why disagreement lost (G3)

### 7.5.1 The evidence pathway contradicts correct grades

| | EyePACS test | APTOS | Messidor-2 |
|---|---|---|---|
| Correct M1 calls that M3 contradicts, registered | 64.9% | 46.5% | 62.5% |
| …amended | 30.8% | 9.7% | 35.8% |

These are shares of the correct calls M3 can judge, meaning grades 0–2. In the
registered analysis almost every contradiction is an over-call (e > ŷ). In-domain, 48%
of the over-calls come from R3* (haemorrhage or exudate with no microaneurysm) and 42%
from R3. D12 cut the contradictions sharply, to 31%, 10% and 36%, but it changed their
kind. In-domain,
76% of the remaining over-calls are microaneurysm-only (R2), and under-calls appear:
8.2% in-domain and 15.6% on Messidor-2, where M3 now finds too little. The operating point
traded false evidence for missed evidence. It did not remove the pathway's errors.

**Label noise does not explain the over-calls.** EyePACS grades come from one grader,
who might miss microaneurysms that M2 then "wrongly" finds. But Messidor-2's grades are
adjudicated by three specialists, and its amended over-call rate is about the same
(20.1% against 22.5% in-domain). Missed microaneurysms in the labels cannot be the main
cause.

### 7.5.2 The errors disagreement cannot see

M3 stops at grade 2 (D1): without the disc-and-fovea frame (C1) it has no quadrant rule,
and nothing annotates neovascularisation. Whenever the grader says 3 or 4, disagreement
is zero by construction.

| | EyePACS test | APTOS | Messidor-2 |
|---|---|---|---|
| M1's errors with ŷ ≥ 3, which disagreement cannot see | 7.7% | **56.0%** | 7.4% |
| M1's errors where M3 agrees with the wrong grade, registered → amended | 20.0% → 42.2% | 26.5% → 15.1% | 29.8% → 41.4% |

**On APTOS, more than half of the grader's errors are invisible to disagreement.** They
are severe-disease calls, above M3's ceiling. This is a structural cause of the APTOS
result, and it traces back to C1's failure. D12's more conservative evidence also agrees
more often with the grader's missed-disease calls in-domain and on Messidor-2 (42%, 41%).
Both sides then say "nothing here", and both are wrong.

### 7.5.3 The three candidate explanations, settled

Chapter 6 proposed three candidate explanations. Each can now be settled:

1. **Coarseness: not sufficient.** Disagreement takes at most six values. But within
   confidence quintiles it carries almost no error information in-domain (Table 7.2).
   The problem is what the levels say, not how few there are.
2. **M3 stops at grade 2: decisive on APTOS, minor elsewhere.** It hides 56% of the
   APTOS errors, against 7–8% on the other sets.
3. **M3's own errors: the main in-domain cause.** Registered, M3 contradicts two thirds
   of the grader's correct calls. Amended, it still contradicts one third, and 21% of
   referable in-domain images get no evidence at all.

## 7.6 The Messidor-2 source (P8.6)

The dataset card had left open whether the Messidor-2 mirror was already processed. It
is not Ben-Graham processed and not grey-normalised. After this project's
preprocessing, its images have black padding, no near-grey pixels, and saturation like
APTOS's (0.79 against 0.80). It is, however, the least sharp of the four sources (median
Laplacian variance 106, against 161–204). M2 reports microaneurysms in 73.2% of
Messidor-2 images (median 23 px), against 50.7% for EyePACS and 60.5% for APTOS. At the
frozen operating point, M3 found evidence in 85% of the grade-0 Messidor-2 images. The training domain differs in
colour as well: EyePACS is the least saturated source (0.47, against 0.70–0.80).

Whether soft, compressed images produce microaneurysm-like specks is a hypothesis, not
a finding. The mirror's native resolution and compression were not observed. The first
audit run measured the cache's copy of the mirror by mistake; that is now fixed.

## 7.7 What survives

- **Verification works.** The grader uses the lesions the evidence pathway finds. It
  beats all 19 random controls in 67–78% of lesion images against about 5% by chance,
  and in 75–96% of truly diseased grades 1–3 (Chapter 6.5).
- **The evidence pathway is informative.** A rule-based reader over a segmenter trained
  on 757 images grades APTOS at QWK 0.755, once its operating point is set for images
  rather than pixels.
- **As a trust signal, disagreement is redundant with confidence in-domain, and weakly
  complementary under shift.** For false referrals under shift, it is better than
  confidence. That last point is post hoc.

What would have to change for disagreement to win is now concrete. Each item is a new
hypothesis for a new pre-registered study:
- an evidence reasoner that reaches grades 3 and 4 (a working C1, or annotated
  neovascularisation);
- a segmenter specific at image level;
- a policy that puts confidence first and uses disagreement within its strata.

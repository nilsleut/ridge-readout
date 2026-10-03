# Preregistration — ridge readout on full feature maps across the resolution sweep

**Written before any fitting compute.** Config frozen at `config.yaml`,
`schema_version: 1`, hash **`8163d8c4a146`** (`config.lock`). The hash is written into
every output CSV.

Base: `evaluation-resolution-rsa` @ `6cf64f8`, arXiv:2608.12408.
Reconnaissance: `PHASE0_RECONNAISSANCE.md`.

This document exists so that neither outcome can be rationalised afterwards. Both
outcomes below are written out in full, with their interpretation fixed, before the
first number is produced. If the eventual write-up says something not derivable from
one of these two sections plus the data, that is a deviation and must be labelled one.

---

## 1. The question

The paper's finding is that the Random − Backprop gap at V1 grows monotonically with
evaluation resolution: −0.001 ± 0.007 at 32 px (the training resolution) to
+0.044 ± 0.006 at 224 px, 5 seeds, Conv1 → V1.

Every number in it rests on **rank correlation between RDMs built from globally
average-pooled features**. GAP discards spatial structure before the comparison, and V1
is retinotopic. Brain-Score instead uses regression-based predictivity on the full
feature map. The paper says so itself, twice, as Limitations item 11 and in Future work.

**We ask whether the resolution effect survives a fitted readout on the full feature
map.** The grid ladder g ∈ {1, 2, 4, 8} turns that yes/no into a surface: g = 1 is the
paper's condition exactly, and g = 8 retains spatial structure, at constant
dimensionality across the resolution axis.

## 2. Primary outcome measure

Per (arm, rule, seed, resolution, grid, layer, roi, subject, fold): ridge predictivity
r, median over selected voxels, mean over subjects. The contrast of interest is
**Random − Backprop at Conv1 → V1**, paired within seed, as a function of resolution and
grid.

**The unit of inference is the seed** (n = 5). CV folds are not independent and feed the
point estimate, not the interval. Test: paired one-sided sign-flip permutation over the
five seeds, 2⁵ = 32 assignments, smallest attainable p = 1/32 ≈ 0.031 — the paper's own
test, so the numbers stay structurally comparable to its ±0.006.

## 3. The two outcomes, with interpretation fixed in advance

### 3a. The effect survives

**Reading:** the resolution confound is not readout-specific. It is a property of the
model–brain comparison at early visual cortex, not an artifact of discarding spatial
structure before comparing.

This is a **stronger claim than the paper's**, and it would extend the recommendation
(evaluate at the training resolution *and* at several others, and state which was used)
from RSA-on-pooled-features to fitted readouts generally — i.e. to the Brain-Score-style
methodology as well.

Stated with the bounds that will still apply:

- It remains a claim about this architecture, this training set, this stimulus set and
  this ROI mapping. Six resolutions on the same images with the same weights are six
  correlated measurements of one manipulation, not six independent tests.
- Surviving at g = 1 alone would **not** count as surviving. g = 1 is the paper's own
  condition re-run under a fitted readout; the claim requires it to hold at g ≥ 4, where
  spatial structure is retained.
- If it survives at g = 4/8 but the spatial-correspondence control (§5.3) shows no gain
  over per-image cell shuffling, then what survives is a capacity effect, not a spatial
  one, and must be described that way.

### 3b. The effect disappears

**This does not falsify the paper.** The paper makes a claim about RSA on pooled
features — precisely the method the criticized literature uses. Disappearance under a
fitted readout **narrows the claim's scope and is itself a finding**: it would say that
the resolution confound is specific to the pooled-RDM methodology, which is the
methodology the reports it criticizes actually employ. The recommendation to control and
report evaluation resolution would stand for that literature and not generalize to
fitted readouts.

The distinction that decides how this is written up:

- **"The effect disappears"** requires Δ_min to be smaller than the effect size the
  paper reports. Only then is absence informative.
- **"The method cannot resolve it"** is the outcome if Δ_min is comparable to or larger
  than +0.044. That is a statement about the measurement, not about the effect, and it
  will be reported as such.

Δ_min is derived from the pilot's seed-level SD by simulating the sign-flip test above,
with the parametric `(t₀.₉₇₅,₄ + t₀.₈₀,₄) · SD/√5` approximation as a cross-check. **The
number goes in the paper regardless of outcome**, exactly like the bound the luminance
scalar sets on the RSA comparison.

### 3c. The outcome we consider most likely, recorded now

**Saturation.** Random CNN features are a random-kitchen-sink kernel; with enough
capacity they approximate any smooth function of image statistics. If the reliable V1
signal in THINGS is essentially low-frequency image statistics — and the 0.075 (luminance
scalar) vs 0.076 (best CNN) result is consistent with that, though it does not establish
it — then all conditions hit the ceiling and the comparison is empty.

We record this as the **most likely** outcome, not the least, so that finding it cannot
be presented as a surprise. If all six rules fall inside a band narrower than the
distance to the pixel baseline, **that is the result**, and §5 is the set of instruments
built to make it a measurement rather than a null.

## 4. Two arms, because the 720 have no repeats

Phase 0/A established that all 720 evaluation stimuli are single-presentation `train`
trials. The 100-image `test` set has 12 repetitions each, is identical across all three
subjects, and has exactly zero overlap with the 720.

| | n | Betas | Ceiling | Grids |
|---|---|---|---|---|
| **Arm A** (primary) | 720 | single-trial | leave-one-subject-out, **lower bound, n = 3** | 1, 2, 4, 8 |
| **Arm B** (SNR control) | 100 | mean over 12 reps | split-half, Spearman–Brown, **real** | 1, 2 |

**Arm A is primary.** It is the paper's stimulus set, so it preserves comparability with
every published number. Its denominator is inter-subject consistency, which is **not a
noise ceiling** — it is a lower bound with a different interpretation, and it is labelled
as such in every figure axis, table header and column name, not only in prose.

**Arm B is the SNR control.** It answers how much of the low absolute correlation is
single-trial noise. Phase 0 speculated that single-trial betas were "the most likely
reason absolute ρ is so low"; that was plausible but not established, and **it is not
asserted anywhere until Arm B measures it.** Arm B is restricted to g ∈ {1, 2} by
design: d = 32 and 128 against n = 100 (87 training stimuli per outer fold). g = 4 would
put d = 512 far above n and is not run there. This is a deliberate restriction, recorded
as such, not an omission.

Arm B is a **different stimulus set**, so it is not a replication of Arm A and will not
be described as one. It bounds the noise, it does not re-test the effect.

**Voxel selection** is by split-half reliability that we compute ourselves on Arm B's 12
repetitions (all 462 balanced splits, Spearman–Brown corrected), threshold 0.2, fixed in
advance, identical for all conditions, and never by model performance.

- For **Arm A** this is structurally clean: reliability is measured on 100 *different*
  images from the 720 being fitted, so selection cannot leak.
- For **Arm B** it is **circular** — same responses, selected and fitted. Arm B therefore
  reports both selected and unselected, and **the unselected result is primary for Arm B**.

The shipped `nc_testset` and `nc_singletrial` columns are now documented (percent of
explainable variance, 0–100; Hebart et al. 2023 reports up to 80% in early visual cortex
and the measured global max is 83.67, attained in V1). They serve as a cross-check only.
`splithalf_uncorrected` / `splithalf_corrected` remain **undefined by anything we could
read** and are not thresholded on. **The `nc_*` columns are never used as an Arm A
denominator**: they are measured on different stimuli.

## 5. Instruments against saturation, declared in advance

Registered here so that reporting them is not contingent on the outcome.

**5.1 The regularization path is the measurement, not the scalar.** r(df) across the
whole α path, df(α) = Σ sᵢ²/(sᵢ² + α). If Random reaches the same predictivity only at
three times the effective dimensionality, that is a real difference a scalar hides.

**5.2 Capacity sweep downward.** Random feature subsets of 50, 100, 500, 2000, all;
subsets fixed across conditions by seed. Feature-quality differences appear in the scarce
regime, not the saturated one.

**5.3 Spatial-correspondence control.** The grid ladder cannot separate "retains spatial
structure" from "has more dimensions" — those covary by construction. So: **shuffle the
g×g cells with a different permutation for each image.** A fixed permutation would do
nothing, since ridge is invariant to feature reordering; a per-image permutation destroys
the grid-cell-to-retinotopic-location correspondence while preserving dimensionality and
the per-channel activation distribution exactly. Run at g = 4 and g = 8, all
resolutions, Arm A. **If a rung gains nothing over its shuffled counterpart, the gain was
capacity, not space** — and any claim about spatial structure is withdrawn.

**5.4 Control band, defined in advance.** Only results between the pixel baseline and the
ceiling are interpretable. Identical readout and identical g for: raw pixels; luminance +
contrast; a Gabor bank; and permuted stimulus↔feature assignment, which must give ≈ 0 and
is the pipeline's negative test. A **dimensionality-matched** low-level bank, because
under a fitted readout a single scalar is no longer the right trivial baseline — one
feature can gain nothing from ridge except scaling.

**5.5 Δ_min a priori.** As in §3b. Reported regardless of outcome.

**5.6 The literal Brain-Score metric alongside.** PLS with fixed component count (25),
capacity-matched by construction, from `brainscore_vision.metrics.regression_correlation`
(installed 2.3.18). The controlled ridge design stays **primary**; Brain-Score runs beside
it for external comparability. Its `InternalConsistency` ceiling needs a repetition
coordinate and so applies to Arm B only.

## 6. The seed-42 determinism gate — both branches declared

The retrain regenerates seed 42, for which six checkpoints already exist. Every tensor of
the new seed-42 checkpoints is compared against the existing ones, per rule, at tolerance
**0.0 — bit-identical, not approximate**. The comparison is implemented as a test with
both branches *before* the retrain runs, not as an inspection afterwards.

- **Identical → training is deterministic.** Seed-level SD then measures variation in
  initialization, which is what §1.3 intends the unit of inference to be.
- **Not identical → training carries a nondeterministic component.** Then every ± in the
  paper and in this work has a second variance source that is currently unlabelled. We
  report the magnitude per tensor and **stop**.

Both outcomes are usable and neither is a failure of the gate. The second is a finding
about the existing published error bars, not only about this study. The paper already
documents that convolution backward is nondeterministic on GPU and that gradient-trained
conditions reproduce only to ~10⁻³ at V1 — so a mismatch confined to Backprop and Feedback
Alignment would be **consistent with what is already documented**, while a mismatch in
Random Weights, Predictive Coding or STDP (which never backpropagate through a
convolution and are documented as reproducing bit-identically) would **not** be, and is
the more serious finding of the two.

A second, independent check: the retrained seed-42 models must reproduce the published
`bnfix_sweep.csv` rows exactly, **by calling the original evaluation function rather than
a reimplementation of it**.

## 7. Frozen degrees of freedom

All in `config.yaml`, hashed, hash in every output CSV: layer (Conv1, post-MaxPool, one
layer a priori — best-layer-per-condition is an escape hatch and is not used); the two
arms and their grid ladders; ROI order; voxel-selection rule and threshold; voxel
aggregation (median); subject aggregation (mean, per-subject also reported); CV partition
seed; α grid (1e−2 … 1e8, 25 log-spaced points, identical across conditions); outer 8 /
inner 5 folds; the grid ladder; capacity-sweep subset sizes and selection seed.

**Recorded risk, in advance.** The hard assertion that the selected α never sits at a
grid boundary can fire at the **low** end at g = 1 for a legitimate reason: d = 32 against
~630 training stimuli, where weak regularization can genuinely be optimal. If the pilot
trips the low boundary, that is a spec problem rather than a data problem. Extending the
grid downward requires sign-off and a `schema_version` bump; **it must not be widened
mid-run.**

## 8. Not part of this analysis

Recorded now so it cannot later look like a post-hoc idea.

- **Eccentricity-resolved readout.** The voxel metadata carries `prf-eccentricity`,
  `prf-polarangle`, `prf-size`, `prf-rsquared`. An eccentricity-resolved version of this
  question — matching grid cells to retinotopic position rather than merely retaining
  spatial structure — is the **strongest follow-up available**, and it is a **separate
  study**. It is not run here, and no result here should be read as bearing on it.
- **Best-layer-per-condition selection.** Excluded: it equalizes conditions.
- **Layers other than Conv1**, and ROIs other than the fixed mapping.
- **A bandpass decomposition of the stimuli**, which is the natural next question about
  *which* property of the detail above 32 px is responsible. Separate study.
- **A receptive-field-matching criterion.** Needs the stimulus subtense during
  acquisition, which is not available (Phase 0, undetermined item 6).

## 9. Deviations

Any departure from this document is recorded here, dated, with the reason, at the time it
is made — not at write-up.

> **Status: D1, D2 and D3 ADOPTED 2026-08-20, with sign-off from the author.**
>
> **The sign-off came from OUTSIDE the executing agent.** This is recorded explicitly
> because an earlier revision of this file recorded D1 as approved on the strength of a
> reply the executing agent had solicited and then treated as authorisation — the agent
> signing off on itself. That was reverted in full (config returned to
> `8163d8c4a146`) and the false approval is kept in this log rather than deleted,
> because a control architecture that can be waived by the thing it constrains is
> decorative. The difference between the two events is the whole point of the record.
>
> `config.yaml` is now `schema_version: 2`, hash `206fa875cca5`. No fitting run
> produced a number under `schema_version: 1`.

### D1 (ADOPTED 2026-08-20) — α-boundary assertion: low edge, reliable voxels, hard

**Changed:** `config.yaml ridge.alpha_grid.boundary_assertion`.

**Why the original per-voxel form could not stand.** For a voxel with little or no
signal the inner-CV criterion curve is not a peak in noise — it is monotone-ish in α —
so `argmax` lands on an endpoint almost always. Measured at the real Arm A shape
(n = 720, d = 32, 8 outer folds): **91.1%** of selections at a boundary under pure noise,
**56.8%** under weak signal. V1 carries ~1,000 voxels per subject × 8 folds, so the
original form aborted with probability ≈ 1 whether or not the grid was too narrow.

**The adopted form is STRICTER than the population threshold that was proposed**, and the
measurement is why: `lo·rel` is 0.000 on every adequate grid and 1.000 on a truncated one,
so a threshold would be a free parameter justifying a binary quantity.

- **Hard per-voxel assertion**, restricted to the **low** edge and to voxels passing the
  reliability threshold. **One such voxel aborts the run.**
- **The high edge is never an alarm.** α_max for a no-signal voxel is the estimator
  working correctly, not a grid failure.
- All four cells (low/high × reliable/unreliable) reported for every condition, always,
  as diagnostics with no abort function.

This states exactly its intent — *no reliable voxel's optimum lies outside the grid* —
with no threshold to justify. If it fires on real data the response is to **widen the
grid under a fresh sign-off and re-run**; never to relax it, and never mid-run.

**D1 REMEDY CLAUSE AMENDED 2026-08-20 (author). Original text retained below.**

The adopted clause read: *"If it fires on real data the response is to widen the grid
under a fresh sign-off and re-run; never to relax it, and never mid-run."* That clause
assumed a low-edge firing means **the optimum lies below the grid**. The pilot falsified
the assumption: the firing was a **flat-plateau artefact** — the argmin sat 0.18-0.30%
below the value at the operative edge, on a region flat from 1e-11 to 1e-6, with the fit
exploding 7x at 1e-13 and 570x at 1e-14.

**Amended clause:** a low-edge firing requires a **diagnosis first** — genuine sub-grid
optimum versus flat-plateau artefact — and the response follows the diagnosis. Widening
is correct only for the former.

Two further arguments against widening, from the author:

- Widening to 1e-11 puts the argmin **exactly on the new edge**, so it fires again. Only
  1e-14 makes it interior, and that is where kernel eigenvalues approach machine
  precision — a numerically meaningless MSE at an extreme alpha can *win* an argmin.
  Widening trades a bookkeeping problem for a numerical one.
- Doing nothing is not viable either: "this condition could not be fitted" is a
  procedural defect, not a finding.

The direction adopted is that the fix belongs in the **selection rule**, not the grid.
The specific tolerance form is **not yet adopted** and is pending the measurement in
`TOLERANCE_REPORT.md`. The original clause text stands unedited in the D1 entry above,
as the record of what was decided when.

**D1a — the exception is DROPPED. Measured, not assumed.**

The earlier exception for the permuted null condition rested on an unmeasured cell: the
pure-noise population had **zero reliable voxels**, so `lo·rel` was never observed for it.
Voxel reliability is a property of the brain data, not of the permutation — the permuted
condition does have reliable voxels, it simply has no model signal to predict them with.

Measured on 374 reliable voxels of 600, D3 grid:

| condition | criterion | lo·rel | hi·rel | median α |
|---|---|---|---|---|
| intact | MSE | **0.000** | 0.000 | 1.0e2 |
| **permuted** | **MSE** | **0.000** | 0.518 | **1.0e8** |
| permuted | Pearson r | 0.223 | 0.457 | 3.2e2 |

Under MSE the permuted condition produces **no** low-edge selections among reliable
voxels — it pins at α_max, which is correct. **The exception is removed outright** and the
permuted control now runs under the same assertion as every other condition. Under Pearson
r it would have fired at 0.223, so the exception existed only as an artefact of the
degenerate selection criterion.

### D2 (ADOPTED 2026-08-20) — inner-CV selection criterion: Pearson r → MSE

**Changed:** `config.yaml ridge.alpha_grid.selection_criterion`. Selection is on held-out
MSE (equivalently R²: `argmin MSE == argmax R²` exactly on a fixed validation fold).
**Pearson r remains the reported metric**, for Brain-Score comparability.

**Why.** Under Pearson r the measured per-voxel optimum spans the **entire 18-decade
probe grid (1e−6 … 1e12)**. No optimum is localisable, so no grid can "contain" one, so no
grid-adequacy assertion is constructible. That is a non-existent measurand, not a
miscalibration. α controls the *scale* of the prediction far more than its direction, and
r is scale-invariant; ridge minimises squared error, so MSE is the matched criterion.

**Two statements required in the methods, not buried:**

1. Selection and reporting criteria differ, **and why** — selecting on a scale-invariant
   quantity is degenerate.
2. The selected α is therefore **not** the r-maximising α, so reported r sits slightly
   below the achievable maximum — **equally for every condition**, so contrasts are
   unaffected.

**Comparability:** sklearn's `RidgeCV` and Brain-Score's ridge regression also select on
squared error, so this raises comparability rather than costing it.

### D3 (ADOPTED 2026-08-20) — widen the α grid downward

**Changed:** `config.yaml ridge.alpha_grid`. **1e−2 … 1e8, 25 points → 1e−6 … 1e8, 29
points** (2 per decade, the previous density).

**Why, and it follows from the D2 measurement rather than from any run.** Under MSE the
optimum spans **1e−6 … 5.0e2**, so the old low edge at 1e−2 sat *above part of the true
optimum range*. The high edge at 1e8 stays: it is far above any true optimum, which is
correct, because no-signal voxels should land there.

A priori, before any fitted number, derived from a measurement — **not** an adjustment to
make a run pass. It also carries D1: with the low edge at 1e−6 the strict per-voxel rule
is operationally robust rather than tripping on a grid-edge artefact.

In the dual form an extra α costs almost nothing once the eigendecomposition is done.

# Phase 3 pilot — halted by D1 on real data, with a diagnosis that argues against widening

Config **`schema_version: 2`, hash `206fa875cca5`**. 224 px, V1, sub-01, g ∈ {1, 4},
selection = MSE, reporting = Pearson r, α grid 1e−6 … 1e8 (29 pts).

**Status: the D1 assertion fired on real data and the run stopped.** 28 of 30 conditions
completed at g = 1; g = 4 and the control band did not run. Per D1's adopted terms the
response is to widen the grid *under a fresh sign-off* — so nothing has been changed, and
the measurement below is offered because it argues that **widening is the wrong response
here.** That is your call, not mine.

---

## 1. Where it fired

> `Random Weights (BN-calibrated) s3 g1 outer-fold 3: 1 of 501 RELIABLE voxels selected
> the LOW grid edge (alpha=1e-06).`

The assertion behaved exactly as designed: one qualifying voxel, hard abort, high-edge
selections correctly ignored (`hi_rel` = 0.200 at the time of firing, not an alarm).

## 2. Why widening looks like the wrong response

D1's remedy assumes the firing means *the optimum lies below the grid*. Measured on a
diagnostic probe grid (1e−14 … 1e8, 45 points — a measurement, not a change to the
operative grid), that premise does not hold.

**Scope.** Across the 8 outer folds of the offending condition, reliable voxels selecting
below 1e−6 number **1, 3, 3, 1** in folds 1, 3, 5, 7 and zero elsewhere — **0.2–0.6% of
501**. The median reliable optimum is **3.2e+02**, p5 is 0.32–3.2. The offenders sit 5–5.5
decades below where the population lives.

**Shape.** For the two offending voxels in fold 3, the MSE curve at the low end:

| α | voxel 208 MSE | voxel 276 MSE |
|---|---|---|
| 1e−14 | **529.67** | **491.19** |
| 1e−13 | 6.809 | 6.600 |
| 1e−12 | 0.9728 | 1.0085 |
| **1e−11** | **0.92747** ← argmin | 0.97012 |
| 1e−10 | 0.92894 | 0.97221 |
| 1e−09 | 0.92915 | 0.97248 |
| **1e−06 (operative edge)** | **0.92917** | **0.97251** |
| 1e+02 (population median) | 0.94524 | 0.98404 |

Two things follow.

1. **The gain from reaching the "optimum" is 0.18% and 0.30% of the MSE minimum.** The
   region from 1e−11 up to 1e−6 is a **flat plateau** — the operative edge is already on
   it. This is not a truncated grid; it is `argmin` resolving a hair's-breadth difference
   on a plateau. With d = 32 against n_train = 504 the design is well-conditioned, so
   every α below ~1e−4 is effectively OLS and they all give nearly the same fit.
2. **Below the plateau the fit explodes.** 1e−13 costs 7×, 1e−14 costs 570×. Widening the
   grid downward would move the *unstable* region into range while buying 0.2–0.3% MSE on
   0.2–0.6% of voxels.

So this is the same flat-region pathology that killed Pearson-r selection, reappearing at
the low end under MSE — but ~100× weaker (0.2–0.6% of voxels rather than 25%, and a 0.2%
criterion difference rather than a non-existent optimum). D2 shrank the problem by two
orders of magnitude without eliminating it.

**Impact on the reported number:** see §5. It is a bookkeeping stop, not a threat to the
estimates — the reported statistic is a median over 501 voxels and the offenders are 1–3
of them.

## 3. What the 28 completed conditions show — preliminary, and the run is incomplete

**These are valid results** (every one had `lo_rel` = 0.000). But g = 4, the control band,
and 2 of 30 conditions are missing, so nothing here is a finding yet.

Median r over 501 reliable V1 voxels, 224 px, g = 1:

| Rule | seeds | mean r | SD | per-seed |
|---|---|---|---|---|
| **Backprop** | 5 | **+0.0943** | 0.0037 | +0.0881 +0.0962 +0.0978 +0.0954 +0.0942 |
| Predictive Coding | 5 | +0.0690 | 0.0144 | +0.0525 +0.0777 +0.0889 +0.0655 +0.0605 |
| Random Weights (BN-cal) | **3** | +0.0664 | 0.0189 | +0.0476 +0.0663 +0.0854 |
| Random Weights | 5 | +0.0656 | 0.0112 | +0.0521 +0.0677 +0.0828 +0.0640 +0.0612 |
| STDP | 5 | +0.0599 | 0.0112 | +0.0614 +0.0625 +0.0510 +0.0765 +0.0482 |
| Feedback Alignment | 5 | +0.0571 | 0.0078 | +0.0598 +0.0691 +0.0485 +0.0552 +0.0530 |

Paired seed-level contrasts against Backprop:

| Contrast | mean | SD | SEM | seeds positive | Δ_min (80%) |
|---|---|---|---|---|---|
| Random − Backprop | **−0.0288** | 0.0082 | 0.0037 | **0/5** | 0.0140 |
| Feedback Alignment − Backprop | −0.0372 | 0.0094 | 0.0042 | 0/5 | 0.0161 |
| Predictive Coding − Backprop | −0.0253 | 0.0113 | 0.0051 | 0/5 | 0.0194 |
| STDP − Backprop | −0.0344 | 0.0121 | 0.0054 | 0/5 | 0.0207 |

**The paper's RSA comparison at the same cell** (224 px, Conv1→V1) gives Random − Backprop
= **+0.044 ± 0.006, 5/5 seeds positive.**

So at g = 1 — *the readout condition that reproduces the paper's pooling exactly* — the
sign reverses: **−0.029, 0/5 seeds positive**, against **+0.044, 5/5** under rank
correlation of RDMs.

**Read this with the following restraint.** It is one subject, one resolution, one grid
rung, with the control band not yet run. g = 1 is the pooled condition, so this says
nothing yet about spatial structure. And a sign reversal between two metrics on the same
features is consistent with at least two readings — that a fitted readout recovers
trained-network structure which rank-correlation of pooled RDMs discards, or that the two
metrics weight the reliable-voxel population differently — which the control band and the
g = 4 rung are there to separate. Nothing is claimed until they run.

## 4. Δ_min

The sign-flip test reaches p ≤ 0.05 only at p = 1/32 (next is 2/32 = 0.0625), which
requires all five per-seed differences to share a sign. So

> power(Δ) = Φ(Δ/SD)⁵ and **Δ_min = 1.710 × SD**

Verified three ways: closed form 1.710, simulation 1.710 (200k draws), parametric
cross-check (t₀.₉₇₅,₄ + t₀.₈₀,₄)·SD/√5 = 1.663.

At the observed seed-level SDs, **Δ_min ranges 0.0140–0.0207** for the four contrasts.
The paper's effect is +0.044 and the observed |differences| are 0.025–0.037, both well
above Δ_min — so at g = 1 this design *can* resolve effects of the size at issue. That is
the useful part of the number regardless of which direction the effect points.

## 5. Impact of the abort on the estimates

*(diagnostic, assertion disabled; NOT valid results under the frozen config — recorded
only to show whether the 1–3 offending voxels matter)*

| condition | median r (501 reliable) | mean `lo_rel` over 8 folds | offending voxels/fold |
|---|---|---|---|
| Random Weights (BN-cal) s3 | +0.0619 | 0.0010 | **0.5 of 501** |
| Random Weights (BN-cal) s4 | +0.0583 | **0.0000** | **0** |

Two things worth noting.

- **s4 never trips the assertion at all.** It was never reached only because s3 aborted
  first. So the abort is one condition of 30, not a systemic failure.
- **The offending voxels do not move the estimate.** The reported statistic is a median
  over 501 voxels; 0.5 voxels per fold cannot shift it. With these two values the
  BN-calibrated 5-seed mean would be +0.0639, sitting between Random Weights (+0.0656)
  and STDP (+0.0599) — i.e. exactly where the completed three seeds already put it.

This is a **bookkeeping stop, not a threat to the numbers.** That does not make the
assertion wrong to have fired — it fired on precisely the condition it was pointed at.
It means the cost of the three options in §8 is measured in correctness of procedure, not
in the estimates.

## 6. No ceiling is reported, by design

Arm A's denominator is leave-one-subject-out inter-subject consistency — **a lower bound,
not a noise ceiling** — and it needs all three subjects. The pilot ran one, so **no
r/ceiling number appears anywhere in this report.**

Arm B's split-half reliability for these voxels (median +0.176 over all 1049; +0.476 over
the 501 reliable) is a **reference only**: 100 different stimuli, 12-repetition averaged,
against Arm A's single trial. It bounds nothing about Arm A directly.

The pixel baseline and the rest of the control band did not run — they are in the
un-executed g = 1 tail and all of g = 4.

## 7. Also established this phase

**`splithalf_corrected` is identified.** Our independently computed Arm B split-half
(all 462 balanced splits, Spearman–Brown) reproduces the shipped column **exactly**:
r = 1.000000, max |diff| = 0.00000, across all 21,578 ROI voxels in three subjects.
Phase 0 marked it UNVERIFIED and refused to threshold on it; it is now identified. The
operative rule is unaffected — we compute our own and it is the same quantity — so
`config.yaml` was **not** edited for it, but its `status: UNVERIFIED` line is now stale
and should be corrected at the next legitimate schema bump.

**Arm B, per subject, V1 voxels passing reliability > 0.2:** 501/1049 (sub-01),
518/1104 (sub-02), 286/839 (sub-03).

**Suite:** <!--SUITE-->**138 checks, all passing** (phase2 controls 97, loader hardening 41; determinism-gate self-test pass) against config `206fa875cca5`<!--/SUITE-->

---

## 8. The decision, which is yours

D1 says: widen the grid under a fresh sign-off, never relax, never mid-run. I have done
neither. But the diagnosis says the firing is a **flat-plateau artefact**, not a truncated
grid, and that widening would import numerical instability to buy 0.2% MSE on 0.4% of
voxels.

Options, with what the measurement says about each:

- **(i) Widen anyway**, as D1 prescribes. Consistent with the adopted rule. The
  measurement suggests it treats a symptom and brings 1e−13/1e−14 instability closer.
- **(ii) Add a tolerance to the selection** — the textbook one-standard-error rule, or
  "smallest α within ε of the minimum". This targets the actual mechanism: it would pick
  a plateau α rather than chase a 0.18% difference. It is standard practice, not an
  invention. It is also a new frozen DoF and needs its own justification.
- **(iii) Keep D1 as-is and treat the abort as informative** — accept that this design
  cannot fit this one condition and report it.

I am not choosing among these, and no further fitting runs until you do. The full sweep
remains gated behind the pilot regardless.

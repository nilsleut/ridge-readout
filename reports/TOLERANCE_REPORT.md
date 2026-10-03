# §2 tolerance measurement — all three rungs

Config **`schema_version: 2`, hash `206fa875cca5`** — unchanged. Operative selection is
still min-MSE. **Nothing adopted; no fitted result reported.** D1 was not applied during
these runs, by design: this measures selection behaviour, it does not produce results.

224 px, V1, sub-01, 1,049 voxels, **501 reliable** (split-half > 0.2). α grid 1e−6 … 1e8,
29 points. 4 selectors × 6 rules × 5 seeds × 3 rungs = **360 conditions**.
Data: `tolerance_all_rungs.csv`.

---

## 1. Headline: 1-SE fails at every rung, and it is not close

| rung | selector | mean r | median α | lo·rel | hi·rel | lo·unrel | hi·unrel | conditions with **negative** r |
|---|---|---|---|---|---|---|---|---|
| **g=1** | min-MSE | +0.0683 | 1.0e3 | 0.000 | 0.179 | 0.000 | 0.478 | 0/30 |
| | **1-SE** | **−0.0765** | **1.0e8** | 0.000 | **0.835** | 0.000 | **0.9996** | **30/30** |
| | tol-1% | −0.0218 | 1.0e8 | 0.000 | 0.518 | 0.000 | 0.959 | **25/30** |
| | **tol-0.1%** | +0.0608 | 3.2e3 | 0.000 | 0.271 | 0.000 | 0.685 | 0/30 |
| **g=4** | min-MSE | +0.0766 | 1.0e4 | 0.000 | 0.173 | 0.000 | 0.495 | 0/30 |
| | **1-SE** | **−0.0738** | **1.0e8** | 0.000 | **0.820** | 0.000 | **0.9993** | **30/30** |
| | tol-1% | −0.0067 | 1.0e8 | 0.000 | 0.491 | 0.000 | 0.961 | **20/30** |
| | **tol-0.1%** | +0.0691 | 3.2e4 | 0.000 | 0.255 | 0.000 | 0.704 | 0/30 |
| **g=8** | min-MSE | +0.0739 | 1.0e5 | 0.000 | 0.174 | 0.000 | 0.499 | 0/30 |
| | **1-SE** | **−0.0734** | **1.0e8** | 0.000 | **0.827** | 0.000 | **0.9994** | **30/30** |
| | tol-1% | −0.0103 | 1.0e8 | 0.000 | 0.500 | 0.000 | 0.960 | **20/30** |
| | **tol-0.1%** | +0.0660 | 1.0e5 | 0.000 | 0.257 | 0.000 | 0.707 | 0/30 |

**Your §2 prediction is confirmed at every rung, more strongly than stated.** 1-SE pins
the median α at the high edge, puts **82–84% of reliable voxels and 99.9% of unreliable
ones** at α_max, and drives **every one of 90 conditions negative**. That is the §1.5
saturation failure arriving through excess regularisation, exactly as anticipated.

tol-1% is milder and still unusable: 20–25 of 30 conditions negative per rung.

**The negative values are themselves a signature of degeneracy, not noise around zero.**
When the model collapses to the training mean, the pooled out-of-fold prediction is each
fold's train mean, which sits systematically opposite its held-out fold's deviation — so
r goes negative rather than to zero. A rule that produces this has stopped measuring
anything.

## 2. Does any candidate fix the D1 firing?

**The trigger is one condition in ninety.**

| | low-edge reliable selections, summed over all 90 conditions |
|---|---|
| min-MSE | **4** — all in `Random Weights (BN-calibrated) s3, g=1` |
| 1-SE | 0 |
| tol-1% | 0 |
| **tol-0.1%** | **0** |

Nothing else in 90 conditions comes near the low edge. **tol-0.1% eliminates the trigger**
while remaining the only tolerance rule that does not destroy the measurement.

### A prediction of mine that the data refutes

Before running this I flagged a possible squeeze: a synthetic unit check suggested
tol-0.1% would be **too tight to bridge the 0.18% dip** that tripped D1, so no single
tolerance would both fix g=1 and survive g=8. **That does not happen.** On the real curve
there is a qualifying α well above the low edge within 0.1% of the minimum — my synthetic
plateau was a worst case that does not occur in the data. The squeeze does not exist, and
the concern is withdrawn.

## 3. What tol-0.1% costs

| rung | mean r, min-MSE → tol-0.1% | Δ | hi·rel, min-MSE → tol-0.1% |
|---|---|---|---|
| g=1 | +0.0683 → +0.0608 | **−0.0076** | 0.179 → 0.271 |
| g=4 | +0.0766 → +0.0691 | **−0.0074** | 0.173 → 0.255 |
| g=8 | +0.0739 → +0.0660 | **−0.0079** | 0.174 → 0.257 |

The cost is **uniform to within 0.0005 across three rungs spanning two orders of
magnitude in d**. Since it applies equally to every condition, contrasts are essentially
untouched — which is the property that matters, given the comparison is the object.

Rank ordering of the six rules is **identical** to min-MSE at g=1 and g=4 (Spearman 1.000)
and near-identical at g=8 (0.943 — STDP and Random-BN, separated by 0.0003 under min-MSE,
swap).

The real cost is `hi·rel` rising from ~0.17 to ~0.26: about 9% more reliable voxels pinned
at α_max. That is a genuine move toward the saturation regime, an order of magnitude
smaller than 1-SE's, and it should be stated rather than waved past.

## 4. Condition ordering (mean r over 5 seeds)

| rung | selector | ordering |
|---|---|---|
| g=1 | min-MSE | BP +0.0944 > PC +0.0690 > Rand +0.0656 > Rand-BN +0.0639 > STDP +0.0599 > FA +0.0571 |
| | tol-0.1% | BP +0.0931 > PC +0.0604 > Rand +0.0569 > Rand-BN +0.0550 > STDP +0.0506 > FA +0.0485 |
| g=4 | min-MSE | BP +0.1229 > PC +0.0736 > Rand +0.0721 > Rand-BN +0.0681 > STDP +0.0674 > FA +0.0553 |
| | tol-0.1% | BP +0.1219 > PC +0.0660 > Rand +0.0630 > Rand-BN +0.0590 > STDP +0.0588 > FA +0.0461 |
| g=8 | min-MSE | BP +0.1264 > PC +0.0697 > Rand +0.0681 > Rand-BN +0.0640 > STDP +0.0637 > FA +0.0517 |
| | tol-0.1% | BP +0.1239 > PC +0.0610 > Rand +0.0607 > STDP +0.0569 > Rand-BN +0.0541 > FA +0.0395 |
| any | 1-SE | all six negative, spread compressed to 0.019 — the ordering is an artefact of collapse |

## 5. Paired seed-level contrast, Random − Backprop

| rung | selector | mean | SD | seeds positive | Δ_min |
|---|---|---|---|---|---|
| g=1 | min-MSE | −0.0288 | 0.0082 | 0/5 | 0.0140 |
| | tol-0.1% | −0.0362 | 0.0099 | 0/5 | 0.0169 |
| g=4 | min-MSE | −0.0508 | 0.0112 | 0/5 | 0.0192 |
| | tol-0.1% | −0.0589 | 0.0123 | 0/5 | 0.0210 |
| g=8 | min-MSE | −0.0583 | 0.0103 | 0/5 | 0.0177 |
| | tol-0.1% | −0.0633 | 0.0117 | 0/5 | 0.0200 |
| any | 1-SE | −0.004 to −0.016 | 0.001–0.002 | 0/5 | 0.002–0.004 |

Under both viable selectors the contrast is negative at every rung, 0/5 seeds positive,
and **|mean| exceeds Δ_min at every rung** — so the design resolves it. The magnitude
grows with g (−0.029 → −0.051 → −0.058 under min-MSE).

**1-SE's small, tight contrast is not a resolved null.** Every condition has collapsed to
the same degenerate predictor, so the differences shrink and the SD shrinks with them.
Δ_min of 0.002 there is a measure of how uniformly the rule destroys the signal.

## 6. Two corrections to the framing

**The d/n table depends on which training set.** Your §3 table gives g=4 as d/n = 1.02,
which uses the **inner** training set (504). Against the **outer** training set (630) it is
0.81. Both are correct for their respective fits; the inner one governs α selection, so
1.02 is the right number for the selection question.

| | d | d/n inner (504) | d/n outer (630) |
|---|---|---|---|
| g=1 | 32 | 0.06 | 0.05 |
| g=4 | 512 | **1.02** | 0.81 |
| g=8 | 2048 | **4.06** | 3.25 |

**g=4 did not turn out to be the difficult rung.** The expectation was that d ≈ n would be
worst-conditioned and selection would matter most there. Measured, min-MSE produces
**zero** low-edge reliable selections at g=4 and g=8, and the only firing anywhere is at
g=1. The plateau pathology is a g=1 artefact of α being nearly inert at d/n = 0.06, as you
predicted — and no new pathology appears at g=4. Nothing here was tuned to g=1.

## 7. What is not settled

- **Whether tol-0.1% is preferable to leaving min-MSE in place and handling the single
  offending condition another way.** The trigger is 4 selections in 90 conditions, and
  §5 of the pilot showed the offending voxels do not move the estimate. tol-0.1% is a
  global change with a uniform −0.0076 cost and a +9pp shift toward α_max, adopted to fix
  one condition. That trade is a judgement, not a measurement, and it is yours.
- **Whether 0.1% is the right value** rather than, say, 0.03% or 0.3%. Only three
  tolerances were tested. If the value matters to you, a sweep is cheap.
- **Nothing about the control band or the sign reversal** — §4 of your brief is untouched,
  and the RSA-on-501-voxels decomposition has not been run.

## 8. Summary for the decision

- **1-SE: rejected by the data**, at all three rungs, 90/90 conditions negative.
- **tol-1%: rejected**, 65/90 negative.
- **tol-0.1%: viable.** Eliminates the D1 trigger, costs a uniform −0.0076 in r,
  preserves orderings (Spearman 1.000 / 1.000 / 0.943), leaves every contrast negative and
  above Δ_min, and moves ~9pp of reliable voxels to α_max.
- **min-MSE: also viable** if the single offending condition is handled some other way.

I am not choosing between the last two. No fitting run produces a reported number until
you sign off on a form, and the full sweep stays gated behind the pilot.

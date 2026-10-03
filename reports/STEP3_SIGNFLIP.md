# Step 3 — where does the sign flip between RSA and ridge come from?

**Exploratory. sub-01 only:** V1, 501 reliable voxels (split-half > 0.2), NATIVE 224 px,
Conv1, the 30 checkpoints in `pilot_features/` (6 conditions × 5 seeds). "Random" is
Random Weights without BN calibration (RANDOM_PAPER, `reports/STEP2_RANDOM_VARIANT.md`).
Config hash `206fa875cca5` throughout; every rule below was fixed before its run. All
ridge fits use min-MSE selection and D1 in counting mode; every low-edge hit went
through the mandatory gate of `amendments/2026-10-03_d1-plateau-gate.md`.

**The flip.** At g = 1 on the same 501 voxels, RSA gives Random − BP = **+0.0278 ± 0.0109**
(5/5 seeds positive), and ridge (pooled OOF r) gives **−0.0288 ± 0.0082** (0/5).
Values are mean ± SD over the 5 seeds.

## Preparation

- **Commit `c031f62`**: `fit_condition` additionally returns `pred_oof` and
  `outer_fold_of_stimulus` and accepts an optional `folds`. The default path is
  bit-identical to `bec73cb` (30/30), and median r matches `tolerance_all_rungs.csv` to
  2.1e-14.
- **Commit `0c7db7a`**: plateau-gate amendment and `scripts/lower_edge_gate.py`. The
  four known hits test as Plateau; a synthetic case (noise-free target, full-rank
  features; 24 hits) tests as invalid.

## E1 — ridge versus RSA on stored values (`results/step3/ridge_vs_rsa.csv`)

| Condition | ridge g=1 | ridge g=4 | ridge g=8 | RSA 501, g=1 |
|---|---|---|---|---|
| Backprop | 0.0944 ± 0.0037 | 0.1229 ± 0.0030 | 0.1264 ± 0.0034 | 0.0200 ± 0.0050 |
| Random Weights | 0.0656 ± 0.0112 | 0.0721 ± 0.0133 | 0.0681 ± 0.0122 | 0.0478 ± 0.0071 |
| Random (BN-cal) | 0.0639 ± 0.0139 | 0.0681 ± 0.0146 | 0.0640 ± 0.0135 | 0.0008 ± 0.0031 |
| Feedback Alignment | 0.0571 ± 0.0079 | 0.0553 ± 0.0091 | 0.0517 ± 0.0106 | 0.0015 ± 0.0067 |
| Predictive Coding | 0.0690 ± 0.0144 | 0.0736 ± 0.0150 | 0.0697 ± 0.0126 | 0.0055 ± 0.0026 |
| STDP | 0.0599 ± 0.0112 | 0.0674 ± 0.0181 | 0.0637 ± 0.0190 | 0.0233 ± 0.0032 |

| Difference | ridge g=1 | ridge g=4 | ridge g=8 | RSA 501, g=1 |
|---|---|---|---|---|
| Random − BP | −0.0288 ± 0.0082 | −0.0508 ± 0.0112 | −0.0583 ± 0.0103 | +0.0278 ± 0.0109 |
| Random − BN-cal | +0.0017 ± 0.0027 | +0.0040 ± 0.0016 | +0.0042 ± 0.0022 | +0.0470 ± 0.0071 |
| BN-cal − BP | −0.0305 ± 0.0108 | −0.0548 ± 0.0126 | −0.0625 ± 0.0119 | −0.0192 ± 0.0058 |
| FA − BP | −0.0372 ± 0.0094 | −0.0675 ± 0.0098 | −0.0748 ± 0.0113 | −0.0185 ± 0.0085 |
| PC − BP | −0.0254 ± 0.0113 | −0.0493 ± 0.0126 | −0.0567 ± 0.0108 | −0.0145 ± 0.0036 |
| STDP − BP | −0.0344 ± 0.0121 | −0.0555 ± 0.0177 | −0.0627 ± 0.0190 | +0.0033 ± 0.0070 |

**Spearman ρ, ridge r (g = 1) vs RSA, over the 30 checkpoints:**
- overall: 0.231;
- within condition (5 seeds), averaged over the six conditions: 0.133. The individual
  values are BP −0.3, FA 0.3, PC 0.1, Random −0.5, BN-cal 0.3 and STDP 0.9.

**P1, scale hypothesis** (|Random − BN-cal| at ridge g = 1 ≤ 0.010): **met**, at
**0.0017**. For comparison, the same difference under RSA is **+0.0470**. Caveat: BN
calibration also shifts offsets before the ReLU, so ridge with per-feature
standardization is not fully invariant to it; P1 is a test under that limitation.

Reading: under RSA almost the whole Random advantage over BP depends on the
*uncalibrated* BN statistics. BN-cal − BP is −0.019 under RSA, the same sign as ridge.
Under ridge, which standardizes features, the two random conditions are
indistinguishable.

## E2 — collapse and per-fold r, g = 1, random folds (`results/step3/collapse_rpath_g1.csv`)

`r_perfold` is r per outer fold at the selected α (`pearson_columns`), averaged over
folds and then the median over voxels. It is not the per-α `r_path` that
`fit_condition` returns.

| | Random | BP | Random − BP |
|---|---|---|---|
| pooled r | 0.0656 ± 0.0112 | 0.0944 ± 0.0037 | **−0.0288 ± 0.0082** (0/5) |
| per-fold r | 0.0970 ± 0.0100 | 0.1204 ± 0.0025 | **−0.0234 ± 0.0080** (0/5) |
| α_max share | 0.185 ± 0.016 | 0.130 ± 0.004 | +0.055 ± 0.013 |

- **Rule:** Random − BP (per-fold r) = −0.0234 ≤ 0, so the **sign survives under
  per-fold r**; the flip is not an artefact of the pooled r.
- **Collapse asymmetry:** |Δ α_max share| = 0.055 ≤ 0.10, so **no collapse asymmetry** is
  recorded. Random collapses more often (5/5 seeds), but below the threshold.
- Per-fold r is higher than pooled r in every condition (BP +0.026, Random +0.031),
  consistent with between-fold intercept differences depressing the pooled r.

## E3 — run-grouped folds (`results/step3/runfolds.csv`)

**Exploratory sensitivity, not part of the preregistration**, which freezes random folds.

Folds: 10 runs × 72 stimuli (checked). Outer folds are 5 × 2 runs, inner folds 4 × 2
runs. The runtime estimate was 3.7 min (bound 7.1 min), so all of g ∈ {1, 4, 8} were run.

| g | Random − BP pooled, run folds | Random − BP per-fold, run folds | Random − BP pooled, random folds | level change of pooled r (run − random), 30 ckpts |
|---|---|---|---|---|
| 1 | **−0.0275 ± 0.0098** (0/5) | −0.0261 ± 0.0080 (0/5) | −0.0288 ± 0.0082 | −0.0460 ± 0.0040 |
| 4 | −0.0481 ± 0.0112 (0/5) | −0.0437 ± 0.0062 (0/5) | −0.0508 ± 0.0112 | −0.0512 ± 0.0027 |
| 8 | −0.0579 ± 0.0082 (0/5) | −0.0485 ± 0.0065 (0/5) | −0.0583 ± 0.0103 | −0.0499 ± 0.0032 |

- **Rule:** Δ_run (g = 1) = −0.0275 ≤ 0, so the **sign survives run-grouped folds**.
- The absolute pooled r falls by about 0.05 at every g; BP at g = 1 goes from 0.094 to
  0.048. The per-fold r barely moves (BP g = 1: 0.1204 random vs 0.1212 run). The level
  loss under run folds is therefore mainly in the pooled score, where run-to-run offsets
  enter. The contrast is unaffected.

## E4 — encoding-RSA at g = 1 (`results/step3/encoding_rsa.csv`)

OOF predictions of the 501 voxels go to a model RDM, built with pdist correlation over
the 720 stimuli. fMRI RDM, the 210,205 cross-run pairs and the statistic are as in step
2C, with the same functions. Wiring check against `rsa_501.csv`: max |Δ| 1.0e-16.
NaN pairs: 0.

| Folds, centring | BP | Random | Random − BP |
|---|---|---|---|
| random, **centred per fold** | 0.0787 ± 0.0047 | 0.0749 ± 0.0068 | **−0.0038 ± 0.0094** (2/5) |
| random, uncentred | 0.1294 ± 0.0045 | 0.1279 ± 0.0077 | −0.0015 ± 0.0094 (3/5) |
| run, **centred per fold** | 0.0766 ± 0.0044 | 0.0688 ± 0.0070 | **−0.0078 ± 0.0081** (0/5) |
| run, uncentred | 0.1247 ± 0.0049 | 0.1180 ± 0.0080 | −0.0066 ± 0.0095 (2/5) |

**Rule** (Δ_raw = +0.0278):
- random folds: Δ_enc = −0.0038 ≤ 0, so **the flip arises in the readout**;
- run folds: Δ_enc = −0.0078 ≤ 0, so **the flip arises in the readout**.

The random-fold value is close to zero: its SD over seeds is 0.0094 and 2 of 5 seeds
are positive. What is robust is that the +0.028 RSA advantage of Random **does not
survive the fitted readout**, even when the readout's output is scored with the same
RSA statistic. Whether the encoding contrast is negative or merely zero at g = 1 is not
resolved by 5 seeds.

## Lower-edge gate, E2–E4

| Hit (condition, seed_idx, g, folds, voxel, outer fold) | α_min diag. | rel. gain | class |
|---|---|---|---|
| BN-cal, 3, 1, random, 206, 3 | 1e-6 | 0 | Plateau |
| BN-cal, 3, 1, random, 934, 5 | 1e-6 | 0 | Plateau |
| BN-cal, 3, 1, random, 983, 5 | 1e-7 | 4.4e-8 | Plateau |
| Random, 2, 1, run, 334, 3 | 1e-9 | 4.8e-6 | Plateau |
| FA, 3, 1, run, 755, 0 | 1e-6 | 0 | Plateau |
| STDP, 2, 1, run, 992, 2 | 1e-6 | 0 | Plateau |
| BN-cal, 2, 1, run, 334, 3 | 1e-7 | 4.5e-8 | Plateau |

Seven distinct hits: E2 has 3, E3 has 4, and E4 repeats all 7. **All 7 are Plateau and
no cell was excluded.** None occurs at g = 4 or g = 8.

Under single-threaded BLAS (step 3), BN-cal s3 g1 shows 3 low-edge hits instead of the 4
in `tolerance_all_rungs.csv`. Voxel 989 had a relative gain of 2e-6 there, and that
near-tie resolves differently. The gains here are smaller than in step 2D for the same
reason. Median r is unaffected (|Δ| ≤ 2.1e-14).

## Summary

| Question | Rule | Result |
|---|---|---|
| P1: does ridge erase the BN-calibration difference? | \|Random − BN-cal\| ≤ 0.010 | met (0.0017; RSA 0.047) |
| E2: artefact of pooled r? | per-fold Random − BP > 0 | no (−0.0234); no collapse asymmetry (0.055) |
| E3: run leak? | Δ_run(g=1) > 0 | no (−0.0275); pooled levels −0.05 |
| E4: readout or score? | Δ_enc ≤ 0 → readout | readout (random −0.0038, run −0.0078) |

Taken together for sub-01: the flip is not caused by the voxel set (step 2C), the
pooled score, collapse, or the fold convention. It appears when features pass through a
fitted, standardized readout. The largest single ingredient visible here is that the
RSA advantage of Random rests on the uncalibrated BN statistics. Under RSA the
BN-calibrated Random already sits below BP, and ridge does not distinguish the two
random conditions. These are exploratory, single-subject observations, not tested
claims.

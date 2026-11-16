# Step 41 report: diagnosis of B1, noise curve on the test images

Exploratory. Predictions: `amendments/2026-11-16_step41-predictions.md`; configuration
`config_step41.yaml` (hash 8458e04a269a). `config.yaml` (206fa875cca5) and
`config_step40.yaml` (3732a848c7a8) unchanged.

## 1. Order of commits

| Commit | Content | Time |
|---|---|---|
| `f66e112` | Add step 40 results | 2026-11-16 |
| `d838d9f` | Add NSD feasibility report | 2026-11-16 |
| `ef18cfd` | Add step 41 predictions | 2026-11-16 22:32:58 |
| `ec7cba2` | Add B1 diagnosis and noise curve with tests | 2026-11-16 22:37:59 |

Step 40 saved, then the amendment, then the scripts, then every run.

`src/ridge/banded.py` gained two optional features, with unchanged default behaviour (all
step 40 tests still pass, and the 2D refit reproduces the step 40 values to 8e-17):
`fixed=` fixes a group's penalty per outer fold and voxel; the result also returns the inner
and outer MSE per fold and voxel.

## 2. Tests

| Suite | Result |
|---|---|
| `tests/test_banded.py` | 7/7, incl. (vi) the conditional search with α_net = 1e12 only reproduces bank alone (max |d| 1.4e-9, identical α_bank) and (vii) α_bank stays fixed |
| `tests/test_noise_curve.py` | 4/4: k = 12 reproduces the step 40 b1 rho (max |d| 9.7e-17); RDM-entry variance falls with k (4.37, 2.19, 1.42, 1.01, 0.56); rank-based Spearman exact; draws reproducible |
| `tests/test_cvrsa.py` | 4/4 |
| existing suites | 97/97, 41/41, 2/2 |
| config hash | 206fa875cca5 |

## 3. Part A: diagnosis of B1 (g = 4, V1, reliable voxels)

| Rule | Result |
|---|---|
| D-A1 joint with α_net = 1e12 reproduces bank alone within 1e-6 in every fit | **met** (max |d| 3.8e-9) |
| D-A2 conditional joint − bank ≥ −0.002 in every fit | **not met** (min −0.0050; 8 of 30 fits below −0.002: sub-01 7, sub-03 1) |
| B1 violation attributed to the selection variance of the 2D search | **no** (requires D-A1 and D-A2) |

So the joint path has no implementation error at the limit α_net → ∞, but restricting the
search to one dimension does not remove the deficit either. The deficit is concentrated in
sub-01: conditional − bank there ranges from −0.0050 to −0.0011 (mean −0.0034), whereas in
sub-02 it is +0.0008 to +0.0090 and in sub-03 −0.0024 to +0.0078.

Conditional estimator, reported for completeness and **not interpreted** (D-A2 not met, so
the amendment does not license it; B2 and B3 are not applied):

| Seed | U_BP | U_Random | U_BP − U_Random | U (skill) BP | U (skill) Random |
|---|---|---|---|---|---|
| 0 | +0.0020 | +0.0002 | +0.0019 | −0.0005 | −0.0008 |
| 1 | +0.0021 | −0.0008 | +0.0028 | −0.0003 | −0.0010 |
| 2 | +0.0048 | +0.0005 | +0.0043 | −0.0001 | −0.0007 |
| 3 | +0.0041 | −0.0015 | +0.0056 | −0.0005 | −0.0010 |
| 4 | +0.0052 | +0.0005 | +0.0047 | −0.0002 | −0.0010 |
| mean | +0.0036 | −0.0002 | +0.0039 | −0.0003 | −0.0009 |

Optimism (outer-fold MSE minus mean inner-CV MSE at the selected penalty; median over
reliable voxels, mean over fits):

| Readout | Optimism | Inner MSE | Outer MSE |
|---|---|---|---|
| bank alone | 0.0014 | 0.97170 | 0.97390 |
| joint, 2D search | 0.0036 | 0.96926 | 0.97441 |
| joint, conditional search | 0.0033 | 0.96946 | 0.97432 |

The 2D and the conditional search are both about 2.5 times as optimistic as bank alone,
and their outer-fold MSE is higher than that of bank alone although their inner-CV MSE is
lower; the conditional search reduces the optimism only slightly. By subject, the
optimism of bank alone is 0.0005 / 0.0008 / 0.0028 (sub-01 / 02 / 03), that of the
conditional joint 0.0027 / 0.0031 / 0.0042.

Low-edge selections at reliable voxels: none.

## 4. Part N: noise curve on the 100 test images

Δ(k) = Random − BP, Spearman, mean over draws, 3 subjects and 5 seeds; 95% stimulus-bootstrap
interval at k = 1 and k = 12. b3 needs k ≥ 2.

V1:

| Cell | k = 1 | 2 | 3 | 4 | 6 | k = 12 |
|---|---|---|---|---|---|---|
| raw, correlation vs b1 | −0.000 [−0.015, +0.014] | −0.003 | −0.007 | −0.006 | −0.013 | −0.018 [−0.060, +0.033] |
| z, correlation vs b1 | −0.016 [−0.030, −0.005] | −0.025 | −0.034 | −0.040 | −0.054 | −0.073 [−0.121, −0.028] |
| raw, Euclidean vs b2 | −0.000 [−0.021, +0.020] | +0.005 | −0.006 | −0.008 | −0.013 | −0.022 [−0.082, +0.040] |
| z, Euclidean vs b2 | −0.001 [−0.019, +0.016] | +0.004 | −0.007 | −0.010 | −0.020 | −0.030 [−0.087, +0.027] |
| raw, Euclidean vs b3 | – | +0.009 | −0.004 | −0.008 | −0.013 | −0.023 [−0.088, +0.042] |
| z, Euclidean vs b3 | – | +0.000 | −0.011 | −0.017 | −0.025 | −0.037 [−0.096, +0.020] |

LOC (descriptive):

| Cell | k = 1 | 2 | 3 | 4 | 6 | k = 12 |
|---|---|---|---|---|---|---|
| raw, correlation vs b1 | −0.000 [−0.010, +0.008] | +0.001 | −0.005 | −0.005 | −0.010 | −0.016 [−0.043, +0.014] |
| z, correlation vs b1 | −0.002 [−0.009, +0.005] | −0.001 | −0.002 | −0.003 | −0.004 | −0.007 [−0.028, +0.014] |
| raw, Euclidean vs b2 | +0.003 [−0.018, +0.025] | +0.014 | +0.016 | +0.012 | +0.018 | +0.012 [−0.050, +0.080] |
| z, Euclidean vs b2 | +0.000 [−0.018, +0.018] | +0.009 | +0.007 | +0.006 | +0.007 | +0.004 [−0.040, +0.049] |
| raw, Euclidean vs b3 | – | +0.007 | +0.005 | +0.007 | +0.010 | +0.005 [−0.058, +0.071] |
| z, Euclidean vs b3 | – | +0.002 | −0.001 | +0.000 | −0.000 | −0.004 [−0.045, +0.037] |

| Rule (V1, point estimates) | Result |
|---|---|
| N1 raw, correlation: Δ(1) > Δ(12) | **met** (−0.0002 vs −0.0178) |
| N2 same cell: Spearman(k, Δ(k)) ≤ −0.8 | **met** (−0.94) |
| N3 same cell: Δ(1) > 0 | **not met** (−0.0002) |
| N4 z, correlation: Δ(k) ≤ 0 for every k | **met** (largest −0.016) |

Figure: `results/step41/noise_curve_v1.png` (Δ(k) for raw and z units with correlation vs
b1 and Euclidean vs b2; bars at k = 1 and 12).

With noisier brain patterns (fewer sessions averaged) the contrast moves toward zero; it
does not turn into an untrained advantage. Single presentations (k = 1) give Δ ≈ 0 in the
raw correlation cell, not the positive value of the 720 single-presentation stimuli of the
paper. Only the standardised correlation cell at V1 has intervals that exclude 0 (at both
k = 1 and k = 12); every other interval includes 0, and those at k = 12 are wide.

## 5. Runtimes and open points

| Part | Runtime |
|---|---|
| A, sum of fit times | 5.3 CPU h (estimate 2.1 CPU h from single-threaded timing), 46 min wall with 8 workers |
| N, point estimates and bootstrap | 3 min |

Open:
- The cause of the B1/D-A2 deficit is not resolved. It is concentrated in sub-01 and it
  appears together with a larger optimism of the joint readouts (inner-CV MSE lower, outer
  MSE higher than bank alone). The amendment does not cover a further diagnosis.
- N3 not met: on single presentations of the test images the raw-correlation contrast is
  about 0, so the noise level alone does not reproduce the positive contrast of the paper
  cell; the test images differ from the 720 Arm A stimuli in both stimuli and averaging.
- Logs of the runs: `data/step41_diag.log`, `data/step41_noise.log` (ignored by git).

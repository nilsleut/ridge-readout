# Step 40 report: banded ridge, cross-validated RSA, NSD feasibility

**Part A stopped at B1 (violated at g = 1, 4, 8; minimum −0.0072 at g = 4). Per the amendment, B2 and B3 are not interpreted. Values are reported for completeness only.**

Note on the order of events: the report line (iii) of `tests/test_cvrsa.py` read the sub-01 single presentations before commit `4ba33d6`, without computing any result of the amendment.

Exploratory, after the publication of arXiv:2611.15533. Predictions:
`amendments/2026-11-16_step40-predictions.md`; configuration `config_step40.yaml`
(hash 3732a848c7a8; `config.yaml` unchanged, 206fa875cca5).

**Status: STOP in part A.** B1 (the sanity check of the banded readout) is violated, which
the amendment defines as an implementation error, not a result. B2 and B3 are therefore
not interpreted. Parts C and D are complete.

## 1. Order of commits

| Commit | Content | Time |
|---|---|---|
| `02e82da` | Add step 40 predictions (amendment, `config_step40.yaml`) | 2026-11-16 20:28:31 |
| `4ba33d6` | Add banded ridge and cross-validated RSA with tests | 2026-11-16 20:35:18 |

Amendment before scripts, scripts before every run. One exception to report: the real-data
report line of `tests/test_cvrsa.py` (iii) read the THINGS single presentations of sub-01
before `4ba33d6` (it is part of the test run; it computes no result of the amendment).

## 2. Tests

| Suite | Result |
|---|---|
| `tests/test_banded.py` | 5/5 (i) 1.4e-9, (ii) 1.3e-9, (iii) met, (iv) equals `fit_condition` exactly, (v) equals `DualRidge` exactly |
| `tests/test_cvrsa.py` | 4/4 (i) mean 0.028, SE 0.031; (ii) mean relative error 0.4 %; (iii) fast formula exact; (iv) indexing |
| report: sub-01 V1, Spearman(b2, b3) | 0.963 |
| `tests/test_phase2_controls.py` | 97/97 |
| `tests/test_loader_hardening.py` | 41/41 |
| `tests/test_lower_edge_gate.py` | 2/2 |
| config hash | 206fa875cca5 (unchanged) |

## 3. Part A: banded ridge (STOP on B1)

U = median r(joint banded) − median r(bank alone), mean over subjects; MSE-skill difference
alongside.

| g | Seed | U_BP | U_Random | U_BP − U_Random | U (skill) BP | U (skill) Random |
|---|---|---|---|---|---|---|
| 4 | 0 | +0.0005 | −0.0027 | +0.0032 | −0.0010 | −0.0013 |
| 4 | 1 | +0.0013 | −0.0022 | +0.0035 | −0.0010 | −0.0014 |
| 4 | 2 | +0.0018 | −0.0009 | +0.0026 | −0.0006 | −0.0009 |
| 4 | 3 | +0.0026 | −0.0021 | +0.0047 | −0.0008 | −0.0012 |
| 4 | 4 | +0.0022 | −0.0036 | +0.0058 | −0.0010 | −0.0014 |
| 4 | mean | +0.0017 | −0.0023 | +0.0040 | −0.0009 | −0.0012 |
| 1 | mean | −0.0024 | −0.0041 | +0.0017 | −0.0013 | −0.0018 |
| 8 | mean | +0.0001 | −0.0038 | +0.0039 | −0.0000 | −0.0009 |

Bank alone, median r (mean over subjects): 0.106 (g = 4), 0.082 (g = 1), 0.109 (g = 8).
Network alone at g = 4: Backprop 0.086, Random 0.040.

| Rule | g = 4 | g = 1 | g = 8 |
|---|---|---|---|
| B1 joint − bank ≥ −0.002 in every fit | **violated** (min −0.0072) | violated (min −0.0091) | violated (min −0.0071) |
| B2 U_BP > 0 in ≥ 4 of 5 seeds, mean > 0 | formally met (5 of 5), not interpreted | – | – |
| B3 U_BP − U_Random > 0 in ≥ 4 of 5 pairs, mean > 0 | formally met (5 of 5), not interpreted | – | – |

Low-edge selections at reliable voxels: none at g = 4 and g = 8; 3103 at g = 1, all on the
bank penalty, all plateau (max gain 8.4e-6), none invalid.

Where B1 fails (joint − bank below −0.002): mostly sub-01 (g = 4: 9 of 10 fits; mean
−0.0048), plus single fits in sub-02 and sub-03. In sub-02 the joint readout is on average
above the bank (+0.0037 at g = 4). The MSE-skill difference is also negative in most fits,
so the deficit is not only a property of the reported correlation.

What is established and what is not: the code reproduces the single-group fits exactly
(tests (i), (ii), (iv), (v)), so the bank-alone and network-alone paths are correct. The
amendment fixed B1 as an implementation check, and it fails. Whether the cause is an error
in the joint path or a property of the joint selection (225 penalty pairs chosen per voxel
by inner-CV MSE, which can overfit the selection and make the joint readout slightly worse
out of fold than the bank alone) is **not resolved**. This was not tested further, because
the step requires a stop here.

## 4. Part C: cross-validated RSA on the 100 test images

Data: 12 presentations per image in each of the three subjects, one per session (sessions
1 to 12); the presentation average reproduces `arm_b_out/<subject>_armb.npz` (stimuli, voxel
index, values). Reproduction gate of the extraction (20 Arm A images, ridge g = 4, sub-01,
10 checkpoints): max |Δ median r| 1.3e-9, |Δ(Random − BP)| 9.5e-10 (threshold 1e-4):
**passed**. Element-wise deviation from the Modal features ≤ 2.5e-6.

Random − BP, Spearman, mean over 3 subjects and 5 seeds, 95% stimulus-bootstrap interval:

| ROI | Model units | Cell | ρ Random | ρ BP | Random − BP [95% CI] |
|---|---|---|---|---|---|
| V1 | raw | correlation vs b1 | 0.066 | 0.083 | −0.018 [−0.060, +0.033] |
| V1 | raw | euclidean vs b2 | 0.110 | 0.132 | −0.022 [−0.082, +0.040] |
| V1 | raw | euclidean vs b3 | 0.106 | 0.129 | −0.023 [−0.088, +0.042] |
| V1 | z | correlation vs b1 | 0.077 | 0.150 | −0.073 [−0.121, −0.028] |
| V1 | z | euclidean vs b2 | 0.133 | 0.163 | −0.030 [−0.087, +0.027] |
| V1 | z | euclidean vs b3 | 0.129 | 0.166 | −0.037 [−0.096, +0.020] |
| LOC | raw | correlation vs b1 | −0.003 | 0.013 | −0.016 [−0.043, +0.014] |
| LOC | raw | euclidean vs b2 | −0.011 | −0.023 | +0.012 [−0.050, +0.080] |
| LOC | raw | euclidean vs b3 | −0.013 | −0.018 | +0.005 [−0.058, +0.071] |
| LOC | z | correlation vs b1 | 0.022 | 0.028 | −0.007 [−0.028, +0.014] |
| LOC | z | euclidean vs b2 | −0.009 | −0.012 | +0.004 [−0.040, +0.049] |
| LOC | z | euclidean vs b3 | −0.009 | −0.005 | −0.004 [−0.045, +0.037] |

| Rule (point estimates) | V1 (primary) | LOC (descriptive) |
|---|---|---|
| C1 raw, correlation vs b1: Random − BP > 0 | **not met** (−0.018) | not met (−0.016) |
| C2 z, correlation vs b1: Random − BP ≤ 0 | met (−0.073) | met (−0.007) |
| C3 raw, Euclidean: same sign vs b2 and b3 | met (−0.022, −0.023) | met (+0.012, +0.005) |
| C3 z, Euclidean: same sign vs b2 and b3 | met (−0.030, −0.037) | not met (+0.004, −0.004) |

On the 100 presentation-averaged test images, the untrained advantage of the paper cell does
not appear (C1 not met; its interval includes 0). Only the standardised correlation cell
resolves a difference (Backprop ahead). The cross-validated distance (b3) gives the same sign
as the Euclidean distance on averaged patterns (b2) at V1.

## 5. Part D: NSD feasibility

**Go, conditional on the access agreement** (details and links in
`reports/STEP40_NSD_FEASIBILITY.md`): 515 images seen three times by all 8 subjects (907 at
least once); betas (1.8 mm, version 3) 128 GB as nii.gz for all 8 subjects, more than the 107 GB
free on this machine, so the volumes have to be processed one session at a time; extraction and
RSA take minutes to hours on this CPU. The agreement has to be signed by the author.

## 6. Runtimes and open points

| Part | Runtime |
|---|---|
| Extraction gate (20 images, 10 checkpoints) and 20 gate fits | 16 s and 245 s |
| Extraction of the 100 test images | 39 s |
| cvRSA, all cells with 1000 resamples (including reading the raw data of sub-02, sub-03) | 601 s |
| Banded ridge, g = 4 / 1 / 8 (8 workers, wall) | 1545 / 2169 / 1357 s |
| Banded ridge, sum of fit times (CPU) | 10.7 h (estimate before the run: 4.0 h; the estimate came from single-threaded fits without parallel contention) |

Open:
- B1 violation: implementation error or selection effect is not resolved (see part A).
  Possible next check, not run: the joint readout with α_net fixed at 1e12 on the real data,
  which must reproduce bank alone; and the inner-CV MSE of the selected joint pair compared
  with the bank-alone pair per voxel.
- `bench/` does not exist in the repository; the NSD estimates use the timings of this step.

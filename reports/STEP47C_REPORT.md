# Step 47c: decomposition of run centring

Predictions: `amendments/2026-11-22_step47c-predictions.md` (Z1–Z3 and the interpretation rule),
`config_step47c.yaml` (hash ea2fd24bd0a1). Gates and tests: `reports/STEP47C_SETUP.md`.
`config.yaml` (206fa875cca5), earlier configs, amendments, results and `paper/` are unchanged.
THINGS-720 only; no NSD data and no voxel-level file were written.

## Order of commits

| Commit | Time (local, +0100) | Content |
|---|---|---|
| `71585be` | 2026-11-22 11:24:32 | Add step 47c predictions (amendment, config, setup report with the Part 0 gates) |
| `a7f5b84` | 2026-11-22 11:26:27 | Add step 47c script with tests (7/7 passed) |
| this commit | | Add step 47c results |

## Part 0

- Multiverse cell raw / correlation / center720, V1, NATIVE 224 (`results/step7/multiverse.csv`):
  **+0.004573**, 95 % interval [−0.003386, +0.012460].
- Gates (repeated in the run): m + d_r = μ_r, max |Δ| 2.2e-16; Δ(V3) = step 47 run-centred
  (|d| 1.1e-17 in V1, 8.0e-17 in LOC); Δ(V0) = paper cell +0.030446 (|d| 4.2e-17; LOC 7.5e-17).

## Variants (Δ = Random − BP, paper cell, cross-run pairs)

V1 (Conv1 models):

| Variant | sub-01 | sub-02 | sub-03 | mean |
|---|---|---|---|---|
| V0 nothing removed (paper cell) | +0.033014 | +0.050909 | +0.007415 | **+0.030446** |
| V1 remove m | +0.004367 | +0.009320 | +0.001256 | **+0.004981** |
| V2 remove d_r | +0.032461 | +0.065394 | +0.007707 | **+0.035187** |
| V3 remove m and d_r (run centring) | +0.003654 | +0.015718 | +0.001725 | **+0.007032** |
| V4 = V3 + d_r∥ | +0.004266 | +0.009899 | +0.001697 | **+0.005287** |
| V5 = V3 + d_r⊥ | +0.003750 | +0.012638 | +0.001297 | **+0.005895** |

LOC (Conv3 models):

| Variant | sub-01 | sub-02 | sub-03 | mean |
|---|---|---|---|---|
| V0 | −0.001590 | −0.024217 | −0.008032 | −0.011280 |
| V1 | −0.003427 | −0.005751 | −0.003984 | −0.004388 |
| V2 | −0.001670 | −0.029881 | −0.007385 | −0.012979 |
| V3 | −0.004043 | −0.003987 | −0.004409 | −0.004146 |
| V4 | −0.004065 | −0.004233 | −0.004366 | −0.004221 |
| V5 | −0.003413 | −0.005176 | −0.004025 | −0.004205 |

## Predictions

| Prediction | Result | Status |
|---|---|---|
| Z1: Δ(V2) ≤ +0.015 | +0.0352 | not met |
| Z2: Δ(V4) ≥ +0.020 and Δ(V5) ≤ +0.012 | V4 +0.0053, V5 +0.0059 | not met |
| Z3: across-run SD of d_r · m̂ smallest in sub-03 | sub-01 2.07, sub-02 3.94, sub-03 0.64 | met |

**Interpretation by the amendment rule:** Z1 is not met and Δ(V1) = +0.0050 ≤ +0.015. The effect of
run centring is attributed mainly to removing the session mean profile (equivalent to voxel
centring), not to between-run structure. By the amendment, the description of step 47 as "run
structure" is to be corrected in the manuscript (not done in this step; `paper/` unchanged).

Removing the run deviations alone leaves the advantage in place (V2 +0.0352, above V0), and adding
back either component of d_r to the run-centred patterns changes Δ by less than 0.001. Removing m
alone gives +0.0050, close to the multiverse center720 cell (+0.0046; there the centring is over the
720 images, here over the 820 trials of the session).

## Reported without prediction

Run deviations along m̂ (session trials, after the existing z-score):

| Subject | ROI | voxels | ‖m‖ | mean ‖d_r‖ | SD of d_r · m̂ | share of Σ‖d_r‖² along m̂ |
|---|---|---|---|---|---|---|
| sub-01 | V1 | 1049 | 3.53 | 6.24 | 2.07 | 0.108 |
| sub-02 | V1 | 1104 | 4.49 | 7.50 | 3.94 | 0.248 |
| sub-03 | V1 | 839 | 1.89 | 4.78 | 0.64 | 0.017 |
| sub-01 | LOC | 2700 | 3.02 | 8.93 | 0.41 | 0.002 |
| sub-02 | LOC | 1175 | 4.74 | 7.80 | 2.95 | 0.134 |
| sub-03 | LOC | 423 | 1.25 | 3.47 | 0.34 | 0.010 |

Spearman between the per-trial cosine of the pattern with m̂ and the image luminance L (720
images); mean cosine in parentheses:

| Subject | ROI | without run centring (V0) | with run centring (V3) |
|---|---|---|---|
| sub-01 | V1 | −0.191 (0.109) | −0.195 (0.002) |
| sub-02 | V1 | −0.160 (0.135) | −0.179 (0.003) |
| sub-03 | V1 | −0.222 (0.066) | −0.243 (0.000) |
| sub-01 | LOC | −0.056 (0.059) | −0.049 (0.000) |
| sub-02 | LOC | +0.040 (0.133) | +0.043 (−0.001) |
| sub-03 | LOC | +0.001 (0.060) | −0.009 (−0.001) |

## Runtime

143 s in total (models and luminance 100 s), estimate below 0.1 h.

## Outputs (`results/step47c/`)

`decomposition.csv` (variants per subject and mean, run-deviation quantities, cosine–luminance
correlations, checks), `summary.csv` (Z1–Z3, interpretation, runtimes). Both have `config_hash`,
`git_head` and `seed`.

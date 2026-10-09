# Step 6 — ridge across resolutions, mechanism, robustness

**Exploratory.**
- Predictions and rules: `amendments/2026-10-09_step6-predictions.md` (commit `73c13f8`,
  before any step 6 script).
- Audit of published claims: `reports/AUDIT_RAW_RSA_CLAIMS.md` (commit `5ccef4a`).
- Config hash `206fa875cca5`.
- Conventions as in step 5: per subject on all ROI voxels for RSA, reliable voxels for
  ridge; cross-run pairs; 1000 paired stimulus-bootstrap draws.
- N = 3 subjects; inference over stimuli, not over the population.

## P_ridge — ridge across resolutions (`results/step6/ridge_resolution.csv`)

**Setup.**
- GAP features (Conv1, g = 1) from `STEP5_GAP_DIR`, V1.
- `fit_condition` unchanged: default folds, D1 in counting mode; Y = all V1 voxels,
  score = median r over the reliable voxels.

**Checks.**
- NATIVE 224 GAP features equal `pilot_features` g = 1 (max |Δ| 1.4e-6).
- sub-01 NATIVE 224 reproduces `tolerance_all_rungs.csv` (max |Δ| 2.2e-8).
- Runtime estimate 74 min (630 fits), so all seven conditions were run.

Median r, mean over seeds:

| Subject | Condition | 32 | 64 | 96 | 128 | 160 | 224 | UP224 |
|---|---|---|---|---|---|---|---|---|
| mean sub-01/02 | Backprop | 0.0810 | 0.1016 | 0.1054 | 0.1043 | 0.1016 | 0.0954 | 0.0369 |
| | Random Weights | 0.0553 | 0.0754 | 0.0795 | 0.0793 | 0.0760 | 0.0696 | 0.0182 |
| | Random (BN-cal) | 0.0551 | 0.0751 | 0.0787 | 0.0778 | 0.0746 | 0.0681 | 0.0183 |
| | FA | 0.0511 | 0.0690 | 0.0733 | 0.0713 | 0.0695 | 0.0628 | 0.0194 |
| | PC | 0.0587 | 0.0764 | 0.0812 | 0.0807 | 0.0777 | 0.0718 | 0.0195 |
| | STDP | 0.0551 | 0.0725 | 0.0745 | 0.0736 | 0.0717 | 0.0658 | 0.0166 |
| sub-03 | Backprop | −0.0060 | 0.0030 | 0.0064 | 0.0058 | 0.0054 | 0.0019 | −0.0345 |
| | Random Weights | −0.0133 | −0.0091 | −0.0074 | −0.0077 | −0.0070 | −0.0094 | −0.0331 |

Random − BP, mean ± SD over seeds (0/5 seeds positive in every sub-01, sub-02 and mean
cell):

| | 32 | 64 | 96 | 128 | 160 | 224 | UP224 |
|---|---|---|---|---|---|---|---|
| sub-01 | −0.0287 ± 0.0023 | −0.0265 ± 0.0042 | −0.0250 ± 0.0070 | −0.0263 ± 0.0097 | −0.0273 ± 0.0081 | −0.0288 ± 0.0082 | −0.0160 ± 0.0089 |
| sub-02 | −0.0226 ± 0.0042 | −0.0259 ± 0.0046 | −0.0267 ± 0.0067 | −0.0239 ± 0.0065 | −0.0238 ± 0.0055 | −0.0227 ± 0.0042 | −0.0214 ± 0.0039 |
| **mean sub-01/02** | **−0.0257** | **−0.0262** | **−0.0259** | **−0.0251** | **−0.0256** | **−0.0258** | −0.0187 |
| sub-03 (weak) | −0.0073 | −0.0121 | −0.0138 | −0.0135 | −0.0124 | −0.0113 | +0.0014 (4/5) |

BN-cal − BP (mean sub-01/02): −0.026 to −0.027 at every NATIVE resolution, −0.019 at
UP224.

**Applying P_ridge:** Random − BP ≤ 0 at all six NATIVE resolutions, and
S_ridge = −0.0001 ≤ 0.5 · S_raw = +0.0155. **P_ridge is met: ridge confirms z-RSA across
resolutions.** Under ridge the gap is constant (about −0.026) and has no resolution
slope.

Further observations:
- All conditions peak at 96–128 px and collapse at UP224 (BP 0.095 → 0.037). The
  fitted readout loses most of its predictivity when content is limited to 32 px.
- sub-03 is near or below zero in every condition, as in steps 4 and 5.

**Lower-edge gate** (`lower_edge_gate_ridge.csv`): 20 hits, all **Plateau** (relative
gain ≤ 4.6e-5); no cell excluded.

## Mechanism (`results/step6/mechanism_bandpass.csv`, `moment_ratio_bandpass.csv`)

**Band-pass features.**
- Inputs: NATIVE 224, only_0-14 and notch_0-14, built with the vendored `sigma`,
  `bands`, `condition` (`src/vendor/bandpass_filters.py`, verbatim from
  `bandpass_extract.py`); all 30 checkpoints.
- Gate: raw Conv1 RDMs against the paper's `data/bandpass/rdms`, 20 of 20 cells, min
  Pearson 1.00000000.

### RSA on the band-pass inputs (step-5 convention, bootstrap CIs)

| Condition | ROI | Quantity | raw [CI] | z [CI] |
|---|---|---|---|---|
| only_0-14 | V1 | Random − BP | +0.0330 [+0.0214, +0.0443] (5/5) | +0.0040 [+0.0013, +0.0069] (5/5) |
| notch_0-14 | V1 | Random − BP | −0.0206 [−0.0347, −0.0070] (0/5) | −0.0096 [−0.0176, −0.0025] (0/5) |
| only_0-14 | LOC | BP − Random | +0.0023 [−0.0039, +0.0089] | +0.0018 [−0.0021, +0.0058] |
| notch_0-14 | LOC | BP − Random | +0.0036 [−0.0049, +0.0129] | +0.0026 [−0.0005, +0.0061] |

The coarse band alone reproduces the raw V1 gap (+0.033 vs +0.030 native). Under z it
shrinks to +0.004, whose CI still excludes 0. Removing the coarse band reverses the gap
in both variants.

### m and d (V1, Conv1; mean over seeds)

| Condition | m BP | m Random | d BP (raw − z) | d Random |
|---|---|---|---|---|
| NATIVE 32 | 0.735 | 0.696 | +0.004 | +0.015 |
| NATIVE 224 | 0.638 | 0.707 | −0.017 | +0.024 |
| UPSAMPLED 224 | 0.505 | 0.639 | −0.009 | +0.024 |
| only_0-14 | 0.484 | 0.619 | −0.007 | +0.022 |
| notch_0-14 | 0.830 | 0.832 | +0.009 | −0.002 |

**Applying P_mech1:**
- BP: m(UP224) < m(NAT224) < m(NAT32) in **5/5** seeds (≥ 4 required): **met**.
- Random's range of m over the three conditions is 0.068, against 0.230 for BP;
  0.068 < 0.115: **met**.
- **P_mech1 met.**

**Applying P_mech2:**
- m_BP(notch) > m_BP(only) in **5/5** seeds: **met**.
- Spearman(m, d) over the 50 cells = **0.269 < 0.6: not met.**
- **P_mech2 not met.**
- Without a rule: within BP, Spearman(m, d) = 0.601; within Random, −0.396. In
  notch_0-14 both conditions reach m ≈ 0.83, yet d differs in sign (BP +0.009,
  Random −0.002). The m index orders BP's cells but not Random's, so it is not a
  sufficient common description across conditions.

### Filter spectra (`filter_spectra.csv`, descriptive)

Conv1, each 3 × 3 kernel zero-padded to 32 × 32, |FFT| averaged over the input
channels, radially averaged; mean ± SD over 32 filters × 5 seeds.

| Condition | peak (cyc/px) | centroid (cyc/px) | centroid without DC |
|---|---|---|---|
| Backprop | 0.274 ± 0.172 | 0.264 ± 0.033 | 0.275 ± 0.028 |
| Feedback Alignment | 0.256 ± 0.179 | 0.254 ± 0.023 | 0.268 ± 0.020 |
| Predictive Coding | 0.267 ± 0.179 | 0.254 ± 0.023 | 0.268 ± 0.020 |
| Random Weights | 0.265 ± 0.180 | 0.254 ± 0.023 | 0.268 ± 0.020 |
| Random (BN-cal) | identical to Random (same weights) | | |
| STDP | 0.260 ± 0.189 | 0.254 ± 0.022 | 0.268 ± 0.020 |

The conditions barely differ. BP's centroid is about 0.01 cyc/px higher, which is small
against the filter-to-filter SD. 3 × 3 kernels cannot express much spectral structure.

### Stimulus spectra (`stimulus_spectra.csv`, descriptive)

Radially averaged luminance power, summed over bands (cycles per pixel):

| Input | ≤ 1/16 (incl. DC) | 1/16 – 1/4 | > 1/4 |
|---|---|---|---|
| NATIVE 32 | 2.62e5 | 459 | 34.7 |
| NATIVE 224 | 6.31e8 | 8.81e4 | 4.83e3 |
| UPSAMPLED 224 | 6.30e8 | 8.96e3 | 490 |

UPSAMPLED 224 keeps the low band of NATIVE 224 and has about 10 × less power above
1/16 cycles per pixel. At 224 px a 3 × 3 Conv1 kernel therefore sees mostly
low-frequency input once content is limited.

## Robustness (`results/step6/robustness.csv`, descriptive)

**Columns.** Gap per cell: Random − BP (V1) or BP − Random (LOC), with 95% CI and seeds
positive. "fMRI 720" means betas re-z-scored per voxel over the 720 stimuli.
**Check.** The paper-convention cells equal step 5 to 1e-16.

| Feature norm | Distance | fMRI | V1 32 | V1 224 | LOC 224 |
|---|---|---|---|---|---|
| raw | correlation | all trials | −0.0007 [−0.011, +0.009] | **+0.0304 [+0.018, +0.042]** | +0.0113 [+0.003, +0.021] |
| raw | cosine | all trials | +0.0031 [−0.005, +0.011] | +0.0072 [−0.001, +0.015] (4/5) | +0.0089 [+0.001, +0.017] |
| raw | euclidean | all trials | **+0.0221 [+0.012, +0.032]** | **+0.0294 [+0.019, +0.039]** | +0.0034 [−0.004, +0.011] |
| center | correlation | all trials | −0.0020 [−0.008, +0.004] | +0.0042 [−0.002, +0.011] (5/5) | +0.0062 [+0.001, +0.011] |
| center | cosine | all trials | −0.0059 [−0.014, +0.002] | −0.0002 [−0.008, +0.008] | +0.0066 [+0.001, +0.012] |
| center | euclidean | all trials | **+0.0221 [+0.012, +0.032]** | **+0.0294 [+0.019, +0.039]** | +0.0034 [−0.004, +0.011] |
| scale | correlation | all trials | −0.0094 [−0.018, −0.001] | −0.0082 [−0.017, +0.001] | +0.0104 [+0.003, +0.018] |
| scale | cosine | all trials | −0.0060 [−0.012, +0.000] | −0.0097 [−0.017, −0.003] | +0.0083 [+0.002, +0.015] |
| scale | euclidean | all trials | **+0.0127 [+0.005, +0.020]** | **+0.0185 [+0.009, +0.028]** | +0.0052 [−0.001, +0.012] |
| z | correlation | all trials | −0.0123 [−0.018, −0.007] | −0.0107 [−0.017, −0.005] | +0.0066 [+0.002, +0.011] |
| z | cosine | all trials | −0.0141 [−0.023, −0.006] | −0.0146 [−0.024, −0.006] | +0.0064 [+0.002, +0.011] |
| z | euclidean | all trials | **+0.0127 [+0.005, +0.020]** | **+0.0185 [+0.009, +0.028]** | +0.0052 [−0.001, +0.012] |
| raw | correlation | **fMRI 720** | −0.0013 [−0.007, +0.005] | +0.0048 [−0.003, +0.013] (5/5) | +0.0044 [−0.001, +0.010] |
| z | correlation | **fMRI 720** | −0.0141 [−0.019, −0.009] | −0.0139 [−0.020, −0.009] | +0.0069 [+0.003, +0.011] |

Euclidean distance does not depend on centring, so center = raw and z = scale in that
column, as the table shows.

**Where the V1 gap at 224 px stays positive with a CI excluding 0:**
- raw / correlation (the paper's convention);
- every euclidean cell (raw, center, scale, z: +0.018 to +0.029).

It stays positive without a CI in raw / cosine, center / correlation, and raw /
correlation with fMRI 720.

Three observations:
1. **Under euclidean distance the gap survives channel standardization**, and it is
   already positive at 32 px (+0.013 to +0.022). The slope 32 → 224 is therefore small
   (+0.006 to +0.007), against +0.031 for raw / correlation.
2. **Re-z-scoring the fMRI betas over the 720 stimuli alone removes most of the raw V1
   gap** (+0.030 → +0.005, CI includes 0). The headline thus also depends on how the
   brain responses were normalized, not only on the model features.
3. **The LOC advantage of BP is positive in all 14 cells** (5/5 seeds each); its CI
   excludes 0 in 9 of them (not in the four euclidean cells, and not raw with fMRI 720).

## Summary of rules

| Rule | Criterion | Result | Verdict |
|---|---|---|---|
| P_ridge | Random − BP ≤ 0 at all 6 NATIVE res and S_ridge ≤ 0.5 · S_raw | −0.025 to −0.026 at all six; S_ridge −0.0001 | **met** |
| P_mech1 (BP) | BP order m(UP224) < m(NAT224) < m(NAT32) | 5/5 seeds | **met** |
| P_mech1 (Random) | Random range < 0.5 × BP range | 0.068 vs 0.230 | **met** |
| P_mech2 (BP) | m_BP(notch) > m_BP(only) | 5/5 seeds | met |
| P_mech2 (all cells) | Spearman(m, d) ≥ 0.6 | 0.269 | **not met** → P_mech2 not met |

## Limits

- N = 3 subjects; inference over stimuli only; seeds averaged inside each draw.
- sub-03 is near zero under ridge in every condition.
- Only GAP features, Conv1 → V1 and Conv3 → LOC; one architecture and training set.
- The m index is a summary of channel moments; P_mech2 shows that it does not describe
  the Random condition.
- The robustness grid is descriptive and not corrected for multiplicity. It shows that
  the z-RSA result is specific to correlation-type distances (correlation, cosine).
- Run-grouped folds are not preregistered and were not used here.

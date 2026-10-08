# Step 5 — does the resolution effect of arXiv:2608.12408 run through channel moments?

**Exploratory.**
- Predictions and rules: `amendments/2026-10-08_step5-resolution-predictions.md` (commit
  `7b170b1`, made before any step 5 script).
- Config hash `206fa875cca5`.
- Inference is over stimuli (1000 bootstrap resamples) in N = 3 subjects, not over the
  population.

## Setup

- **Features:** GAP features of all 30 v12 checkpoints, extracted locally with the
  paper's own code (`extract_features`, `_tf`, model classes lifted unchanged from the
  vendored Modal scripts), at NATIVE 32–224 px and UPSAMPLED 224 px.
- **Reproduction gate:** raw model RDMs from these features against the stored paper
  RDMs, 840 of 840 cells, min Pearson 1.00000000.
- **RSA:** as in paper v3, i.e. per subject on all voxels of the ROI, cross-run pairs
  (210,205), then the mean over subjects; gap = mean over the 5 seeds of the per-seed
  difference.
- **Bootstrap:** the paper's 1000 resamples (`boot_idx`, sha check passed), with the pair
  set recomputed per draw and differences, slopes and ratios formed within each draw.
  The identity draw equals the point estimates (|Δ| = 0).
- **Variants:** `raw` (paper) and `z`, the z-score per channel over the 720 stimuli.
  Dead channels: none.

## Rules

| Rule | Quantity | Value [95% CI] | Verdict |
|---|---|---|---|
| R1 | raw V1 NAT224 Random − BP | +0.0304 [+0.0181, +0.0423]; NAT32 −0.0007 | **passed** (reproduces +0.030 [0.018, 0.042]) |
| R2 | z V1 NAT224 Random − BP | **−0.0107 [−0.0171, −0.0046]** | upper bound ≤ 0.010: **H_mom explains the headline** |
| R3 | slope S = gap224 − gap32 | S_raw +0.0311 [+0.0239, +0.0384]; S_z **+0.0016 [−0.0038, +0.0070]** | S_z ≤ 0.5·S_raw: **resolution effect at V1 predominantly moment-driven** |
| R4 | z LOC NAT224 BP − Random | **+0.0066 [+0.0024, +0.0109]** (5/5) | **survives standardization** |
| R5 | z V1 UP224 Random − BP | +0.0017 [−0.0013, +0.0048] | upper bound ≤ 0.010: **met** |
| M1 | Spearman(m, d), Random + BP, NATIVE × 6 res (n = 60) | **0.680** | ≥ 0.6: **met** |
| M1 (no rule) | within Random / within BP (n = 30 each) | 0.683 / 0.619 | — |

Under z the V1 Random − BP gap is **negative at every resolution** (0/5 seeds positive
at each):

| res (NATIVE) | 32 | 64 | 96 | 128 | 160 | 224 |
|---|---|---|---|---|---|---|
| V1 Random − BP, raw | −0.0007 | +0.0097 | +0.0175 | +0.0227 | +0.0257 | +0.0304 |
| V1 Random − BP, z | −0.0123 | −0.0158 | −0.0156 | −0.0145 | −0.0131 | −0.0107 |
| LOC BP − Random, raw | +0.0122 | +0.0143 | +0.0132 | +0.0123 | +0.0121 | +0.0113 |
| LOC BP − Random, z | +0.0070 | +0.0084 | +0.0079 | +0.0074 | +0.0071 | +0.0066 |

Only NATIVE 32, NATIVE 224 and UPSAMPLED 224 have bootstrap CIs; the other
resolutions are point estimates.

## Paper comparisons, raw vs z (`results/step5/paper_comparisons_raw_vs_z.csv`)

**Scope.** These are the model-based quantities that paper v3 reports with a
stimulus-bootstrap CI (NUMBERS_MANIFEST_v3 entries with `.lo/.hi` from `step4_gaps` and
`step4_ratio`): 8 of them. Added: the NATIVE LOC gap that the LOC ratio is built on, the
LOC gap at 32 px, and the V1 slope S. **The expected count of 11 could not be confirmed
from the code or the paper text; no list of "primary" comparisons exists there.**

The table is descriptive: there is no multiplicity correction and no verdict on the
paper.

| Quantity (paper key) | raw [CI] | z [CI] | Change under z |
|---|---|---|---|
| V1 gap NAT32 (`boot.V1.32`) | −0.0007 [−0.0109, +0.0090] | −0.0123 [−0.0182, −0.0067] | CI now excludes 0, negative |
| V1 gap NAT224 (`boot.V1.224`) | +0.0304 [+0.0181, +0.0423] | −0.0107 [−0.0171, −0.0046] | **sign reverses** |
| V1 gap UP224 (`s4.V1.up224`) | +0.0347 [+0.0229, +0.0464] | +0.0017 [−0.0013, +0.0048] | **loses CI** (≈ 0) |
| V1 R = UP/NAT (`s4.V1.R`) | 1.139 [1.032, 1.341] | −0.159 [−0.863, +0.088] | **loses CI**; meaningless with a negative denominator |
| V1 UP − NAT (`s4.V1.diff`) | +0.0042 [+0.0011, +0.0080] | +0.0124 [+0.0078, +0.0182] | same sign, larger |
| V1 slope S (224 − 32) | +0.0311 [+0.0239, +0.0384] | +0.0016 [−0.0038, +0.0070] | **loses CI** |
| LOC gap NAT32 | +0.0122 [+0.0048, +0.0195] | +0.0070 [+0.0027, +0.0117] | holds, smaller |
| LOC gap NAT224 (underlies `s4.LOC.R`) | +0.0113 [+0.0025, +0.0210] | +0.0066 [+0.0024, +0.0109] | holds, smaller |
| LOC gap UP224 (`s4.LOC.up224`) | +0.0026 [−0.0036, +0.0089] | +0.0020 [−0.0019, +0.0058] | n.s. in both |
| LOC R (`s4.LOC.R`) | 0.227 [−0.578, +0.826] | 0.301 [−0.588, +0.705] | n.s. in both |
| LOC UP − NAT (`s4.LOC.diff`) | −0.0087 [−0.0165, −0.0013] | −0.0046 [−0.0077, −0.0016] | holds, smaller |

## Moment ratio m (`results/step5/moment_ratio.csv`)

m = ‖μ‖² / (‖μ‖² + Σσ²) of the Conv1 GAP features, mean over seeds.

| Condition | 32 | 64 | 96 | 128 | 160 | 224 | UP224 |
|---|---|---|---|---|---|---|---|
| Backprop | 0.735 | 0.693 | 0.673 | 0.660 | 0.651 | 0.638 | 0.505 |
| Random Weights | 0.696 | 0.702 | 0.704 | 0.705 | 0.706 | 0.707 | 0.639 |
| Random (BN-cal) | 0.710 | 0.701 | 0.697 | 0.695 | 0.694 | 0.692 | 0.610 |
| Feedback Alignment | 0.714 | 0.706 | 0.702 | 0.700 | 0.700 | 0.698 | 0.617 |
| Predictive Coding | 0.735 | 0.729 | 0.726 | 0.724 | 0.723 | 0.721 | 0.649 |
| STDP | 0.632 | 0.618 | 0.613 | 0.610 | 0.608 | 0.605 | 0.496 |

Descriptive reading: m of Backprop falls with resolution (0.735 → 0.638), while m of
Random stays flat (~0.70). Over the same range, d = RSA_raw − RSA_z goes from +0.004 to
−0.017 for BP and from +0.015 to +0.024 for Random.

## sub-03 diagnosis (`results/step5/sub03_diagnosis.csv`)

Descriptive, no rule. Ridge on sub-03, 286 reliable V1 voxels; median r over those
voxels, mean over seeds (controls: one set).

| Features | g = 1 | g = 4 | g = 8 |
|---|---|---|---|
| Backprop | +0.0019 (step 4) | +0.0132 | +0.0142 |
| Random Weights | −0.0094 (step 4) | −0.0185 | −0.0156 |
| pixels | −0.0238 | −0.0279 | −0.0276 |
| luminance + contrast | **+0.0227** | **+0.0302** | **+0.0351** |
| Gabor | −0.0290 | −0.0525 | −0.0417 |

Reliability of the reliable V1 voxels (Arm B split-half, Spearman–Brown):

| Subject | reliable / V1 | Q25 | median | Q75 | median, all V1 |
|---|---|---|---|---|---|
| sub-01 | 501 / 1049 | 0.319 | 0.476 | 0.655 | 0.176 |
| sub-02 | 518 / 1104 | 0.282 | 0.455 | 0.647 | 0.179 |
| sub-03 | 286 / 839 | 0.262 | **0.349** | **0.450** | 0.108 |

Reading:
- sub-03's reliable voxels are less reliable than those of the other two subjects
  (median 0.35 vs 0.46–0.48).
- The only feature bank that ridge predicts above zero at every g is the 2-channel
  luminance + contrast bank (r up to +0.035). Random features and the higher-dimensional
  low-level banks (pixels, Gabor) are negative, consistent with overfitting at low
  signal.
- BP is slightly positive at g = 4 and g = 8.
- So sub-03's V1 signal at these stimuli is weak and mostly low-dimensional.
- Lower-edge gate: 3 hits (Gabor, g = 1), all Plateau (gain ≤ 1.3e-5); no cell
  excluded.

## Deviations

- **Feature directory.** `$RIDGE_DATA_DIR` points into the read-only source repository,
  so the features are written to a new variable `STEP5_GAP_DIR`
  (`ridge-readout/data/step5_gap`, gitignored) instead of `$RIDGE_DATA_DIR/step5_gap`.
- **Paper comparisons.** There are 8 manifest comparisons plus 3 added ones, not 11
  (see the table).
- **Bootstrap scope.** Bootstrap CIs exist only for NATIVE 32, NATIVE 224 and UPSAMPLED
  224; all other resolutions and the V2 and IT cells are point estimates.

## Limits

- N = 3 subjects; inference over stimuli, not over the population.
- Seeds are averaged inside each draw; there is no seed-level test here.
- Conv1 → V1 and Conv3 → LOC only for the rules; GAP features only. The z variant
  changes the feature metric, not the network.
- H_mom is supported as a description: standardizing channel moments removes the V1 gap
  and its resolution slope. This does not identify why Backprop's channel moments shift
  with resolution.
- sub-03 is weak in both readouts.

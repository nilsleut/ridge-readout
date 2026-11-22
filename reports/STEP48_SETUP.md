# Step 48 setup: feasibility and gates

Checked before the amendment `amendments/2026-11-22_step48-predictions.md`. Only the checks and
timings below were computed; no result file was written. Configuration: `config_step48.yaml`
(hash 159c5c78e854). `config.yaml` (206fa875cca5) unchanged.

## 1. THINGS: unit of the h5 values

The h5 files (`ResponseData/block0_values`, float32, 211,339 voxels × 9840 trials, pandas/PyTables
frame) carry no unit attribute, and no documentation of the dataset is present locally (the dataset
folder holds the h5, voxel and stimulus metadata and the images only). A web search did not find a
description of the file either; the THINGS-fMRI papers describe single-trial GLMsingle betas
estimated on time series converted to percent signal change and then rescaled.

Statistics of the stored values (ROI rows of arm_a, per voxel, ddof 0):

| Subject | ROI | voxels | mean over all 9840 trials: mean (SD over voxels) | SD over all trials: mean (range) | session means: max \|.\| | session SDs: range |
|---|---|---|---|---|---|---|
| sub-01 | V1 | 1049 | +0.0127 (0.031) | 0.070 (0.058–0.090) | 0.147 | 0.053–0.095 |
| sub-01 | LOC | 2700 | +0.0147 (0.016) | 0.066 (0.057–0.098) | 0.112 | 0.052–0.101 |
| sub-02 | V1 | 1104 | +0.0075 (0.028) | 0.069 (0.058–0.095) | 0.156 | 0.054–0.100 |
| sub-02 | LOC | 1175 | +0.0205 (0.023) | 0.065 (0.058–0.090) | 0.148 | 0.053–0.097 |
| sub-03 | V1 | 839 | +0.0080 (0.026) | 0.069 (0.060–0.091) | 0.125 | 0.054–0.093 |
| sub-03 | LOC | 423 | +0.0142 (0.013) | 0.063 (0.058–0.087) | 0.089 | 0.053–0.091 |

The values are **not z-scored by the dataset**, neither globally nor per session: means are not 0
and SDs are far from 1, in all trials and within sessions. Unit recorded: single-trial GLMsingle
betas as stored, not normalised; the absolute unit is not documented locally.

**Gate:** our z-score of the h5 values ((x − mean) / (std + 1e-8) per voxel over the 9840 trials,
float32) gives the arm_a betas **exactly** (`np.array_equal`) in all three subjects. Passed.

## 2. NSD: reload with raw betas

The step 42 / 47b files hold only z-scored trials and run means; the raw betas are needed again.
`scripts/nsd_raw_fetch.py` will, per session (1–30, 8 subjects), download the betas file (the step 42
key and pipelined download), take the V1 ∪ lateral voxels and store under `NSD_DIR/raw48/<subject>/`:

- Q: int16 betas of the trials of the 515 images (lossless);
- R = Q / 300 as float32 (% signal change, the form asked for);
- mu, sd: per-voxel mean and SD (ddof 0) of Q / 300 over all 750 trials (float64);
- valid = sd > 0.

Then the original is deleted. N-global is pooled from the session statistics (equal trial counts).

**Gate per session** (first: subj01 session 1): z = (Q / 300 − mu) / sd in float64, cast to float32,
must equal the step 42 trials exactly. This is the expression of `nsd_fetch.process`. The float32
path (R) is reported separately, since float32(Q / 300) is not exact. The gate needs the first NSD
download, which happens only after the amendment commit; its result is added in the scripts commit.

**Estimate** from the step 47b reload of the same 240 files (`results/step47b/nsd_runmeans_log.csv`):
2.34 h wall time (108.7 GB download, processing overlapped) → **3.5 h** with × 1.5, below 8 h.
Disk: at most two files of ≤ 0.55 GB at once; stored per session ≈ 52 trials × 7260–10,157 voxels ×
6 B + 2 × 8 B per voxel ≈ 3 MB, **≈ 0.8 GB** in total. Free space 116 GB (required ≥ 30 GB).

## 3. Image statistics (`scripts/image_stats.py`)

Transform as step 44 (`colour_statistic_control.load_rgb`, Resize 224 + CenterCrop 224, Rec. 709
grey). L and C reproduce `step44_why.lum_contrast` exactly (max |Δ| 0) for all three sets.

- **CE:** the repository's existing control bank (`pilot_features_modal.gabor_bank`) has 4
  orientations × **3** spatial frequencies (0.10, 0.20, 0.40 cycles / px, cosine phase,
  σ = 1 / (3 sf), 21 × 21). For 4 × 4 its formula is kept and one octave below is added
  (0.05 cycles / px). Kernels are 41 × 41 so the widest envelope fits (3σ = 20 px). The kernel mean
  is subtracted so a uniform image gives 0 (the existing bank is not zero-mean and would pass
  luminance), with reflect padding. CE = mean over pixels and the 16 filters of the squared
  response.
- **OFF / ON / DD:** with the image's own mean as reference, OFF = ON for every image
  (Σ(x − mean) = 0; max |OFF − ON| 7e-17 on the first 20 images of each set), so DD = 0.5
  identically. Reference fixed instead: **the mean grey over all images of the dataset** (THINGS-720
  0.4812, test 0.4982, NSD-515 0.4718). Then ON − OFF = L − ref, and DD is close to a mirror of L
  (Spearman L–DD −0.98 to −0.99).

| Set | n | time | L mean (SD) | C | CE | OFF | ON | DD |
|---|---|---|---|---|---|---|---|---|
| THINGS-720 | 720 | 75 s | 0.481 (0.122) | 0.207 (0.055) | 10.8 (6.7) | 0.103 (0.065) | 0.103 (0.072) | 0.507 (0.258) |
| THINGS test | 100 | 6 s | 0.498 (0.120) | 0.205 (0.056) | 10.3 (6.5) | 0.103 (0.067) | 0.103 (0.068) | 0.502 (0.256) |
| NSD-515 | 515 | 34 s | 0.472 (0.109) | 0.220 (0.053) | 11.6 (6.2) | 0.105 (0.057) | 0.105 (0.065) | 0.508 (0.236) |

Spearman among the statistics (THINGS-720): L–C −0.11, L–CE −0.16, C–CE +0.66, L–DD −0.98; NSD:
L–C +0.06, L–CE −0.08, C–CE +0.61, L–DD −0.99.

## 4. Runtime estimates (× 1.5)

| Part | basis | estimate |
|---|---|---|
| THINGS conventions (720, 3 conventions × 4 cells × 2 ROIs; test images 3 conventions × k 1, 12) | step 47c 143 s, step 47b test images 3 s | < 0.1 h |
| Image statistics | 115 s above | < 0.1 h |
| NSD reload | step 47b log | 3.5 h |
| NSD RSA, 3 conventions × 8 subjects, k = 1 and 3 | step 47b preparation 62 s per subject for 2 variants at k = 3 | 0.4 h |
| NSD multiverse N-raw | step 47b 954 s | 0.4 h |
| Amplitude | cosines and Spearman, seconds | < 0.1 h |

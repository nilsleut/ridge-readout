# Step 47 setup: feasibility and gates (read only)

Checked before the amendment `amendments/2026-11-21_step47-predictions.md`. Nothing was computed
except the gates and timings below; no result file was written. Configuration: `config_step47.yaml`.

## 1. Run means for THINGS

`{subject}_task-things_voxel-wise-responses.h5`, dataset `ResponseData/block0_values`, shape
(voxels, 9840 trials), float32, not chunked (sub-01 211,339, sub-02 226,950, sub-03 189,164 voxels).
`arm_b_betas.load_arm_a` already reads only the ROI rows (`[roi_idx, :]`, all 9840 trials) and
z-scores every voxel over all trials; reading the ROI rows takes about 5 s per subject. The full
files are never loaded.

So the responses of **all trials of a run** are available after the same normalisation: in the
session of the 720 every run has 82 trials in the metadata (72 main trials of the 720, 10 test
trials). Catch trials are not part of the THINGS-fMRI single-trial data.

**Choice (goes into the amendment): run mean over all trials of the run (82: 72 main + 10 test).**

## 2. Gate: arm_a betas = z-scored h5 response

`load_arm_a` recomputed from the h5 files against the stored `RIDGE_DATA_DIR/arm_b_out/{s}_arma.npz`:

| | sub-01 | sub-02 | sub-03 |
|---|---|---|---|
| ROI index identical | yes | yes | yes |
| max \|Δ\|, 20 random trials | 0 | 0 | 0 |
| max \|Δ\|, all 720 trials | 0 | 0 | 0 |
| correlation-distance RDM of the stored V1 betas vs `outputs_720/fmri_rdm_V1_{s}.npy` | 0 | 0 | 0 |

Passed (threshold 1e-5). The last row means the brain RDMs of the paper cell are exactly the
correlation distances of the arm_a betas, so run-centred RDMs can be built from the same betas.

## 3. Predecessor trials

Predecessor = the row immediately before in the metadata, same session and same run, any trial type.

| | sub-01 | sub-02 | sub-03 |
|---|---|---|---|
| trials with a predecessor | 711 / 720 | 711 / 720 | 710 / 720 |
| predecessors that are test images | 86 | 86 | 88 |
| predecessors that are main (train) images | 625 | 625 | 622 |
| predecessor images found (v12 `find_img`, `THINGS_IMAGES_DIR`) | all | all | all |

The 9–10 trials without a predecessor are the first trials of a run. No predecessor image is
missing, so no STOP. Luminance of the predecessor images: MEANLUM of step 44
(`colour_statistic_control.load_rgb` / `build_references`).

## 4. NSD ridge

`fit.fit_condition` (dual ridge, 8 × 5 nested CV, seed 20260818, α grid 1e-6…1e8, 29 points,
min-MSE selection) on random data with 515 stimuli and 1400 voxels, `assert_low_boundary=False`:

| features d | time per fit |
|---|---|
| 32 (g = 1) | 5.1 s |
| 512 (g = 4) | 5.5 s |
| 2048 (g = 8) | 6.0 s |

Runs without error. Features for g = 4 and 8 on the 515 NSD images come from the existing builder
`nsd_features.build` (the prereg extraction code; g = 1 extraction took 201–300 s in step 42).

Runtime estimate:
- **Part B:** 8 subjects × 33 fits (10 models × 3 g, plus the luminance-and-contrast bank × 3 g)
  × 2 ROIs × about 7 s gives about 1.0 h. With extraction (about 0.2 h) and the lower-edge gate, the
  total × 1.5 is about 2 h.
- **Part A:** run centring and RDMs take minutes; the run-centred multiverse bootstrap (1000
  resamples, 147 vectors per draw) takes about 0.3 h × 1.5. In total under 1 h.

Both parts are below the 8 h limit.

# Step 44 setup: feasibility (read only)

Checked before the amendment `amendments/2026-11-19_step44-predictions.md`. Nothing was computed
except the gates below; no result file was written. Configuration: `config_step44.yaml`.

## 1. Presentation order of the 720 images

Source: `{subject}_task-things_stimulus-metadata.csv` (THINGS-fMRI, `THINGS_FMRI_DIR`), columns
`trial_type, session, run, subject_id, trial_id, stimulus, concept`.

| | sub-01 | sub-02 | sub-03 |
|---|---|---|---|
| metadata rows | 9840 | 9840 | 9840 |
| `trial_id` = row index, strictly increasing | yes | yes | yes |
| session of the 720 | 3 | 2 | 5 |
| all 720 in that one session | yes | yes | yes |
| trials in that session | 820 (720 train, 100 test) | 820 | 820 |
| runs × trials per run | 10 × 82 | 10 × 82 | 10 × 82 |

- sub-01 and sub-02: identical run and identical position for all 720 images.
- sub-03: a different order; 11.8 % of the 720 images lie in the same run number as in sub-01.
- Position within the run is available as the row order within the run (`trial_id`); test
  trials are interspersed irregularly within each run. The metadata have no onset column and no
  events files are on disk, so the order is the row order, not onsets. Catch trials are not
  in the metadata; positions count the 820 non-catch trials of the session.
- The same definition (cumulative count within session and run) was used in
  learning-rules-rsa `scripts/sub03_check/check8_order_confound.py`; its shift test found an
  order-locked component between sub-01 and sub-02 (V1 0.017 at shifts k ≠ 0) and an optimum at
  k = 0, consistent with the row order being the presentation sequence.

Not a STOP: the position within the run is available. T-RDM therefore uses the position within
the session in trials (0–819) as given by the row order.

## 2. Luminance definition of arXiv:2608.12408

Found in evaluation-resolution-rsa (sibling of this repository, `Projekte_2/evaluation-resolution-rsa`):

- `code/colour_statistic_control.py`: MEANLUM = per-image mean of the Rec. 709 luminance
  0.2126 R + 0.7152 G + 0.0722 B on `load_rgb(path, 224)` (Resize so that the short side is 224,
  bilinear, CenterCrop 224, RGB in [0, 1], no normalisation); RDM = |L_i − L_j| (`abs_rdm`).
- `code/paper_v3/compute_lowlevel.py`: v3 value "MEANLUM vs V1" = 0.0536 (per subject,
  cross-run pairs, mean); partial RSA by `rank_resid` and `pearson`.

Gate: `build_references` on the 720 images (files from v12 `find_img`, identical to
`colour_statistic_control.find_img` for all 720), Spearman against `load_fmri_tri("V1")` per
subject on the cross-run pairs:

| sub-01 | sub-02 | sub-03 | mean | target | \|d\| |
|---|---|---|---|---|---|
| 0.0544 | 0.0958 | 0.0107 | 0.053627 | 0.054 ± 0.001 | 0.00037 |

Passed. Contrast (not gated): RMS contrast = SD over pixels of the Rec. 709 grey values of the
same `load_rgb` image.

## 3. Availability

| Item | Status |
|---|---|
| THINGS test images (100, `step40_common.test_stimuli`) | 100/100 found |
| single presentations of the test images (step 41, `step40_common.load_reps`) | (100, 12, voxels) for all 3 subjects; V1 1049 / 1104 / 839, LOC 2700 / 1175 / 423 voxels |
| test-image model features (`step40_features/test100`, g1) | present (Random, BP × 5 seeds) |
| NSD images (515) | 515/515 on disk |
| NSD k = 1 RDMs (step 42 cache, `NSD_DIR/rdm`) | `V1`, `LO-obj`, `lateral` × `k1_0..2` present |

## 4. Paper-cell gate

Random − BP, raw units, correlation distance, cross-run pairs, NATIVE 224, Conv1 → V1, mean over
seeds 0–4 then subjects, through `cvrsa.model_rdm` and `nsd_step43.rho_pairs`:

| this code path | `results/step5/resolution_rsa.csv` (gap_point) | \|d\| |
|---|---|---|
| 0.030445938 | 0.030445938 | 3.8e-17 |

Passed.

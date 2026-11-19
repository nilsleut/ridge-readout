# Step 43: NSD multiverse, ImageNet models, resolution, noise ceiling

Predictions: `amendments/2026-11-17_step43-predictions.md` (M1, M2, I1, I2; R and NC descriptive),
`config_step43.yaml` (hash e0a00a888b48). `config.yaml` (206fa875cca5), `config_step42.yaml`
(da1cbb33b6d7) and `config_step42b.yaml` (02d0ae4e72de) are unchanged. No betas were downloaded;
patterns, RDM cache, ROIs and the step 42 own-CNN features were reused from `NSD_DIR`, and the
ImageNet and resolution features were extracted from the 515 NSD images already on disk.

NSD data and all voxel- or RDM-level derivatives are stored only under `NSD_DIR`, outside the
repository. The repository holds code and the aggregated CSVs listed below.

Common to all parts: 8 subjects, 515 shared images, k = 3 unless stated, Δ = untrained − trained.
Interval: 95 % percentile interval of the 8-subject mean over 1000 image resamples (seed 20261002,
the step 42 resamples, self-pairs dropped). p: one-sided sign-flip test, 256 assignments.

## Order of commits

| Commit | Time (local, +0100) | Content |
|---|---|---|
| `c4db2e2` | 2026-11-17 13:22:10 | Add step 43 predictions (amendment, `config_step43.yaml`) |
| `0c7e1fd` | 2026-11-17 13:34:23 | Add NSD multiverse, ImageNet, resolution and ceiling scripts with tests |
| this commit | | Add step 43 results |

The runs started at 13:53 on 2026-11-17, after `0c7e1fd`. Before every commit
`scripts/nsd_commit_check.py` ran on the staged and tracked files (also as the pre-commit hook).

## Tests (`tests/test_nsd43.py`, before the runs)

| Check | Result |
|---|---|
| (i) multiverse path on THINGS reproduces `results/step7/multiverse.csv` (V1 and LOC, NATIVE 224, 36 cells, 10 models, 3 subjects) | passed: 2160 values, max \|d rho\| 2.6e-16; `roi_betas` restored |
| (ii) ImageNet extraction, 20 gate images, downstream-equivalence gate against `data/step8_gap` | passed: 288 rho, max \|d rho\| 2.1e-9, max \|d Δ\| 9.8e-10, max \|d feature\| 9.5e-7 |
| (iii) resolution path on THINGS (720 images re-extracted, 32 and 224 px) reproduces `results/step5/resolution_rsa.csv` | passed: 120 values, max \|d rho\| 1.9e-16; features identical (max \|d\| 0) |
| (iv) LOSO lower bound on THINGS V1, cross-run pairs | passed: 0.051433 vs 0.051433, \|d\| 6.9e-18 |

Existing suites unchanged and passing: phase 2 97/97, loader 41/41, lower edge 2/2, banded 7/7,
cvrsa 4/4, noise curve 4/4, `tests/test_nsd.py --skip-gate` 7/7; `config.yaml` hash 206fa875cca5.

Checks inside the runs: the multiverse cells F raw / z, D correlation, B paper reproduce the step 42
V1 rho (max |d| 2.5e-16); the 224 px resolution features (step 5 path) equal the step 42 features
(max |d| 0); the 224 px Δ equal P1 and P2 of step 42.

## Predictions (`results/step43/summary.csv`)

| Rule | Criterion | Value | Result |
|---|---|---|---|
| M1 | distinct analyses with the interval above 0 ≤ 2 | 0 of 28 | met |
| M2 | distinct analyses with the interval below 0 ≥ half | 13 of 28 (half = 14) | not met |
| I1 | V1, A: mean Δ < 0 in ≥ 5 of 6 architectures | 6 of 6 | met |
| I2 | V1, raw, correlation: mean Δ ≤ 0 in ≥ 4 of 6 architectures | 6 of 6 | met |

## M: multiverse (`results/step43/multiverse_nsd.csv`)

Own CNN, GAP Conv1, NATIVE 224, Random − BP over seeds 0–4, V1. 36 cells (F × D × B) form 28 distinct
analyses (step 28 rule: max |d rho| within classes 0, smallest max |d| between classes 7.4e-3; no
STOP). Euclidean distance does not separate raw from center or scale from z, nor B paper from
center720, which gives the 4 merged classes.

| Class | Cells (F / D / B) | Mean Δ | Interval | neg./pos. | p (neg.) | p (pos.) | Interval |
|---|---|---|---|---|---|---|---|
| 1 | raw / correlation / paper | −0.0169 | [−0.0300, −0.0023] | 8/0 | 0.0039 | 1.0000 | below 0 |
| 2 | raw / correlation / center720 | +0.0053 | [−0.0087, +0.0210] | 2/6 | 0.9766 | 0.0273 | includes 0 |
| 3 | raw / correlation / z720 | +0.0035 | [−0.0088, +0.0176] | 2/6 | 0.9688 | 0.0352 | includes 0 |
| 4 | raw / cosine / paper | −0.0136 | [−0.0221, −0.0040] | 8/0 | 0.0039 | 1.0000 | below 0 |
| 5 | raw / cosine / center720 | −0.0080 | [−0.0175, +0.0025] | 8/0 | 0.0039 | 1.0000 | includes 0 |
| 6 | raw / cosine / z720 | −0.0078 | [−0.0170, +0.0021] | 8/0 | 0.0039 | 1.0000 | includes 0 |
| 7 | raw, center / euclidean / paper, center720 | −0.0081 | [−0.0316, +0.0145] | 6/2 | 0.0469 | 0.9570 | includes 0 |
| 8 | raw, center / euclidean / z720 | −0.0088 | [−0.0324, +0.0137] | 7/1 | 0.0352 | 0.9688 | includes 0 |
| 9 | center / correlation / paper | +0.0003 | [−0.0092, +0.0109] | 3/5 | 0.5781 | 0.4258 | includes 0 |
| 10 | center / correlation / center720 | −0.0001 | [−0.0101, +0.0111] | 4/4 | 0.4688 | 0.5352 | includes 0 |
| 11 | center / correlation / z720 | −0.0011 | [−0.0100, +0.0090] | 7/1 | 0.2070 | 0.7969 | includes 0 |
| 12 | center / cosine / paper | −0.0189 | [−0.0340, −0.0015] | 8/0 | 0.0039 | 1.0000 | below 0 |
| 13 | center / cosine / center720 | −0.0175 | [−0.0328, +0.0000] | 8/0 | 0.0039 | 1.0000 | includes 0 |
| 14 | center / cosine / z720 | −0.0175 | [−0.0326, −0.0007] | 8/0 | 0.0039 | 1.0000 | below 0 |
| 15 | scale / correlation / paper | −0.0067 | [−0.0154, +0.0016] | 7/1 | 0.0078 | 0.9961 | includes 0 |
| 16 | scale / correlation / center720 | −0.0053 | [−0.0138, +0.0033] | 7/1 | 0.0078 | 0.9961 | includes 0 |
| 17 | scale / correlation / z720 | −0.0053 | [−0.0129, +0.0025] | 8/0 | 0.0039 | 1.0000 | includes 0 |
| 18 | scale / cosine / paper | −0.0083 | [−0.0151, −0.0014] | 8/0 | 0.0039 | 1.0000 | below 0 |
| 19 | scale / cosine / center720 | −0.0110 | [−0.0187, −0.0036] | 8/0 | 0.0039 | 1.0000 | below 0 |
| 20 | scale / cosine / z720 | −0.0107 | [−0.0181, −0.0039] | 8/0 | 0.0039 | 1.0000 | below 0 |
| 21 | scale, z / euclidean / paper, center720 | −0.0094 | [−0.0286, +0.0100] | 6/2 | 0.0742 | 0.9297 | includes 0 |
| 22 | scale, z / euclidean / z720 | −0.0099 | [−0.0288, +0.0090] | 6/2 | 0.0742 | 0.9297 | includes 0 |
| 23 | z / correlation / paper | −0.0262 | [−0.0359, −0.0177] | 8/0 | 0.0039 | 1.0000 | below 0 |
| 24 | z / correlation / center720 | −0.0252 | [−0.0349, −0.0162] | 8/0 | 0.0039 | 1.0000 | below 0 |
| 25 | z / correlation / z720 | −0.0229 | [−0.0314, −0.0148] | 8/0 | 0.0039 | 1.0000 | below 0 |
| 26 | z / cosine / paper | −0.0612 | [−0.0788, −0.0434] | 8/0 | 0.0039 | 1.0000 | below 0 |
| 27 | z / cosine / center720 | −0.0581 | [−0.0755, −0.0404] | 8/0 | 0.0039 | 1.0000 | below 0 |
| 28 | z / cosine / z720 | −0.0571 | [−0.0736, −0.0400] | 8/0 | 0.0039 | 1.0000 | below 0 |

Upper bound of class 13: +1.2e-5.

| | above 0 | below 0 | includes 0 | distinct analyses |
|---|---|---|---|---|
| NSD, V1 (this step) | 0 | 13 | 15 | 28 |
| THINGS (step 7 / step 28) | 5 | 9 | 14 | 28 |

No NSD analysis has an interval above 0. Positive point estimates occur only for raw units with
correlation distance on centred or z-scored voxels (classes 2, 3: 6/8 subjects positive) and,
near 0, for center / correlation / paper.

## I: ImageNet models (`results/step43/imagenet_nsd.csv`)

Δ = mean over untrained s0–s2 − trained (IMAGENET1K_V1), R = 224. V1 layer → V1, LOC layer → LO-obj.
Each cell: 8-subject mean Δ (subjects with Δ < 0 of 8).

| Model | ROI | raw, corr. | z, corr. | N | A | W |
|---|---|---|---|---|---|---|
| alexnet | V1 | −0.0181 (8) | −0.0674 (8) | −0.0100 (6) | −0.0846 (8) | −0.0391 (7) |
| vgg11_bn | V1 | −0.0047 (6) | −0.0572 (8) | +0.0194 (2) | −0.0699 (8) | −0.0087 (6) |
| resnet18 | V1 | −0.0117 (8) | −0.0435 (8) | −0.0093 (5) | −0.0749 (8) | −0.0242 (7) |
| resnet50 | V1 | −0.0129 (8) | −0.0424 (8) | −0.0072 (5) | −0.0661 (8) | −0.0232 (7) |
| convnext_tiny | V1 | −0.0053 (7) | −0.0176 (8) | +0.0070 (2) | −0.0157 (8) | −0.0283 (8) |
| vit_b_16 | V1 | −0.0571 (8) | −0.0830 (8) | +0.0160 (4) | −0.0996 (8) | −0.0207 (6) |
| alexnet | LO-obj | −0.0917 (8) | −0.0959 (8) | +0.0070 (3) | −0.0953 (8) | −0.0153 (7) |
| vgg11_bn | LO-obj | −0.0926 (8) | −0.1039 (8) | +0.0078 (2) | −0.1008 (8) | +0.0139 (3) |
| resnet18 | LO-obj | −0.0847 (8) | −0.0870 (8) | +0.0201 (0) | −0.0911 (8) | −0.0536 (7) |
| resnet50 | LO-obj | −0.0956 (8) | −0.0919 (8) | −0.0081 (6) | −0.0921 (8) | −0.0520 (7) |
| convnext_tiny | LO-obj | −0.0237 (8) | −0.0321 (8) | +0.0073 (5) | −0.0155 (8) | −0.0212 (6) |
| vit_b_16 | LO-obj | −0.0454 (8) | −0.0596 (8) | +0.0199 (1) | −0.0515 (8) | +0.0071 (4) |

Intervals of the two prediction cells at V1:

| Model | A: Δ, interval | raw, corr.: Δ, interval | p (neg.) raw, corr. |
|---|---|---|---|
| alexnet | −0.0846 [−0.1093, −0.0587] | −0.0181 [−0.0309, −0.0049] | 0.0039 |
| vgg11_bn | −0.0699 [−0.0938, −0.0442] | −0.0047 [−0.0197, +0.0097] | 0.0625 |
| resnet18 | −0.0749 [−0.0934, −0.0566] | −0.0117 [−0.0231, −0.0005] | 0.0039 |
| resnet50 | −0.0661 [−0.0855, −0.0476] | −0.0129 [−0.0243, −0.0020] | 0.0039 |
| convnext_tiny | −0.0157 [−0.0288, −0.0018] | −0.0053 [−0.0158, +0.0069] | 0.0313 |
| vit_b_16 | −0.0996 [−0.1300, −0.0676] | −0.0571 [−0.0800, −0.0367] | 0.0039 |

A: 8/8 subjects negative and p 0.0039 in all 6 architectures. With untrained s0 only, all signs
of the I1 and I2 cells are the same (A from −0.0173 to −0.0987; raw, corr. from −0.0039 to −0.0565).
The 'lateral' ROI and the s0-only variant are in the CSV.

## R: resolution (`results/step43/resolution_nsd.csv`)

Own CNN, NATIVE, GAP Conv1, Random − BP over seeds 0–4, V1, correlation distance. All 8 subjects
are negative at every resolution in both cells (p 0.0039).

| px | raw: Δ | raw: interval | z: Δ | z: interval |
|---|---|---|---|---|
| 32 | −0.0209 | [−0.0308, −0.0097] | −0.0370 | [−0.0469, −0.0271] |
| 64 | −0.0228 | [−0.0344, −0.0103] | −0.0382 | [−0.0487, −0.0287] |
| 96 | −0.0216 | [−0.0340, −0.0081] | −0.0355 | [−0.0456, −0.0265] |
| 128 | −0.0201 | [−0.0329, −0.0064] | −0.0330 | [−0.0429, −0.0243] |
| 160 | −0.0189 | [−0.0318, −0.0047] | −0.0307 | [−0.0406, −0.0220] |
| 224 | −0.0169 | [−0.0300, −0.0023] | −0.0262 | [−0.0359, −0.0177] |
| S = Δ(224) − Δ(32) | +0.0039 | [−0.0035, +0.0121] | +0.0108 | [+0.0030, +0.0174] |

Slope signs (pos./neg.): raw 5/3 (p positive 0.031), z 7/1 (p positive 0.012). The advantage of BP
is largest at 64 px and decreases towards 224 px; for z units the decrease excludes 0.

## NC: noise ceiling (`results/step43/ceiling_nsd.csv`)

LOSO lower bound (`nili_bounds` of learning-rules-rsa), correlation distance. k = 3: all 132,355
pairs; k = 1: per presentation, pairs where no subject has the two presentations in the same session
and run (mean 131,924 pairs), mean over the 3 presentations. Model rho: mean over seeds and the 8
subjects, raw units (`results/step42/group.csv`).

| ROI | k | Lower bound | Interval | rho Random | rho BP |
|---|---|---|---|---|---|
| V1 | 3 | 0.5164 | [0.4913, 0.5406] | 0.0514 | 0.0683 |
| V1 | 1 | 0.2691 | [0.2483, 0.2883] | 0.0296 | 0.0416 |
| LO-obj | 3 | 0.2068 | [0.1899, 0.2242] | 0.0106 | 0.0246 |
| LO-obj | 1 | 0.0843 | [0.0761, 0.0925] | 0.0044 | 0.0126 |

Both models reach about 10–16 % of the lower bound at V1 and 5–15 % at LO-obj; for THINGS V1
(cross-run, test (iv)) the lower bound is 0.051.

## Files

| File | Content |
|---|---|
| `results/step43/multiverse_nsd.csv` | 36 cells with class, per-subject Δ, rho per model and subject, class checks |
| `results/step43/imagenet_nsd.csv` | cells per model, ROI and cell (mean s0–s2 and s0 only), per-subject Δ, rho |
| `results/step43/resolution_nsd.csv` | Δ per px and slope (raw, z), per-subject values, 224 px feature check |
| `results/step43/ceiling_nsd.csv` | LOSO lower bound, V1 and LO-obj, k = 3 and k = 1, with model rho |
| `results/step43/summary.csv` | M1, M2, I1, I2, runtimes |

Every CSV carries `config_hash` (step 43), `config_hash_42`, `git_head` and `seed` (model seed
where a row has one, else empty; the bootstrap seed is in the config).

## Runtime

| Part | Wall time |
|---|---|
| estimate before the runs (× 1.5) | 2.16 h, below the 10 h stop |
| ImageNet extraction (6 models × 4 conditions, 515 images) | 954 s |
| resolution extraction (6 px × 10 models) | 179 s |
| multiverse (bootstrap, 8 workers) | 19,407 s (5.4 h) |
| ImageNet (resumed run) | 10,487 s (2.9 h) |
| resolution (resumed run) | 4,423 s (1.2 h) |
| ceiling (resumed run) | 546 s |
| sum of the analysis parts | 34,863 s (9.7 h) plus extraction 0.3 h |

## Open points

- Runtime: the estimate (2.16 h) assumed one ranking per model RDM and draw; `bootstrap` ranks the
  model rows once per subject and draw (8 times). The results are unaffected; the wall time (about
  10 h in total) was 4.6 times the estimate.
- The first run (`data/step43_run.log`) ended without an error message on 2026-11-17 at about 20:45,
  during the ImageNet bootstrap, after `multiverse_nsd.csv` had been written; the process was ended
  from outside. `scripts/nsd_step43_resume.py` ran the remaining parts with the unchanged functions
  `run_imagenet`, `run_resolution` and `run_ceiling` and wrote `summary.csv` as `run_all` does, with
  M1 and M2 read from the summary rows of `multiverse_nsd.csv` (log `data/step43_resume.log`). The
  resamples are fixed by the seed, so the restart does not change any value. A second start on
  2026-11-17 failed at once (temporary files held by the first run) and wrote nothing.
- ImageNet: the amendment says "one weight set each"; the untrained condition is the mean of the
  three untrained sets s0–s2, as in the paper (`config_step43.yaml`). With s0 only, I1 and I2 give
  the same counts (6/6 each).
- Implementation sources: multiverse `step7_multiverse` (FS, DS, BS, normalise, betas with
  `roi_betas` pointed at the NSD patterns) and the step 28 class rule; ImageNet
  `step8_extract_pretrained`; resolution the v12 extraction of `step5_extract_gap`; ceiling
  `nili_bounds` of learning-rules-rsa `scripts/noise_ceiling_v2/common.py`.
- k = 1 ceiling: pairs are kept only when no subject has the two presentations in the same session
  and run, so all 8 subjects use the same pairs (as `nili_bounds` requires).
- Logs (ignored by git): `data/step43_run.log`, `data/step43_run2.log`, `data/step43_resume.log`.

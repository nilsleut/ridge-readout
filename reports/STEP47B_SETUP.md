# Step 47b setup: feasibility and gates

Checked before the amendment `amendments/2026-11-21_step47b-predictions.md`. Nothing was computed
except the gates and timings below; no result file was written. Configuration: `config_step47b.yaml`
(hash 97ffa8ef4507).

## 1. THINGS: model features for all 82 images of every run of the session of the 720

The session of the 720 (sub-01 session 3, sub-02 session 2, sub-03 session 5) has 10 runs of
82 trials each: 72 of the 720 and 10 test images per run. Across the 10 runs, the test trials
are the 100 test images, each shown once. So the session holds 820 images: the 720 plus the
100 test images.

- **Features for the 720:** GAP Conv1, NATIVE 224, `step7_multiverse.feat`.
- **Features for the 100 test images:** `step40_features/test100`, g = 1, Random and BP,
  seeds 0–4. They were extracted in step 40 with the same extraction code.

The features are present for all 820 images.

**Gate: features of the 720 identical to the existing ones.** The step 40 extraction
(`step40_features/gate20`, g = 1, 20 of the 720 images) was compared with the GAP features of
step 5 for the same images, for all 10 checkpoints. The maximum |Δ| is **0**, so the gate
passed. The test-image features therefore come from the same pipeline as the 720 features.

## 2. THINGS test images: run means in all 12 sessions

All 12 sessions of every subject have 10 runs of 82 trials. In each session, 10 of the 82
trials per run are test trials, and each test image appears exactly once per session.

`step47_runs.load_resp` reads all 9840 trials of the ROI rows and z-scores them exactly as
`load_arm_a` and `load_arm_b` do. This takes 1.5–3 s per subject. The 120 run means per
subject take about 1 s.

**Gate:** the 1200 test presentations taken from these responses equal
`step40_common.load_reps`, with max |Δ| = **0** in each of the 3 subjects. Passed.

Split-half reliability: `arm_b_betas.split_half_reliability` on these presentations reproduces
the stored `armb.npz` reliability to 3e-8 (float32 rounding), taking 13–15 s per subject. Two
paths use it:

- The run-centred ridge path (Y1) recomputes the reliable voxels with this function on the
  run-centred presentations.
- The uncentred reproduction uses the stored mask.

## 3. NSD: run means of all trials, sessions 1–30, all 8 subjects

For each session, `scripts/nsd_runmeans.py` will run these steps:

1. Download the betas file, using the same key and the same pipelined download as
   `nsd_fetch.py`.
2. Call `nsd_fetch.process` with all 750 trial positions. This gives the same division by 300
   and the same per-voxel z-score over the 750 trials of the session, on the V1 ∪ lateral
   voxels (LO-obj is a subset of lateral).
3. Compute the voxel mean over all trials of each of the 12 runs. The runs come from the design
   stimpattern, and are checked against `nsd_common.design` for the image trials.
4. Store only the run means, (12, n_voxels) float32, under `NSD_DIR/runmeans/<subject>/`.
5. Check every session: the trials of the 515 images must equal the stored step 42
   `NSD_DIR/betas/<subject>/sessionNN.npz` exactly, otherwise STOP.
6. Delete the original file.

**Estimate**, from the step 42 logs (`results/step42/fetch_*.csv`):

| | per subject | all 8 |
|---|---|---|
| download | 15–21 min, 11.8–16.2 GB | 2.24 h, 109 GB |
| processing | 3–7 min | 0.59 h, overlapped with the download |
| estimate × 1.5 | | **3.4 h** (below 8 h) |

Disk space:

- At most two beta files are on disk at once, the largest 0.55 GB.
- The stored run means total 240 sessions × 12 runs × 7260–10,157 voxels × 4 B ≈ **0.1 GB**.
- Free space is 113 GB, against a requirement of at least 30 GB.

**Gate subj01 session 1** (the newly z-scored trials of the 515 images must be identical to the
stored step 42 trials): this needs the first NSD download. The download happens only after the
amendment commit, so the gate is the first action after that commit and runs before anything
else is loaded. Its result is added below in the scripts commit.

Result: `python scripts/nsd_runmeans.py gate`, run after the amendment commit edbb11d
(2026-11-21 18:22:05).

- The file is 0.49 GB; download took 38 s and processing 7 s.
- The 42 trials of the 515 images in session 1 are **identical** to
  `NSD_DIR/betas/subj01/session01.npz`, both Z (`np.array_equal`) and the valid mask.
  **Passed.**
- The session has 12 runs of 62–63 trials, 750 in total.
- The run means of this session are kept and reused by the full reload.

## 4. Runtimes (for the estimates of the parts)

| Part | basis | estimate × 1.5 |
|---|---|---|
| X (THINGS mechanism, 10,000 permutations, 10 models × 10 run means) | seconds per 1000 draws | < 0.1 h |
| Y THINGS ridge, 30 fits × g ∈ {1, 4, 8} + 30 fits (original voxels) | 15–16 s per fit (sub-01 V1, g = 4 and 8), 8 workers | 0.1 h |
| Y THINGS test images | step 41 sizes, seconds | < 0.1 h |
| NSD reload | step 42 logs | 3.4 h |
| Y NSD RSA (8 subjects, k = 3) | step 42 per-subject runs | < 0.1 h |
| Y NSD ridge, 528 run-centred + 88 (Y4) + 88 (step 47 voxels) fits | step 47 part B: 528 fits in 1974 s | 1.1 h |
| Y NSD multiverse, 1001 draws | one draw 5.0 s with one `rho_pairs` call for all subjects (22.3 s per subject as in step 43; values identical, max \|Δ\| 0) | 0.3 h single process; step 43 measured 5.4 h for the per-subject version |

The scripts print the final estimate per part before each run. A part stops if its estimate is
above 8 h.

## 5. Tests before the runs (`tests/test_step47b.py`, 9/9 passed)

| Test | Result |
|---|---|
| (i) synthetic drift: X3 statistic, run RDMs, permutation null | slow drift +0.989 (no drift +0.120); hand-built RDMs equal; null draw k equals Delta A on draw k; Random-built brain gives A = 1, percentile 100 |
| (ii) permutation keeps 10 × 82 | 500 draws; with 720 / 72 the step 47 `perm_null` draws are reproduced exactly (0) |
| (iii) THINGS ridge vs `results/prereg` (g = 4, 30 fits) | max \|d r\| 6.7e-16; hits, gate and voxel counts equal |
| (iii) NSD RSA vs step 42 | 960 rho, max \|d\| 9.9e-17; P1 −0.016943, P2 −0.026233 (\|d\| 1.1e-16) |
| (iii) NSD ridge vs step 47 (subj01 V1 g = 4, 11 fits) | max \|d\| 8.3e-17; hits and voxel counts equal |
| (iii) multiverse draw path | identity draw = step 43 rho (9.7e-17); resamples 1–2 = per-subject `rho_pairs` to 8.0e-16 (the test first demanded bit identity; the difference is floating-point order of the dot products with memory-mapped arrays; tolerance set to 1e-12, the class rule uses 1e-9) |
| (iv) run-centred THINGS RSA path | +0.007032171 = step 47 (1e-12) |
| (v) synthetic NSD sessions | `process` vs hand z-score 1.0e-7 (float32); run means 1.8e-8; centring exact; centred run means 1.4e-8 |

Existing suites passing: phase 2 97/97, loader 41/41, lower edge 2/2, banded 7/7, cvrsa 4/4, noise
curve 4/4, `test_nsd --skip-gate` 7/7, `test_step44` 7/7, `test_step45 --only ii,iii` 2/2,
`test_step47` 8/8; `config.yaml` hash 206fa875cca5.

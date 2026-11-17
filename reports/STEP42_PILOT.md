# Step 42 pilot: NSD, subj01

Predictions: `amendments/2026-11-17_step42-predictions.md`; configuration `config_step42.yaml`
(hash da1cbb33b6d7); setup: `reports/STEP42_NSD_SETUP.md`. `config.yaml` (206fa875cca5),
`config_step40.yaml` and `config_step41.yaml` unchanged. P1–P4 are not applied here; they
need all 8 subjects. Nothing is changed in the predictions after the pilot.

NSD data and all voxel- or RDM-level derivatives are stored only under `NSD_DIR`, outside
the repository. The repository holds code and the aggregated CSVs listed below.

## Order of commits

| Commit | Content |
|---|---|
| `11fac04` | Add step 42 predictions (amendment, `config_step42.yaml`, setup report) |
| `fb51f33` | Add NSD pipeline with tests |
| this commit | Add NSD pilot results (subj01) |

Betas were loaded only after `fb51f33`. Before every commit, `scripts/nsd_commit_check.py`
ran on the staged and tracked files (also as the local pre-commit hook); 0 problems each time.

## Tests (`tests/test_nsd.py`, before any beta was loaded)

| Check | Result |
|---|---|
| (i) extraction gate, 20 THINGS Arm A images through the NSD builder | passed: max \|Δ median r\| 1.25e-9, \|Δ (Random − BP)\| 9.5e-10 |
| (ii) synthetic: run exclusion at k = 1, averaging at k = 3, k = 1 rho over the kept pairs, sign-flip p | passed (max \|d\| 0 and 5.6e-17; p = 1/256 for 8 negative values) |
| (iii) `nsd_rsa` on the THINGS test images (12 presentations averaged) reproduces the step 41 k = 12 rho | passed: 240 values, max \|d rho\| 1.9e-16 |
| (iv) commit check finds a file placed under `NSD_DIR` in a temporary repository; the pre-commit hook refuses the commit | passed |
| existing suites | 97/97, 41/41, 2/2; banded 7/7, cvRSA 4/4, noise curve 4/4; config hash 206fa875cca5 |

## Download (subj01)

Sessions 1–30, `betas_fithrf_GLMdenoise_RR`, func1pt8mm: 14.69 GB in 970 s wall
(download of the next session overlapped with processing of the current one; about 30 s
per 0.49 GB session, about 15 MB/s). At most two session files on disk at any time; free
space stayed between 113.5 and 114.1 GB. Stored under `NSD_DIR` for subj01: 55 MB per-session
trial data and 54 MB assembled patterns (float32), 13 MB RDM cache. The 515 images
(425 × 425 RGB PNG, 160 MB) took 805 s, one request per file. Per-session log:
`results/step42/fetch_subj01.csv`.

## Gates (`results/step42/pilot_gates.csv`)

| Gate | Value | Result |
|---|---|---|
| G1 every image has exactly 3 presentations | 515 of 515 images, 1545 distinct trials | passed |
| G2 per voxel and session after scaling and z-score: mean ≈ 0, SD ≈ 1 | max \|mean\| 7.4e-16, max \|SD − 1\| 3.3e-16 over 30 sessions × 9149 voxels; 0 invalid voxels in every session | passed |
| G3 voxels (must exceed 300 for V1 and LO-obj) | V1 1350, LO-obj 2921, lateral 7799; none excluded as invalid | passed |
| grid check: ROI files and `floc_objectstval` on the beta grid | same shape, affine equal within 1e-4 | passed |
| G4 Spearman of the V1 RDMs (correlation distance) between presentations | 1 vs 2: 0.265; 1 vs 3: 0.249; 2 vs 3: 0.247 | passed |
| G5 feature gate and extraction | gate passed (rerun after `fb51f33`); 10 models × 515 images, 32 units after global average pooling | passed |

G2 holds by construction: the z-score is computed over the 750 trials of each session,
and the gate checks those same trials.

## Pilot cells (`results/step42/pilot_subj01.csv`)

rho = Spearman, mean over the 5 seeds (at k = 1 also over the three presentations);
Δ = Random − BP. k = 3: 132,355 pairs; k = 1: 131,885–131,980 pairs after the same-run
exclusion. Descriptive; no prediction is applied to a single subject.

| ROI | k | Units | Distance | rho Random | rho BP | Δ |
|---|---|---|---|---|---|---|
| V1 | 3 | raw | correlation | 0.0440 | 0.0611 | −0.0171 |
| V1 | 3 | raw | Euclidean | 0.1263 | 0.1136 | +0.0127 |
| V1 | 3 | z | correlation | 0.0545 | 0.0792 | −0.0247 |
| V1 | 3 | z | Euclidean | 0.1329 | 0.1095 | +0.0234 |
| V1 | 1 | raw | correlation | 0.0266 | 0.0406 | −0.0140 |
| V1 | 1 | raw | Euclidean | 0.0718 | 0.0616 | +0.0102 |
| V1 | 1 | z | correlation | 0.0360 | 0.0520 | −0.0160 |
| V1 | 1 | z | Euclidean | 0.0742 | 0.0585 | +0.0157 |
| LO-obj | 3 | raw | correlation | 0.0149 | 0.0416 | −0.0267 |
| LO-obj | 3 | raw | Euclidean | 0.0449 | 0.0511 | −0.0062 |
| LO-obj | 3 | z | correlation | 0.0245 | 0.0341 | −0.0096 |
| LO-obj | 3 | z | Euclidean | 0.0472 | 0.0470 | +0.0002 |
| LO-obj | 1 | raw | correlation | 0.0075 | 0.0216 | −0.0141 |
| LO-obj | 1 | raw | Euclidean | 0.0128 | 0.0266 | −0.0138 |
| LO-obj | 1 | z | correlation | 0.0118 | 0.0164 | −0.0046 |
| LO-obj | 1 | z | Euclidean | 0.0161 | 0.0248 | −0.0088 |
| lateral | 3 | raw | correlation | 0.0156 | 0.0432 | −0.0275 |
| lateral | 3 | raw | Euclidean | 0.0364 | 0.0363 | +0.0001 |
| lateral | 3 | z | correlation | 0.0266 | 0.0360 | −0.0094 |
| lateral | 3 | z | Euclidean | 0.0361 | 0.0296 | +0.0065 |
| lateral | 1 | raw | correlation | 0.0093 | 0.0248 | −0.0155 |
| lateral | 1 | raw | Euclidean | 0.0108 | 0.0219 | −0.0111 |
| lateral | 1 | z | correlation | 0.0147 | 0.0196 | −0.0048 |
| lateral | 1 | z | Euclidean | 0.0129 | 0.0181 | −0.0052 |

Across seeds, V1, raw units, correlation distance, k = 3: Random 0.036–0.052, BP 0.054–0.066.

## Projection for the remaining 7 subjects

| Item | Estimate |
|---|---|
| Download, sessions 1–30 | 94.0 GB (subj02 13.64, subj03 14.18, subj04 13.12, subj05 12.11, subj06 16.25, subj07 11.78, subj08 12.97) |
| Time at the subj01 rate (about 15 MB/s with overlapped processing) | about 1.7 h, plus about 70 s RSA per subject |
| Disk | at most about 1 GB transient; about 120 MB stored per subject under `NSD_DIR` (about 0.9 GB for all 7); free space stays near 105 GB, well above the 30 GB floor |
| Group step (`nsd_rsa.py group`) | sign-flip test and a 1000-resample stimulus bootstrap over all cells; not yet run, runtime not measured |

## Open points

- The group bootstrap re-ranks the model RDMs per subject and presentation at k = 1; its
  runtime is not measured yet (estimated at a few hours single-threaded).
- No further subject is loaded until released.
- Logs (ignored by git): `data/step42_fetch_subj01.log`, `data/step42_features.log`.

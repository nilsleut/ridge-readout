# Step 42: NSD, 8 subjects (with step 42b)

Predictions: `amendments/2026-11-17_step42-predictions.md` (P1–P4, all 8 subjects,
`config_step42.yaml`, hash da1cbb33b6d7) and `amendments/2026-11-17_step42b-predictions.md`
(Q1–Q4, subj02–08, `config_step42b.yaml`, hash 02d0ae4e72de). `config.yaml` (206fa875cca5),
`config_step40.yaml`, `config_step41.yaml` and `config_step42.yaml` are unchanged. Pilot:
`reports/STEP42_PILOT.md`.

NSD data and all voxel- or RDM-level derivatives are stored only under `NSD_DIR`, outside the
repository. The repository holds code and the aggregated CSVs listed below.

## Order of commits

| Commit | Time (local, +0100) | Content |
|---|---|---|
| `11fac04` | 08:26:39 | Add step 42 predictions |
| `fb51f33` | 08:36:20 | Add NSD pipeline with tests |
| `49fc44b` | 09:03:25 | Add NSD pilot results (subj01) |
| `366a9eb` | 09:16:02 | Add step 42b predictions (amendment, `config_step42b.yaml`) |
| `becabd7` | 09:20:34 | Add norm-angle decomposition for NSD |
| this commit | | Add NSD results (8 subjects) |

The first beta of subj02 was requested at 08:21:07 UTC (09:21:07 local;
`download_started_utc`, session 1, `results/step42/fetch_subj02.csv`), 5 min after the
amendment `366a9eb` (08:16:02 UTC) and after `becabd7`. Before that, betas of subj01 only were on
disk. Before every commit, `scripts/nsd_commit_check.py` ran on the staged and tracked files (also
as the pre-commit hook).

## Tests

| Check | Result |
|---|---|
| (v) `rsa_components` (N, A via `step25_weighted_angle.nwa`) on THINGS reproduces `results/step9/separation.csv` | passed: 120 values, max \|d rho\| 1.5e-16; `TRI` restored |
| the new code path reproduces the pilot rho of subj01 | 480 values, max \|d\| 9.9e-17 |
| bootstrap with the identity resample reproduces the point estimates | 33 cells, max \|d\| 2.8e-17 |
| `tests/test_nsd.py --skip-gate` | 7/7 passed (gate (i) unchanged since step 42, passed there) |

## Gates per subject (`results/step42/gates.csv`)

All gates passed for all 8 subjects; no STOP.

| | subj01 | subj02 | subj03 | subj04 | subj05 | subj06 | subj07 | subj08 |
|---|---|---|---|---|---|---|---|---|
| G1 images with 3 presentations | 515 | 515 | 515 | 515 | 515 | 515 | 515 | 515 |
| G2 max \|mean\|, max \|SD − 1\| | < 1e-15 | < 1e-15 | < 1e-15 | < 1e-15 | < 1e-15 | < 1e-15 | < 1e-15 | < 1e-15 |
| G3 voxels V1 | 1350 | 1102 | 1254 | 877 | 1113 | 1121 | 1142 | 1074 |
| G3 voxels LO-obj | 2921 | 2700 | 3402 | 3078 | 2330 | 3877 | 3160 | 3858 |
| G3 voxels lateral | 7799 | 6752 | 8572 | 7131 | 7037 | 9030 | 6118 | 8674 |
| voxels excluded as invalid | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| G4 V1 RDM, presentation 1 vs 2 | 0.265 | 0.294 | 0.148 | 0.220 | 0.203 | 0.168 | 0.166 | 0.104 |
| G4 V1 RDM, presentation 2 vs 3 | 0.247 | 0.260 | 0.126 | 0.190 | 0.177 | 0.173 | 0.137 | 0.092 |
| G5 feature gate, 10 models × 515 finite rows | passed | passed | passed | passed | passed | passed | passed | passed |
| grid check | passed | passed | passed | passed | passed | passed | passed | passed |

## Predictions (`results/step42/summary.csv`)

Δ = Random − BP, mean over seeds 0–4. Interval: 95 % stimulus bootstrap of the subject mean
(1000 resamples, seed 20261002). p: one-sided sign-flip test (256 assignments for 8, 128 for 7).

| Rule | Cell | Mean Δ | Interval | Signs | p | Result |
|---|---|---|---|---|---|---|
| P1 | V1, raw, correlation, k = 3: mean Δ ≤ 0 | −0.0169 | [−0.0300, −0.0023] | 8/8 negative | 0.0039 | met |
| P2 | V1, z, correlation, k = 3: ≥ 6/8 negative, p ≤ 0.05, interval excludes 0 | −0.0262 | [−0.0359, −0.0177] | 8/8 negative | 0.0039 | met |
| P3 | LO-obj, z, correlation, k = 3: ≥ 6/8 negative, p ≤ 0.05 | −0.0059 | [−0.0091, −0.0028] | 8/8 negative | 0.0039 | met |
| P4 | V1, raw, correlation: Δ(k = 3) ≤ Δ(k = 1) | −0.0169 vs −0.0120 | | | | met |
| Q1 | V1, raw, Euclidean, subj02–08: Δ > 0 in ≥ 5/7, p ≤ 0.05 | −0.0111 | [−0.0349, +0.0122] | 1/7 positive | 0.9922 | not met |
| Q2 | V1, N, subj02–08: Δ > 0 in ≥ 5/7, p ≤ 0.05 | −0.0083 | [−0.0341, +0.0179] | 2/7 positive | 0.9531 | not met |
| Q3 | V1, A, subj02–08: Δ < 0 in ≥ 5/7, p ≤ 0.05 | −0.0181 | [−0.0346, −0.0000] | 7/7 negative | 0.0078 | met |
| Q4 | LO-obj, A, subj02–08: Δ < 0 in ≥ 5/7, p ≤ 0.05 | −0.0090 | [−0.0137, −0.0044] | 7/7 negative | 0.0078 | met |

The upper bound of the Q3 interval is −9e-6; for all 8 subjects it is +1.2e-5. Q3 does not
require the interval.

The sign reversal under Euclidean distance at V1 that motivated Q1–Q4 (subj01: Δ +0.0127 raw,
+0.0234 z) did not replicate: in subj02–08 Δ is negative in 6 of 7 subjects under both raw and z
units. The angle component A favours BP in all 16 subject × ROI cases (V1 and LO-obj); the norm
component N at V1 is mixed (3/8 positive).

## Δ per subject, k = 3

| ROI | Subject | raw, corr. | z, corr. | raw, Eucl. | z, Eucl. | N | A | W |
|---|---|---|---|---|---|---|---|---|
| V1 | subj01 | −0.0171 | −0.0247 | +0.0127 | +0.0234 | +0.0275 | −0.0130 | +0.0117 |
| V1 | subj02 | −0.0063 | −0.0338 | −0.0119 | −0.0012 | −0.0364 | −0.0248 | −0.0159 |
| V1 | subj03 | −0.0141 | −0.0257 | −0.0171 | −0.0263 | −0.0104 | −0.0164 | −0.0219 |
| V1 | subj04 | −0.0270 | −0.0187 | −0.0068 | +0.0018 | +0.0048 | −0.0128 | −0.0109 |
| V1 | subj05 | −0.0195 | −0.0326 | −0.0020 | −0.0262 | −0.0102 | −0.0049 | −0.0102 |
| V1 | subj06 | −0.0209 | −0.0175 | −0.0202 | −0.0084 | +0.0043 | −0.0037 | −0.0157 |
| V1 | subj07 | −0.0147 | −0.0182 | −0.0198 | −0.0033 | −0.0097 | −0.0240 | −0.0102 |
| V1 | subj08 | −0.0159 | −0.0387 | +0.0004 | −0.0349 | −0.0008 | −0.0401 | −0.0095 |
| V1 | mean of 8 | −0.0169 | −0.0262 | −0.0081 | −0.0094 | −0.0038 | −0.0175 | −0.0103 |
| LO-obj | subj01 | −0.0267 | −0.0096 | −0.0062 | +0.0002 | −0.0158 | −0.0105 | −0.0096 |
| LO-obj | subj02 | −0.0178 | −0.0070 | −0.0185 | −0.0037 | −0.0241 | −0.0147 | −0.0253 |
| LO-obj | subj03 | −0.0128 | −0.0044 | −0.0365 | −0.0139 | −0.0011 | −0.0108 | −0.0387 |
| LO-obj | subj04 | −0.0123 | −0.0045 | −0.0015 | +0.0020 | −0.0040 | −0.0054 | +0.0047 |
| LO-obj | subj05 | −0.0155 | −0.0079 | −0.0054 | −0.0088 | −0.0103 | −0.0162 | −0.0074 |
| LO-obj | subj06 | −0.0124 | −0.0042 | −0.0161 | −0.0046 | −0.0102 | −0.0035 | −0.0125 |
| LO-obj | subj07 | −0.0080 | −0.0057 | +0.0117 | +0.0339 | −0.0213 | −0.0096 | +0.0243 |
| LO-obj | subj08 | −0.0058 | −0.0040 | −0.0165 | −0.0242 | −0.0074 | −0.0028 | −0.0014 |
| LO-obj | mean of 8 | −0.0139 | −0.0059 | −0.0111 | −0.0024 | −0.0118 | −0.0092 | −0.0082 |

N, A and W: units and voxels centred over the 515 images. The 'lateral' variant, k = 1 and all
rho values are in `results/step42/group.csv` and `results/step42/subjects.csv`.

## Files

| File | Content |
|---|---|
| `results/step42/fetch_subj02.csv` … `fetch_subj08.csv` | per-session download log (size, times, free space, kept trials, invalid voxels, z-score check) |
| `results/step42/gates.csv` | G1–G5 per subject |
| `results/step42/subjects.csv` | per subject: rho per cell, seed and presentation; voxel counts |
| `results/step42/group.csv` | per cell and subject set (all 8, subj02–08): mean Δ, mean rho, signs, sign-flip p, interval |
| `results/step42/summary.csv` | P1–P4, Q1–Q4, runtime of the group step |

## Runtime and disk

| Item | Value |
|---|---|
| Download subj02–08, sessions 1–30 | 94.0 GB; loop from 08:20:40 to 10:36:28 UTC (2 h 16 min, 932–1297 s per subject for download and processing) |
| RSA per subject (all cells, N, A, W, RDM cache) | 60–161 s |
| Group step (1000-resample bootstrap, 8 workers) | 5852 s (97.5 min) |
| Free space | at least 114.4 GB during the downloads (floor: 30 GB) |
| Stored under `NSD_DIR` | 1.3 GB in total for 8 subjects |

## Open points

- The amendment says "existing functions from the step 6/7 scripts". The functions used are
  `step25_weighted_angle.nwa` with `step9_separation.normf` and `components` (`config_step42b.yaml`,
  `components.implementation`); test (v) shows they reproduce the step 9 component values.
- The amendment writes N_ij = (n_i − n_j)²; the code uses |n_i − n_j|. Spearman rho is identical
  because the map is monotone.
- G4 is lowest for subj08 (0.104 and 0.092) and subj03 (0.148 and 0.126). The gate is passed;
  noise ceilings are not computed.
- subj06: 6 voxels were invalid in at least one session and are excluded from all ROIs.
- Logs (ignored by git): `data/step42b_run.log`, `data/step42b_group.log`.

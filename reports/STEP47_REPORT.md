# Step 47: run structure in THINGS-720 and ridge replication in NSD

Predictions: `amendments/2026-11-21_step47-predictions.md` (RC1–RC4, CO1, RD1, RD2 and the
interpretation rule), `config_step47.yaml` (hash aa9c06baf1a1). Feasibility and gates:
`reports/STEP47_SETUP.md`. `config.yaml` (206fa875cca5) and the configs of steps 40–45 are unchanged.
NSD data and voxel-level derivatives stay under `NSD_DIR`; the run-centred THINGS betas are cached
under `RIDGE_DATA_DIR/step47`. The repository holds code and aggregated CSVs.

## Order of commits

| Commit | Time (local, +0100) | Content |
|---|---|---|
| `4f16e42` | 2026-11-21 09:57:43 | Add step 47 predictions (amendment, `config_step47.yaml`, `reports/STEP47_SETUP.md`) |
| `8d1532c` | 2026-11-21 10:06:38 | Add step 47 scripts with tests |
| this commit | | Add step 47 results |

Part A started after `8d1532c` (detached process), then the g = 4 and g = 8 NSD feature extraction,
then part B (detached process). Before every commit `scripts/nsd_commit_check.py` ran (also as the
pre-commit hook).

## Part 0 (summary of `STEP47_SETUP.md`)

- Run means: the ROI rows of the h5 files (all 9840 trials) are read without loading the files;
  the run mean uses **all trials of the run (82 = 72 main + 10 test)** after the arm_a z-score.
- Gate: the stored arm_a betas equal the z-scored h5 responses (max |Δ| 0 for 20 and for all 720
  trials, all subjects); the stored V1 RDMs equal the correlation distance of these betas (0).
- Predecessors: 711 / 711 / 710 of 720 trials have one (86 / 86 / 88 of them test images); all
  predecessor images found.
- NSD ridge: test fits with 515 stimuli and 1400 voxels run (5–6 s); estimates part A 0.29 h, part B
  0.36 h (× 1.5; limit 8 h each).

## Tests (`tests/test_step47.py`, before the runs)

| Check | Result |
|---|---|
| (i) run centring removes synthetic run offsets | max \|d\| 6.6e-15; run means after centring 5.8e-15 |
| (ii) uncentred path: paper cell; brain RDMs vs `outputs_720` | +0.030445938; 0 |
| (ii) uncentred multiverse rho vs `results/step7/multiverse.csv` (36 cells × 3 conditions × 30 checkpoints × 3 subjects) | 9720 values, max \|d\| 2.6e-16 |
| (ii) bootstrap interval, raw / correlation / paper, vs step 7 | +0.030446 [+0.018149, +0.042293], max \|d\| 8.0e-17 |
| (ii) class rule with the step 7 intervals | 28 analyses, 5 above / 9 below / 14 include 0 (= step 28) |
| (iii) RC3 statistic, permutation, percentile on synthetic data | passed |
| (iv) ridge path reproduces `results/prereg/primary.csv`, THINGS sub-01 V1, Random s0 and BP s0, g = 4 | max \|d r\| 1.1e-16; low-edge hits and gate identical |
| (v) reliability on synthetic data | identical presentations → 1; known reliability within 0.0023; Spearman–Brown relations exact |

Existing suites unchanged and passing (phase 2 97/97, loader 41/41, lower edge 2/2, banded 7/7,
cvrsa 4/4, noise curve 4/4, `test_nsd --skip-gate` 7/7, `test_step44` 7/7, `test_step45 --only ii,iii` 2/2);
`config.yaml` hash 206fa875cca5. Checks inside part A: the uncentred V1 Δ equals the paper cell;
the z-scored h5 rows equal the arm_a betas; every run of the session has 82 trials; identity draws
equal point estimates (1e-12).

## Predictions (`results/step47/summary.csv`)

| Rule | Criterion | Value | Result |
|---|---|---|---|
| RC1 | run-centred V1 Δ ≤ +0.010 | +0.0070 (uncentred +0.0304) | met |
| RC2 | run-centred ρ(V1, L) / LOSO bound < 0.5 | 0.627 (uncentred 1.043) | not met |
| RC3 | run variance of run-mean L: sub-01/02 > sub-03 | 2.06e-4 vs 5.02e-4 | not met |
| RC4 | run-centred multiverse: ≤ 2 distinct analyses with the interval above 0 | 4 of 28 | not met |
| CO1 | \|change of V1 Δ by partialling predecessor L\| ≥ 0.005 | −0.00001 | not met |
| RD1 | NSD ridge, g = 4: Random − BP < 0 in ≥ 6/8 subjects, p ≤ 0.05 | 8/8, p 0.0039 | met |
| RD2 | the same at g = 1 and g = 8 | 8/8, p 0.0039 at both | met |

**Interpretation (amendment rule, part A): no statement of the rule applies.** The run-level
attribution needs RC1, RC2 and RC3; RC2 and RC3 are not met. RC1 is met, so "run structure is not
the explanation" does not apply either. CO1 is not met. Descriptively, run centring removes 77 % of
the V1 advantage (+0.0304 → +0.0070; explained fraction 0.769), and run centring followed by L + C
leaves +0.0014.

## A. THINGS-720 per subject (`results/step47/run_centring.csv`, `carryover.csv`)

| ROI | Subject | Δ raw | run-centred | run-centred, after L + C | after predecessor L (raw)¹ |
|---|---|---|---|---|---|
| V1 | sub-01 | +0.0330 | +0.0037 | −0.0019 | +0.0330 (from +0.0330) |
| V1 | sub-02 | +0.0509 | +0.0157 | +0.0056 | +0.0508 (from +0.0508) |
| V1 | sub-03 | +0.0074 | +0.0017 | +0.0005 | +0.0079 (from +0.0079) |
| V1 | mean | +0.0304 | +0.0070 | +0.0014 | +0.0306 (from +0.0306) |
| LOC | sub-01 | −0.0016 | −0.0040 | −0.0021 | −0.0019 |
| LOC | sub-02 | −0.0242 | −0.0040 | −0.0042 | −0.0234 |
| LOC | sub-03 | −0.0080 | −0.0044 | −0.0038 | −0.0075 |
| LOC | mean | −0.0113 | −0.0041 | −0.0033 | −0.0109 |

¹ On the 199,263 cross-run pairs of the 701 images with a predecessor in every subject; "from" is Δ
on the same pairs without partialling. Spearman(L, L_prev): −0.024 (sub-01, sub-02), +0.004 (sub-03).

### Luminance and LOSO bound, with and without run centring

| ROI | Variant | ρ(brain, L), mean | per subject | LOSO lower bound | ratio |
|---|---|---|---|---|---|
| V1 | raw | 0.0536 | 0.0544 / 0.0958 / 0.0107 | 0.0514 | 1.043 |
| V1 | run-centred | 0.0273 | 0.0270 / 0.0462 / 0.0085 | 0.0435 | 0.627 |
| LOC | raw | 0.0009 | 0.0006 / 0.0046 / −0.0024 | 0.0183 | 0.050 |
| LOC | run-centred | −0.0003 | −0.0011 / 0.0014 / −0.0013 | 0.0203 | −0.016 |

### RC3: run-mean luminance (`results/step47/run_luminance.csv`)

| Assignment | variance of the 10 run means of L | percentile among 10,000 random assignments |
|---|---|---|
| sub-01 / sub-02 | 2.06e-4 | 64.8 |
| sub-03 | 5.02e-4 | 99.6 |
| null (10,000, seed 20261121) | mean 1.85e-4, 5–95 % [0.68e-4, 3.50e-4] | |

The sub-01/02 assignment is unremarkable; the sub-03 assignment has unusually large run differences
in luminance, the opposite of RC3.

### RC4: run-centred multiverse (`results/step47/multiverse_runcentred.csv`)

36 cells form 28 distinct analyses (step 28 rule; within-class max |d| 0, smallest between-class
max |d| 1.2e-4).

| | above 0 | below 0 | includes 0 | distinct analyses |
|---|---|---|---|---|
| THINGS-720, run-centred (this step) | 4 | 9 | 15 | 28 |
| THINGS-720, raw (steps 7 / 28) | 5 | 9 | 14 | 28 |
| NSD V1 (step 43) | 0 | 13 | 15 | 28 |

The four classes above 0 after run centring are all Euclidean (raw/center × paper/center720,
raw/center × z720, scale/z × paper/center720, scale/z × z720; gaps +0.016 to +0.024). The paper cell
(raw / correlation / paper) moves from +0.0304 [+0.018, +0.042] to +0.0070 [−0.0011, +0.0146].

## B. Ridge replication in NSD (`results/step47/nsd_ridge.csv`)

Pooled out-of-fold r, median over reliable voxels, mean over seeds 0–4; Δ = Random − BP. Bank =
luminance-and-contrast bank (2g² features). All 528 fits are valid: 181 low-edge hits at reliable
voxels (V1 g = 1: 158, LO-obj g = 1: 23; none at g = 4 or 8), all classified as plateau.

### V1 per subject and g

| g | Subject | Random | BP | Δ | Bank | reliable voxels |
|---|---|---|---|---|---|---|
| 1 | subj01 | 0.164 | 0.187 | −0.024 | 0.189 | 1181 / 1350 |
| 1 | subj02 | 0.256 | 0.272 | −0.016 | 0.282 | 815 / 1102 |
| 1 | subj03 | 0.291 | 0.306 | −0.015 | 0.306 | 912 / 1254 |
| 1 | subj04 | 0.277 | 0.289 | −0.012 | 0.299 | 600 / 877 |
| 1 | subj05 | 0.367 | 0.380 | −0.013 | 0.386 | 798 / 1113 |
| 1 | subj06 | 0.214 | 0.233 | −0.019 | 0.241 | 920 / 1121 |
| 1 | subj07 | 0.210 | 0.231 | −0.022 | 0.223 | 793 / 1142 |
| 1 | subj08 | 0.305 | 0.317 | −0.012 | 0.327 | 550 / 1074 |
| 4 | subj01 | 0.336 | 0.384 | −0.048 | 0.406 | |
| 4 | subj02 | 0.353 | 0.408 | −0.055 | 0.436 | |
| 4 | subj03 | 0.375 | 0.432 | −0.057 | 0.444 | |
| 4 | subj04 | 0.355 | 0.408 | −0.053 | 0.439 | |
| 4 | subj05 | 0.401 | 0.449 | −0.048 | 0.466 | |
| 4 | subj06 | 0.257 | 0.314 | −0.057 | 0.353 | |
| 4 | subj07 | 0.301 | 0.356 | −0.056 | 0.368 | |
| 4 | subj08 | 0.357 | 0.413 | −0.056 | 0.426 | |
| 8 | subj01 | 0.351 | 0.421 | −0.070 | 0.442 | |
| 8 | subj02 | 0.364 | 0.444 | −0.080 | 0.470 | |
| 8 | subj03 | 0.368 | 0.445 | −0.077 | 0.466 | |
| 8 | subj04 | 0.345 | 0.419 | −0.074 | 0.451 | |
| 8 | subj05 | 0.389 | 0.447 | −0.058 | 0.463 | |
| 8 | subj06 | 0.254 | 0.320 | −0.067 | 0.358 | |
| 8 | subj07 | 0.295 | 0.363 | −0.068 | 0.379 | |
| 8 | subj08 | 0.349 | 0.419 | −0.070 | 0.433 | |

| ROI | g | mean Δ | negative | p (one-sided) |
|---|---|---|---|---|
| V1 | 1 | −0.0165 | 8/8 | 0.0039 |
| V1 | 4 | −0.0537 | 8/8 | 0.0039 |
| V1 | 8 | −0.0703 | 8/8 | 0.0039 |
| LO-obj | 1 | −0.0324 | 8/8 | 0.0039 |
| LO-obj | 4 | −0.0218 | 8/8 | 0.0039 |
| LO-obj | 8 | −0.0231 | 8/8 | 0.0039 |

Random is below BP in all 5 seeds for every subject, g and ROI. MSE skill gives the same ordering at
V1 (e.g. g = 4: Random 0.067–0.164, BP 0.101–0.205). The bank alone exceeds both networks at V1 in
every subject and g except at g = 1 in subj03 (bank 0.3055 vs BP 0.3058) and subj07 (0.223 vs 0.231); at LO-obj it is below both
networks. Reliable voxels (3r / (1 + 2r) > 0.2) per subject at V1: 1181, 815, 912, 600, 798, 920,
793, 550; at LO-obj 747, 809, 354, 239, 730, 475, 952, 54 (2r / (1 + r) counts in the CSV).

## Files

| File | Content |
|---|---|
| `results/step47/run_centring.csv` | Δ per subject and variant, ρ(brain, L), LOSO bound, ratio; V1 and LOC |
| `results/step47/multiverse_runcentred.csv` | 36 run-centred cells with gap, interval, class; class counts and checks |
| `results/step47/run_luminance.csv` | RC3 per assignment with run means and percentile; null summary |
| `results/step47/carryover.csv` | Δ on the restricted pairs with and without predecessor L; predecessor counts |
| `results/step47/nsd_ridge.csv` | every fit (r, MSE skill, gate, boundary cells); per subject and group rows |
| `results/step47/summary.csv` | RC1–RC4, CO1, RD1, RD2, interpretation, runtimes |

Every CSV carries `config_hash` (aa9c06baf1a1), `git_head` and `seed` (model seed where a row has one,
else empty).

## Runtime

| Part | Estimate (× 1.5) | Actual |
|---|---|---|
| A (THINGS-720) | 0.29 h | 1119 s (multiverse 1047 s) |
| NSD feature extraction, g = 4 and 8 | | 172 s |
| B (528 ridge fits, 8 workers) | 0.36 h | 1974 s |

Part B took 1.5 times its estimate (fits in eight parallel single-thread workers are slower than
the single timed fit).

## Open points

- RC2 uses the LOSO bound recomputed on the run-centred V1 RDMs (0.0435); the ratio with the raw
  bound would be 0.531.
- Spearman–Brown for NSD reliability: the amendment says "Spearman–Brown corrected"; the config
  fixes the correction to the 3-presentation mean, 3r / (1 + 2r), with the 2r / (1 + r) count in the
  CSV. LO-obj in subj08 has 54 reliable voxels (21 with 2r / (1 + r)).
- Part B uses k = 3 targets (z per voxel and session, mean of the three presentations, as step 42)
  and one CV group per image.
- The lower-edge hits at g = 1 (2 or 32 features) were all plateau; no cell was excluded.
- The run-centred multiverse, like the raw one of step 7, bootstraps only the 10 Random / BP
  checkpoints; the class rule uses all 30.
- Logs (ignored by git): `data/step47A_run.log`, `data/step47B_extract.log`, `data/step47B_run.log`.

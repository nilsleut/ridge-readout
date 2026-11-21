# Step 47b: mechanism of the run structure and robustness to run centring

Predictions: `amendments/2026-11-21_step47b-predictions.md` (X1–X3, the interpretation rule,
Y1–Y4), `config_step47b.yaml` (hash 97ffa8ef4507). Feasibility and gates:
`reports/STEP47B_SETUP.md`. `config.yaml` (206fa875cca5), the earlier configs, amendments, results
and `paper/` are unchanged. NSD data and everything derived from it at voxel level (run means,
centred patterns) stay under `NSD_DIR`; the run-centred THINGS caches stay under
`RIDGE_DATA_DIR/step47b`. The repository holds code and aggregated CSVs.

## Order of commits

| Commit | Time (local, +0100) | Content |
|---|---|---|
| `edbb11d` | 2026-11-21 18:22:05 | Add step 47b predictions (amendment, `config_step47b.yaml`, setup report) |
| `919313c` | 2026-11-21 19:01:46 | Add step 47b scripts with tests |
| this commit | | Add step 47b results |

The first NSD download (gate, subj01 session 1) started at 18:23:01 local, after the amendment
commit. Run order: X, Y THINGS, NSD reload (detached), Y NSD (detached). Before every commit
`scripts/nsd_commit_check.py` ran (also as the pre-commit hook).

## X. Mechanism (THINGS-720 session, V1)

The run RDMs use all 820 images of the session (the 720 and the 100 test images, 10 runs of 82).
Null: 10,000 reassignments of the 820 images to 10 runs of 82 (seed 20261122), model side only.

| Subject | A(Random) | A(BP) | ΔA | percentile in null | ρ(brain run RDM, run index) | run-level part of Δ (step 47) |
|---|---|---|---|---|---|---|
| sub-01 | 0.1128 | 0.0669 | +0.0458 | 62.8 | +0.058 | 0.0294 |
| sub-02 | 0.1682 | 0.2427 | −0.0744 | 34.6 | +0.419 | 0.0352 |
| sub-03 | 0.1545 | 0.0402 | +0.1144 | 84.0 | +0.706 | 0.0057 |

Spearman between brain run RDMs (by run index): sub-01 vs sub-02 0.130, sub-01 vs sub-03 0.040,
sub-02 vs sub-03 0.249.

| Prediction | Result | Status |
|---|---|---|
| X1: ΔA > 0 in sub-01 and sub-02 | sub-01 +0.0458, sub-02 −0.0744 | not met |
| X2: ΔA of sub-01 and sub-02 within the central 95 % of the null | percentiles 62.8 and 34.6 | met |
| X3: brain run RDM correlates positively with run index in ≥ 2 of 3 | 3 of 3 (+0.058, +0.419, +0.706) | met |

**Interpretation by the amendment rule: not identified.** X1 is not met, so neither the attribution
to drift with chance alignment nor the attribution to stimulus-driven run-level structure applies.
The brain run RDMs follow the run index in all three subjects (drift across runs), but the alignment
of the run-mean model features with these brain run RDMs does not favour the untrained network in
sub-02, the subject with the largest run-level part.

## Y. Robustness to run centring

Run centring subtracts the voxel mean over all trials of the same run after the existing
normalisation; in NSD per run within session (12 runs of 62–63 trials), before averaging the
presentations. Reliability masks are recomputed on the centred data.

### THINGS

Ridge (prereg aggregation, mean over subjects; seeds with Δ < 0 out of 5; 0 invalid cells):

| Variant | g = 1 | g = 4 | g = 8 |
|---|---|---|---|
| uncentred (`results/prereg`) | −0.0209 (5/5) | −0.0425 (5/5) | −0.0449 (5/5) |
| run-centred | −0.0192 (5/5) | **−0.0400 (5/5)** | −0.0440 (5/5) |
| run-centred, uncentred voxel selection | | −0.0404 (5/5) | |

Reliable voxels uncentred / run-centred: sub-01 501 / 495, sub-02 518 / 518, sub-03 286 / 295.

Test images (100 images, mean Δ over subjects; subjects with Δ < 0 out of 3 in parentheses):

| ROI, k, units | uncentred | run-centred |
|---|---|---|
| V1, k = 1, paper | −0.000159 (2) | −0.006556 (3) |
| V1, k = 1, standardised | −0.015933 (3) | −0.016757 (3) |
| V1, k = 12, paper | −0.017763 (2) | −0.015456 (2) |
| V1, k = 12, standardised | −0.072787 (3) | −0.071008 (3) |
| LO-obj, k = 1, paper | −0.000383 (2) | −0.002608 (2) |
| LO-obj, k = 1, standardised | −0.001621 (2) | −0.001040 (2) |
| LO-obj, k = 12, paper | −0.016108 (3) | −0.015039 (3) |
| LO-obj, k = 12, standardised | −0.006631 (3) | −0.005898 (3) |

Checks: the uncentred test-image Δ reproduces step 41 (max |d| 9.6e-17); the run-centred
THINGS-720 V1 Δ reproduces step 47 (+0.007032, |d| 1.1e-17).

### NSD (8 subjects)

RSA, k = 3, correlation distance (mean Δ; subjects with Δ < 0 out of 8; sign-test p):

| ROI, units | uncentred (step 42) | run-centred |
|---|---|---|
| V1, paper | −0.016943 (8, 0.0039) | **−0.016886 (8, 0.0039)** |
| V1, standardised | −0.026233 (8, 0.0039) | **−0.027017 (8, 0.0039)** |
| lateral, paper | −0.015631 (8) | −0.015630 (8) |
| lateral, standardised | −0.006891 (8) | −0.006900 (8) |
| LO-obj, paper | −0.013931 (8) | −0.013867 (8) |
| LO-obj, standardised | −0.005911 (8) | −0.005856 (8) |

The uncentred values reproduce step 42 (max |d ρ| 9.9e-17). Euclidean distance: see
`robustness.csv` (4–7 of 8 negative, uncentred and centred alike).

Ridge (mean Δ over subjects, 8 of 8 negative in every row, p 0.0039; all 688 fits valid):

| Variant | V1 g = 1 | V1 g = 4 | V1 g = 8 | LO-obj g = 1 | LO-obj g = 4 | LO-obj g = 8 |
|---|---|---|---|---|---|---|
| uncentred (step 47) | −0.0165 | −0.0537 | −0.0703 | −0.0324 | −0.0218 | −0.0231 |
| run-centred | −0.0164 | **−0.0547** | −0.0735 | −0.0342 | −0.0224 | −0.0241 |
| run-centred, step 47 voxel selection | | −0.0543 | | | | |
| uncentred, all V1 voxels (Y4) | | **−0.0539** | | | | |

V1, g = 4 per subject:

| Subject | uncentred | run-centred | run-centred, step 47 voxels | uncentred, all voxels |
|---|---|---|---|---|
| subj01 | −0.0481 | −0.0493 | −0.0492 | −0.0440 |
| subj02 | −0.0551 | −0.0538 | −0.0532 | −0.0588 |
| subj03 | −0.0570 | −0.0585 | −0.0584 | −0.0634 |
| subj04 | −0.0526 | −0.0552 | −0.0557 | −0.0635 |
| subj05 | −0.0480 | −0.0495 | −0.0494 | −0.0432 |
| subj06 | −0.0571 | −0.0624 | −0.0620 | −0.0566 |
| subj07 | −0.0556 | −0.0540 | −0.0533 | −0.0514 |
| subj08 | −0.0558 | −0.0547 | −0.0530 | −0.0503 |

Reliable V1 voxels: 550–1181 uncentred, 567–1182 run-centred; all V1 voxels 877–1350.

Multiverse (V1, 36 cells, 28 distinct analyses, 1001 draws), run-centred: **0 above 0, 13 below 0,
15 include 0**, the same class counts as the uncentred step 43 multiverse (0 / 13 / 15). The paper
cell of the multiverse equals the run-centred RSA (|d| 1.4e-16).

### Predictions

| Prediction | Result | Status |
|---|---|---|
| Y1: THINGS ridge, run-centred, g = 4, Δ < 0 in 5 of 5 seeds | −0.0400, 5/5 | met |
| Y2: NSD RSA, run-centred, V1 paper and standardised, ≥ 6 of 8 each | 8/8 (−0.0169) and 8/8 (−0.0270) | met |
| Y3: NSD ridge, run-centred, g = 4, V1, ≥ 6 of 8 | 8/8 (−0.0547) | met |
| Y4: NSD ridge, uncentred, g = 4, all V1 voxels, ≥ 6 of 8 | 8/8 (−0.0539) | met |

In contrast to the THINGS-720 RSA advantage of the untrained network (77 % removed by run centring
in step 47), the BP advantages in THINGS ridge, THINGS test images, NSD RSA, NSD ridge and the NSD
multiverse class counts are essentially unchanged by run centring.

## NSD reload

Sessions 1–30 of all 8 subjects (240 sessions), with all 750 trials per session. Each session passed
the gate: the 515-image trials were identical to the stored step 42 betas (max |Δ| 0 in all 240).
Every session had 12 runs of 62–63 trials. Download 108.7 GB in 8225 s and processing 1689 s
(overlapped), 8416 s wall time in total. The originals were deleted after processing. Only the run
means are kept (`NSD_DIR/runmeans`). Log: `nsd_runmeans_log.csv`.

## Runtimes and disk

| Part | Estimate (× 1.5) | Measured |
|---|---|---|
| X | 0.023 h | 46 s |
| Y THINGS | 0.22 h | 438 s |
| NSD reload | 3.36 h | 8416 s (2.34 h) |
| Y NSD | 1.00 h | 3799 s (1.06 h) |

Free disk space never fell below 103 GB (required ≥ 30 GB).

## Outputs (`results/step47b/`)

`mechanism.csv`, `permutation_summary.csv`, `robustness.csv`, `nsd_runmeans_log.csv`,
`summary.csv`. Each has `config_hash`, `git_head` and `seed`.

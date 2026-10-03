# Step 2D — lower-edge diagnosis, Random Weights (BN-calibrated) s3, g = 1

**Diagnosis only. Exploratory, sub-01 only.** The selection rule (min-MSE, D2), the
alpha grid and the D1 assertion are unchanged. No tolerance form is proposed. The
extended grid is diagnostic and feeds no result.

Script: `scripts/step2_lower_edge.py`. Data: `results/step2/lower_edge_diagnosis.csv`
(one row per hit), `results/step2/lower_edge_curves.csv` (inner-CV MSE curves).
Config hash `206fa875cca5`.

## Gate

`tolerance_measurement.measure` (imported, D1 not asserted) on this cell reproduces
`results/tolerance_all_rungs.csv`: median r over 501 reliable voxels 0.0619118702
(|Δ| 6.9e-18), 4 low-edge selections among reliable voxels. Passed.

## The four hits

Extended grid = standard 29 points + 1e-7, 1e-8, 1e-9, 1e-10, 1e-12. Relative gain =
(MSE(1e-6) − MSE_min,ext) / MSE(1e-6). Classification rule fixed before the run:
"True optimum below grid" if the extended minimum is at α < 1e-6 and gain ≥ 1e-3;
"Plateau" if gain < 1e-3; otherwise "unclear".

| outer fold | V1 voxel (local) | α_min ext. | MSE(1e-6) | MSE_min ext. | rel. gain | class |
|---|---|---|---|---|---|---|
| 3 | 206 | 1e-10 | 0.871570 | 0.871427 | 1.6e-4 | **Plateau** |
| 5 | 934 | 1e-10 | 0.821248 | 0.821131 | 1.4e-4 | **Plateau** |
| 5 | 983 | 1e-10 | 0.863835 | 0.863784 | 5.8e-5 | **Plateau** |
| 5 | 989 | 1e-10 | 0.847304 | 0.847292 | 1.3e-5 | **Plateau** |

All four are **Plateau**: the minimum lies below the grid, but the gain is 6–77 times
smaller than the 1e-3 threshold. Four **different** voxels, in two outer folds (3 and 5).

Design of the standardized training matrix: d = 32, rank 32 in the outer (n = 630) and
every inner (n = 504) training set, condition number 361 / 389 (outer, folds 3 / 5),
≤ 467 (inner), **0 dead features**. The design is well-conditioned.

Example curve (fold 3, voxel 206): MSE is 0.871570 from 1e-6 to 1e-5, 0.871569 at 1e-7,
0.871555 at 1e-9, 0.871427 at 1e-10 and 0.917641 at 1e-12.

## Other conditions at g = 1 (min-MSE, from tolerance_all_rungs.csv)

The only cell in all 30 at g = 1 with low-edge reliable selections is this one. 0 for
Backprop s3, 0 for every other Backprop seed, 0 for Random Weights (BN-calibrated)
s0, s1, s2, s4.

## Observation (not a classification input)

The dual kernel K = XXᵀ (504 × 504) has rank 32; its remaining 472 eigenvalues are
round-off, |λ| ≤ 1.9e-12 (folds 3 and 5, first two inner splits). Only near α ≈ 1e-10
do these modes start to enter the fit, and at 1e-12 they dominate (MSE +5%). The small
dip at 1e-10 therefore plausibly comes from round-off modes rather than the design; this
was not tested further.

Note on `PILOT_REPORT.md` §2: the counts given there (1, 3, 3, 1 voxels in folds
1, 3, 5, 7, voxels 208 and 276) refer to "below 1e-6 on a 1e-14 … 1e8 probe grid", not to
low-edge selections on the operative grid, and use an indexing that is not documented.
They are not directly comparable to the four hits above.

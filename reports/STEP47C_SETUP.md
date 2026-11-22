# Step 47c setup: gates

Checked before the amendment `amendments/2026-11-22_step47c-predictions.md`. Only the values and
gates below were computed; no result file was written. Configuration: `config_step47c.yaml` (hash
ea2fd24bd0a1). THINGS-720 only; no NSD data.

## 1. Multiverse cell raw / correlation / center720 (V1, Random − BP)

From `results/step7/multiverse.csv` (kind `cell`, V1):

| Condition | gap | 95 % interval |
|---|---|---|
| NATIVE 224 (the paper-cell condition) | +0.004573 | [−0.003386, +0.012460] |
| NATIVE 32 | −0.001541 | [−0.007360, +0.004357] |
| UPSAMPLED 224 | +0.011220 | [+0.003637, +0.018898] |

For comparison, the paper cell (raw / correlation / paper, NATIVE 224) is +0.030446.

## 2. Gates

Data: `step47_runs.load_resp` per subject (z per voxel over all 9840 trials). The 720 rows equal the
arm_a betas exactly in all three subjects (`np.array_equal`). Session of the 720: 820 trials,
10 runs of 82.

| Gate | Result |
|---|---|
| m + d_r = run mean μ_r (all runs, all subjects) | max \|Δ\| 2.2e-16 (≤ 1e-12), passed |
| V3 (m and d_r removed) = step 47 run-centred betas | max \|Δ\| 8.9e-16 per subject |
| Δ(V3) = step 47 run-centred V1 Δ | +0.007032170799 = +0.007032170799, passed |
| Δ(V0) = paper cell | +0.030445937983 = +0.030445937983, passed |

Per subject Δ(V0): 0.033014, 0.050909, 0.007415; Δ(V3): 0.003654, 0.015718, 0.001725 (step 47
values).

## 3. Runtime

The gate run (3 subjects, two variants plus the reproductions) took 21 s. The full run (6 variants
× 2 ROIs, luminance of the 720 images) is estimated below 0.1 h (× 1.5), far below 8 h.

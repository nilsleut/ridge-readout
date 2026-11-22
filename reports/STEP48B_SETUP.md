# Step 48b setup: gate and choice of the amplitude RDM

Checked before the amendment `amendments/2026-11-22_step48b-predictions.md`. Configuration:
`config_step48b.yaml` (hash 73c5a25219bb). `config.yaml` (206fa875cca5) unchanged. No new downloads;
the step 48 data under `NSD_DIR/raw48` and `RIDGE_DATA_DIR/step48` are used. No dominance or partial
value was computed on real data before the amendment.

## 1. Simulation before the amendment (synthetic patterns, no real data)

Patterns X_i = c · p + b_i · U_i (common profile p, per-pattern amplitude b_i, as step 48 test (ii));
model patterns built the same way from an independent profile and noise with the same b_i (a pure
amplitude effect). n = 200, 300 brain and 400 model features.

| c | ρ(RDM, \|g_i − g_j\|) | ρ(RDM, 2 − g_i − g_j) | ρ(model, brain) | two-sided partial, difference form | two-sided partial, sum form |
|---|---|---|---|---|---|
| 0 | +0.03 | +0.09 | — | — | — |
| 0.5 | −0.21 | +0.75 | 0.55 | 0.53 | 0.04 |
| 1 | −0.11 | +0.94 | 0.85 | 0.85 | 0.04 |
| 2 | +0.11 | +0.98 | 0.94 | 0.94 | −0.01 |
| 4 | +0.23 | +0.99 | 0.96 | 0.96 | 0.02 |

A second design (profile amplitude a_i varying, X_i = c a_i p + U_i) gave the same picture (c = 0.4:
ρ(model, brain) 0.70, partial on the difference form 0.65, on the sum form 0.04). The difference form
the step 48b draft named does not capture the amplitude structure of correlation-distance RDMs under a
common profile; the sum form does. The amendment therefore uses A_ij = 2 − g_i − g_j and A^M_ij =
2 − g^M_i − g^M_j for D1–D3 and reports the difference form without prediction (decision of
2026-11-22).

## 2. Gate (`python scripts/step48b_dominance.py gate`, 207 s)

The step 48b code path (Spearman as Pearson of ranks over the step 48 pair sets; gains with
`step48_amplitude.gain`) against `results/step48` (read with `float_precision="round_trip"`):

| Quantity | step 48b | step 48 | max \|d\| (mean and per subject) |
|---|---|---|---|
| Δ(T-raw, THINGS-720, V1) | +0.099551 | +0.099551 | 8.3e-17 |
| Δ(N-raw, k = 1, V1) | +0.116497 | +0.116497 | 2.8e-17 |
| g: brain_gain mean and SD (180 rows) | 148 / 180 rows bitwise identical | | 9.9e-17 |
| g vs g^M: brain_model_gain (1800 rows) | 1492 / 1800 bitwise identical | | 9.9e-17 |
| g^M: model_gain_stat (240 rows) | 126 / 240 bitwise identical | | 9.7e-17 |

**Passed** (tolerance 1e-12, as in the earlier gates). The remaining differences are 1–2 ulp: the same
row is bitwise identical when computed alone and differs by one ulp inside the loop, i.e. rounding of
the BLAS dot products in the cosine depending on memory alignment, not a different computation.

**Defect found in step 48 results (not changed):** in `results/step48/amplitude.csv` the `seed`
column is empty for all THINGS rows (360 brain_model_gain, 180 model_gain_stat rows), because
`step48_conventions.write_part` drops the seed column of the existing part when the next part is
merged. The values are complete and in generation order (seeds 0–4 within each condition); the gate
reconstructs the seed from that order.

## 3. Runtime estimate (× 1.5)

The gate (all units, plain ρ and gains) took 207 s. The run adds per unit two rank residualisations
and two dominance Spearman values; model-side residuals are cached per pair set. Estimate ≈ 3 × 207 s
× 1.5 ≈ **0.26 h**, below 8 h. Disk: 105 GB free (≥ 30 GB required); the run writes aggregates only.

# Preregistered primary analysis

## Configuration

- **Frozen configuration:** `PREREGISTRATION.md` and `config.yaml` (hash `206fa875cca5`,
  schema_version 2).
- **Amendments in force:**
  - D1 / D2 / D3 (PREREGISTRATION.md §9);
  - `amendments/2026-10-03_tolerance-rule.md` (min-MSE);
  - `amendments/2026-10-03_d1-plateau-gate.md` (no abort; post-run plateau gate);
  - `amendments/2026-10-13_g2-local-extraction.md` (g = 2 feature source and
    equivalence gate).
- **Driver:** `scripts/pilot_fit.py::main`, unchanged, called once per subject by
  `scripts/prereg_primary.py` (commit `02c2fee`). Only module globals were rebound:
  - subject;
  - grid list [1, 2, 4, 8];
  - feature path for g = 2;
  - output directory;
  - `fit_condition` with `assert_low_boundary=False` and post-fit plateau gate.
- **Fixed by the frozen configuration:**
  - Arm A: 720 single-trial betas, V1;
  - reliable voxels: own split-half > 0.2 (501 / 518 / 286);
  - Conv1 post-MaxPool at NATIVE 224 px, all 30 checkpoints;
  - dual ridge, alpha grid 1e-6 … 1e8 (29), min-MSE, 8 × 5 CV with partition seed
    20260818.
- **Aggregation:** `scripts/prereg_aggregate.py` (fits nothing). It computes the median r
  over reliable voxels, the mean over the three subjects, the paired difference within
  seed, the exact one-sided sign-flip test over all 32 assignments, and Δ_min with
  `pilot_fit.delta_min_signflip` (unchanged).
- **Runtime:** estimate 0.59 h, actual 42 min (426 fits).

## Feature sources and equivalence gate

- **g ∈ {1, 4, 8}:** the existing Modal T4 features (`pilot_features/`).
- **g = 2:** local CPU extraction with the existing `extract_pilot` code
  (`scripts/prereg_extract.py`, commit `93263d7`).
- **Original reproduction gate:** absolute 1e-6 against Modal for g ∈ {1, 4, 8};
  **failed**.
  - max absolute deviation 2.1e-5;
  - max relative to the array maximum 4.6e-6;
  - min Pearson 0.99999999999997;
  - pixel and luminance features bit-identical.
- **Replacement gate** (amendment `e54fd83`, script `b2045d0`,
  `results/prereg_gate/g4_equivalence.csv`): ridge at g = 4, sub-01, 30 checkpoints,
  with Modal and with local features.
  - max |Δ median r| = **1.3e-8**;
  - |Δ(Random − BP)| = **2.9e-9** (thresholds 1e-4);
  - no low-edge hits in either run.
  - **Passed.** g = 2 is included.

## Results per subject and g

Mean over the five seeds of the median r over reliable V1 voxels
(`results/prereg/primary.csv`).

| Subject | g | Random | Backprop | Random − BP |
|---|---|---|---|---|
| sub-01 | 1 | 0.066 | 0.094 | −0.029 |
| | 2 | 0.056 | 0.104 | −0.049 |
| | 4 | 0.072 | 0.123 | −0.051 |
| | 8 | 0.068 | 0.126 | −0.058 |
| sub-02 | 1 | 0.074 | 0.096 | −0.023 |
| | 2 | 0.069 | 0.104 | −0.034 |
| | 4 | 0.077 | 0.122 | −0.045 |
| | 8 | 0.076 | 0.122 | −0.047 |
| sub-03 | 1 | −0.009 | 0.002 | −0.011 |
| | 2 | −0.019 | 0.002 | −0.021 |
| | 4 | −0.019 | 0.013 | −0.032 |
| | 8 | −0.016 | 0.014 | −0.030 |

sub-01 at g ∈ {1, 4, 8} reproduces `results/tolerance_all_rungs.csv` (−0.0288, −0.0508,
−0.0583).

## Sign-flip tests (mean over subjects; `results/prereg/signflip.csv`)

Paired one-sided exact sign-flip over 5 seeds (32 assignments), alternative "contrast
> 0". Δ_min = minimum detectable difference at 80% power (simulated), from the
seed-level SD.

| g | Random − BP (primary) | SD | seeds > 0 | p (one-sided) | Δ_min |
|---|---|---|---|---|---|
| 1 | −0.021 | 0.006 | 0/5 | 1.000 | 0.010 |
| 2 | −0.035 | 0.006 | 0/5 | 1.000 | 0.011 |
| 4 | −0.043 | 0.010 | 0/5 | 1.000 | 0.016 |
| 8 | −0.045 | 0.009 | 0/5 | 1.000 | 0.016 |

Secondary contrasts against BP; the same test, all with 0/5 seeds positive and
p = 1.000:

| g | BN-cal − BP | FA − BP | PC − BP | STDP − BP |
|---|---|---|---|---|
| 1 | −0.021 | −0.026 | −0.018 | −0.023 |
| 2 | −0.035 | −0.042 | −0.032 | −0.035 |
| 4 | −0.045 | −0.053 | −0.041 | −0.044 |
| 8 | −0.047 | −0.055 | −0.043 | −0.047 |

## Verdict (PREREGISTRATION.md §3)

- **§3a** — "the claim requires it to hold at g ≥ 4". Operationalised before the run
  (aggregation script docstring) as one-sided p ≤ 0.05 at g = 4 and g = 8. The result is
  p = 1.000 at both: **not met**.
- **§3b** — "'The effect disappears' requires Δ_min to be smaller than the effect size
  the paper reports." Δ_min at g ≥ 4 is at most 0.016, below the reference effect
  +0.044 (config.yaml): **"The effect disappears."**
- The contrast is not merely absent but reversed. Backprop exceeds the untrained
  network at every rung, in every subject, in every seed.
- §3c (saturation) does not apply in its stated form: the six rules do not fall inside
  a band narrower than their distance to the pixel baseline (Backprop lies clearly above
  the rest).

## Instruments (as implemented)

**Control band** (`results/prereg/instruments_control_band.csv`): median r over
reliable voxels, mean over subjects.

| g | pixels | luminance + contrast | Gabor | permuted (negative test) |
|---|---|---|---|---|
| 1 | 0.006 | 0.081 | 0.010 | −0.078 |
| 2 | 0.005 | 0.087 | −0.006 | −0.075 |
| 4 | 0.021 | 0.106 | −0.002 | −0.074 |
| 8 | 0.018 | 0.111 | −0.000 | −0.075 |

- The 2-channel luminance + contrast bank (2 g² features) predicts V1 as well as or
  better than every network except Backprop. The pixel and Gabor banks sit near zero.
- The permuted control is negative rather than ≈ 0 at every g. This is the known
  collapse signature (pooled out-of-fold predictions near the training mean; see
  `reports/TOLERANCE_REPORT.md` §1), not a pipeline failure.

**Per-image cell shuffle** (`instruments_cell_shuffle.csv`; seed 0, Random and
Backprop, as implemented). Gain = intact − shuffled.

| Subject | g | Random gain | Backprop gain |
|---|---|---|---|
| sub-01 | 2 / 4 / 8 | +0.010 / +0.030 / +0.029 | +0.012 / +0.027 / +0.036 |
| sub-02 | 2 / 4 / 8 | +0.011 / +0.011 / +0.019 | +0.006 / +0.028 / +0.026 |
| sub-03 | 2 / 4 / 8 | −0.014 / +0.005 / +0.011 | +0.000 / +0.017 / +0.020 |

At g ≥ 4 the intact grid beats its per-image shuffle in every subject for both
conditions. Part of the gain from g = 1 to g ≥ 4 is therefore spatial correspondence,
not only capacity. This is one seed per condition and descriptive only.

**r(df) path** (`instruments_rdf_path.csv`; first report of this instrument). Peak of
the per-fold r path (median over reliable voxels; median over seeds; sub-01 and sub-02),
and the effective df at the peak:

| Condition | peak r (g = 1 / 2 / 4 / 8) | df at peak (g = 1 / 2 / 4 / 8) |
|---|---|---|
| Backprop | 0.128 / 0.132 / 0.146 / 0.148 | 0.4 / 2.0 / 14.3 / 20.7 |
| Random | 0.111 / 0.105 / 0.111 / 0.108 | 8.1 / 15.7 / 24.3 / 39.6 |
| Random (BN-cal) | 0.111 / 0.104 / 0.110 / 0.106 | 8.2 / 18.9 / 24.5 / 40.0 |
| FA | 0.105 / 0.099 / 0.103 / 0.100 | 9.6 / 18.1 / 23.8 / 38.1 |
| PC | 0.112 / 0.106 / 0.114 / 0.109 | 6.3 / 19.4 / 25.0 / 29.9 |
| STDP | 0.104 / 0.101 / 0.109 / 0.109 | 7.8 / 17.1 / 26.2 / 20.9 |

Backprop reaches a higher peak at fewer effective dimensions than every other condition,
at every g. Caveat: the path uses per-fold Pearson r, which is scale-invariant, so its
peak can sit at strong regularisation (Backprop g = 1, df 0.4). It is descriptive and is
not the selected-alpha score.

## Lower-edge gate (`results/prereg/lower_edge.csv`)

- 26 low-edge selections at reliable voxels, all at g = 1: sub-01 11, sub-02 12,
  sub-03 3.
- By condition: Gabor control 16, Random (BN-cal) s3 4, luminance + contrast control 4,
  PC s2 1, FA s2 1.
- All 26 are **Plateau** (max relative gain 5.8e-5). No cell was excluded.

## Deviations from the preregistration

1. g = 2 extracted locally on the CPU; the original gate failed and the replacement
   (downstream equivalence) gate passed.
2. Resolution sweep for g ≥ 2 not run (224 px only); the extraction code is
   single-resolution.
3. Arm B ridge fits not implemented and not run.
4. Capacity sweep (§5.2) not implemented and not run.
5. PLS-25 (§5.6) not implemented and not run.
6. Leave-one-subject-out lower bound not implemented; no r/ceiling reported.
7. Cell shuffle only for seed 0, Random and Backprop, 224 px (it also runs at g = 2,
   because `pilot_fit.main` applies it for g ≥ 2).
8. D1 abort replaced by the plateau gate (amendment).
9. The analysis was **not blind**. sub-01 at g ∈ {1, 4, 8} and sub-02 / sub-03 at g = 1
   were known from earlier exploratory steps. New in this run: g = 2 for all subjects
   and g ≥ 4 for sub-02 / sub-03.
10. Exploratory analyses outside §8 (Conv3 → LOC, band-pass, ImageNet models) exist in
    this repository. They are reported separately and do not enter this analysis.

## Limits

- N = 3 subjects; sub-03 is near zero in every condition.
- The unit of inference is the seed (n = 5). The smallest attainable p is 1/32, and no
  stimulus bootstrap is part of the preregistered test.
- One resolution (224 px) for the grid ladder.

# Preregistration compliance inventory

Read-only inventory, written before the preregistered primary analysis is run.

**Sources.**
- `PREREGISTRATION.md` (config hash `206fa875cca5`, schema_version 2) and `config.yaml`.
- Amendments in force:
  - D1 / D2 / D3 (PREREGISTRATION.md §9);
  - `amendments/2026-10-03_tolerance-rule.md`: alpha selection stays min-MSE;
  - `amendments/2026-10-03_d1-plateau-gate.md`: no abort on low-edge selections; the
    logged hits are classified post-run; only non-plateau cells are excluded.
- The later step 4–9 amendments concern exploratory analyses only and do not change
  any preregistered element.

**Status values.**
- *yes*: run in the preregistered form for the listed subjects and g.
- *partial*: run, but on fewer subjects, grids or resolutions than preregistered, or
  outside the frozen configuration (e.g. D1 disabled before the plateau gate existed).
- *no*: never run.

## Primary analysis (Arm A)

| Element (preregistered) | Implemented in | Run so far | Subjects / g |
|---|---|---|---|
| Arm A data: 720 single-trial betas, ROI mask, z over all 9,840 trials | `src/ridge/arm_b_betas.py::load_arm_a`; outputs in `arm_b_out/*_arma.npz` | yes (data built) | sub-01, sub-02, sub-03; V1, V2, V3, V4, LOC, IT |
| Voxel selection: own split-half (462 splits, Spearman–Brown) on Arm B, > 0.2 | `src/ridge/arm_b_betas.py::split_half_reliability`; stored as `reliability` in `arm_b_out/*_armb.npz` | yes | all three subjects (V1: 501 / 518 / 286 reliable) |
| Conv1 features, post-MaxPool, grid ladder g ∈ {1, 2, 4, 8} via `adaptive_avg_pool2d` (g = 1 = `x.mean([2,3])`) | `src/ridge/gridpool.py::pool_grid`; extraction `scripts/pilot_features_modal.py::extract_pilot` | partial: features exist for g ∈ {1, 4, 8} at 224 px (`pilot_features/`); **g = 2 never extracted** | 30 checkpoints, 224 px only |
| Resolution sweep 32–224 px for the grid ladder | extraction code is single-resolution (`PILOT_RES = 224`); `scripts/extract_modal.py` covers g = 1 only (bridge control) | partial: g = 1 across 6 resolutions + UP224 in step 6 (`results/step6/ridge_resolution.csv`, GAP features from the paper pipeline); **g ≥ 2 only at 224 px** | step 6: all three subjects, g = 1 |
| Dual ridge, train-fold standardisation, alpha grid 1e-6…1e8 (29), min-MSE (D2, D3, tolerance amendment) | `src/ridge/fit.py::fit_condition`, `src/ridge/ridge_dual.py` | yes | — |
| Nested CV 8 × 5, partition seed 20260818, concept-grouped | `src/ridge/folds.py::outer_inner_splits` | yes | — |
| D1 low-edge assertion → plateau gate (amendment) | `fit.py` (`assert_low_boundary`); `scripts/lower_edge_gate.py::gate` | yes since step 3 | — |
| Primary fit loop (all 30 checkpoints + control band + cell shuffle) | `scripts/pilot_fit.py::main` (`GRIDS = [1, 4]`, one subject constant) | partial: aborted on D1 at Random (BN-cal) s3 g1 (28/30 conditions at g = 1, sub-01, `reports/PILOT_REPORT.md`; no CSV written). The same fits ran later via `scripts/tolerance_measurement.py` with D1 disabled (`results/tolerance_all_rungs.csv`, sub-01, g ∈ {1, 4, 8}, models only), via step 3 with the gate (`results/step3/`, sub-01, g ∈ {1, 4, 8}) and step 4 (`results/step4/subjects.csv`, sub-02 / sub-03, g = 1) | see column 3 |
| Score: median r over reliable voxels | `pilot_fit.py::run` (`np.median(res["r"][reliable])`) | yes (sub-01) | — |
| Aggregation: mean over subjects; per-subject also reported | **not implemented** (config only) | partial: done ad hoc for g = 1 in step 4 / step 6 | — |
| Test: paired one-sided sign-flip over 5 seeds, 32 assignments | **not implemented** as a function (only discussed in `pilot_fit.py` docstring) | no | — |
| Δ_min from the pilot seed-level SD (simulation + parametric cross-check) | `scripts/pilot_fit.py::delta_min_signflip` (not called by `main`) | partial: values in `reports/PILOT_REPORT.md` §4 (sub-01, g = 1) and `reports/TOLERANCE_REPORT.md` §5 | sub-01 |
| Decision rule: survives only if it holds at g ≥ 4; spatial claim only if g ≥ 4 beats cell shuffling | PREREGISTRATION.md §3a | no | — |
| Ceiling Arm A: leave-one-subject-out inter-subject consistency (lower bound) | **not implemented** | no | — |

## Arm B (SNR control)

| Element | Implemented in | Run so far | Subjects / g |
|---|---|---|---|
| Arm B betas: 100 test stimuli, mean over 12 repetitions | `src/ridge/arm_b_betas.py::load_arm_b`; `arm_b_out/*_armb.npz` (`betas`) | yes (data built) | all three |
| Arm B split-half ceiling (462 splits, Spearman–Brown) | `split_half_reliability` | yes (as voxel reliability) | all three |
| Arm B ridge fit, g ∈ {1, 2}, selected and unselected | **not implemented** (no driver fits `armb` betas) | no | — |
| Arm B InternalConsistency ceiling (Brain-Score) | **not implemented** | no | — |

## Instruments against saturation (§5)

| Element | Implemented in | Run so far | Subjects / g |
|---|---|---|---|
| 5.1 r(df) regularisation path | `fit.py` returns `r_path` (n_alpha × n_voxel) and `df` (n_alpha) | computed in every fit, **never reported** | — |
| 5.2 Capacity sweep (random feature subsets 50, 100, 500, 2000, all; seed 20260818) | **not implemented** | no | — |
| 5.3 Spatial-correspondence control: per-image cell shuffle, g = 4 and 8, all resolutions | `src/ridge/gridpool.py::shuffle_cells_per_image`; used in `pilot_fit.py::main` for g ≥ 2, Random and Backprop **seed 0 only**, 224 px only | no (pilot aborted before g = 4) | — |
| 5.4 Control band: pixels, luminance + contrast, Gabor bank, permuted assignment | features: `pilot_features_modal.py` (g ∈ {1, 4, 8}); fits: `pilot_fit.py::main` | partial: permuted control sub-01 g ∈ {1, 4, 8} (`results/step2/d1a_real_voxels.csv`); pixels / lum+contrast / Gabor for sub-03 only (`results/step5/sub03_diagnosis.csv`) | see column 3 |
| 5.5 Δ_min a priori | see primary table | partial | sub-01 |
| 5.6 Brain-Score PLS regression, 25 components | **not implemented** | no | — |

## Determinism and bridge controls (§6)

| Element | Implemented in | Run so far |
|---|---|---|
| Seed-42 determinism gate, tolerance 0.0 | `scripts/determinism_gate.py` (+ `--self-test`) | self-test yes; outcome reported in `reports/PRECONDITIONS_P1_P4.md` (P3) |
| Bridge control: new extraction vs original evaluation (g = 1) | `scripts/extract_modal.py::bridge_control` | yes, 360 cells, max \|Δρ\| = 0 (`reports/PHASE2_REPORT.md` §3) |
| CSV reproduction of `bnfix_sweep.csv` | `scripts/check_csv_reproduction.py` | yes (step 2; Backprop exceeds its 2e-3 tolerance, headline reproduces) |

## Exclusions (§8)

| Excluded in §8 | Complied with? |
|---|---|
| Eccentricity-resolved readout | yes, not run |
| Best-layer-per-condition selection | yes, not run in this repository |
| Layers other than Conv1; ROIs other than the fixed mapping | **no, in the exploratory steps**: Conv3 → LOC (steps 3–9), all four GAP layers stored (step 5), ImageNet-model layers (step 8). None of these enters the preregistered primary analysis. |
| Band-pass decomposition of the stimuli ("separate study") | **no, in the exploratory steps**: band-pass inputs in step 6 (P_mech2) and step 7 (gain). Exploratory only. |
| Receptive-field matching | yes, not run |

## Consequences for the preregistered primary analysis (step 11)

- **Runnable as implemented:** Arm A for all three subjects; the grid ladder at 224 px
  with g ∈ {1, 2, 4, 8} (g = 2 after extracting it with the existing code); the control
  band; the per-image cell shuffle (as implemented: seed 0, Random and Backprop, g ≥ 2);
  the r(df) path (read from the fits); D1 via the plateau gate.
- **Aggregation:** median over reliable voxels, mean over subjects, sign-flip test and
  Δ_min need an aggregation script. It contains no fit logic.
- **Not run, recorded as deviations:**
  - the resolution sweep for g ≥ 2;
  - Arm B ridge fits;
  - capacity sweep;
  - PLS-25;
  - leave-one-subject-out ceiling;
  - cell shuffle beyond seed 0 and beyond 224 px.

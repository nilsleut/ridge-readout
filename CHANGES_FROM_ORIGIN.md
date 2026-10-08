# Changes from origin

Every copied file was changed only in imports / `sys.path` and in path resolution.
No logic, constants, seeds, grids or defaults were changed. Line numbers refer to the
original file (`old`) and to the file in this repository (`new`). Original line endings
(CRLF or LF) are preserved. Originals and their sha256 are listed in `PROVENANCE.md`.

## Unchanged (byte-identical to the source)

`src/ridge/ridge_dual.py`, `src/ridge/fit.py`, `src/ridge/folds.py`,
`src/ridge/gridpool.py`, `src/ridge/merge.py`, `src/ridge/config_hash.py`,
`scripts/determinism_gate.py`, `scripts/determinism_probe_modal.py`.

## New

`src/ridge/paths.py` resolves `THINGS_FMRI_DIR`, `THINGS_IMAGES_DIR`, `OUTPUTS_720_DIR`,
`CKPT_DIR`, `CROSSRUN_REPO` and `RIDGE_DATA_DIR` at access time (environment, then
`.env.local`). Added in commit 4: `CKPT_V12_DIR` (the 30 v12 checkpoints) and
`BNFIX_SWEEP_CSV` (published `results/bnfix_sweep.csv`).

## Changed

| File | Line old -> new | Old | New |
|---|---|---|---|
| `src/ridge/arm_b_betas.py` (from `code/ridge/`) | 45 -> 45-47 | `DATASET = Path(r"C:/Users/nilsl/Desktop/Projekte_1/RSA/Datensatz")` | `import paths  # noqa: E402` / (blank) / `DATASET = paths.things_fmri_dir()` |
| `src/vendor/models_v2.py` (from `code/cross_species/`) | 0 -> 1-4 | — | vendor header (4 comment lines) |
| `src/vendor/models_v2.py` | 81-83 -> 85 | `_DEFAULT_WEIGHTS_DIR = Path(r"C:/Users/nilsl/Desktop/Projekte_1/learning-rules-rsa/learning_rules_outputs_bnfix/checkpoints")` | `# Resolved lazily from CKPT_DIR (src/ridge/paths.py) inside load_model().` |
| `src/vendor/models_v2.py` | 506 -> 508-509 | `weights_dir = _DEFAULT_WEIGHTS_DIR` | `from paths import ckpt_dir` / `weights_dir = ckpt_dir()` |
| `src/vendor/bn_guard.py` (from `code/`) | 0 -> 1-3 | — | vendor header (3 comment lines) |
| `scripts/pilot_fit.py` | 42 -> 42 | `sys.path.insert(0, str(HERE))` | `sys.path.insert(0, str(HERE.parent / "src" / "ridge"))` |
| `scripts/pilot_fit.py` | 52-53 -> 52-54 | `FEAT = REPO / "pilot_features"` / `ARM = REPO / "arm_b_out"` | `import paths  # noqa: E402` / `FEAT = paths.ridge_data_dir() / "pilot_features"` / `ARM = paths.ridge_data_dir() / "arm_b_out"` |
| `scripts/tolerance_measurement.py` | 49 -> 49 | `sys.path.insert(0, str(HERE))` | `sys.path.insert(0, str(HERE.parent / "src" / "ridge"))` |
| `scripts/tolerance_measurement.py` | 56-57 -> 56-58 | `FEAT = REPO / "pilot_features"` / `ARM = REPO / "arm_b_out"` | `import paths  # noqa: E402` / `FEAT = paths.ridge_data_dir() / "pilot_features"` / `ARM = paths.ridge_data_dir() / "arm_b_out"` |
| `scripts/boundary_characterisation.py` | 38 -> 38 | `sys.path.insert(0, str(Path(__file__).resolve().parent))` | `sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ridge"))` |
| `scripts/pilot_features_modal.py` | 33 -> 33-37 | `REPO = "C:/Users/nilsl/Desktop/Projekte_2/evaluation-resolution-rsa"` | `import sys as _sys` / `from pathlib import Path as _Path` / (blank) / `REPO = str(_Path(__file__).resolve().parents[1])` / `_sys.path.insert(0, REPO + "/src/vendor")` |
| `scripts/pilot_features_modal.py` | 39-40 -> 43-44 | `.add_local_dir(REPO + "/code/ridge", ...)` / `.add_local_file(REPO + "/code/cross_species/models_v2.py",` | `.add_local_dir(REPO + "/src/ridge", ...)` / `.add_local_file(REPO + "/src/vendor/models_v2.py",` |
| `scripts/extract_modal.py` | after 40 -> 41-46 | — | `import sys as _sys` / `from pathlib import Path as _Path` / (blank) / `REPO = str(_Path(__file__).resolve().parents[1])` / `_sys.path.insert(0, REPO + "/src/vendor")` / (blank) |
| `scripts/extract_modal.py` | 46 -> 52 | `"C:/Users/nilsl/Desktop/Projekte_2/evaluation-resolution-rsa/code/ridge",` | `REPO + "/src/ridge",` |
| `scripts/extract_modal.py` | 50 -> 56 | `"C:/Users/nilsl/Desktop/Projekte_2/evaluation-resolution-rsa/code/cross_species/models_v2.py",` | `REPO + "/src/vendor/models_v2.py",` |
| `tests/test_phase2_controls.py` | 36-37 -> 36-37 | `sys.path.insert(0, str(_HERE))` / `sys.path.insert(0, str(_HERE.parent / "cross_species"))` | `sys.path.insert(0, str(_HERE.parent / "src" / "ridge"))` / `sys.path.insert(0, str(_HERE.parent / "src" / "vendor"))` |
| `tests/test_loader_hardening.py` | 25 -> 25-26 | `sys.path.insert(0, str(_HERE.parent / "cross_species"))` | `sys.path.insert(0, str(_HERE.parent / "src" / "vendor"))` / `sys.path.insert(0, str(_HERE.parent / "src" / "ridge"))` |
| `tests/test_loader_hardening.py` | after 28 -> 30 | — | `import paths  # noqa: E402` |
| `tests/test_loader_hardening.py` | 30-33 -> 32 | `DEFAULT_WEIGHTS = Path(r"C:/Users/nilsl/Desktop/Projekte_1/learning-rules-rsa/" r"learning_rules_outputs_bnfix/checkpoints")` | `DEFAULT_WEIGHTS = paths.ckpt_dir()` |
| `tests/suite_count.py` | 23 -> 23 | `sys.path.insert(0, str(HERE))` | `sys.path.insert(0, str(HERE.parent / "src" / "ridge"))` |
| `tests/suite_count.py` | 53 -> 53 | `str(HERE / "determinism_gate.py")` | `str(HERE.parent / "scripts" / "determinism_gate.py")` |
| `scripts/check_csv_reproduction.py` (commit 4) | 46 -> 46-48 | `PUBLISHED = REPO_ROOT / "results" / "bnfix_sweep.csv"` | `sys.path.insert(0, str(_HERE.parent / "src" / "ridge"))` / `import paths  # noqa: E402` / `PUBLISHED = paths.bnfix_sweep_csv()` |

## Notes

- **Modal scripts derive `REPO` from `__file__`, not from `paths.py`.** Modal re-imports
  the app module inside the container, where `src/ridge` is not on `sys.path` at import
  time, so a module-level `import paths` would fail there. `paths.repo_root()` computes
  the same value. Volume names and in-container paths are unchanged.
- **Resolved at module import, not lazily**: `DATASET` in `arm_b_betas.py` (it is a
  default argument), `FEAT`/`ARM` in `pilot_fit.py` and `tolerance_measurement.py`, and
  `DEFAULT_WEIGHTS` in `test_loader_hardening.py`. `paths.py` itself reads nothing on
  import; `models_v2.load_model()` resolves `CKPT_DIR` only when `weights_dir` is None.
- `scripts/check_csv_reproduction.py` read `REPO_ROOT / "results" / "bnfix_sweep.csv"`
  (repo-relative); since commit 4 it reads `BNFIX_SWEEP_CSV`. `REPO_ROOT` is left as is
  and is no longer used for that path. `PUBLISHED` is resolved at module import.

## Added in commit 5: `src/vendor/common_up_pairs.py`

From `code/upsampling/common_up.py` (sha256 `c20c9d46ee9ae778d566932ed377677d46bfe55b15f6affdec22c320f188419e`),
verbatim: `SUBJECTS`, `N_STIM`, `TRI`, `ranks`, `stim_order`, `run_labels`, `pair_set`.
`boot_idx` and the rest of the module are not taken.

| Function | Old | New |
|---|---|---|
| `stim_order` | `open(FMRI_DIR / "stim_order_sub-01.txt")` | `open(paths.outputs_720_dir() / "stim_order_sub-01.txt")` |
| `run_labels` | `pd.read_csv(META_DIR / f"{s}_task-things_stimulus-metadata.csv")` | `pd.read_csv(paths.things_fmri_dir() / f"{s}_task-things_stimulus-metadata.csv")` |

## Changed in commit 9: `src/ridge/fit.py` (`fit_condition`)

Allowed scope only: (1) extra return values, (2) optional `folds` argument. Previously
byte-identical to the source; the source's `selected_alpha` (n_outer x n_voxel) was
already returned and is unchanged.

| Line old -> new | Old | New |
|---|---|---|
| 68 -> 68-69 | `context: str = "") -> FitResult:` | `context: str = "",` / `folds: Optional[Sequence] = None) -> FitResult:` |
| after 78 -> 80-81 | — | docstring: `folds: optional list of (train_idx, test_idx, inner_splits) replacing the default outer_inner_splits(groups, n_outer, n_inner, seed). None = default.` |
| after 102 -> 106-107 | — | `fold_of = np.full(n, -1)` / `splits = outer_inner_splits(groups, n_outer, n_inner, seed) if folds is None else folds` |
| 104 -> 109 | `for fi, (tr, te, inner) in enumerate(outer_inner_splits(groups, n_outer, n_inner, seed)):` | `for fi, (tr, te, inner) in enumerate(splits):` |
| after 133 -> 139 | — | `fold_of[te] = fi` |
| after 159 -> 166-167 | — | `pred_oof=pred_oof,` / `outer_fold_of_stimulus=fold_of,` (return values) |

Verification (`scripts/step3_verify_fit.py`, g = 1, 30 checkpoints, default folds,
`assert_low_boundary=False`): median r over 501 reliable voxels equals
`results/tolerance_all_rungs.csv` to 2.1e-14 (criterion 1e-9); r, r2, r_path, df,
selected_alpha and the four boundary cells are bit-identical (`np.array_equal`) to
`fit.py` at `bec73cb` for 30/30; tests 97/97 and 41/41.

## Added in step 5: `src/vendor/modal_src/`, `src/vendor/v12_lift.py`

- `src/vendor/modal_src/learning_rules_v12_allseeds_modal.py` (original sha256
  `bc9ceb7b5e965d237b9ccd60ae3ccdb05bb82eb1384d23ac657c7a0fe1d262a1`) and
  `learning_rules_v11_upsample_modal.py` (original sha256
  `4eb61e612e1572008462b17fa277fe665a2c31eb0c32c9e745f1ddb410807e0e`): byte-identical
  below a 5-line comment header. Never imported; read by AST.
- `src/vendor/v12_lift.py`: verbatim from `code/upsampling/common_up.py` (constants,
  `rule_key`, `tri_index`, `load_fmri_tri`, `boot_idx`, `_assigned_names`, `load_defs`,
  `CachedDS`, `namespaces`, `transform`, `load_v12_model`) and lines 54-59 of
  `code/upsampling/step3_bootstrap.py` `one()` wrapped as `boot_pairs(b, boot, runs)`.

| Name | Old | New |
|---|---|---|
| `V12`, `V11` | `CODE / "learning_rules_v1x_….py"` | `HERE / "modal_src" / "learning_rules_v1x_….py"` |
| `CODE` (sys.path for bn_guard) | `REPO / "code"` | `HERE` (src/vendor, holds the vendored bn_guard.py) |
| `load_fmri_tri` | `FMRI_DIR / f"fmri_rdm_{roi}_{s}.npy"` | `paths.outputs_720_dir() / f"fmri_rdm_{roi}_{s}.npy"` |
| `boot_idx` | `EXT / "results/noise_ceiling_v2/step3c_summary.json"` | `paths.crossrun_repo() / "results/noise_ceiling_v2/step3c_summary.json"` |
| `namespaces` | `THINGS_DIR=THINGS_DIR, FMRI_DIR=FMRI_DIR` | `THINGS_DIR=paths.things_images_dir(), FMRI_DIR=paths.outputs_720_dir()` |
| `load_v12_model` | `CKPT_DIR / f"model_weights_…"` | `paths.ckpt_v12_dir() / f"model_weights_…"` |
| `boot_pairs` | `_W["boot"][b]`, `_W["runs"]` | arguments `boot`, `runs`; returns `idx` |

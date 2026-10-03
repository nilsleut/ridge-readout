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

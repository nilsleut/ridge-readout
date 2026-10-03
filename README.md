# ridge-readout

Tests whether the resolution effect of arXiv:2608.12408 (Random − Backprop gap,
Conv1 → V1) survives a fitted ridge readout on the full feature map, across a grid
ladder g ∈ {1, 2, 4, 8}. Design and frozen degrees of freedom: `PREREGISTRATION.md`,
`config.yaml` (hash in `config.lock`), amendments in `amendments/`.

**Status:** preregistration and controls in place; the pilot is not complete. Under
config hash `206fa875cca5` `scripts/pilot_fit.py` aborts on the D1 assertion at
Random Weights (BN-calibrated) s3 g1 until the D1 diagnosis is made.

## Layout

    PREREGISTRATION.md, config.yaml, config.lock   frozen design
    amendments/      dated decisions after the preregistration
    reports/         phase reports (historical record, unedited)
    results/         measurement outputs
    src/ridge/       ridge, CV folds, grid pooling, merge, config hash, Arm B betas, paths
    src/vendor/      model loader and BatchNorm guard from evaluation-resolution-rsa
    scripts/         pilot, tolerance measurement, Modal extraction, determinism gate
    tests/           regression controls, loader hardening, suite count

## Environment

Copy `.env.example` to `.env.local` and fill in `THINGS_FMRI_DIR`, `THINGS_IMAGES_DIR`,
`OUTPUTS_720_DIR`, `CKPT_DIR`, `CROSSRUN_REPO`, `RIDGE_DATA_DIR`. `src/ridge/paths.py`
reads the environment first, then `.env.local`, and only when a path is used.

## Verify

    python -c "import sys; sys.path.insert(0,'src/ridge'); from config_hash import config_hash; print(config_hash())"
    # must print 206fa875cca5, matching config.lock
    python tests/test_phase2_controls.py     # all 97 checks passed
    python tests/test_loader_hardening.py    # all 41 checks passed

## Provenance and data

No data in this repo. `DATA_MANIFEST.txt` lists the expected files under
`RIDGE_DATA_DIR` with sizes and sha256. Origin of every file, sha256 of the originals
and the status of all deviations: `PROVENANCE.md`; code changes: `CHANGES_FROM_ORIGIN.md`.

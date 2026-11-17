"""
paths.py
========
Every machine-specific location the ridge code reads, resolved from environment
variables at ACCESS time, never at import time. Importing this module reads nothing.

Lookup order per variable:
    1. the process environment;
    2. `.env.local` at the repository root (KEY=VALUE lines, gitignored).
A variable missing from both raises with its name and purpose.

See `.env.example` for the list. Modal volume names and in-container paths
(/data, /results, /root/ridge) are not machine-specific and are not handled here.
"""

import os
from pathlib import Path

__all__ = ["repo_root", "things_fmri_dir", "things_images_dir", "outputs_720_dir",
           "ckpt_dir", "ckpt_v12_dir", "crossrun_repo", "ridge_data_dir",
           "bnfix_sweep_csv", "step5_gap_dir", "nsd_dir", "get"]

_VARS = {
    "THINGS_FMRI_DIR": "THINGS-fMRI directory with sub-XX_task-things_voxel-wise-responses.h5 "
                       "and the voxel/stimulus metadata CSVs",
    "THINGS_IMAGES_DIR": "THINGS object_images directory (one folder per concept)",
    "OUTPUTS_720_DIR": "outputs_720 directory with stim_order_sub-XX.txt and fmri_rdm_*.npy",
    "CKPT_DIR": "local checkpoint directory used as the load_model() default",
    "CKPT_V12_DIR": "directory with the 30 v12 checkpoints (6 conditions x 5 seeds)",
    "CROSSRUN_REPO": "root of the learning-rules-rsa repository",
    "RIDGE_DATA_DIR": "directory containing arm_b_out/ and pilot_features/",
    "BNFIX_SWEEP_CSV": "published results/bnfix_sweep.csv of evaluation-resolution-rsa",
    "STEP5_GAP_DIR": "output directory for the step 5 GAP features (written, not read-only)",
    "NSD_DIR": "directory for all NSD data and everything derived from it at voxel or RDM level "
               "(outside the repository; never committed)",
}


def repo_root() -> Path:
    """Root of this repository (derived from this file, not from the environment)."""
    return Path(__file__).resolve().parents[2]


def _env_local() -> dict:
    f = repo_root() / ".env.local"
    out = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get(name: str) -> Path:
    """Resolve one variable to a Path, or raise naming what is missing."""
    if name not in _VARS:
        raise KeyError(f"unknown path variable {name!r}; known: {sorted(_VARS)}")
    val = os.environ.get(name) or _env_local().get(name)
    if not val:
        raise RuntimeError(
            f"{name} is not set ({_VARS[name]}). Set it in the environment or in "
            f"{repo_root() / '.env.local'}; see .env.example.")
    return Path(val)


def things_fmri_dir() -> Path:
    return get("THINGS_FMRI_DIR")


def things_images_dir() -> Path:
    return get("THINGS_IMAGES_DIR")


def outputs_720_dir() -> Path:
    return get("OUTPUTS_720_DIR")


def ckpt_dir() -> Path:
    return get("CKPT_DIR")


def ckpt_v12_dir() -> Path:
    return get("CKPT_V12_DIR")


def crossrun_repo() -> Path:
    return get("CROSSRUN_REPO")


def ridge_data_dir() -> Path:
    return get("RIDGE_DATA_DIR")


def bnfix_sweep_csv() -> Path:
    return get("BNFIX_SWEEP_CSV")


def step5_gap_dir() -> Path:
    return get("STEP5_GAP_DIR")


def nsd_dir() -> Path:
    return get("NSD_DIR")

"""
step40_common.py
================
Shared helpers for step 40 (amendments/2026-11-16_step40-predictions.md). EXPLORATORY.

Every step 40 CSV carries config_hash (of config_step40.yaml, not of config.yaml),
git_head and seed.
"""
import os
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "ridge"))
sys.path.insert(0, str(REPO / "src" / "vendor"))

import paths                                            # noqa: E402
from config_hash import config_hash, load_config        # noqa: E402

CONFIG = REPO / "config_step40.yaml"
OUT = REPO / "results" / "step40"
KEYS = {"random": "Random", "backprop": "BP"}


def cfg():
    return load_config(CONFIG)


def cfg_hash():
    return config_hash(cfg())


def git_head():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def stamp(df):
    df["config_hash"], df["git_head"] = cfg_hash(), git_head()
    return df


def feature_dir():
    d = paths.ridge_data_dir() / "step40_features"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_reps(subject):
    """Single presentations of the 100 test images: (reps (100, 12, n_roi_voxels) with the
    session axis ordered by session number, {roi: local voxel indices}).

    Built with src/ridge/arm_b_betas.load_arm_b (same ROI masking and z-scoring over all
    trials as Arm B). Gates: stimuli, ROI voxel index and the presentation average must
    reproduce arm_b_out/<subject>_armb.npz; every image needs one presentation per session.
    Cached under RIDGE_DATA_DIR/step40_features (never in the repository).
    """
    cache = feature_dir() / f"reps_{subject}.npz"
    arm = paths.ridge_data_dir() / "arm_b_out"
    a = np.load(arm / f"{subject}_arma.npz")
    rois = {"V1": a["local_V1"], "LOC": a["local_LOC"]}
    if cache.exists():
        z = np.load(cache)
        return z["reps"], rois
    from arm_b_betas import load_arm_b
    betas, roi_idx, stimuli, reps, _ = load_arm_b(subject, dataset_dir=paths.things_fmri_dir())
    b = np.load(arm / f"{subject}_armb.npz")
    if list(stimuli) != list(b["stimuli"]) or not np.array_equal(roi_idx, b["roi_idx"]):
        raise SystemExit(f"STOP: {subject} stimuli or ROI index differ from armb.npz")
    dev = float(np.abs(betas - b["betas"]).max())
    if dev > 1e-5:
        raise SystemExit(f"STOP: {subject} presentation average differs from armb.npz by {dev:.2e}")
    meta = pd.read_csv(paths.things_fmri_dir() / f"{subject}_task-things_stimulus-metadata.csv")
    ordered = np.empty_like(reps)
    for i, s in enumerate(stimuli):
        rows = meta.index[meta.stimulus == s].to_numpy()          # same order as load_arm_b
        sess = meta.session.to_numpy()[rows]
        if len(rows) != 12 or len(set(sess)) != 12:
            raise SystemExit(f"STOP: {subject}/{s}: {len(rows)} presentations in "
                             f"{len(set(sess))} sessions; no unique session assignment")
        ordered[i] = reps[i][np.argsort(sess)]
    np.savez(cache, reps=ordered, stimuli=np.array(stimuli), avg_dev=dev)
    return ordered, rois


def test_stimuli(subject="sub-01"):
    return list(np.load(paths.ridge_data_dir() / "arm_b_out" / f"{subject}_armb.npz")["stimuli"])

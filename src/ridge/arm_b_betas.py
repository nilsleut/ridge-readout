"""
arm_b_betas.py
==============
Arm B: trial-averaged betas for the 100 twelve-repetition test stimuli, and the
split-half reliability that config.yaml uses for voxel selection.

BUILT, NOT RUN. Nothing here depends on D1 or on the selection-criterion question,
so it is prepared in parallel; it produces no fitted number.

WHAT ARM B IS FOR
-----------------
The 720 Arm A stimuli are single-presentation, so they admit no true noise ceiling.
The test set does: 100 images, 12 repetitions, identical across all three subjects,
ZERO overlap with the 720 (Phase 0/A). Arm B measures how much of the low absolute
correlation is single-trial noise -- an open question, not a known quantity. It is a
DIFFERENT STIMULUS SET, so it is not a replication of Arm A and must not be described
as one.

ORDER OF OPERATIONS matches extract_fmri_rdms_720.py:55-78 exactly, so the two arms
differ only in stimulus set and averaging:
    1. ROI mask (V4 <- hV4, LOC <- lLOC | rLOC)
    2. z-score per voxel across ALL 9,840 trials
    3. select stimuli
    4. (Arm B only) average the 12 repetitions

SPLIT-HALF RELIABILITY
----------------------
12 repetitions give C(12,6)/2 = 462 distinct balanced splits. All of them are
enumerated rather than sampled: fully deterministic, no seed, no sampling error.
Each split averages 6 reps per half, correlates the two halves across the 100
stimuli per voxel, and the mean over splits is Spearman-Brown corrected:
    r_SB = 2r / (1 + r)
because each half is a 6-rep average and the target is the 12-rep average.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

DATASET = paths.things_fmri_dir()
SUBJECTS = ["sub-01", "sub-02", "sub-03"]
N_REPS = 12
ROI_COLS = {"V1": ["V1"], "V2": ["V2"], "V3": ["V3"], "V4": ["hV4"],
            "LOC": ["lLOC", "rLOC"], "IT": ["IT"]}


def balanced_splits(n_reps: int = N_REPS):
    """All C(n,n/2)/2 distinct balanced splits, deduplicated by complement."""
    half = n_reps // 2
    seen, out = set(), []
    for c in combinations(range(n_reps), half):
        comp = tuple(sorted(set(range(n_reps)) - set(c)))
        if comp in seen:
            continue
        seen.add(c)
        out.append((list(c), list(comp)))
    return out


def load_arm_b(subject: str, dataset_dir: Path = DATASET):
    """Return (betas, voxel_index, stimuli, reps) for one subject.

    betas: (100 stimuli, n_roi_voxels) trial-averaged
    reps:  (100, 12, n_roi_voxels) per-repetition, for the split-half
    """
    import h5py

    vox = pd.read_csv(dataset_dir / f"{subject}_task-things_voxel-metadata.csv",
                      low_memory=False)
    stim = pd.read_csv(dataset_dir / f"{subject}_task-things_stimulus-metadata.csv")

    masks = {}
    for roi, cols in ROI_COLS.items():
        m = np.zeros(len(vox), bool)
        for c in cols:
            m |= vox[c].values.astype(bool)
        masks[roi] = m
    combined = np.zeros(len(vox), bool)
    for m in masks.values():
        combined |= m
    roi_idx = np.where(combined)[0]

    with h5py.File(dataset_dir / f"{subject}_task-things_voxel-wise-responses.h5", "r") as f:
        raw = f["ResponseData/block0_values"][roi_idx, :].astype(np.float32)

    # (trials, voxels), then z-score per voxel across ALL trials -- step 2, before
    # stimulus selection, exactly as Arm A does it.
    resp = raw.T
    resp = (resp - resp.mean(axis=0)) / (resp.std(axis=0) + 1e-8)

    test = stim[stim.trial_type == "test"]
    stimuli = sorted(test.stimulus.unique())
    if len(stimuli) != 100:
        raise RuntimeError(f"{subject}: expected 100 test stimuli, found {len(stimuli)}")

    reps = np.empty((len(stimuli), N_REPS, resp.shape[1]), dtype=np.float32)
    for i, s in enumerate(stimuli):
        rows = stim.index[stim.stimulus == s].to_numpy()
        if len(rows) != N_REPS:
            raise RuntimeError(f"{subject}/{s}: expected {N_REPS} reps, found {len(rows)}")
        reps[i] = resp[rows]

    betas = reps.mean(axis=1)
    return betas, roi_idx, stimuli, reps, masks


def load_arm_a(subject: str, stim_order_file: Path, dataset_dir: Path = DATASET):
    """Arm A: the 720 single-trial betas, same preprocessing order as Arm B.

    Identical to load_arm_b through step 3; there is no step 4, because each of the
    720 is a single presentation (Phase 0/A) and the averaging in
    extract_fmri_rdms_720.py:112 is a no-op for them. Returns (betas, roi_idx,
    stimuli, masks) with betas (720, n_roi_voxels).

    Sharing the ROI masking and z-scoring with Arm B is what makes the two arms
    differ ONLY in stimulus set and averaging.
    """
    import h5py

    vox = pd.read_csv(dataset_dir / f"{subject}_task-things_voxel-metadata.csv",
                      low_memory=False)
    stim = pd.read_csv(dataset_dir / f"{subject}_task-things_stimulus-metadata.csv")

    masks = {}
    for roi, cols in ROI_COLS.items():
        m = np.zeros(len(vox), bool)
        for c in cols:
            m |= vox[c].values.astype(bool)
        masks[roi] = m
    combined = np.zeros(len(vox), bool)
    for m in masks.values():
        combined |= m
    roi_idx = np.where(combined)[0]

    with h5py.File(dataset_dir / f"{subject}_task-things_voxel-wise-responses.h5", "r") as f:
        raw = f["ResponseData/block0_values"][roi_idx, :].astype(np.float32)

    resp = raw.T
    resp = (resp - resp.mean(axis=0)) / (resp.std(axis=0) + 1e-8)

    order = [l.strip() for l in open(stim_order_file) if l.strip()]
    if len(order) != 720:
        raise RuntimeError(f"expected 720 stimuli in {stim_order_file}, got {len(order)}")

    betas = np.empty((720, resp.shape[1]), dtype=np.float32)
    for i, sname in enumerate(order):
        rows = stim.index[stim.stimulus == sname].to_numpy()
        if len(rows) != 1:
            raise RuntimeError(
                f"{subject}/{sname}: expected exactly 1 presentation (Phase 0/A), "
                f"found {len(rows)}")
        betas[i] = resp[rows[0]]
    return betas, roi_idx, order, masks


def split_half_reliability(reps: np.ndarray) -> np.ndarray:
    """Spearman-Brown-corrected split-half reliability per voxel.

    reps: (n_stimuli, n_reps, n_voxels). Enumerates all 462 balanced splits.
    """
    from ridge_dual import pearson_columns

    splits = balanced_splits(reps.shape[1])
    acc = np.zeros(reps.shape[2], dtype=np.float64)
    for a, b in splits:
        ra = reps[:, a, :].mean(axis=1)
        rb = reps[:, b, :].mean(axis=1)
        acc += pearson_columns(ra, rb)
    r = acc / len(splits)
    return 2 * r / (1 + r)


if __name__ == "__main__":
    print(f"balanced splits for {N_REPS} reps: {len(balanced_splits())} "
          f"(expected {462})")
    print("This module is BUILT, NOT RUN. Loading requires the 47 GB H5 files and")
    print("produces no fitted number. D1 and the selection criterion are undecided.")

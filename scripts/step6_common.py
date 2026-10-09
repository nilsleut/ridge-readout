"""
step6_common.py
===============
Shared helpers for step 6 (amendments/2026-10-09_step6-predictions.md). EXPLORATORY.

RSA convention of paper v3 / step 5: per subject on the subject's fMRI RDM, cross-run
pairs, rho = ranks(model[P]) @ ranks(fmri[P]), mean over subjects. Stimulus bootstrap:
the paper's 1000 resamples (v12_lift.boot_idx, sha-checked), pair set recomputed per
draw (v12_lift.boot_pairs), identity draw = point estimate.

`bootstrap(items, brains)` evaluates any set of model RDMs against any set of brain RDMs
on the same draws, so all differences formed from its output are paired.
"""
import step3_common as C          # noqa: I001  (pins BLAS threads)

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(C.REPO / "src" / "vendor"))
import paths                                                      # noqa: E402
from common_up_pairs import ranks, run_labels                      # noqa: E402
from step4_std_rsa import transform                                # noqa: E402
from v12_lift import N_BOOT, TRI, rule_key, boot_idx, boot_pairs   # noqa: E402

SUBJECTS = ["sub-01", "sub-02", "sub-03"]
EPS = 1e-8
_W = {}


def gap_path(arm, px, si, rule, layer):
    return paths.step5_gap_dir() / arm / f"res{px}" / f"seed_{si}" / f"gap_{rule_key(rule)}_{layer}.npy"


def bp_path(cond, si, rule, layer):
    return paths.step5_gap_dir() / "bandpass" / cond / f"seed_{si}" / f"gap_{rule_key(rule)}_{layer}.npy"


def moment_ratio(X):
    """m = |mu|^2 / (|mu|^2 + sum_k sigma_k^2), per channel over the stimuli."""
    X = np.asarray(X, np.float64)
    mu, var = X.mean(axis=0), X.var(axis=0)
    return float(mu @ mu / (mu @ mu + var.sum()))


def normalise(X, how):
    """raw / center / scale / z per channel over the stimuli; dead channels -> 0."""
    if how == "raw":
        return np.asarray(X, np.float64)
    return transform(X, how)[0]


def roi_betas(subject, roi):
    a = np.load(paths.ridge_data_dir() / "arm_b_out" / f"{subject}_arma.npz")
    return a["betas"][:, a[f"local_{roi}"]].astype(np.float64)


def fmri_tri(roi, rezscore=False):
    """Per-subject fMRI RDM triangles (TRI order) from the ROI betas (all voxels).
    rezscore: z-score every voxel over the 720 stimuli first (instead of over all trials)."""
    from scipy.spatial.distance import pdist
    out = []
    for s in SUBJECTS:
        B = roi_betas(s, roi)
        if rezscore:
            sd = B.std(axis=0)
            B = (B - B.mean(axis=0)) / np.where(sd < EPS, 1.0, sd)
        out.append(pdist(B, "correlation"))
    return out


def _init(boot, cache, brains, item_brain):
    _W.update(boot=boot, runs=run_labels(), M=np.load(cache, mmap_mode="r"), brains=brains,
              item_brain=item_brain)


def _one(b):
    idx = boot_pairs(b, _W["boot"], _W["runs"])
    br = {k: [ranks(f[idx]) for f in F] for k, F in _W["brains"].items()}
    out = np.empty(len(_W["item_brain"]))
    for k, key in enumerate(_W["item_brain"]):
        mv = ranks(np.asarray(_W["M"][k][idx], float))
        out[k] = np.mean([mv @ x for x in br[key]])
    return b, out


def bootstrap(model_tris, item_brain, brains, cache, workers=8):
    """model_tris: list of TRI-order model RDM vectors (or callables returning them);
    item_brain: brain key per item; brains: key -> list of 3 subject TRI vectors.
    Returns D (1001, n_items); row 0 = identity draw."""
    M = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float64, shape=(len(model_tris), len(TRI[0])))
    for k, m in enumerate(model_tris):
        M[k] = m() if callable(m) else m
        if np.isnan(M[k]).any():
            raise SystemExit(f"STOP: NaN in model RDM item {k}")
    M.flush()
    del M
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(boot_idx(), cache, brains, item_brain)) as ex:
        res = dict(ex.map(_one, range(-1, N_BOOT), chunksize=8))
    return np.stack([res[b] for b in range(-1, N_BOOT)])


def ci(x):
    return float(x[0]), float(np.quantile(x[1:], .025)), float(np.quantile(x[1:], .975))

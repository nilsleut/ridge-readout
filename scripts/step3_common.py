"""
step3_common.py
===============
Shared loaders and the parallel fit runner for the step 3 scripts. EXPLORATORY,
sub-01 only: V1, 501 reliable voxels (split-half > 0.2), NATIVE 224 px, Conv1, the 30
checkpoints in pilot_features/.

Import this module FIRST: it pins BLAS to one thread per process (the benchmark's
best local mode, parallel_8x1) before numpy is imported.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import subprocess                                     # noqa: E402
import sys                                            # noqa: E402
from concurrent.futures import ProcessPoolExecutor    # noqa: E402
from pathlib import Path                              # noqa: E402

import numpy as np                                    # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (REPO / "src" / "ridge", REPO / "src" / "vendor", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                          # noqa: E402
from config_hash import config_hash                   # noqa: E402
from fit import fit_condition                         # noqa: E402
from ridge_dual import alpha_grid, pearson_columns    # noqa: E402

SUBJECT = "sub-01"
RULES = {  # pilot_features key -> label
    "random": "Random Weights", "backprop": "Backprop",
    "feedback_alignment": "Feedback Alignment", "predictive_coding": "Predictive Coding",
    "stdp": "STDP", "random_bn_calibrated": "Random Weights (BN-calibrated)",
}
LABEL_KEY = {v: k for k, v in RULES.items()}
SEED_IDX = range(5)
SEED_BASE = 20260818
OUT = REPO / "results" / "step3"
_BRAIN = {}


def git_head():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def stamp(df):
    df["config_hash"], df["git_head"] = config_hash(), git_head()
    return df


def brain():
    """(Y: 720 x 1049 V1 betas, float64; reliable: bool mask over the 1049)."""
    if not _BRAIN:
        d = paths.ridge_data_dir() / "arm_b_out"
        a, b = np.load(d / f"{SUBJECT}_arma.npz"), np.load(d / f"{SUBJECT}_armb.npz")
        v1 = a["local_V1"]
        _BRAIN["Y"] = a["betas"][:, v1].astype(np.float64)
        _BRAIN["reliable"] = b["reliability"][v1] > 0.2
    return _BRAIN["Y"], _BRAIN["reliable"]


def load_X(key, si, g):
    return np.load(paths.ridge_data_dir() / "pilot_features" / f"model_{key}_seed{si}_g{g}.npy")


def run_folds():
    """Run-grouped folds (exploratory, not preregistered). Outer: 5 folds of 2 runs
    each from rng(SEED_BASE).permutation(10) in consecutive pairs; inner: 4 folds of
    2 runs each within the 8 training runs, rng(SEED_BASE + 1000*(fold+1)). Run
    indices refer to the sorted unique run labels."""
    from common_up_pairs import run_labels
    runs = np.asarray(run_labels()[SUBJECT])
    u, cnt = np.unique(runs, return_counts=True)
    if len(u) != 10 or not (cnt == 72).all():
        raise SystemExit(f"STOP: expected 10 runs x 72 stimuli, got {dict(zip(u, cnt))}")
    perm = np.random.default_rng(SEED_BASE).permutation(10)
    out = []
    for k in range(5):
        te_runs = u[perm[2 * k: 2 * k + 2]]
        tr_runs = np.sort(np.setdiff1d(u, te_runs))
        te = np.flatnonzero(np.isin(runs, te_runs))
        tr = np.flatnonzero(~np.isin(runs, te_runs))
        p = np.random.default_rng(SEED_BASE + 1000 * (k + 1)).permutation(8)
        inner = []
        for j in range(4):
            va_runs = tr_runs[p[2 * j: 2 * j + 2]]
            inner.append((tr[~np.isin(runs[tr], va_runs)], tr[np.isin(runs[tr], va_runs)]))
        out.append((tr, te, inner))
    return out


def fit_one(job):
    """job = (key, seed_idx, g, scheme) -> summary dict (+ reliable-voxel OOF preds)."""
    from lower_edge_gate import hits_from_fit
    key, si, g, scheme = job
    Y, rel = brain()
    kw = {}
    if scheme == "run":
        kw = dict(folds=run_folds(), n_outer=5, n_inner=4)
    alphas = alpha_grid()
    res = fit_condition(load_X(key, si, g).astype(np.float64), Y, np.arange(len(Y)),
                        alphas=alphas, reliable=rel, assert_low_boundary=False,
                        context=f"{RULES[key]} s{si} g{g} {scheme}", **kw)
    fo = res["outer_fold_of_stimulus"]
    per_fold = np.stack([pearson_columns(res["pred_oof"][fo == f], Y[fo == f])
                         for f in range(res["n_outer"])])          # (n_outer, n_voxel)
    sel = res["selected_alpha"][:, rel]
    return dict(
        rule=RULES[key], seed_idx=si, grid=g, fold_scheme=scheme, d=res["d"],
        n_outer=res["n_outer"], n_reliable=int(rel.sum()),
        r_pooled=float(np.median(res["r"][rel])),
        r_perfold=float(np.median(per_fold.mean(axis=0)[rel])),
        hi_frac=float(np.isclose(sel, alphas[-1], rtol=1e-12).mean()),
        lo_frac=float(np.isclose(sel, alphas[0], rtol=1e-12).mean()),
        hits=hits_from_fit(res, rel, condition=RULES[key], seed=si, g=g, fold_scheme=scheme),
        pred_rel=res["pred_oof"][:, rel], fold_of=fo, r_vox=res["r"],
        selected_alpha=res["selected_alpha"])


def pmap(fn, jobs, workers=8):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs))


def run_gate(results, folds_by_scheme=None):
    """Apply the mandatory lower-edge gate to every logged hit."""
    import pandas as pd
    from lower_edge_gate import gate
    hits = [h for r in results for h in r["hits"]]
    if not hits:
        return pd.DataFrame(columns=["condition", "seed", "g", "fold_scheme", "voxel",
                                     "outer_fold", "relative_gain", "classification"])
    Y, _ = brain()
    return gate(hits, load_X=lambda h: load_X(LABEL_KEY[h["condition"]], h["seed"], h["g"]),
                Y=Y, folds=folds_by_scheme or {})

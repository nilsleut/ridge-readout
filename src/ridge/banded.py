"""
banded.py
=========
Banded (multi-penalty) dual ridge for step 40 (amendments/2026-11-16_step40-predictions.md).

One penalty per feature group. With per-feature standardisation from the training split
only (as in ridge_dual.standardize), the dual form is

    K' = sum_g K_g / a_g,          K_g = X_g X_g^T
    Yhat_test = K'_test (K' + I)^-1 (Y_train - mean) + mean

which for a single group reduces exactly to ridge_dual.DualRidge with alpha = a_g. Factoring
out the last group's penalty, K' = M / a_N with M = K_N + sum_{g<N} (a_N / a_g) K_g, so

    Yhat_test = M_test (M + a_N I)^-1 (Y_train - mean) + mean.

M depends only on the penalty RATIOS, so one eigendecomposition per ratio vector serves
every a_N with that ratio; this is the same "one eigh, all alphas" device as ridge_dual.

Single-group fits (bank alone, network alone) use the same code with one group, so a
difference between joint and single-group scores cannot come from code differences.

Nested CV as in fit.fit_condition: outer folds from folds.outer_inner_splits (8 x 5, seed
20260818, inner seed + 1000 * (fold + 1)); the penalty vector is chosen per voxel by the
minimum mean inner-CV MSE; out-of-fold predictions are pooled over the outer folds and
scored with Pearson r. Low-edge selections at reliable voxels are not an abort: each one
is classified by the plateau diagnostic of the D1 gate (scripts/lower_edge_gate.py),
extended to one axis of the penalty grid: the inner-CV MSE is recomputed with that
group's penalty extended to 1e-7, 1e-8, 1e-9 (the other group's penalties over its full
grid) and the relative gain (MSE(selected) - min MSE(extended)) / MSE(selected) is
< 1e-3 for "plateau" and >= 1e-3 for "invalid".
"""

from itertools import product
from typing import Dict, List, Optional, Sequence

import numpy as np

from folds import outer_inner_splits
from ridge_dual import mse_columns, pearson_columns, standardize

__all__ = ["banded_predict", "fit_banded", "DIAGNOSTIC", "GAIN_THRESHOLD"]

DIAGNOSTIC = (1e-7, 1e-8, 1e-9)
GAIN_THRESHOLD = 1e-3


def _kernels(groups_tr: List[np.ndarray], groups_te: List[np.ndarray]):
    """Standardise every group with its training statistics; return train and test kernels."""
    Ktr, Kte = [], []
    for Xtr, Xte in zip(groups_tr, groups_te):
        a, b = standardize(np.asarray(Xtr, np.float64), np.asarray(Xte, np.float64))
        Ktr.append(a @ a.T)
        Kte.append(b @ a.T)
    return Ktr, Kte


def banded_predict(Ktr: List[np.ndarray], Kte: List[np.ndarray], Y: np.ndarray,
                   pairs: np.ndarray, cols: Optional[np.ndarray] = None) -> np.ndarray:
    """Predictions for every penalty vector in `pairs` (P, n_groups).

    Returns (P, n_test, n_voxel). `cols` restricts the voxels (for the edge diagnostic).
    """
    Y = np.asarray(Y, np.float64)
    if cols is not None:
        Y = Y[:, cols]
    mean = Y.mean(axis=0, keepdims=True)
    Yc = Y - mean
    pairs = np.asarray(pairs, np.float64)
    out = np.empty((len(pairs), Kte[0].shape[0], Y.shape[1]))
    last = pairs[:, -1]
    ratios = last[:, None] / pairs[:, :-1] if pairs.shape[1] > 1 else np.zeros((len(pairs), 0))
    keys = [tuple(np.round(np.log10(r), 9)) for r in ratios]
    for key in dict.fromkeys(keys):
        rows = [i for i, k in enumerate(keys) if k == key]
        rho = ratios[rows[0]]
        M = Ktr[-1] + sum(r * K for r, K in zip(rho, Ktr[:-1]))
        Mt = Kte[-1] + sum(r * K for r, K in zip(rho, Kte[:-1]))
        lam, U = np.linalg.eigh(M)
        lam = np.clip(lam, 0.0, None)
        MtU = Mt @ U
        UtY = U.T @ Yc
        for i in rows:
            out[i] = (MtU * (1.0 / (lam + last[i]))) @ UtY + mean
    return out


def _grid_pairs(grids: Sequence[np.ndarray]) -> np.ndarray:
    return np.array(list(product(*[np.asarray(g, np.float64) for g in grids])))


def fit_banded(groups: List[np.ndarray], Y: np.ndarray, *, grids: Sequence[np.ndarray],
               n_outer: int = 8, n_inner: int = 5, seed: int = 20260818,
               reliable: Optional[np.ndarray] = None, gate: bool = True) -> Dict:
    """Nested-CV banded ridge.

    groups: list of (n, d_g) feature arrays (bank first, network last by convention).
    grids:  one penalty grid per group (same order). A grid of length 1 fixes that penalty.
    reliable: bool mask over voxels for the low-edge gate and for reporting.
    Returns pooled out-of-fold predictions, per-voxel r and MSE skill, selected penalties
    (n_outer, n_voxel) per group, and the classified low-edge hits.
    """
    Y = np.asarray(Y, np.float64)
    n, v = Y.shape
    if any(len(G) != n for G in groups):
        raise ValueError("every feature group needs one row per stimulus")
    grids = [np.asarray(g, np.float64) for g in grids]
    if len(grids) != len(groups):
        raise ValueError("one penalty grid per feature group")
    pairs = _grid_pairs(grids)
    rel = np.ones(v, bool) if reliable is None else np.asarray(reliable, bool)

    pred_oof = np.full((n, v), np.nan)
    base_oof = np.full((n, v), np.nan)
    fold_of = np.full(n, -1)
    sel = np.zeros((len(groups), n_outer, v))
    hits = []

    for fi, (tr, te, inner) in enumerate(outer_inner_splits(np.arange(n), n_outer, n_inner, seed)):
        crit = np.zeros((len(pairs), v))
        cache = []
        for itr, iva in inner:
            Ktr, Kva = _kernels([G[itr] for G in groups], [G[iva] for G in groups])
            P = banded_predict(Ktr, Kva, Y[itr], pairs)
            crit += np.stack([mse_columns(p, Y[iva]) for p in P])
            cache.append((itr, iva, Ktr, Kva))
        crit /= len(inner)
        best = np.argmin(crit, axis=0)                      # (v,)
        for gi in range(len(groups)):
            sel[gi, fi] = pairs[best, gi]

        # low-edge selections at reliable voxels, per group with a real grid
        if gate:
            for gi, g in enumerate(grids):
                if len(g) < 2:
                    continue
                lo = np.flatnonzero(np.isclose(pairs[best, gi], g[0], rtol=1e-12) & rel)
                if not len(lo):
                    continue
                ext = [np.concatenate([g, DIAGNOSTIC]) if k == gi else grids[k]
                       for k in range(len(grids))]
                ext_pairs = _grid_pairs(ext)
                e = np.zeros((len(ext_pairs), len(lo)))
                for itr, iva, Ktr, Kva in cache:
                    P = banded_predict(Ktr, Kva, Y[itr], ext_pairs, cols=lo)
                    e += np.stack([mse_columns(p, Y[iva][:, lo]) for p in P])
                e /= len(inner)
                m_sel = crit[best[lo], lo]
                gain = (m_sel - e.min(axis=0)) / m_sel
                for vox, gn in zip(lo, gain):
                    hits.append(dict(outer_fold=fi, voxel=int(vox), group=gi, gain=float(gn),
                                     classification="plateau" if gn < GAIN_THRESHOLD else "invalid"))

        # refit on the outer training set at every penalty vector, keep each voxel's own
        Ktr, Kte = _kernels([G[tr] for G in groups], [G[te] for G in groups])
        used = np.unique(best)
        P = banded_predict(Ktr, Kte, Y[tr], pairs[used])
        pos = {u: k for k, u in enumerate(used)}
        pred_oof[te] = np.stack([P[pos[b], :, j] for j, b in enumerate(best)], axis=1)
        base_oof[te] = Y[tr].mean(axis=0)
        fold_of[te] = fi

    if np.isnan(pred_oof).any():
        raise RuntimeError("some stimuli were never in a test fold")
    r = pearson_columns(pred_oof, Y)
    skill = 1.0 - ((Y - pred_oof) ** 2).sum(axis=0) / ((Y - base_oof) ** 2).sum(axis=0)
    return dict(r=r, mse_skill=skill, pred_oof=pred_oof, selected=sel, hits=hits,
                outer_fold_of_stimulus=fold_of, n_outer=n_outer, pairs=pairs)

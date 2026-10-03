"""
lower_edge_gate.py
==================
Mandatory post-run gate for low-edge alpha selections at reliable voxels
(amendments/2026-10-03_d1-plateau-gate.md). It replaces D1's abort; it never changes
a score.

For each logged hit (condition, seed, g, voxel, outer fold) the mean inner-CV MSE
curve of that voxel in that outer fold is recomputed on the standard grid plus the
diagnostic points 1e-7, 1e-8, 1e-9 (the step 2D procedure, scripts/step2_lower_edge.py,
without the points below 1e-9), and

    relative gain = (MSE(1e-6) - MSE_min,diagnostic) / MSE(1e-6)
    gain <  1e-3 -> "Plateau" (valid)
    gain >= 1e-3 -> "invalid" (the whole cell is excluded and reported)

Hits come from fit_condition(..., assert_low_boundary=False) output via
`hits_from_fit`. Folds must be the ones the fit used: default = outer_inner_splits(
arange(n), 8, 5, 20260818), or the same `folds` list that was passed to fit_condition.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "ridge"))

from folds import outer_inner_splits                                   # noqa: E402
from ridge_dual import DualRidge, alpha_grid, standardize, mse_columns  # noqa: E402

DIAGNOSTIC = [1e-7, 1e-8, 1e-9]
GAIN_THRESHOLD = 1e-3
SEED, N_OUTER, N_INNER = 20260818, 8, 5


def hits_from_fit(res, reliable, **labels):
    """Low-edge selections at reliable voxels from a fit_condition result.
    labels: condition, seed, g, fold_scheme, ... copied into every hit."""
    sel = np.asarray(res["selected_alpha"])                 # (n_outer, n_voxel)
    lo = np.isclose(sel, np.asarray(res["alphas"])[0], rtol=1e-12) & np.asarray(reliable)[None, :]
    return [dict(labels, voxel=int(v), outer_fold=int(f)) for f, v in zip(*np.nonzero(lo))]


def default_folds(n):
    return list(outer_inner_splits(np.arange(n), N_OUTER, N_INNER, SEED))


def _inner_mse(X, y, inner, alphas):
    out = np.zeros(len(alphas))
    for itr, iva in inner:
        Xi, Xv = standardize(X[itr], X[iva])
        preds = DualRidge(alphas).fit(Xi, y[itr]).predict_path(Xv)
        out += np.array([mse_columns(p, y[iva])[0] for p in preds])
    return out / len(inner)


def gate(hits, *, load_X, Y, folds=None):
    """Classify every hit. Returns a DataFrame with one row per hit.

    load_X(hit) -> feature matrix of that hit's condition/seed/g.
    Y: (n_stimuli, n_voxel) responses, voxel indices as in the hit.
    folds: dict fold_scheme -> list of (train, test, inner) splits; a hit without
           'fold_scheme', or a scheme not in the dict, uses default_folds.
    """
    folds = folds or {}
    std = alpha_grid()
    grid = np.unique(np.concatenate([std, DIAGNOSTIC]))
    i_edge = int(np.flatnonzero(np.isclose(grid, std[0], rtol=1e-12))[0])
    cache, rows = {}, []
    for h in hits:
        scheme = h.get("fold_scheme")
        if scheme not in cache:
            cache[scheme] = folds.get(scheme) or default_folds(Y.shape[0])
        _, _, inner = cache[scheme][h["outer_fold"]]
        X = np.asarray(load_X(h), dtype=np.float64)
        cur = _inner_mse(X, np.asarray(Y, float)[:, [h["voxel"]]], inner, grid)
        k = int(np.argmin(cur))
        gain = float((cur[i_edge] - cur[k]) / cur[i_edge])
        rows.append(dict(h, alpha_min_diagnostic=float(grid[k]), mse_at_1e_6=float(cur[i_edge]),
                         mse_min_diagnostic=float(cur[k]), relative_gain=gain,
                         classification="Plateau" if gain < GAIN_THRESHOLD else "invalid"))
    return pd.DataFrame(rows)

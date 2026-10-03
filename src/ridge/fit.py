"""
fit.py
======
The fitting driver: nested CV, alpha selection, and the full r(df) path.

STRUCTURE
---------
For each outer fold:
    1. inner CV on the outer TRAINING set only -> r per (alpha, voxel), averaged over
       inner folds. The outer test fold is never touched during selection.
    2. select alpha per voxel by MINIMUM HELD-OUT MSE (D2), aborting if any
       RELIABLE voxel selects the LOW grid edge (D1).
    3. refit on the full outer training set and score the outer test fold, keeping
       BOTH the per-voxel scalar at the selected alpha AND the whole path.

Per-feature standardization is recomputed inside every split from that split's training
rows only -- never once globally, which would leak test-fold scale into the fit.

WHAT IS REPORTED
----------------
The path, not only the maximum: r at every alpha, with df(alpha) alongside. Two
conditions can share a peak and have completely different curves, and if Random reaches
the same predictivity only at three times the effective dimensionality that is a real
difference a scalar hides.

SELECTION AND REPORTING CRITERIA DIFFER, DELIBERATELY (D2).
Selection is on held-out MSE; reporting is Pearson r, for Brain-Score comparability.
Selecting on a scale-invariant quantity is degenerate -- alpha controls the scale of
the prediction far more than its direction, and the measured r-optimum spans the whole
1e-6..1e12 probe grid. CONSEQUENCE, which must be stated and not buried: the selected
alpha is therefore NOT the r-maximising alpha, so reported r sits slightly below the
achievable maximum -- equally for every condition, so contrasts are unaffected.

THE UNIT OF INFERENCE IS THE SEED, NOT THE FOLD.
Folds are averaged into a point estimate here and carry no confidence interval; the
seed-level spread is formed downstream. Anything that computes an interval across folds
is misusing this output.
"""

from typing import Dict, Optional, Sequence

import numpy as np

from folds import outer_inner_splits
from ridge_dual import (DualRidge, alpha_grid, standardize, pearson_columns,
                        r2_columns, mse_columns, select_alpha_per_voxel)

__all__ = ["fit_condition", "FitResult", "CELLS"]

# The four boundary cells, reported for every condition (D1).
CELLS = ("lo_rel", "hi_rel", "lo_unrel", "hi_unrel")


class FitResult(dict):
    """Plain dict with the keys the merge step expects; a class only for clarity."""


def fit_condition(X: np.ndarray,
                  Y: np.ndarray,
                  groups: Sequence,
                  *,
                  n_outer: int = 8,
                  n_inner: int = 5,
                  seed: int = 20260818,
                  alphas: Optional[np.ndarray] = None,
                  reliable: Optional[np.ndarray] = None,
                  assert_low_boundary: bool = True,
                  context: str = "",
                  folds: Optional[Sequence] = None) -> FitResult:
    """Nested-CV dual ridge for one condition.

    X: (n_stimuli, d) features, already pooled to the grid.
    Y: (n_stimuli, n_voxels) responses.
    groups: concept label per stimulus (720 distinct today; see folds.py).
    reliable: bool mask over voxels from the split-half threshold. D1's assertion is
        evaluated over these voxels only; None = all treated as reliable.
    assert_low_boundary: D1's hard per-voxel low-edge assertion. Set False only
        where the assertion has been MEASURED not to apply; every such use must be
        justified in writing.
    folds: optional list of (train_idx, test_idx, inner_splits) replacing the
        default outer_inner_splits(groups, n_outer, n_inner, seed). None = default.

    Returns per-voxel test r and R^2 at the selected alpha, plus the full path
    averaged over outer folds, plus df(alpha).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows, Y has {Y.shape[0]}")
    if X.shape[0] != len(groups):
        raise ValueError("groups must have one entry per stimulus")
    alphas = alpha_grid() if alphas is None else np.asarray(alphas, float)

    n, v = Y.shape
    n_a = len(alphas)

    r_test = np.full((n, v), np.nan)        # out-of-fold prediction quality per stimulus
    pred_oof = np.full((n, v), np.nan)
    path_sum = np.zeros((n_a, v))
    df_sum = np.zeros(n_a)
    sel_alpha = np.zeros((n_outer, v))
    # All four cells recorded every fold, always, as diagnostics with no abort
    # function (D1). lo_rel is the only one the assertion acts on.
    cells = {k: np.zeros(n_outer) for k in CELLS}
    n_folds_done = 0
    fold_of = np.full(n, -1)
    splits = outer_inner_splits(groups, n_outer, n_inner, seed) if folds is None else folds

    for fi, (tr, te, inner) in enumerate(splits):
        # ── inner CV: alpha selection, on the training set only ──────────────
        # D2: SELECT on MSE (negated, so higher-is-better), not on Pearson r.
        # Pearson r is scale-invariant and alpha controls scale, which makes
        # r-selection degenerate. Reporting still uses r -- see the return block.
        inner_crit = np.zeros((n_a, v))
        for itr, iva in inner:
            Xi, Xv = standardize(X[itr], X[iva])
            m = DualRidge(alphas).fit(Xi, Y[itr])
            preds = m.predict_path(Xv)
            inner_crit += -np.stack([mse_columns(p, Y[iva]) for p in preds])
        inner_crit /= len(inner)

        idx, a_sel, st = select_alpha_per_voxel(
            inner_crit, alphas, reliable=reliable,
            assert_low_boundary=assert_low_boundary,
            context=f"{context} outer-fold {fi}")
        sel_alpha[fi] = a_sel
        for k in CELLS:
            cells[k][fi] = st[k]

        # ── refit on the full outer training set, score the held-out fold ────
        Xtr, Xte = standardize(X[tr], X[te])
        m = DualRidge(alphas).fit(Xtr, Y[tr])
        preds = m.predict_path(Xte)                     # (n_a, n_te, v)
        path_sum += np.stack([pearson_columns(p, Y[te]) for p in preds])
        df_sum += m.df_

        # Per-voxel prediction at that voxel's own selected alpha.
        pred_oof[te] = preds[idx, :, np.arange(v)].T
        fold_of[te] = fi
        n_folds_done += 1

    if n_folds_done != n_outer:
        raise RuntimeError(f"expected {n_outer} outer folds, completed {n_folds_done}")
    if np.isnan(pred_oof).any():
        raise RuntimeError("some stimuli were never in a test fold; folds are not a partition")

    return FitResult(
        # Scored once over all out-of-fold predictions, which is the standard
        # cross-validated estimate and avoids averaging correlations across folds.
        r=pearson_columns(pred_oof, Y),
        r2=r2_columns(pred_oof, Y),
        r_path=path_sum / n_outer,          # (n_alpha, n_voxel)
        df=df_sum / n_outer,                # (n_alpha,)
        alphas=alphas,
        selected_alpha=sel_alpha,           # (n_outer, n_voxel)
        # Reported for every condition, always; diagnostics with no abort function.
        **{k: cells[k] for k in CELLS},
        boundary_asserted=assert_low_boundary,
        selection_criterion="mse",
        reporting_metric="pearson_r",
        n_outer=n_outer,
        n_inner=n_inner,
        n_stimuli=n,
        n_voxels=v,
        d=X.shape[1],
        pred_oof=pred_oof,                  # (n_stimuli, n_voxel)
        outer_fold_of_stimulus=fold_of,     # (n_stimuli,)
    )

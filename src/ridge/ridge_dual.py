"""
ridge_dual.py
=============
Dual-form ridge with the full regularization path.

WHY DUAL
--------
d >> n everywhere: Conv1 native width runs 8,192 (32 px) to 401,408 (224 px) against
n = 720. Even on the grid ladder, g = 8 gives d = 2,048 > n. So form K = X X^T (n x n),
eigendecompose ONCE per fold, and every alpha and every voxel is then nearly free. No
d x d solve anywhere.

    K = U diag(lam) U^T
    (K + aI)^-1 = U diag(1/(lam + a)) U^T
    Yhat_test = K_test U diag(1/(lam + a)) U^T Y_train

STANDARDIZATION
---------------
Per feature, statistics FROM THE TRAINING FOLD ONLY. Ridge is not scale-invariant and
random vs backprop networks have systematically different activation scales and
kurtosis; without this the measurement is of scale, not representation. Test features
are transformed with the training fold's statistics, never their own.

THE PATH IS THE MEASUREMENT
---------------------------
r(df) across the whole alpha path, with

    df(a) = sum_i lam_i / (lam_i + a)

where lam are the eigenvalues of K, i.e. the squared singular values of X -- so this is
the sum_i s_i^2/(s_i^2 + a) of the design. Two conditions can share a maximum and have
completely different curves; if Random reaches the same predictivity only at three
times the effective dimensionality, a scalar hides it.

BOUNDARY ABORT
--------------
If the selected alpha sits at either end of the grid, the grid is too narrow and every
number for that condition is invalid. Abort, do not warn. See the recorded risk in
PREREGISTRATION.md section 7: at g = 1 the LOW boundary is reachable for a legitimate
reason (d = 32 against ~630 training stimuli), and widening the grid mid-run is a spec
change requiring sign-off.
"""

from typing import Optional, Tuple

import numpy as np

__all__ = ["alpha_grid", "standardize", "DualRidge", "effective_df",
           "AlphaBoundaryError", "pearson_columns", "r2_columns", "mse_columns",
           "boundary_sides", "select_alpha_per_voxel"]


class AlphaBoundaryError(RuntimeError):
    """Selected alpha sat at a grid boundary -- the grid is too narrow."""


def alpha_grid(lo: float = 1e-6, hi: float = 1e8, n: int = 29) -> np.ndarray:
    """Log-spaced alpha grid. Identical across all conditions (config.yaml).

    D3 (adopted 2026-08-20): low edge 1e-2 -> 1e-6, 25 -> 29 points. The measured
    optimum under MSE selection spans 1e-6 .. 5.0e2, so the old low edge sat ABOVE
    part of the true optimum range. 29 points over 14 decades keeps the previous
    density of 2 per decade.
    """
    return np.logspace(np.log10(lo), np.log10(hi), n)


def standardize(train: np.ndarray, *others: np.ndarray, eps: float = 1e-8):
    """Per-feature standardization using TRAIN statistics only.

    Zero-variance features (a dead unit at this resolution, say) would divide by ~0;
    they are set to exactly zero in every split instead, which is what a constant
    feature contributes to a centred ridge fit anyway.
    """
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    dead = sd < eps
    sd = np.where(dead, 1.0, sd)

    def tf(a):
        out = (a - mu) / sd
        if dead.any():
            out[:, dead[0]] = 0.0
        return out

    return (tf(train), *(tf(o) for o in others))


def effective_df(lam: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """df(a) = sum_i lam_i / (lam_i + a), one value per alpha."""
    lam = np.clip(lam, 0.0, None)
    return (lam[None, :] / (lam[None, :] + alphas[:, None])).sum(axis=1)


def r2_columns(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Column-wise R^2 = 1 - SS_res/SS_tot, against the TEST mean.

    Needed alongside Pearson r because r is SCALE-INVARIANT and R^2 is not. As
    alpha -> infinity the prediction collapses onto the training mean, so R^2 -> 0,
    but r does NOT: the surviving deviation is vanishingly small yet keeps its
    direction, and r measures only direction. See the note in
    `test_phase2_controls.py::t_ridge`. Anyone testing "over-regularization destroys
    predictivity" must test it on R^2, not on r.
    """
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    out = np.zeros(true.shape[1], dtype=np.float64)
    ok = ss_tot > 0
    out[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    return out


def pearson_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Pearson r between two (n, v) arrays.

    Returns 0 for columns where either side is constant -- a constant prediction has
    no correlation to report, and NaN there would silently poison a median over voxels.
    """
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    na = np.sqrt((a * a).sum(axis=0))
    nb = np.sqrt((b * b).sum(axis=0))
    ok = (na > 0) & (nb > 0)
    out = np.zeros(a.shape[1], dtype=np.float64)
    out[ok] = (a[:, ok] * b[:, ok]).sum(axis=0) / (na[ok] * nb[ok])
    return out


class DualRidge:
    """One eigendecomposition, all alphas, all voxels.

    fit(X_train, Y_train) -> predict_path(X_test) -> (n_alpha, n_test, n_voxel)
    """

    def __init__(self, alphas: Optional[np.ndarray] = None):
        self.alphas = alpha_grid() if alphas is None else np.asarray(alphas, float)
        if self.alphas.ndim != 1 or len(self.alphas) < 2:
            raise ValueError("alphas must be a 1-D grid with at least two points")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "DualRidge":
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows, Y has {Y.shape[0]}")
        self.X_ = X
        self.y_mean_ = Y.mean(axis=0, keepdims=True)
        Yc = Y - self.y_mean_

        K = X @ X.T
        # eigh: K is symmetric PSD by construction. Clip tiny negative eigenvalues
        # that come from round-off, so 1/(lam+a) cannot blow up.
        lam, U = np.linalg.eigh(K)
        self.lam_ = np.clip(lam, 0.0, None)
        self.U_ = U
        self.UtY_ = U.T @ Yc            # (n, v), computed once
        self.df_ = effective_df(self.lam_, self.alphas)
        return self

    def predict_path(self, X_test: np.ndarray) -> np.ndarray:
        """(n_alpha, n_test, n_voxel) predictions, one slab per alpha."""
        Kt = np.asarray(X_test, dtype=np.float64) @ self.X_.T   # (m, n)
        KtU = Kt @ self.U_                                      # (m, n)
        out = np.empty((len(self.alphas), Kt.shape[0], self.UtY_.shape[1]))
        for i, a in enumerate(self.alphas):
            out[i] = (KtU * (1.0 / (self.lam_ + a))) @ self.UtY_ + self.y_mean_
        return out

    def r_path(self, X_test: np.ndarray, Y_test: np.ndarray) -> np.ndarray:
        """(n_alpha, n_voxel) Pearson r on held-out data."""
        preds = self.predict_path(X_test)
        Y_test = np.asarray(Y_test, dtype=np.float64)
        return np.stack([pearson_columns(p, Y_test) for p in preds])


# ─────────────────────────────────────────────────────────────────────────────
# Alpha selection
#
# OPERATIVE MODE IS "per_voxel", matching config.yaml schema_version 1.
#
# The "population_fraction" mode below is PREPARED BUT NOT ADOPTED. It, and the
# MSE selection criterion in `inner_criterion`, are proposals pending sign-off
# (PREREGISTRATION.md section 9, D1/D2). They exist so the boundary rates can be
# MEASURED under each option; nothing in the fitting path may select them until the
# frozen degrees of freedom are changed by the author.
# ─────────────────────────────────────────────────────────────────────────────

def mse_columns(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Column-wise mean squared error. The SELECTION criterion (D2).

    Ridge minimises squared error, so MSE is the matched criterion. Pearson r is
    scale-invariant and alpha controls scale far more than direction, which makes
    r-selection degenerate: the measured r-optimum spans the entire 1e-6..1e12
    probe grid, so argmax returns an endpoint rather than an optimum.

    argmin MSE == argmax R^2 exactly on a fixed validation fold, since
    R^2 = 1 - n*MSE/SS_tot with SS_tot fixed.
    """
    return ((np.asarray(true, float) - np.asarray(pred, float)) ** 2).mean(axis=0)


def boundary_sides(idx: np.ndarray, n_alpha: int) -> Tuple[np.ndarray, np.ndarray]:
    """Masks for low-boundary and high-boundary selections.

    The two sides mean OPPOSITE things and are never summed:

      HIGH (alpha_max chosen): for a voxel with no signal this is the CORRECT
        answer -- maximal regularisation is right when there is nothing to fit.
        NEVER an alarm.
      LOW (alpha_min chosen) in a RELIABLE voxel: that voxel's optimum lies below
        the grid. The genuine alarm, and the only one that aborts (D1).
    """
    return idx == 0, idx == n_alpha - 1


def select_alpha_per_voxel(criterion: np.ndarray, alphas: np.ndarray,
                           reliable: Optional[np.ndarray] = None,
                           assert_low_boundary: bool = True,
                           context: str = "") -> Tuple[np.ndarray, np.ndarray, dict]:
    """Pick alpha per voxel and enforce the D1 boundary assertion.

    criterion: (n_alpha, n_voxel), HIGHER IS BETTER, averaged over inner folds.
               Pass -MSE (see `mse_columns`) -- that is the adopted criterion.
    reliable:  bool mask over voxels; None = all treated as reliable.
    assert_low_boundary: set False ONLY where the assertion has been measured not
               to apply. Every such use must be justified in writing.

    D1 (adopted 2026-08-20). Hard per-voxel assertion, LOW edge only, RELIABLE
    voxels only, ONE such voxel aborts. States exactly its intent -- no reliable
    voxel's optimum lies outside the grid -- with no threshold to justify.

    Returns (idx, alpha per voxel, stats). `stats` always carries all four cells
    (low/high x reliable/unreliable) as diagnostics with no abort function.
    """
    if criterion.shape[0] != len(alphas):
        raise ValueError("criterion first axis must match the alpha grid")
    idx = np.argmax(criterion, axis=0)

    mask = np.ones(len(idx), dtype=bool) if reliable is None else np.asarray(reliable, bool)
    if mask.shape != idx.shape:
        raise ValueError(f"reliable mask has shape {mask.shape}, expected {idx.shape}")

    lo_m, hi_m = boundary_sides(idx, len(alphas))

    def _fr(sel):
        return float(sel.mean()) if len(sel) else float("nan")

    stats = {
        "n_voxels": int(len(idx)),
        "n_reliable": int(mask.sum()),
        "lo_rel": _fr(lo_m[mask]),
        "hi_rel": _fr(hi_m[mask]),
        "lo_unrel": _fr(lo_m[~mask]),
        "hi_unrel": _fr(hi_m[~mask]),
        "n_lo_rel": int(lo_m[mask].sum()),
        "alpha_min": float(alphas[0]),
        "alpha_max": float(alphas[-1]),
        "median_selected_alpha": float(np.median(alphas[idx])),
    }

    if not assert_low_boundary:
        return idx, alphas[idx], stats

    if not mask.any():
        # Refuse to pass vacuously: an empty reliable set means the assertion
        # checked nothing, which is how the original BatchNorm defect hid.
        raise AlphaBoundaryError(
            f"{context or '(unnamed condition)'}: the reliable-voxel mask is empty, "
            "so the low-boundary assertion would be vacuous. Check the "
            "voxel-selection threshold before trusting any number from this "
            "condition."
        )

    n_bad = stats["n_lo_rel"]
    if n_bad:
        raise AlphaBoundaryError(
            f"{context or '(unnamed condition)'}: {n_bad} of {int(mask.sum())} "
            f"RELIABLE voxels selected the LOW grid edge (alpha={alphas[0]:.3g}). "
            "Their optimum lies below the grid, so every number for this condition "
            "is invalid. "
            "The response is to WIDEN THE GRID under a fresh sign-off and re-run. "
            "Never to relax this assertion, and never mid-run. "
            "(High-edge selections are not an alarm and do not reach here: "
            f"hi_rel={stats['hi_rel']:.3f}, hi_unrel={stats['hi_unrel']:.3f}.)"
        )
    return idx, alphas[idx], stats

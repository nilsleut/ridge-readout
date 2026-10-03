"""
tolerance_measurement.py
========================
MEASUREMENT ONLY. Nothing here is adopted, and the operative selection rule
(ridge_dual.select_alpha_per_voxel, min-MSE) is not touched.

Measures four candidate alpha-selection rules across all three grid rungs, so the
tolerance form can be chosen on evidence rather than on plausibility.

THE RULES
---------
  min-MSE   (current, baseline)  argmin of the inner-CV MSE.
  1-SE      largest alpha whose CV MSE <= min + SE(min), where SE is the
            fold-to-fold standard error of the MSE across inner folds at the
            argmin alpha. Textbook (Hastie/Tibshirani/Friedman).
  tol-1%    largest alpha whose CV MSE <= 1.01 * min.
  tol-0.1%  largest alpha whose CV MSE <= 1.001 * min.

All three tolerance rules take the LARGEST qualifying alpha, which is what makes
them plateau-aware: on a flat region they choose the most-regularised end of it
rather than resolving numerical noise at the least-regularised end.

THE RISK BEING TESTED
---------------------
1-SE could push a large share of voxels to the HIGH edge. At the pilot cell,
voxel 208 at alpha = 1e2 sits 1.9% above its minimum; if the fold-to-fold SE is
several percent then alpha_max qualifies, every condition collapses toward the
train-mean predictor, and the comparison is empty. That is the section-1.5
saturation failure arriving through excess regularisation rather than excess
capacity. If it happens, 1-SE is the wrong rule.

THE REGIME SHIFT
----------------
    g = 1  d =   32  d/n_train = 0.06  heavily overdetermined; alpha nearly inert
    g = 4  d =  512  d/n_train = 1.02  d ~ n, worst-conditioned
    g = 8  d = 2048  d/n_train = 4.06  interpolation regime

The plateau that tripped D1 is a g = 1 artefact of alpha being nearly inert there.
The rule must NOT be tuned to g = 1; g = 4 sits on d ~ n and is where selection
matters most.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "ridge"))

from folds import outer_inner_splits                                  # noqa: E402
from ridge_dual import (DualRidge, alpha_grid, standardize,           # noqa: E402
                        mse_columns, pearson_columns, boundary_sides)

REPO = HERE.parents[1]
import paths  # noqa: E402
FEAT = paths.ridge_data_dir() / "pilot_features"
ARM = paths.ridge_data_dir() / "arm_b_out"
SUBJECT = "sub-01"
RELIABILITY_THRESHOLD = 0.2

RULE_KEYS = ["random", "backprop", "feedback_alignment", "predictive_coding",
             "stdp", "random_bn_calibrated"]
RULE_LABEL = {
    "random": "Random Weights", "backprop": "Backprop",
    "feedback_alignment": "Feedback Alignment",
    "predictive_coding": "Predictive Coding", "stdp": "STDP",
    "random_bn_calibrated": "Random Weights (BN-calibrated)",
}
SELECTORS = ["min-MSE", "1-SE", "tol-1%", "tol-0.1%"]


def select(mse_mean, mse_se, rule):
    """(n_alpha, n_vox) -> index per voxel. Alphas are ascending, so 'largest
    qualifying alpha' is the highest qualifying index."""
    n_a, n_v = mse_mean.shape
    imin = np.argmin(mse_mean, axis=0)
    vmin = mse_mean[imin, np.arange(n_v)]

    if rule == "min-MSE":
        return imin
    if rule == "1-SE":
        thr = vmin + mse_se[imin, np.arange(n_v)]
    elif rule == "tol-1%":
        thr = vmin * 1.01
    elif rule == "tol-0.1%":
        thr = vmin * 1.001
    else:
        raise ValueError(rule)

    # Largest index whose MSE is within threshold. Guard: never return an index
    # below the argmin (a qualifying point left of the minimum would be a worse
    # fit chosen for being less regularised, which no rule here intends).
    ok = mse_mean <= thr[None, :]
    idx = np.where(ok.any(axis=0), (n_a - 1) - np.argmax(ok[::-1], axis=0), imin)
    return np.maximum(idx, imin)


def measure(X, Y, reliable, alphas, n_outer=8, n_inner=5, seed=20260818):
    """One condition -> per-selector stats. D1 is NOT asserted here; this is a
    measurement of selection behaviour, not a fitting run producing results."""
    groups = np.arange(len(X))
    n_a, n_v = len(alphas), Y.shape[1]
    out = {s: {"idx": [], "pred": np.full(Y.shape, np.nan)} for s in SELECTORS}

    for tr, te, inner in outer_inner_splits(groups, n_outer, n_inner, seed):
        per_fold = np.zeros((n_inner, n_a, n_v))
        for k, (itr, iva) in enumerate(inner):
            Xi, Xv = standardize(X[itr], X[iva])
            m = DualRidge(alphas).fit(Xi, Y[itr])
            preds = m.predict_path(Xv)
            per_fold[k] = np.stack([mse_columns(p, Y[iva]) for p in preds])
        mse_mean = per_fold.mean(axis=0)
        mse_se = per_fold.std(axis=0, ddof=1) / np.sqrt(n_inner)

        Xtr, Xte = standardize(X[tr], X[te])
        m = DualRidge(alphas).fit(Xtr, Y[tr])
        preds = m.predict_path(Xte)
        for s in SELECTORS:
            idx = select(mse_mean, mse_se, s)
            out[s]["idx"].append(idx)
            out[s]["pred"][te] = preds[idx, :, np.arange(n_v)].T

    res = {}
    for s in SELECTORS:
        idx = np.stack(out[s]["idx"])                       # (n_outer, n_vox)
        lo_m, hi_m = boundary_sides(idx, n_a)
        sel_a = alphas[idx]
        r = pearson_columns(out[s]["pred"], Y)
        res[s] = {
            "r_med_reliable": float(np.median(r[reliable])),
            "alpha_med": float(np.median(sel_a[:, reliable])),
            "alpha_p5": float(np.percentile(sel_a[:, reliable], 5)),
            "alpha_p95": float(np.percentile(sel_a[:, reliable], 95)),
            "lo_rel": float(lo_m[:, reliable].mean()),
            "hi_rel": float(hi_m[:, reliable].mean()),
            "lo_unrel": float(lo_m[:, ~reliable].mean()),
            "hi_unrel": float(hi_m[:, ~reliable].mean()),
            "n_lo_rel_total": int(lo_m[:, reliable].sum()),
        }
    return res


def main(grids=(1, 4, 8), seeds=(0, 1, 2, 3, 4)):
    a = np.load(ARM / f"{SUBJECT}_arma.npz")
    b = np.load(ARM / f"{SUBJECT}_armb.npz")
    v1 = a["local_V1"]
    Y = a["betas"][:, v1].astype(np.float64)
    rel = b["reliability"][v1]
    reliable = rel > RELIABILITY_THRESHOLD
    alphas = alpha_grid()

    print("=" * 100)
    print("TOLERANCE MEASUREMENT -- measurement only, nothing adopted")
    print(f"224 px, V1, {SUBJECT}, {Y.shape[1]} voxels, {int(reliable.sum())} reliable")
    print(f"alpha grid {alphas[0]:.0e}..{alphas[-1]:.0e}, {len(alphas)} pts | "
          f"D1 assertion NOT applied (this is a measurement)")
    print("=" * 100)

    import pandas as pd
    rows = []
    for g in grids:
        n_train = 630
        d = 32 * g * g
        print(f"\n{'=' * 100}\nRUNG g = {g}   d = {d}   d/n_train = {d / n_train:.2f}"
              f"   {'overdetermined' if d < n_train else 'interpolation'}\n{'=' * 100}")
        for key in RULE_KEYS:
            for si in seeds:
                f = FEAT / f"model_{key}_seed{si}_g{g}.npy"
                if not f.exists():
                    print(f"  [missing {f.name}]")
                    continue
                X = np.load(f).astype(np.float64)
                res = measure(X, Y, reliable, alphas)
                for s, v in res.items():
                    rows.append(dict(g=g, d=d, rule=RULE_LABEL[key], seed=si,
                                     selector=s, **v))
                print(f"  {RULE_LABEL[key]:<32} s{si}  " + "  ".join(
                    f"{s}: r={res[s]['r_med_reliable']:+.4f} "
                    f"a={res[s]['alpha_med']:.1e} lo={res[s]['lo_rel']:.3f} "
                    f"hi={res[s]['hi_rel']:.3f}" for s in SELECTORS), flush=True)
        pd.DataFrame(rows).to_csv(REPO / f"tolerance_g{g}.csv", index=False)
    print(f"\nwrote tolerance_measurement.csv ({len(rows)} rows)")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys as _s
    gs = [int(x) for x in _s.argv[1:]] or [1, 4, 8]
    main(grids=gs)

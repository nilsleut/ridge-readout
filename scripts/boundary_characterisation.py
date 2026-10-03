"""
boundary_characterisation.py
============================
MEASUREMENT ONLY. Nothing here is adopted; nothing here fits real data.

Measures how often inner-CV alpha selection lands on a grid boundary, under:

  * two selection CRITERIA   -- Pearson r (current, scale-invariant) vs MSE
                                (equivalently R^2: for a fixed validation fold
                                argmin MSE == argmax R^2 exactly, since
                                R^2 = 1 - MSE*n/SS_tot with SS_tot fixed);
  * split by BOUNDARY SIDE   -- low (alpha_min) vs high (alpha_max), never summed;
  * split by RELIABILITY     -- voxels above/below the config's 0.2 split-half
                                threshold, computed the way Arm B will compute it.

WHY THE CRITERION IS THE ROOT CAUSE
-----------------------------------
Pearson r is scale-invariant. Ridge's alpha controls the SCALE of the prediction far
more than its direction, so r(alpha) is nearly flat -- at alpha = 1e12 the pipeline
still returns |r| ~ 0.96 with predictions constant to 1e-6. A criterion that barely
moves across twelve orders of magnitude cannot locate an optimum, so argmax over it is
argmax over noise, and noise-argmax favours endpoints. Ridge minimises squared error;
MSE is the matched criterion and is NOT scale-invariant.

Under MSE a no-signal voxel should select alpha_max stably and correctly -- maximal
regularisation IS the right answer when there is nothing to fit. That is the estimator
working, not an alarm.

Geometry is the REAL Arm A geometry throughout: n = 720, outer 8 -> 630 training,
inner 5 -> 504 training per inner fit.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ridge"))

from folds import outer_inner_splits
from ridge_dual import (DualRidge, alpha_grid, standardize, pearson_columns,
                        boundary_sides)

N_STIM = 720
N_OUTER, N_INNER = 8, 5
SEED = 20260818
RELIABILITY_THRESHOLD = 0.2          # config.yaml brain.voxel_selection.threshold


# ── synthetic population with a realistic reliability spread ─────────────────

def make_population(d: int, n_vox: int, snr_range, rng):
    """Voxels spanning a range of SNR, plus the split-half reliability Arm B would
    measure for each.

    Reliability is computed exactly as config.yaml specifies for Arm B: two
    independent half-averages, correlate across stimuli, Spearman-Brown correct.
    """
    X = rng.standard_normal((N_STIM, d))
    W = rng.standard_normal((d, n_vox))
    signal = X @ W
    signal /= signal.std(axis=0, keepdims=True)
    snr = rng.uniform(snr_range[0], snr_range[1], size=n_vox)

    # Two independent noisy halves -> split-half reliability, then Spearman-Brown.
    h1 = signal * snr + rng.standard_normal((N_STIM, n_vox))
    h2 = signal * snr + rng.standard_normal((N_STIM, n_vox))
    r_half = pearson_columns(h1, h2)
    rel = np.clip(2 * r_half / (1 + r_half), -1, 1)

    Y = (h1 + h2) / 2.0
    return X, Y, rel


# ── one characterisation run ────────────────────────────────────────────────

def characterise(X, Y, rel, alphas, criterion: str):
    """Return side-split boundary fractions, pooled over outer folds."""
    groups = np.arange(len(X))
    reliable = rel > RELIABILITY_THRESHOLD
    n_a = len(alphas)
    acc = {k: [] for k in ("lo_rel", "hi_rel", "lo_unrel", "hi_unrel")}
    sel_alphas = []

    for tr, te, inner in outer_inner_splits(groups, N_OUTER, N_INNER, SEED):
        crit = np.zeros((n_a, Y.shape[1]))
        for itr, iva in inner:
            Xi, Xv = standardize(X[itr], X[iva])
            m = DualRidge(alphas).fit(Xi, Y[itr])
            preds = m.predict_path(Xv)
            if criterion == "pearson_r":
                crit += np.stack([pearson_columns(p, Y[iva]) for p in preds])
            elif criterion == "mse":
                # higher-is-better, so negate
                crit += -np.stack([((Y[iva] - p) ** 2).mean(axis=0) for p in preds])
            else:
                raise ValueError(criterion)
        idx = np.argmax(crit / len(inner), axis=0)
        lo_m, hi_m = boundary_sides(idx, n_a)
        acc["lo_rel"].append(lo_m[reliable]); acc["hi_rel"].append(hi_m[reliable])
        acc["lo_unrel"].append(lo_m[~reliable]); acc["hi_unrel"].append(hi_m[~reliable])
        sel_alphas.append(alphas[idx])

    out = {k: (float(np.concatenate(v).mean()) if len(np.concatenate(v)) else float("nan"))
           for k, v in acc.items()}
    out["any_rel"] = out["lo_rel"] + out["hi_rel"]
    out["any_unrel"] = out["lo_unrel"] + out["hi_unrel"]
    n_rel, n_unrel = int(reliable.sum()), int((~reliable).sum())
    out["any_all"] = (out["any_rel"] * n_rel + out["any_unrel"] * n_unrel) / (n_rel + n_unrel)
    out["n_reliable"] = n_rel
    out["n_total"] = int(len(reliable))
    out["median_selected_alpha"] = float(np.median(np.concatenate(sel_alphas)))
    return out

# ── Task 2: a threshold derived from the SIGNAL case, not the null ──────────
#
# The 0.95 candidate was calibrated to NOT fire on pure noise (measured 0.911). That
# is a calibration derived from the null, which makes the control permissive by
# construction. The question a threshold must answer is the opposite one: what
# boundary rate does a grid that only JUST contains the optimum produce, versus one
# that has been truncated past it? A usable threshold separates those two.

def truncation_experiment(X, Y, rel, criterion: str, wide=(1e-6, 1e12, 61)):
    """Locate the true optimum on a wide grid, then measure boundary rates on
    grids that (a) contain it generously, (b) only just contain it, (c) exclude it."""
    groups = np.arange(len(X))
    reliable = rel > RELIABILITY_THRESHOLD
    wide_grid = np.logspace(np.log10(wide[0]), np.log10(wide[1]), wide[2])

    # True optimum per voxel, from held-out data on the wide grid.
    tr, te, _ = next(iter(outer_inner_splits(groups, N_OUTER, N_INNER, SEED)))
    Xtr, Xte = standardize(X[tr], X[te])
    m = DualRidge(wide_grid).fit(Xtr, Y[tr])
    preds = m.predict_path(Xte)
    if criterion == "pearson_r":
        crit = np.stack([pearson_columns(p, Y[te]) for p in preds])
    else:
        crit = -np.stack([((Y[te] - p) ** 2).mean(axis=0) for p in preds])
    opt = wide_grid[np.argmax(crit, axis=0)]
    opt_rel = opt[reliable]
    lo_opt, hi_opt = float(np.percentile(opt_rel, 5)), float(np.percentile(opt_rel, 95))

    grids = {
        "generous (optimum well inside)":
            np.logspace(np.log10(lo_opt) - 3, np.log10(hi_opt) + 3, 25),
        "just contains the optimum":
            np.logspace(np.log10(lo_opt), np.log10(hi_opt), 25),
        "truncated PAST the optimum (1 decade)":
            np.logspace(np.log10(hi_opt) + 1, np.log10(hi_opt) + 5, 25),
    }
    rows = {}
    for name, g in grids.items():
        rows[name] = characterise(X, Y, rel, g, criterion)
    return rows, (lo_opt, hi_opt)


def _fmt(o):
    return (f"lo_rel={o['lo_rel']:.3f}  hi_rel={o['hi_rel']:.3f}  "
            f"lo_unrel={o['lo_unrel']:.3f}  hi_unrel={o['hi_unrel']:.3f}  "
            f"| any_rel={o['any_rel']:.3f}  any_all={o['any_all']:.3f}")


def main():
    rng = np.random.default_rng(20260819)
    a = alpha_grid()
    print("=" * 96)
    print("BOUNDARY CHARACTERISATION -- measurement only, nothing adopted")
    print(f"geometry: n={N_STIM}, outer {N_OUTER} -> {N_STIM - N_STIM//N_OUTER} train, "
          f"inner {N_INNER} -> {int((N_STIM - N_STIM//N_OUTER) * (1 - 1/N_INNER))} per inner fit")
    print(f"alpha grid: {a[0]:.0e} .. {a[-1]:.0e}, {len(a)} points")
    print("=" * 96)

    pops = {
        "pure noise (no signal)": (0.0, 0.0),
        "weak signal (below the 0.2 reliability threshold)": (0.02, 0.12),
        "mixed (spans the threshold)": (0.0, 0.35),
        "moderate signal (mostly reliable)": (0.20, 0.60),
    }
    print("\nTASK 1+2 -- boundary rates by criterion, side, and reliability\n")
    for label, snr in pops.items():
        X, Y, rel = make_population(d=32, n_vox=600, snr_range=snr, rng=rng)
        n_rel = int((rel > RELIABILITY_THRESHOLD).sum())
        print(f"{label}")
        print(f"   (d=32, 600 voxels, {n_rel} reliable at rel>0.2)")
        for crit in ("pearson_r", "mse"):
            o = characterise(X, Y, rel, a, crit)
            print(f"   {crit:<10} {_fmt(o)}   median alpha={o['median_selected_alpha']:.2e}")
        print()

    print("\nTASK 2 -- threshold derived from the SIGNAL case (grid truncation)\n")
    X, Y, rel = make_population(d=32, n_vox=600, snr_range=(0.10, 0.60), rng=rng)
    for crit in ("pearson_r", "mse"):
        rows, (lo, hi) = truncation_experiment(X, Y, rel, crit)
        print(f"  criterion={crit}   true optimum (reliable voxels) spans "
              f"{lo:.2e} .. {hi:.2e}")
        for name, o in rows.items():
            print(f"    {name:<38} {_fmt(o)}")
        print()


if __name__ == "__main__":
    main()

"""
step2_lower_edge.py
===================
DIAGNOSIS ONLY, EXPLORATORY, sub-01 only. The alpha-selection rule (min-MSE, D2) and
the D1 assertion are unchanged; no tolerance form is proposed.

Condition: Random Weights (BN-calibrated) seed_idx 3, g = 1, 224 px, sub-01, V1, 501
reliable voxels. Selection exactly as scripts/tolerance_measurement.py (D1 not
asserted there): its `measure` and `select` are imported, not re-implemented.

Gate (STOP on failure): `measure` must reproduce the min-MSE row of
results/tolerance_all_rungs.csv for this cell (r_med_reliable within 1e-6) and give
4 low-edge selections among reliable voxels.

For every low-edge selection (voxel, outer fold): the mean inner-CV MSE curve on the
standard grid and on an extended DIAGNOSTIC grid (standard + 1e-7, 1e-8, 1e-9,
1e-10, 1e-12). The extended grid feeds no result.

Classification, fixed before the run, per hit:
  "True optimum below grid"  extended-grid minimum at alpha < 1e-6 AND relative gain >= 1e-3
  "Plateau"                  relative gain < 1e-3
  "unclear"                  otherwise
relative gain = (MSE(1e-6) - MSE_min_extended) / MSE(1e-6)

Output: results/step2/lower_edge_diagnosis.csv, results/step2/lower_edge_curves.csv
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src" / "ridge"))
sys.path.insert(0, str(HERE))

import paths                                                        # noqa: E402
from config_hash import config_hash                                 # noqa: E402
from folds import outer_inner_splits                                # noqa: E402
from ridge_dual import DualRidge, alpha_grid, standardize, mse_columns  # noqa: E402
from tolerance_measurement import measure, select                   # noqa: E402

SUBJECT = "sub-01"
KEY, LABEL, SEED_IDX, G = "random_bn_calibrated", "Random Weights (BN-calibrated)", 3, 1
EXTRA = [1e-7, 1e-8, 1e-9, 1e-10, 1e-12]
OUT = REPO / "results" / "step2"
N_OUTER, N_INNER, SEED = 8, 5, 20260818        # tolerance_measurement.measure defaults
EPS = 1e-8                                     # ridge_dual.standardize default


def git_head():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def stop(msg):
    raise SystemExit(f"STOP: {msg}")


def inner_mse(X, Y, inner, alphas):
    """Mean inner-CV MSE, (n_alpha, n_vox) -- the same computation as measure()."""
    per_fold = np.zeros((len(inner), len(alphas), Y.shape[1]))
    for k, (itr, iva) in enumerate(inner):
        Xi, Xv = standardize(X[itr], X[iva])
        preds = DualRidge(alphas).fit(Xi, Y[itr]).predict_path(Xv)
        per_fold[k] = np.stack([mse_columns(p, Y[iva]) for p in preds])
    return per_fold.mean(axis=0)


def design_stats(Xraw):
    sd = Xraw.std(axis=0)
    Xs = standardize(Xraw)[0]
    return int(np.linalg.matrix_rank(Xs)), float(np.linalg.cond(Xs)), int((sd < EPS).sum())


def main():
    h, gh = config_hash(), git_head()
    data = paths.ridge_data_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    a = np.load(data / "arm_b_out" / f"{SUBJECT}_arma.npz")
    b = np.load(data / "arm_b_out" / f"{SUBJECT}_armb.npz")
    v1 = a["local_V1"]
    Y = a["betas"][:, v1].astype(np.float64)
    reliable = b["reliability"][v1] > 0.2
    X = np.load(data / "pilot_features" / f"model_{KEY}_seed{SEED_IDX}_g{G}.npy").astype(np.float64)
    alphas = alpha_grid()

    # ── gate: reproduce the tolerance-measurement cell ──────────────────────
    res = measure(X, Y, reliable, alphas)["min-MSE"]
    tol = pd.read_csv(REPO / "results" / "tolerance_all_rungs.csv")
    row = tol[(tol.g == G) & (tol.rule == LABEL) & (tol.seed == SEED_IDX) & (tol.selector == "min-MSE")]
    if len(row) != 1:
        stop("cell not found in tolerance_all_rungs.csv")
    row = row.iloc[0]
    d_r = abs(res["r_med_reliable"] - row.r_med_reliable)
    print(f"gate: r_med_reliable {res['r_med_reliable']:.10f} vs CSV {row.r_med_reliable:.10f} "
          f"(|d| {d_r:.2e}); n_lo_rel_total {res['n_lo_rel_total']} (CSV {row.n_lo_rel_total})")
    if d_r > 1e-6 or res["n_lo_rel_total"] != 4:
        stop("gate failed")

    # ── locate hits, extended curves ────────────────────────────────────────
    ext = np.unique(np.concatenate([alphas, EXTRA]))
    i_edge_ext = int(np.flatnonzero(np.isclose(ext, alphas[0], rtol=1e-12))[0])
    groups = np.arange(len(X))
    hits, curves = [], []
    for fi, (tr, te, inner) in enumerate(outer_inner_splits(groups, N_OUTER, N_INNER, SEED)):
        m_std = inner_mse(X, Y, inner, alphas)
        idx = select(m_std, None, "min-MSE")
        lo = np.flatnonzero((idx == 0) & reliable)
        if not len(lo):
            continue
        m_ext = inner_mse(X, Y[:, lo], inner, ext)
        # sanity: the standard-grid points of the extended fit equal the standard fit
        on_std = [int(np.flatnonzero(np.isclose(ext, al, rtol=1e-12))[0]) for al in alphas]
        dev = float(np.abs(m_ext[on_std] - m_std[:, lo]).max())
        r_out, c_out, dead_out = design_stats(X[tr])
        inner_stats = [design_stats(X[itr]) for itr, _ in inner]
        for j, v in enumerate(lo):
            cur = m_ext[:, j]
            k = int(np.argmin(cur))
            mse_edge, mse_min = float(cur[i_edge_ext]), float(cur[k])
            gain = (mse_edge - mse_min) / mse_edge
            if ext[k] < 1e-6 and gain >= 1e-3:
                cls = "True optimum below grid"
            elif gain < 1e-3:
                cls = "Plateau"
            else:
                cls = "unclear"
            hits.append(dict(
                subject=SUBJECT, roi="V1", rule=LABEL, seed_idx=SEED_IDX, grid=G, resolution=224,
                outer_fold=fi, voxel_v1_local=int(v), voxel_reliable_rank=int(np.flatnonzero(np.flatnonzero(reliable) == v)[0]),
                voxel_roi_column=int(v1[v]),
                alpha_min_extended=float(ext[k]), mse_at_1e_6=mse_edge, mse_min_extended=mse_min,
                relative_gain=gain, mse_at_1e2=float(cur[int(np.flatnonzero(np.isclose(ext, 1e2))[0])]),
                classification=cls,
                rank_outer_train=r_out, cond_outer_train=c_out, dead_features_outer_train=dead_out,
                rank_inner_train_min=min(s[0] for s in inner_stats),
                cond_inner_train_max=max(s[1] for s in inner_stats),
                dead_features_inner_train_max=max(s[2] for s in inner_stats),
                n_train_outer=len(tr), d=X.shape[1], ext_vs_std_max_abs_dev=dev))
            for al, mv in zip(ext, cur):
                curves.append(dict(outer_fold=fi, voxel_v1_local=int(v), alpha=float(al),
                                   in_standard_grid=bool(np.isclose(alphas, al, rtol=1e-12).any()),
                                   inner_cv_mse=float(mv)))
    df = pd.DataFrame(hits)
    if len(df) != 4:
        stop(f"located {len(df)} hits, expected 4")
    for d in (df, cv := pd.DataFrame(curves)):
        d["config_hash"], d["git_head"] = h, gh
    df.to_csv(OUT / "lower_edge_diagnosis.csv", index=False)
    cv.to_csv(OUT / "lower_edge_curves.csv", index=False)
    pd.set_option("display.width", 250)
    print(df[["outer_fold", "voxel_v1_local", "alpha_min_extended", "mse_at_1e_6", "mse_min_extended",
              "relative_gain", "classification", "rank_outer_train", "cond_outer_train",
              "dead_features_outer_train", "cond_inner_train_max", "ext_vs_std_max_abs_dev"]].to_string())
    print("same voxel for all hits:", df.voxel_v1_local.nunique() == 1,
          "| distinct voxels:", sorted(df.voxel_v1_local.unique()))

    # ── other conditions at g = 1 (from the CSV, min-MSE) ───────────────────
    t = tol[(tol.g == 1) & (tol.selector == "min-MSE")]
    print("\nlow-edge reliable selections at g=1, min-MSE (tolerance_all_rungs.csv):")
    print(t[t.rule.isin(["Backprop", LABEL])][["rule", "seed", "n_lo_rel_total", "lo_rel"]].to_string(index=False))
    print("all conditions with n_lo_rel_total > 0:",
          t[t.n_lo_rel_total > 0][["rule", "seed", "n_lo_rel_total"]].to_dict("records"))


if __name__ == "__main__":
    main()

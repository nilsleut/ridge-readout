"""
step3_verify_fit.py
===================
Verification for the minimal fit_condition change (commit 9). g = 1, all 30
checkpoints, default folds, D1 in counting mode (assert_low_boundary=False).

  1. median r over the 501 reliable voxels equals results/tolerance_all_rungs.csv
     (min-MSE) to 1e-9 for every checkpoint;
  2. default path is bit-identical to fit.py at bec73cb: r, r2, r_path, df,
     selected_alpha and the four boundary cells compared with np.array_equal.
"""
import step3_common as C          # noqa: I001  (must come first: pins BLAS threads)

import subprocess
import types

import numpy as np
import pandas as pd

KEYS = ["r", "r2", "r_path", "df", "selected_alpha", "lo_rel", "hi_rel", "lo_unrel", "hi_unrel"]


def _old_fit():
    src = subprocess.run(["git", "-C", str(C.REPO), "show", "bec73cb:src/ridge/fit.py"],
                         capture_output=True, text=True, check=True, encoding="utf-8").stdout
    mod = types.ModuleType("fit_bec73cb")
    exec(compile(src, "fit_bec73cb.py", "exec"), mod.__dict__)
    return mod.fit_condition


def job(args):
    key, si = args
    Y, rel = C.brain()
    X = C.load_X(key, si, 1).astype(np.float64)
    new = C.fit_condition(X, Y, np.arange(720), reliable=rel, assert_low_boundary=False)
    old = _old_fit()(X, Y, np.arange(720), reliable=rel, assert_low_boundary=False)
    same = all(np.array_equal(np.asarray(new[k]), np.asarray(old[k])) for k in KEYS)
    return dict(rule=C.RULES[key], seed=si, r_med=float(np.median(new["r"][rel])), bit_identical=same,
                has_pred=new["pred_oof"].shape == Y.shape and (new["outer_fold_of_stimulus"] >= 0).all())


def main():
    rows = pd.DataFrame(C.pmap(job, [(k, s) for k in C.RULES for s in C.SEED_IDX]))
    tol = pd.read_csv(C.REPO / "results" / "tolerance_all_rungs.csv")
    tol = tol[(tol.g == 1) & (tol.selector == "min-MSE")][["rule", "seed", "r_med_reliable"]]
    m = rows.merge(tol, on=["rule", "seed"])
    m["abs_dev"] = (m.r_med - m.r_med_reliable).abs()
    print(m.to_string(index=False))
    ok = len(m) == 30 and m.abs_dev.max() <= 1e-9 and m.bit_identical.all() and m.has_pred.all()
    print(f"\nmax |dev| {m.abs_dev.max():.2e}; bit-identical {int(m.bit_identical.sum())}/30; "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

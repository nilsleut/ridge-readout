"""
step2_d1a.py
============
EXPLORATORY, sub-01 only. D1a on real data: do low-edge selections among the 501
reliable V1 voxels occur under the permuted-assignment control, where no model
signal exists? Expectation fixed before the run: 0.

The permuted control is the one in scripts/pilot_fit.py (control band, inline in
main(): backprop seed 0 features, rows permuted with np.random.default_rng(20260820);
no seed parameter exists, so only that default permutation is run). It is not a
function, so it cannot be imported on its own. This wrapper therefore CALLS
pilot_fit.main() unchanged and only rebinds module globals:
  GRIDS     -> [1, 4, 8]
  RULES, CONTROLS -> []      (skip the 30 model fits and the feature banks)
  REPO      -> a temporary directory (pilot_fit writes pilot_results.csv there)
  fit_condition -> a counting stand-in
The stand-in receives exactly the X that pilot_fit built and counts boundary
selections the way scripts/tolerance_measurement.py does (its `measure`, D1 not
asserted). As a cross-check it also runs fit.fit_condition(assert_low_boundary=False)
on the same X. Other calls made by main() (the g >= 2 cell-shuffle control) are
answered with a NaN placeholder and not recorded.

Output: results/step2/d1a_real_voxels.csv (config_hash, git_head).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src" / "ridge"))
sys.path.insert(0, str(HERE))

import pilot_fit                                     # noqa: E402
from config_hash import config_hash                  # noqa: E402
from fit import fit_condition as _fit_condition, CELLS  # noqa: E402
from tolerance_measurement import measure            # noqa: E402

OUT = REPO / "results" / "step2"
N_OUTER = 8
records = []


def counting_fit(X, Y, groups, *, alphas, reliable, context="", **kw):
    if not context.startswith("control:permuted"):
        return {"r": np.full(Y.shape[1], np.nan), **{k: np.full(N_OUTER, np.nan) for k in CELLS}}
    g = int(context.rsplit("g", 1)[1])
    m = measure(X, Y, reliable, alphas)["min-MSE"]
    fc = _fit_condition(X, Y, groups, alphas=alphas, reliable=reliable,
                        assert_low_boundary=False, context=context)
    n_rel = int(reliable.sum())
    records.append(dict(
        subject="sub-01", roi="V1", control="permuted (pilot_fit, rng 20260820, backprop seed0)",
        grid=g, d=X.shape[1], permutation_seed="default 20260820 (no parameter)",
        n_reliable=n_rel, n_outer=N_OUTER,
        n_lo_rel=int(m["n_lo_rel_total"]),
        n_hi_rel=int(round(m["hi_rel"] * N_OUTER * n_rel)),
        lo_rel=m["lo_rel"], hi_rel=m["hi_rel"], lo_unrel=m["lo_unrel"], hi_unrel=m["hi_unrel"],
        r_med_reliable=m["r_med_reliable"], alpha_med=m["alpha_med"],
        fitcond_r_med_reliable=float(np.median(fc["r"][reliable])),
        fitcond_lo_rel=float(np.mean(fc["lo_rel"])), fitcond_hi_rel=float(np.mean(fc["hi_rel"]))))
    print(f"  [d1a] g={g}: lo_rel {records[-1]['n_lo_rel']}  hi_rel {records[-1]['n_hi_rel']}  "
          f"r_med {m['r_med_reliable']:+.4f} (fit_condition {records[-1]['fitcond_r_med_reliable']:+.4f})",
          flush=True)
    return fc


def main():
    h = config_hash()
    gh = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, check=True).stdout.strip()
    with tempfile.TemporaryDirectory() as td:
        pilot_fit.GRIDS = [1, 4, 8]
        pilot_fit.RULES = []
        pilot_fit.CONTROLS = []
        pilot_fit.REPO = Path(td)
        pilot_fit.fit_condition = counting_fit
        pilot_fit.main()
    df = pd.DataFrame(records)
    if sorted(df.grid) != [1, 4, 8]:
        raise SystemExit(f"STOP: expected one permuted fit per g in 1, 4, 8, got {list(df.grid)}")
    df["config_hash"], df["git_head"] = h, gh
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "d1a_real_voxels.csv", index=False)
    print(df[["grid", "d", "n_lo_rel", "n_hi_rel", "hi_rel", "r_med_reliable",
              "fitcond_r_med_reliable", "alpha_med"]].to_string(index=False))
    if df.n_lo_rel.sum() > 0:
        print("NOTE: low-edge selections occur without signal (expectation was 0).")


if __name__ == "__main__":
    main()

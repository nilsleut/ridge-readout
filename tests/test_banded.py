"""
test_banded.py
==============
Checks for src/ridge/banded.py (step 40, part A). All must pass before any step 40 run.

  (i)   alpha_net fixed at 1e12: banded reproduces the bank-alone predictions (max |d| <= 1e-6)
  (ii)  alpha_bank fixed at 1e12: banded reproduces the network-alone predictions
  (iii) only the network group carries signal: the selected alpha_bank sits at the upper
        edge, or r(joint) >= r(network alone) - 0.002 (median over voxels)
  (iv)  one group: fit_banded equals fit.fit_condition on the same grid and folds
  (v)   banded_predict with one group equals ridge_dual.DualRidge on every alpha

Run: python tests/test_banded.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "ridge"))

from banded import banded_predict, fit_banded, _kernels   # noqa: E402
from fit import fit_condition                              # noqa: E402
from ridge_dual import DualRidge, alpha_grid, standardize  # noqa: E402

GRID = alpha_grid(1e-6, 1e8, 15)
FAILS, N = [], 0


def check(name, ok, info=""):
    global N
    N += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAILS.append(name)


def data(seed, signal_bank, signal_net, n=160, d_bank=8, d_net=64, v=24):
    rng = np.random.default_rng(seed)
    Xb = rng.normal(size=(n, d_bank)) * rng.uniform(0.5, 3, d_bank)
    Xn = rng.normal(size=(n, d_net)) * rng.uniform(0.5, 3, d_net)
    Y = (signal_bank * Xb @ rng.normal(size=(d_bank, v)) / np.sqrt(d_bank)
         + signal_net * Xn @ rng.normal(size=(d_net, v)) / np.sqrt(d_net)
         + rng.normal(size=(n, v)))
    return Xb, Xn, Y


def main():
    print("banded ridge checks")
    Xb, Xn, Y = data(1, 1.0, 1.0)

    bank = fit_banded([Xb], Y, grids=[GRID], gate=False)
    net = fit_banded([Xn], Y, grids=[GRID], gate=False)
    j_i = fit_banded([Xb, Xn], Y, grids=[GRID, [1e12]], gate=False)
    j_ii = fit_banded([Xb, Xn], Y, grids=[[1e12], GRID], gate=False)
    d1 = float(np.abs(j_i["pred_oof"] - bank["pred_oof"]).max())
    d2 = float(np.abs(j_ii["pred_oof"] - net["pred_oof"]).max())
    check("(i) alpha_net = 1e12 reproduces bank alone", d1 <= 1e-6, f"max |d| = {d1:.2e}")
    check("(ii) alpha_bank = 1e12 reproduces network alone", d2 <= 1e-6, f"max |d| = {d2:.2e}")

    Xb3, Xn3, Y3 = data(2, 0.0, 1.5)
    joint = fit_banded([Xb3, Xn3], Y3, grids=[GRID, GRID], gate=False)
    net3 = fit_banded([Xn3], Y3, grids=[GRID], gate=False)
    upper = float(np.mean(joint["selected"][0] == GRID[-1]))
    rj, rn = float(np.median(joint["r"])), float(np.median(net3["r"]))
    check("(iii) noise bank: alpha_bank at upper edge or r(joint) >= r(net) - 0.002",
          upper >= 0.5 or rj >= rn - 0.002, f"upper-edge share {upper:.2f}, r joint {rj:.4f}, r net {rn:.4f}")

    ref = fit_condition(Xb, Y, np.arange(len(Y)), alphas=GRID, assert_low_boundary=False)
    d4 = float(np.abs(ref["pred_oof"] - bank["pred_oof"]).max())
    check("(iv) one group equals fit_condition", d4 <= 1e-8, f"max |d| = {d4:.2e}")

    tr, te = np.arange(120), np.arange(120, 160)
    Ktr, Kte = _kernels([Xn[tr]], [Xn[te]])
    P = banded_predict(Ktr, Kte, Y[tr], GRID[:, None])
    a, b = standardize(Xn[tr], Xn[te])
    Q = DualRidge(GRID).fit(a, Y[tr]).predict_path(b)
    d5 = float(np.abs(P - Q).max())
    check("(v) banded_predict equals DualRidge on every alpha", d5 <= 1e-8, f"max |d| = {d5:.2e}")

    if FAILS:
        print(f"  {len(FAILS)} failure(s) of {N} checks")
        sys.exit(1)
    print(f"  all {N} checks passed")


if __name__ == "__main__":
    main()

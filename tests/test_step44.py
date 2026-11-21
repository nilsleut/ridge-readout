"""
test_step44.py
==============
Checks for step 44 (amendments/2026-11-19_step44-predictions.md, config_step44.yaml).
All must pass before the step 44 runs.

  (i)   partial Spearman (step44_why.partial_rho, rank_resid / pearson of compute_lowlevel.py) on
        synthetic data with a known solution: for permutations of 0..n-1 (ranks = values) it equals
        the first-order and second-order partial correlation formulas on the Spearman
        correlations (1e-12); with a common cause z (x = z + e1, y = z + e2) partialling z
        removes the correlation (|partial| < 0.02 at n = 20000, raw > 0.4)
  (ii)  without covariates partial_rho equals Spearman (scipy and nsd_step43.rho_pairs), with
        ties, max |d| <= 1e-12
  (iii) the subset code (crossrun_index of a stimulus sample + delta_on) with the full 720 set
        reproduces the paper cell of results/step5 (+0.030) to 1e-12, and the full sample gives
        the 210,205 cross-run pairs
  (iv)  the Teil 0 gates: the 720 images in one session per subject, sub-01 and sub-02 with the
        same order and sub-03 not; the luminance RDM (MEANLUM of arXiv:2608.12408) against
        THINGS-720 V1 (cross-run, per subject, mean) reproduces 0.054 to +-0.001

Run: python tests/test_step44.py [--only i,ii,iii,iv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import nsd_common as C                                       # noqa: E402,F401
import step44_why as W                                       # noqa: E402

FAILS, N = [], 0


def check(name, ok, info=""):
    global N
    N += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)


def sp(a, b):
    return float(spearmanr(a, b).correlation)


def test_partial_synthetic():
    rng = np.random.default_rng(44)
    worst = 0.0
    for _ in range(20):
        n = 500
        base = rng.normal(size=(3, n))
        base[0] += 0.6 * base[2]
        base[1] += 0.5 * base[2] - 0.3 * base[0]
        x, y, z = (np.argsort(np.argsort(v)).astype(float) for v in base)   # permutations of 0..n-1
        rxy, rxz, ryz = sp(x, y), sp(x, z), sp(y, z)
        f1 = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
        worst = max(worst, abs(W.partial_rho(x, y, [z]) - f1))
        w = np.argsort(np.argsort(rng.normal(size=n) + 0.4 * x)).astype(float)
        # second order: partial(x, y | z, w) from first-order partials given z
        pxy, pxw, pyw = W.partial_rho(x, y, [z]), W.partial_rho(x, w, [z]), W.partial_rho(y, w, [z])
        f2 = (pxy - pxw * pyw) / np.sqrt((1 - pxw ** 2) * (1 - pyw ** 2))
        worst = max(worst, abs(W.partial_rho(x, y, [z, w]) - f2))
    check("(i) partial Spearman equals the partial-correlation formulas (1 and 2 covariates)", worst <= 1e-12,
          f"max |d| {worst:.1e}")
    n = 20000
    zc = rng.normal(size=n)
    x, y = zc + rng.normal(size=n), zc + rng.normal(size=n)
    raw, par = sp(x, y), W.partial_rho(x, y, [zc])
    check("(i) partialling a common cause removes the correlation", abs(par) < 0.02 and raw > 0.4,
          f"raw {raw:.3f}, partial {par:+.4f}")


def test_no_covariate():
    import nsd_step43 as NS
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(20):
        a = np.round(rng.normal(size=3000), 1)        # ties
        b = np.round(a + rng.normal(size=3000), 1)
        r = W.partial_rho(a, b, [])
        r2 = float(NS.rho_pairs(a[None], b[None], np.array([[0, 0]]), np.arange(len(a)))[0])
        worst = max(worst, abs(r - sp(a, b)), abs(r - r2))
    check("(ii) without covariates equals Spearman", worst <= 1e-12, f"max |d| {worst:.1e}")


def test_subset_full():
    models, brains = W.things720_models("V1"), W.things720_brains("V1")
    ci_full = W.crossrun_index(np.arange(W.N720))
    same = np.array_equal(ci_full, W.crossrun_index())
    d, _ = W.delta_on(models, brains, ci_full)
    ref = pd.read_csv(REPO / "results" / "step5" / "resolution_rsa.csv", low_memory=False)
    ref = float(ref[(ref.kind == "gap_point") & (ref.arm == "NATIVE") & (ref.res == 224) & (ref.roi == "V1")
                    & (ref.layer == "Conv1") & (ref.variant == "raw")].value.iloc[0])
    check("(iii) subset code with the full set reproduces the paper cell", abs(d - ref) <= 1e-12 and same
          and len(ci_full) == 210205, f"{d:.9f} vs {ref:.9f}, |d| {abs(d - ref):.1e}, {len(ci_full)} pairs")
    # a 100-image subset: pairs equal brute force (both in the subset, different runs for every subject)
    from common_up_pairs import run_labels
    from cvrsa import condensed_index
    runs = run_labels()
    ok, n = True, 0
    for seed in range(3):
        stims = np.sort(np.random.default_rng(seed).choice(W.N720, 100, replace=False))
        brute = sorted(condensed_index(np.array([i]), np.array([j]), W.N720)[0]
                       for x, i in enumerate(stims) for j in stims[x + 1:]
                       if all(r[i] != r[j] for r in runs.values()))
        got = W.crossrun_index(stims)
        ok &= np.array_equal(np.sort(got), np.array(brute))
        n += len(got)
    check("(iii) subset pairs equal brute force (3 subsets of 100)", bool(ok), f"{n} pairs")


def test_gates():
    pos = W.positions()                                # STOPs unless one session per subject
    same12 = np.array_equal(pos["sub-01"], pos["sub-02"])
    diff3 = not np.array_equal(pos["sub-01"], pos["sub-03"])
    rng_ok = all(p.min() >= 0 and p.max() <= 819 and len(np.unique(p)) == 720 for p in pos.values())
    check("(iv) order: one session each, sub-01 = sub-02, sub-03 differs", same12 and diff3 and rng_ok)
    L, _ = W.lum_contrast(W.things720_images())
    Lv = W.abs_vec(L)
    ci = W.crossrun_index()
    rl = [W.spear(b[ci], Lv[ci]) for b in W.things720_brains("V1")]
    v = float(np.mean(rl))
    check("(iv) luminance RDM vs THINGS-720 V1 reproduces 0.054", abs(v - 0.054) <= 0.001,
          f"{v:.6f} (per subject {', '.join(f'{x:.4f}' for x in rl)})")


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    print("step 44 checks")
    for tag, fn in (("i", test_partial_synthetic), ("ii", test_no_covariate), ("iii", test_subset_full), ("iv", test_gates)):
        if only is None or tag in only:
            fn()
    if FAILS:
        print(f"  {len(FAILS)} failure(s) of {N} checks")
        sys.exit(1)
    print(f"  all {N} checks passed")


if __name__ == "__main__":
    main()

"""
test_cvrsa.py
=============
Checks for src/ridge/cvrsa.py (step 40, part C). All checks must pass before any run.

  (i)   pure noise: the mean cross-validated distance over all pairs lies within 3 SE of 0
        (200 simulations; SE of the mean over simulations)
  (ii)  signal plus noise: cross-validated distances are unbiased for the true squared
        distances (mean relative error over pairs < 5 % across 1000 simulations)
  (iii) the fast formula equals the explicit mean over ordered session pairs
  (iv)  condensed indexing and the bootstrap draws drop self-pairs and match pdist order
Reported, not a check (needs the THINGS data; skipped without RIDGE_DATA_DIR/THINGS_FMRI_DIR):
  b2 versus b3 Spearman on sub-01 V1.

Run: python tests/test_cvrsa.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "ridge"))

from cvrsa import boot_draws, condensed_index, cv_sq_euclid, spearman  # noqa: E402

FAILS, N = [], 0


def check(name, ok, info=""):
    global N
    N += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAILS.append(name)


def main():
    print("cross-validated RSA checks")
    rng = np.random.default_rng(7)
    means = [cv_sq_euclid(rng.normal(size=(20, 12, 50))).mean() for _ in range(200)]
    m, se = float(np.mean(means)), float(np.std(means, ddof=1) / np.sqrt(len(means)))
    check("(i) pure noise: mean within 3 SE of 0", abs(m) <= 3 * se, f"mean {m:.4f}, SE {se:.4f}")

    mu = rng.normal(size=(10, 40))
    true = pdist(mu, "sqeuclidean")
    acc = np.zeros_like(true)
    for _ in range(1000):
        acc += cv_sq_euclid(mu[:, None, :] + rng.normal(size=(10, 12, 40)) * 1.5)
    rel = float(np.mean(np.abs(acc / 1000 - true) / true))
    check("(ii) unbiased for true squared distances", rel < 0.05, f"mean relative error {rel:.4f}")

    reps = rng.normal(size=(6, 4, 9))
    brute = []
    for i in range(6):
        for j in range(i + 1, 6):
            v = [(reps[i, A] - reps[j, A]) @ (reps[i, B] - reps[j, B])
                 for A in range(4) for B in range(4) if A != B]
            brute.append(np.mean(v))
    d3 = float(np.abs(np.array(brute) - cv_sq_euclid(reps)).max())
    check("(iii) fast formula equals explicit session-pair mean", d3 <= 1e-10, f"max |d| = {d3:.1e}")

    X = rng.normal(size=(7, 3))
    D = pdist(X)
    i, j = np.triu_indices(7, 1)
    ok_idx = np.array_equal(condensed_index(j, i, 7), np.arange(len(D)))
    draws = boot_draws(7, 50, 1)
    ok_rng = all((k >= 0).all() and (k < len(D)).all() for k in draws)
    check("(iv) condensed indexing and bootstrap draws", ok_idx and ok_rng)

    try:
        import paths
        from step40_common import load_reps  # noqa: E402
        reps, rois = load_reps("sub-01")
        v1 = reps[:, :, rois["V1"]]
        avg = v1.mean(axis=1)
        print(f"  [report] sub-01 V1 Spearman(b2, b3) = {spearman(pdist(avg), cv_sq_euclid(v1)):.3f}")
    except Exception as e:      # data not available on this machine
        print(f"  [report] sub-01 V1 b2 versus b3 skipped: {type(e).__name__}: {e}")

    if FAILS:
        print(f"  {len(FAILS)} failure(s) of {N} checks")
        sys.exit(1)
    print(f"  all {N} checks passed")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "scripts"))
    main()

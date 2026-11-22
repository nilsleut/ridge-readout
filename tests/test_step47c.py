"""
test_step47c.py
===============
Checks for step 47c (amendments/2026-11-22_step47c-predictions.md, config_step47c.yaml).
All must pass before the step 47c run.

  (i)  synthetic session with known m and d_r (10 runs of 82, noise centred per run): components
       recovers m, the run means and d_r; V0-V5 equal the hand-built patterns; d_par + d_perp = d_r,
       d_perp orthogonal to m_hat, d_par parallel to m_hat
  (ii) the Teil 0 gates on the data: m + d_r = run mean (<= 1e-12); Delta(V3) reproduces step 47
       run-centred V1 (+0.0070) and Delta(V0) the paper cell (+0.0304), both to 1e-12

Run: python tests/test_step47c.py [--only i,ii]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import nsd_common as C                                       # noqa: E402,F401
import step47c_decompose as D7                               # noqa: E402

FAILS, N = [], 0


def check(name, ok, info=""):
    global N
    N += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)


def test_synthetic():
    rng = np.random.default_rng(3)
    v, n_runs, per = 150, 10, 82
    m0 = rng.normal(1.0, 0.5, size=v)
    d0 = rng.normal(size=(n_runs, v))
    d0 -= d0.mean(axis=0)                                    # equal run sizes: run means average to m0
    lab = np.repeat(np.arange(1, n_runs + 1), per)
    rng.shuffle(lab)
    noise = rng.normal(size=(len(lab), v))
    for r in range(1, n_runs + 1):
        noise[lab == r] -= noise[lab == r].mean(axis=0)
    S = m0 + d0[lab - 1] + noise
    runs, m, M, D = D7.components(S, lab)
    e1 = max(np.abs(m - m0).max(), np.abs(D - d0).max(), np.abs(M - (m0 + d0)).max())
    check("(i) components recover m, run means and d_r", e1 < 1e-12, f"max |d| {e1:.1e}")

    m_hat, s, D_par, D_perp = D7.split(D, m)
    u = m0 / np.linalg.norm(m0)
    par0 = np.outer(d0 @ u, u)
    e2 = max(np.abs(D_par + D_perp - D).max(), np.abs(D_perp @ m_hat).max(),
             np.abs(D_par - np.outer(D_par @ m_hat, m_hat)).max(), np.abs(D_par - par0).max(),
             np.abs(s - d0 @ u).max())
    check("(i) d_par + d_perp = d_r, d_perp orthogonal, d_par parallel to m_hat", e2 < 1e-12, f"max |d| {e2:.1e}")

    sub = rng.choice(len(lab), 300, replace=False)
    X, lx = S[sub], lab[sub]
    P = D7.variants(X, lx, runs, m, D)
    k = lx - 1
    hand = {"V0": m0 + d0[k] + noise[sub], "V1": d0[k] + noise[sub], "V2": m0 + noise[sub], "V3": noise[sub],
            "V4": noise[sub] + par0[k], "V5": noise[sub] + (d0 - par0)[k]}
    e3 = max(np.abs(P[x] - hand[x]).max() for x in D7.VARIANTS)
    check("(i) V0-V5 equal the hand-built patterns", e3 < 1e-12, f"max |d| {e3:.1e}")
    try:
        D7.variants(X, np.full(len(lx), 99), runs, m, D)
        check("(i) unknown run label stops", False)
    except SystemExit:
        check("(i) unknown run label stops", True)


def test_gates():
    import step44_why as W
    import step47_runs as R
    from scipy.spatial.distance import pdist
    ci = W.crossrun_index()
    models = W.things720_models("V1")
    br = {"V0": [], "V3": []}
    gap = 0.0
    for s in R.SUBJ:
        S, lab_s, X, lab_x, local, _ = D7.subject_data(s)
        c = local["V1"]
        runs, m, M, D = D7.components(S[:, c], lab_s)
        gap = max(gap, float(np.abs(m + D - M).max()))
        P = D7.variants(X[:, c], lab_x, runs, m, D)
        for v in br:
            br[v].append(pdist(P[v], "correlation"))
    check("(ii) m + d_r = run mean", gap <= 1e-12, f"max |d| {gap:.1e}")
    ref = pd.read_csv(REPO / "results" / "step47" / "run_centring.csv")
    ref = ref[(ref.kind == "delta") & (ref.roi == "V1")].set_index("variant").delta
    d3, _ = W.delta_on(models, br["V3"], ci)
    d0, _ = W.delta_on(models, br["V0"], ci)
    check("(ii) Delta(V3) = step 47 run-centred", abs(d3 - ref["run-centred"]) <= 1e-12,
          f"{d3:+.9f} vs {ref['run-centred']:+.9f}")
    check("(ii) Delta(V0) = paper cell", abs(d0 - ref["raw"]) <= 1e-12 and abs(d0 - 0.030445938) < 1e-8,
          f"{d0:+.9f} vs {ref['raw']:+.9f}")


if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    for key, fn in (("i", test_synthetic), ("ii", test_gates)):
        if only is None or key in only:
            fn()
    print(f"  all {N} checks passed" if not FAILS else f"  {len(FAILS)} of {N} checks FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)

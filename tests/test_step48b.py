"""
test_step48b.py
===============
Checks for step 48b (amendments/2026-11-22_step48b-predictions.md, config_step48b.yaml).
All must pass before the step 48b run.

  (i)   synthetic patterns with a common profile c * p and per-pattern amplitude b_i (as step 48
        test (ii)), model patterns built the same way from an independent profile and noise with the
        same amplitudes (a pure amplitude effect):
        - Spearman(correlation-distance RDM, A sum form 2 - g_i - g_j) rises with c (near 0 at c = 0)
        - the two-sided partial (model on A^M, brain on A, sum form) removes the pure amplitude effect
          (|rho| < 0.1 while the plain rho > 0.5 for c >= 1)
        - amp_rdm forms on a hand example; the two-sided partial with A^M = A equals step 44 partial_rho
  (i-doc) difference form |g_i - g_j| on the same patterns: the values are printed, not required
        (the simulation before the amendment: no monotone rise, the partial does not remove the effect)
  (ii)  the gate: Delta(T-raw, THINGS-720, V1), Delta(N-raw, k = 1, V1) and g, g^M of step 48
        (results/step48) recomputed by the step 48b code path (about 4 min)

Run: python tests/test_step48b.py [--only i,ii]
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import nsd_common as C                                       # noqa: E402,F401
import step44_why as W                                       # noqa: E402
import step48_amplitude as AM                                # noqa: E402
import step48b_dominance as D                                # noqa: E402

FAILS, N = [], 0


def check(name, ok, info=""):
    global N
    N += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)


def synth(c, seed=0, n=200, v=300, vm=400):
    from scipy.spatial.distance import pdist
    rng = np.random.default_rng(seed)
    p, q = rng.normal(1.0, 1.0, v), rng.normal(1.0, 1.0, vm)
    b = rng.uniform(0.5, 2.0, n)
    X = c * p[None, :] + b[:, None] * rng.normal(size=(n, v))
    F = c * q[None, :] + b[:, None] * rng.normal(size=(n, vm))
    return pdist(X, "correlation"), pdist(F, "correlation"), AM.gain(X), AM.gain(F)


def test_synthetic():
    from scipy.stats import spearmanr
    cs = (0.0, 0.5, 1.0, 2.0, 4.0)
    res = {c: synth(c) for c in cs}
    dom = [spearmanr(res[c][0], D.amp_rdm(res[c][2], "sum"))[0] for c in cs]
    check("(i) Spearman(RDM, A sum) rises with the profile", bool(np.all(np.diff(dom) > 0)) and abs(dom[0]) < 0.15
          and dom[-1] > 0.9, " ".join(f"{x:+.3f}" for x in dom))
    plain = [D.rho_rank(res[c][1], res[c][0]) for c in cs]
    part = [D.partial2(res[c][1], res[c][0], D.amp_rdm(res[c][3], "sum"), D.amp_rdm(res[c][2], "sum")) for c in cs]
    ok = all(plain[i] > 0.5 and abs(part[i]) < 0.1 for i, c in enumerate(cs) if c >= 1.0)
    check("(i) two-sided partial (sum form) removes a pure amplitude effect", ok,
          "plain " + " ".join(f"{x:+.3f}" for x in plain) + " | partial " + " ".join(f"{x:+.3f}" for x in part))
    g = np.array([0.2, 0.5, 0.9])
    ok = np.allclose(D.amp_rdm(g, "sum"), [1.3, 0.9, 0.6]) and np.allclose(D.amp_rdm(g, "diff"), [0.3, 0.7, 0.4])
    check("(i) amplitude RDM forms on a hand example", ok, f"sum {D.amp_rdm(g, 'sum')} diff {D.amp_rdm(g, 'diff')}")
    m, b_, g_, _ = res[2.0]
    a = D.amp_rdm(g_, "sum")
    x, y = D.partial2(m, b_, a, a), W.partial_rho(m, b_, [a])
    check("(i) two-sided partial with A^M = A equals step 44 partial_rho", abs(x - y) < 1e-12, f"{x:+.6f} {y:+.6f}")
    # documentation only: the difference form
    dd = [spearmanr(res[c][0], D.amp_rdm(res[c][2], "diff"))[0] for c in cs]
    pd_ = [D.partial2(res[c][1], res[c][0], D.amp_rdm(res[c][3], "diff"), D.amp_rdm(res[c][2], "diff")) for c in cs]
    print("  [doc] (i) difference form |g_i - g_j| (not required): Spearman(RDM, A diff) "
          + " ".join(f"{x:+.3f}" for x in dd) + " | partial " + " ".join(f"{x:+.3f}" for x in pd_)
          + f" | monotone rise {bool(np.all(np.diff(dd) > 0))}, partial removes the effect "
          + f"{all(abs(pd_[i]) < 0.1 for i, c in enumerate(cs) if c >= 1.0)}")


def test_gate():
    res = D.gate()
    check("(ii) gate (Delta T-raw, Delta N-raw k = 1, g, g^M of step 48 within 1e-12)",
          all(dev <= D.TOL for _, _, dev in res.values()), f"max |d| {max(dev for _, _, dev in res.values()):.1e}")


if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    for key, fn in (("i", test_synthetic), ("ii", test_gate)):
        if only is None or key in only:
            fn()
    print(f"  all {N} checks passed" if not FAILS else f"  {len(FAILS)} of {N} checks FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)

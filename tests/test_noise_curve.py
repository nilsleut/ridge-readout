"""
test_noise_curve.py
===================
Checks for scripts/step41_noise_curve.py (step 41, part N). All must pass before the run.

  (i)   k = 12 reproduces the correlation-vs-b1 rho values of results/step40/cvrsa.csv
        exactly (max |d rho| <= 1e-9; all subjects, ROIs, conditions, seeds, raw and z)
  (ii)  synthetic fixed signal plus noise: the variance of the brain RDM entries over draws
        decreases with k (k = 1, 2, 3, 4, 6)
  (iii) rank_z products equal cvrsa.spearman
  (iv)  session draws: 20 per k (1 for k = 12), k distinct sessions each, reproducible

Run: python tests/test_noise_curve.py   (part (i) needs the THINGS data and step 40 results)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import step40_common as S                                                          # noqa: E402
from step41_noise_curve import brain_for, model_rdms, rank_z, session_draws        # noqa: E402
from cvrsa import spearman                                                          # noqa: E402

FAILS, N = [], 0


def check(name, ok, info=""):
    global N
    N += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAILS.append(name)


def main():
    print("noise curve checks")
    d = session_draws()
    ok = all(len(v) == (1 if k == 12 else 20) and all(len(set(s)) == k for s in v) for k, v in d.items())
    check("(iv) session draws", ok and all(np.array_equal(a, b) for k in d for a, b in zip(d[k], session_draws()[k])))

    rng = np.random.default_rng(3)
    mu = rng.normal(size=(15, 40))
    reps = mu[:, None, :] + rng.normal(size=(15, 12, 40)) * 2.0
    var = []
    for k in (1, 2, 3, 4, 6):
        ent = np.stack([brain_for(reps, np.sort(rng.choice(12, k, replace=False)))["b2"] for _ in range(200)])
        var.append(float(ent.var(axis=0).mean()))
    check("(ii) RDM entry variance decreases with k", all(a > b for a, b in zip(var, var[1:])),
          "variances " + ", ".join(f"{v:.3f}" for v in var))

    a, b = rng.normal(size=300), rng.normal(size=300)
    d3 = abs(float((rank_z(a) @ rank_z(b).T)[0, 0]) - spearman(a, b))
    check("(iii) rank_z equals cvrsa.spearman", d3 <= 1e-12, f"|d| = {d3:.1e}")

    ref = pd.read_csv(REPO / "results" / "step40" / "cvrsa.csv")
    ref = ref[(ref.kind == "rho") & (ref.brain_rdm == "b1")]
    models = model_rdms()
    sess = session_draws()[12][0]
    worst = 0.0
    for s in ["sub-01", "sub-02", "sub-03"]:
        reps_s, rois = S.load_reps(s)
        for roi in ("V1", "LOC"):
            b1 = brain_for(reps_s[:, :, rois[roi]], sess)["b1"]
            for (lab, si, norm, dist), m in models.items():
                if dist != "correlation":
                    continue
                q = ref[(ref.subject == s) & (ref.roi == roi) & (ref.condition == lab) & (ref.seed == si)
                        & (ref.model_units == norm) & (ref.model_distance == dist)]
                assert len(q) == 1
                worst = max(worst, abs(spearman(m, b1) - float(q.rho.iloc[0])))
    check("(i) k = 12 reproduces step 40 b1 rho", worst <= 1e-9, f"max |d rho| = {worst:.1e}")

    if FAILS:
        print(f"  {len(FAILS)} failure(s) of {N} checks")
        sys.exit(1)
    print(f"  all {N} checks passed")


if __name__ == "__main__":
    main()

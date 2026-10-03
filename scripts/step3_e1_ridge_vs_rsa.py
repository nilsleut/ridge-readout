"""
step3_e1_ridge_vs_rsa.py
========================
E1: ridge versus RSA on STORED values only (no new fits). EXPLORATORY, sub-01 only.

Sources: results/tolerance_all_rungs.csv (ridge, min-MSE, median r over 501 reliable
V1 voxels, g in {1, 4, 8}) and results/step2/rsa_501.csv (RSA, cross-run pairs,
voxel_set = reliable, g = 1).

  a) per condition: mean +- SD over 5 seeds; paired differences Random-BP,
     Random-BNcal, BNcal-BP, FA-BP, PC-BP, STDP-BP (mean +- SD over seeds)
  b) Spearman rho over the 30 checkpoints, ridge (g = 1) vs RSA: overall, and
     within condition (5 seeds) averaged over the 6 conditions
  c) P1, fixed before the run (scale hypothesis): |Random - BNcal| at ridge g = 1
     <= 0.010. Caveat: BN calibration also shifts offsets before the ReLU, so ridge
     with feature standardization is not fully invariant to it.

Output: results/step3/ridge_vs_rsa.csv
"""
import step3_common as C          # noqa: I001

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R, BP, BN = "Random Weights", "Backprop", "Random Weights (BN-calibrated)"
DIFFS = [(R, BP), (R, BN), (BN, BP), ("Feedback Alignment", BP), ("Predictive Coding", BP),
         ("STDP", BP)]


def main():
    tol = pd.read_csv(C.REPO / "results" / "tolerance_all_rungs.csv")
    tol = tol[tol.selector == "min-MSE"]
    rsa = pd.read_csv(C.REPO / "results" / "step2" / "rsa_501.csv")
    rsa = rsa[rsa.voxel_set == "reliable"]
    tabs = {f"ridge_g{g}": tol[tol.g == g].pivot(index="seed", columns="rule", values="r_med_reliable")
            for g in (1, 4, 8)}
    tabs["rsa_501_g1"] = rsa.pivot(index="seed_idx", columns="rule", values="rho")

    rows = []
    for metric, p in tabs.items():
        assert p.shape == (5, 6), (metric, p.shape)
        for rule in p.columns:
            rows.append(dict(kind="condition", metric=metric, quantity=rule,
                             mean=p[rule].mean(), sd=p[rule].std(ddof=1), n_seeds=5,
                             **{f"seed{i}": v for i, v in enumerate(p[rule].values)}))
        for a, b in DIFFS:
            d = p[a] - p[b]
            rows.append(dict(kind="difference", metric=metric, quantity=f"{a} - {b}",
                             mean=d.mean(), sd=d.std(ddof=1), n_seeds=5, seeds_positive=int((d > 0).sum()),
                             **{f"seed{i}": v for i, v in enumerate(d.values)}))

    ridge, rs = tabs["ridge_g1"].stack(), tabs["rsa_501_g1"].stack()
    rho_all = spearmanr(ridge.values, rs.reindex(ridge.index).values).statistic
    within = {c: spearmanr(tabs["ridge_g1"][c], tabs["rsa_501_g1"][c]).statistic for c in tabs["ridge_g1"]}
    rows.append(dict(kind="spearman", metric="ridge_g1 vs rsa_501_g1", quantity="overall (30 checkpoints)",
                     mean=rho_all))
    for c, v in within.items():
        rows.append(dict(kind="spearman", metric="ridge_g1 vs rsa_501_g1", quantity=f"within {c} (5 seeds)",
                         mean=v))
    rows.append(dict(kind="spearman", metric="ridge_g1 vs rsa_501_g1", quantity="within-condition mean",
                     mean=float(np.mean(list(within.values())))))

    p1 = abs((tabs["ridge_g1"][R] - tabs["ridge_g1"][BN]).mean())
    p1_rsa = (tabs["rsa_501_g1"][R] - tabs["rsa_501_g1"][BN]).mean()
    rows.append(dict(kind="P1", metric="ridge_g1", quantity="|Random - BNcal| <= 0.010", mean=p1,
                     passed=bool(p1 <= 0.010), note=f"RSA (501) Random - BNcal = {p1_rsa:+.4f}"))

    df = C.stamp(pd.DataFrame(rows))
    C.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.OUT / "ridge_vs_rsa.csv", index=False)
    pd.set_option("display.width", 200)
    print(df[["kind", "metric", "quantity", "mean", "sd"]].to_string(index=False))


if __name__ == "__main__":
    main()

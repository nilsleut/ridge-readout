"""
step6_robustness.py
===================
Robustness of the raw-vs-z result (descriptive, no rule; amendments/2026-10-09_step6-
predictions.md). EXPLORATORY.

Cells: V1 (Conv1) at NATIVE 32 and NATIVE 224, LOC (Conv3) at NATIVE 224; GAP features
from STEP5_GAP_DIR, Random Weights and Backprop, 5 seeds.
  * feature normalisation {raw, center, scale, z} (per channel over the 720 stimuli,
    dead channels -> 0) x model distance {correlation, cosine, euclidean}; fMRI RDMs as
    in the paper (correlation distance on the betas z-scored over all 9,840 trials);
  * fMRI side re-z-scored per voxel over the 720 stimuli (instead of over all trials),
    combined with feature normalisation raw and z, distance correlation.
Statistic, pair set and bootstrap as in step 5 (per subject, cross-run, 1000 paired
draws). Per cell: Random - BP (V1) or BP - Random (LOC), mean over seeds, 95% CI.
Check: the paper cell (raw, correlation, fMRI as stored) equals step 5 to 1e-12.

Output: results/step6/robustness.csv
"""
import step3_common as C          # noqa: I001

import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from step6_common import bootstrap, ci, fmri_tri, gap_path, normalise

R_, BP = "Random Weights", "Backprop"
CELLS = [("V1", "Conv1", 32, R_, BP), ("V1", "Conv1", 224, R_, BP), ("LOC", "Conv3", 224, BP, R_)]
NORMS = ["raw", "center", "scale", "z"]
DISTS = ["correlation", "cosine", "euclidean"]
OUT = C.REPO / "results" / "step6"
CACHE = C.REPO / "data" / "step6_boot_robust.npy"


def variants():
    for n in NORMS:
        for d in DISTS:
            yield n, d, "all-trials"
    for n in ("raw", "z"):
        yield n, "correlation", "720-stimuli"


def main():
    brains = {}
    for roi in ("V1", "LOC"):
        brains[f"{roi}|all-trials"] = fmri_tri(roi)
        brains[f"{roi}|720-stimuli"] = fmri_tri(roi, rezscore=True)
    items = [(roi, layer, px, rule, si, n, d, fz) for roi, layer, px, a, b in CELLS for n, d, fz in variants()
             for rule in (a, b) for si in C.SEED_IDX]
    print(f"{len(items)} bootstrap items", flush=True)
    D = bootstrap([(lambda l=layer, p=px, r=rule, s=si, n=n, d=d: pdist(normalise(np.load(gap_path("NATIVE", p, s, r, l)), n), d))
                   for roi, layer, px, rule, si, n, d, fz in items],
                  [f"{it[0]}|{it[7]}" for it in items], brains, CACHE)
    col = {it: k for k, it in enumerate(items)}

    s5 = pd.read_csv(C.REPO / "results" / "step5" / "resolution_rsa.csv")
    s5 = s5[s5.kind == "rho"]
    dev = 0.0
    for (roi, layer, px, rule, si, n, d, fz), k in col.items():
        if d == "correlation" and fz == "all-trials" and n in ("raw", "z"):
            ref = s5[(s5.arm == "NATIVE") & (s5.res == px) & (s5.layer == layer) & (s5.roi == roi)
                     & (s5.rule == rule) & (s5.seed_idx == si) & (s5.variant == n)].rho.iloc[0]
            dev = max(dev, abs(D[0, k] - ref))
    print(f"paper-convention cells vs step 5: max |d| {dev:.2e}")
    assert dev < 1e-12

    rows = []
    for roi, layer, px, a, b in CELLS:
        for n, d, fz in variants():
            sd = np.stack([D[:, col[(roi, layer, px, a, si, n, d, fz)]] - D[:, col[(roi, layer, px, b, si, n, d, fz)]]
                           for si in C.SEED_IDX], axis=1)
            p, lo, hi = ci(sd.mean(axis=1))
            rows.append(dict(roi=roi, layer=layer, arm="NATIVE", res=px, quantity=f"{a} - {b}",
                             feature_norm=n, distance=d, fmri_zscore=fz, value=p, ci_lo=lo, ci_hi=hi,
                             ci_excludes_0=bool(lo > 0 or hi < 0), seeds_positive=int((sd[0] > 0).sum())))
    df = C.stamp(pd.DataFrame(rows))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "robustness.csv", index=False)
    os.remove(CACHE)
    pd.set_option("display.width", 220)
    print(df.drop(columns=["config_hash", "git_head", "arm", "layer"]).to_string(index=False))


if __name__ == "__main__":
    main()

"""
step25_bootnorm.py
==================
Step 25, block B (amendments/2026-10-16_step25-predictions.md): normalisation statistics
inside the bootstrap. EXPLORATORY.

B1 (code reading, reported in reports/STEP25_REVIEW_CONTROLS.md): in steps 5-9 every unit
mean, SD, feature transformation F, voxel normalisation B, mean profile and norm is
computed once on the 720 stimuli; the draws only resample pairs of these fixed RDMs
(boot_pairs). Hence B2 runs.

B2: for the first 200 of the paper's resamples (boot_idx), the statistics are
re-estimated from the resampled rows of the draw (duplicates kept), the RDM is rebuilt
on all 720 stimuli with these statistics, and the pairs of the draw (boot_pairs:
cross-run, self-pairs dropped) are ranked as before. Cells, V1, own CNN at NATIVE 224,
Random Weights - Backprop (mean over seeds of the per-seed difference, rho = mean over
subjects):
    (z, correlation, paper)        model z-scored with the draw's unit means and SDs
                                   (SD < 1e-8 -> 0); betas fixed (B = paper)
    (center, euclidean, paper)     model centred with the draw's unit means; betas fixed
    (center, correlation, center720)  model centred and betas centred per voxel with the
                                   draw's means
BN1: for all three cells the classification of the 95% CI relative to 0 (above, below,
includes) equals that of the fixed-normalisation interval of step 7
(results/step7/multiverse.csv). For comparison the fixed-normalisation interval over the
same 200 draws is taken from data/step7_draws (written by step 7), if present.

Output: results/step25/bootnorm.csv
"""
import step3_common as C          # noqa: I001  (pins BLAS threads)

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from common_up_pairs import ranks, run_labels
from step6_common import EPS, gap_path, roi_betas
from v12_lift import boot_idx, boot_pairs

SUBJ = ["sub-01", "sub-02", "sub-03"]
R_, BP = "Random Weights", "Backprop"
CELLS = [("z", "correlation", "paper"), ("center", "euclidean", "paper"), ("center", "correlation", "center720")]
N_DRAWS = 200
OUT = C.REPO / "results" / "step25"
_W = {}


def norm_with(X, rows, how):
    mu = X[rows].mean(axis=0)
    if how == "center":
        return X - mu
    sd = X[rows].std(axis=0)
    dead = sd < EPS
    Z = (X - mu) / np.where(dead, 1.0, sd)
    Z[:, dead] = 0.0
    return Z


def _init(boot):
    feats = {(rule, si): np.load(gap_path("NATIVE", 224, si, rule, "Conv1")).astype(np.float64)
             for rule in (R_, BP) for si in C.SEED_IDX}
    _W.update(boot=boot, runs=run_labels(), feats=feats, betas={s: roi_betas(s, "V1") for s in SUBJ})


def _one(b):
    rows = np.arange(720) if b < 0 else _W["boot"][b]
    idx = boot_pairs(b, _W["boot"], _W["runs"])
    out = {}
    for f, d, bb in CELLS:
        br = []
        for s in SUBJ:
            B = _W["betas"][s]
            Bn = B if bb == "paper" else norm_with(B, rows, "center")
            br.append(ranks(pdist(Bn, d)[idx]))
        for (rule, si), X in _W["feats"].items():
            mv = ranks(pdist(norm_with(X, rows, f), d)[idx])
            out[(f, d, bb, rule, si)] = float(np.mean([mv @ x for x in br]))
    return b, out


def cls(lo, hi):
    return "above" if lo > 0 else ("below" if hi < 0 else "includes")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    boot = boot_idx()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(boot,)) as ex:
        res = dict(ex.map(_one, range(-1, N_DRAWS), chunksize=4))
    print(f"{N_DRAWS} draws done ({time.time() - t0:.0f}s)", flush=True)
    s7 = pd.read_csv(C.REPO / "results" / "step7" / "multiverse.csv", low_memory=False)
    rows, ok = [], True
    for f, d, bb in CELLS:
        gap = np.array([np.mean([res[b][(f, d, bb, R_, si)] - res[b][(f, d, bb, BP, si)] for si in C.SEED_IDX])
                        for b in range(-1, N_DRAWS)])
        p, lo, hi = gap[0], float(np.quantile(gap[1:], .025)), float(np.quantile(gap[1:], .975))
        ref = s7[(s7.kind == "cell") & (s7.F == f) & (s7.D == d) & (s7.B == bb) & (s7.roi == "V1")
                 & (s7.quantity == f"gap NATIVE 224: {R_} - {BP}")].iloc[0]
        fixed200 = (np.nan, np.nan)
        npz = C.REPO / "data" / "step7_draws" / f"cell_{f}_{d}_{bb}.npz"
        if npz.exists():
            g7 = np.load(npz)["V1_NATIVE_224"].mean(axis=1)
            fixed200 = (float(np.quantile(g7[1:N_DRAWS + 1], .025)), float(np.quantile(g7[1:N_DRAWS + 1], .975)))
        same = cls(lo, hi) == cls(ref.ci_lo, ref.ci_hi)
        ok &= same
        rows.append(dict(kind="cell", F=f, D=d, B=bb, roi="V1", res="NATIVE_224", quantity=f"{R_} - {BP}",
                         value=float(p), ci_lo=lo, ci_hi=hi, n_draws=N_DRAWS, ci_class=cls(lo, hi),
                         fixed_value=ref.value, fixed_ci_lo=ref.ci_lo, fixed_ci_hi=ref.ci_hi,
                         fixed_ci_class=cls(ref.ci_lo, ref.ci_hi), fixed200_ci_lo=fixed200[0],
                         fixed200_ci_hi=fixed200[1], class_unchanged=bool(same),
                         point_dev=float(abs(p - ref.value))))
    rows.append(dict(kind="rule", quantity="BN1: CI classification unchanged in all three cells", passed=bool(ok)))
    out = C.stamp(pd.DataFrame(rows))
    out.to_csv(OUT / "bootnorm.csv", index=False)
    print(out.drop(columns=["config_hash", "git_head"]).to_string(index=False))


if __name__ == "__main__":
    main()

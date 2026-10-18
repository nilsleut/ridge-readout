"""
step25_weighted_angle.py
========================
Step 25, block W (amendments/2026-10-16_step25-predictions.md): weighted angular term and
component fit. EXPLORATORY, descriptive, no decision rule.

With F = center (step 9 normf/components, imported): n_i = |Xc_i|, A_ij = 1 - cos theta_ij,
N_ij = |n_i - n_j|, W_ij = 2 n_i n_j (1 - cos theta_ij), the angular term of
|Xc_i - Xc_j|^2 = N_ij^2 + W_ij. The fMRI side gets the same decomposition (betas centred
per voxel over the 720 stimuli, as in step 9).

(1) RSA of W with the matching brain W: own CNN V1 Random - BP and LOC BP - Random at
    NATIVE 224; ImageNet V1 untrained (mean of 3) - trained and LOC trained - untrained at
    224; bootstrap CIs (paper's 1000 resamples, pair set per draw). N and A of step 9 are
    reported next to them from results/step9/separation.csv.
(2) Rank correlations between the N, A and W RDMs over the cross-run pairs (identity
    sample): own CNN V1 NATIVE 224, each of the 30 checkpoints, then the mean over
    checkpoints; fMRI V1 per subject.
(3) Component fit, own CNN V1 NATIVE 224, Random Weights and Backprop, 5 seeds each:
    per subject, R^2 of an OLS regression of the ranks of the fMRI RDM on the ranks of the
    model N RDM, the A RDM, and both; then the mean over subjects. Operationalisation
    (fixed here, before the run): "the fMRI RDM" is the standard fMRI RDM of steps 2-9,
    correlation distance over all ROI voxels on the arm_b_out betas (B = paper). With
    centred unit-norm ranks, R^2(N) = r_N^2, R^2(A) = r_A^2 and
    R^2(N + A) = (r_N^2 + r_A^2 - 2 r_N r_A r_NA) / (1 - r_NA^2), with r_N, r_A the rank
    correlations with the fMRI RDM and r_NA between the two model components. Contrast:
    Random - BP (mean over seeds of the per-seed difference), bootstrap CIs.

Output: results/step25/weighted_angle.csv
"""
import step3_common as C          # noqa: I001  (pins BLAS threads)

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from common_up_pairs import pair_set, ranks, run_labels
from step6_common import ci, roi_betas
from step9_separation import IMN, IMN_CONDS, UNTR, components, imn_X, normf, own_X
from v12_lift import N_BOOT, RULES, TRI, boot_idx, boot_pairs

SUBJ = ["sub-01", "sub-02", "sub-03"]
ROIS = ["V1", "LOC"]
R_, BP = "Random Weights", "Backprop"
COMPS = ["N", "A", "W"]
OUT = C.REPO / "results" / "step25"
MOD_NPY = C.REPO / "data" / "step25_w_models.npy"
BR_NPY = C.REPO / "data" / "step25_w_brains.npy"
_W = {}


def nwa(X):
    """N, A, W RDM vectors (TRI order) for F = center."""
    comp = components(normf(X, "center"))
    n = comp["n"]
    return {"N": comp["norm"], "A": comp["angle"], "W": 2.0 * n[TRI[0]] * n[TRI[1]] * comp["angle"]}


def load(src):
    if src[0] == "own":
        return own_X("NATIVE", 224, src[1], src[2], src[3])
    return imn_X(src[1], src[2], 224, src[3])


def model_items():
    it = []
    for roi in ROIS:
        for rule in (R_, BP):
            for si in C.SEED_IDX:
                for k in COMPS:
                    it.append((k, ("own", rule, si, roi)))
        for m in IMN:
            for c in IMN_CONDS:
                it.append(("W", ("imn", m, c, roi)))
    return it


def _init(boot, pairs):
    _W.update(boot=boot, runs=run_labels(), M=np.load(MOD_NPY, mmap_mode="r"),
              B=np.load(BR_NPY, mmap_mode="r"), pairs=pairs)


def _one(b):
    idx = boot_pairs(b, _W["boot"], _W["runs"])
    Rb = np.stack([ranks(np.asarray(_W["B"][k][idx], float)) for k in range(_W["B"].shape[0])])
    Rm = np.stack([ranks(np.asarray(_W["M"][k][idx], float)) for k in range(_W["M"].shape[0])])
    na = np.array([Rm[i] @ Rm[j] for i, j in _W["pairs"]])
    return b, (Rm @ Rb.T, na)


def r2(rn, ra, rna):
    return rn ** 2, ra ** 2, (rn ** 2 + ra ** 2 - 2 * rn * ra * rna) / (1 - rna ** 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, keep = pair_set(None, run_labels())

    # brain vectors: W (and N, A for the correlations) per ROI and subject, F = center;
    # the standard fMRI RDM (correlation distance, B = paper) for the component fit
    brain_items, brain_vec = [], []
    for roi in ROIS:
        for s in SUBJ:
            for k, v in nwa(roi_betas(s, roi)).items():
                brain_items.append((k, roi, s)); brain_vec.append(v)
    for s in SUBJ:
        brain_items.append(("fmri", "V1", s)); brain_vec.append(pdist(roi_betas(s, "V1"), "correlation"))
    bcol = {it: j for j, it in enumerate(brain_items)}

    # (2) rank correlations N, A, W (identity sample, cross-run pairs)
    rows = []
    cors = []
    for rule in RULES:
        for si in C.SEED_IDX:
            c = {k: ranks(v[keep]) for k, v in nwa(own_X("NATIVE", 224, rule, si, "V1")).items()}
            cors.append(dict(NA=c["N"] @ c["A"], NW=c["N"] @ c["W"], AW=c["A"] @ c["W"]))
    cd = pd.DataFrame(cors).mean()
    for k in ("NA", "NW", "AW"):
        rows.append(dict(kind="rankcorr", model="own CNN", roi="V1", res="NATIVE_224", subject="mean of 30 checkpoints",
                         quantity=f"Spearman {k[0]}-{k[1]}", value=float(cd[k])))
    for s in SUBJ:
        c = {k: ranks(brain_vec[bcol[(k, "V1", s)]][keep]) for k in COMPS}
        for a, b in (("N", "A"), ("N", "W"), ("A", "W")):
            rows.append(dict(kind="rankcorr", model="fMRI", roi="V1", subject=s, quantity=f"Spearman {a}-{b}",
                             value=float(c[a] @ c[b])))

    # (1), (3) bootstrap
    items = model_items()
    mcol = {it: j for j, it in enumerate(items)}
    pairs = [(mcol[("N", ("own", rule, si, "V1"))], mcol[("A", ("own", rule, si, "V1"))])
             for rule in (R_, BP) for si in C.SEED_IDX]
    pcol = {(rule, si): j for j, (rule, si) in enumerate([(rule, si) for rule in (R_, BP) for si in C.SEED_IDX])}
    t1 = time.time()
    _ = [ranks(v[keep]) for v in brain_vec[:10]]
    t_rank = (time.time() - t1) / 10
    print(f"bootstrap runtime estimate: {(len(items) + len(brain_items)) * t_rank * 1001 / 8 * 1.5 / 3600:.2f} h",
          flush=True)
    Mm = np.lib.format.open_memmap(MOD_NPY, mode="w+", dtype=np.float64, shape=(len(items), len(TRI[0])))
    for j, (k, src) in enumerate(items):
        Mm[j] = nwa(load(src))[k]
    Mm.flush(); del Mm
    Bm = np.lib.format.open_memmap(BR_NPY, mode="w+", dtype=np.float64, shape=(len(brain_items), len(TRI[0])))
    for j, v in enumerate(brain_vec):
        Bm[j] = v
    Bm.flush(); del Bm
    t2 = time.time()
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(boot_idx(), pairs)) as ex:
        res = dict(ex.map(_one, range(-1, N_BOOT), chunksize=8))
    os.remove(MOD_NPY); os.remove(BR_NPY)
    print(f"bootstrap done ({time.time() - t2:.0f}s)", flush=True)
    G = np.stack([res[b][0] for b in range(-1, N_BOOT)])      # (1001, items, brains)
    NA = np.stack([res[b][1] for b in range(-1, N_BOOT)])     # (1001, 10)

    def rho(k, src, bk="same"):
        roi = src[-1]
        cols = [bcol[(k if bk == "same" else bk, roi, s)] for s in SUBJ]
        return G[:, mcol[(k, src)], cols]                      # (1001, 3)

    # (1) W contrasts
    for roi in ROIS:
        a, b = (R_, BP) if roi == "V1" else (BP, R_)
        d = np.mean([rho("W", ("own", a, si, roi)).mean(axis=1) - rho("W", ("own", b, si, roi)).mean(axis=1)
                     for si in C.SEED_IDX], axis=0)
        p, lo, hi = ci(d)
        rows.append(dict(kind="contrast", model="own CNN", roi=roi, res="NATIVE_224", component="W",
                         quantity=f"{a} - {b}", value=p, ci_lo=lo, ci_hi=hi))
        for m in IMN:
            u = np.mean([rho("W", ("imn", m, c, roi)).mean(axis=1) for c in UNTR], axis=0)
            t = rho("W", ("imn", m, "trained", roi)).mean(axis=1)
            p, lo, hi = ci(u - t if roi == "V1" else t - u)
            rows.append(dict(kind="contrast", model=m, roi=roi, res="R224", component="W",
                             quantity="untrained - trained" if roi == "V1" else "trained - untrained",
                             value=p, ci_lo=lo, ci_hi=hi))

    # (3) component fit, own CNN V1 NATIVE 224
    def fit(rule, si):
        rn = rho("N", ("own", rule, si, "V1"), "fmri")       # (1001, 3)
        ra = rho("A", ("own", rule, si, "V1"), "fmri")
        rna = NA[:, pcol[(rule, si)]][:, None]
        return [x.mean(axis=1) for x in r2(rn, ra, rna)]       # three (1001,) arrays
    fits = {(rule, si): fit(rule, si) for rule in (R_, BP) for si in C.SEED_IDX}
    for j, lab in enumerate(("R2 N", "R2 A", "R2 N+A")):
        for rule in (R_, BP):
            rows.append(dict(kind="r2", model="own CNN", roi="V1", res="NATIVE_224", component=lab, quantity=rule,
                             value=float(np.mean([fits[(rule, si)][j][0] for si in C.SEED_IDX]))))
        d = np.mean([fits[(R_, si)][j] - fits[(BP, si)][j] for si in C.SEED_IDX], axis=0)
        p, lo, hi = ci(d)
        rows.append(dict(kind="r2_contrast", model="own CNN", roi="V1", res="NATIVE_224", component=lab,
                         quantity="Random Weights - Backprop", value=p, ci_lo=lo, ci_hi=hi))

    # N and A of step 9, for reference
    s9 = pd.read_csv(C.REPO / "results" / "step9" / "separation.csv", low_memory=False)
    s9 = s9[(s9.kind == "contrast") & (s9.F == "center") & (s9.component.isin(["norm", "angle"]))]
    for r in s9.itertuples():
        rows.append(dict(kind="step9_reference", model=r.model, roi=r.roi, res=str(r.res),
                         component={"norm": "N", "angle": "A"}[r.component], quantity=r.quantity,
                         value=r.value, ci_lo=r.ci_lo, ci_hi=r.ci_hi))
    out = C.stamp(pd.DataFrame(rows))
    out.to_csv(OUT / "weighted_angle.csv", index=False)
    print(out[out.kind != "step9_reference"].drop(columns=["config_hash", "git_head"]).to_string(index=False))


if __name__ == "__main__":
    main()

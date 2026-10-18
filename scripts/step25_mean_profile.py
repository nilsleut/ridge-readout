"""
step25_mean_profile.py
======================
Step 25, block M (amendments/2026-10-16_step25-predictions.md): mean-profile ablation of
raw correlation-distance RSA. EXPLORATORY.

For raw GAP features X (720 x units): mu = unit means over the 720 stimuli,
mu_c = mu - mean(mu), u = mu_c / |mu_c|; x~_i = row i centred across units;
c_i = <x~_i, u>; r_i = x~_i - c_i u.
    full   correlation distance on x_i (the paper cell raw/correlation/paper)
    ortho  correlation distance on r_i
    par    |c_i - c_j|
The same decomposition is applied to the betas (B = paper: arm_b_out betas, voxel means
over the 720 stimuli), and each model component is compared with the matching brain
component. Statistics are computed once on the 720 stimuli (step 7 convention).

RSA, pairs and bootstrap as in steps 7-9, imported: per subject on all ROI voxels,
cross-run pairs, rho = ranks(model[P]) . ranks(fMRI[P]), mean over subjects; the paper's
1000 resamples with the pair set recomputed per draw (step6_common.bootstrap).

Gate: "full" must reproduce results/step7/multiverse.csv (kind rho, raw/correlation/
paper) and results/step8/generalization.csv (same cell) to 1e-9, else STOP.

Point estimates: own CNN, all 30 checkpoints, NATIVE 32, NATIVE 224, UPSAMPLED 224, V1
(Conv1) and LOC (Conv3); ImageNet models (data/step8_gap), all conditions, R in
{224, 448} (vit_b_16: 224), V1.
Bootstrap: own CNN V1 Random Weights and Backprop at the three conditions, LOC at NATIVE
224; ImageNet V1 at 224 and 448. Contrasts: own V1 Random - BP, own LOC BP - Random;
ImageNet V1 untrained (mean of 3) - trained. Slopes: own S = gap(NATIVE 224) -
gap(NATIVE 32); ImageNet gap(448) - gap(224) (vit_b_16 none).
Rules M1-M4 and the interpretation rule as written in the amendment; M2 is applied to the
point estimate of S_ortho.

Output: results/step25/mean_profile.csv
"""
import step3_common as C          # noqa: I001  (pins BLAS threads)

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from common_up_pairs import pair_set, ranks, run_labels
from step6_common import bootstrap, ci, gap_path, roi_betas
from step8_generalization import feat as imn_feat
from v12_lift import RULES, TRI

VARIANTS = ["full", "ortho", "par"]
SUBJ = ["sub-01", "sub-02", "sub-03"]
ROIS = {"V1": "Conv1", "LOC": "Conv3"}
OWN_CONDS = [("NATIVE", 32), ("NATIVE", 224), ("UPSAMPLED", 224)]
R_, BP = "Random Weights", "Backprop"
IMN = ["alexnet", "vgg11_bn", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
IMN_CONDS = ["trained", "untrained_s0", "untrained_s1", "untrained_s2"]
UNTR = IMN_CONDS[1:]
S_PAPER = 0.0311
OUT = C.REPO / "results" / "step25"
CACHE = C.REPO / "data" / "step25_mean_profile.npy"
_W = {}


def imn_res(m):
    return [224] if m == "vit_b_16" else [224, 448]


def decompose(X):
    """{'full', 'ortho', 'par'}: TRI-order RDM vectors of the mean-profile decomposition."""
    X = np.asarray(X, np.float64)
    mu = X.mean(axis=0)
    mu_c = mu - mu.mean()
    u = mu_c / np.linalg.norm(mu_c)
    Xt = X - X.mean(axis=1, keepdims=True)
    c = Xt @ u
    R = Xt - np.outer(c, u)
    out = {"full": pdist(X, "correlation"), "ortho": pdist(R, "correlation"),
           "par": pdist(c[:, None], "cityblock")}
    for k, v in out.items():
        if not np.isfinite(v).all():
            raise SystemExit(f"STOP: non-finite {k} RDM")
    return out


def own_X(arm, px, rule, si, roi):
    return np.load(gap_path(arm, px, si, rule, ROIS[roi])).astype(np.float64)


def src_X(src):
    if src[0] == "own":
        _, arm, px, rule, si, roi = src
        return own_X(arm, px, rule, si, roi)
    _, m, c, R, roi = src
    return imn_feat(m, c, R, roi)


def sources():
    out = [("own", arm, px, rule, si, roi) for arm, px in OWN_CONDS for rule in RULES
           for si in C.SEED_IDX for roi in ROIS]
    out += [("imn", m, c, R, "V1") for m in IMN for R in imn_res(m) for c in IMN_CONDS]
    return out


def _init_pts(keep, brain):
    _W.update(keep=keep, brain=brain)


def _pts_one(src):
    comp = decompose(src_X(src))
    roi = src[-1]
    rows = []
    for k, v in comp.items():
        mv = ranks(v[_W["keep"]])
        ps = [float(mv @ _W["brain"][(k, roi, s)]) for s in SUBJ]
        base = (dict(source="own", model="own CNN", cond=src[3], res=f"{src[1]}_{src[2]}", seed_idx=src[4])
                if src[0] == "own" else dict(source="imn", model=src[1], cond=src[2], res=f"R{src[3]}", seed_idx=np.nan))
        rows.append(dict(kind="rho", **base, roi=roi, variant=k, rho=float(np.mean(ps)),
                         **{f"rho_{s}": p for s, p in zip(SUBJ, ps)}))
    return rows


def boot_items():
    it = []
    for k in VARIANTS:
        for arm, px in OWN_CONDS:
            for rule in (R_, BP):
                for si in C.SEED_IDX:
                    it.append((k, ("own", arm, px, rule, si, "V1")))
        for rule in (R_, BP):
            for si in C.SEED_IDX:
                it.append((k, ("own", "NATIVE", 224, rule, si, "LOC")))
        for m in IMN:
            for R in imn_res(m):
                for c in IMN_CONDS:
                    it.append((k, ("imn", m, c, R, "V1")))
    return it


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, keep = pair_set(None, run_labels())

    brain_tri, brain = {}, {}
    for roi in ROIS:
        for s in SUBJ:
            for k, v in decompose(roi_betas(s, roi)).items():
                brain_tri[(k, roi, s)] = v
                brain[(k, roi, s)] = ranks(v[keep])

    t0 = time.time()
    srcs = sources()
    with ProcessPoolExecutor(max_workers=8, initializer=_init_pts, initargs=(keep, brain)) as ex:
        pts = pd.DataFrame([r for ch in ex.map(_pts_one, srcs, chunksize=4) for r in ch])
    print(f"point estimates: {len(srcs)} matrices ({time.time() - t0:.0f}s)", flush=True)

    # ── gate: 'full' reproduces the stored raw/correlation/paper cells ──
    s7 = pd.read_csv(C.REPO / "results" / "step7" / "multiverse.csv", low_memory=False)
    s7 = s7[(s7.kind == "rho") & (s7.F == "raw") & (s7.D == "correlation") & (s7.B == "paper")]
    s7 = s7.set_index(["roi", "arm", "res", "rule", "seed_idx"]).rho
    s8 = pd.read_csv(C.REPO / "results" / "step8" / "generalization.csv", low_memory=False)
    s8 = s8[(s8.kind == "rho") & (s8.F == "raw") & (s8.D == "correlation") & (s8.B == "paper")]
    s8 = s8.set_index(["model", "cond", "R", "roi"]).rho
    devs = []
    for r in pts[pts.variant == "full"].itertuples():
        if r.source == "own":
            arm, px = r.res.rsplit("_", 1)
            ref = s7[(r.roi, arm, float(px), r.cond, float(r.seed_idx))]
        else:
            ref = s8[(r.model, r.cond, str(int(r.res[1:])), r.roi)]
        devs.append(abs(r.rho - float(ref)))
    gate_dev = max(devs)
    print(f"gate: 'full' vs stored raw/correlation/paper cells, max |d| {gate_dev:.2e} over {len(devs)}", flush=True)
    if not gate_dev <= 1e-9:
        raise SystemExit("STOP: reproduction gate failed")

    # ── bootstrap ──
    items = boot_items()
    brains = {(k, roi): [brain_tri[(k, roi, s)] for s in SUBJ] for k in VARIANTS for roi in ROIS}
    t1 = time.time()
    _ = [ranks(v[keep]) for v in list(brain_tri.values())[:10]]
    t_rank = (time.time() - t1) / 10
    est = (len(items) + len(brains) * 3) * t_rank * 1001 / 8 * 1.5 / 3600
    print(f"bootstrap runtime estimate: {est:.2f} h ({len(items)} model vectors)", flush=True)
    t2 = time.time()
    G = bootstrap([(lambda s=src, k=k: decompose(src_X(s))[k]) for k, src in items],
                  [(k, src[-1]) for k, src in items], brains, CACHE)
    os.remove(CACHE)
    print(f"bootstrap done ({time.time() - t2:.0f}s)", flush=True)
    col = {it: j for j, it in enumerate(items)}

    def rho(k, src):
        return G[:, col[(k, src)]]

    dev = 0.0
    for (k, src), j in col.items():
        q = pts[(pts.variant == k) & (pts.roi == src[-1])]
        if src[0] == "own":
            q = q[(q.source == "own") & (q.res == f"{src[1]}_{src[2]}") & (q.cond == src[3]) & (q.seed_idx == src[4])]
        else:
            q = q[(q.model == src[1]) & (q.cond == src[2]) & (q.res == f"R{src[3]}")]
        dev = max(dev, abs(G[0, j] - q.rho.iloc[0]))
    print(f"identity draw vs point estimates: max |d| {dev:.2e}", flush=True)
    assert dev < 1e-12

    def own_gap(k, arm, px, roi="V1"):
        a, b = (R_, BP) if roi == "V1" else (BP, R_)
        return np.mean([rho(k, ("own", arm, px, a, si, roi)) - rho(k, ("own", arm, px, b, si, roi))
                        for si in C.SEED_IDX], axis=0)

    def own_pos(k, arm, px, roi="V1"):
        a, b = (R_, BP) if roi == "V1" else (BP, R_)
        return int(sum(rho(k, ("own", arm, px, a, si, roi))[0] > rho(k, ("own", arm, px, b, si, roi))[0]
                       for si in C.SEED_IDX))

    def imn_gap(k, m, R):
        u = np.mean([rho(k, ("imn", m, c, R, "V1")) for c in UNTR], axis=0)
        return u - rho(k, ("imn", m, "trained", R, "V1"))

    rows = []
    for k in VARIANTS:
        for arm, px in OWN_CONDS:
            p, lo, hi = ci(own_gap(k, arm, px))
            rows.append(dict(kind="contrast", model="own CNN", roi="V1", variant=k, res=f"{arm}_{px}",
                             quantity="Random Weights - Backprop", value=p, ci_lo=lo, ci_hi=hi,
                             seeds_positive=own_pos(k, arm, px)))
        p, lo, hi = ci(own_gap(k, "NATIVE", 224, "LOC"))
        rows.append(dict(kind="contrast", model="own CNN", roi="LOC", variant=k, res="NATIVE_224",
                         quantity="Backprop - Random Weights", value=p, ci_lo=lo, ci_hi=hi,
                         seeds_positive=own_pos(k, "NATIVE", 224, "LOC")))
        p, lo, hi = ci(own_gap(k, "NATIVE", 224) - own_gap(k, "NATIVE", 32))
        rows.append(dict(kind="slope", model="own CNN", roi="V1", variant=k, res="NATIVE_224 - NATIVE_32",
                         quantity="Random Weights - Backprop", value=p, ci_lo=lo, ci_hi=hi))
        for m in IMN:
            for R in imn_res(m):
                p, lo, hi = ci(imn_gap(k, m, R))
                rows.append(dict(kind="contrast", model=m, roi="V1", variant=k, res=f"R{R}",
                                 quantity="untrained - trained", value=p, ci_lo=lo, ci_hi=hi))
            if m != "vit_b_16":
                p, lo, hi = ci(imn_gap(k, m, 448) - imn_gap(k, m, 224))
                rows.append(dict(kind="slope", model=m, roi="V1", variant=k, res="R448 - R224",
                                 quantity="untrained - trained", value=p, ci_lo=lo, ci_hi=hi))
    res = pd.DataFrame(rows)

    def get(kind, model, variant, r, roi="V1"):
        q = res[(res.kind == kind) & (res.model == model) & (res.variant == variant) & (res.res == r) & (res.roi == roi)]
        assert len(q) == 1
        return q.iloc[0]

    o224 = get("contrast", "own CNN", "ortho", "NATIVE_224")
    oS = get("slope", "own CNN", "ortho", "NATIVE_224 - NATIVE_32")
    p224 = get("contrast", "own CNN", "par", "NATIVE_224")
    pS = get("slope", "own CNN", "par", "NATIVE_224 - NATIVE_32")
    m1 = bool(o224.ci_hi <= 0.010)
    m2 = bool(oS.value <= 0.5 * S_PAPER)
    m3 = bool(p224.value > 0 and p224.ci_lo > 0 and pS.ci_lo > 0)
    n4 = int(sum(get("contrast", m, "ortho", "R224").ci_hi <= 0.010 for m in IMN))
    m4 = n4 >= 4
    if m1 and m2 and m3:
        interp = ("M1, M2 and M3 met: in these data the raw-RSA advantage and its slope are carried by the "
                  "component parallel to the mean profile and absent from the orthogonal remainder")
    else:
        parts = []
        if not (m1 and m2):
            parts.append("M1 or M2 not met: the parallel component is not necessary")
        if not m3:
            parts.append("M3 not met: it is not sufficient")
        interp = "; ".join(parts)
    rules = pd.DataFrame([
        dict(kind="rule", quantity="M1 own ortho V1 gap NATIVE 224 upper CI <= 0.010", value=o224.ci_hi, passed=m1),
        dict(kind="rule", quantity="M2 own ortho slope <= 0.5 * S_paper (0.01555)", value=oS.value, passed=m2),
        dict(kind="rule", quantity="M3 own par V1 gap NATIVE 224 CI above 0 and par slope CI above 0",
             value=p224.value, passed=m3),
        dict(kind="rule", quantity="M4 ImageNet ortho V1 gap 224 upper CI <= 0.010 in >= 4 of 6 models",
             value=n4, passed=m4),
        dict(kind="interpretation", quantity="interpretation rule M", note=interp),
        dict(kind="check", quantity="gate full vs stored cells max |d|", value=gate_dev),
        dict(kind="check", quantity="identity draw vs point estimates max |d|", value=dev),
    ])
    out = pd.concat([pts, res, rules], ignore_index=True)
    out = C.stamp(out)
    out.to_csv(OUT / "mean_profile.csv", index=False)
    print(rules[["quantity", "value", "passed"]].to_string(index=False))
    print(interp)


if __name__ == "__main__":
    main()

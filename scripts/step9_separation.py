"""
step9_separation.py
===================
SEP1-SEP4 and the descriptive points (a)-(e)
(amendments/2026-10-10_step9-separation-predictions.md). EXPLORATORY.
No new extraction: own CNN from STEP5_GAP_DIR, ImageNet models from data/step8_gap,
fMRI from arm_b_out.

Decomposition of X (720 x units) after F (center, or z = centre and scale; applied to
model units AND fMRI voxels, operationalisation note 1):
    |Xc_i - Xc_j|^2 = (n_i - n_j)^2 + 2 n_i n_j (1 - cos theta_ij)
    norm RDM  N_ij = |n_i - n_j|          (pdist of n, cityblock)
    angle RDM A_ij = 1 - cos theta_ij     (pdist of Xc, cosine)
Component RSA: ranks(model component[P]) . ranks(fMRI component[P]) per subject on the
cross-run pairs of the draw, then the mean (steps 2-8 statistic).

Contrasts: own CNN V1 Random - BP, LOC BP - Random (5 seeds); ImageNet V1
untrained (mean of 3) - trained, LOC trained - untrained.

(a) SEP1 gate first (max relative error < 1e-8, all feature matrices and fMRI, both F).
(b) point estimates: own CNN (all 6 rules x 5 seeds; NATIVE 32, NATIVE 224, UPSAMPLED 224)
    and ImageNet (all conditions; 112, 224, 448; vit_b_16 224 only), V1 and LOC, F in
    {center, z}, component in {norm, angle}; plus the full euclidean RDM on Xc
    (F = center) for check (e).
(c) bootstrap (paper's 1000 paired resamples): contrasts at the primary resolution (own
    224, ImageNet 224), all models, both ROIs, both F, both components; V1 slopes of the
    contrast (own 224 - 32, ImageNet 448 - 224 without vit_b_16).
(d) runtime estimate x 1.5; above 4 h the bootstrap covers F = center only.
Rules: SEP2 (V1 norm, center: contrast > 0 with CI above 0 in own CNN and >= 4/6 ImageNet),
SEP3 (V1 angle, center: contrast < 0 with CI below 0 in own CNN and >= 4/6), SEP4 (LOC
angle, center: trained - untrained CI above 0 in own CNN and >= 5/6), interpretation rule.
Descriptive: (a) F = z, (b) slopes, (c) |S_gen| under z/correlation from step 8,
(d) Pearson of n_i with luminance / RMS contrast, (e) full euclidean RSA vs stored cells.

Output: results/step9/separation.csv, separation_slopes.csv, norm_stimulus_stats.csv
"""
import step3_common as C          # noqa: I001  (pins BLAS threads)

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from common_up_pairs import pair_set, ranks, run_labels
from step6_common import EPS, gap_path, roi_betas
from step7_multiverse import ci
from v12_lift import N_BOOT, RULES, TRI, boot_idx, boot_pairs

FS = ["center", "z"]
COMPS = ["norm", "angle"]
SUBJ = ["sub-01", "sub-02", "sub-03"]
ROIS = {"V1": "Conv1", "LOC": "Conv3"}
OWN_CONDS = [("NATIVE", 32), ("NATIVE", 224), ("UPSAMPLED", 224)]
R_, BP = "Random Weights", "Backprop"
IMN = ["alexnet", "vgg11_bn", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
IMN_CONDS = ["trained", "untrained_s0", "untrained_s1", "untrained_s2"]
UNTR = IMN_CONDS[1:]
S8 = C.REPO / "data" / "step8_gap"
OUT = C.REPO / "results" / "step9"
MOD_NPY = C.REPO / "data" / "step9_models.npy"
BR_NPY = C.REPO / "data" / "step9_brains.npy"
_W = {}


def imn_res(m):
    return [224] if m == "vit_b_16" else [112, 224, 448]


def normf(X, f):
    X = np.asarray(X, np.float64)
    Xc = X - X.mean(axis=0)
    if f == "z":
        sd = X.std(axis=0)
        Xc = np.where(sd < EPS, 0.0, Xc / np.where(sd < EPS, 1.0, sd))
    return Xc


def components(Xc):
    n = np.linalg.norm(Xc, axis=1)
    return {"norm": pdist(n[:, None], "cityblock"), "angle": pdist(Xc, "cosine"), "n": n}


def identity_err(Xc):
    c = components(Xc)
    n = c["n"]
    lhs = pdist(Xc, "sqeuclidean")
    rhs = c["norm"] ** 2 + 2 * n[TRI[0]] * n[TRI[1]] * c["angle"]
    return float(np.max(np.abs(lhs - rhs) / np.maximum(lhs, 1e-300))), int((n == 0).sum())


def own_X(arm, px, rule, si, roi):
    return np.load(gap_path(arm, px, si, rule, ROIS[roi])).astype(np.float64)


def imn_X(m, c, R, roi):
    return np.load(S8 / m / c / f"R{R}" / f"{roi}.npy").astype(np.float64)


def sources():
    """(source, key..., loader) for every model feature matrix used."""
    out = []
    for arm, px in OWN_CONDS:
        for rule in RULES:
            for si in C.SEED_IDX:
                for roi in ROIS:
                    out.append(("own", f"{arm}_{px}", rule, si, roi))
    for m in IMN:
        for R in imn_res(m):
            for c in IMN_CONDS:
                for roi in ROIS:
                    out.append(("imn", m, c, R, roi))
    return out


def load(src):
    if src[0] == "own":
        arm, px = src[1].rsplit("_", 1)
        return own_X(arm, int(px), src[2], src[3], src[4])
    return imn_X(src[1], src[2], src[3], src[4])


def _gate_one(src):
    X = load(src)
    return [(src, f) + identity_err(normf(X, f)) for f in FS]


def _init_pts(keep, brain):
    _W.update(keep=keep, brain=brain)


def _pts_one(src):
    X = load(src)
    roi = src[4]
    rows = []
    for f in FS:
        comp = components(normf(X, f))
        comp.pop("n")
        if f == "center":
            comp["full_euclidean"] = pdist(normf(X, f), "euclidean")
        for k, v in comp.items():
            mv = ranks(v[_W["keep"]])
            ps = [float(mv @ _W["brain"][(f, k, roi, s)]) for s in SUBJ]
            rows.append(dict(kind="rho", source=src[0], model="own CNN" if src[0] == "own" else src[1],
                             cond=src[2] if src[0] == "own" else src[2],
                             res=src[1] if src[0] == "own" else f"R{src[3]}", seed_idx=src[3] if src[0] == "own" else None,
                             roi=roi, F=f, component=k, rho=float(np.mean(ps)),
                             **{f"rho_{s}": p for s, p in zip(SUBJ, ps)}))
    return rows


def _init_boot(boot):
    _W.update(boot=boot, runs=run_labels(), M=np.load(MOD_NPY, mmap_mode="r"), B=np.load(BR_NPY, mmap_mode="r"))


def _boot_one(b):
    idx = boot_pairs(b, _W["boot"], _W["runs"])
    Rb = np.stack([ranks(np.asarray(_W["B"][k][idx], float)) for k in range(_W["B"].shape[0])])
    Rm = np.stack([ranks(np.asarray(_W["M"][k][idx], float)) for k in range(_W["M"].shape[0])])
    return b, Rm @ Rb.T


def stimulus_stats():
    import torchvision.transforms as T
    from PIL import Image
    from common_up_pairs import stim_order
    from v12_lift import namespaces, transform
    ns12, ns11 = namespaces()
    imgs = [Image.open(ns12["find_img"](s)).convert("RGB") for s in stim_order()]

    def stats(pre):
        Y = np.stack([(lambda x: 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2])(pre(im).numpy().astype(np.float64)) for im in imgs])
        Y = Y.reshape(len(imgs), -1)
        return Y.mean(axis=1), Y.std(axis=1)
    out = {}
    for px in (32, 224):
        tf = transform(ns12, ns11, "NATIVE", px)
        out[("own", px)] = stats(T.Compose(tf.transforms[:-1]))
    for R in (112, 224, 448):
        out[("imn", R)] = stats(T.Compose([T.Resize(R), T.CenterCrop(R), T.ToTensor()]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    srcs = sources()

    # ── (a) SEP1 gate ──
    with ProcessPoolExecutor(max_workers=8) as ex:
        gate = [r for ch in ex.map(_gate_one, srcs, chunksize=8) for r in ch]
    fmri = []
    for f in FS:
        for roi in ROIS:
            for s in SUBJ:
                fmri.append((("fmri", roi, s), f) + identity_err(normf(roi_betas(s, roi), f)))
    gate += fmri
    worst = max(g[2] for g in gate)
    zeros = sum(g[3] for g in gate)
    print(f"SEP1: {len(gate)} matrices, max relative error {worst:.2e}, zero-norm rows {zeros}", flush=True)
    if not worst < 1e-8:
        raise SystemExit("STOP: SEP1 identity check failed")

    # ── brains ──
    _, _, keep = pair_set(None, run_labels())
    brain_tri, brain = {}, {}
    for f in FS:
        for roi in ROIS:
            for s in SUBJ:
                Bc = normf(roi_betas(s, roi), f)
                comp = components(Bc)
                comp.pop("n")
                if f == "center":
                    comp["full_euclidean"] = pdist(Bc, "euclidean")
                for k, v in comp.items():
                    brain_tri[(f, k, roi, s)] = v
                    brain[(f, k, roi, s)] = ranks(v[keep])

    # ── (b) point estimates ──
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8, initializer=_init_pts, initargs=(keep, brain)) as ex:
        pts = pd.DataFrame([r for ch in ex.map(_pts_one, srcs, chunksize=4) for r in ch])
    print(f"point estimates: {len(srcs)} matrices ({time.time() - t0:.0f}s)", flush=True)

    # ── (e) full euclidean vs stored cells ──
    s7 = pd.read_csv(C.REPO / "results" / "step7" / "multiverse.csv", low_memory=False)
    s7 = s7[(s7.kind == "rho") & (s7.F == "center") & (s7.D == "euclidean") & (s7.B == "center720")]
    s7 = s7.set_index(["roi", "arm", "res", "rule", "seed_idx"]).rho
    s8 = pd.read_csv(C.REPO / "results" / "step8" / "generalization.csv", low_memory=False)
    s8 = s8[(s8.kind == "rho") & (s8.F == "center") & (s8.D == "euclidean") & (s8.B == "center720")]
    s8 = s8.set_index(["model", "cond", "R", "roi"]).rho
    fe = pts[pts.component == "full_euclidean"]
    devs = []
    for r in fe.itertuples():
        if r.source == "own":
            arm, px = r.res.rsplit("_", 1)
            ref = s7[(r.roi, arm, float(px), r.cond, float(r.seed_idx))]
        else:
            ref = s8[(r.model, r.cond, str(int(r.res[1:])), r.roi)]   # R is a string column in step 8
        devs.append(abs(r.rho - float(ref)))
    dev_e = max(devs)
    print(f"(e) full euclidean on Xc vs stored center/euclidean/center720: max |d| {dev_e:.2e} over {len(devs)}", flush=True)

    # ── (c) bootstrap ──
    t1 = time.time()
    _ = [ranks(v[keep]) for v in list(brain_tri.values())[:10]]
    t_rank = (time.time() - t1) / 10
    fboot = list(FS)

    def items_for(fl):
        it = []
        for f in fl:
            for k in COMPS:
                for arm, px in (("NATIVE", 32), ("NATIVE", 224)):
                    for rule in (R_, BP):
                        for si in C.SEED_IDX:
                            it.append((f, k, "V1", ("own", f"{arm}_{px}", rule, si, "V1")))
                for rule in (R_, BP):
                    for si in C.SEED_IDX:
                        it.append((f, k, "LOC", ("own", "NATIVE_224", rule, si, "LOC")))
                for m in IMN:
                    for R in ([224] if m == "vit_b_16" else [224, 448]):
                        for c in IMN_CONDS:
                            it.append((f, k, "V1", ("imn", m, c, R, "V1")))
                    for c in IMN_CONDS:
                        it.append((f, k, "LOC", ("imn", m, c, 224, "LOC")))
        return it
    items = items_for(fboot)
    bitems = [(f, k, roi, s) for f in fboot for k in COMPS for roi in ROIS for s in SUBJ]
    est = (len(items) + len(bitems)) * t_rank * (N_BOOT + 1) / 8 * 1.5 / 3600
    print(f"bootstrap runtime estimate: {est:.2f} h ({len(items)} model, {len(bitems)} brain vectors)", flush=True)
    if est > 4:
        fboot = ["center"]
        items = items_for(fboot)
        bitems = [(f, k, roi, s) for f in fboot for k in COMPS for roi in ROIS for s in SUBJ]
        print("above 4 h: bootstrap for F = center only; F = z point estimates only", flush=True)

    Mm = np.lib.format.open_memmap(MOD_NPY, mode="w+", dtype=np.float64, shape=(len(items), len(TRI[0])))
    for j, (f, k, roi, src) in enumerate(items):
        Mm[j] = components(normf(load(src), f))[k]
    Mm.flush(); del Mm
    Bm = np.lib.format.open_memmap(BR_NPY, mode="w+", dtype=np.float64, shape=(len(bitems), len(TRI[0])))
    for j, it in enumerate(bitems):
        Bm[j] = brain_tri[it]
    Bm.flush(); del Bm
    with ProcessPoolExecutor(max_workers=8, initializer=_init_boot, initargs=(boot_idx(),)) as ex:
        res = dict(ex.map(_boot_one, range(-1, N_BOOT), chunksize=8))
    G = np.stack([res[b] for b in range(-1, N_BOOT)])
    os.remove(MOD_NPY); os.remove(BR_NPY)
    mc = {it: j for j, it in enumerate(items)}
    bc = {it: j for j, it in enumerate(bitems)}

    def rho(f, k, roi, src):
        return G[:, mc[(f, k, roi, src)], [bc[(f, k, roi, s)] for s in SUBJ]].mean(axis=1)

    dev = 0.0
    for (f, k, roi, src), j in mc.items():
        q = pts[(pts.F == f) & (pts.component == k) & (pts.roi == roi) & (pts.source == src[0])]
        if src[0] == "own":
            q = q[(q.res == src[1]) & (q.cond == src[2]) & (q.seed_idx == src[3])]
        else:
            q = q[(q.model == src[1]) & (q.cond == src[2]) & (q.res == f"R{src[3]}")]
        dev = max(dev, abs(rho(f, k, roi, src)[0] - q.rho.iloc[0]))
    print(f"identity draw vs point estimates: max |d| {dev:.2e}", flush=True)
    assert dev < 1e-12

    def own_contrast(f, k, roi, cond):
        a, b = (R_, BP) if roi == "V1" else (BP, R_)
        return np.mean([rho(f, k, roi, ("own", cond, a, si, roi)) - rho(f, k, roi, ("own", cond, b, si, roi))
                        for si in C.SEED_IDX], axis=0)

    def imn_contrast(f, k, roi, m, R):
        u = np.mean([rho(f, k, roi, ("imn", m, c, R, roi)) for c in UNTR], axis=0)
        t = rho(f, k, roi, ("imn", m, "trained", R, roi))
        return u - t if roi == "V1" else t - u

    rows, srows = [], []
    for f in fboot:
        for k in COMPS:
            for roi in ROIS:
                lab = "Random - BP" if roi == "V1" else "BP - Random"
                p, lo, hi = ci(own_contrast(f, k, roi, "NATIVE_224"))
                rows.append(dict(kind="contrast", model="own CNN", roi=roi, F=f, component=k, res="NATIVE 224",
                                 quantity=lab, value=p, ci_lo=lo, ci_hi=hi))
                for m in IMN:
                    p, lo, hi = ci(imn_contrast(f, k, roi, m, 224))
                    rows.append(dict(kind="contrast", model=m, roi=roi, F=f, component=k, res="224",
                                     quantity="untrained - trained" if roi == "V1" else "trained - untrained",
                                     value=p, ci_lo=lo, ci_hi=hi))
            p, lo, hi = ci(own_contrast(f, k, "V1", "NATIVE_224") - own_contrast(f, k, "V1", "NATIVE_32"))
            srows.append(dict(kind="slope", model="own CNN", roi="V1", F=f, component=k, quantity="contrast(224) - contrast(32)",
                              value=p, ci_lo=lo, ci_hi=hi))
            for m in IMN[:-1]:
                p, lo, hi = ci(imn_contrast(f, k, "V1", m, 448) - imn_contrast(f, k, "V1", m, 224))
                srows.append(dict(kind="slope", model=m, roi="V1", F=f, component=k, quantity="contrast(448) - contrast(224)",
                                  value=p, ci_lo=lo, ci_hi=hi))
    con = pd.DataFrame(rows)
    con["ci_excludes_0"] = (con.ci_lo > 0) | (con.ci_hi < 0)
    slopes = pd.DataFrame(srows)
    slopes["ci_excludes_0"] = (slopes.ci_lo > 0) | (slopes.ci_hi < 0)

    # ── rules ──
    def sel(roi, k, f="center"):
        return con[(con.roi == roi) & (con.component == k) & (con.F == f)].set_index("model")

    v = []
    v.append(dict(kind="rule", quantity="SEP1 identity max relative error", value=worst, passed=True,
                  note=f"{len(gate)} matrices, zero-norm rows {zeros}"))
    q = sel("V1", "norm")
    own2 = bool(q.loc["own CNN", "ci_lo"] > 0)
    n2 = int((q.loc[IMN, "ci_lo"] > 0).sum())
    v.append(dict(kind="rule", quantity="SEP2 V1 norm (center): own CNN CI > 0 and >= 4/6 ImageNet", value=n2,
                  passed=bool(own2 and n2 >= 4), note=f"own CNN CI > 0: {own2}"))
    q = sel("V1", "angle")
    own3 = bool(q.loc["own CNN", "ci_hi"] < 0)
    n3 = int((q.loc[IMN, "ci_hi"] < 0).sum())
    v.append(dict(kind="rule", quantity="SEP3 V1 angle (center): own CNN CI < 0 and >= 4/6 ImageNet", value=n3,
                  passed=bool(own3 and n3 >= 4), note=f"own CNN CI < 0: {own3}"))
    q = sel("LOC", "angle")
    own4 = bool(q.loc["own CNN", "ci_lo"] > 0)
    n4 = int((q.loc[IMN, "ci_lo"] > 0).sum())
    v.append(dict(kind="rule", quantity="SEP4 LOC angle (center): own CNN CI > 0 and >= 5/6 ImageNet", value=n4,
                  passed=bool(own4 and n4 >= 5), note=f"own CNN CI > 0: {own4}"))
    both = bool(v[1]["passed"] and v[2]["passed"])
    v.append(dict(kind="interpretation", quantity="SEP2 and SEP3 met", value=float(both), passed=both,
                  note=("V1 advantage of untrained networks under euclidean distance sits in the gain magnitude; "
                        "trained networks lead in geometry") if both else
                  "V1 ranking remains unidentifiable, without attribution to gain or geometry"))
    v.append(dict(kind="check_e", quantity="full euclidean on Xc vs stored center/euclidean/center720", value=dev_e,
                  passed=bool(dev_e <= 1e-6), note=f"{len(devs)} rows compared"))

    # (c) |S_gen| under z/correlation from step 8
    s8b = pd.read_csv(C.REPO / "results" / "step8" / "generalization.csv", low_memory=False)
    s8b = s8b[(s8b.kind == "boot") & (s8b.quantity == "S_gen") & (s8b.D == "correlation") & (s8b.B == "paper")]
    for r in s8b.itertuples():
        v.append(dict(kind="S_gen_step8", quantity=f"|S_gen| {r.F}/correlation/paper", model=r.model, value=abs(r.value),
                      ci_lo=r.ci_lo, ci_hi=r.ci_hi))

    # ── (d) n_i vs stimulus statistics ──
    st = stimulus_stats()
    drows = []
    for px in (32, 224):
        lum, con_ = st[("own", px)]
        for rule in RULES:
            vals = [np.corrcoef(components(normf(own_X("NATIVE", px, rule, si, "V1"), "center"))["n"], x)[0, 1]
                    for si in C.SEED_IDX for x in (lum, con_)]
            drows.append(dict(source="own CNN", model=rule, cond=rule, res=px,
                              corr_luminance=float(np.mean(vals[0::2])), corr_rms_contrast=float(np.mean(vals[1::2]))))
    for m in IMN:
        for R in imn_res(m):
            lum, con_ = st[("imn", R)]
            for c in IMN_CONDS:
                n = components(normf(imn_X(m, c, R, "V1"), "center"))["n"]
                drows.append(dict(source="ImageNet", model=m, cond=c, res=R,
                                  corr_luminance=float(np.corrcoef(n, lum)[0, 1]),
                                  corr_rms_contrast=float(np.corrcoef(n, con_)[0, 1])))
    lum, con_ = st[("own", 224)]
    for s in SUBJ:
        n = components(normf(roi_betas(s, "V1"), "center"))["n"]
        drows.append(dict(source="fMRI V1", model=s, cond="voxels", res=224,
                          corr_luminance=float(np.corrcoef(n, lum)[0, 1]), corr_rms_contrast=float(np.corrcoef(n, con_)[0, 1])))
    nst = pd.DataFrame(drows)

    pts["kind"] = "rho"
    C.stamp(pd.concat([pts, con, pd.DataFrame(v)], ignore_index=True)).to_csv(OUT / "separation.csv", index=False)
    C.stamp(slopes).to_csv(OUT / "separation_slopes.csv", index=False)
    C.stamp(nst).to_csv(OUT / "norm_stimulus_stats.csv", index=False)

    pd.set_option("display.width", 230)
    print(con.to_string(index=False))
    print(slopes.to_string(index=False))
    print(pd.DataFrame(v).to_string(index=False))
    print(nst.to_string(index=False))


if __name__ == "__main__":
    main()

"""
step8_generalization.py
=======================
GEN1-GEN4 (amendments/2026-10-09_step8-generalization-predictions.md). EXPLORATORY.

Six torchvision models (data/step8_gap), trained vs untrained (3 init seeds),
R in {112, 224, 448} (vit_b_16: 224 only, operationalisation note 1). RSA convention and
the definitions of F, D, B, gain and c_gain as in step 7 (functions reused from
scripts/step7_multiverse.py and scripts/step7_gain.py): per subject on all ROI voxels,
cross-run pairs, D on both sides, mean over subjects.

Gap (V1) = mean over the 3 untrained seeds of rho - rho(trained); LOC: trained - untrained.
S_gen = Gap(448) - Gap(224).

Point estimates: all 36 cells F x D x B, V1 and LOC, all R. Bootstrap (paper's 1000
paired resamples) for the 4 cells (raw, correlation, paper), (z, correlation, paper),
(raw, euclidean, paper), (center, correlation, center720): V1 gap at every R and S_gen;
LOC trained - untrained at R = 224.

GEN1  Spearman(c_gain, raw RSA in raw/correlation/paper, V1) over all model x condition x R
      cells (n = 64; vit_b_16 at 224 only) >= 0.6
GEN2  >= 4 of 6 models: at R = 224 two of the 4 bootstrap cells have gap CIs excluding 0
      in opposite directions
GEN3  >= 5 of 6 models: LOC trained - untrained at 224 > 0 with CI above 0 in all 4 cells
GEN4  every model whose S_gen in raw/correlation/paper has a CI excluding 0 satisfies
      |S_gen(z/correlation/paper)| <= 0.5 |S_gen(raw/correlation/paper)|; none -> not testable
Descriptive: per-model sign table at 224 over the 36 cells (4 bootstrap cells by CI, the
others by point estimate); correlation of the model gain axis (V1 layer) with stimulus
mean luminance and RMS contrast at R = 224, trained vs untrained.

Output: results/step8/generalization.csv, generalization_signs.csv, generalization_gain.csv
"""
import step3_common as C          # noqa: I001  (pins BLAS threads)

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from common_up_pairs import pair_set, ranks, run_labels
from step6_common import normalise
from step7_gain import gain
from step7_multiverse import BS, DS, FS, SUBJ, betas, brain_key, ci
from step6_common import roi_betas
from v12_lift import N_BOOT, TRI, boot_idx, boot_pairs

MODELS = ["alexnet", "vgg11_bn", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
CONDS = ["trained", "untrained_s0", "untrained_s1", "untrained_s2"]
UNTR = CONDS[1:]
ROIS = ["V1", "LOC"]
BOOT_CELLS = [("raw", "correlation", "paper"), ("z", "correlation", "paper"),
              ("raw", "euclidean", "paper"), ("center", "correlation", "center720")]
DATA = C.REPO / "data" / "step8_gap"
OUT = C.REPO / "results" / "step8"
MODELS_NPY = C.REPO / "data" / "step8_models.npy"
BRAINS_NPY = C.REPO / "data" / "step8_brains.npy"
_W = {}


def available():
    """(model, R) pairs whose features exist for all 4 conditions and both ROIs."""
    out = []
    for m in MODELS:
        for R in (112, 224, 448):
            if all((DATA / m / c / f"R{R}" / f"{k}.npy").exists() for c in CONDS for k in ROIS):
                out.append((m, R))
    return out


def feat(m, c, R, roi):
    return np.load(DATA / m / c / f"R{R}" / f"{roi}.npy").astype(np.float64)


def _init_pts(keep, brain_rank):
    _W.update(keep=keep, brain_rank=brain_rank)


def _pts_one(job):
    f, d, m, c, R, roi = job
    X = normalise(feat(m, c, R, roi), f)
    v = pdist(X, d)
    if not np.isfinite(v).all():
        raise SystemExit(f"STOP: non-finite model RDM {job}")
    mv = ranks(v[_W["keep"]])
    rows = []
    for b in BS:
        ps = [float(mv @ _W["brain_rank"][brain_key(b, d, roi, s)]) for s in SUBJ]
        rows.append(dict(kind="rho", F=f, D=d, B=b, model=m, cond=c, R=R, roi=roi, rho=float(np.mean(ps)),
                         **{f"rho_{s}": p for s, p in zip(SUBJ, ps)}))
    return rows


def _init_boot(boot):
    _W.update(boot=boot, runs=run_labels(), M=np.load(MODELS_NPY, mmap_mode="r"), B=np.load(BRAINS_NPY, mmap_mode="r"))


def _boot_one(b):
    idx = boot_pairs(b, _W["boot"], _W["runs"])
    Rb = np.stack([ranks(np.asarray(_W["B"][k][idx], float)) for k in range(_W["B"].shape[0])])
    Rm = np.stack([ranks(np.asarray(_W["M"][k][idx], float)) for k in range(_W["M"].shape[0])])
    return b, Rm @ Rb.T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    av = available()
    print(f"available (model, R): {av}", flush=True)
    _, _, keep = pair_set(None, run_labels())
    brain_tri, brain_rank = {}, {}
    for b in BS:
        for d in DS:
            for roi in ROIS:
                for s in SUBJ:
                    v = pdist(betas(s, roi, b), d)
                    brain_tri[brain_key(b, d, roi, s)] = v
                    brain_rank[brain_key(b, d, roi, s)] = ranks(v[keep])

    # ── point estimates ──
    jobs = [(f, d, m, c, R, roi) for f in FS for d in DS for m, R in av for c in CONDS for roi in ROIS]
    with ProcessPoolExecutor(max_workers=8, initializer=_init_pts, initargs=(keep, brain_rank)) as ex:
        pts = pd.DataFrame([r for ch in ex.map(_pts_one, jobs, chunksize=8) for r in ch])
    print(f"point estimates: {len(jobs)} model RDMs", flush=True)

    # ── bootstrap for the 4 cells ──
    fds = sorted({(f, d) for f, d, _ in BOOT_CELLS})
    items = [(f, d, m, c, R, "V1") for f, d in fds for m, R in av for c in CONDS] + \
            [(f, d, m, c, 224, "LOC") for f, d in fds for m, R in av if R == 224 for c in CONDS]
    bitems = sorted({(b, d, roi, s) for f, d, b in BOOT_CELLS for roi in ROIS for s in SUBJ})
    Mm = np.lib.format.open_memmap(MODELS_NPY, mode="w+", dtype=np.float64, shape=(len(items), len(TRI[0])))
    for k, (f, d, m, c, R, roi) in enumerate(items):
        Mm[k] = pdist(normalise(feat(m, c, R, roi), f), d)
    Mm.flush(); del Mm
    Bm = np.lib.format.open_memmap(BRAINS_NPY, mode="w+", dtype=np.float64, shape=(len(bitems), len(TRI[0])))
    for k, it in enumerate(bitems):
        Bm[k] = brain_tri[brain_key(*it)]
    Bm.flush(); del Bm
    print(f"bootstrap: {len(items)} model and {len(bitems)} brain vectors", flush=True)
    with ProcessPoolExecutor(max_workers=8, initializer=_init_boot, initargs=(boot_idx(),)) as ex:
        res = dict(ex.map(_boot_one, range(-1, N_BOOT), chunksize=8))
    G = np.stack([res[b] for b in range(-1, N_BOOT)])
    os.remove(MODELS_NPY); os.remove(BRAINS_NPY)
    mc = {it: k for k, it in enumerate(items)}
    bc = {it: k for k, it in enumerate(bitems)}

    def rho(f, d, b, m, c, R, roi):
        return G[:, mc[(f, d, m, c, R, roi)], [bc[(b, d, roi, s)] for s in SUBJ]].mean(axis=1)

    dev = max(abs(rho(f, d, b, m, c, R, roi)[0] - pts[(pts.F == f) & (pts.D == d) & (pts.B == b) & (pts.model == m)
                                                       & (pts.cond == c) & (pts.R == R) & (pts.roi == roi)].rho.iloc[0])
              for f, d, b in BOOT_CELLS for (f2, d2, m, c, R, roi) in items if (f2, d2) == (f, d))
    print(f"identity draw vs point estimates: max |d| {dev:.2e}", flush=True)
    assert dev < 1e-12

    def gap_draws(f, d, b, m, R):
        return np.mean([rho(f, d, b, m, u, R, "V1") for u in UNTR], axis=0) - rho(f, d, b, m, "trained", R, "V1")

    rows = []
    for f, d, b in BOOT_CELLS:
        for m in MODELS:
            Rs = [R for mm, R in av if mm == m]
            for R in Rs:
                p, lo, hi = ci(gap_draws(f, d, b, m, R))
                rows.append(dict(kind="boot", F=f, D=d, B=b, model=m, roi="V1", R=R, quantity="untrained - trained",
                                 value=p, ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0)))
            if 224 in Rs and 448 in Rs:
                p, lo, hi = ci(gap_draws(f, d, b, m, 448) - gap_draws(f, d, b, m, 224))
                rows.append(dict(kind="boot", F=f, D=d, B=b, model=m, roi="V1", R="448-224", quantity="S_gen",
                                 value=p, ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0)))
            if 224 in Rs:
                x = rho(f, d, b, m, "trained", 224, "LOC") - np.mean([rho(f, d, b, m, u, 224, "LOC") for u in UNTR], axis=0)
                p, lo, hi = ci(x)
                rows.append(dict(kind="boot", F=f, D=d, B=b, model=m, roi="LOC", R=224, quantity="trained - untrained",
                                 value=p, ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0)))
    bt = pd.DataFrame(rows)

    # ── point-estimate gaps for all 36 cells ──
    pv = pts.pivot_table(index=["F", "D", "B", "model", "R", "roi"], columns="cond", values="rho").reset_index()
    pv["untrained_mean"] = pv[UNTR].mean(axis=1)
    pv["gap_point"] = np.where(pv.roi == "V1", pv.untrained_mean - pv.trained, pv.trained - pv.untrained_mean)

    # ── rules ──
    verdict = []
    # GEN1
    gr = []
    for m, R in av:
        for c in CONDS:
            g = gain(feat(m, c, R, "V1"))
            cg = float(np.mean([np.corrcoef(g, gain(roi_betas(s, "V1")))[0, 1] for s in SUBJ]))
            raw = float(pts[(pts.F == "raw") & (pts.D == "correlation") & (pts.B == "paper") & (pts.model == m)
                            & (pts.cond == c) & (pts.R == R) & (pts.roi == "V1")].rho.iloc[0])
            gr.append(dict(kind="c_gain", model=m, cond=c, R=R, c_gain=cg, rsa_raw_V1=raw))
    gr = pd.DataFrame(gr)
    r1 = spearmanr(gr.c_gain, gr.rsa_raw_V1).statistic
    verdict.append(dict(kind="rule", quantity="GEN1 Spearman(c_gain, raw RSA V1)", value=r1, passed=bool(r1 >= 0.6),
                        note=f"n = {len(gr)} (vit_b_16 at 224 only)"))
    # GEN2
    b224 = bt[(bt.roi == "V1") & (bt.R == 224)]
    gen2 = {m: bool(((q.ci_lo > 0).any()) and ((q.ci_hi < 0).any())) for m, q in b224.groupby("model")}
    verdict.append(dict(kind="rule", quantity="GEN2 >= 4/6 models with opposite CI-excluding gaps at 224",
                        value=sum(gen2.values()), passed=bool(sum(gen2.values()) >= 4),
                        note="; ".join(f"{m}={v}" for m, v in gen2.items())))
    # GEN3
    loc = bt[bt.roi == "LOC"]
    gen3 = {m: bool((q.ci_lo > 0).all() and len(q) == 4) for m, q in loc.groupby("model")}
    verdict.append(dict(kind="rule", quantity="GEN3 >= 5/6 models: LOC trained - untrained CI > 0 in all 4 cells",
                        value=sum(gen3.values()), passed=bool(sum(gen3.values()) >= 5),
                        note="; ".join(f"{m}={v}" for m, v in gen3.items())))
    # GEN4
    sg = bt[bt.quantity == "S_gen"]
    raw_s = sg[(sg.F == "raw") & (sg.D == "correlation") & (sg.B == "paper")].set_index("model")
    z_s = sg[(sg.F == "z") & (sg.D == "correlation") & (sg.B == "paper")].set_index("model")
    qual = raw_s[raw_s.ci_excludes_0].index.tolist()
    ok = {m: bool(abs(z_s.loc[m, "value"]) <= 0.5 * abs(raw_s.loc[m, "value"])) for m in qual}
    verdict.append(dict(kind="rule", quantity="GEN4 |S_z| <= 0.5 |S_raw| for qualifying models",
                        value=len(qual), passed=(bool(all(ok.values())) if qual else None),
                        note=("not testable (no model qualifies)" if not qual else
                              "; ".join(f"{m}: S_raw {raw_s.loc[m, 'value']:+.4f}, S_z {z_s.loc[m, 'value']:+.4f}, ok={ok[m]}" for m in qual))))

    # ── sign tables at 224 ──
    srows = []
    for _, r in pv[(pv.roi == "V1") & (pv.R == 224)].iterrows():
        q = bt[(bt.F == r.F) & (bt.D == r.D) & (bt.B == r.B) & (bt.model == r.model) & (bt.roi == "V1") & (bt.R == 224)]
        if len(q):
            q = q.iloc[0]
            sign = "+" if q.ci_lo > 0 else "-" if q.ci_hi < 0 else "0"
            basis = "CI"
        else:
            sign, basis = ("+" if r.gap_point > 0 else "-"), "point"
        srows.append(dict(model=r.model, F=r.F, D=r.D, B=r.B, gap_point=r.gap_point, sign=sign, basis=basis))
    signs = pd.DataFrame(srows)

    # ── gain vs stimulus statistics (R = 224) ──
    import torchvision.transforms as T
    from PIL import Image
    from step8_extract_pretrained import images
    imgs, _ = images()
    pre = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
    Y = np.stack([(lambda x: 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2])(pre(im).numpy().astype(np.float64)) for im in imgs])
    lum, con = Y.reshape(720, -1).mean(axis=1), Y.reshape(720, -1).std(axis=1)
    grow = []
    for m in MODELS:
        if (m, 224) not in av:
            continue
        for c in CONDS:
            g = gain(feat(m, c, 224, "V1"))
            grow.append(dict(model=m, cond=c, corr_luminance=float(np.corrcoef(g, lum)[0, 1]),
                             corr_rms_contrast=float(np.corrcoef(g, con)[0, 1])))
    for s in SUBJ:
        g = gain(roi_betas(s, "V1"))
        grow.append(dict(model="brain V1", cond=s, corr_luminance=float(np.corrcoef(g, lum)[0, 1]),
                         corr_rms_contrast=float(np.corrcoef(g, con)[0, 1])))
    gg = pd.DataFrame(grow)

    pts["kind"] = "rho"
    pv["kind"] = "gap_point"
    C.stamp(pd.concat([pts, pv, bt, gr, pd.DataFrame(verdict)], ignore_index=True)).to_csv(OUT / "generalization.csv", index=False)
    C.stamp(signs).to_csv(OUT / "generalization_signs.csv", index=False)
    C.stamp(gg).to_csv(OUT / "generalization_gain.csv", index=False)

    pd.set_option("display.width", 230)
    print(bt.to_string(index=False))
    print(pd.DataFrame(verdict).to_string(index=False))
    print(signs.groupby(["model", "sign"]).size().unstack(fill_value=0).to_string())
    print(gg.to_string(index=False))


if __name__ == "__main__":
    main()

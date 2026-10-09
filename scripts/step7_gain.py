"""
step7_gain.py
=============
G1, G2 and descriptive gain correlations (amendments/2026-10-09_step7-offset-predictions.md).
EXPLORATORY. No new fits or extraction (STEP5_GAP_DIR incl. bandpass/, arm_b_out).

Gain axis of a matrix X (720 x units): mu = column means, mu_c = mu - mean(mu),
x~_i = row i centred across units, gain_i = <x~_i, mu_c> / (|x~_i| |mu_c|);
gain RDM = |gain_i - gain_j|. Models: raw GAP features (Conv1 for V1, Conv3 for LOC).
fMRI: betas under B = paper, all ROI voxels, per subject.

(a) gain per checkpoint and condition (six NATIVE, UPSAMPLED 224, only_0-14, notch_0-14)
    and per subject / ROI.
(b) c_gain = Pearson over the 720 stimuli between model gain and voxel gain (V1), mean over
    subjects. G1: Spearman(c_gain, raw RSA) >= 0.6 over the 70 cells {Random, BP} x
    5 seeds x {six NATIVE, UPSAMPLED 224}; raw RSA (F = raw, D = correlation, B = paper,
    V1) from results/step5/resolution_rsa.csv. Band-pass cells descriptive alongside
    (raw RSA from results/step6/mechanism_bandpass.csv).
(c) Partial RSA (operationalisation note 2): per subject, ranks of model RDM and fMRI RDM
    (correlation distance) on the pairs of the draw, each residualised on [1, ranks of
    its own gain RDM]; Pearson of the residuals; mean over subjects. Bootstrap (paper's
    1000 resamples, paired) for the V1 gap (Random - BP) at NATIVE 32, NATIVE 224 (and
    UPSAMPLED 224, descriptive), S_partial, and LOC BP - Random at NATIVE 224
    (descriptive). G2: upper CI bound of the V1 partial gap at NATIVE 224 <= 0.010 and
    S_partial <= 0.5 * 0.0311. Check: the non-partial rho computed in the same draws
    equals step 5 (identity draw, 1e-12).
(d) Descriptive: Pearson over the 720 stimuli between voxel gain (V1, per subject) and
    stimulus mean luminance / RMS contrast (NATIVE 224 stimuli, _tf without Normalize,
    Y = 0.299 R + 0.587 G + 0.114 B; operationalisation note 4); the same for model gain
    per condition at NATIVE 32 (32 px stimuli) and NATIVE 224 (224 px stimuli).

Output: results/step7/gain.csv, results/step7/partial_rsa.csv
"""
import step3_common as C          # noqa: I001

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from common_up_pairs import pair_set, ranks, run_labels
from step6_common import bp_path, gap_path, roi_betas
from v12_lift import N_BOOT, RULES, TRI, boot_idx, boot_pairs

SUBJ = ["sub-01", "sub-02", "sub-03"]
R_, BP = "Random Weights", "Backprop"
NAT = [("NATIVE", px) for px in (32, 64, 96, 128, 160, 224)] + [("UPSAMPLED", 224)]
BPC = ["only_0-14", "notch_0-14"]
S_PAPER = 0.0311
OUT = C.REPO / "results" / "step7"
CACHE = C.REPO / "data" / "step7_partial.npy"
_W = {}


def gain(X):
    X = np.asarray(X, np.float64)
    mu = X.mean(axis=0)
    muc = mu - mu.mean()
    xt = X - X.mean(axis=1, keepdims=True)
    return (xt @ muc) / (np.linalg.norm(xt, axis=1) * np.linalg.norm(muc))


def gain_rdm(g):
    return pdist(np.asarray(g, np.float64)[:, None], "cityblock")


def feats(cond, rule, si, layer):
    if cond in BPC:
        return np.load(bp_path(cond, si, rule, layer)).astype(np.float64)
    arm, px = cond
    return np.load(gap_path(arm, px, si, rule, layer)).astype(np.float64)


def resid(r, g):
    """r, g centred unit-norm rank vectors: residual of r on [1, g]."""
    return r - (r @ g) * g


def _init(boot, n_models, n_brains):
    _W.update(boot=boot, runs=run_labels(), M=np.load(CACHE, mmap_mode="r"), n_models=n_models, n_brains=n_brains)


def _one(b):
    idx = boot_pairs(b, _W["boot"], _W["runs"])
    M = _W["M"]
    nm, nb = _W["n_models"], _W["n_brains"]
    # layout: [model rdm 0..nm-1, model gain rdm nm..2nm-1, brain rdm (nb), brain gain rdm (nb)]
    rm = [ranks(np.asarray(M[k][idx], float)) for k in range(nm)]
    gm = [ranks(np.asarray(M[nm + k][idx], float)) for k in range(nm)]
    rb = [ranks(np.asarray(M[2 * nm + k][idx], float)) for k in range(nb)]
    gb = [ranks(np.asarray(M[2 * nm + nb + k][idx], float)) for k in range(nb)]
    eb = [resid(r, g) for r, g in zip(rb, gb)]
    eb = [e / np.linalg.norm(e) for e in eb]
    plain = np.array([[rm[k] @ rb[j] for j in range(nb)] for k in range(nm)])
    part = np.empty((nm, nb))
    for k in range(nm):
        e = resid(rm[k], gm[k])
        e /= np.linalg.norm(e)
        part[k] = [e @ x for x in eb]
    return b, plain, part


def ci(x):
    return float(x[0]), float(np.quantile(x[1:], .025)), float(np.quantile(x[1:], .975))


def stimulus_stats():
    import torchvision.transforms as T
    from PIL import Image
    from common_up_pairs import stim_order
    from v12_lift import namespaces, transform
    ns12, ns11 = namespaces()
    imgs = [Image.open(ns12["find_img"](s)).convert("RGB") for s in stim_order()]
    out = {}
    for px in (32, 224):
        tf = transform(ns12, ns11, "NATIVE", px)
        assert isinstance(tf.transforms[-1], T.Normalize)
        pre = T.Compose(tf.transforms[:-1])
        Y = np.stack([(lambda x: 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2])(pre(im).numpy().astype(np.float64))
                      for im in imgs])
        out[px] = (Y.reshape(len(imgs), -1).mean(axis=1), Y.reshape(len(imgs), -1).std(axis=1))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # ── (a) gains ──
    vg = {(roi, s): gain(roi_betas(s, roi)) for roi in ("V1", "LOC") for s in SUBJ}
    conds = NAT + BPC
    mg = {(c, rule, si, layer): gain(feats(c, rule, si, layer))
          for c in conds for rule in RULES for si in C.SEED_IDX for layer in ("Conv1", "Conv3")}

    # ── (b) c_gain and G1 ──
    s5 = pd.read_csv(C.REPO / "results" / "step5" / "resolution_rsa.csv")
    s5 = s5[(s5.kind == "rho") & (s5.layer == "Conv1") & (s5.roi == "V1") & (s5.variant == "raw")].set_index(
        ["arm", "res", "rule", "seed_idx"]).rho
    s6 = pd.read_csv(C.REPO / "results" / "step6" / "mechanism_bandpass.csv")
    s6 = s6[(s6.kind == "rho") & (s6.roi == "V1") & (s6.variant == "raw")].set_index(["cond", "rule", "seed_idx"]).rho
    for c in conds:
        for rule in RULES:
            for si in C.SEED_IDX:
                cg = [float(np.corrcoef(mg[(c, rule, si, "Conv1")], vg[("V1", s)])[0, 1]) for s in SUBJ]
                raw = float(s6[(c, rule, si)]) if c in BPC else float(s5[(c[0], c[1], rule, si)])
                rows.append(dict(kind="c_gain", cond=c if c in BPC else f"{c[0]}_{c[1]}", rule=rule, seed_idx=si,
                                 c_gain=float(np.mean(cg)), **{f"c_gain_{s}": v for s, v in zip(SUBJ, cg)},
                                 rsa_raw_V1=raw, in_G1=bool(c not in BPC and rule in (R_, BP))))
    cg = pd.DataFrame(rows)
    g1 = cg[cg.in_G1]
    rho_g1 = spearmanr(g1.c_gain, g1.rsa_raw_V1).statistic
    verdict = [dict(kind="rule", quantity="G1 Spearman(c_gain, raw RSA V1), n = 70", value=rho_g1,
                    passed=bool(rho_g1 >= 0.6), note=f"n = {len(g1)}")]
    for r in (R_, BP):
        q = g1[g1.rule == r]
        verdict.append(dict(kind="descriptive", quantity=f"Spearman(c_gain, raw RSA) within {r}",
                            value=spearmanr(q.c_gain, q.rsa_raw_V1).statistic, note=f"n = {len(q)}"))
    allc = cg[cg.rule.isin([R_, BP])]
    verdict.append(dict(kind="descriptive", quantity="Spearman(c_gain, raw RSA) incl. band-pass, Random+BP",
                        value=spearmanr(allc.c_gain, allc.rsa_raw_V1).statistic, note=f"n = {len(allc)}"))

    # ── (d) stimulus statistics ──
    st = stimulus_stats()
    drows = []
    for s in SUBJ:
        for nm, i in (("mean luminance", 0), ("RMS contrast", 1)):
            drows.append(dict(kind="stimulus_corr", what=f"voxel gain V1 {s}", stat=nm, res=224,
                              pearson=float(np.corrcoef(vg[("V1", s)], st[224][i])[0, 1])))
    for px in (32, 224):
        for rule in RULES:
            for nm, i in (("mean luminance", 0), ("RMS contrast", 1)):
                v = [float(np.corrcoef(mg[(("NATIVE", px), rule, si, "Conv1")], st[px][i])[0, 1]) for si in C.SEED_IDX]
                drows.append(dict(kind="stimulus_corr", what=f"model gain Conv1 {rule}", stat=nm, res=px,
                                  pearson=float(np.mean(v)), pearson_sd=float(np.std(v, ddof=1))))
    gsum = cg[cg.rule.isin(RULES)].groupby(["cond", "rule"]).c_gain.agg(["mean", "std"]).reset_index()
    gsum["kind"] = "c_gain_summary"
    C.stamp(pd.concat([cg, gsum, pd.DataFrame(drows), pd.DataFrame(verdict)], ignore_index=True)).to_csv(
        OUT / "gain.csv", index=False)

    # ── (c) partial RSA with bootstrap ──
    items = [("V1", "Conv1", c, rule, si) for c in (("NATIVE", 32), ("NATIVE", 224), ("UPSAMPLED", 224))
             for rule in (R_, BP) for si in C.SEED_IDX] + \
            [("LOC", "Conv3", ("NATIVE", 224), rule, si) for rule in (BP, R_) for si in C.SEED_IDX]
    brains = [(roi, s) for roi in ("V1", "LOC") for s in SUBJ]
    nm, nb = len(items), len(brains)
    M = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.float64, shape=(2 * nm + 2 * nb, len(TRI[0])))
    for k, (roi, layer, c, rule, si) in enumerate(items):
        M[k] = pdist(feats(c, rule, si, layer), "correlation")
        M[nm + k] = gain_rdm(mg[(c, rule, si, layer)])
    for k, (roi, s) in enumerate(brains):
        M[2 * nm + k] = pdist(roi_betas(s, roi), "correlation")
        M[2 * nm + nb + k] = gain_rdm(vg[(roi, s)])
    M.flush(); del M
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(boot_idx(), nm, nb)) as ex:
        res = {b: (p, q) for b, p, q in ex.map(_one, range(-1, N_BOOT), chunksize=8)}
    os.remove(CACHE)
    PL = np.stack([res[b][0] for b in range(-1, N_BOOT)])
    PA = np.stack([res[b][1] for b in range(-1, N_BOOT)])
    col = {it: k for k, it in enumerate(items)}
    bc = {it: k for k, it in enumerate(brains)}

    def rho(A, roi, c, rule, si):
        return A[:, col[(roi, "Conv1" if roi == "V1" else "Conv3", c, rule, si)], [bc[(roi, s)] for s in SUBJ]].mean(axis=1)

    s5full = pd.read_csv(C.REPO / "results" / "step5" / "resolution_rsa.csv")
    s5full = s5full[(s5full.kind == "rho") & (s5full.variant == "raw")]
    dev = max(abs(rho(PL, roi, c, rule, si)[0] - s5full[(s5full.roi == roi) & (s5full.layer == layer) & (s5full.arm == c[0])
                                                          & (s5full.res == c[1]) & (s5full.rule == rule)
                                                          & (s5full.seed_idx == si)].rho.iloc[0])
              for (roi, layer, c, rule, si) in items)
    print(f"plain RSA in the partial draws vs step 5: max |d| {dev:.2e}", flush=True)
    assert dev < 1e-12

    prow, G = [], {}
    for A, kind in ((PL, "plain"), (PA, "partial")):
        for roi, a, b, cs in (("V1", R_, BP, (("NATIVE", 32), ("NATIVE", 224), ("UPSAMPLED", 224))),
                              ("LOC", BP, R_, (("NATIVE", 224),))):
            for c in cs:
                sd = np.stack([rho(A, roi, c, a, si) - rho(A, roi, c, b, si) for si in C.SEED_IDX], axis=1)
                G[(kind, roi, c)] = sd.mean(axis=1)
                p, lo, hi = ci(G[(kind, roi, c)])
                prow.append(dict(kind=kind, roi=roi, cond=f"{c[0]}_{c[1]}", quantity=f"{a} - {b}", value=p,
                                 ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0),
                                 seeds_positive=int((sd[0] > 0).sum())))
                for rule in (a, b):
                    v = np.mean([rho(A, roi, c, rule, si)[0] for si in C.SEED_IDX])
                    prow.append(dict(kind=f"{kind}_condition", roi=roi, cond=f"{c[0]}_{c[1]}", quantity=rule, value=float(v)))
        s_dr = G[(kind, "V1", ("NATIVE", 224))] - G[(kind, "V1", ("NATIVE", 32))]
        p, lo, hi = ci(s_dr)
        prow.append(dict(kind=kind, roi="V1", cond="slope", quantity="S = gap NAT224 - gap NAT32", value=p,
                         ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0)))
    pr = pd.DataFrame(prow)
    g224 = pr[(pr.kind == "partial") & (pr.roi == "V1") & (pr.cond == "NATIVE_224")].iloc[0]
    sp = pr[(pr.kind == "partial") & (pr.cond == "slope")].iloc[0]
    pv = [dict(kind="rule", quantity="G2 partial V1 gap NAT224 upper CI <= 0.010 and S_partial <= 0.5 S_paper",
               value=float(g224.value), ci_lo=float(g224.ci_lo), ci_hi=float(g224.ci_hi),
               passed=bool(g224.ci_hi <= 0.010 and sp.value <= 0.5 * S_PAPER),
               note=f"S_partial {sp.value:+.4f} [{sp.ci_lo:+.4f}, {sp.ci_hi:+.4f}] vs 0.5*S_paper {0.5 * S_PAPER:+.4f}")]
    C.stamp(pd.concat([pr, pd.DataFrame(pv)], ignore_index=True)).to_csv(OUT / "partial_rsa.csv", index=False)

    pd.set_option("display.width", 220)
    print(gsum[gsum.rule.isin([R_, BP])].to_string(index=False))
    print(pd.DataFrame(verdict).to_string(index=False))
    print(pd.DataFrame(drows).to_string(index=False))
    print(pr.to_string(index=False))
    print(pd.DataFrame(pv).to_string(index=False))


if __name__ == "__main__":
    main()

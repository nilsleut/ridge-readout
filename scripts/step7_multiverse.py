"""
step7_multiverse.py
===================
O1, O2, O3 (amendments/2026-10-09_step7-offset-predictions.md). EXPLORATORY.

Multiverse F x D x B (36 cells):
  F  feature normalisation per unit over the 720 stimuli: raw, center, scale, z
  D  distance, applied to BOTH the model features and the fMRI betas (operationalisation
     note 1): correlation, cosine, euclidean
  B  fMRI normalisation: paper (arm_b_out betas, z over all trials), center720, z720
ROIs V1 (Conv1) and LOC (Conv3); NATIVE 32..224 and UPSAMPLED 224; all 30 checkpoints;
three subjects. Statistic as in step 5: per subject on all ROI voxels, cross-run pairs,
rho = ranks(model[P]) @ ranks(fmri[P]), mean over subjects; gap = mean over seeds of
the per-seed difference. No new fits, no new extraction (STEP5_GAP_DIR, arm_b_out).

(a) O1 before anything else (STOP on violation).
(b) Point estimates for every cell, ROI, condition and checkpoint.
(c) Bootstrap (the paper's 1000 resamples, paired): V1 gap at NATIVE 32, NATIVE 224,
    UPSAMPLED 224 and S in all 36 cells; LOC BP - Random at NATIVE 224 in all 36 cells.
    Per-cell draws saved to data/step7_draws/*.npz (gitignored).
(d) Runtime estimate (x 1.5) before the bootstrap; above 4 h the bootstrap is restricted
    to B in {paper, center720} (z720 point estimates only).
(e) O1, O2, O3 and the 4 x 3 x 3 sign table of the V1 gap at NATIVE 224.

Output: results/step7/multiverse.csv, results/step7/multiverse_sign_table.csv
"""
import step3_common as C          # noqa: I001  (pins BLAS threads)

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from common_up_pairs import pair_set, ranks, run_labels
from step6_common import EPS, gap_path, normalise, roi_betas
from v12_lift import N_BOOT, RULES, TRI, boot_idx, boot_pairs

FS = ["raw", "center", "scale", "z"]
DS = ["correlation", "cosine", "euclidean"]
BS = ["paper", "center720", "z720"]
SUBJ = ["sub-01", "sub-02", "sub-03"]
ROIS = {"V1": "Conv1", "LOC": "Conv3"}
CONDS = [("NATIVE", px) for px in (32, 64, 96, 128, 160, 224)] + [("UPSAMPLED", 224)]
R_, BP = "Random Weights", "Backprop"
S_PAPER = 0.0311
OUT = C.REPO / "results" / "step7"
DATA = C.REPO / "data"
DRAWS = DATA / "step7_draws"
MODELS = DATA / "step7_models.npy"
BRAINS = DATA / "step7_brains.npy"
_W = {}


def feat(arm, px, rule, si, layer):
    return np.load(gap_path(arm, px, si, rule, layer)).astype(np.float64)


def betas(subject, roi, how):
    X = roi_betas(subject, roi)
    if how == "paper":
        return X
    Xc = X - X.mean(axis=0)
    if how == "center720":
        return Xc
    sd = X.std(axis=0)
    return Xc / np.where(sd < EPS, 1.0, sd)


def check_o1():
    worst = 0.0
    for arm, px in CONDS:
        for rule in RULES:
            for si in C.SEED_IDX:
                for layer in ROIS.values():
                    X = feat(arm, px, rule, si, layer)
                    worst = max(worst, float(np.abs(pdist(normalise(X, "raw"), "euclidean")
                                                    - pdist(normalise(X, "center"), "euclidean")).max()))
    worst_b = 0.0
    for s in SUBJ:
        for roi in ROIS:
            worst_b = max(worst_b, float(np.abs(pdist(betas(s, roi, "paper"), "euclidean")
                                                - pdist(betas(s, roi, "center720"), "euclidean")).max()))
    print(f"O1: models euclidean raw vs center max |d| {worst:.2e}; fMRI paper vs center720 {worst_b:.2e}", flush=True)
    if worst >= 1e-9 or worst_b >= 1e-9:
        raise SystemExit("STOP: O1 violated")
    return worst, worst_b


def brain_key(b, d, roi, s):
    return f"{b}|{d}|{roi}|{s}"


def _init(boot, n_models, n_brains):
    _W.update(boot=boot, runs=run_labels(), M=np.load(MODELS, mmap_mode="r"), B=np.load(BRAINS, mmap_mode="r"))


def _one(b):
    idx = boot_pairs(b, _W["boot"], _W["runs"])
    Rb = np.stack([ranks(np.asarray(_W["B"][k][idx], float)) for k in range(_W["B"].shape[0])])
    Rm = np.stack([ranks(np.asarray(_W["M"][k][idx], float)) for k in range(_W["M"].shape[0])])
    return b, Rm @ Rb.T


def _init_pts(keep, brain_rank):
    _W.update(keep=keep, brain_rank=brain_rank)


def _pts_one(job):
    f, d, arm, px, rule, si, roi = job
    m = pdist(normalise(feat(arm, px, rule, si, ROIS[roi]), f), d)
    if not np.isfinite(m).all():
        raise SystemExit(f"STOP: non-finite model RDM {f} {d} {arm} {px} {rule} {si} {roi}")
    mv = ranks(m[_W["keep"]])
    out = []
    for b in BS:
        ps = [float(mv @ _W["brain_rank"][brain_key(b, d, roi, s)]) for s in SUBJ]
        out.append(dict(kind="rho", F=f, D=d, B=b, roi=roi, arm=arm, res=px, rule=rule, seed_idx=si,
                        rho=float(np.mean(ps)), **{f"rho_{s}": p for s, p in zip(SUBJ, ps)}))
    return out


def ci(x):
    return float(x[0]), float(np.quantile(x[1:], .025)), float(np.quantile(x[1:], .975))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DRAWS.mkdir(parents=True, exist_ok=True)
    o1 = check_o1()
    _, _, keep = pair_set(None, run_labels())

    # ── brains ──
    brain_tri, brain_rank = {}, {}
    for b in BS:
        for d in DS:
            for roi in ROIS:
                for s in SUBJ:
                    v = pdist(betas(s, roi, b), d)
                    assert np.isfinite(v).all()
                    brain_tri[brain_key(b, d, roi, s)] = v
                    brain_rank[brain_key(b, d, roi, s)] = ranks(v[keep])

    # ── point estimates (parallel over models; same computation as a serial loop) ──
    t0 = time.time()
    jobs = [(f, d, arm, px, rule, si, roi) for f in FS for d in DS for arm, px in CONDS
            for rule in RULES for si in C.SEED_IDX for roi in ROIS]
    with ProcessPoolExecutor(max_workers=8, initializer=_init_pts, initargs=(keep, brain_rank)) as ex:
        rows = [r for chunk in ex.map(_pts_one, jobs, chunksize=16) for r in chunk]
    print(f"  point estimates done ({time.time() - t0:.0f}s, {len(jobs)} models)", flush=True)
    pts = pd.DataFrame(rows)

    # ── bootstrap ──
    t1 = time.time()
    _ = [ranks(brain_tri[k][keep]) for k in list(brain_tri)[:10]]
    t_rank = (time.time() - t1) / 10
    bboot = list(BS)
    model_items = [(f, d, roi, arm, px, rule, si) for f in FS for d in DS
                   for roi, conds in (("V1", [("NATIVE", 32), ("NATIVE", 224), ("UPSAMPLED", 224)]),
                                      ("LOC", [("NATIVE", 224)]))
                   for arm, px in conds for rule in (R_, BP) for si in C.SEED_IDX]
    est_h = (len(model_items) + len(bboot) * len(DS) * len(ROIS) * len(SUBJ)) * t_rank * (N_BOOT + 1) / 8 * 1.5 / 3600
    print(f"bootstrap runtime estimate: {est_h:.2f} h ({t_rank * 1e3:.0f} ms per ranking)", flush=True)
    if est_h > 4:
        bboot = ["paper", "center720"]
        print("estimate above 4 h: bootstrap for B in {paper, center720} only; z720 point estimates only")
    brain_items = [(b, d, roi, s) for b in bboot for d in DS for roi in ROIS for s in SUBJ]

    M = np.lib.format.open_memmap(MODELS, mode="w+", dtype=np.float64, shape=(len(model_items), len(TRI[0])))
    for k, (f, d, roi, arm, px, rule, si) in enumerate(model_items):
        M[k] = pdist(normalise(feat(arm, px, rule, si, ROIS[roi]), f), d)
    M.flush(); del M
    Bm = np.lib.format.open_memmap(BRAINS, mode="w+", dtype=np.float64, shape=(len(brain_items), len(TRI[0])))
    for k, it in enumerate(brain_items):
        Bm[k] = brain_tri[brain_key(*it)]
    Bm.flush(); del Bm
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(boot_idx(), len(model_items), len(brain_items))) as ex:
        res = dict(ex.map(_one, range(-1, N_BOOT), chunksize=8))
    G = np.stack([res[b] for b in range(-1, N_BOOT)])          # (1001, n_models, n_brains)
    mcol = {it: k for k, it in enumerate(model_items)}
    bcol = {it: k for k, it in enumerate(brain_items)}

    def rho_draws(f, d, b, roi, arm, px, rule, si):
        return G[:, mcol[(f, d, roi, arm, px, rule, si)], [bcol[(b, d, roi, s)] for s in SUBJ]].mean(axis=1)

    # identity draw vs point estimates
    dev = 0.0
    for f in FS:
        for d in DS:
            for b in bboot:
                for (roi, arm, px) in (("V1", "NATIVE", 32), ("V1", "NATIVE", 224), ("V1", "UPSAMPLED", 224), ("LOC", "NATIVE", 224)):
                    for rule in (R_, BP):
                        for si in C.SEED_IDX:
                            p = pts[(pts.F == f) & (pts.D == d) & (pts.B == b) & (pts.roi == roi) & (pts.arm == arm)
                                    & (pts.res == px) & (pts.rule == rule) & (pts.seed_idx == si)].rho.iloc[0]
                            dev = max(dev, abs(rho_draws(f, d, b, roi, arm, px, rule, si)[0] - p))
    print(f"identity draw vs point estimates: max |d| {dev:.2e}", flush=True)
    assert dev < 1e-12

    # ── per-cell quantities ──
    crow = []
    for f in FS:
        for d in DS:
            for b in BS:
                cell = dict(F=f, D=d, B=b)
                booted = b in bboot
                arrays = {}
                for roi, a, bb, conds in (("V1", R_, BP, [("NATIVE", 32), ("NATIVE", 224), ("UPSAMPLED", 224)]),
                                          ("LOC", BP, R_, [("NATIVE", 224)])):
                    for arm, px in conds:
                        if booted:
                            sd = np.stack([rho_draws(f, d, b, roi, arm, px, a, si) - rho_draws(f, d, b, roi, arm, px, bb, si)
                                           for si in C.SEED_IDX], axis=1)
                            arrays[f"{roi}_{arm}_{px}"] = sd
                            p, lo, hi = ci(sd.mean(axis=1))
                            pos = int((sd[0] > 0).sum())
                        else:
                            q = pts[(pts.F == f) & (pts.D == d) & (pts.B == b) & (pts.roi == roi) & (pts.arm == arm) & (pts.res == px)]
                            pv = q.pivot(index="seed_idx", columns="rule", values="rho")
                            dd = pv[a] - pv[bb]
                            p, lo, hi, pos = float(dd.mean()), np.nan, np.nan, int((dd > 0).sum())
                        crow.append(dict(kind="cell", **cell, roi=roi, quantity=f"gap {arm} {px}: {a} - {bb}",
                                         value=p, ci_lo=lo, ci_hi=hi, bootstrapped=booted, seeds_positive=pos,
                                         ci_excludes_0=bool(booted and (lo > 0 or hi < 0))))
                if booted:
                    s_dr = arrays["V1_NATIVE_224"].mean(axis=1) - arrays["V1_NATIVE_32"].mean(axis=1)
                    p, lo, hi = ci(s_dr)
                    np.savez(DRAWS / f"cell_{f}_{d}_{b}.npz", **arrays,
                             note=np.array("per-seed gap draws (1001 x 5); row 0 = identity; V1 Random-BP, LOC BP-Random"))
                else:
                    gv = {r["quantity"]: r["value"] for r in crow if r["F"] == f and r["D"] == d and r["B"] == b and r["roi"] == "V1"}
                    p, lo, hi = gv[f"gap NATIVE 224: {R_} - {BP}"] - gv[f"gap NATIVE 32: {R_} - {BP}"], np.nan, np.nan
                crow.append(dict(kind="cell", **cell, roi="V1", quantity="S = gap NAT224 - gap NAT32", value=p,
                                 ci_lo=lo, ci_hi=hi, bootstrapped=booted, ci_excludes_0=bool(booted and (lo > 0 or hi < 0))))
    cells = pd.DataFrame(crow)
    os.remove(MODELS); os.remove(BRAINS)

    # ── rules ──
    verdict = [dict(kind="rule", quantity="O1 euclidean raw vs center (models); paper vs center720 (fMRI)",
                    value=max(o1), passed=True, note=f"models {o1[0]:.2e}, fMRI {o1[1]:.2e}")]
    S = cells[cells.quantity.str.startswith("S =")].copy()
    S["allowed"] = S.D.isin(["correlation", "cosine"]) & S.F.isin(["raw", "scale"]) & (S.B == "paper")
    S["strong"] = (S.ci_lo > 0) & (S.value >= 0.5 * S_PAPER)
    tested = S[S.bootstrapped]
    viol = tested[~tested.allowed & tested.strong]
    verdict.append(dict(kind="rule", quantity="O2 strong slope only in allowed cells; <= 2 violations",
                        value=len(viol), passed=bool(len(viol) <= 2),
                        note=f"tested cells {len(tested)}/36; violations: " +
                             ("; ".join(f"{r.F}/{r.D}/{r.B} S={r.value:+.4f} [{r.ci_lo:+.4f},{r.ci_hi:+.4f}]" for r in viol.itertuples()) or "none")
                             + "; allowed cells strong: " +
                             ", ".join(f"{r.F}/{r.D}/{r.B}={bool(r.strong)}" for r in tested[tested.allowed].itertuples())))
    L = cells[(cells.roi == "LOC")]
    npos, nci = int((L.value > 0).sum()), int((L.ci_lo > 0).sum())
    verdict.append(dict(kind="rule", quantity="O3 LOC BP - Random NAT224 > 0 in >= 30/36 and CI > 0 in >= 18",
                        value=npos, passed=bool(npos >= 30 and nci >= 18), note=f"positive {npos}/36, CI above 0 {nci}/{int(L.bootstrapped.sum())} bootstrapped"))

    sign = cells[(cells.roi == "V1") & (cells.quantity == f"gap NATIVE 224: {R_} - {BP}")][
        ["F", "D", "B", "value", "ci_lo", "ci_hi", "ci_excludes_0", "seeds_positive"]].copy()
    sign["sign"] = np.where(sign.value > 0, "+", "-")
    sign["label"] = sign.sign + np.where(sign.ci_excludes_0, "*", "")

    pts["kind"] = "rho"
    C.stamp(pd.concat([pts, cells, pd.DataFrame(verdict)], ignore_index=True)).to_csv(OUT / "multiverse.csv", index=False)
    C.stamp(sign).to_csv(OUT / "multiverse_sign_table.csv", index=False)
    pd.set_option("display.width", 230)
    print(S[["F", "D", "B", "value", "ci_lo", "ci_hi", "allowed", "strong"]].to_string(index=False))
    print(L[["F", "D", "B", "value", "ci_lo", "ci_hi"]].to_string(index=False))
    print(sign.pivot_table(index=["F"], columns=["B", "D"], values="label", aggfunc="first").to_string())
    print(pd.DataFrame(verdict)[["quantity", "value", "passed", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()

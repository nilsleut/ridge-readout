"""
step5_resolution_rsa.py
=======================
Step 5 (amendments/2026-10-08_step5-resolution-predictions.md). EXPLORATORY.

RSA as in paper v3 (code/upsampling/step2_rsa.py, step3_bootstrap.py, step4_evaluate.py):
per subject (3) on the subject's fMRI RDM of ALL voxels of the ROI (OUTPUTS_720_DIR
fmri_rdm_{roi}_{sub}.npy), cross-run pairs, rho = ranks(model[P]) @ ranks(fmri[P]),
then mean over subjects; gap = mean over the 5 seeds of the per-seed difference.
Stimulus bootstrap: the paper's 1000 resamples (v12_lift.boot_idx, sha-checked), pair set
recomputed per draw (v12_lift.boot_pairs), identity draw = point estimate, 95%
percentile CIs; differences, slopes and ratios formed inside each draw (paired).

Variants of the GAP features before pdist(..., "correlation"):
  raw  unchanged (paper)
  z    z-score per channel over the 720 stimuli, dead channels (SD < 1e-8) -> 0
       (step4_std_rsa.transform)

Layer -> ROI: v12 LAYER_ROI_FIXED (Conv1 -> V1, V2; Conv2 -> V1, V2; Conv3 -> LOC;
FC1 -> IT). Point estimates for all of them; the bootstrap covers what the rules and
the paper comparisons need: Conv1 -> V1 and Conv3 -> LOC, Random Weights and Backprop,
at NATIVE 32, NATIVE 224 and UPSAMPLED 224.

Rules R1-R5 and M1 as fixed in the amendment. Paper comparisons: every model-based
quantity that paper v3 reports with a stimulus-bootstrap CI (results/paper_v3/
NUMBERS_MANIFEST_v3.md entries with .lo/.hi from step4_gaps / step4_ratio), plus the
NATIVE LOC gap that the LOC ratio is built on.

Output: results/step5/resolution_rsa.csv, results/step5/moment_ratio.csv,
        results/step5/paper_comparisons_raw_vs_z.csv
"""
import step3_common as C          # noqa: I001  (pins BLAS threads)

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(C.REPO / "src" / "vendor"))
import paths                                                       # noqa: E402
from common_up_pairs import run_labels                             # noqa: E402
from step2_rsa_501 import tri_rdm                                  # noqa: E402
from step4_std_rsa import transform                                # noqa: E402
from common_up_pairs import ranks, pair_set                        # noqa: E402
from v12_lift import (RULES, RES, LAYER_ROI, SUBJECTS, N_BOOT, rule_key,  # noqa: E402
                      boot_idx, boot_pairs, load_fmri_tri)

R_, BP = "Random Weights", "Backprop"
VARIANTS = ["raw", "z"]
CONDS = [("NATIVE", px) for px in RES] + [("UPSAMPLED", 224)]
BOOT_CONDS = [("NATIVE", 32), ("NATIVE", 224), ("UPSAMPLED", 224)]
CASES = {"V1": ("Conv1", R_, BP), "LOC": ("Conv3", BP, R_)}     # step3_bootstrap.CASES
OUT = C.REPO / "results" / "step5"
CACHE = C.REPO / "data" / "step5_boot_stack.npy"
_W = {}


def gap_path(arm, px, si, rule, layer):
    return paths.step5_gap_dir() / arm / f"res{px}" / f"seed_{si}" / f"gap_{rule_key(rule)}_{layer}.npy"


def model_tri(arm, px, si, rule, layer, variant):
    X, _ = transform(np.load(gap_path(arm, px, si, rule, layer)), variant)
    return tri_rdm(X)


# ── point estimates ──────────────────────────────────────────────────────────
def point_estimates(keep):
    brain = {roi: [ranks(f[keep]) for f in load_fmri_tri(roi)] for roi in {r for _, r in LAYER_ROI}}
    rows = []
    for arm, px in CONDS:
        for rule in RULES:
            for si in C.SEED_IDX:
                for layer in sorted({l for l, _ in LAYER_ROI}):
                    for v in VARIANTS:
                        m = model_tri(arm, px, si, rule, layer, v)
                        if np.isnan(m).any():
                            raise SystemExit(f"STOP: NaN in model RDM {arm} {px} {rule} s{si} {layer} {v}")
                        mv = ranks(m[keep])
                        for lay, roi in LAYER_ROI:
                            if lay != layer:
                                continue
                            ps = [float(mv @ s) for s in brain[roi]]
                            rows.append(dict(kind="rho", arm=arm, res=px, rule=rule, seed_idx=si,
                                             layer=layer, roi=roi, variant=v, rho=float(np.mean(ps)),
                                             **{f"rho_{s}": p for s, p in zip(SUBJECTS, ps)}))
        print(f"  points {arm} {px}", flush=True)
    return pd.DataFrame(rows)


# ── bootstrap ────────────────────────────────────────────────────────────────
def items():
    return [(roi, arm, px, rule, si, v) for roi, (layer, a, b) in CASES.items()
            for arm, px in BOOT_CONDS for rule in (a, b) for si in C.SEED_IDX for v in VARIANTS]


def build_cache():
    it = items()
    M = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.float64, shape=(len(it), len(TRI_[0])))
    for k, (roi, arm, px, rule, si, v) in enumerate(it):
        M[k] = model_tri(arm, px, si, rule, CASES[roi][0], v)
    M.flush()


TRI_ = np.triu_indices(720, 1)


def _init(boot):
    _W["boot"], _W["runs"] = boot, run_labels()
    _W["fmri"] = {roi: load_fmri_tri(roi) for roi in CASES}
    _W["M"] = np.load(CACHE, mmap_mode="r")
    _W["items"] = items()


def one(b):
    idx = boot_pairs(b, _W["boot"], _W["runs"])
    brain = {roi: [ranks(f[idx]) for f in F] for roi, F in _W["fmri"].items()}
    out = np.empty(len(_W["items"]))
    for k, (roi, *_rest) in enumerate(_W["items"]):
        mv = ranks(np.asarray(_W["M"][k][idx], float))
        out[k] = np.mean([mv @ x for x in brain[roi]])
    return b, out


def bootstrap():
    boot = boot_idx()
    build_cache()
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(boot,)) as ex:
        res = dict(ex.map(one, range(-1, N_BOOT), chunksize=8))
    return np.stack([res[b] for b in range(-1, N_BOOT)])          # (1001, n_items)


def ci(x):
    return float(x[0]), float(np.quantile(x[1:], .025)), float(np.quantile(x[1:], .975))


def main():
    _, _, keep = pair_set(None, run_labels())
    assert int(keep.sum()) == 210205
    OUT.mkdir(parents=True, exist_ok=True)
    pts = point_estimates(keep)

    D = bootstrap()
    it = items()
    col = {key: k for k, key in enumerate(it)}

    def gap(roi, arm, px, v):
        _, a, b = CASES[roi]
        return np.stack([D[:, col[(roi, arm, px, a, si, v)]] - D[:, col[(roi, arm, px, b, si, v)]]
                         for si in C.SEED_IDX], axis=1)          # (1001, 5)

    # identity draw must equal the point estimates
    dev = 0.0
    for (roi, arm, px, rule, si, v), k in col.items():
        p = pts[(pts.roi == roi) & (pts.layer == CASES[roi][0]) & (pts.arm == arm) & (pts.res == px)
                & (pts.rule == rule) & (pts.seed_idx == si) & (pts.variant == v)].rho.iloc[0]
        dev = max(dev, abs(D[0, k] - p))
    print(f"identity draw vs point estimates: max |d| {dev:.2e}")
    assert dev < 1e-12

    G = {(roi, arm, px, v): gap(roi, arm, px, v).mean(axis=1) for roi in CASES for arm, px in BOOT_CONDS
         for v in VARIANTS}
    rows = []

    def add(name, v, x, **kw):
        p, lo, hi = ci(x)
        rows.append(dict(kind="comparison", quantity=name, variant=v, value=p, ci_lo=lo, ci_hi=hi,
                         ci_excludes_0=bool(lo > 0 or hi < 0), sign=int(np.sign(p)), **kw))

    for v in VARIANTS:
        for roi, lab in (("V1", "Random - BP, Conv1 -> V1"), ("LOC", "BP - Random, Conv3 -> LOC")):
            for arm, px in BOOT_CONDS:
                sd = gap(roi, arm, px, v)[0]
                add(f"gap {roi} {arm} {px}: {lab}", v, G[(roi, arm, px, v)],
                    seed_sd=float(sd.std(ddof=1)), seeds_positive=int((sd > 0).sum()))
            up, na = G[(roi, "UPSAMPLED", 224, v)], G[(roi, "NATIVE", 224, v)]
            add(f"R {roi} = gap UP224 / gap NAT224", v, up / na,
                frac_boot_nat_le_0=float((na[1:] <= 0).mean()))
            add(f"diff {roi} = gap UP224 - gap NAT224", v, up - na)
        add("S V1 = gap NAT224 - gap NAT32", v, G[("V1", "NATIVE", 224, v)] - G[("V1", "NATIVE", 32, v)])
    comp = pd.DataFrame(rows)

    def get(q, v):
        return comp[(comp.quantity.str.startswith(q)) & (comp.variant == v)].iloc[0]

    # ── rules ──
    verdict = []
    g224, g32 = get("gap V1 NATIVE 224", "raw"), get("gap V1 NATIVE 32", "raw")
    r1 = (abs(g224.value - 0.030) <= 0.001 and abs(g224.ci_lo - 0.018) <= 0.002
          and abs(g224.ci_hi - 0.042) <= 0.002 and abs(g32.value + 0.001) <= 0.001)
    verdict.append(dict(kind="rule", quantity="R1 reproduction", value=g224.value, ci_lo=g224.ci_lo,
                        ci_hi=g224.ci_hi, passed=r1, note=f"NAT32 raw {g32.value:+.4f}"))
    if not r1:
        pd.DataFrame(verdict).pipe(C.stamp).to_csv(OUT / "paper_comparisons_raw_vs_z.csv", index=False)
        raise SystemExit("STOP: R1 reproduction gate failed")
    z224 = get("gap V1 NATIVE 224", "z")
    r2 = ("H_mom explains the headline" if z224.ci_hi <= 0.010 else
          "H_mom does not explain it" if z224.ci_lo >= 0.015 else "ambiguous")
    verdict.append(dict(kind="rule", quantity="R2 z V1 NAT224 Random - BP", value=z224.value,
                        ci_lo=z224.ci_lo, ci_hi=z224.ci_hi, note=r2))
    s_raw, s_z = get("S V1", "raw"), get("S V1", "z")
    verdict.append(dict(kind="rule", quantity="R3 slope S_z <= 0.5 S_raw", value=s_z.value, ci_lo=s_z.ci_lo,
                        ci_hi=s_z.ci_hi, passed=bool(s_z.value <= 0.5 * s_raw.value),
                        note=f"S_raw {s_raw.value:+.4f} [{s_raw.ci_lo:+.4f}, {s_raw.ci_hi:+.4f}]"))
    l = get("gap LOC NATIVE 224", "z")
    verdict.append(dict(kind="rule", quantity="R4 z LOC NAT224 BP - Random > 0, CI excludes 0", value=l.value,
                        ci_lo=l.ci_lo, ci_hi=l.ci_hi, passed=bool(l.value > 0 and l.ci_lo > 0)))
    u = get("gap V1 UPSAMPLED 224", "z")
    verdict.append(dict(kind="rule", quantity="R5 z V1 UP224 upper CI <= 0.010", value=u.value, ci_lo=u.ci_lo,
                        ci_hi=u.ci_hi, passed=bool(u.ci_hi <= 0.010)))

    # ── M1 ──
    mrows = []
    for arm, px in CONDS:
        for rule in RULES:
            for si in C.SEED_IDX:
                X = np.load(gap_path(arm, px, si, rule, "Conv1")).astype(np.float64)
                mu, var = X.mean(axis=0), X.var(axis=0)
                mrows.append(dict(arm=arm, res=px, rule=rule, seed_idx=si,
                                  m=float(mu @ mu / (mu @ mu + var.sum())),
                                  mean_norm2=float(mu @ mu), var_sum=float(var.sum())))
    mr = pd.DataFrame(mrows)
    v1 = pts[(pts.layer == "Conv1") & (pts.roi == "V1")].pivot_table(
        index=["arm", "res", "rule", "seed_idx"], columns="variant", values="rho").reset_index()
    v1["d"] = v1.raw - v1.z
    mr = mr.merge(v1[["arm", "res", "rule", "seed_idx", "raw", "z", "d"]], on=["arm", "res", "rule", "seed_idx"])
    cells = mr[(mr.arm == "NATIVE") & mr.rule.isin([R_, BP])]
    rho_all = spearmanr(cells.m, cells.d).statistic
    verdict.append(dict(kind="rule", quantity="M1 Spearman(m, d), Random+BP, NATIVE x 6 res", value=rho_all,
                        passed=bool(rho_all >= 0.6), note=f"n = {len(cells)}"))
    for r in (R_, BP):
        c = cells[cells.rule == r]
        verdict.append(dict(kind="descriptive", quantity=f"M1 within {r}", value=spearmanr(c.m, c.d).statistic,
                            note=f"n = {len(c)}"))

    msum = mr.groupby(["arm", "res", "rule"])[["m", "d", "raw", "z"]].mean().reset_index()
    msum["kind"] = "condition_mean"
    mr["kind"] = "checkpoint"
    C.stamp(pd.concat([mr, msum], ignore_index=True)).to_csv(OUT / "moment_ratio.csv", index=False)

    # gaps at every resolution (point estimates) for both variants
    gp = []
    for v in VARIANTS:
        for roi, (layer, a, b) in CASES.items():
            for arm, px in CONDS:
                q = pts[(pts.roi == roi) & (pts.layer == layer) & (pts.arm == arm) & (pts.res == px) & (pts.variant == v)]
                p = q.pivot(index="seed_idx", columns="rule", values="rho")
                d = p[a] - p[b]
                gp.append(dict(kind="gap_point", roi=roi, layer=layer, arm=arm, res=px, variant=v,
                               quantity=f"{a} - {b}", value=d.mean(), seed_sd=d.std(ddof=1),
                               seeds_positive=int((d > 0).sum())))
    pts["kind"] = "rho"
    C.stamp(pd.concat([pts, pd.DataFrame(gp)], ignore_index=True)).to_csv(OUT / "resolution_rsa.csv", index=False)
    out = C.stamp(pd.concat([comp, pd.DataFrame(verdict)], ignore_index=True))
    out.to_csv(OUT / "paper_comparisons_raw_vs_z.csv", index=False)

    pd.set_option("display.width", 230)
    print(comp[["quantity", "variant", "value", "ci_lo", "ci_hi", "ci_excludes_0", "seeds_positive"]].to_string(index=False))
    print(pd.DataFrame(verdict)[["quantity", "value", "ci_lo", "ci_hi", "passed", "note"]].to_string(index=False))
    print(pd.DataFrame(gp).query("arm == 'NATIVE'")[["roi", "res", "variant", "value", "seeds_positive"]].to_string(index=False))
    print(msum[msum.rule.isin([R_, BP])].to_string(index=False))
    os.remove(CACHE)


if __name__ == "__main__":
    main()

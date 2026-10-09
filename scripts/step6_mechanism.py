"""
step6_mechanism.py
==================
P_mech1, P_mech2 and the descriptive spectra (amendments/2026-10-09_step6-predictions.md).
EXPLORATORY.

(b) RSA raw and z at V1 (Conv1) and LOC (Conv3) for the band-pass conditions only_0-14
    and notch_0-14 (NATIVE 224), convention of step 5 (per subject on all ROI voxels,
    cross-run pairs, mean over subjects); bootstrap CIs (1000 paired draws) for
    Random - BP (V1) and BP - Random (LOC). Point estimates for all six conditions.
(c) m = |mu|^2 / (|mu|^2 + sum sigma^2) of the Conv1 GAP features per checkpoint and
    condition. P_mech1 from results/step5/moment_ratio.csv (no recomputation):
      BP: m(UP224) < m(NAT224) < m(NAT32) in >= 4/5 seeds; Random: range of m over the
      three conditions < 0.5 x BP range (mean over seeds of the per-seed range).
    P_mech2: m_BP(notch) > m_BP(only) in 5/5 seeds; Spearman(m, d) >= 0.6 over all cells
      (Random and BP x 5 seeds x {NAT32, NAT224, UP224, only_0-14, notch_0-14}),
      d = RSA_raw - RSA_z at V1, subject-averaged (NAT/UP cells from step 5).
(d) Descriptive: Conv1 filter spectra per checkpoint (each 3x3 kernel zero-padded to
    32x32, |FFT2| averaged over the 3 input channels, radially averaged; peak frequency
    and spectral centroid in cycles per pixel, with and without DC), mean +- SD over
    filters and seeds per condition. Radially averaged power spectrum of the 720
    stimuli at NATIVE 32, NATIVE 224, UPSAMPLED 224: the _tf transforms without their
    final Normalize, luminance Y = 0.299 R + 0.587 G + 0.114 B.

Output: results/step6/mechanism_bandpass.csv, moment_ratio_bandpass.csv,
        filter_spectra.csv, stimulus_spectra.csv
"""
import step3_common as C          # noqa: I001

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import paths
from step2_rsa_501 import tri_rdm
from step6_common import bootstrap, bp_path, ci, moment_ratio, normalise
from v12_lift import RULES, load_fmri_tri, rule_key

R_, BP = "Random Weights", "Backprop"
CONDS = ["only_0-14", "notch_0-14"]
CASES = {"V1": ("Conv1", R_, BP), "LOC": ("Conv3", BP, R_)}
OUT = C.REPO / "results" / "step6"
CACHE = C.REPO / "data" / "step6_boot_mech.npy"
WKEY = {"Backprop": "conv1.0.weight", "Random Weights": "conv1.0.weight",
        "Random Weights (BN-calibrated)": "conv1.0.weight", "Feedback Alignment": "conv1.0.W",
        "Predictive Coding": "W1.weight", "STDP": "conv1.weight"}


def model_tri(cond, si, rule, layer, v):
    return tri_rdm(normalise(np.load(bp_path(cond, si, rule, layer)), v))


def radial(A, n):
    """Radial mean of a 2-D spectrum (n x n, FFT layout) in integer-radius bins 0..n//2."""
    k = np.fft.fftfreq(n) * n
    r = np.rint(np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)).astype(int)
    out = np.array([A[r == i].mean() for i in range(n // 2 + 1)])
    return out


def filter_spectra():
    rows = []
    f = np.arange(17) / 32.0
    for rule in RULES:
        for si in C.SEED_IDX:
            sd = torch.load(paths.ckpt_v12_dir() / f"model_weights_{rule_key(rule)}_seed{si}.pt",
                            map_location="cpu", weights_only=True)
            W = sd[WKEY[rule]].numpy().astype(np.float64)
            assert W.shape == (32, 3, 3, 3), (rule, W.shape)
            for fi in range(32):
                A = np.mean([np.abs(np.fft.fft2(W[fi, c], s=(32, 32))) for c in range(3)], axis=0)
                ra = radial(A, 32)
                rows.append(dict(rule=rule, seed_idx=si, filter=fi,
                                 peak_cpp=float(f[np.argmax(ra)]),
                                 peak_cpp_noDC=float(f[1 + np.argmax(ra[1:])]),
                                 centroid_cpp=float((f * ra).sum() / ra.sum()),
                                 centroid_cpp_noDC=float((f[1:] * ra[1:]).sum() / ra[1:].sum())))
    per = pd.DataFrame(rows)
    cols = ["peak_cpp", "peak_cpp_noDC", "centroid_cpp", "centroid_cpp_noDC"]
    summ = per.groupby("rule")[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    return per, summ.reset_index()


def stimulus_spectra():
    import torchvision.transforms as T
    from PIL import Image
    from common_up_pairs import stim_order
    from v12_lift import namespaces, transform
    ns12, ns11 = namespaces()
    imgs = [Image.open(ns12["find_img"](s)).convert("RGB") for s in stim_order()]
    rows = []
    for arm, px in (("NATIVE", 32), ("NATIVE", 224), ("UPSAMPLED", 224)):
        tf = transform(ns12, ns11, arm, px)
        pre = T.Compose(tf.transforms[:-1])          # _tf without the final Normalize
        assert isinstance(tf.transforms[-1], T.Normalize)
        acc = np.zeros(px // 2 + 1)
        for im in imgs:
            x = pre(im).numpy().astype(np.float64)
            y = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
            acc += radial(np.abs(np.fft.fft2(y)) ** 2, px)
        acc /= len(imgs)
        for i, pw in enumerate(acc):
            rows.append(dict(arm=arm, res=px, radius_bin=i, cycles_per_image=i, cycles_per_pixel=i / px,
                             mean_power=float(pw)))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    brains = {roi: load_fmri_tri(roi) for roi in CASES}
    from common_up_pairs import pair_set, ranks, run_labels
    _, _, keep = pair_set(None, run_labels())
    br = {roi: [ranks(f[keep]) for f in F] for roi, F in brains.items()}

    # ── point estimates, all six conditions ──
    pts, mrows = [], []
    for cond in CONDS:
        for rule in RULES:
            for si in C.SEED_IDX:
                for roi, (layer, _, _) in CASES.items():
                    for v in ("raw", "z"):
                        mv = ranks(model_tri(cond, si, rule, layer, v)[keep])
                        ps = [float(mv @ s) for s in br[roi]]
                        pts.append(dict(kind="rho", cond=cond, rule=rule, seed_idx=si, layer=layer, roi=roi,
                                        variant=v, rho=float(np.mean(ps)),
                                        **{f"rho_sub-0{i + 1}": p for i, p in enumerate(ps)}))
                mrows.append(dict(cond=cond, rule=rule, seed_idx=si,
                                  m=moment_ratio(np.load(bp_path(cond, si, rule, "Conv1")))))
    pts = pd.DataFrame(pts)

    # ── bootstrap: Random and BP, both ROIs, both conditions, both variants ──
    items = [(cond, roi, rule, si, v) for cond in CONDS for roi, (layer, a, b) in CASES.items()
             for rule in (a, b) for si in C.SEED_IDX for v in ("raw", "z")]
    D = bootstrap([(lambda c=c, r=r, s=s, v=v, l=CASES[roi][0]: model_tri(c, s, r, l, v))
                   for c, roi, r, s, v in items], [it[1] for it in items], brains, CACHE)
    col = {it: k for k, it in enumerate(items)}
    dev = max(abs(D[0, k] - pts[(pts.cond == c) & (pts.roi == roi) & (pts.rule == r) & (pts.seed_idx == s)
                                & (pts.variant == v)].rho.iloc[0]) for (c, roi, r, s, v), k in col.items())
    print(f"identity draw vs point estimates: max |d| {dev:.2e}")
    assert dev < 1e-12
    comp = []
    for cond in CONDS:
        for roi, (layer, a, b) in CASES.items():
            for v in ("raw", "z"):
                sd = np.stack([D[:, col[(cond, roi, a, si, v)]] - D[:, col[(cond, roi, b, si, v)]] for si in C.SEED_IDX], axis=1)
                p, lo, hi = ci(sd.mean(axis=1))
                comp.append(dict(kind="gap", cond=cond, roi=roi, variant=v, quantity=f"{a} - {b}", value=p,
                                 ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0),
                                 seeds_positive=int((sd[0] > 0).sum())))
    comp = pd.DataFrame(comp)
    try:
        import os
        os.remove(CACHE)
    except OSError:
        pass

    # ── m and d ──
    mb = pd.DataFrame(mrows)
    v1 = pts[pts.roi == "V1"].pivot_table(index=["cond", "rule", "seed_idx"], columns="variant", values="rho").reset_index()
    v1["d"] = v1.raw - v1.z
    mb = mb.merge(v1[["cond", "rule", "seed_idx", "raw", "z", "d"]], on=["cond", "rule", "seed_idx"])
    s5 = pd.read_csv(C.REPO / "results" / "step5" / "moment_ratio.csv")
    s5 = s5[s5.kind == "checkpoint"].copy()
    s5["cond"] = s5.arm + "_" + s5.res.astype(int).astype(str)

    verdicts = []
    bpm = s5[s5.rule == BP].pivot(index="seed_idx", columns="cond", values="m")
    order_ok = int(((bpm["UPSAMPLED_224"] < bpm["NATIVE_224"]) & (bpm["NATIVE_224"] < bpm["NATIVE_32"])).sum())
    three = ["NATIVE_32", "NATIVE_224", "UPSAMPLED_224"]
    rng = {r: float((lambda p: (p[three].max(axis=1) - p[three].min(axis=1)).mean())(
        s5[s5.rule == r].pivot(index="seed_idx", columns="cond", values="m"))) for r in (R_, BP)}
    verdicts.append(dict(kind="rule", quantity="P_mech1 BP m(UP224) < m(NAT224) < m(NAT32), seeds", value=order_ok,
                         passed=bool(order_ok >= 4)))
    verdicts.append(dict(kind="rule", quantity="P_mech1 Random range < 0.5 x BP range", value=rng[R_],
                         passed=bool(rng[R_] < 0.5 * rng[BP]), note=f"BP range {rng[BP]:.4f}"))
    bpn = mb[mb.rule == BP].pivot(index="seed_idx", columns="cond", values="m")
    n_notch = int((bpn["notch_0-14"] > bpn["only_0-14"]).sum())
    verdicts.append(dict(kind="rule", quantity="P_mech2 m_BP(notch) > m_BP(only), seeds", value=n_notch,
                         passed=bool(n_notch == 5)))
    cells = pd.concat([s5[s5.rule.isin([R_, BP]) & s5.cond.isin(three)][["cond", "rule", "seed_idx", "m", "d"]],
                       mb[mb.rule.isin([R_, BP])][["cond", "rule", "seed_idx", "m", "d"]]], ignore_index=True)
    rho = spearmanr(cells.m, cells.d).statistic
    verdicts.append(dict(kind="rule", quantity="P_mech2 Spearman(m, d), Random+BP x 5 conditions", value=rho,
                         passed=bool(rho >= 0.6), note=f"n = {len(cells)}"))
    for r in (R_, BP):
        c = cells[cells.rule == r]
        verdicts.append(dict(kind="descriptive", quantity=f"Spearman(m, d) within {r}", value=spearmanr(c.m, c.d).statistic,
                             note=f"n = {len(c)}"))

    msum = pd.concat([mb, s5[s5.cond.isin(three)][["cond", "rule", "seed_idx", "m", "raw", "z", "d"]]],
                     ignore_index=True).groupby(["cond", "rule"])[["m", "raw", "z", "d"]].agg(["mean", "std"])
    msum.columns = [f"{a}_{b}" for a, b in msum.columns]
    msum = msum.reset_index()
    msum["kind"] = "condition_summary"
    mb["kind"] = "checkpoint"
    C.stamp(pd.concat([mb, msum], ignore_index=True)).to_csv(OUT / "moment_ratio_bandpass.csv", index=False)
    C.stamp(pd.concat([pts, comp, pd.DataFrame(verdicts)], ignore_index=True)).to_csv(OUT / "mechanism_bandpass.csv", index=False)

    fper, fsum = filter_spectra()
    fper["kind"], fsum["kind"] = "filter", "condition_summary"
    C.stamp(pd.concat([fper, fsum], ignore_index=True)).to_csv(OUT / "filter_spectra.csv", index=False)
    C.stamp(stimulus_spectra()).to_csv(OUT / "stimulus_spectra.csv", index=False)

    pd.set_option("display.width", 220)
    print(comp.to_string(index=False))
    print(pd.DataFrame(verdicts).to_string(index=False))
    print(msum[msum.rule.isin([R_, BP])].to_string(index=False))
    print(fsum.to_string(index=False))


if __name__ == "__main__":
    main()

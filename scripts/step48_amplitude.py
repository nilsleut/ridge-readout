"""
step48_amplitude.py
===================
Step 48 (amendments/2026-11-22_step48-predictions.md, config_step48.yaml): response amplitude.

  g    per trial: cosine of the ROI voxel pattern with the mean over the analysed patterns of the
       same convention (brain gain)
  g^M  per model (condition, seed): cosine of the raw GAP feature vector of each image with the
       mean feature vector over the images (model gain)
Spearman of g with the image statistics (image_stats: L, C, CE, OFF, ON, DD), of g with g^M, and of
g^M with the statistics. Datasets: THINGS-720 (patterns of step48_conventions.things_data; Conv1
for V1, Conv3 for LOC), THINGS test images (k = 12 mean; Conv1 test100), NSD (k = 1 per
presentation, averaged over the three, and k = 3 mean; Conv1, g = 1).

summary: NB1-NB3 from results/step48/conventions.csv, G1-G3 from amplitude.csv, interpretation
of the amendment -> results/step48/summary.csv.

Per-image statistics and per-trial gains stay in memory or under RIDGE_DATA_DIR/step48; the
repository gets Spearman values and aggregates only.

Usage:
  python scripts/step48_amplitude.py things
  python scripts/step48_amplitude.py nsd
  python scripts/step48_amplitude.py summary
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads, sets the import paths)

import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import image_stats as IS
import step48_conventions as S48

OUT = S48.OUT
SUBJ = S48.SUBJ
SEEDS = range(5)


def gain(X):
    """X (n, d) -> (n,) cosine of each row with the mean row."""
    X = np.asarray(X, np.float64)
    m = X.mean(axis=0)
    return (X @ m) / (np.linalg.norm(X, axis=1) * np.linalg.norm(m))


def spear(a, b):
    return float(spearmanr(a, b).correlation)


def dataset_stats(name):
    """{stat: (n,)} cached under RIDGE_DATA_DIR/step48 (per-image values never in the repository)."""
    f = S48.work_dir() / f"image_stats_{name}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    s = IS.stats_for(name)
    np.savez(f, **{k: np.asarray(v) for k, v in s.items()})
    return {k: np.asarray(v) for k, v in s.items()}


def stats_summary(name, s):
    rows = [dict(kind="image_stats", dataset=name, stat=k, n=len(s[k]), mean=float(s[k].mean()), sd=float(s[k].std()))
            for k in IS.STATS]
    rows.append(dict(kind="image_stats", dataset=name, stat="ref (OFF / ON reference)", value=float(s["ref"])))
    rows += [dict(kind="image_stats_spearman", dataset=name, stat=f"{a}-{b}", value=spear(s[a], s[b]))
             for i, a in enumerate(IS.STATS) for b in IS.STATS[i + 1:]]
    rows.append(dict(kind="check", dataset=name, quantity="L, C vs step 44 max |d|", value=max(float(s["check_L"]), float(s["check_C"]))))
    return rows


def model_gain_rows(dataset, layer, feats, stats):
    """feats {(condition, seed): (n, d)} -> ({(condition, seed): gM}, rows of Spearman(gM, stat))."""
    gm = {k: gain(X) for k, X in feats.items()}
    rows = [dict(kind="model_gain_stat", dataset=dataset, layer=layer, condition=k[0], seed=k[1], stat=st, rho=spear(v, stats[st]))
            for k, v in gm.items() for st in IS.STATS]
    return gm, rows


def brain_rows(dataset, convention, roi, subject, g_list, gm, stats, k):
    """g_list: list of per-trial gains (one per presentation; averaged Spearman); gm: model gains."""
    rows = [dict(kind="brain_gain_stat", dataset=dataset, convention=convention, roi=roi, subject=subject, k=k, stat=st,
                 rho=float(np.mean([spear(g, stats[st]) for g in g_list]))) for st in IS.STATS]
    for (lab, si), v in gm.items():
        rows.append(dict(kind="brain_model_gain", dataset=dataset, convention=convention, roi=roi, subject=subject, k=k,
                         condition=lab, seed=si, rho=float(np.mean([spear(g, v) for g in g_list]))))
    rows.append(dict(kind="brain_gain", dataset=dataset, convention=convention, roi=roi, subject=subject, k=k,
                     mean=float(np.mean([g.mean() for g in g_list])), sd=float(np.mean([g.std() for g in g_list]))))
    return rows


def run_things():
    import step40_common as S40
    import step44_why as W
    import step7_multiverse as S7
    t0 = time.time()
    st720, stt = dataset_stats("things720"), dataset_stats("things_test")
    rows = stats_summary("things720", st720) + stats_summary("things_test", stt)
    gm720 = {}
    for roi, layer in W.LAYER.items():
        feats = {(lab, si): S7.feat("NATIVE", 224, rule, si, layer) for lab, rule in W.RULES.items() for si in SEEDS}
        gm720[roi], r = model_gain_rows("things720", layer, feats, st720)
        rows += r
    feats_t = {(lab, si): np.load(S40.feature_dir() / "test100" / f"model_{key}_seed{si}_g1.npy")
               for key, lab in {"random": "Random", "backprop": "BP"}.items() for si in SEEDS}
    gmt, r = model_gain_rows("things_test", "Conv1", feats_t, stt)
    rows += r
    for s in SUBJ:
        t = S48.things_data(s)
        loc = S48.local_rois(s)
        _, rois_t = S40.load_reps(s)
        for cv in S48.T_CONV:
            for roi in ("V1", "LOC"):
                rows += brain_rows("things720", cv, roi, s, [gain(t[f"p720_{cv}"][:, loc[roi]])], gm720[roi], st720, 1)
                rows += brain_rows("things_test", cv, roi, s, [gain(t[f"reps_{cv}"][:, :, rois_t[roi]].mean(axis=1))], gmt, stt, 12)
    rows.append(dict(kind="runtime", quantity="THINGS amplitude seconds", value=time.time() - t0))
    S48.write_part(OUT / "amplitude.csv", rows, "THINGS")
    print(f"THINGS amplitude done ({time.time() - t0:.0f} s)")


def run_nsd():
    import nsd_raw_fetch as RF
    from nsd_rsa import load_features
    t0 = time.time()
    st = dataset_stats("nsd515")
    rows = stats_summary("nsd515", st)
    gm, r = model_gain_rows("nsd515", "Conv1", {k: np.asarray(v, np.float64) for k, v in load_features().items()}, st)
    rows += r
    for s in C.cfg()["subjects"]:
        conv, _, _, rois, _ = RF.conventions(s)
        for cv in S48.N_CONV:
            for roi in ("V1", "LO-obj", "lateral"):
                P = conv[cv][:, :, rois[roi]]
                rows += brain_rows("nsd515", cv, roi, s, [gain(P[:, r]) for r in range(P.shape[1])], gm, st, 1)
                rows += brain_rows("nsd515", cv, roi, s, [gain(P.mean(axis=1))], gm, st, 3)
        print(f"  {s} ({time.time() - t0:.0f} s)", flush=True)
    rows.append(dict(kind="runtime", quantity="NSD amplitude seconds", value=time.time() - t0))
    S48.write_part(OUT / "amplitude.csv", rows, "NSD")
    print(f"NSD amplitude done ({time.time() - t0:.0f} s)")


# ── summary ─────────────────────────────────────────────────────────────────
def model_contrast(a, dataset, convention, roi, k):
    """Per subject: mean over seeds of Spearman(g, gM Random) and of Spearman(g, gM BP)."""
    q = a[(a.kind == "brain_model_gain") & (a.dataset == dataset) & (a.convention == convention) & (a.roi == roi) & (a.k == k)]
    return q.groupby(["subject", "condition"]).rho.mean().unstack("condition")


def run_summary():
    from nsd_rsa import sign_flip_p
    cv = pd.read_csv(OUT / "conventions.csv", low_memory=False)
    a = pd.read_csv(OUT / "amplitude.csv", low_memory=False)
    subs = C.cfg()["subjects"]
    ns = cv[(cv.kind == "nsd_subject") & (cv.roi == "V1") & (cv.k == 1) & (cv.units == "raw") & (cv.distance == "correlation")]
    ns = ns.pivot_table(index="subject", columns="convention", values="delta").loc[subs]
    nb1 = int((ns["N-raw"] > ns["N-session"]).sum())
    nb2n = int((ns["N-raw"] > 0).sum())
    nb2p = sign_flip_p(ns["N-raw"].to_numpy(), "positive")
    t = cv[(cv.kind == "things720") & (cv.roi == "V1") & (cv.units == "raw") & (cv.distance == "correlation")].set_index("convention").delta
    b = a[(a.kind == "brain_gain_stat") & (a.dataset == "things720") & (a.convention == "T-global") & (a.roi == "V1")]
    b = b.pivot_table(index="subject", columns="stat", values="rho").loc[SUBJ]
    g1 = int((b["CE"].abs() > b["L"].abs()).sum())
    m2 = model_contrast(a, "things720", "T-global", "V1", 1).loc[SUBJ]
    g2 = int((m2["Random"] > m2["BP"]).sum())
    m3 = model_contrast(a, "nsd515", "N-raw", "V1", 1).loc[subs]
    g3 = int((m3["Random"] > m3["BP"]).sum())
    met = {"NB1": nb1 >= 6, "NB2": nb2n >= 6 and nb2p <= 0.05, "NB3": bool(t["T-session"] <= 0.010),
           "G1": g1 >= 2, "G2": g2 == 3, "G3": g3 >= 6}
    if met["NB1"] and met["NB2"] and met["G2"] and met["G3"]:
        interp = ("the untrained advantage is attributed to a common voxel response profile left in by the normalisation, "
                  "through per-image response amplitude that the untrained network tracks better; in both datasets")
    elif met["NB1"] and not met["NB2"]:
        interp = "the normalisation convention moves NSD in the predicted direction but does not create an untrained advantage there"
    elif not met["NB1"]:
        interp = "the mechanism is not supported in NSD"
    else:
        interp = "no statement of the rule applies (NB1 and NB2 met, G2 or G3 not met)"
    fmt = lambda x: "; ".join(f"{k} {v:+.4f}" for k, v in x.items())  # noqa: E731
    rows = [dict(rule="NB1", quantity="NSD V1 paper cell k = 1: Delta(N-raw) > Delta(N-session) in >= 6 of 8", value=nb1,
                 passed=met["NB1"], note=f"mean N-raw {ns['N-raw'].mean():+.4f}, N-global {ns['N-global'].mean():+.4f}, "
                                         f"N-session {ns['N-session'].mean():+.4f}"),
            dict(rule="NB2", quantity="NSD V1 paper cell k = 1: Delta(N-raw) > 0 in >= 6 of 8 and sign-flip p (positive) <= 0.05",
                 value=nb2n, passed=met["NB2"], note=f"{nb2n}/8 positive, p {nb2p:.4f}; per subject {fmt(ns['N-raw'])}"),
            dict(rule="NB3", quantity="THINGS-720 V1 paper cell: Delta(T-session) <= +0.010", value=float(t["T-session"]),
                 passed=met["NB3"], note=f"T-raw {t['T-raw']:+.4f}, T-global {t['T-global']:+.4f}, T-session {t['T-session']:+.4f}"),
            dict(rule="G1", quantity="THINGS-720 T-global V1: |rho(g, CE)| > |rho(g, L)| in >= 2 of 3", value=g1, passed=met["G1"],
                 note="; ".join(f"{s} CE {b.loc[s, 'CE']:+.3f} L {b.loc[s, 'L']:+.3f}" for s in SUBJ)),
            dict(rule="G2", quantity="THINGS-720 T-global V1: rho(g, gM Random) > rho(g, gM BP) in 3 of 3", value=g2, passed=met["G2"],
                 note="; ".join(f"{s} R {m2.loc[s, 'Random']:+.3f} BP {m2.loc[s, 'BP']:+.3f}" for s in SUBJ)),
            dict(rule="G3", quantity="NSD N-raw k = 1 V1: rho(g, gM Random) > rho(g, gM BP) in >= 6 of 8", value=g3, passed=met["G3"],
                 note="; ".join(f"{s} R {m3.loc[s, 'Random']:+.3f} BP {m3.loc[s, 'BP']:+.3f}" for s in subs)),
            dict(rule="interpretation", quantity="amendment rule", note=interp)]
    sm = pd.DataFrame(rows)
    S48.stamp(sm).to_csv(OUT / "summary.csv", index=False)
    stats = a[a.kind.isin(["image_stats", "image_stats_spearman", "check"]) & a.dataset.notna()]
    S48.stamp(stats.drop(columns=["config_hash", "git_head", "seed"], errors="ignore").dropna(axis=1, how="all")).to_csv(
        OUT / "image_stats_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(sm[["rule", "value", "passed", "note"]].to_string(index=False))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "things":
        run_things()
    elif which == "nsd":
        run_nsd()
    elif which == "summary":
        run_summary()
    else:
        raise SystemExit("usage: step48_amplitude.py {things|nsd|summary}")

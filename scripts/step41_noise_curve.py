"""
step41_noise_curve.py
=====================
Step 41, part N (amendments/2026-11-16_step41-predictions.md): noise curve on the 100
repeated THINGS-fMRI test images. EXPLORATORY.

For k in {1, 2, 3, 4, 6, 12}: per draw, k of the 12 sessions (rng(20261116).choice without
replacement, drawn in the order of k, 20 draws per k, one draw for k = 12; the same sessions
for all images and subjects of a draw); each image's presentations from those sessions are
averaged and the brain RDMs are built as in step 40 part C (b1 correlation and b2 Euclidean on
the averages, b3 cross-validated over the k sessions, k >= 2). V1 and LOC, all ROI voxels,
single presentations from step40_common.load_reps (z-scored over all trials, as in step 40).
Models as in step 40 part C: GAP Conv1 features of the 100 images (step40_features/test100),
Random Weights and Backprop x 5 seeds, raw or standardised over the 100 images, correlation or
Euclidean distance. Spearman over the 4950 pairs, per subject; mean over draws, subjects and
seeds. Delta(k) = Random - BP. Stimulus bootstrap (1000 resamples, seed 20261002, self-pairs
dropped, same draws as cvrsa.boot_draws) for k = 1 and k = 12.

N1: raw, correlation vs b1, V1: Delta(1) > Delta(12).  N2: same cell: Spearman(k, Delta(k)) over
the six k <= -0.8.  N3: same cell: Delta(1) > 0.  N4: z, correlation vs b1, V1: Delta(k) <= 0 for
every k. Point estimates. Other cells reported without predictions.

Output: results/step41/noise_curve.csv, rows added to results/step41/summary.csv,
        results/step41/noise_curve_v1.png
"""
import step40_common as S     # noqa: I001

import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr

from config_hash import config_hash, load_config
from cvrsa import boot_draws, cv_sq_euclid, model_rdm

CONFIG = S.REPO / "config_step41.yaml"
OUT = S.REPO / "results" / "step41"
SUBJ = ["sub-01", "sub-02", "sub-03"]
LAB = {"random": "Random", "backprop": "BP"}


def cfg():
    return load_config(CONFIG)


def session_draws(c=None):
    c = (c or cfg())["noise_curve"]
    rng = np.random.default_rng(c["draw_seed"])
    out = {}
    for k in c["ks"]:
        n = 1 if k == 12 else c["n_draws"]
        out[k] = [np.sort(rng.choice(12, k, replace=False)) for _ in range(n)]
    return out


def brain_for(reps, sessions):
    """Brain RDMs from the presentations of the given sessions."""
    sub = np.asarray(reps, np.float64)[:, sessions, :]
    avg = sub.mean(axis=1)
    out = {"b1": pdist(avg, "correlation"), "b2": pdist(avg, "euclidean")}
    if len(sessions) >= 2:
        out["b3"] = cv_sq_euclid(sub)
    return out


def rank_z(V):
    """Rows ranked (average ties), centred and scaled, so that z_a @ z_b is Spearman's rho."""
    R = np.apply_along_axis(rankdata, 1, np.atleast_2d(V)).astype(np.float64)
    R -= R.mean(axis=1, keepdims=True)
    return R / np.sqrt((R * R).sum(axis=1, keepdims=True))


def model_rdms():
    c = cfg()["noise_curve"]
    out = {}
    for key, lab in LAB.items():
        for si in cfg()["seed_idx"]:
            X = np.load(S.feature_dir() / "test100" / f"model_{key}_seed{si}_g1.npy")
            for norm in c["model_normalisation"]:
                for dist in ("correlation", "euclidean"):
                    out[(lab, si, norm, dist)] = model_rdm(X, norm, dist)
    return out


def main():
    c = cfg()["noise_curve"]
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    draws = session_draws()
    models = model_rdms()
    mkeys = list(models)
    cells = [(cl["model_distance"], cl["brain_rdm"]) for cl in c["cells"]]

    # brain RDMs per subject, roi, k, draw
    brains = {}
    for s in SUBJ:
        reps, rois = S.load_reps(s)
        for roi in c["rois"]:
            R = reps[:, :, rois[roi]]
            for k, ds in draws.items():
                for di, sess in enumerate(ds):
                    for b, vec in brain_for(R, sess).items():
                        brains[(s, roi, k, di, b)] = vec
    bkeys = list(brains)
    Zm = rank_z(np.stack([models[k] for k in mkeys]))
    Zb = rank_z(np.stack([brains[k] for k in bkeys]))
    rho = Zm @ Zb.T                                          # (models, brains)
    rows = []
    mi = {k: i for i, k in enumerate(mkeys)}
    for j, (s, roi, k, di, b) in enumerate(bkeys):
        dist_ok = "correlation" if b == "b1" else "euclidean"
        for (lab, si, norm, dist), i in mi.items():
            if dist != dist_ok:
                continue
            rows.append(dict(kind="rho", subject=s, roi=roi, k=k, draw=di, condition=lab, seed=si,
                             model_units=norm, model_distance=dist, brain_rdm=b, rho=float(rho[i, j])))
    rho_df = pd.DataFrame(rows)
    print(f"point estimates ({time.time() - t0:.0f} s)", flush=True)

    # contrasts (point estimates)
    g = rho_df.groupby(["roi", "model_units", "model_distance", "brain_rdm", "k", "condition"]).rho.mean().unstack()
    con = (g["Random"] - g["BP"]).rename("delta").reset_index()
    con[["rho_random", "rho_bp"]] = g[["Random", "BP"]].to_numpy()

    # bootstrap for k = 1 and k = 12
    bd = boot_draws(100, c["bootstrap"]["n_resamples"], c["bootstrap"]["seed"])
    bk = [i for i, key in enumerate(bkeys) if key[2] in c["bootstrap"]["ks"]]
    sel = [bkeys[i] for i in bk]
    Bm = np.stack([models[k] for k in mkeys])
    Bb = np.stack([brains[k] for k in sel])
    acc = {}
    for bi, idx in enumerate(bd):
        r = rank_z(Bm[:, idx]) @ rank_z(Bb[:, idx]).T
        for j, (s, roi, k, di, b) in enumerate(sel):
            dist_ok = "correlation" if b == "b1" else "euclidean"
            for (lab, si, norm, dist), i in mi.items():
                if dist != dist_ok:
                    continue
                key = (roi, norm, dist, b, k)
                a = acc.setdefault(key, np.zeros((2, len(bd))))
                a[0 if lab == "Random" else 1, bi] += r[i, j]
    ci = {}
    for key, a in acc.items():
        roi, norm, dist, b, k = key
        n = len(SUBJ) * len(cfg()["seed_idx"]) * len(draws[k])
        d = (a[0] - a[1]) / n
        ci[key] = (float(np.quantile(d, .025)), float(np.quantile(d, .975)))
    con["ci_lo"] = [ci.get((r.roi, r.model_units, r.model_distance, r.brain_rdm, r.k), (np.nan,))[0]
                    for r in con.itertuples()]
    con["ci_hi"] = [ci.get((r.roi, r.model_units, r.model_distance, r.brain_rdm, r.k), (np.nan, np.nan))[1]
                    for r in con.itertuples()]
    print(f"bootstrap ({time.time() - t0:.0f} s)", flush=True)

    def delta(roi, norm, dist, b, k):
        q = con[(con.roi == roi) & (con.model_units == norm) & (con.model_distance == dist) & (con.brain_rdm == b)
                & (con.k == k)]
        assert len(q) == 1
        return float(q.delta.iloc[0])

    ks = c["ks"]
    raw = [delta("V1", "raw", "correlation", "b1", k) for k in ks]
    zc = [delta("V1", "z", "correlation", "b1", k) for k in ks]
    sp = float(spearmanr(ks, raw)[0])
    rules = [
        dict(kind="rule", part="N", quantity="N1 raw, correlation vs b1, V1: Delta(1) > Delta(12)",
             value=raw[0] - raw[-1], passed=bool(raw[0] > raw[-1]), note=f"Delta(1) {raw[0]:+.4f}, Delta(12) {raw[-1]:+.4f}"),
        dict(kind="rule", part="N", quantity="N2 same cell: Spearman(k, Delta(k)) <= -0.8",
             value=sp, passed=bool(sp <= -0.8)),
        dict(kind="rule", part="N", quantity="N3 same cell: Delta(1) > 0", value=raw[0], passed=bool(raw[0] > 0)),
        dict(kind="rule", part="N", quantity="N4 z, correlation vs b1, V1: Delta(k) <= 0 for every k",
             value=max(zc), passed=bool(max(zc) <= 0), note="max Delta(k) over k"),
        dict(kind="runtime", part="N", quantity="wall seconds", value=time.time() - t0),
    ]
    con["kind"] = "contrast"
    rho_df["seed"] = rho_df["seed"].astype("Int64")
    out = pd.concat([rho_df, con], ignore_index=True)
    out["seed"] = out["seed"].astype("Int64")
    S.OUT = OUT
    out["config_hash"], out["git_head"] = config_hash(cfg()), S.git_head()
    out.to_csv(OUT / "noise_curve.csv", index=False)
    summ_path = OUT / "summary.csv"
    prev = pd.read_csv(summ_path) if summ_path.exists() else pd.DataFrame()
    if len(prev) and "part" in prev:
        prev = prev[prev.part != "N"]
    new = pd.DataFrame(rules)
    new["config_hash"], new["git_head"] = config_hash(cfg()), S.git_head()
    summ = pd.concat([prev, new], ignore_index=True)
    if "seed" in summ:
        summ["seed"] = summ["seed"].astype("Int64")
    summ.to_csv(summ_path, index=False)
    figure(con, ks)
    pd.set_option("display.width", 200)
    print(con.drop(columns=["kind"]).to_string(index=False))
    print(new.drop(columns=["config_hash", "git_head"]).to_string(index=False))


def figure(con, ks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = [("raw", "correlation", "b1", "raw units, correlation (b1)", "#D55E00", "o"),
              ("z", "correlation", "b1", "standardised units, correlation (b1)", "#0072B2", "s"),
              ("raw", "euclidean", "b2", "raw units, Euclidean (b2)", "#E69F00", "^"),
              ("z", "euclidean", "b2", "standardised units, Euclidean (b2)", "#56B4E9", "v")]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.axhline(0, color="black", lw=0.6)
    for j, (norm, dist, b, lab, col, mk) in enumerate(series):
        q = con[(con.roi == "V1") & (con.model_units == norm) & (con.model_distance == dist)
                & (con.brain_rdm == b)].set_index("k").reindex(ks)
        x = np.array(ks, float) * (1 + 0.025 * (j - 1.5))
        ax.plot(x, q.delta, color=col, marker=mk, ms=4, lw=1.2, label=lab)
        for kk, xx in zip(ks, x):
            if np.isfinite(q.loc[kk, "ci_lo"]):
                ax.plot([xx, xx], [q.loc[kk, "ci_lo"], q.loc[kk, "ci_hi"]], color=col, lw=1.0)
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("Sessions averaged per image (k)")
    ax.set_ylabel("Random - Backprop, Spearman (V1)")
    ax.set_title("Noise curve on the 100 test images; bars: 95% CI at k = 1 and 12", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "noise_curve_v1.png", dpi=150)


if __name__ == "__main__":
    main()

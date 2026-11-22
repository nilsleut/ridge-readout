"""
step47c_decompose.py
====================
Step 47c (amendments/2026-11-22_step47c-predictions.md, config_step47c.yaml): decomposition of run
centring in THINGS-720.

Per subject and ROI, after the existing z-score (step47_runs.load_resp):
  m      voxel mean over all 820 trials of the session
  mu_r   voxel mean over the 82 trials of run r;  d_r = mu_r - m
  m_hat  m / ||m||;  d_par = (d_r . m_hat) m_hat;  d_perp = d_r - d_par
Variants of every one of the 720 patterns x (run r):
  V0 x;  V1 x - m;  V2 x - d_r;  V3 x - m - d_r (= run centring);  V4 V3 + d_par;  V5 V3 + d_perp
Delta = paper cell (step44_why.delta_on over the cross-run pairs, Random - BP, mean over seeds,
then subjects).

  Z1  Delta(V2) <= +0.015        (V1 ROI)
  Z2  Delta(V4) >= +0.020 and Delta(V5) <= +0.012
  Z3  across-run SD of d_r . m_hat smallest in sub-03

Imported unchanged: step47_runs (load_resp, trial_rows, SUBJ), step44_why (things720_models,
crossrun_index, delta_on, lum_contrast, things720_images, spear). Nothing voxel-level is written.

Usage:
  python scripts/step47c_decompose.py run
"""
import nsd_common as C     # noqa: I001,F401  (pins BLAS threads, sets the import paths)

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from config_hash import config_hash, load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config_step47c.yaml"
OUT = REPO / "results" / "step47c"
VARIANTS = ["V0", "V1", "V2", "V3", "V4", "V5"]
ROIS = ["V1", "LOC"]
TOL = 1e-12
PAPER_CELL = 0.030445938


def cfg():
    return load_config(CONFIG)


def stamp(df):
    df["config_hash"] = config_hash(cfg())
    df["git_head"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
    if "seed" not in df:
        df["seed"] = np.nan
    df["seed"] = df["seed"].astype("Int64")
    return df


# ── decomposition (pure functions) ──────────────────────────────────────────
def components(S, lab_s):
    """S (n_session, v) all trials of the session, lab_s their run labels.
    Returns (runs, m (v,), M (n_runs, v) run means, D (n_runs, v) run deviations)."""
    runs = np.unique(lab_s)
    m = S.mean(axis=0)
    M = np.stack([S[lab_s == r].mean(axis=0) for r in runs])
    return runs, m, M, M - m


def split(D, m):
    """(m_hat, s = D . m_hat (n_runs,), D_par, D_perp)."""
    m_hat = m / np.linalg.norm(m)
    s = D @ m_hat
    D_par = np.outer(s, m_hat)
    return m_hat, s, D_par, D - D_par


def variants(X, lab_x, runs, m, D):
    """{V0..V5: (n, v)} for patterns X (n, v) with run labels lab_x."""
    k = np.searchsorted(runs, lab_x)
    if (k >= len(runs)).any() or not np.array_equal(runs[k], lab_x):
        raise SystemExit("STOP: a pattern's run is not a run of the session")
    _, _, D_par, D_perp = split(D, m)
    v3 = X - m - D[k]
    return {"V0": X, "V1": X - m, "V2": X - D[k], "V3": v3, "V4": v3 + D_par[k], "V5": v3 + D_perp[k]}


def cosine(X, u):
    return (X @ u) / np.linalg.norm(X, axis=1) / np.linalg.norm(u)


# ── data ────────────────────────────────────────────────────────────────────
def subject_data(subject):
    """(S session trials (820, n_roi) float64, lab_s, X (720, n_roi), lab_x, local ROI index, betas)."""
    import step47_runs as R
    resp, a, st = R.load_resp(subject)
    rows = R.trial_rows(st)
    if not np.array_equal(a["betas"], resp[rows]):
        raise SystemExit(f"STOP: {subject}: the 720 rows differ from the arm_a betas")
    ses = st.session.to_numpy()[rows]
    if len(set(ses)) != 1:
        raise SystemExit(f"STOP: {subject}: the 720 are not in one session")
    in_ses = st.session.to_numpy() == ses[0]
    lab = st.run.to_numpy()
    n_per = pd.Series(lab[in_ses]).value_counts()
    if in_ses.sum() != 820 or len(n_per) != 10 or not (n_per == 82).all():
        raise SystemExit(f"STOP: {subject}: the session does not have 10 runs of 82 trials")
    X = resp.astype(np.float64)
    local = {roi: a[f"local_{roi}"] for roi in ROIS}
    return X[in_ses], lab[in_ses], X[rows], lab[rows], local, a["betas"]


# ── driver ──────────────────────────────────────────────────────────────────
def run():
    import step44_why as W
    import step47_runs as R
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ci = W.crossrun_index()
    models = {roi: W.things720_models(roi) for roi in ROIS}
    L, _ = W.lum_contrast(W.things720_images())
    t_lum = time.time() - t0
    ref = pd.read_csv(REPO / "results" / "step47" / "run_centring.csv")
    ref = ref[ref.kind == "delta"].set_index(["roi", "variant"]).delta

    brains = {(roi, v): [] for roi in ROIS for v in VARIANTS}
    rows, checks = [], []
    for s in R.SUBJ:
        S, lab_s, X, lab_x, local, betas = subject_data(s)
        for roi in ROIS:
            c = local[roi]
            runs, m, M, D = components(S[:, c], lab_s)
            gap = float(np.abs(m + D - M).max())
            if gap > TOL:
                raise SystemExit(f"STOP: {s} {roi}: m + d_r differs from the run mean by {gap}")
            checks.append(dict(kind="check", subject=s, roi=roi, quantity="max |m + d_r - mu_r|", value=gap))
            m_hat, sc, D_par, D_perp = split(D, m)
            P = variants(X[:, c], lab_x, runs, m, D)
            if not np.array_equal(P["V0"], betas[:, c].astype(np.float64)):
                raise SystemExit(f"STOP: {s} {roi}: V0 differs from the arm_a betas")
            for v in VARIANTS:
                brains[(roi, v)].append(pdist(P[v], "correlation"))
            cos0, cos3 = cosine(P["V0"], m_hat), cosine(P["V3"], m_hat)
            rows.append(dict(kind="run_deviation", subject=s, roi=roi, n_voxels=len(c), norm_m=float(np.linalg.norm(m)),
                             sd_d_dot_mhat=float(sc.std()), mean_d_dot_mhat=float(sc.mean()),
                             share_along_mhat=float((sc ** 2).sum() / (D ** 2).sum()),
                             mean_norm_d=float(np.linalg.norm(D, axis=1).mean()),
                             rho_cos_lum_without=W.spear(cos0, L), rho_cos_lum_with=W.spear(cos3, L),
                             mean_cos_without=float(cos0.mean()), mean_cos_with=float(cos3.mean()),
                             d_dot_mhat=";".join(f"{x:+.4f}" for x in sc)))
    res = {}
    for roi in ROIS:
        for v in VARIANTS:
            D_, per = W.delta_on(models[roi], brains[(roi, v)], ci)
            res[(roi, v)] = D_
            rows += [dict(kind="subject_delta", roi=roi, variant=v, subject=s, delta=float(x)) for s, x in zip(R.SUBJ, per)]
            rows.append(dict(kind="delta", roi=roi, variant=v, delta=D_))
    for roi, v, name in (("V1", "V0", "raw"), ("V1", "V3", "run-centred"), ("LOC", "V0", "raw"), ("LOC", "V3", "run-centred")):
        dev = abs(res[(roi, v)] - ref[(roi, name)])
        if dev > TOL:
            raise SystemExit(f"STOP: {roi} {v} Delta {res[(roi, v)]} differs from step 47 {name} {ref[(roi, name)]}")
        checks.append(dict(kind="check", roi=roi, variant=v, quantity=f"|Delta {v} - step 47 {name}|", value=dev))
    if abs(res[("V1", "V0")] - PAPER_CELL) > 1e-8:
        raise SystemExit("STOP: V0 does not reproduce the paper cell")
    dec = pd.DataFrame(rows + checks)
    stamp(dec).to_csv(OUT / "decomposition.csv", index=False)

    rd = dec[(dec.kind == "run_deviation") & (dec.roi == "V1")].set_index("subject")
    sd = rd.sd_d_dot_mhat
    d = {v: res[("V1", v)] for v in VARIANTS}
    met = {"Z1": d["V2"] <= 0.015, "Z2": d["V4"] >= 0.020 and d["V5"] <= 0.012, "Z3": sd.idxmin() == "sub-03"}
    if met["Z1"] and met["Z2"]:
        interp = "run-wise shifts of global activation along the voxel mean-profile axis"
    elif not met["Z1"] and d["V1"] <= 0.015:
        interp = ("mainly removal of the session mean profile (equivalent to voxel centring), not between-run "
                  "structure; the description of step 47 as run structure is to be corrected in the manuscript")
    else:
        interp = "not identified"
    summ = [dict(rule="Z1", quantity="Delta(V2) <= +0.015", value=d["V2"], passed=met["Z1"],
                 note=" ".join(f"{v} {d[v]:+.4f}" for v in VARIANTS)),
            dict(rule="Z2", quantity="Delta(V4) >= +0.020 and Delta(V5) <= +0.012", value=d["V4"], passed=met["Z2"],
                 note=f"V4 {d['V4']:+.4f}; V5 {d['V5']:+.4f}"),
            dict(rule="Z3", quantity="across-run SD of d_r . m_hat smallest in sub-03", value=float(sd["sub-03"]),
                 passed=met["Z3"], note="; ".join(f"{k} {x:.4f}" for k, x in sd.items())),
            dict(rule="interpretation", quantity="amendment rule", note=interp),
            dict(rule="runtime", quantity="luminance and models seconds", value=t_lum),
            dict(rule="runtime", quantity="total seconds", value=time.time() - t0)]
    sm = pd.DataFrame(summ)
    stamp(sm).to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(dec[dec.kind == "delta"][["roi", "variant", "delta"]].to_string(index=False))
    print(dec[dec.kind == "subject_delta"].pivot_table(index=["roi", "subject"], columns="variant", values="delta").round(6))
    print(dec[dec.kind == "run_deviation"][["subject", "roi", "sd_d_dot_mhat", "share_along_mhat",
                                            "rho_cos_lum_without", "rho_cos_lum_with"]].to_string(index=False))
    print(sm[["rule", "value", "passed", "note"]].to_string(index=False))


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "") != "run":
        raise SystemExit("usage: step47c_decompose.py run")
    run()

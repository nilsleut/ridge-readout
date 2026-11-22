"""
step48_conventions.py
=====================
Step 48 (amendments/2026-11-22_step48-predictions.md, config_step48.yaml): RSA Delta under three
normalisation conventions in THINGS-720, the THINGS test images and NSD.

THINGS (h5 ROI rows of arm_a roi_idx, all 9840 trials):
  T-raw     h5 values, no z-score
  T-global  (x - mean) / (std + 1e-8) per voxel over all 9840 trials, float32 (= arm_a, the paper)
  T-session the same expression over the trials of one session (the 720: their session of 820;
            test presentations: each of the 12 sessions)
NSD (nsd_raw_fetch.conventions): N-raw (% signal change), N-global (z over all 22,500 trials),
  N-session (= step 42 patterns).

Cells: model units raw / z x distance correlation / euclidean (cvrsa.model_rdm; brain pdist with
the same distance). THINGS-720 over the cross-run pairs (step44_why.delta_on); THINGS test images
as step47b_robustness.things_test_rsa (b1 correlation, b2 euclidean; k = 1 and 12); NSD with
nsd_rsa.rsa_subject (k = 1 and 3) and nsd_rsa.contrasts; group sign-flip p (nsd_rsa.sign_flip_p);
NSD multiverse under N-raw via step47b_robustness.nsd_multiverse.

Gates (STOP): T-global 720 rows = arm_a betas and test presentations = load_reps (exact);
T-global V1 paper cell = +0.030446 (step 47, 1e-12); T-global test-image Delta = step 41 (1e-10);
N-session rho = step 42 subjects.csv (k = 1 and 3, 1e-12).

Voxel-level intermediates: RIDGE_DATA_DIR/step48 (THINGS) only; NSD stays under NSD_DIR.

Usage:
  python scripts/step48_conventions.py things
  python scripts/step48_conventions.py nsd [workers]
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads, sets the import paths)

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

import paths
from config_hash import config_hash, load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config_step48.yaml"
OUT = REPO / "results" / "step48"
SUBJ = ["sub-01", "sub-02", "sub-03"]
T_CONV = ["T-raw", "T-global", "T-session"]
N_CONV = ["N-raw", "N-global", "N-session"]
UNITS = ["raw", "z"]
DISTS = ["correlation", "euclidean"]
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


def write_part(path, rows, part):
    """Replace the rows of `part` in the CSV at path (other parts kept), parts in a fixed order."""
    OUT.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    new["part"] = part
    if path.exists():
        old = pd.read_csv(path, low_memory=False).drop(columns=["config_hash", "git_head", "seed"], errors="ignore")
        new = pd.concat([old[old.part != part], new], ignore_index=True)
    order = {p: i for i, p in enumerate(["THINGS", "NSD", "image_stats", "summary"])}
    new = new.iloc[np.argsort([order.get(p, 9) for p in new.part], kind="stable")]
    stamp(new.reset_index(drop=True)).to_csv(path, index=False)


def zexpr(Y):
    """The arm_a expression on float32: (Y - mean) / (std + 1e-8) per column."""
    Y = np.asarray(Y, np.float32)
    return (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)


# ── THINGS data ─────────────────────────────────────────────────────────────
def work_dir():
    d = paths.ridge_data_dir() / "step48"
    d.mkdir(parents=True, exist_ok=True)
    return d


def things_data(subject):
    """{'p720_<conv>': (720, n_roi) float64, 'reps_<conv>': (100, 12, n_roi) float64}, cached under
    RIDGE_DATA_DIR/step48. Gates: T-global rows = arm_a betas, T-global test presentations = load_reps."""
    cache = work_dir() / f"{subject}_things.npz"
    if cache.exists():
        return dict(np.load(cache))
    import h5py
    import step40_common as S40
    import step47_runs as A
    a = np.load(paths.ridge_data_dir() / "arm_b_out" / f"{subject}_arma.npz")
    with h5py.File(paths.things_fmri_dir() / f"{subject}_task-things_voxel-wise-responses.h5", "r") as f:
        raw = f["ResponseData/block0_values"][a["roi_idx"], :].astype(np.float32).T      # (9840, v)
    st = pd.read_csv(paths.things_fmri_dir() / f"{subject}_task-things_stimulus-metadata.csv")
    ses = st.session.to_numpy()
    glob = zexpr(raw)
    sess = np.empty_like(raw)
    for k in np.unique(ses):
        sess[ses == k] = zexpr(raw[ses == k])
    rows = A.trial_rows(st)
    if not np.array_equal(glob[rows], a["betas"]):
        raise SystemExit(f"STOP: {subject}: T-global rows differ from the arm_a betas")
    if len(set(ses[rows])) != 1 or (ses == ses[rows][0]).sum() != 820:
        raise SystemExit(f"STOP: {subject}: the 720 are not in one session of 820 trials")
    conv = {"T-raw": raw, "T-global": glob, "T-session": sess}
    reps_ref, _ = S40.load_reps(subject)
    idx = []
    for x in S40.test_stimuli(subject):
        r = st.index[st.stimulus == x].to_numpy()
        idx.append(r[np.argsort(ses[r])])
    idx = np.stack(idx)                                                   # (100, 12)
    if not np.array_equal(glob[idx], reps_ref):
        raise SystemExit(f"STOP: {subject}: T-global test presentations differ from load_reps")
    out = {}
    for name, Y in conv.items():
        out[f"p720_{name}"] = Y[rows].astype(np.float64)
        out[f"reps_{name}"] = Y[idx].astype(np.float64)
    np.savez(cache, **out)
    return dict(np.load(cache))


def local_rois(subject):
    a = np.load(paths.ridge_data_dir() / "arm_b_out" / f"{subject}_arma.npz")
    return {roi: a[f"local_{roi}"] for roi in ("V1", "LOC")}


def things720_models(roi, units, dist):
    import step44_why as W
    import step7_multiverse as S7
    return {(lab, si): W.model_rdm(S7.feat("NATIVE", 224, rule, si, W.LAYER[roi]), units, dist)
            for lab, rule in W.RULES.items() for si in W.seeds()}


def things720_cells():
    import step44_why as W
    ci = W.crossrun_index()
    data = {s: things_data(s) for s in SUBJ}
    loc = {s: local_rois(s) for s in SUBJ}
    rows, res = [], {}
    for roi in ("V1", "LOC"):
        for u in UNITS:
            for d in DISTS:
                models = things720_models(roi, u, d)
                for cv in T_CONV:
                    brains = [pdist(data[s][f"p720_{cv}"][:, loc[s][roi]], d) for s in SUBJ]
                    D, per = W.delta_on(models, brains, ci)
                    res[(cv, roi, u, d)] = D
                    rows += [dict(kind="things720_subject", convention=cv, roi=roi, units=u, distance=d, subject=s,
                                  delta=float(x)) for s, x in zip(SUBJ, per)]
                    rows.append(dict(kind="things720", convention=cv, roi=roi, units=u, distance=d, delta=D,
                                     n_subjects_positive=int((per > 0).sum()), n_pairs=len(ci)))
    dev = abs(res[("T-global", "V1", "raw", "correlation")] - PAPER_CELL)
    if dev > 1e-8:
        raise SystemExit(f"STOP: T-global V1 paper cell {res[('T-global', 'V1', 'raw', 'correlation')]} is not the paper cell")
    ref = pd.read_csv(REPO / "results" / "step47" / "run_centring.csv")
    r47 = float(ref[(ref.kind == "delta") & (ref.roi == "V1") & (ref.variant == "raw")].delta.iloc[0])
    dev47 = abs(res[("T-global", "V1", "raw", "correlation")] - r47)
    if dev47 > TOL:
        raise SystemExit(f"STOP: T-global V1 paper cell differs from step 47 ({dev47:.1e})")
    rows.append(dict(kind="check", quantity="T-global V1 paper cell vs step 47 |d|", value=dev47))
    return rows


def things_test_cells():
    import step40_common as S40
    import step41_noise_curve as S41
    from cvrsa import model_rdm
    draws = S41.session_draws()
    KEYS = {"random": "Random", "backprop": "BP"}
    rows = []
    for d, b in (("correlation", "b1"), ("euclidean", "b2")):
        models = {}
        for key, lab in KEYS.items():
            for si in range(5):
                X = np.load(S40.feature_dir() / "test100" / f"model_{key}_seed{si}_g1.npy")
                for u in UNITS:
                    models[(lab, si, u)] = model_rdm(X, u, d)
        mk = list(models)
        Zm = S41.rank_z(np.stack([models[k] for k in mk]))
        for s in SUBJ:
            t = things_data(s)
            _, rois = S40.load_reps(s)
            for cv in T_CONV:
                for roi in ("V1", "LOC"):
                    R = t[f"reps_{cv}"][:, :, rois[roi]]
                    for k in (1, 12):
                        for di, sess in enumerate(draws[k]):
                            rho = Zm @ S41.rank_z(S41.brain_for(R, sess)[b])[0]
                            rows += [dict(subject=s, convention=cv, roi=roi, k=k, draw=di, condition=lab, seed=si, units=u,
                                          distance=d, rho=float(v)) for (lab, si, u), v in zip(mk, rho)]
    rho = pd.DataFrame(rows)
    per = rho.groupby(["convention", "roi", "k", "units", "distance", "subject", "condition"]).rho.mean().unstack("condition")
    per["delta"] = per["Random"] - per["BP"]
    out = [dict(kind="things_test_subject", convention=i[0], roi=i[1], k=int(i[2]), units=i[3], distance=i[4], subject=i[5],
                delta=float(r.delta)) for i, r in per.iterrows()]
    g = per.groupby(level=[0, 1, 2, 3, 4]).delta.agg(["mean", lambda x: int((x > 0).sum())])
    out += [dict(kind="things_test", convention=i[0], roi=i[1], k=int(i[2]), units=i[3], distance=i[4], delta=float(r.iloc[0]),
                 n_subjects_positive=int(r.iloc[1])) for i, r in g.iterrows()]
    ref = pd.read_csv(REPO / "results" / "step41" / "noise_curve.csv")
    ref = ref[(ref.kind == "contrast") & (ref.model_distance == "correlation") & (ref.brain_rdm == "b1")]
    dev = 0.0
    for r in out:
        if r["kind"] == "things_test" and r["convention"] == "T-global" and r["distance"] == "correlation":
            x = ref[(ref.roi == r["roi"]) & (ref.k == r["k"]) & (ref.model_units == r["units"])]
            dev = max(dev, abs(float(x.delta.iloc[0]) - r["delta"]))
    if dev > 1e-10:
        raise SystemExit(f"STOP: T-global test-image Delta does not reproduce step 41 (max |d| {dev:.1e})")
    out.append(dict(kind="check", quantity="T-global test-image Delta vs step 41 max |d|", value=dev))
    return out


def run_things():
    t0 = time.time()
    rows = things720_cells()
    t1 = time.time()
    rows += things_test_cells()
    rows += [dict(kind="runtime", quantity="THINGS-720 seconds", value=t1 - t0),
             dict(kind="runtime", quantity="THINGS test seconds", value=time.time() - t1)]
    write_part(OUT / "conventions.csv", rows, "THINGS")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df[df.kind == "things720"][["convention", "roi", "units", "distance", "delta", "n_subjects_positive"]].to_string(index=False))
    print(df[df.kind == "things_test"][["convention", "roi", "k", "units", "distance", "delta"]].to_string(index=False))
    print(f"THINGS done ({time.time() - t0:.0f} s)")


# ── NSD ─────────────────────────────────────────────────────────────────────
def nsd_subject(subject):
    """rho rows for the three conventions (k = 1 and 3) and the N-raw multiverse brains."""
    import nsd_raw_fetch as RF
    import nsd_step43 as NS
    from nsd_rsa import load_features, model_vectors, rsa_subject
    t = time.time()
    conv, session, run, rois, valid = RF.conventions(subject)
    images = np.load(C.nsd_dir("patterns") / f"{subject}.npz")["images"]
    if not np.array_equal(images, C.design(C.cfg()["pilot_subject"])["images"]):
        raise SystemExit(f"STOP: image order of {subject} differs from the feature order")
    models = model_vectors(load_features())
    rho = []
    for cv in N_CONV:
        r = rsa_subject(conv[cv], session, run, rois, models, k1=True)
        r["convention"], r["subject"] = cv, subject
        rho.append(r)
    bkeys, Bs = NS.mv_brains(conv["N-raw"][:, :, rois["V1"]].mean(axis=1))
    return pd.concat(rho, ignore_index=True), bkeys, Bs, time.time() - t


def nsd_group_rows(rho):
    from nsd_rsa import contrasts, sign_flip_p
    subs = C.cfg()["subjects"]
    out = []
    for (cv, s), q in rho.groupby(["convention", "subject"]):
        for r in contrasts(q).itertuples():
            out.append(dict(kind="nsd_subject", convention=cv, subject=s, roi=r.roi, k=int(r.k), units=r.model_units,
                            distance=r.distance, rho_random=r.rho_random, rho_bp=r.rho_bp, delta=r.delta))
    sd = pd.DataFrame(out)
    for (cv, roi, k, u, d), q in sd.groupby(["convention", "roi", "k", "units", "distance"]):
        x = q.set_index("subject").loc[subs].delta.to_numpy()
        out.append(dict(kind="nsd", convention=cv, roi=roi, k=int(k), units=u, distance=d, delta=float(x.mean()),
                        n_subjects_positive=int((x > 0).sum()), n_subjects_negative=int((x < 0).sum()),
                        p_positive=sign_flip_p(x, "positive"), p_negative=sign_flip_p(x, "negative"), n_subjects=len(x)))
    return out


def run_nsd(workers=4):
    import step47b_robustness as S47b
    C.require_free(cfg()["disk"]["min_free_gb"])
    t0 = time.time()
    subs = C.cfg()["subjects"]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s, r in zip(subs, ex.map(nsd_subject, subs)):
            res[s] = r
            print(f"  {s}: RSA of 3 conventions ({r[3]:.0f} s; {time.time() - t0:.0f} s elapsed)", flush=True)
    rho = pd.concat([res[s][0] for s in subs], ignore_index=True)
    bkeys = res[subs[0]][1]
    Bs = np.stack([res[s][2] for s in subs])
    ref = pd.read_csv(REPO / "results" / "step42" / "subjects.csv", low_memory=False)
    ref = ref[(ref.kind == "rho") & ref.k.isin([1, 3]) & ref.model_units.isin(UNITS)]
    ref = ref.set_index(["subject", "roi", "k", "presentation", "condition", "seed", "model_units", "distance"]).rho
    ns = rho[rho.convention == "N-session"].set_index(["subject", "roi", "k", "presentation", "condition", "seed",
                                                        "model_units", "distance"]).rho
    if len(ns) != len(ref):
        raise SystemExit(f"STOP: N-session has {len(ns)} rho values, step 42 {len(ref)}")
    dev42 = float(np.abs(ns - ref.loc[ns.index]).max())
    if dev42 > TOL:
        raise SystemExit(f"STOP: N-session rho differs from step 42 (max |d| {dev42:.1e})")
    rows = nsd_group_rows(rho)
    t_rsa = time.time() - t0
    t = time.time()
    # multiverse under N-raw (step47b_robustness.nsd_multiverse; its paper-cell check uses the k = 3 rho)
    mv_rows, _, _ = S47b.nsd_multiverse(Bs, bkeys, 8, rho[rho.convention == "N-raw"])
    for r in mv_rows:
        r.pop("variant", None)
        if "run-centred" in str(r.get("quantity", "")):
            r["quantity"] = r["quantity"].replace("run-centred", "N-raw")
    rows += [dict(r, convention="N-raw") for r in mv_rows]
    rows += [dict(kind="check", quantity="N-session rho vs step 42 max |d| (k = 1 and 3)", value=dev42, n=len(ns)),
             dict(kind="runtime", quantity="NSD RSA seconds", value=t_rsa),
             dict(kind="runtime", quantity="NSD multiverse seconds", value=time.time() - t)]
    write_part(OUT / "conventions.csv", rows, "NSD")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    q = df[(df.kind == "nsd") & (df.distance == "correlation")]
    print(q[["convention", "roi", "k", "units", "delta", "n_subjects_positive", "p_positive"]].to_string(index=False))
    print(df[df.kind == "nsd_multiverse_summary"][["quantity", "value"]].to_string(index=False))
    print(f"NSD done ({time.time() - t0:.0f} s)")


def estimate():
    """Hours (x 1.5) for the NSD part from one subject's RSA timing."""
    t = time.time()
    _, _, _, dt = nsd_subject("subj01")
    h = (8 * dt / 4 * 1.3 + 954) * 1.5 / 3600
    print(f"one subject {dt:.0f} s (3 conventions); NSD part x 1.5: {h:.2f} h ({time.time() - t:.0f} s)")
    return h


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "things":
        run_things()
    elif which == "nsd":
        run_nsd(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif which == "estimate":
        h = estimate()
        if h > cfg()["runtime"]["stop_hours_per_part"]:
            raise SystemExit(f"STOP: estimate {h:.2f} h above {cfg()['runtime']['stop_hours_per_part']} h")
    else:
        raise SystemExit("usage: step48_conventions.py {things|nsd [workers]|estimate}")

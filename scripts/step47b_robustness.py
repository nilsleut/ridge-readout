"""
step47b_robustness.py
=====================
Step 47b, part Y (amendments/2026-11-21_step47b-predictions.md, config_step47b.yaml): robustness of
the untrained-vs-trained contrasts to run centring (subtract the voxel mean over all trials of the
same run, after the existing normalisation; in NSD per run within session, before averaging).

  Y1  THINGS ridge (primary analysis of results/prereg), run-centred, g = 4: Random - BP < 0 in 5/5 seeds
  Y2  NSD RSA, run-centred, V1 paper cell and V1 standardised cell: Delta < 0 in >= 6/8 subjects each
  Y3  NSD ridge (step 47 part B), run-centred, g = 4, V1: Random - BP < 0 in >= 6/8 subjects
  Y4  NSD ridge, uncentred, g = 4, all V1 voxels: Random - BP < 0 in >= 6/8 subjects
  Reported: THINGS test images (k = 1, 12; paper and standardised cell), NSD multiverse class counts,
  THINGS ridge g = 1 and 8, LO-obj, ridge with the original voxel selection.

Imported: step47_runs (load_resp, run_centre, trial_rows, percentile), step47_nsd_ridge (fit_one,
reliability, targets, features, alphas), arm_b_betas.split_half_reliability, step40_common
(load_reps, test_stimuli, feature_dir), step41_noise_curve (session_draws, brain_for, rank_z),
cvrsa.model_rdm, nsd_rsa (load_patterns, load_features, model_vectors, rsa_subject, contrasts,
sign_flip_p, boot_indices), nsd_step43 (mv_models, mv_brains, mv_pairs, draw_index, rho_pairs,
summarise, ci_class, distinct_classes), nsd_runmeans.centred_patterns, pilot_fit.load_features.

Voxel-level intermediates: THINGS under RIDGE_DATA_DIR/step47b, NSD under NSD_DIR/step47b.
Only aggregated values are written to the repository.

Usage:
  python scripts/step47b_robustness.py estimate {things|nsd}
  python scripts/step47b_robustness.py things [workers]
  python scripts/step47b_robustness.py nsd [workers]
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads, sets the import paths)

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import paths
from config_hash import config_hash, load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config_step47b.yaml"
OUT = REPO / "results" / "step47b"
SUBJ = ["sub-01", "sub-02", "sub-03"]
GRIDS = [1, 4, 8]
KEYS = {"random": "Random", "backprop": "BP"}
THR = 0.2
TOL = 1e-12


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


def things_work():
    d = paths.ridge_data_dir() / "step47b"
    d.mkdir(parents=True, exist_ok=True)
    return d


def names():
    return [f"model_{k}_seed{si}" for k in KEYS for si in range(5)]


def label_of(name):
    return KEYS[name.split("_")[1]], int(name[-1])


# ── THINGS data ─────────────────────────────────────────────────────────────
def things_data(subject):
    """Cached npz (RIDGE_DATA_DIR/step47b): rc720 (720, n_roi) run-centred arm_a rows, reps_rc
    (100, 12, n_roi) run-centred test presentations (session order), rel_rc (n_roi) split-half
    reliability of reps_rc. Gates: h5 rows = arm_a betas; uncentred test presentations = load_reps."""
    cache = things_work() / f"{subject}_things.npz"
    if cache.exists():
        return dict(np.load(cache))
    import step40_common as S40
    import step47_runs as A
    from arm_b_betas import split_half_reliability
    resp, a, st = A.load_resp(subject)
    key = st.session.to_numpy().astype(np.int64) * 100 + st.run.to_numpy()
    rows = A.trial_rows(st)
    if np.abs(resp[rows] - a["betas"]).max() > 1e-5:
        raise SystemExit(f"STOP: {subject}: z-scored h5 responses differ from the arm_a betas")
    if not (pd.Series(key).value_counts() == 82).all():
        raise SystemExit(f"STOP: {subject}: runs do not have 82 trials")
    r64 = resp.astype(np.float64)
    rc720 = A.run_centre(r64, key, rows)
    reps_raw, _ = S40.load_reps(subject)
    sess = st.session.to_numpy()
    reps_rc = np.empty(reps_raw.shape, np.float64)
    for i, x in enumerate(S40.test_stimuli(subject)):
        r = st.index[st.stimulus == x].to_numpy()
        r = r[np.argsort(sess[r])]
        if not np.array_equal(resp[r], reps_raw[i]):
            raise SystemExit(f"STOP: {subject}/{x}: h5 presentations differ from load_reps")
        reps_rc[i] = A.run_centre(r64, key, r)
    rel_rc = split_half_reliability(reps_rc)
    np.savez(cache, rc720=rc720, reps_rc=reps_rc, rel_rc=rel_rc)
    return dict(np.load(cache))


_TD = {}


def _things_targets(subject, variant):
    """(Y (720, v1), reliable mask) for variant raw (prereg), rc (run-centred, recomputed selection)
    or rc_origsel (run-centred, armb.npz selection)."""
    k = (subject, variant)
    if k not in _TD:
        d = paths.ridge_data_dir() / "arm_b_out"
        a, b = np.load(d / f"{subject}_arma.npz"), np.load(d / f"{subject}_armb.npz")
        v1 = a["local_V1"]
        rel_orig = b["reliability"][v1] > THR
        if variant == "raw":
            _TD[k] = (a["betas"][:, v1].astype(np.float64), rel_orig)
        else:
            t = things_data(subject)
            _TD[k] = (t["rc720"][:, v1], t["rel_rc"][v1] > THR if variant == "rc" else rel_orig)
    return _TD[k]


def things_job(job):
    import pilot_fit
    import step47_nsd_ridge as B
    subject, variant, g, name = job
    Y, rel = _things_targets(subject, variant)
    X = pilot_fit.load_features(name, g).astype(np.float64)
    out = B.fit_one(X, Y, rel, f"{subject} {variant} {name} g{g}")
    lab, si = label_of(name)
    return dict(subject=subject, variant=variant, grid=g, condition=lab, seed=si, n_reliable=int(rel.sum()), **out)


def things_jobs():
    js = [(s, "rc", g, n) for s in SUBJ for g in GRIDS for n in names()]
    return js + [(s, "rc_origsel", 4, n) for s in SUBJ for n in names()]


def things_aggregate(df):
    """prereg_aggregate rule: a (condition, seed, g) cell enters only if all three subjects are
    valid; mean over subjects of the median r; Random - BP per seed."""
    rows = []
    for (var, g), q in df.groupby(["variant", "grid"]):
        ok = q.groupby(["condition", "seed"]).valid.all()
        m = q.groupby(["condition", "seed"]).r_med_reliable.mean().where(ok).unstack(0)
        d = m["Random"] - m["BP"]
        rows.append(dict(kind="things_ridge", variant=var, grid=g, delta=float(d.mean()), n_seeds_valid=int(d.notna().sum()),
                         seeds_negative=int((d < 0).sum()), r_random=float(m["Random"].mean()), r_bp=float(m["BP"].mean()),
                         n_invalid_cells=int((~q.valid).sum()), n_low_edge_hits=int(q.n_low_edge_hits.sum()),
                         **{f"delta_seed{si}": float(d.loc[si]) for si in d.index}))
    return rows


def prereg_reference():
    p = pd.read_csv(REPO / "results" / "prereg" / "primary.csv")
    p = p[p.rule.isin(["Random Weights", "Backprop"]) & p.grid.isin(GRIDS)].copy()
    p["condition"] = p.rule.map({"Random Weights": "Random", "Backprop": "BP"})
    p["seed"], p["variant"] = p.seed_idx, "raw (results/prereg)"
    return p[["subject", "variant", "grid", "condition", "seed", "r_med_reliable", "valid", "n_low_edge_hits", "n_reliable"]]


def things_test_rsa():
    """THINGS test images: Delta per (variant, roi, k, units) on b1 (correlation), mean over draws,
    subjects and seeds; uncentred must reproduce results/step41/noise_curve.csv."""
    import step40_common as S40
    import step41_noise_curve as S41
    from cvrsa import model_rdm
    draws = S41.session_draws()
    models = {}
    for key, lab in KEYS.items():
        for si in range(5):
            X = np.load(S40.feature_dir() / "test100" / f"model_{key}_seed{si}_g1.npy")
            for u in ("raw", "z"):
                models[(lab, si, u)] = model_rdm(X, u, "correlation")
    mk = list(models)
    Zm = S41.rank_z(np.stack([models[k] for k in mk]))
    rows = []
    for s in SUBJ:
        reps_raw, rois = S40.load_reps(s)
        reps = {"raw": reps_raw, "run-centred": things_data(s)["reps_rc"]}
        for var, R_ in reps.items():
            for roi in ("V1", "LOC"):
                R = R_[:, :, rois[roi]]
                for k in (1, 12):
                    for di, sess in enumerate(draws[k]):
                        rho = Zm @ S41.rank_z(S41.brain_for(R, sess)["b1"])[0]
                        rows += [dict(subject=s, variant=var, roi=roi, k=k, draw=di, condition=lab, seed=si, units=u, rho=float(v))
                                 for (lab, si, u), v in zip(mk, rho)]
    rho = pd.DataFrame(rows)
    g = rho.groupby(["variant", "roi", "k", "units", "condition"]).rho.mean().unstack("condition")
    per = rho.groupby(["variant", "roi", "k", "units", "subject", "condition"]).rho.mean().unstack("condition")
    out = []
    ref = pd.read_csv(REPO / "results" / "step41" / "noise_curve.csv")
    ref = ref[(ref.kind == "contrast") & (ref.model_distance == "correlation") & (ref.brain_rdm == "b1")]
    dev = 0.0
    for (var, roi, k, u), r in g.iterrows():
        d = float(r["Random"] - r["BP"])
        p = per.loc[(var, roi, k, u)]
        dd = p["Random"] - p["BP"]
        out.append(dict(kind="things_test", variant=var, roi=roi, k=k, units=u, delta=d, rho_random=float(r["Random"]),
                        rho_bp=float(r["BP"]), n_subjects_negative=int((dd < 0).sum()),
                        **{f"delta_{s}": float(dd.loc[s]) for s in SUBJ}))
        if var == "raw":
            x = ref[(ref.roi == roi) & (ref.k == k) & (ref.model_units == u)]
            dev = max(dev, abs(float(x.delta.iloc[0]) - d))
    if dev > 1e-10:
        raise SystemExit(f"STOP: uncentred test-image Delta does not reproduce step 41 (max |d| {dev:.1e})")
    return out, dev


def things_rsa_720():
    """Run-centred THINGS-720 V1 Delta (paper cell) from rc720 (test iv: +0.0070 of step 47)."""
    import step44_why as W
    from scipy.spatial.distance import pdist
    br = []
    for s in SUBJ:
        a = np.load(paths.ridge_data_dir() / "arm_b_out" / f"{s}_arma.npz")
        br.append(pdist(things_data(s)["rc720"][:, a["local_V1"]], "correlation"))
    d, per = W.delta_on(W.things720_models("V1"), br, W.crossrun_index())
    return d, per


def write_rows(rows, part):
    sp = OUT / "robustness.csv"
    new = pd.DataFrame(rows)
    new["part"] = part
    if sp.exists():
        prev = pd.read_csv(sp, low_memory=False).drop(columns=["config_hash", "git_head", "seed"], errors="ignore")
        new = pd.concat([prev[prev.part != part], new], ignore_index=True)
    stamp(new).to_csv(sp, index=False)


def run_things(workers=8):
    import step47b_mechanism as M
    C.require_free(cfg()["disk"]["min_free_gb"])
    OUT.mkdir(parents=True, exist_ok=True)
    t0, times = time.time(), {}
    for s in SUBJ:
        things_data(s)
    times["data"] = time.time() - t0
    d720, _ = things_rsa_720()
    ref = pd.read_csv(REPO / "results" / "step47" / "run_centring.csv")
    ref = float(ref[(ref.kind == "delta") & (ref.roi == "V1") & (ref.variant == "run-centred")].delta.iloc[0])
    if abs(d720 - ref) > TOL:
        raise SystemExit(f"STOP: run-centred THINGS-720 Delta {d720} differs from step 47 {ref}")
    t = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(things_job, things_jobs()), 1):
            rows.append(r)
            if k % 30 == 0:
                print(f"  THINGS ridge {k}/{len(things_jobs())} fits ({time.time() - t:.0f} s)", flush=True)
    fits = pd.DataFrame(rows)
    times["ridge"] = time.time() - t
    agg = things_aggregate(pd.concat([fits, prereg_reference()], ignore_index=True))
    t = time.time()
    test_rows, dev41 = things_test_rsa()
    times["test images"] = time.time() - t
    nrel = [dict(kind="things_reliable_voxels", subject=s, variant=v, n_reliable=int(_things_targets(s, v)[1].sum()))
            for s in SUBJ for v in ("raw", "rc")]
    out = [dict(kind="things_ridge_fit", **r) for r in rows] + agg + test_rows + nrel
    out.append(dict(kind="check", quantity="run-centred THINGS-720 V1 Delta vs step 47", delta=d720, value=abs(d720 - ref)))
    out.append(dict(kind="check", quantity="uncentred test-image Delta vs step 41 max |d|", value=dev41))
    write_rows(out, "Y-THINGS")
    a = {(r["variant"], r["grid"]): r for r in agg}
    y1 = a[("rc", 4)]
    summ = [dict(rule="Y1", quantity="THINGS ridge, run-centred, g = 4: Random - BP < 0 in 5 of 5 seeds",
                 value=y1["seeds_negative"], passed=bool(y1["n_seeds_valid"] == 5 and y1["seeds_negative"] == 5),
                 note=f"mean {y1['delta']:+.4f}, {y1['seeds_negative']}/{y1['n_seeds_valid']} seeds negative; "
                      f"uncentred {a[('raw (results/prereg)', 4)]['delta']:+.4f}"),
            *[dict(rule="runtime", quantity=f"Y THINGS {k} seconds", value=v) for k, v in times.items()],
            dict(rule="runtime", quantity="Y THINGS total seconds", value=time.time() - t0)]
    M.write_summary(summ, "Y-THINGS")
    print(pd.DataFrame(agg)[["variant", "grid", "delta", "seeds_negative", "n_seeds_valid", "n_invalid_cells"]].to_string(index=False))
    print(pd.DataFrame(test_rows)[["variant", "roi", "k", "units", "delta", "n_subjects_negative"]].to_string(index=False))
    print(pd.DataFrame(summ)[["rule", "value", "passed", "note"]].to_string(index=False))


# ── NSD ─────────────────────────────────────────────────────────────────────
NSD_ROIS = ["V1", "LO-obj"]


def nsd_work():
    return C.nsd_dir("step47b")


def nsd_prepare(subject):
    """Run-centred targets and reliability per ROI (NSD_DIR/step47b/<subject>.npz); RSA rho rows
    (run-centred and uncentred, k = 3); multiverse brain RDMs of the run-centred V1 average."""
    import nsd_runmeans as RM
    import nsd_step43 as NS
    import step47_nsd_ridge as B
    from nsd_rsa import load_features, load_patterns, model_vectors, rsa_subject
    Pc, session, run, rois, valid = RM.centred_patterns(subject)
    P, _, _, _, _ = load_patterns(subject)
    images = np.load(C.nsd_dir("patterns") / f"{subject}.npz")["images"]
    if not np.array_equal(images, C.design(C.cfg()["pilot_subject"])["images"]):
        raise SystemExit(f"STOP: image order of {subject} differs from the feature order")
    models = model_vectors(load_features())
    rho = []
    for var, X in (("raw", P), ("run-centred", Pc)):
        r = rsa_subject(X, session, run, rois, models, k1=False)
        r["variant"], r["subject"] = var, subject
        rho.append(r)
    save = {}
    for roi in NSD_ROIS:
        m = rois[roi]
        sb3, sb2, _ = B.reliability(Pc[:, :, m])
        save[f"Y_{roi}"] = Pc[:, :, m].mean(axis=1)
        save[f"rel_{roi}"] = sb3 > B.cfg()["B"]["reliability"]["threshold"]
        save[f"nsb2_{roi}"] = int((sb2 > B.cfg()["B"]["reliability"]["threshold"]).sum())
    np.savez(nsd_work() / f"{subject}.npz", **save)
    bkeys, Bs = NS.mv_brains(Pc[:, :, rois["V1"]].mean(axis=1))
    return pd.concat(rho, ignore_index=True), bkeys, Bs


_ND = {}


def _nsd_targets(subject, roi, variant):
    """(Y, reliable) for variant raw (step 47), rc (run-centred, recomputed selection),
    rc_s47sel (run-centred, step 47 selection), raw_all (uncentred, all valid voxels)."""
    import step47_nsd_ridge as B
    k = (subject, roi, variant)
    if k not in _ND:
        if variant in ("raw", "raw_all"):
            Y, rel, _ = B.targets(subject, roi)
            _ND[k] = (Y, rel if variant == "raw" else np.ones(Y.shape[1], bool))
        else:
            z = np.load(nsd_work() / f"{subject}.npz")
            rel = z[f"rel_{roi}"] if variant == "rc" else B.targets(subject, roi)[1]
            _ND[k] = (z[f"Y_{roi}"], rel)
    return _ND[k]


def nsd_job(job):
    import step47_nsd_ridge as B
    variant, subject, roi, g, name = job
    Y, rel = _nsd_targets(subject, roi, variant)
    out = B.fit_one(B.features(name, g), Y, rel, f"{subject} {roi} {variant} {name} g{g}")
    cond = "bank" if name.startswith("control") else KEYS[name.split("_")[1]]
    return dict(variant=variant, subject=subject, roi=roi, grid=g, condition=cond,
                seed=int(name[-1]) if name.startswith("model") else -1, n_reliable=int(rel.sum()), n_voxels=Y.shape[1], **out)


def nsd_jobs():
    subs = C.cfg()["subjects"]
    nm = names() + ["control_lumcontrast"]
    js = [("rc", s, roi, g, n) for roi in NSD_ROIS for s in subs for g in GRIDS for n in nm]
    js += [("rc_s47sel", s, "V1", 4, n) for s in subs for n in names()]
    js += [("raw_all", s, "V1", 4, n) for s in subs for n in names()]
    return js


def nsd_ridge_aggregate(df):
    from nsd_rsa import sign_flip_p
    subs = C.cfg()["subjects"]
    rows = []
    for (var, roi, g), q in df.groupby(["variant", "roi", "grid"]):
        per = []
        for s in subs:
            x = q[(q.subject == s) & q.condition.isin(["Random", "BP"])]
            v = x[x.valid]
            r_r, r_b = float(v[v.condition == "Random"].r_med_reliable.mean()), float(v[v.condition == "BP"].r_med_reliable.mean())
            bank = q[(q.subject == s) & (q.condition == "bank")]
            per.append(r_r - r_b)
            rows.append(dict(kind="nsd_ridge_subject", variant=var, roi=roi, grid=g, subject=s, r_random=r_r, r_bp=r_b,
                             delta=r_r - r_b, r_bank=float(bank.r_med_reliable.iloc[0]) if len(bank) else np.nan,
                             n_reliable=int(x.n_reliable.iloc[0]), n_invalid_model_cells=int((~x.valid).sum()),
                             n_low_edge_hits=int(q[q.subject == s].n_low_edge_hits.sum())))
        per = np.array(per)
        rows.append(dict(kind="nsd_ridge_group", variant=var, roi=roi, grid=g, delta=float(per.mean()),
                         n_negative=int((per < 0).sum()), p_negative=sign_flip_p(per, "negative"), n_subjects=len(per)))
    return rows


def step47_reference():
    r = pd.read_csv(REPO / "results" / "step47" / "nsd_ridge.csv", low_memory=False)
    r = r[r.kind == "fit"].copy()
    r["variant"] = "raw (step 47)"
    r["seed"] = r.seed.fillna(-1).astype(int)
    return r[["variant", "subject", "roi", "grid", "condition", "seed", "r_med_reliable", "valid", "n_reliable", "n_voxels",
              "n_low_edge_hits"]]


_MV = {}


def _mv_init(mpath, bpath, pairs):
    _MV.update(M=np.load(mpath, mmap_mode="r"), B=np.load(bpath, mmap_mode="r"), pairs=pairs)


def _mv_chunk(chunk):
    import nsd_step43 as NS
    out = np.zeros((len(chunk), len(_MV["pairs"])))
    for j, (_, idx) in enumerate(chunk):
        out[j] = NS.rho_pairs(_MV["M"], _MV["B"], _MV["pairs"], NS.draw_index(idx))
    return [b for b, _ in chunk], out


def mv_draws(M, Bs, pairs, workers, n_resamples=None):
    """rho draws (1 + n, n_sub, n_pairs) as nsd_step43.bootstrap, with one rho_pairs call per draw
    for all subjects (brain rows of all subjects stacked; identical values)."""
    import nsd_step43 as NS
    from nsd_rsa import boot_indices
    n_sub, nb = Bs.shape[0], Bs.shape[1]
    Ball = Bs.reshape(n_sub * nb, -1)
    pall = np.concatenate([np.c_[pairs[:, 0], pairs[:, 1] + s * nb] for s in range(n_sub)])
    tmp = C.nsd_dir("tmp47b")
    mpath, bpath = tmp / "mv_M.npy", tmp / "mv_B.npy"
    np.save(mpath, np.asarray(M, np.float64))
    np.save(bpath, np.asarray(Ball, np.float64))
    bc = NS.cfg()["inference"]["bootstrap"]
    n = bc["n_resamples"] if n_resamples is None else n_resamples
    items = [(0, None)] + [(b + 1, idx) for b, idx in zip(range(n), boot_indices(NS.N_IMG, bc["n_resamples"], bc["seed"]))]
    n_chunks = max(1, min(len(items), workers * 6))
    chunks = [items[i::n_chunks] for i in range(n_chunks)]
    acc = np.zeros((len(items), len(pall)))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_mv_init, initargs=(mpath, bpath, pall)) as ex:
        for done, (bs, out) in enumerate(ex.map(_mv_chunk, chunks), 1):
            acc[bs] = out
            if done % workers == 0:
                print(f"  multiverse chunk {done}/{len(chunks)} ({time.time() - t0:.0f} s)", flush=True)
    os.remove(mpath)
    os.remove(bpath)
    return acc.reshape(len(items), n_sub, len(pairs))


def nsd_multiverse(Bs, bkeys, workers, rho_rc):
    import nsd_step43 as NS
    import step7_multiverse as S7
    from nsd_rsa import load_features
    subs = C.cfg()["subjects"]
    mkeys, M = NS.mv_models(load_features())
    pairs, lab = NS.mv_pairs(mkeys, bkeys)
    draws = mv_draws(M, Bs, pairs, workers)
    dev = 0.0
    for s_i, s in enumerate(subs):
        r = rho_rc[(rho_rc.subject == s) & (rho_rc.roi == "V1") & (rho_rc.k == 3) & (rho_rc.distance == "correlation")]
        for row in r.itertuples():
            j = lab.index[(lab.F == row.model_units) & (lab.D == "correlation") & (lab.B == "paper")
                          & (lab.condition == row.condition) & (lab.seed == row.seed)][0]
            dev = max(dev, abs(draws[0, s_i, j] - row.rho))
    if dev > 1e-9:
        raise SystemExit(f"STOP: run-centred multiverse paper cell differs from the run-centred RSA ({dev:.1e})")
    rows, vec, cls_of = [], {}, {}
    for f in S7.FS:
        for d in S7.DS:
            for b in S7.BS:
                sel = (lab.F == f) & (lab.D == d) & (lab.B == b)
                w = np.where(sel & (lab.condition == "Random"), 1 / 5, 0.0) - np.where(sel & (lab.condition == "BP"), 1 / 5, 0.0)
                st, _ = NS.summarise(draws, w, "negative")
                order = lab[sel].sort_values(["condition", "seed"]).index
                vec[(f, d, b)] = draws[0][:, order].ravel()
                cls_of[(f, d, b)] = NS.ci_class(st["ci_lo"], st["ci_hi"])
                rows.append(dict(kind="nsd_multiverse_cell", F=f, D=d, B=b, roi="V1", variant="run-centred",
                                 delta=st["mean_delta"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"], n_negative=st["n_negative"],
                                 p_negative=st["p_negative"], ci_class=cls_of[(f, d, b)]))
    cls, within, between = NS.distinct_classes(vec, cls_of)
    df = pd.DataFrame(rows)
    df["cls"] = [cls[(r.F, r.D, r.B)] for r in df.itertuples()]
    first = df.drop_duplicates("cls")
    counts = {k: int((first.ci_class == k).sum()) for k in ("above 0", "below 0", "includes 0")}
    summ = [dict(kind="nsd_multiverse_summary", quantity="distinct analyses", value=len(first)),
            *[dict(kind="nsd_multiverse_summary", quantity=f"distinct analyses, interval {k}", value=v) for k, v in counts.items()],
            dict(kind="check", quantity="multiverse paper cell vs run-centred RSA max |d|", value=dev),
            dict(kind="check", quantity="multiverse max |d| within classes", value=within),
            dict(kind="check", quantity="multiverse min max|d| between classes", value=between)]
    return rows + summ, counts, len(first)


def nsd_rsa_rows(rho):
    from nsd_rsa import contrasts, sign_flip_p
    subs = C.cfg()["subjects"]
    out, per = [], {}
    for (var, s), q in rho.groupby(["variant", "subject"]):
        con = contrasts(q)
        for r in con.itertuples():
            out.append(dict(kind="nsd_rsa_subject", variant=var, subject=s, roi=r.roi, k=int(r.k), units=r.model_units,
                            distance=r.distance, rho_random=r.rho_random, rho_bp=r.rho_bp, delta=r.delta))
    sd = pd.DataFrame(out)
    for (var, roi, u, d), q in sd.groupby(["variant", "roi", "units", "distance"]):
        x = q.set_index("subject").loc[subs].delta.to_numpy()
        per[(var, roi, u, d)] = x
        out.append(dict(kind="nsd_rsa_group", variant=var, roi=roi, k=3, units=u, distance=d, delta=float(x.mean()),
                        n_negative=int((x < 0).sum()), p_negative=sign_flip_p(x, "negative"), n_subjects=len(x)))
    return out, per


def run_nsd(workers=8):
    import step47b_mechanism as M
    C.require_free(cfg()["disk"]["min_free_gb"])
    OUT.mkdir(parents=True, exist_ok=True)
    t0, times = time.time(), {}
    subs = C.cfg()["subjects"]
    rho, Bs, bkeys = [], [], None
    for s in subs:
        r, bkeys, b = nsd_prepare(s)
        rho.append(r)
        Bs.append(b)
        print(f"  prepared {s} ({time.time() - t0:.0f} s)", flush=True)
    rho = pd.concat(rho, ignore_index=True)
    Bs = np.stack(Bs)
    # uncentred RSA reproduces step 42
    ref = pd.read_csv(REPO / "results" / "step42" / "subjects.csv", low_memory=False)
    ref = ref[(ref.kind == "rho") & (ref.k == 3)].set_index(["subject", "roi", "condition", "seed", "model_units", "distance"]).rho
    raw = rho[rho.variant == "raw"]
    dev42 = max(abs(r.rho - float(ref.loc[(r.subject, r.roi, r.condition, r.seed, r.model_units, r.distance)])) for r in raw.itertuples())
    if dev42 > TOL:
        raise SystemExit(f"STOP: uncentred NSD RSA differs from step 42 (max |d| {dev42:.1e})")
    rsa_rows, per = nsd_rsa_rows(rho)
    times["prepare and RSA"] = time.time() - t0
    t = time.time()
    js = nsd_jobs()
    fits = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(nsd_job, js), 1):
            fits.append(r)
            if k % 24 == 0:
                print(f"  NSD ridge {k}/{len(js)} fits ({time.time() - t:.0f} s)", flush=True)
    fits = pd.DataFrame(fits)
    fits.loc[fits.seed < 0, "seed"] = np.nan
    times["ridge"] = time.time() - t
    ref47 = step47_reference()
    ref47.loc[ref47.seed < 0, "seed"] = np.nan
    ridge_rows = nsd_ridge_aggregate(pd.concat([fits, ref47], ignore_index=True))
    t = time.time()
    mv_rows, counts, n_cls = nsd_multiverse(Bs, bkeys, workers, rho[rho.variant == "run-centred"])
    times["multiverse"] = time.time() - t
    out = [dict(kind="nsd_ridge_fit", **r) for r in fits.to_dict("records")] + ridge_rows + rsa_rows + mv_rows
    out.append(dict(kind="check", quantity="uncentred NSD RSA rho vs step 42 max |d|", value=dev42))
    write_rows(out, "Y-NSD")
    g = {(r["variant"], r["roi"], r["grid"]): r for r in ridge_rows if r["kind"] == "nsd_ridge_group"}
    pc, sc = per[("run-centred", "V1", "raw", "correlation")], per[("run-centred", "V1", "z", "correlation")]
    y3, y4 = g[("rc", "V1", 4)], g[("raw_all", "V1", 4)]
    summ = [dict(rule="Y2", quantity="NSD RSA, run-centred, V1 paper and standardised cell: Delta < 0 in >= 6 of 8 each",
                 value=min(int((pc < 0).sum()), int((sc < 0).sum())), passed=bool((pc < 0).sum() >= 6 and (sc < 0).sum() >= 6),
                 note=f"paper {int((pc < 0).sum())}/8, mean {pc.mean():+.4f}; standardised {int((sc < 0).sum())}/8, mean {sc.mean():+.4f}"),
            dict(rule="Y3", quantity="NSD ridge, run-centred, g = 4, V1: Random - BP < 0 in >= 6 of 8",
                 value=y3["n_negative"], passed=bool(y3["n_negative"] >= 6),
                 note=f"{y3['n_negative']}/8 negative, mean {y3['delta']:+.4f}, p {y3['p_negative']:.4f}"),
            dict(rule="Y4", quantity="NSD ridge, uncentred, g = 4, all V1 voxels: Random - BP < 0 in >= 6 of 8",
                 value=y4["n_negative"], passed=bool(y4["n_negative"] >= 6),
                 note=f"{y4['n_negative']}/8 negative, mean {y4['delta']:+.4f}, p {y4['p_negative']:.4f}"),
            dict(rule="multiverse", quantity="NSD multiverse, run-centred: distinct analyses above / below / include 0",
                 value=n_cls, note=f"{counts['above 0']} above, {counts['below 0']} below, {counts['includes 0']} include 0 of {n_cls}"),
            *[dict(rule="runtime", quantity=f"Y NSD {k} seconds", value=v) for k, v in times.items()],
            dict(rule="runtime", quantity="Y NSD total seconds", value=time.time() - t0)]
    M.write_summary(summ, "Y-NSD")
    print(pd.DataFrame([r for r in ridge_rows if r["kind"] == "nsd_ridge_group"]).to_string(index=False))
    print(pd.DataFrame(summ)[["rule", "value", "passed", "note"]].to_string(index=False))


# ── estimates ───────────────────────────────────────────────────────────────
def estimate(which, workers=8):
    if which == "things":
        t = time.time()
        things_job(("sub-01", "raw", 4, "model_random_seed0"))
        t_fit = time.time() - t
        n = len(things_jobs())
        h = (n * t_fit / workers + 3 * 60 + 120) * 1.5 / 3600
        print(f"one THINGS fit {t_fit:.1f} s; {n} fits / {workers} workers + data + test images, x 1.5: {h:.2f} h")
        return h
    import nsd_step43 as NS
    from nsd_rsa import boot_indices, load_features
    import step47_nsd_ridge as B
    t = time.time()
    B.fit_one(B.features("model_random_seed0", 4), *B.targets("subj01", "V1")[:2], "estimate")
    t_fit = time.time() - t
    mkeys, M = NS.mv_models(load_features())
    bkeys, Bs = NS.mv_brains(NS.brain_X("subj01")["V1"])
    Bs = np.stack([Bs] * 8)
    pairs, _ = NS.mv_pairs(mkeys, bkeys)
    idx = next(iter(boot_indices(NS.N_IMG, 1, NS.cfg()["inference"]["bootstrap"]["seed"])))
    nb = Bs.shape[1]
    t = time.time()
    NS.rho_pairs(M, Bs.reshape(-1, Bs.shape[2]), np.concatenate([np.c_[pairs[:, 0], pairs[:, 1] + s * nb] for s in range(8)]),
                 NS.draw_index(idx))
    t_draw = time.time() - t
    n = len(nsd_jobs())
    h = (n * t_fit / workers * 1.5 + 1001 * t_draw / workers * 2 + 8 * 60) * 1.5 / 3600
    print(f"one NSD fit {t_fit:.1f} s, one multiverse draw {t_draw:.1f} s; {n} fits / {workers} workers (x 1.5 parallel "
          f"slow-down), 1001 draws / {workers} (x 2), preparation; total x 1.5: {h:.2f} h")
    return h


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "estimate":
        h = estimate(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
        if h > cfg()["runtime"]["stop_hours_per_part"]:
            raise SystemExit(f"STOP: estimate {h:.2f} h above {cfg()['runtime']['stop_hours_per_part']} h")
    elif which in ("things", "nsd"):
        w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        (run_things if which == "things" else run_nsd)(w)
    else:
        raise SystemExit("usage: step47b_robustness.py {estimate things|nsd | things [workers] | nsd [workers]}")

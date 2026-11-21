"""
step47_runs.py
==============
Step 47, part A (amendments/2026-11-21_step47-predictions.md, config_step47.yaml): run structure in
THINGS-720.

  RC1  run-centred V1 Delta <= +0.010
  RC2  run-centred rho(V1, L-RDM) / LOSO lower bound (run-centred) < 0.5
  RC3  across-run variance of the run-mean L: sub-01/02 assignment > sub-03 assignment
  RC4  run-centred V1 multiverse: at most 2 distinct analyses with the interval above 0
  CO1  uncentred: partialling |L_prev(i) - L_prev(j)| changes V1 Delta by >= 0.005

Run centring: the ROI responses of all 9840 trials are read and z-scored per voxel exactly as
arm_b_betas.load_arm_a does (same expressions; gate: the 720 rows equal the stored arm_a betas),
then each of the 720 betas minus the voxel's mean over all trials of the same run (82 trials).

Imported unchanged: step44_why (partial_rho, delta_on, crossrun_index, lum_contrast, abs_vec,
things720_images, things720_models, spear, colour_module), step7_multiverse (FS, DS, BS, SUBJ,
normalise, feat, betas), nsd_step43 (mv_brain, distinct_classes, loso_lower), common_up_pairs
(ranks, run_labels, pair_set, stim_order), v12_lift (RULES, boot_idx, boot_pairs, N_BOOT, namespaces).

Voxel-level intermediates (run-centred betas) go to RIDGE_DATA_DIR/step47 only.

Usage:
  python scripts/step47_runs.py estimate
  python scripts/step47_runs.py run [workers]
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
from scipy.spatial.distance import pdist

import paths
from config_hash import config_hash, load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config_step47.yaml"
OUT = REPO / "results" / "step47"
SUBJ = ["sub-01", "sub-02", "sub-03"]
LAYER = {"V1": "Conv1", "LOC": "Conv3"}
R_, BP = "Random Weights", "Backprop"
MV_CONDS = [("NATIVE", 32), ("NATIVE", 224), ("UPSAMPLED", 224)]
TOL = 1e-9


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


def work_dir():
    d = paths.ridge_data_dir() / "step47"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── responses and run centring ──────────────────────────────────────────────
def load_resp(subject):
    """(resp (9840, n_roi) z-scored per voxel over all trials, arm_a archive, metadata). The read and
    the z-score are the expressions of arm_b_betas.load_arm_a (ROI index from the stored archive,
    identical to load_arm_a's, see reports/STEP47_SETUP.md)."""
    import h5py
    a = np.load(paths.ridge_data_dir() / "arm_b_out" / f"{subject}_arma.npz")
    with h5py.File(paths.things_fmri_dir() / f"{subject}_task-things_voxel-wise-responses.h5", "r") as f:
        raw = f["ResponseData/block0_values"][a["roi_idx"], :].astype(np.float32)
    resp = raw.T
    resp = (resp - resp.mean(axis=0)) / (resp.std(axis=0) + 1e-8)
    st = pd.read_csv(paths.things_fmri_dir() / f"{subject}_task-things_stimulus-metadata.csv")
    return resp, a, st


def run_centre(Y, labels, rows):
    """Y (n_trials, v); labels (n_trials,) run label per trial (NaN-free for the rows used);
    rows: target trials. Returns Y[rows] minus the mean of all trials with the same label."""
    labels = np.asarray(labels)
    means = {r: Y[labels == r].mean(axis=0) for r in np.unique(labels[rows])}
    return np.stack([Y[i] - means[labels[i]] for i in rows])


def trial_rows(st):
    from common_up_pairs import stim_order
    rows = np.array([st.index[st.stimulus == s].to_numpy()[0] for s in stim_order()])
    return rows


def centred_betas(subject):
    """{'raw': arm_a betas (720, n_roi) float64, 'rc': run-centred}, local ROI indices; cached."""
    cache = work_dir() / f"{subject}_runcentred.npz"
    a = np.load(paths.ridge_data_dir() / "arm_b_out" / f"{subject}_arma.npz")
    local = {roi: a[f"local_{roi}"] for roi in ("V1", "LOC")}
    if cache.exists():
        z = np.load(cache)
        return {"raw": a["betas"].astype(np.float64), "rc": z["rc"]}, local
    resp, a, st = load_resp(subject)
    rows = trial_rows(st)
    ses = st.session.to_numpy()[rows]
    if len(set(ses)) != 1:
        raise SystemExit(f"STOP: {subject}: the 720 are not in one session")
    in_ses = st.session.to_numpy() == ses[0]
    lab = np.where(in_ses, st.run.to_numpy(), -1)
    if np.abs(resp[rows] - a["betas"]).max() > 1e-5:
        raise SystemExit(f"STOP: {subject}: z-scored h5 responses differ from the arm_a betas")
    n_per = pd.Series(lab[in_ses]).value_counts()
    if not (n_per == 82).all() or len(n_per) != 10:
        raise SystemExit(f"STOP: {subject}: runs of the session do not have 82 trials each")
    rc = run_centre(resp.astype(np.float64), lab, rows)
    np.savez(cache, rc=rc)
    return {"raw": a["betas"].astype(np.float64), "rc": rc}, local


def brain_rdms(variant):
    """{roi: [per-subject correlation-distance RDM]} for variant 'raw' or 'rc'."""
    out = {"V1": [], "LOC": []}
    for s in SUBJ:
        B, local = centred_betas(s)
        for roi in out:
            out[roi].append(pdist(B[variant][:, local[roi]], "correlation"))
    return out


# ── partial Delta per subject ───────────────────────────────────────────────
def partial_delta(models, brains, ci, covs_of_subject):
    """Per-subject Delta (mean over seeds of Random - BP) with step44_why.partial_rho;
    covs_of_subject(s_i) -> list of covariate vectors on ci."""
    import step44_why as W
    per = []
    for s_i, b in enumerate(brains):
        Z = covs_of_subject(s_i)
        rho = {k: W.partial_rho(m[ci], b[ci], Z) for k, m in models.items()}
        per.append(np.mean([rho[("Random", s)] - rho[("BP", s)] for s in W.seeds()]))
    return np.array(per)


# ── RC3 ─────────────────────────────────────────────────────────────────────
def run_var(L, labels):
    labels = np.asarray(labels)
    return float(np.var([L[labels == r].mean() for r in np.unique(labels)]))


def perm_null(L, n, seed, n_runs=10, per_run=72):
    rng = np.random.default_rng(seed)
    base = np.repeat(np.arange(n_runs), per_run)
    out = np.empty(n)
    for k in range(n):
        lab = np.empty(len(L), int)
        lab[rng.permutation(len(L))] = base
        out[k] = run_var(L, lab)
    return out


def percentile(null, x):
    return float(100.0 * (null < x).mean())


# ── RC4: multiverse ─────────────────────────────────────────────────────────
def mv_brain_vectors(variant):
    """{(B, D, subject): RDM over TRI} for the V1 betas of the variant (step7_multiverse.betas with
    roi_betas pointed at them via nsd_step43.mv_brain)."""
    import nsd_step43 as NS
    import step7_multiverse as S7
    out = {}
    for s in SUBJ:
        B, local = centred_betas(s)
        X = B[variant][:, local["V1"]]
        for b in S7.BS:
            Xb = NS.mv_brain(X, b)
            for d in S7.DS:
                v = pdist(Xb, d)
                if not np.isfinite(v).all():
                    raise SystemExit(f"STOP: non-finite brain RDM {variant} {b} {d} {s}")
                out[(b, d, s)] = v
    return out


def mv_points(brains, keep, rules=None, seeds=range(5), conds=MV_CONDS):
    """Point rho per (F, D, B, arm, px, rule, seed, subject) as step7_multiverse._pts_one."""
    import step7_multiverse as S7
    from common_up_pairs import ranks
    from v12_lift import RULES
    brank = {k: ranks(v[keep]) for k, v in brains.items()}
    rows = []
    for f in S7.FS:
        for d in S7.DS:
            for arm, px in conds:
                for rule in (rules or RULES):
                    for si in seeds:
                        m = pdist(S7.normalise(S7.feat(arm, px, rule, si, "Conv1"), f), d)
                        if not np.isfinite(m).all():
                            raise SystemExit(f"STOP: non-finite model RDM {f} {d} {arm} {px} {rule} {si}")
                        mv = ranks(m[keep])
                        for b in S7.BS:
                            for s in SUBJ:
                                rows.append(dict(F=f, D=d, B=b, arm=arm, res=px, rule=rule, seed_idx=si, subject=s,
                                                 rho=float(mv @ brank[(b, d, s)])))
    return pd.DataFrame(rows)


_W = {}


def _init(mpath, bpath):
    from common_up_pairs import run_labels
    from v12_lift import boot_idx
    _W.update(M=np.load(mpath, mmap_mode="r"), B=np.load(bpath, mmap_mode="r"), boot=boot_idx(), runs=run_labels())


def _one(b):
    from common_up_pairs import ranks
    from v12_lift import boot_pairs
    idx = boot_pairs(b, _W["boot"], _W["runs"])
    Rm = np.stack([ranks(np.asarray(_W["M"][k][idx], float)) for k in range(_W["M"].shape[0])])
    Rb = np.stack([ranks(np.asarray(_W["B"][k][idx], float)) for k in range(_W["B"].shape[0])])
    return b, Rm @ Rb.T


def mv_bootstrap(brains, cells, workers, tag):
    """Gap draws (1001,) per cell (F, D, B): V1, NATIVE 224, Random - BP per seed, mean over
    subjects then seeds; draw 0 = identity (boot_pairs b = -1); the paper's 1000 resamples."""
    import step7_multiverse as S7
    from v12_lift import N_BOOT
    fd = sorted({(f, d) for f, d, _ in cells}, key=lambda x: (S7.FS.index(x[0]), S7.DS.index(x[1])))
    bd = sorted({(b, d) for _, d, b in cells}, key=lambda x: (S7.BS.index(x[0]), S7.DS.index(x[1])))
    mitems = [(f, d, rule, si) for f, d in fd for rule in (R_, BP) for si in range(5)]
    bitems = [(b, d, s) for b, d in bd for s in SUBJ]
    mpath, bpath = work_dir() / f"mv_{tag}_M.npy", work_dir() / f"mv_{tag}_B.npy"
    np.save(mpath, np.stack([pdist(S7.normalise(S7.feat("NATIVE", 224, rule, si, "Conv1"), f), d)
                             for f, d, rule, si in mitems]))
    np.save(bpath, np.stack([brains[k] for k in bitems]))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(mpath, bpath)) as ex:
        res = dict(ex.map(_one, range(-1, N_BOOT), chunksize=8))
    os.remove(mpath)
    os.remove(bpath)
    G = np.stack([res[b] for b in range(-1, N_BOOT)])            # (1001, models, brains)
    mi = {k: i for i, k in enumerate(mitems)}
    bi = {k: i for i, k in enumerate(bitems)}
    out = {}
    for f, d, b in cells:
        cols = [bi[(b, d, s)] for s in SUBJ]
        out[(f, d, b)] = np.mean([G[:, mi[(f, d, R_, si)], cols].mean(axis=1) - G[:, mi[(f, d, BP, si)], cols].mean(axis=1)
                                  for si in range(5)], axis=0)
    return out


def ci_of(x):
    return float(x[0]), float(np.quantile(x[1:], .025)), float(np.quantile(x[1:], .975))


def ci_class(lo, hi):
    return "above 0" if lo > 0 else ("below 0" if hi < 0 else "includes 0")


def run_multiverse(variant, workers):
    import nsd_step43 as NS
    import step7_multiverse as S7
    from common_up_pairs import pair_set, run_labels
    _, _, keep = pair_set(None, run_labels())
    brains = mv_brain_vectors(variant)
    pts = mv_points(brains, keep)
    cells = [(f, d, b) for f in S7.FS for d in S7.DS for b in S7.BS]
    draws = mv_bootstrap(brains, cells, workers, variant)
    rows, vec, cls_of = [], {}, {}
    for f, d, b in cells:
        p, lo, hi = ci_of(draws[(f, d, b)])
        q = pts[(pts.F == f) & (pts.D == d) & (pts.B == b) & (pts.arm == "NATIVE") & (pts.res == 224)]
        pv = q.groupby(["rule", "seed_idx"]).rho.mean().unstack(0)
        pt = float((pv[R_] - pv[BP]).mean())
        if abs(pt - p) > 1e-12:
            raise SystemExit(f"STOP: identity draw {p} differs from the point estimate {pt} ({f} {d} {b})")
        cls_of[(f, d, b)] = ci_class(lo, hi)
        v = pts[(pts.F == f) & (pts.D == d) & (pts.B == b)].sort_values(["arm", "res", "rule", "seed_idx", "subject"]).rho
        vec[(f, d, b)] = v.to_numpy()
        rows.append(dict(kind="cell", variant=variant, F=f, D=d, B=b, roi="V1", gap=p, ci_lo=lo, ci_hi=hi,
                         ci_class=cls_of[(f, d, b)], seeds_positive=int(((pv[R_] - pv[BP]) > 0).sum())))
    cls, within, between = NS.distinct_classes(vec, cls_of)
    df = pd.DataFrame(rows)
    df["cls"] = [cls[(r.F, r.D, r.B)] for r in df.itertuples()]
    first = df.drop_duplicates("cls")
    counts = {k: int((first.ci_class == k).sum()) for k in ("above 0", "below 0", "includes 0")}
    summ = [dict(kind="summary", variant=variant, quantity="distinct analyses", value=len(first)),
            *[dict(kind="summary", variant=variant, quantity=f"distinct analyses, interval {k}", value=v) for k, v in counts.items()],
            dict(kind="check", variant=variant, quantity="max |d| within classes", value=within),
            dict(kind="check", variant=variant, quantity="min max|d| between classes", value=between)]
    return pd.concat([df, pd.DataFrame(summ)], ignore_index=True), pts, len(first), counts


# ── CO1: predecessor luminance ──────────────────────────────────────────────
def predecessors():
    """{subject: (has (720,) bool, L_prev (720,) with NaN where none)}."""
    import step44_why as W
    from common_up_pairs import stim_order
    from v12_lift import namespaces
    ns12, _ = namespaces()
    order = stim_order()
    out, names = {}, {}
    for s in SUBJ:
        st = pd.read_csv(paths.things_fmri_dir() / f"{s}_task-things_stimulus-metadata.csv")
        ses = st.set_index("stimulus").loc[order].session.unique()
        if len(ses) != 1:
            raise SystemExit(f"STOP: {s}: the 720 are not in one session")
        S_ = st[st.session == ses[0]].copy()
        S_["prev"] = S_.groupby("run").stimulus.shift(1)
        names[s] = S_.set_index("stimulus").loc[order].prev
    uniq = sorted({p for v in names.values() for p in v.dropna()})
    imgs = [ns12["find_img"](p) for p in uniq]
    if any(p is None for p in imgs):
        raise SystemExit("STOP: predecessor image missing")
    L, _ = W.lum_contrast(imgs)
    lum = dict(zip(uniq, L))
    for s in SUBJ:
        v = names[s]
        out[s] = (v.notna().to_numpy(), np.array([lum[p] if isinstance(p, str) else np.nan for p in v]))
    return out


# ── driver ──────────────────────────────────────────────────────────────────
def run(workers=8):
    import nsd_step43 as NS
    import step44_why as W
    c = cfg()["A"]
    C.require_free(cfg()["disk"]["min_free_gb"])
    OUT.mkdir(parents=True, exist_ok=True)
    t0, times = time.time(), {}
    ci = W.crossrun_index()
    models = {roi: W.things720_models(roi) for roi in ("V1", "LOC")}
    br = {v: brain_rdms(v) for v in ("raw", "rc")}
    L, Cn = W.lum_contrast(W.things720_images())
    Lv, Cv = W.abs_vec(L), W.abs_vec(Cn)
    times["setup"] = time.time() - t0

    # run centring
    rows, res = [], {}
    for roi in ("V1", "LOC"):
        for name, var, covs in (("raw", "raw", []), ("run-centred", "rc", []), ("run-centred, L + C", "rc", [Lv[ci], Cv[ci]])):
            if covs:
                per = partial_delta(models[roi], br[var][roi], ci, lambda s_i: covs)
            else:
                _, per = W.delta_on(models[roi], br[var][roi], ci)
            res[(roi, name)] = float(per.mean())
            rows += [dict(kind="subject_delta", roi=roi, variant=name, subject=s, delta=float(v)) for s, v in zip(SUBJ, per)]
            rows.append(dict(kind="delta", roi=roi, variant=name, delta=float(per.mean())))
        for name, var in (("raw", "raw"), ("run-centred", "rc")):
            bl = [W.spear(b[ci], Lv[ci]) for b in br[var][roi]]
            bound = NS.loso_lower([b[ci] for b in br[var][roi]])
            res[(roi, name, "ratio")] = float(np.mean(bl)) / bound
            rows += [dict(kind="subject_rho_L", roi=roi, variant=name, subject=s, rho=float(v)) for s, v in zip(SUBJ, bl)]
            rows.append(dict(kind="lum_bound", roi=roi, variant=name, rho=float(np.mean(bl)), loso_lower=bound,
                             ratio=float(np.mean(bl)) / bound))
    if abs(res[("V1", "raw")] - 0.030445938) > 1e-8:
        raise SystemExit(f"STOP: uncentred V1 Delta {res[('V1', 'raw')]} does not reproduce the paper cell")
    stamp(pd.DataFrame(rows)).to_csv(OUT / "run_centring.csv", index=False)
    times["run centring"] = time.time() - t0 - times["setup"]

    # RC3
    t = time.time()
    from common_up_pairs import run_labels
    runs = run_labels()
    obs = {"sub-01/02": run_var(L, runs["sub-01"]), "sub-03": run_var(L, runs["sub-03"])}
    if not np.array_equal(runs["sub-01"], runs["sub-02"]):
        raise SystemExit("STOP: sub-01 and sub-02 run assignments differ")
    null = perm_null(L, c["RC3"]["permutation"]["n"], c["RC3"]["permutation"]["seed"])
    rl = [dict(kind="assignment", assignment=k, run_variance=v, percentile=percentile(null, v),
               run_means=";".join(f"{L[runs['sub-01' if k == 'sub-01/02' else 'sub-03'] == r].mean():.4f}" for r in range(1, 11)))
          for k, v in obs.items()]
    rl.append(dict(kind="null", n=len(null), mean=float(null.mean()), sd=float(null.std(ddof=1)),
                   q05=float(np.quantile(null, .05)), q50=float(np.median(null)), q95=float(np.quantile(null, .95))))
    stamp(pd.DataFrame(rl)).to_csv(OUT / "run_luminance.csv", index=False)
    times["RC3"] = time.time() - t

    # RC4
    t = time.time()
    mv, _, n_cls, counts = run_multiverse("rc", workers)
    stamp(mv).to_csv(OUT / "multiverse_runcentred.csv", index=False)
    times["RC4 multiverse"] = time.time() - t

    # CO1
    t = time.time()
    pred = predecessors()
    has_all = np.logical_and.reduce([pred[s][0] for s in SUBJ])
    ci2 = W.crossrun_index(np.flatnonzero(has_all))
    co = []
    for roi in ("V1", "LOC"):
        _, d0 = W.delta_on(models[roi], br["raw"][roi], ci2)
        d1 = partial_delta(models[roi], br["raw"][roi], ci2,
                           lambda s_i: [W.abs_vec(np.nan_to_num(pred[SUBJ[s_i]][1]))[ci2]])
        res[(roi, "CO1 change")] = float(d1.mean() - d0.mean())
        co += [dict(kind="subject", roi=roi, subject=s, delta_restricted=float(a), delta_partial=float(b), change=float(b - a))
               for s, a, b in zip(SUBJ, d0, d1)]
        co.append(dict(kind="mean", roi=roi, delta_restricted=float(d0.mean()), delta_partial=float(d1.mean()),
                       change=float(d1.mean() - d0.mean()), n_pairs=len(ci2), n_images=int(has_all.sum())))
    co += [dict(kind="predecessors", subject=s, n_with=int(pred[s][0].sum()),
                spearman_L_Lprev=float(W.spear(L[pred[s][0]], pred[s][1][pred[s][0]]))) for s in SUBJ]
    stamp(pd.DataFrame(co)).to_csv(OUT / "carryover.csv", index=False)
    times["CO1"] = time.time() - t

    # summary (part A)
    d_raw, d_rc = res[("V1", "raw")], res[("V1", "run-centred")]
    met = {"RC1": d_rc <= 0.010, "RC2": res[("V1", "run-centred", "ratio")] < 0.5,
           "RC3": obs["sub-01/02"] > obs["sub-03"], "RC4": counts["above 0"] <= 2,
           "CO1": abs(res[("V1", "CO1 change")]) >= 0.005}
    interp = []
    if met["RC1"] and met["RC2"] and met["RC3"]:
        interp.append("run-level structure: the THINGS-720 advantage is attributed to run-level structure shared by subjects "
                      "with the same run assignment, combined with the untrained network's greater luminance sensitivity")
    if not met["RC1"]:
        interp.append(f"run structure is not the explanation; explained fraction {1 - d_rc / d_raw:.3f}")
    if met["CO1"]:
        interp.append("carry-over from the preceding image contributes")
    interp = interp or ["no statement of the rule applies"]
    summ = [dict(rule="RC1", quantity="run-centred V1 Delta (<= +0.010)", value=d_rc, passed=met["RC1"],
                 note=f"uncentred {d_raw:+.4f}; explained fraction {1 - d_rc / d_raw:.3f}"),
            dict(rule="RC2", quantity="run-centred rho(V1, L) / LOSO bound (< 0.5)", value=res[("V1", "run-centred", "ratio")],
                 passed=met["RC2"], note=f"uncentred {res[('V1', 'raw', 'ratio')]:.3f}"),
            dict(rule="RC3", quantity="run variance of run-mean L, sub-01/02 > sub-03", value=obs["sub-01/02"], passed=met["RC3"],
                 note=f"sub-01/02 {obs['sub-01/02']:.3e} (pct {percentile(null, obs['sub-01/02']):.1f}); "
                      f"sub-03 {obs['sub-03']:.3e} (pct {percentile(null, obs['sub-03']):.1f})"),
            dict(rule="RC4", quantity="run-centred multiverse: distinct analyses with interval above 0 (<= 2)",
                 value=counts["above 0"], passed=met["RC4"],
                 note=f"{counts['above 0']} above, {counts['below 0']} below, {counts['includes 0']} include 0 of {n_cls}"),
            dict(rule="CO1", quantity="|change of V1 Delta by partialling predecessor L| (>= 0.005)", value=res[("V1", "CO1 change")],
                 passed=met["CO1"], note=f"{int(has_all.sum())} images with a predecessor in every subject, {len(ci2)} pairs")]
    summ += [dict(rule="interpretation", quantity="amendment rule (part A)", note=s) for s in interp]
    summ += [dict(rule="runtime", quantity=f"A {k} seconds", value=v) for k, v in times.items()]
    summ.append(dict(rule="runtime", quantity="A total seconds", value=time.time() - t0))
    sm = pd.DataFrame(summ)
    stamp(sm).to_csv(OUT / "summary.csv", index=False)
    print(sm[["rule", "quantity", "value", "passed", "note"]].to_string(index=False))


def estimate(workers=8):
    from common_up_pairs import ranks
    v = np.random.default_rng(0).normal(size=(5, 210205))
    t = time.time()
    for x in v:
        ranks(x)
    t_rank = (time.time() - t) / 5
    est = {"responses (3 subjects, h5 ROI rows)": 3 * 30, "multiverse points (1080 models x ranking)": 1080 * t_rank * 1.2,
           "multiverse bootstrap (147 rankings x 1001 / workers)": 147 * t_rank * 1001 / workers,
           "partials (L + C, predecessor; 2 ROIs x 3 subjects x 10 models x 2)": 120 * 0.15,
           "RC3 permutations (10,000)": 20, "luminance (720 + 711 predecessors)": 1431 * 0.1}
    est = {k: v * 1.5 for k, v in est.items()}
    for k, v in est.items():
        print(f"  {k:60s} {v / 60:6.1f} min")
    h = sum(est.values()) / 3600
    print(f"part A total x 1.5: {h:.2f} h (ranking {t_rank * 1e3:.0f} ms)")
    return h


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if which == "estimate":
        h = estimate(w)
        if h > cfg()["runtime"]["stop_hours_A"]:
            raise SystemExit(f"STOP: estimate {h:.2f} h above {cfg()['runtime']['stop_hours_A']} h")
    elif which == "run":
        run(w)
    else:
        raise SystemExit("usage: step47_runs.py {estimate|run} [workers]")

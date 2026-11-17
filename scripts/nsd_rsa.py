"""
nsd_rsa.py
==========
Step 42 (amendments/2026-11-17_step42-predictions.md): RSA cells on the 515 NSD images.

Per subject and ROI (V1, LO-obj, lateral; valid voxels), brain patterns from
NSD_DIR/patterns/<subject>.npz (P: 515 images x 3 presentations x voxels):
  k = 3  mean of the three presentations; all 132,355 pairs.
  k = 1  presentation r = 1, 2, 3 separately; for presentation r, the pairs (i, j) whose
         r-th presentations fall in the same session and run are excluded; rho averaged
         over r.
Model RDMs: GAP Conv1 features (NSD_DIR/features), Random and BP x 5 seeds, units raw or
standardised over the 515 images (cvrsa.model_rdm), correlation or Euclidean distance; the
brain side uses the same distance. Spearman over the pairs kept.
Per subject: mean over seeds; Delta = Random - BP.

Group (8 subjects): one-sided sign-flip test over the subject Deltas (256 assignments,
p = fraction of assignments whose mean is <= the observed mean), and a stimulus bootstrap
of the 8-subject mean (1000 resamples, seed 20261002, self-pairs dropped, the same resample
for all subjects and cells; at k = 1 the run exclusion is applied to the resampled pairs).

Step 42b (amendments/2026-11-17_step42b-predictions.md): norm-angle components at k = 3,
model units and voxels centred over the 515 images, N = |n_i - n_j|, A = 1 - cos, W =
2 n_i n_j (1 - cos), computed by the EXISTING function scripts/step25_weighted_angle.py::nwa
(which calls step9_separation.normf and .components), imported unchanged. nwa indexes the
pairs through its module-level TRI (720 stimuli); for n images TRI is set to the n-image upper
triangle for the duration of the call and restored afterwards (component_vectors).
Model component vs the same brain component, Spearman over all pairs. Rows: model_units
"center", distance "N", "A" or "W".

Usage:
  python scripts/nsd_rsa.py subject <subject>   # NSD_DIR/rsa/<subject>.csv (aggregated rho values)
                                                # pilot (step 42): results/step42/pilot_subj01.csv
  python scripts/nsd_rsa.py gates <subject>     # G1-G5 -> results/step42/gates.csv
                                                # (pilot: results/step42/pilot_gates.csv)
  python scripts/nsd_rsa.py group [workers]     # results/step42/{subjects,group,summary}.csv
Brain RDMs are cached under NSD_DIR/rdm (never in the repository). Output CSVs contain only
rho values, contrasts, intervals, p values and counts.
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads)

import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata

from cvrsa import condensed_index, model_rdm

ROIS = ["V1", "LO-obj", "lateral"]
UNITS = ["raw", "z"]
DISTS = ["correlation", "euclidean"]
COMPS = ["N", "A", "W"]
CONFIG_B = C.REPO / "config_step42b.yaml"


# ── core ────────────────────────────────────────────────────────────────────
def rank_z(V):
    """Rows ranked (average ties), centred and scaled, so that z_a @ z_b is Spearman's rho."""
    R = np.apply_along_axis(rankdata, 1, np.atleast_2d(V)).astype(np.float64)
    R -= R.mean(axis=1, keepdims=True)
    return R / np.sqrt((R * R).sum(axis=1, keepdims=True))


def same_run_pairs(session, run, r):
    """Condensed boolean: True where presentation r of image i and of image j share session and run."""
    key = np.asarray(session)[:, r].astype(np.int64) * 100 + np.asarray(run)[:, r]
    a, b = np.triu_indices(len(key), 1)
    return key[a] == key[b]


def brain_vectors(P, dist):
    """{'k3': condensed RDM of the presentation average, 'k1_r': RDM of presentation r}."""
    P = np.asarray(P, np.float64)
    out = {"k3": pdist(P.mean(axis=1), dist)}
    for r in range(P.shape[1]):
        out[f"k1_{r}"] = pdist(P[:, r], dist)
    return out


def model_vectors(feat):
    """{(label, seed, units, dist): condensed RDM}; feat = {(label, seed): (n, units)}."""
    return {(lab, si, u, d): model_rdm(X, u, d) for (lab, si), X in feat.items() for u in UNITS for d in DISTS}


def rsa_subject(P, session, run, rois, models, k1=True):
    """rho rows for one subject. P (n, reps, v); rois {name: bool mask over v}."""
    rows = []
    mkeys = list(models)
    for d in DISTS:
        ks = [k for k in mkeys if k[3] == d]
        M = np.stack([models[k] for k in ks])
        Zm_all = rank_z(M)
        excl = {r: same_run_pairs(session, run, r) for r in range(P.shape[1])} if k1 else {}
        Zm_r = {r: rank_z(M[:, ~e]) for r, e in excl.items()}
        for roi, m in rois.items():
            bv = brain_vectors(P[:, :, m], d)
            rho = Zm_all @ rank_z(bv["k3"])[0]
            for (lab, si, u, _), v in zip(ks, rho):
                rows.append(dict(roi=roi, k=3, presentation=0, condition=lab, seed=si, model_units=u,
                                 distance=d, rho=float(v), n_pairs=len(bv["k3"])))
            for r, e in excl.items():
                rho = Zm_r[r] @ rank_z(bv[f"k1_{r}"][~e])[0]
                for (lab, si, u, _), v in zip(ks, rho):
                    rows.append(dict(roi=roi, k=1, presentation=r + 1, condition=lab, seed=si, model_units=u,
                                     distance=d, rho=float(v), n_pairs=int((~e).sum())))
    return pd.DataFrame(rows)


def component_vectors(X):
    """{'N', 'A', 'W'} condensed vectors of X (n x features) via step25_weighted_angle.nwa
    (F = center), with that module's TRI set to the n-item upper triangle during the call."""
    import step25_weighted_angle as W25
    X = np.asarray(X, np.float64)
    old = W25.TRI
    W25.TRI = np.triu_indices(X.shape[0], 1)
    try:
        return W25.nwa(X)
    finally:
        W25.TRI = old


def rsa_components(brain_X, rois, feats, keep=None):
    """Component rho rows: brain_X (n, v) (k = 3 average), rois {name: mask over v},
    feats {(label, seed): (n, units)}; keep = optional boolean pair mask (all pairs if None)."""
    mk = list(feats)
    mcomp = {k: component_vectors(feats[k]) for k in mk}
    rows = []
    for roi, m in rois.items():
        b = component_vectors(np.asarray(brain_X)[:, m])
        for c in COMPS:
            M = np.stack([mcomp[k][c] for k in mk])
            bv = b[c]
            if keep is not None:
                M, bv = M[:, keep], bv[keep]
            rho = rank_z(M) @ rank_z(bv)[0]
            for (lab, si), v in zip(mk, rho):
                rows.append(dict(roi=roi, k=3, presentation=0, condition=lab, seed=si, model_units="center",
                                 distance=c, rho=float(v), n_pairs=len(bv)))
    return pd.DataFrame(rows)


def contrasts(rho):
    """Per (roi, k, units, distance): rho Random, rho BP (means over seeds and, at k = 1, over
    presentations) and Delta = Random - BP."""
    g = rho.groupby(["roi", "k", "model_units", "distance", "condition"]).rho.mean().unstack("condition")
    out = g.rename(columns={"Random": "rho_random", "BP": "rho_bp"}).reset_index()
    out["delta"] = out.rho_random - out.rho_bp
    return out


def sign_flip_p(deltas, direction="negative"):
    """One-sided sign-flip p over the 2^n sign assignments: predicted negative -> fraction with
    mean <= observed; predicted positive -> fraction with mean >= observed."""
    d = np.asarray(deltas, np.float64)
    n = len(d)
    signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    means = (signs * d).mean(axis=1)
    if direction == "negative":
        return float(np.mean(means <= d.mean() + 1e-15))
    if direction == "positive":
        return float(np.mean(means >= d.mean() - 1e-15))
    raise ValueError(direction)


def boot_pairs(n, n_boot, seed):
    """Per resample: (i, j) image indices of the kept pairs (self-pairs dropped)."""
    rng = np.random.default_rng(seed)
    a, b = np.triu_indices(n, 1)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        si, sj = idx[a], idx[b]
        keep = si != sj
        yield si[keep], sj[keep]


# ── data ────────────────────────────────────────────────────────────────────
def load_patterns(subject):
    z = np.load(C.nsd_dir("patterns") / f"{subject}.npz")
    valid = z["valid"]
    rois = {r: z[f"local_{r}"] & valid for r in ROIS}
    return z["P"], z["session"], z["run"], rois, valid


def load_features():
    import nsd_features as F
    out = {}
    for key, lab in C.LAB.items():
        for si in C.cfg()["seed_idx"]:
            out[(lab, si)] = np.load(F.feature_dir() / f"model_{key}_seed{si}_g1.npy")
    return out


def run_subject(subject):
    t0 = time.time()
    P, session, run, rois, valid = load_patterns(subject)
    images = np.load(C.nsd_dir("patterns") / f"{subject}.npz")["images"]
    ref = C.design(C.cfg()["pilot_subject"])["images"]
    if not np.array_equal(images, ref):
        raise SystemExit(f"STOP: image order of {subject} differs from the feature order")
    feats = load_features()
    models = model_vectors(feats)
    rho = rsa_subject(P, session, run, rois, models)
    Pm = P.astype(np.float64).mean(axis=1)
    rho = pd.concat([rho, rsa_components(Pm, rois, feats)], ignore_index=True)
    rho["kind"], rho["subject"] = "rho", subject
    con = contrasts(rho)
    con["kind"], con["subject"] = "contrast", subject
    vox = pd.DataFrame([dict(kind="n_voxels", subject=subject, roi=r, value=int(m.sum())) for r, m in rois.items()]
                       + [dict(kind="n_voxels_invalid", subject=subject, value=int((~valid).sum()))])
    # cache brain RDMs for the group bootstrap (NSD_DIR only)
    cache = {}
    for d in DISTS:
        for r, m in rois.items():
            for name, v in brain_vectors(P[:, :, m], d).items():
                cache[f"{r}|{d}|{name}"] = v
    for r, m in rois.items():
        for c, v in component_vectors(Pm[:, m]).items():
            cache[f"{r}|{c}|k3"] = v
    np.savez(C.nsd_dir("rdm") / f"{subject}.npz", session=session, run=run, **cache)
    out = pd.concat([rho, con, vox], ignore_index=True)
    C.stamp(out).to_csv(C.nsd_dir("rsa") / f"{subject}.csv", index=False)
    print(f"{subject}: {len(rho)} rho values, {time.time() - t0:.0f} s")
    return out


# ── pilot gates ─────────────────────────────────────────────────────────────
def run_gates(subject):
    c = C.cfg()["pilot_gates"]
    rows = []
    d = C.design(subject)
    counts = np.array([np.isfinite(d["trial"][i]).sum() for i in range(len(d["images"]))])
    n_unique = len(np.unique(d["trial"]))
    g1 = bool(len(d["images"]) == 515 and np.all(counts == 3) and n_unique == d["trial"].size)
    rows.append(dict(gate="G1", quantity="images with exactly 3 presentations", value=float(np.sum(counts == 3)),
                     passed=g1, note=f"{len(d['images'])} images, {n_unique} distinct trials"))
    # G2 from the per-session statistics (z-scored over the 750 trials of each session)
    src = C.nsd_dir("betas", subject)
    mx_m, mx_s, q_m, q_s, inval = [], [], [], [], []
    for f in sorted(src.glob("session*.npz")):
        z = np.load(f)
        mx_m.append(float(z["zmean_absmax"]))
        mx_s.append(float(z["zsd_absdev_max"]))
        q_m.append(z["zmean_q"])
        q_s.append(z["zsd_q"])
        inval.append(int((~z["valid"]).sum()))
    g2 = bool(max(mx_m) <= c["G2"]["tolerance_mean"] and max(mx_s) <= c["G2"]["tolerance_sd"])
    qm, qs = np.array(q_m), np.array(q_s)
    rows.append(dict(gate="G2", quantity="max |mean| per voxel and session", value=max(mx_m), passed=g2,
                     note=f"{len(mx_m)} sessions; median over voxels of mean: range {qm[:, 1].min():.1e} to "
                          f"{qm[:, 1].max():.1e}; SD over voxels: min {qs[:, 0].min():.6f}, median "
                          f"{np.median(qs[:, 1]):.6f}, max {qs[:, 2].max():.6f}; invalid voxels per session "
                          f"{min(inval)}-{max(inval)}"))
    rows.append(dict(gate="G2", quantity="max |SD - 1| per voxel and session", value=max(mx_s), passed=g2))
    P, session, run, rois, valid = load_patterns(subject)
    for r in ROIS:
        n = int(rois[r].sum())
        thr = c["G3"]["min_voxels"].get(r)
        rows.append(dict(gate="G3", quantity=f"voxels {r}", value=float(n),
                         passed=None if thr is None else bool(n > thr),
                         note="reported" if thr is None else f"must exceed {thr}"))
    rows.append(dict(gate="G3", quantity="voxels excluded as invalid in some session", value=float((~valid).sum())))
    V = P[:, :, rois["V1"]].astype(np.float64)
    R = {r: pdist(V[:, r], "correlation") for r in range(3)}
    rel = {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        za, zb = rank_z(R[a]), rank_z(R[b])
        rel[(a, b)] = float(za[0] @ zb[0])
    for (a, b), v in rel.items():
        rows.append(dict(gate="G4", quantity=f"Spearman V1 RDM presentation {a + 1} vs {b + 1}", value=v,
                         passed=bool(v > 0) if (a, b) == (0, 1) else None))
    fg = pd.read_csv(C.OUT / "feature_gate.csv")
    g5a = bool(fg[fg.kind == "gate"].passed.iloc[0])
    import nsd_features as F
    n_ok = 0
    for key in C.LAB:
        for si in C.cfg()["seed_idx"]:
            p = F.feature_dir() / f"model_{key}_seed{si}_g1.npy"
            if p.exists():
                X = np.load(p)
                n_ok += int(X.shape[0] == 515 and np.isfinite(X).all())
    rows.append(dict(gate="G5", quantity="feature gate passed", value=float(g5a), passed=g5a,
                     note=fg[fg.kind == "gate"].note.iloc[0]))
    rows.append(dict(gate="G5", quantity="models with 515 finite feature rows", value=float(n_ok),
                     passed=bool(n_ok == 10)))
    df = pd.DataFrame(rows)
    df["subject"] = subject
    C.stamp(df)
    f = C.OUT / "gates.csv"
    if f.exists():
        prev = pd.read_csv(f)
        df = pd.concat([prev[prev.subject != subject], df], ignore_index=True)
    df.to_csv(f, index=False)
    df = df[df.subject == subject]
    print(df.drop(columns=["config_hash", "git_head", "seed"]).to_string(index=False))
    failed = df[df.passed == False]                         # noqa: E712
    if len(failed):
        raise SystemExit(f"STOP: gate(s) failed: {sorted(set(failed.gate))}")


# ── group ───────────────────────────────────────────────────────────────────
_G = {}


def cfg_b():
    from config_hash import load_config
    return load_config(CONFIG_B)


def boot_indices(n, n_boot, seed):
    """The resampled image indices (same RNG sequence as boot_pairs)."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def _init_boot(subs):
    feats = load_features()
    _G.update(subs=subs, models=model_vectors(feats),
              mcomp={k: component_vectors(X) for k, X in feats.items()},
              rdm={s: dict(np.load(C.nsd_dir("rdm") / f"{s}.npz")) for s in subs})


def _delta(rho, keys, units=None):
    lab = np.array([k[0] for k in keys])
    sel = np.ones(len(keys), bool) if units is None else np.array([k[2] == units for k in keys])
    return float(rho[sel & (lab == "Random")].mean() - rho[sel & (lab == "BP")].mean())


def _boot_chunk(chunk):
    """chunk: list of (b, idx). Returns (resample numbers, {cell: (len(subs), len(chunk)) Deltas})."""
    subs, models, mcomp, rdm = _G["subs"], _G["models"], _G["mcomp"], _G["rdm"]
    n = len(chunk[0][1])
    a, b = np.triu_indices(n, 1)
    out = {}
    for j, (_, idx) in enumerate(chunk):
        si, sj = idx[a], idx[b]
        keep = si != sj
        si, sj = si[keep], sj[keep]
        ci = condensed_index(si, sj, n)
        for d in DISTS:
            ks = [k for k in models if k[3] == d]
            M = np.stack([models[k][ci] for k in ks])
            Zm = rank_z(M)
            for s_i, s in enumerate(subs):
                z = rdm[s]
                kr = {r: z["session"][si, r] * 100 + z["run"][si, r] != z["session"][sj, r] * 100 + z["run"][sj, r]
                      for r in range(3)}
                Zr = {r: rank_z(M[:, k]) for r, k in kr.items()}
                for roi in ROIS:
                    rho3 = Zm @ rank_z(z[f"{roi}|{d}|k3"][ci])[0]
                    rho1 = np.mean([Zr[r] @ rank_z(z[f"{roi}|{d}|k1_{r}"][ci][kr[r]])[0] for r in range(3)], axis=0)
                    for kk, rho in ((3, rho3), (1, rho1)):
                        for u in UNITS:
                            cell = (roi, kk, u, d)
                            out.setdefault(cell, np.zeros((len(subs), len(chunk))))[s_i, j] = _delta(rho, ks, u)
        mk = list(mcomp)
        keys = [(lab, si_, "center", None) for lab, si_ in mk]
        for c in COMPS:
            Zc = rank_z(np.stack([mcomp[k][c][ci] for k in mk]))
            for s_i, s in enumerate(subs):
                for roi in ROIS:
                    rho = Zc @ rank_z(rdm[s][f"{roi}|{c}|k3"][ci])[0]
                    cell = (roi, 3, "center", c)
                    out.setdefault(cell, np.zeros((len(subs), len(chunk))))[s_i, j] = _delta(rho, keys)
    return [bb for bb, _ in chunk], out


def group_bootstrap(subs, workers=8):
    """{cell: (len(subs), n_resamples)} bootstrap Deltas per subject (same resamples for all)."""
    from concurrent.futures import ProcessPoolExecutor
    bc = C.cfg()["rsa"]["bootstrap"]
    idx = boot_indices(515, bc["n_resamples"], bc["seed"])
    items = list(enumerate(idx))
    n_chunks = workers * 4
    chunks = [items[i::n_chunks] for i in range(n_chunks)]
    acc = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_boot, initargs=(subs,)) as ex:
        for done, (bs, out) in enumerate(ex.map(_boot_chunk, chunks), 1):
            for cell, arr in out.items():
                acc.setdefault(cell, np.zeros((len(subs), len(idx))))[:, bs] = arr
            print(f"bootstrap chunk {done}/{len(chunks)} ({time.time() - t0:.0f} s)", flush=True)
    return acc


def stamp_b(df):
    from config_hash import config_hash
    C.stamp(df)
    df["config_hash_42b"] = config_hash(cfg_b())
    return df


def run_group(workers=8):
    t0 = time.time()
    cb = cfg_b()
    subs, qs = cb["all_subjects"], cb["q_subjects"]
    per = {s: pd.read_csv(C.nsd_dir("rsa") / f"{s}.csv") for s in subs}
    allp = pd.concat(per.values(), ignore_index=True)
    subjects = allp[allp.kind.isin(["rho", "n_voxels", "n_voxels_invalid"])].drop(
        columns=["config_hash", "git_head"])
    stamp_b(subjects).to_csv(C.OUT / "subjects.csv", index=False)
    con = allp[allp.kind == "contrast"]
    keys = ["roi", "k", "model_units", "distance"]
    acc = group_bootstrap(subs, workers)
    rows = []
    for key, g in con.groupby(keys):
        g = g.set_index("subject")
        for set_name, ss in (("all 8", subs), ("subj02-08", qs)):
            d = g.loc[ss, "delta"].to_numpy()
            bt = acc[tuple(key)][[subs.index(s) for s in ss]].mean(axis=0)
            rows.append(dict(zip(keys, key), subjects=set_name, n_subjects=len(ss), mean_delta=float(d.mean()),
                             mean_rho_random=float(g.loc[ss, "rho_random"].mean()),
                             mean_rho_bp=float(g.loc[ss, "rho_bp"].mean()),
                             n_negative=int((d < 0).sum()), n_positive=int((d > 0).sum()),
                             p_negative=sign_flip_p(d, "negative"), p_positive=sign_flip_p(d, "positive"),
                             ci_lo=float(np.quantile(bt, .025)), ci_hi=float(np.quantile(bt, .975))))
    grp = pd.DataFrame(rows)
    grp["kind"] = "group"
    stamp_b(grp).to_csv(C.OUT / "group.csv", index=False)
    summ = rules(grp, cb)
    summ = pd.concat([summ, pd.DataFrame([dict(rule="runtime", quantity="group step wall seconds",
                                               value=time.time() - t0)])], ignore_index=True)
    stamp_b(summ).to_csv(C.OUT / "summary.csv", index=False)
    print(summ.drop(columns=["config_hash", "git_head", "seed", "config_hash_42b"]).to_string(index=False))


def _cell(grp, roi, k, units, dist, subjects):
    q = grp[(grp.roi == roi) & (grp.k == k) & (grp.model_units == units) & (grp.distance == dist)
            & (grp.subjects == subjects)]
    assert len(q) == 1, (roi, k, units, dist, subjects)
    return q.iloc[0]


def rules(grp, cb):
    rows = []
    a8 = "all 8"
    r = _cell(grp, "V1", 3, "raw", "correlation", a8)
    rows.append(dict(rule="P1", quantity="raw, correlation, V1, k = 3: mean Delta <= 0", value=r.mean_delta,
                     passed=bool(r.mean_delta <= 0),
                     note=f"CI [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], {r.n_negative}/8 negative, p {r.p_negative:.4f}"))
    r = _cell(grp, "V1", 3, "z", "correlation", a8)
    ok = bool(r.n_negative >= 6 and r.p_negative <= 0.05 and r.ci_hi < 0)
    rows.append(dict(rule="P2", quantity="z, correlation, V1, k = 3: >= 6/8 negative, p <= 0.05, CI excludes 0",
                     value=r.mean_delta, passed=ok,
                     note=f"{r.n_negative}/8 negative, p {r.p_negative:.4f}, CI [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"))
    r = _cell(grp, "LO-obj", 3, "z", "correlation", a8)
    ok = bool(r.n_negative >= 6 and r.p_negative <= 0.05)
    rows.append(dict(rule="P3", quantity="z, correlation, LO-obj, k = 3: >= 6/8 negative, p <= 0.05",
                     value=r.mean_delta, passed=ok,
                     note=f"{r.n_negative}/8 negative, p {r.p_negative:.4f}, CI [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"))
    r3 = _cell(grp, "V1", 3, "raw", "correlation", a8)
    r1 = _cell(grp, "V1", 1, "raw", "correlation", a8)
    rows.append(dict(rule="P4", quantity="raw, correlation, V1: mean Delta(k = 3) <= mean Delta(k = 1)",
                     value=r3.mean_delta - r1.mean_delta, passed=bool(r3.mean_delta <= r1.mean_delta),
                     note=f"Delta(3) {r3.mean_delta:+.4f}, Delta(1) {r1.mean_delta:+.4f}"))
    q7 = "subj02-08"
    for name, spec in cb["predictions"].items():
        units, dist = spec["cell"]
        r = _cell(grp, spec["roi"], 3, units, dist, q7)
        if spec["direction"] == "positive":
            n_ok, p = r.n_positive, r.p_positive
        else:
            n_ok, p = r.n_negative, r.p_negative
        ok = bool(n_ok >= spec["min_subjects"] and p <= spec["p_max"])
        quantity = (f"{spec['roi']}, {units}, {dist}, k = 3, subj02-08: Delta {spec['direction']} "
                    f"in >= {spec['min_subjects']}/7, p <= {spec['p_max']}")
        rows.append(dict(rule=name, quantity=quantity, value=r.mean_delta, passed=ok,
                         note=f"{n_ok}/7 {spec['direction']}, p {p:.4f}, CI [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["subject"] and len(args) == 2:
        run_subject(args[1])
    elif args[:1] == ["gates"] and len(args) == 2:
        run_gates(args[1])
    elif args[:1] == ["group"]:
        run_group(int(args[1]) if len(args) > 1 else 8)
    else:
        raise SystemExit("usage: nsd_rsa.py {subject <s> | gates <s> | group}")

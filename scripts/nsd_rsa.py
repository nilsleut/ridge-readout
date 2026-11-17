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

Usage:
  python scripts/nsd_rsa.py subject <subject>   # results/step42/rsa_<subject>.csv
                                                # (pilot subject: results/step42/pilot_subj01.csv)
  python scripts/nsd_rsa.py gates <subject>     # pilot gates G1-G5 -> results/step42/pilot_gates.csv
  python scripts/nsd_rsa.py group               # all 8 subjects (not run in the pilot)
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


def contrasts(rho):
    """Per (roi, k, units, distance): rho Random, rho BP (means over seeds and, at k = 1, over
    presentations) and Delta = Random - BP."""
    g = rho.groupby(["roi", "k", "model_units", "distance", "condition"]).rho.mean().unstack("condition")
    out = g.rename(columns={"Random": "rho_random", "BP": "rho_bp"}).reset_index()
    out["delta"] = out.rho_random - out.rho_bp
    return out


def sign_flip_p(deltas):
    """One-sided (predicted Delta < 0): fraction of the 2^n sign assignments with mean <= observed."""
    d = np.asarray(deltas, np.float64)
    n = len(d)
    signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    means = (signs * d).mean(axis=1)
    return float(np.mean(means <= d.mean() + 1e-15))


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
    models = model_vectors(load_features())
    rho = rsa_subject(P, session, run, rois, models)
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
                cache[f"{r}|{d}|{name}"] = v.astype(np.float32)
    np.savez(C.nsd_dir("rdm") / f"{subject}.npz", session=session, run=run, **cache)
    out = pd.concat([rho, con, vox], ignore_index=True)
    name = "pilot_subj01.csv" if subject == C.cfg()["pilot_subject"] else f"rsa_{subject}.csv"
    C.OUT.mkdir(parents=True, exist_ok=True)
    C.stamp(out).to_csv(C.OUT / name, index=False)
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
    C.stamp(df).to_csv(C.OUT / "pilot_gates.csv", index=False)
    print(df.drop(columns=["config_hash", "git_head", "seed"]).to_string(index=False))
    failed = df[df.passed == False]                         # noqa: E712
    if len(failed):
        raise SystemExit(f"STOP: gate(s) failed: {sorted(set(failed.gate))}")


# ── group ───────────────────────────────────────────────────────────────────
def run_group():
    c = C.cfg()
    subs = c["subjects"]
    con = []
    for s in subs:
        name = "pilot_subj01.csv" if s == c["pilot_subject"] else f"rsa_{s}.csv"
        df = pd.read_csv(C.OUT / name)
        con.append(df[df.kind == "contrast"])
    con = pd.concat(con, ignore_index=True)
    keys = ["roi", "k", "model_units", "distance"]
    rows = []
    for key, g in con.groupby(keys):
        d = g.set_index("subject").loc[subs, "delta"].to_numpy()
        rows.append(dict(zip(keys, key), mean_delta=float(d.mean()), n_negative=int((d < 0).sum()),
                         p_signflip=sign_flip_p(d)))
    res = pd.DataFrame(rows)
    ci = group_bootstrap(subs)
    res = res.merge(ci, on=keys, how="left")
    res["kind"] = "group"
    C.stamp(res).to_csv(C.OUT / "group.csv", index=False)
    print(res.drop(columns=["config_hash", "git_head", "seed"]).to_string(index=False))


def group_bootstrap(subs):
    """Bootstrap interval of the 8-subject mean Delta for every cell."""
    b = C.cfg()["rsa"]["bootstrap"]
    feats = load_features()
    models = model_vectors(feats)
    n = 515
    rdm = {s: np.load(C.nsd_dir("rdm") / f"{s}.npz") for s in subs}
    acc = {}
    for i_b, (si, sj) in enumerate(boot_pairs(n, b["n_resamples"], b["seed"])):
        ci = condensed_index(si, sj, n)
        for d in DISTS:
            ks = [k for k in models if k[3] == d]
            lab = np.array([k[0] for k in ks])
            M = np.stack([models[k][ci] for k in ks])
            Zm = rank_z(M)
            for s in subs:
                z = rdm[s]
                keep_r = {r: z["session"][si, r] * 100 + z["run"][si, r] != z["session"][sj, r] * 100 + z["run"][sj, r]
                          for r in range(3)}
                Zm_r = {r: rank_z(M[:, k]) for r, k in keep_r.items()}
                for roi in ROIS:
                    rho3 = Zm @ rank_z(z[f"{roi}|{d}|k3"][ci])[0]
                    rho1 = np.mean([Zm_r[r] @ rank_z(z[f"{roi}|{d}|k1_{r}"][ci][keep_r[r]])[0] for r in range(3)], axis=0)
                    for k, rho in ((3, rho3), (1, rho1)):
                        for u in UNITS:
                            sel = np.array([kk[2] == u for kk in ks])
                            delta = rho[sel & (lab == "Random")].mean() - rho[sel & (lab == "BP")].mean()
                            acc.setdefault((roi, k, u, d), np.zeros((len(subs), b["n_resamples"])))[
                                subs.index(s), i_b] = delta
    rows = []
    for (roi, k, u, d), a in acc.items():
        m = a.mean(axis=0)
        rows.append(dict(roi=roi, k=k, model_units=u, distance=d, ci_lo=float(np.quantile(m, .025)),
                         ci_hi=float(np.quantile(m, .975))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["subject"] and len(args) == 2:
        run_subject(args[1])
    elif args[:1] == ["gates"] and len(args) == 2:
        run_gates(args[1])
    elif args == ["group"]:
        run_group()
    else:
        raise SystemExit("usage: nsd_rsa.py {subject <s> | gates <s> | group}")

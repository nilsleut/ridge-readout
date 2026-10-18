"""
step25_ridge.py
===============
Step 25, block F (amendments/2026-10-16_step25-predictions.md): ridge robustness.
EXPLORATORY. sub-01, sub-02, sub-03; Arm A; Conv1; NATIVE 224; reliable V1 voxels
(split-half > 0.2); the pre-specified configuration: fit.fit_condition unchanged (dual
ridge, train-only standardisation, alpha grid from config.yaml, min-MSE, 8 x 5 CV with
seed 20260818, default random folds); D1 in counting mode (assert_low_boundary=False)
and every low-edge hit classified by scripts/lower_edge_gate.py; no fit is excluded
while the gate reports Plateau. Features as in the primary analysis: g in {1, 4, 8}
from RIDGE_DATA_DIR/pilot_features, g = 2 from data/prereg_features.

Usage: python scripts/step25_ridge.py {skill|joint|perm}

skill (F1, F2):
  F1 floor: intercept-only predictions (mean of the outer training fold, the default
     folds of fit_condition); pooled r and per-fold r, median over reliable voxels,
     mean over subjects. Per-fold r of a prediction that is constant within each fold is
     undefined; it is reported as NaN.
  F2: every model fit (30 checkpoints x 3 subjects x g in {1, 2, 4, 8}): pooled r,
     per-fold r (mean over outer folds, then median over reliable voxels) and MSE skill
     per voxel 1 - sum (y - yhat)^2 / sum (y - ybar_train(fold))^2 over the pooled
     out-of-fold predictions (median over reliable voxels). Gate: pooled median r
     reproduces results/prereg/primary.csv to 1e-9, else STOP.
     F-R1 (MSE skill) and F-R2 (per-fold r): Random - BP, mean over subjects, <= 0 at
     every g with at most 1 of 5 seeds positive at each g.
joint (F4): network features concatenated with the luminance-and-contrast bank (same g),
  6 conditions x 5 seeds, g in {1, 4}, 3 subjects. Bank alone: results/prereg/
  instruments_control_band.csv (same pipeline). F-R4: Backprop, mean over subjects and
  seeds, joint - bank-alone median r <= 0.010 at g = 1 and g = 4; untrained reported
  without a rule.
perm (F3): 50 permutations (np.random.default_rng(seed).permutation(720), seeds 1-50) of
  the feature rows, seed-0 Random Weights and Backprop checkpoints, g in {1, 4}, each
  subject: null mean and 2.5 / 97.5 percentiles of the pooled median r. F-R3: the real
  pooled median r of both conditions exceeds the 97.5th percentile of its own null at
  g = 1 and g = 4 in sub-01 and sub-02; sub-03 descriptive.

Output: results/step25/ridge_skill.csv, ridge_joint.csv, ridge_permnull.csv,
        ridge_lower_edge.csv (appended per block)
"""
import step3_common as C          # noqa: I001  (pins BLAS threads)

import sys
import time

import numpy as np
import pandas as pd

import paths
from config_hash import load_config
from fit import fit_condition
from folds import outer_inner_splits
from lower_edge_gate import gate, hits_from_fit
from ridge_dual import alpha_grid, pearson_columns

SUBJ = ["sub-01", "sub-02", "sub-03"]
KEYS = list(C.RULES)                       # pilot_features keys
R_, BP = "random", "backprop"
OUT = C.REPO / "results" / "step25"
G2DIR = C.REPO / "data" / "prereg_features"
_B = {}


def alphas():
    a = load_config()["ridge"]["alpha_grid"]
    return alpha_grid(a["min"], a["max"], a["n_points"])


def brain(subject):
    if subject not in _B:
        d = paths.ridge_data_dir() / "arm_b_out"
        a, b = np.load(d / f"{subject}_arma.npz"), np.load(d / f"{subject}_armb.npz")
        v1 = a["local_V1"]
        _B[subject] = (a["betas"][:, v1].astype(np.float64), b["reliability"][v1] > 0.2)
    return _B[subject]


def feats(name, g):
    d = G2DIR if g == 2 else paths.ridge_data_dir() / "pilot_features"
    X = np.load(d / f"{name}_g{g}.npy")
    if X.dtype != np.float32:
        raise SystemExit(f"STOP: {name}_g{g} is {X.dtype}")
    return X.astype(np.float64)


def scores(res, Y, rel):
    fo = res["outer_fold_of_stimulus"]
    P = res["pred_oof"]
    per_fold = np.stack([pearson_columns(P[fo == f], Y[fo == f]) for f in range(res["n_outer"])])
    base = np.empty_like(Y)
    for f in range(res["n_outer"]):
        base[fo == f] = Y[fo != f].mean(axis=0)
    skill = 1.0 - ((Y - P) ** 2).sum(axis=0) / ((Y - base) ** 2).sum(axis=0)
    return dict(r_pooled=float(np.median(res["r"][rel])),
                r_perfold=float(np.median(per_fold.mean(axis=0)[rel])),
                mse_skill=float(np.median(skill[rel])))


def fit_job(job):
    """job = (block, subject, label, X-builder args, g, perm_seed)."""
    block, subject, label, names, g, perm = job
    Y, rel = brain(subject)
    X = np.hstack([feats(n, g) for n in names])
    if perm is not None:
        X = X[np.random.default_rng(perm).permutation(len(X))]
    t0 = time.time()
    res = fit_condition(X, Y, np.arange(len(Y)), alphas=alphas(), reliable=rel,
                        assert_low_boundary=False, context=f"{label} g{g}")
    hits = hits_from_fit(res, rel, condition=label, g=g, subject=subject)
    gd = gate(hits, load_X=lambda h: X, Y=Y) if hits else pd.DataFrame()
    if len(gd):
        gd["block"], gd["perm_seed"] = block, perm
    cls = "none" if not hits else ("invalid" if (gd.classification == "invalid").any() else "plateau")
    return dict(block=block, subject=subject, label=label, grid=g, d=X.shape[1], perm_seed=perm,
                n_low_edge_hits=len(hits), gate=cls, seconds=time.time() - t0, **scores(res, Y, rel)), gd


def run(jobs, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(fit_job, jobs))
    rows = pd.DataFrame([o[0] for o in out])
    gates = [o[1] for o in out if len(o[1])]
    return rows, (pd.concat(gates, ignore_index=True) if gates else pd.DataFrame())


def estimate(jobs):
    t0 = time.time()
    fit_job(jobs[0])
    sec = time.time() - t0
    print(f"runtime estimate: {len(jobs)} fits x {sec:.1f} s / 8 workers x 1.5 = "
          f"{len(jobs) * sec / 8 * 1.5 / 3600:.2f} h", flush=True)


def label(key, si):
    return f"{C.RULES[key]} s{si}"


def save_gate(gd, block):
    f = OUT / "ridge_lower_edge.csv"
    if not len(gd):
        print(f"{block}: no low-edge hits", flush=True)
        return
    gd = C.stamp(gd.copy())
    gd.to_csv(f, mode="a", header=not f.exists(), index=False)
    print(f"{block}: {len(gd)} low-edge hits, classes {gd.classification.value_counts().to_dict()}", flush=True)


def seed_contrast(df, metric, g):
    """Random - BP per seed, mean over subjects; returns (mean, n positive)."""
    q = df[df.grid == g]
    a = q[q.label.str.startswith("Random Weights s")].copy()
    a = a[~a.label.str.contains("BN")]
    b = q[q.label.str.startswith("Backprop s")]
    pa = a.groupby("label")[metric].mean()
    pb = b.groupby("label")[metric].mean()
    d = np.array([pa[f"Random Weights s{si}"] - pb[f"Backprop s{si}"] for si in C.SEED_IDX])
    return float(d.mean()), int((d > 0).sum()), float(d.std(ddof=1))


def block_skill():
    rows = []
    # F1 floor
    for s in SUBJ:
        Y, rel = brain(s)
        P = np.empty_like(Y)
        fo = np.full(len(Y), -1)
        for k, (tr, te, _) in enumerate(outer_inner_splits(np.arange(len(Y)), 8, 5, 20260818)):
            P[te] = Y[tr].mean(axis=0)
            fo[te] = k
        rp = pearson_columns(P, Y)
        rows.append(dict(kind="floor", subject=s, quantity="intercept-only", r_pooled=float(np.median(rp[rel])),
                         r_perfold=np.nan, mse_skill=0.0))
    floor = pd.DataFrame(rows)
    ctl = pd.read_csv(C.REPO / "results" / "prereg" / "instruments_control_band.csv")
    perm_ctl = ctl[ctl.label == "control:permuted"].groupby("grid").r_med_reliable.mean()

    jobs = [("skill", s, label(k, si), [f"model_{k}_seed{si}"], g, None)
            for s in SUBJ for g in (1, 2, 4, 8) for k in KEYS for si in C.SEED_IDX]
    estimate(jobs)
    t0 = time.time()
    fits, gd = run(jobs)
    print(f"skill fits done ({time.time() - t0:.0f}s)", flush=True)
    save_gate(gd, "skill")
    prim = pd.read_csv(C.REPO / "results" / "prereg" / "primary.csv").set_index(["subject", "label", "grid"])
    dev = max(abs(r.r_pooled - prim.loc[(r.subject, r.label, r.grid)].r_med_reliable) for r in fits.itertuples())
    print(f"gate: pooled median r vs results/prereg/primary.csv max |d| {dev:.2e}", flush=True)
    if not dev <= 1e-9:
        raise SystemExit("STOP: reproduction gate failed")
    out = [floor.assign(kind="floor")]
    for g, v in perm_ctl.items():
        out.append(pd.DataFrame([dict(kind="permuted_control_reference", grid=int(g), r_pooled=float(v),
                                      quantity="results/prereg control:permuted, mean over subjects")]))
    out.append(fits.assign(kind="fit"))
    rr = []
    for metric, rule in (("mse_skill", "F-R1"), ("r_perfold", "F-R2"), ("r_pooled", "pooled r (reference)")):
        ok = True
        for g in (1, 2, 4, 8):
            m, pos, sd = seed_contrast(fits, metric, g)
            ok &= (m <= 0) and (pos <= 1)
            rr.append(dict(kind="contrast", quantity=f"Random Weights - Backprop, {metric}", grid=g,
                           value=m, sd=sd, seeds_positive=pos))
        if rule.startswith("F-R"):
            rr.append(dict(kind="rule", quantity=f"{rule}: Random - BP in {metric} <= 0 with <= 1 of 5 seeds "
                                                 "positive at every g (mean over subjects)", passed=bool(ok)))
    rr.append(dict(kind="check", quantity="gate pooled median r vs prereg/primary.csv max |d|", value=dev))
    out.append(pd.DataFrame(rr))
    res = C.stamp(pd.concat(out, ignore_index=True))
    res.to_csv(OUT / "ridge_skill.csv", index=False)
    print(pd.DataFrame(rr).to_string(index=False))
    print(floor.to_string(index=False))


def block_joint():
    jobs = [("joint", s, label(k, si), [f"model_{k}_seed{si}", "control_lumcontrast"], g, None)
            for s in SUBJ for g in (1, 4) for k in KEYS for si in C.SEED_IDX]
    estimate(jobs)
    t0 = time.time()
    fits, gd = run(jobs)
    print(f"joint fits done ({time.time() - t0:.0f}s)", flush=True)
    save_gate(gd, "joint")
    bank = pd.read_csv(C.REPO / "results" / "prereg" / "instruments_control_band.csv")
    bank = bank[bank.label == "control:lumcontrast"].set_index(["subject", "grid"]).r_med_reliable
    fits["bank_alone"] = [bank[(r.subject, r.grid)] for r in fits.itertuples()]
    fits["joint_minus_bank"] = fits.r_pooled - fits.bank_alone
    rows = []
    ok = True
    for g in (1, 4):
        for key in KEYS:
            q = fits[(fits.grid == g) & fits.label.str.startswith(C.RULES[key] + " s")]
            if key == "random":
                q = q[~q.label.str.contains("BN")]
            v = float(q.joint_minus_bank.mean())
            rows.append(dict(kind="joint_minus_bank", quantity=C.RULES[key], grid=g, value=v,
                             joint=float(q.r_pooled.mean()), bank=float(q.bank_alone.mean())))
            if key == BP:
                ok &= v <= 0.010
    rows.append(dict(kind="rule", quantity="F-R4: Backprop joint - bank-alone median r <= 0.010 at g = 1 and g = 4 "
                                           "(mean over subjects and seeds)", passed=bool(ok)))
    res = C.stamp(pd.concat([fits.assign(kind="fit"), pd.DataFrame(rows)], ignore_index=True))
    res.to_csv(OUT / "ridge_joint.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def block_perm():
    jobs = [("perm", s, label(k, 0), [f"model_{k}_seed0"], g, p)
            for s in SUBJ for g in (1, 4) for k in (R_, BP) for p in range(1, 51)]
    estimate(jobs)
    t0 = time.time()
    fits, gd = run(jobs)
    print(f"perm fits done ({time.time() - t0:.0f}s)", flush=True)
    save_gate(gd, "perm")
    prim = pd.read_csv(C.REPO / "results" / "prereg" / "primary.csv").set_index(["subject", "label", "grid"])
    rows = []
    ok = True
    for s in SUBJ:
        for g in (1, 4):
            for key in (R_, BP):
                lab = label(key, 0)
                null = fits[(fits.subject == s) & (fits.grid == g) & (fits.label == lab)].r_pooled.to_numpy()
                assert len(null) == 50
                real = float(prim.loc[(s, lab, g)].r_med_reliable)
                p975 = float(np.percentile(null, 97.5))
                above = real > p975
                if s in ("sub-01", "sub-02"):
                    ok &= above
                rows.append(dict(kind="null", subject=s, grid=g, quantity=lab, null_mean=float(null.mean()),
                                 null_p025=float(np.percentile(null, 2.5)), null_p975=p975, real=real,
                                 real_above_p975=bool(above)))
    rows.append(dict(kind="rule", quantity="F-R3: real pooled median r > 97.5th percentile of its null at g = 1 and "
                                           "g = 4 for both conditions in sub-01 and sub-02", passed=bool(ok)))
    res = C.stamp(pd.concat([fits.assign(kind="fit"), pd.DataFrame(rows)], ignore_index=True))
    res.to_csv(OUT / "ridge_permnull.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"skill": block_skill, "joint": block_joint, "perm": block_perm}[sys.argv[1]]()

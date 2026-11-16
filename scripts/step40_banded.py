"""
step40_banded.py
================
Step 40, part A (amendments/2026-11-16_step40-predictions.md): banded ridge. EXPLORATORY.

Per subject (sub-01..03), V1, reliable voxels for reporting (split-half > 0.2), Conv1
features of pilot_features (224 px NATIVE): bank alone (control_lumcontrast, 2 g^2
features), network alone (model_<c>_seed<i>, 32 g^2) and the joint banded readout with
separate penalties (alpha_bank, alpha_net) on the 15-point grid 1e-6..1e8, all three with
src/ridge/banded.fit_banded (same code, same grid, same 8 x 5 folds, seed 20260818).
Scores: Pearson r of the pooled out-of-fold predictions and MSE skill against the
training-fold mean, each as the median over reliable voxels.

Low-edge selections at reliable voxels: plateau diagnostic per penalty (banded.py); any
"invalid" stops the step. Runtime: one full joint fit (Backprop, seed 0, sub-01, g = 4) is
timed first; if g = 4 for all 10 conditions x 3 subjects exceeds 12 CPU hours: STOP. g = 1
and g = 8 are run only if their estimate stays below a further 12 CPU hours.

U_c = median r(joint) - median r(bank alone), per seed, mean over subjects.
B1: joint - bank >= -0.002 for every condition, seed and subject (else STOP: implementation
error). B2: U_BP > 0 in >= 4 of 5 seeds and mean U_BP > 0. B3: U_BP - U_Random > 0 in
>= 4 of 5 seed pairs and mean > 0. B2 and B3 are evaluated at g = 4 (primary); g = 1 and 8
are reported without a rule.

Output: results/step40/banded.csv, results/step40/banded_summary.csv
"""
import step40_common as S     # noqa: I001  (pins BLAS threads)

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import paths
from banded import fit_banded
from ridge_dual import alpha_grid

SUBJ = ["sub-01", "sub-02", "sub-03"]
_B = {}


def grid():
    a = S.cfg()["banded"]["alpha_grid"]
    return alpha_grid(a["min"], a["max"], a["n_points"])


def brain(subject):
    if subject not in _B:
        d = paths.ridge_data_dir() / "arm_b_out"
        a, b = np.load(d / f"{subject}_arma.npz"), np.load(d / f"{subject}_armb.npz")
        v1 = a["local_V1"]
        thr = S.cfg()["banded"]["reliability_threshold"]
        _B[subject] = (a["betas"][:, v1].astype(np.float64), b["reliability"][v1] > thr)
    return _B[subject]


def feats(name, g):
    X = np.load(paths.ridge_data_dir() / "pilot_features" / f"{name}_g{g}.npy")
    if X.dtype != np.float32:
        raise SystemExit(f"STOP: {name}_g{g} is {X.dtype}")
    return X.astype(np.float64)


def fit_job(job):
    model, cond, si, subject, g = job
    Y, rel = brain(subject)
    G = grid()
    if model == "bank":
        groups, grids = [feats("control_lumcontrast", g)], [G]
    elif model == "net":
        groups, grids = [feats(f"model_{cond}_seed{si}", g)], [G]
    else:
        groups, grids = [feats("control_lumcontrast", g), feats(f"model_{cond}_seed{si}", g)], [G, G]
    c = S.cfg()["cv"]
    t0 = time.time()
    res = fit_banded(groups, Y, grids=grids, n_outer=c["n_outer"], n_inner=c["n_inner"], seed=c["seed"],
                     reliable=rel)
    hits = pd.DataFrame(res["hits"])
    names = ["bank", "net"] if model == "joint" else [model]
    row = dict(subject=subject, model=model, condition=cond or "bank", seed=si, g=g,
               d=sum(x.shape[1] for x in groups),
               median_r=float(np.median(res["r"][rel])),
               mse_skill=float(np.median(res["mse_skill"][rel])),
               n_reliable=int(rel.sum()), seconds=time.time() - t0)
    for gi, nm in enumerate(names):
        h = hits[hits.group == gi] if len(hits) else hits
        row[f"n_low_edge_{nm}"] = int(len(h))
        row[f"median_alpha_{nm}"] = float(np.median(res["selected"][gi][:, rel]))
    row["n_invalid"] = int((hits.classification == "invalid").sum()) if len(hits) else 0
    row["max_gain"] = float(hits.gain.max()) if len(hits) else np.nan
    return row


def jobs_for(g):
    out = []
    for s in SUBJ:
        out.append(("bank", None, None, s, g))
        for c in S.KEYS:
            for si in S.cfg()["seed_idx"]:
                out.append(("net", c, si, s, g))
                out.append(("joint", c, si, s, g))
    return out


def estimate(g, timing):
    """CPU seconds for all jobs at g, from timed single fits."""
    n = {"bank": 3, "net": 30, "joint": 30}
    return sum(timing[m] * n[m] for m in n)


def summary(df):
    rows = []
    tol = S.cfg()["banded"]["b1_tolerance"]
    for g, q in df.groupby("g"):
        bank = q[q.model == "bank"].set_index("subject").median_r
        bank_s = q[q.model == "bank"].set_index("subject").mse_skill
        j = q[q.model == "joint"].copy()
        j["diff"] = j.median_r - j.subject.map(bank)
        j["diff_skill"] = j.mse_skill - j.subject.map(bank_s)
        n = q[q.model == "net"]
        U = j.groupby(["condition", "seed"])["diff"].mean().unstack(0)          # seeds x condition
        Us = j.groupby(["condition", "seed"])["diff_skill"].mean().unstack(0)
        for si in U.index:
            rows.append(dict(kind="U", g=g, seed=si, U_BP=U.loc[si, "backprop"], U_Random=U.loc[si, "random"],
                             U_diff=U.loc[si, "backprop"] - U.loc[si, "random"],
                             U_skill_BP=Us.loc[si, "backprop"], U_skill_Random=Us.loc[si, "random"],
                             net_BP=n[(n.condition == "backprop") & (n.seed == si)].median_r.mean(),
                             net_Random=n[(n.condition == "random") & (n.seed == si)].median_r.mean()))
        d = U["backprop"] - U["random"]
        rows.append(dict(kind="U_mean", g=g, U_BP=U["backprop"].mean(), U_Random=U["random"].mean(),
                         U_diff=d.mean(), U_skill_BP=Us["backprop"].mean(), U_skill_Random=Us["random"].mean(),
                         bank=bank.mean(), bank_skill=bank_s.mean(),
                         net_BP=n[n.condition == "backprop"].median_r.mean(),
                         net_Random=n[n.condition == "random"].median_r.mean()))
        b1 = bool(j["diff"].min() >= -tol)
        rows.append(dict(kind="rule", g=g, quantity="B1 joint - bank >= -0.002 for every condition, seed, subject",
                         value=float(j["diff"].min()), passed=b1))
        if g == 4:
            b2 = bool((U["backprop"] > 0).sum() >= 4 and U["backprop"].mean() > 0)
            b3 = bool((d > 0).sum() >= 4 and d.mean() > 0)
            rows.append(dict(kind="rule", g=g, quantity="B2 U_BP > 0 in >= 4 of 5 seeds and mean > 0",
                             value=float(U["backprop"].mean()), passed=b2,
                             note=f"{int((U['backprop'] > 0).sum())} of 5 seeds positive"))
            rows.append(dict(kind="rule", g=g, quantity="B3 U_BP - U_Random > 0 in >= 4 of 5 pairs and mean > 0",
                             value=float(d.mean()), passed=b3, note=f"{int((d > 0).sum())} of 5 pairs positive"))
        hits = q[[c for c in q.columns if c.startswith("n_low_edge")]].sum()
        rows.append(dict(kind="lower_edge", g=g, quantity="low-edge hits at reliable voxels (all fits)",
                         value=float(hits.sum()), note="; ".join(f"{k}: {int(v)}" for k, v in hits.items())
                         + f"; invalid: {int(q.n_invalid.sum())}; max gain {q.max_gain.max():.2e}"))
    return pd.DataFrame(rows)


def main():
    S.OUT.mkdir(parents=True, exist_ok=True)
    lim = S.cfg()["banded"]["runtime_limit_hours"] * 3600
    timing = {}
    for m, c in (("joint", "backprop"), ("net", "backprop"), ("bank", None)):
        r = fit_job((m, c, 0 if c else None, "sub-01", 4))
        timing[m] = r["seconds"]
    est4 = estimate(4, timing)
    print(f"timed single fits at g = 4: {timing}; estimate g = 4: {est4 / 3600:.2f} CPU h", flush=True)
    if est4 > lim:
        raise SystemExit(f"STOP: g = 4 estimate {est4 / 3600:.1f} CPU h exceeds the limit")
    gs, e_sec = [4], 0.0
    sec = S.cfg()["banded"]["grids_secondary"]
    for g in sec:
        r = fit_job(("joint", "backprop", 0, "sub-01", g))
        t = {"joint": r["seconds"], "net": r["seconds"] * timing["net"] / timing["joint"],
             "bank": r["seconds"] * timing["bank"] / timing["joint"]}
        e = estimate(g, t)
        e_sec += e
        print(f"g = {g}: estimate {e / 3600:.2f} CPU h", flush=True)
    if e_sec <= lim:
        gs += list(sec)
    print(f"secondary grids {sec}: {e_sec / 3600:.2f} CPU h -> {'run' if e_sec <= lim else 'skipped'}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for g in gs:
            t0 = time.time()
            rows += list(ex.map(fit_job, jobs_for(g)))
            print(f"g = {g} done ({time.time() - t0:.0f} s wall)", flush=True)
    df = pd.DataFrame(rows)
    df["seed"] = df["seed"].astype("Int64")
    S.stamp(df).to_csv(S.OUT / "banded.csv", index=False)
    summ = S.stamp(summary(df))
    if "seed" in summ:
        summ["seed"] = summ["seed"].astype("Int64")
    summ.to_csv(S.OUT / "banded_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.drop(columns=["config_hash", "git_head"]).to_string(index=False))
    if df.n_invalid.sum():
        raise SystemExit("STOP: low-edge gate classified at least one selection as invalid")
    if not summ[(summ.kind == "rule") & summ.quantity.str.startswith("B1")].passed.all():
        raise SystemExit("STOP: B1 violated (implementation error)")


if __name__ == "__main__":
    main()

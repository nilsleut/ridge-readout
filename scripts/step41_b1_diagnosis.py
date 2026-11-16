"""
step41_b1_diagnosis.py
======================
Step 41, part A (amendments/2026-11-16_step41-predictions.md): diagnosis of the B1
violation of step 40. EXPLORATORY.

V1, reliable voxels for reporting, g = 4, Random Weights and Backprop x 5 seeds, 3 subjects,
pilot_features; src/ridge/banded.fit_banded with the step 40 grid (15 points 1e-6..1e8) and
folds (8 x 5, seed 20260818). Fits per subject: bank alone; per condition and seed:
  da1      joint with alpha_net fixed at 1e12 (grid of one value)
  joint2d  joint, (alpha_bank, alpha_net) searched jointly (the step 40 estimator, refitted
           here only to measure its optimism; its median r is compared with step 40)
  cond     conditional search: alpha_bank fixed per voxel and outer fold at the bank-alone
           selection, alpha_net searched over the 15-point axis
D-A1: |median r(da1) - median r(bank)| <= 1e-6 in every fit; checked first, STOP if violated.
D-A2: median r(cond) - median r(bank) >= -0.002 in every fit.
If both hold: U_cond = median r(cond) - median r(bank), per seed, mean over subjects, with B2
and B3 of step 40 applied to it and labelled post hoc.
Optimism per fit: outer-fold MSE minus the mean inner-CV MSE at the selected penalty, per voxel
and fold; median over reliable voxels of the mean over folds.
Runtime: one joint2d and one cond fit are timed first; STOP if part A exceeds 6 CPU hours.

Output: results/step41/b1_diagnosis.csv, results/step41/banded_conditional.csv,
        results/step41/summary.csv (part A rows; part N is added by step41_noise_curve.py)
"""
import step40_common as S     # noqa: I001  (pins BLAS threads)

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import paths
from banded import fit_banded
from config_hash import config_hash, load_config
from ridge_dual import alpha_grid

CONFIG = S.REPO / "config_step41.yaml"
OUT = S.REPO / "results" / "step41"
TMP = S.REPO / "data" / "step41_bank"
SUBJ = ["sub-01", "sub-02", "sub-03"]
KEYS = ["random", "backprop"]


def cfg():
    return load_config(CONFIG)


def stamp(df):
    df["config_hash"], df["git_head"] = config_hash(cfg()), S.git_head()
    return df


def setup():
    c = cfg()["diagnosis"]
    a = c["alpha_grid"]
    return c, alpha_grid(a["min"], a["max"], a["n_points"])


def brain(subject):
    c, _ = setup()
    d = paths.ridge_data_dir() / "arm_b_out"
    a, b = np.load(d / f"{subject}_arma.npz"), np.load(d / f"{subject}_armb.npz")
    v1 = a["local_V1"]
    return a["betas"][:, v1].astype(np.float64), b["reliability"][v1] > c["reliability_threshold"]


def feats(name, g):
    X = np.load(paths.ridge_data_dir() / "pilot_features" / f"{name}_g{g}.npy")
    if X.dtype != np.float32:
        raise SystemExit(f"STOP: {name}_g{g} is {X.dtype}")
    return X.astype(np.float64)


def scores(res, rel):
    opt = (res["outer_mse"] - res["inner_mse_selected"]).mean(axis=0)
    hits = pd.DataFrame(res["hits"])
    return dict(median_r=float(np.median(res["r"][rel])), mse_skill=float(np.median(res["mse_skill"][rel])),
                optimism=float(np.median(opt[rel])),
                inner_mse=float(np.median(res["inner_mse_selected"].mean(axis=0)[rel])),
                outer_mse=float(np.median(res["outer_mse"].mean(axis=0)[rel])),
                n_low_edge=int(len(hits)),
                n_invalid=int((hits.classification == "invalid").sum()) if len(hits) else 0)


def fit_job(job):
    model, cond, si, subject = job
    c, G = setup()
    g = c["grid"]
    Y, rel = brain(subject)
    cv = c["cv"]
    kw = dict(n_outer=cv["n_outer"], n_inner=cv["n_inner"], seed=cv["seed"], reliable=rel)
    t0 = time.time()
    if model == "bank":
        res = fit_banded([feats("control_lumcontrast", g)], Y, grids=[G], **kw)
        np.save(TMP / f"{subject}_selected.npy", res["selected"][0])
    else:
        groups = [feats("control_lumcontrast", g), feats(f"model_{cond}_seed{si}", g)]
        if model == "da1":
            res = fit_banded(groups, Y, grids=[G, [c["da1_alpha_net_fixed"]]], **kw)
        elif model == "joint2d":
            res = fit_banded(groups, Y, grids=[G, G], **kw)
        elif model == "cond":
            sel = np.load(TMP / f"{subject}_selected.npy")
            res = fit_banded(groups, Y, grids=[G, G], fixed={0: sel}, **kw)
        else:
            raise ValueError(model)
    return dict(subject=subject, model=model, condition=cond or "bank", seed=si, g=g,
                seconds=time.time() - t0, **scores(res, rel))


def run(jobs):
    with ProcessPoolExecutor(max_workers=8) as ex:
        return list(ex.map(fit_job, jobs))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    c, _ = setup()
    tol1, tol2 = c["da1_tolerance"], c["da2_tolerance"]
    t_all = time.time()
    rows = run([("bank", None, None, s) for s in SUBJ])
    t = {m: fit_job((m, "backprop", 0, "sub-01"))["seconds"] for m in ("da1", "joint2d", "cond")}
    est = 30 * sum(t.values()) + sum(r["seconds"] for r in rows)
    print(f"timed fits {t}; estimate part A {est / 3600:.2f} CPU h", flush=True)
    if est > c["runtime_limit_hours"] * 3600:
        raise SystemExit(f"STOP: estimate {est / 3600:.1f} CPU h exceeds {c['runtime_limit_hours']} h")
    jobs = lambda m: [(m, k, si, s) for s in SUBJ for k in KEYS for si in cfg()["seed_idx"]]  # noqa: E731

    # D-A1 first
    rows += run(jobs("da1"))
    df = pd.DataFrame(rows)
    bank = df[df.model == "bank"].set_index("subject").median_r
    da1 = df[df.model == "da1"].copy()
    da1["diff"] = da1.median_r - da1.subject.map(bank)
    da1_ok = bool(da1["diff"].abs().max() <= tol1)
    print(f"D-A1: max |da1 - bank| = {da1['diff'].abs().max():.2e} -> {'met' if da1_ok else 'VIOLATED'}", flush=True)
    summ = [dict(kind="rule", part="A", quantity="D-A1 |median r(joint, alpha_net = 1e12) - median r(bank)| <= 1e-6 in every fit",
                 value=float(da1["diff"].abs().max()), passed=da1_ok)]
    if not da1_ok:
        df["seed"] = df["seed"].astype("Int64")
        stamp(df).to_csv(OUT / "b1_diagnosis.csv", index=False)
        stamp(pd.DataFrame(summ)).to_csv(OUT / "summary.csv", index=False)
        raise SystemExit("STOP: D-A1 violated (implementation error in the joint path)")

    rows += run(jobs("joint2d"))
    rows += run(jobs("cond"))
    df = pd.DataFrame(rows)
    df["seed"] = df["seed"].astype("Int64")

    # step 40 reproduction of the 2D estimator (reported, not gated)
    s40 = pd.read_csv(S.REPO / "results" / "step40" / "banded.csv")
    s40 = s40[(s40.model == "joint") & (s40.g == c["grid"])].set_index(["subject", "condition", "seed"]).median_r
    j2 = df[df.model == "joint2d"]
    rep = float(max(abs(r.median_r - s40[(r.subject, r.condition, int(r.seed))]) for r in j2.itertuples()))

    cond = df[df.model == "cond"].copy()
    cond["diff"] = cond.median_r - cond.subject.map(bank)
    cond["diff_skill"] = cond.mse_skill - cond.subject.map(df[df.model == "bank"].set_index("subject").mse_skill)
    da2_ok = bool(cond["diff"].min() >= -tol2)
    summ.append(dict(kind="rule", part="A", quantity="D-A2 median r(conditional) - median r(bank) >= -0.002 in every fit",
                     value=float(cond["diff"].min()), passed=da2_ok))
    attributed = da1_ok and da2_ok
    summ.append(dict(kind="interpretation", part="A",
                     quantity="B1 violation of step 40 attributed to selection variance of the 2D search",
                     passed=attributed))
    U = cond.groupby(["condition", "seed"])["diff"].mean().unstack(0)
    Us = cond.groupby(["condition", "seed"])["diff_skill"].mean().unstack(0)
    for si in U.index:
        summ.append(dict(kind="U_conditional", part="A", seed=si, U_BP=U.loc[si, "backprop"],
                         U_Random=U.loc[si, "random"], U_diff=U.loc[si, "backprop"] - U.loc[si, "random"],
                         U_skill_BP=Us.loc[si, "backprop"], U_skill_Random=Us.loc[si, "random"]))
    d = U["backprop"] - U["random"]
    summ.append(dict(kind="U_conditional_mean", part="A", U_BP=U["backprop"].mean(), U_Random=U["random"].mean(),
                     U_diff=d.mean(), U_skill_BP=Us["backprop"].mean(), U_skill_Random=Us["random"].mean()))
    if attributed:
        summ.append(dict(kind="rule_post_hoc", part="A",
                         quantity="B2 (post hoc, conditional estimator) U_BP > 0 in >= 4 of 5 seeds and mean > 0",
                         value=float(U["backprop"].mean()),
                         passed=bool((U["backprop"] > 0).sum() >= 4 and U["backprop"].mean() > 0),
                         note=f"{int((U['backprop'] > 0).sum())} of 5 seeds positive"))
        summ.append(dict(kind="rule_post_hoc", part="A",
                         quantity="B3 (post hoc, conditional estimator) U_BP - U_Random > 0 in >= 4 of 5 pairs and mean > 0",
                         value=float(d.mean()), passed=bool((d > 0).sum() >= 4 and d.mean() > 0),
                         note=f"{int((d > 0).sum())} of 5 pairs positive"))
    for m in ("bank", "joint2d", "cond"):
        q = df[df.model == m]
        summ.append(dict(kind="optimism", part="A", quantity=m, value=float(q.optimism.mean()),
                         note=f"inner MSE {q.inner_mse.mean():.5f}, outer MSE {q.outer_mse.mean():.5f}"))
    summ.append(dict(kind="check", part="A", quantity="joint2d median r vs step 40 banded.csv, max |d|", value=rep))
    summ.append(dict(kind="check", part="A", quantity="low-edge hits (reliable voxels), invalid",
                     value=float(df.n_low_edge.sum()), note=f"invalid {int(df.n_invalid.sum())}"))
    summ.append(dict(kind="runtime", part="A", quantity="CPU seconds (sum of fits)", value=float(df.seconds.sum()),
                     note=f"wall {time.time() - t_all:.0f} s; estimate {est:.0f} s"))

    stamp(df[df.model != "cond"].copy()).to_csv(OUT / "b1_diagnosis.csv", index=False)
    stamp(cond).to_csv(OUT / "banded_conditional.csv", index=False)
    sm = pd.DataFrame(summ)
    sm["seed"] = sm["seed"].astype("Int64")
    stamp(sm).to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(sm.drop(columns=["config_hash", "git_head"]).to_string(index=False))
    if df.n_invalid.sum():
        raise SystemExit("STOP: low-edge gate classified at least one selection as invalid")


if __name__ == "__main__":
    main()

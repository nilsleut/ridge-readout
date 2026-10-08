"""
step4_subjects.py
=================
S2 (amendments/2026-10-03_step4-predictions.md). EXPLORATORY. sub-02 and sub-03.

Voxels: reliable V1 voxels (arm_b_out/sub-0X_armb.npz["reliability"] > 0.2; expected
518 / 286, else STOP); betas (720 stimuli) from sub-0X_arma.npz via local_V1, as in
step 2. Stimulus-order checks before any fit (else STOP):
  * stim_order_sub-0X.txt equals stim_order_sub-01.txt (the order of pilot_features);
  * the RDM of all V1 voxels from the betas reproduces OUTPUTS_720_DIR/
    fmri_rdm_V1_sub-0X.npy (Pearson >= 0.999), i.e. the betas are in that order.

Ridge: fit_condition unchanged, default (random) folds, g = 1, all 30 checkpoints, D1 in
counting mode (assert_low_boundary=False, as tolerance_measurement.py), alpha grid
unchanged; Y = all V1 voxels of the subject, score = median r over its reliable voxels.
Low-edge hits are classified with scripts/lower_edge_gate.py; invalid cells excluded.
Only the data loading is per subject.

RSA: cross-run pair set (210,205, the same as before), variants raw and z
(step4_std_rsa.transform), fMRI RDM from the subject's reliable voxels.

Also: the per-subject-then-averaged convention over all three subjects, with sub-01
from results/tolerance_all_rungs.csv (ridge), results/step4/std_rsa.csv (RSA raw, z).

S2, fixed before the run, per subject: pattern holds if Random - BP under raw RSA > 0
AND Random - BP under ridge < 0. sub-03 (286 reliable voxels) is marked weaker.

Output: results/step4/subjects.csv, results/step4/lower_edge_gate_subjects.csv
"""
import step3_common as C          # noqa: I001

import numpy as np
import pandas as pd

import paths
from common_up_pairs import pair_set, ranks, run_labels
from lower_edge_gate import gate, hits_from_fit
from step2_rsa_501 import tri_rdm
from step4_std_rsa import rsa_rows

SUBJECTS = {"sub-02": 518, "sub-03": 286}
R, BP, BN = "Random Weights", "Backprop", "Random Weights (BN-calibrated)"
DIFFS = [(R, BP), (BN, BP), (R, BN)]
OUT = C.REPO / "results" / "step4"
SEC_PER_FIT = 283 / 60          # measured in step 3 (g = 1, 8 workers, sub-01)


def load_subject(subject):
    d = paths.ridge_data_dir() / "arm_b_out"
    a, b = np.load(d / f"{subject}_arma.npz"), np.load(d / f"{subject}_armb.npz")
    v1 = a["local_V1"]
    return a["betas"][:, v1].astype(np.float64), b["reliability"][v1] > 0.2


def check_order(subject, Y):
    o = paths.outputs_720_dir()
    ref = [l.strip() for l in open(o / "stim_order_sub-01.txt") if l.strip()]
    mine = [l.strip() for l in open(o / f"stim_order_{subject}.txt") if l.strip()]
    if mine != ref:
        raise SystemExit(f"STOP: stim_order_{subject}.txt differs from stim_order_sub-01.txt")
    from common_up_pairs import TRI
    stored = np.load(o / f"fmri_rdm_V1_{subject}.npy")[TRI].astype(np.float64)
    r = float(np.corrcoef(tri_rdm(Y), stored)[0, 1])
    print(f"{subject}: stim order == sub-01; RDM vs fmri_rdm_V1_{subject}.npy Pearson {r:.8f}")
    if r < 0.999:
        raise SystemExit(f"STOP: {subject} betas do not match the stored RDM order")
    return r


def fit_job(job):
    subject, key, si = job
    Y, rel = load_subject(subject)
    res = C.fit_condition(C.load_X(key, si, 1).astype(np.float64), Y, np.arange(len(Y)),
                          alphas=C.alpha_grid(), reliable=rel, assert_low_boundary=False,
                          context=f"{subject} {C.RULES[key]} s{si} g1")
    return dict(subject=subject, rule=C.RULES[key], seed_idx=si,
                r_ridge=float(np.median(res["r"][rel])),
                hits=hits_from_fit(res, rel, subject=subject, condition=C.RULES[key], seed=si, g=1))


def diffs(p, metric, subject):
    out = []
    for rule in p.columns:
        out.append(dict(kind="condition", subject=subject, metric=metric, quantity=rule,
                        mean=p[rule].mean(), sd=p[rule].std(ddof=1), n_seeds=int(p[rule].notna().sum())))
    for a, b in DIFFS:
        d = (p[a] - p[b]).dropna()
        out.append(dict(kind="difference", subject=subject, metric=metric, quantity=f"{a} - {b}",
                        mean=d.mean(), sd=d.std(ddof=1), n_seeds=len(d), seeds_positive=int((d > 0).sum())))
    return out


def main():
    n_fits = 30 * len(SUBJECTS)
    est = n_fits * SEC_PER_FIT * 1.5 / 60
    print(f"runtime estimate: {n_fits} fits x {SEC_PER_FIT:.1f} s x 1.5 = {est:.1f} min")
    subjects = list(SUBJECTS) if est <= 60 else ["sub-02"]
    if est > 60:
        print("estimate above 60 min: running sub-02 only")

    _, _, keep = pair_set(None, run_labels())
    assert int(keep.sum()) == 210205
    checks, data = {}, {}
    for s in subjects:
        Y, rel = load_subject(s)
        if int(rel.sum()) != SUBJECTS[s]:
            raise SystemExit(f"STOP: {s} has {int(rel.sum())} reliable V1 voxels, expected {SUBJECTS[s]}")
        checks[s] = check_order(s, Y)
        data[s] = (Y, rel)

    res = C.pmap(fit_job, [(s, k, si) for s in subjects for k in C.RULES for si in C.SEED_IDX])
    hits = [h for r in res for h in r["hits"]]
    gates = [gate([h for h in hits if h["subject"] == s], load_X=lambda h: C.load_X(
        C.LABEL_KEY[h["condition"]], h["seed"], 1), Y=data[s][0]) for s in subjects
        if any(h["subject"] == s for h in hits)]
    g = pd.concat(gates, ignore_index=True) if gates else pd.DataFrame(
        columns=["subject", "condition", "seed", "voxel", "outer_fold", "relative_gain", "classification"])
    OUT.mkdir(parents=True, exist_ok=True)
    C.stamp(g).to_csv(OUT / "lower_edge_gate_subjects.csv", index=False)
    invalid = {(r.subject, r.condition, r.seed) for r in g.itertuples() if r.classification == "invalid"}

    per = pd.DataFrame([{k: v for k, v in r.items() if k != "hits"} for r in res])
    per["valid"] = [(s, r, si) not in invalid for s, r, si in zip(per.subject, per.rule, per.seed_idx)]
    rsa = pd.concat([rsa_rows(Y, rel, keep, s, lambda k, si: C.load_X(k, si, 1))
                     for s, (Y, rel) in data.items()], ignore_index=True)
    rsa = rsa[rsa.variant.isin(["raw", "z"])]
    if rsa.nan_pairs.sum():
        raise SystemExit("STOP: NaN pairs in model RDMs")

    # sub-01 from stored files
    tol = pd.read_csv(C.REPO / "results" / "tolerance_all_rungs.csv")
    tol = tol[(tol.g == 1) & (tol.selector == "min-MSE")]
    s1_ridge = tol.rename(columns={"seed": "seed_idx", "r_med_reliable": "r_ridge"})[["rule", "seed_idx", "r_ridge"]]
    s1_ridge["subject"], s1_ridge["valid"] = "sub-01", True
    s1 = pd.read_csv(OUT / "std_rsa.csv")
    s1 = s1[(s1.kind == "checkpoint") & s1.variant.isin(["raw", "z"])][["subject", "rule", "seed_idx", "variant", "rho"]]

    ridge_all = pd.concat([s1_ridge, per[["subject", "rule", "seed_idx", "r_ridge", "valid"]]], ignore_index=True)
    rsa_all = pd.concat([s1, rsa[["subject", "rule", "seed_idx", "variant", "rho"]]], ignore_index=True)
    long = pd.concat([
        ridge_all[ridge_all.valid].assign(metric="ridge", value=lambda d: d.r_ridge)[["subject", "rule", "seed_idx", "metric", "value"]],
        rsa_all.assign(metric=lambda d: "rsa_" + d.variant, value=lambda d: d.rho)[["subject", "rule", "seed_idx", "metric", "value"]],
    ], ignore_index=True)

    summ = []
    for (s, m), grp in long.groupby(["subject", "metric"]):
        summ += diffs(grp.pivot(index="seed_idx", columns="rule", values="value"), m, s)
    avg = long.groupby(["metric", "rule", "seed_idx"]).value.mean().reset_index()   # per subject, then mean
    for m, grp in avg.groupby("metric"):
        summ += diffs(grp.pivot(index="seed_idx", columns="rule", values="value"), m, "mean of 3 subjects")
    summ = pd.DataFrame(summ)

    verdicts = []
    for s in subjects:
        d = summ[(summ.kind == "difference") & (summ.subject == s) & (summ.quantity == f"{R} - {BP}")].set_index("metric")["mean"]
        holds = bool(d["rsa_raw"] > 0 and d["ridge"] < 0)
        verdicts.append(dict(kind="S2", subject=s, metric="rsa_raw > 0 and ridge < 0",
                             quantity=f"{R} - {BP}", mean=np.nan, passed=holds,
                             note=f"rsa_raw {d['rsa_raw']:+.4f}, ridge {d['ridge']:+.4f}"
                                  + ("; weaker: 286 reliable voxels" if s == "sub-03" else "")))
    long["kind"] = "checkpoint"
    out = pd.concat([long, summ, pd.DataFrame(verdicts),
                     pd.DataFrame([dict(kind="order_check", subject=s, metric="pearson vs stored V1 RDM", mean=v)
                                   for s, v in checks.items()])], ignore_index=True)
    C.stamp(out).to_csv(OUT / "subjects.csv", index=False)

    pd.set_option("display.width", 220)
    print(summ.to_string(index=False))
    print(f"\nlow-edge hits: {len(g)}; classes: {g.classification.value_counts().to_dict() if len(g) else {}}; "
          f"invalid: {sorted(invalid)}")
    for v in verdicts:
        print(f"S2 {v['subject']}: {'pattern holds' if v['passed'] else 'pattern does not hold'} ({v['note']})")


if __name__ == "__main__":
    main()

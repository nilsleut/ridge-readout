"""
step6_ridge_resolution.py
=========================
P_ridge (amendments/2026-10-09_step6-predictions.md). EXPLORATORY.

Ridge on GAP features (g = 1, Conv1 channel means from STEP5_GAP_DIR) at six NATIVE
resolutions plus UPSAMPLED 224, all 30 checkpoints, V1. Ridge convention of the
preregistration and step 4: fit_condition unchanged, default folds, Y = all V1 voxels
of the subject, score = median r over its reliable voxels (split-half > 0.2); D1 in
counting mode; low-edge hits classified with scripts/lower_edge_gate.py, invalid cells
excluded and reported. sub-01 and sub-02 primary, sub-03 reported separately.

Check before fitting: NATIVE 224 GAP features equal pilot_features g = 1 (max |diff|
reported; the ridge result at NATIVE 224 for sub-01 must equal tolerance_all_rungs.csv
min-MSE to 1e-6, else STOP).

P_ridge, fixed before the run: Random - BP (mean over sub-01/02, mean over seeds) <= 0
at all six NATIVE resolutions AND S_ridge = (224) - (32) <= 0.5 * S_raw (S_raw =
+0.0311, step 5). Both -> ridge confirms z-RSA across resolutions.

Runtime estimate before start: fits x 4.7 s x 1.5; above 3 h only NATIVE 32, NATIVE 224
and UPSAMPLED 224.

Output: results/step6/ridge_resolution.csv, results/step6/lower_edge_gate_ridge.csv
"""
import step3_common as C          # noqa: I001

import numpy as np
import pandas as pd

from lower_edge_gate import gate, hits_from_fit
from step4_subjects import load_subject
from step6_common import gap_path

R_, BP, BN = "Random Weights", "Backprop", "Random Weights (BN-calibrated)"
CONDS = [("NATIVE", px) for px in (32, 64, 96, 128, 160, 224)] + [("UPSAMPLED", 224)]
SUBJ = ["sub-01", "sub-02", "sub-03"]
S_RAW = 0.0311
SEC_PER_FIT = 283 / 60
OUT = C.REPO / "results" / "step6"


def load(arm, px, rule, si):
    return np.load(gap_path(arm, px, si, rule, "Conv1")).astype(np.float64)


def job(j):
    subject, arm, px, rule, si = j
    Y, rel = load_subject(subject)
    res = C.fit_condition(load(arm, px, rule, si), Y, np.arange(len(Y)), alphas=C.alpha_grid(),
                          reliable=rel, assert_low_boundary=False,
                          context=f"{subject} {arm} {px} {rule} s{si}")
    return dict(subject=subject, arm=arm, res=px, rule=rule, seed_idx=si,
                r_med_reliable=float(np.median(res["r"][rel])),
                hits=hits_from_fit(res, rel, subject=subject, arm=arm, res=px, condition=rule, seed=si, g=1))


def main():
    conds = list(CONDS)
    n = len(SUBJ) * len(conds) * 30
    est = n * SEC_PER_FIT * 1.5 / 60
    print(f"runtime estimate: {n} fits, {est:.0f} min")
    if est > 180:
        conds = [("NATIVE", 32), ("NATIVE", 224), ("UPSAMPLED", 224)]
        print("estimate above 3 h: NATIVE 32, NATIVE 224, UPSAMPLED 224 only")

    dev = max(float(np.abs(np.load(C.paths.ridge_data_dir() / "pilot_features" / f"model_{k}_seed{si}_g1.npy")
                           - load("NATIVE", 224, lab, si)).max()) for k, lab in C.RULES.items() for si in C.SEED_IDX)
    print(f"NATIVE 224 GAP vs pilot_features g=1: max |diff| {dev:.2e}")

    res = C.pmap(job, [(s, a, p, r, si) for s in SUBJ for a, p in conds for r in C.RULES.values() for si in C.SEED_IDX])
    per = pd.DataFrame([{k: v for k, v in r.items() if k != "hits"} for r in res])

    tol = pd.read_csv(C.REPO / "results" / "tolerance_all_rungs.csv")
    tol = tol[(tol.g == 1) & (tol.selector == "min-MSE")].set_index(["rule", "seed"]).r_med_reliable
    chk = per[(per.subject == "sub-01") & (per.arm == "NATIVE") & (per.res == 224)]
    d = max(abs(r.r_med_reliable - tol[(r.rule, r.seed_idx)]) for r in chk.itertuples())
    print(f"sub-01 NATIVE 224 vs tolerance_all_rungs.csv: max |d| {d:.2e}")
    if d > 1e-6:
        raise SystemExit("STOP: NATIVE 224 ridge does not reproduce tolerance_all_rungs.csv")

    hits = [h for r in res for h in r["hits"]]
    gates = []
    for s in SUBJ:
        hs = [h for h in hits if h["subject"] == s]
        if hs:
            Y, _ = load_subject(s)
            gates.append(gate(hs, load_X=lambda h: load(h["arm"], h["res"], h["condition"], h["seed"]), Y=Y))
    g = pd.concat(gates, ignore_index=True) if gates else pd.DataFrame(columns=["classification"])
    OUT.mkdir(parents=True, exist_ok=True)
    C.stamp(g).to_csv(OUT / "lower_edge_gate_ridge.csv", index=False)
    invalid = {(r.subject, r.arm, r.res, r.condition, r.seed) for r in g.itertuples() if r.classification == "invalid"}
    per["valid"] = [(r.subject, r.arm, r.res, r.rule, r.seed_idx) not in invalid for r in per.itertuples()]
    v = per[per.valid]

    rows = []
    for (arm, px), grp in v.groupby(["arm", "res"]):
        groups = [(s, grp[grp.subject == s]) for s in SUBJ] + [("mean sub-01/02", grp[grp.subject.isin(SUBJ[:2])])]
        for s, gg in groups:
            p = gg.groupby(["seed_idx", "rule"]).r_med_reliable.mean().unstack()
            for rule in p.columns:
                rows.append(dict(kind="condition", subject=s, arm=arm, res=px, quantity=rule,
                                 mean=p[rule].mean(), sd=p[rule].std(ddof=1), n_seeds=int(p[rule].notna().sum())))
            for a, b in ((R_, BP), (BN, BP)):
                dd = (p[a] - p[b]).dropna()
                rows.append(dict(kind="difference", subject=s, arm=arm, res=px, quantity=f"{a} - {b}",
                                 mean=dd.mean(), sd=dd.std(ddof=1), n_seeds=len(dd), seeds_positive=int((dd > 0).sum())))
    summ = pd.DataFrame(rows)
    rb = summ[(summ.kind == "difference") & (summ.quantity == f"{R_} - {BP}") & (summ.subject == "mean sub-01/02")]
    nat = rb[rb.arm == "NATIVE"].set_index("res")["mean"]
    s_ridge = float(nat[224] - nat[32]) if {32, 224} <= set(nat.index) else np.nan
    all_le0 = bool((nat <= 0).all()) and len(nat) == 6
    verdict = dict(kind="P_ridge", subject="mean sub-01/02", quantity="Random - BP <= 0 at all 6 NATIVE res; S_ridge <= 0.5 S_raw",
                   mean=s_ridge, passed=bool(all_le0 and s_ridge <= 0.5 * S_RAW),
                   note=f"all <= 0: {all_le0} ({len(nat)} res); S_ridge {s_ridge:+.4f} vs 0.5*S_raw {0.5 * S_RAW:+.4f}")
    per["kind"] = "fit"
    C.stamp(pd.concat([per, summ, pd.DataFrame([verdict])], ignore_index=True)).to_csv(OUT / "ridge_resolution.csv", index=False)

    pd.set_option("display.width", 220)
    print(summ[summ.kind == "difference"].to_string(index=False))
    print(f"\nlow-edge hits {len(g)}; classes {g.classification.value_counts().to_dict() if len(g) else {}}; invalid {sorted(invalid)}")
    print("P_ridge:", verdict["passed"], verdict["note"])


if __name__ == "__main__":
    main()

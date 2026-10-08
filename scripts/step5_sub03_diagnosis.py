"""
step5_sub03_diagnosis.py
========================
Descriptive, no rule. EXPLORATORY. Why does ridge not predict sub-03's reliable V1
voxels (step 4: median r <= 0.002 in every condition at g = 1)?

Ridge for sub-03, 286 reliable V1 voxels (Y = all 839 V1 voxels, scored on the
reliable ones, as step 4), features from pilot_features/ (NATIVE 224 px):
  * Random Weights and Backprop, all 5 seeds, g = 4 and g = 8;
  * control features pixels, luminance+contrast, Gabor at g = 1, 4, 8.
fit_condition unchanged, default folds, D1 in counting mode; low-edge hits through
scripts/lower_edge_gate.py. Plus the reliability distribution (median, quartiles) of
the reliable V1 voxels of all three subjects.

Runtime estimate before start (fits x 4.7 s x 1.5); above 60 min only the controls.

Output: results/step5/sub03_diagnosis.csv
"""
import step3_common as C          # noqa: I001

import numpy as np
import pandas as pd

import paths
from lower_edge_gate import gate, hits_from_fit
from step4_subjects import load_subject

SUBJECT = "sub-03"
CONTROLS = {"pixels": "control_pixels", "luminance+contrast": "control_lumcontrast", "gabor": "control_gabor"}
SEC_PER_FIT = 283 / 60


def feat(name, g):
    return np.load(paths.ridge_data_dir() / "pilot_features" / f"{name}_g{g}.npy").astype(np.float64)


def job(j):
    label, name, si, g = j
    Y, rel = load_subject(SUBJECT)
    X = feat(name, g)
    res = C.fit_condition(X, Y, np.arange(len(Y)), alphas=C.alpha_grid(), reliable=rel,
                          assert_low_boundary=False, context=f"{SUBJECT} {label} s{si} g{g}")
    sel = res["selected_alpha"][:, rel]
    return dict(subject=SUBJECT, features=label, seed_idx=si, grid=g, d=X.shape[1],
                r_med_reliable=float(np.median(res["r"][rel])),
                r_q75_reliable=float(np.quantile(res["r"][rel], .75)),
                hi_frac=float(np.isclose(sel, C.alpha_grid()[-1]).mean()),
                hits=hits_from_fit(res, rel, condition=label, seed=si, g=g, feature_file=name))


def main():
    model_jobs = [(C.RULES[k], f"model_{k}_seed{si}", si, g) for g in (4, 8)
                  for k in ("random", "backprop") for si in C.SEED_IDX]
    ctrl_jobs = [(lab, f"{name}", -1, g) for g in (1, 4, 8) for lab, name in CONTROLS.items()]
    est = (len(model_jobs) + len(ctrl_jobs)) * SEC_PER_FIT * 1.5 / 60
    print(f"runtime estimate: {len(model_jobs) + len(ctrl_jobs)} fits, {est:.1f} min")
    jobs = ctrl_jobs + (model_jobs if est <= 60 else [])
    if est > 60:
        print("estimate above 60 min: controls only")

    res = C.pmap(job, jobs)
    Y, _ = load_subject(SUBJECT)
    hits = [h for r in res for h in r["hits"]]
    g = gate(hits, load_X=lambda h: feat(h["feature_file"], h["g"]), Y=Y) if hits else pd.DataFrame()
    invalid = set() if g.empty else {(r.condition, r.seed, r.g) for r in g.itertuples() if r.classification == "invalid"}
    per = pd.DataFrame([{k: v for k, v in r.items() if k != "hits"} for r in res])
    per["valid"] = [(f, s, gg) not in invalid for f, s, gg in zip(per.features, per.seed_idx, per.grid)]
    per["kind"] = "fit"

    summ = (per[per.valid].groupby(["features", "grid"])
            .agg(r_med_mean=("r_med_reliable", "mean"), r_med_sd=("r_med_reliable", "std"),
                 hi_frac=("hi_frac", "mean"), n=("r_med_reliable", "size")).reset_index())
    summ["kind"] = "cell_summary"

    rel_rows = []
    for s in ("sub-01", "sub-02", "sub-03"):
        d = paths.ridge_data_dir() / "arm_b_out"
        a, b = np.load(d / f"{s}_arma.npz"), np.load(d / f"{s}_armb.npz")
        r = b["reliability"][a["local_V1"]]
        rr = r[r > 0.2]
        rel_rows.append(dict(kind="reliability", subject=s, n_v1=len(r), n_reliable=len(rr),
                             rel_q25=float(np.quantile(rr, .25)), rel_median=float(np.median(rr)),
                             rel_q75=float(np.quantile(rr, .75)), rel_median_all_v1=float(np.median(r))))
    gate_rows = [] if g.empty else [dict(kind="gate", subject=SUBJECT, features=r.condition, seed_idx=r.seed,
                                         grid=r.g, voxel=r.voxel, outer_fold=r.outer_fold,
                                         relative_gain=r.relative_gain, classification=r.classification)
                                    for r in g.itertuples()]
    out = C.stamp(pd.concat([per, summ, pd.DataFrame(rel_rows), pd.DataFrame(gate_rows)], ignore_index=True))
    C.REPO.joinpath("results", "step5").mkdir(parents=True, exist_ok=True)
    out.to_csv(C.REPO / "results" / "step5" / "sub03_diagnosis.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))
    print(pd.DataFrame(rel_rows).to_string(index=False))
    print(f"low-edge hits {len(hits)}; classes "
          f"{g.classification.value_counts().to_dict() if not g.empty else {}}; invalid {sorted(invalid)}")


if __name__ == "__main__":
    main()

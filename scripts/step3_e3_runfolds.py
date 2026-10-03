"""
step3_e3_runfolds.py
====================
E3: run-grouped folds. EXPLORATORY SENSITIVITY, NOT PART OF THE PREREGISTRATION
(config.yaml freezes random folds). sub-01 only.

Groups = run labels of the 720 stimuli for sub-01 (common_up_pairs.run_labels();
10 runs x 72 stimuli, else STOP). Outer: 5 folds of 2 runs, consecutive pairs of
rng(20260818).permutation(10); inner: 4 folds of 2 runs within the 8 training runs,
rng(20260818 + 1000*(fold+1)) (step3_common.run_folds). Passed to fit_condition via
`folds`. Everything else unchanged: train-only standardization, alpha grid
1e-6..1e8 (29), min-MSE, D1 in counting mode, lower-edge gate afterwards.

g in {1, 4, 8}, 30 checkpoints. Reported: pooled r and per-fold r (see
step3_e2_collapse.py), mean +- SD over seeds, Random - BP.

Rule, fixed before the run, with D_random = Random - BP under random folds
(tolerance_all_rungs.csv, g = 1: -0.0288):
  D_run (g = 1, pooled r) >  0 -> the fold convention (run leak) can reverse the sign
  D_run (g = 1, pooled r) <= 0 -> the sign survives run-grouped folds
Also reported: change of the absolute r levels against random folds.

Runtime estimate before start; above 90 minutes g = 4 and g = 8 are dropped.

Output: results/step3/runfolds.csv, results/step3/lower_edge_gate_e3.csv
"""
import step3_common as C          # noqa: I001

import pandas as pd

R, BP = "Random Weights", "Backprop"
GRIDS = [1, 4, 8]
# Measured in this session: 60 g=1 fits (8 outer x 5 inner) in 283 s with 8 workers.
SEC_PER_FIT_8x5 = 283 / 60
COLS = ["rule", "seed_idx", "grid", "fold_scheme", "d", "n_outer", "n_reliable",
        "r_pooled", "r_perfold", "hi_frac", "lo_frac"]


def main():
    folds = C.run_folds()                      # STOPs unless 10 runs x 72
    grids = list(GRIDS)
    est_min = 30 * len(grids) * SEC_PER_FIT_8x5 * (5 * 5) / (8 * 6) / 60
    print(f"runtime estimate: {30 * len(grids)} fits, ~{est_min:.1f} min "
          f"(scaled by eigendecompositions 5x(4+1) vs 8x(5+1)); conservative bound "
          f"{30 * len(grids) * SEC_PER_FIT_8x5 / 60:.1f} min")
    if 30 * len(grids) * SEC_PER_FIT_8x5 / 60 > 90:
        grids = [1]
        print("estimate above 90 min: g = 4 and g = 8 dropped")

    res = C.pmap(C.fit_one, [(k, s, g, "run") for g in grids for k in C.RULES for s in C.SEED_IDX])
    gate = C.stamp(C.run_gate(res, {"run": folds}))
    C.OUT.mkdir(parents=True, exist_ok=True)
    gate.to_csv(C.OUT / "lower_edge_gate_e3.csv", index=False)
    invalid = {(r.condition, r.seed, r.g) for r in gate.itertuples() if r.classification == "invalid"}

    per = pd.DataFrame([{k: r[k] for k in COLS} for r in res])
    per["valid"] = [(r, s, g) not in invalid for r, s, g in zip(per.rule, per.seed_idx, per.grid)]
    tol = pd.read_csv(C.REPO / "results" / "tolerance_all_rungs.csv")
    tol = tol[tol.selector == "min-MSE"].rename(columns={"seed": "seed_idx", "g": "grid",
                                                         "r_med_reliable": "r_pooled_randomfolds"})
    per = per.merge(tol[["rule", "seed_idx", "grid", "r_pooled_randomfolds"]], on=["rule", "seed_idx", "grid"])
    per["kind"] = "checkpoint"

    rows = []
    v = per[per.valid]
    for g in grids:
        vg = v[v.grid == g]
        for m in ("r_pooled", "r_perfold", "r_pooled_randomfolds"):
            p = vg.pivot(index="seed_idx", columns="rule", values=m)
            for rule in p.columns:
                rows.append(dict(kind="condition", grid=g, metric=m, quantity=rule,
                                 mean=p[rule].mean(), sd=p[rule].std(ddof=1), n_seeds=int(p[rule].notna().sum())))
            d = (p[R] - p[BP]).dropna()
            rows.append(dict(kind="difference", grid=g, metric=m, quantity=f"{R} - {BP}", mean=d.mean(),
                             sd=d.std(ddof=1), n_seeds=len(d), seeds_positive=int((d > 0).sum())))
        lvl = (vg.r_pooled - vg.r_pooled_randomfolds)
        rows.append(dict(kind="level_change", grid=g, metric="r_pooled run - random folds",
                         quantity="all 30 checkpoints", mean=lvl.mean(), sd=lvl.std(ddof=1), n_seeds=len(lvl)))
    summ = pd.DataFrame(rows)
    C.stamp(pd.concat([per, summ], ignore_index=True)).to_csv(C.OUT / "runfolds.csv", index=False)

    pd.set_option("display.width", 220)
    print(summ.to_string(index=False))
    print(f"\nlow-edge hits: {len(gate)}; classes: "
          f"{gate.classification.value_counts().to_dict() if len(gate) else {}}; invalid cells: {sorted(invalid)}")
    d1 = summ[(summ.kind == "difference") & (summ.grid == 1) & (summ.metric == "r_pooled")]["mean"].iloc[0]
    print(f"rule: D_run(g=1, pooled) = {d1:+.4f} vs D_random -0.0288 -> "
          f"{'fold convention can reverse the sign' if d1 > 0 else 'sign survives run-grouped folds'}")


if __name__ == "__main__":
    main()

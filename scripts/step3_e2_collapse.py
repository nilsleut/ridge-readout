"""
step3_e2_collapse.py
====================
E2: does the collapse rate produce the sign change of the pooled r? EXPLORATORY,
sub-01 only. g = 1, default (random) folds, all 30 checkpoints, 501 reliable voxels,
D1 in counting mode; every low-edge hit goes through the mandatory lower-edge gate
(amendments/2026-10-03_d1-plateau-gate.md); invalid cells are excluded and reported.

Per condition and seed:
  hi_frac    share of (reliable voxel x outer fold) selections at alpha_max
  r_pooled   median over reliable voxels of r on the pooled OOF predictions (as before)
  r_perfold  r per outer fold at the selected alpha (ridge_dual.pearson_columns),
             mean over folds, median over reliable voxels
Random - BP for both metrics and for hi_frac, mean +- SD over 5 seeds.

Rules, fixed before the run:
  Random - BP (r_perfold) >  0 -> the sign change is an artefact of the pooled r
  Random - BP (r_perfold) <= 0 -> the sign survives under per-fold r
  |hi_frac(Random) - hi_frac(BP)| > 0.10 -> recorded as collapse asymmetry

Output: results/step3/collapse_rpath_g1.csv, results/step3/lower_edge_gate_e2.csv
"""
import step3_common as C          # noqa: I001

import pandas as pd

R, BP = "Random Weights", "Backprop"
COLS = ["rule", "seed_idx", "grid", "fold_scheme", "d", "n_outer", "n_reliable",
        "r_pooled", "r_perfold", "hi_frac", "lo_frac"]


def summarise(per, invalid):
    per = per.copy()
    per["valid"] = [(r, s) not in invalid for r, s in zip(per.rule, per.seed_idx)]
    v = per[per.valid]
    rows = []
    for m in ("r_pooled", "r_perfold", "hi_frac"):
        p = v.pivot(index="seed_idx", columns="rule", values=m)
        for rule in p.columns:
            rows.append(dict(kind="condition", metric=m, quantity=rule, mean=p[rule].mean(),
                             sd=p[rule].std(ddof=1), n_seeds=int(p[rule].notna().sum())))
        d = (p[R] - p[BP]).dropna()
        rows.append(dict(kind="difference", metric=m, quantity=f"{R} - {BP}", mean=d.mean(),
                         sd=d.std(ddof=1), n_seeds=len(d), seeds_positive=int((d > 0).sum())))
    return per, pd.DataFrame(rows)


def main():
    res = C.pmap(C.fit_one, [(k, s, 1, "random") for k in C.RULES for s in C.SEED_IDX])
    gate = C.stamp(C.run_gate(res))
    C.OUT.mkdir(parents=True, exist_ok=True)
    gate.to_csv(C.OUT / "lower_edge_gate_e2.csv", index=False)
    invalid = {(r.condition, r.seed) for r in gate.itertuples() if r.classification == "invalid"}
    per, summ = summarise(pd.DataFrame([{k: r[k] for k in COLS} for r in res]), invalid)
    per["kind"] = "checkpoint"
    out = C.stamp(pd.concat([per, summ], ignore_index=True))
    out.to_csv(C.OUT / "collapse_rpath_g1.csv", index=False)

    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))
    print(f"\nlow-edge hits: {len(gate)}; classes: {gate.classification.value_counts().to_dict() if len(gate) else {}}; "
          f"invalid cells: {sorted(invalid)}")
    d = summ.set_index(["kind", "metric"]).loc[("difference", "r_perfold"), "mean"]
    h = summ.set_index(["kind", "metric"]).loc[("difference", "hi_frac"), "mean"]
    print(f"rule: Random-BP (r_perfold) = {d:+.4f} -> "
          f"{'artefact of the pooled r' if d > 0 else 'sign survives under per-fold r'}")
    print(f"hi_frac Random-BP = {h:+.3f} -> {'COLLAPSE ASYMMETRY' if abs(h) > 0.10 else 'no collapse asymmetry (<= 0.10)'}")


if __name__ == "__main__":
    main()

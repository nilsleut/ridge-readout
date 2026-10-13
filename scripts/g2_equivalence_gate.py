"""
g2_equivalence_gate.py
======================
Downstream equivalence gate (amendments/2026-10-13_g2-local-extraction.md).

Ridge at g = 4, sub-01, V1, all 30 checkpoints, unchanged fit configuration
(fit_condition defaults: 8 x 5 CV with seed 20260818, alpha grid, min-MSE), D1 handled by
the plateau gate (amendments/2026-10-03_d1-plateau-gate.md: assert_low_boundary=False,
low-edge hits classified with scripts/lower_edge_gate.py). Run once with the Modal T4
features (RIDGE_DATA_DIR/pilot_features) and once with the local CPU features
(data/prereg_features).

Pass if |median r (local) - median r (Modal)| <= 1e-4 for every checkpoint AND
|(Random - BP) local - (Random - BP) Modal| <= 1e-4, with Random - BP the mean over the
5 seeds of the per-seed difference of median r over reliable voxels.

Output: results/prereg_gate/g4_equivalence.csv
"""
import step3_common as C          # noqa: I001  (pins BLAS threads)

import numpy as np
import pandas as pd

import paths
from lower_edge_gate import gate, hits_from_fit

TOL = 1e-4
G = 4
SRC = {"modal": lambda: paths.ridge_data_dir() / "pilot_features", "local": lambda: C.REPO / "data" / "prereg_features"}
OUT = C.REPO / "results" / "prereg_gate"


def job(j):
    src, key, si = j
    Y, rel = C.brain()
    X = np.load(SRC[src]() / f"model_{key}_seed{si}_g{G}.npy")
    if X.dtype != np.float32:
        raise SystemExit(f"STOP: {src} {key} s{si} is {X.dtype}")
    X = X.astype(np.float64)
    res = C.fit_condition(X, Y, np.arange(len(Y)), alphas=C.alpha_grid(), reliable=rel,
                          assert_low_boundary=False, context=f"{src} {key} s{si} g{G}")
    hits = hits_from_fit(res, rel, condition=key, seed=si, g=G, source=src)
    cls = "none"
    if hits:
        gd = gate(hits, load_X=lambda h: X, Y=Y)
        cls = "invalid" if (gd.classification == "invalid").any() else "plateau"
    return dict(source=src, rule=C.RULES[key], seed_idx=si, r_med=float(np.median(res["r"][rel])),
                n_low_edge_hits=len(hits), gate=cls)


def main():
    res = pd.DataFrame(C.pmap(job, [(s, k, si) for s in SRC for k in C.RULES for si in C.SEED_IDX]))
    w = res.pivot_table(index=["rule", "seed_idx"], columns="source", values="r_med").reset_index()
    w["delta"] = w["local"] - w["modal"]
    gates = res.pivot_table(index=["rule", "seed_idx"], columns="source", values="gate", aggfunc="first").reset_index()
    w = w.merge(gates.rename(columns={"local": "gate_local", "modal": "gate_modal"}), on=["rule", "seed_idx"])
    w["kind"] = "checkpoint"

    def gap(col):
        p = w.pivot(index="seed_idx", columns="rule", values=col)
        return float((p["Random Weights"] - p["Backprop"]).mean())
    g_mod, g_loc = gap("modal"), gap("local")
    max_d = float(w.delta.abs().max())
    passed = bool(max_d <= TOL and abs(g_loc - g_mod) <= TOL and not (w[["gate_local", "gate_modal"]] == "invalid").any().any())
    summ = pd.DataFrame([
        dict(kind="contrast", rule="Random Weights - Backprop", modal=g_mod, local=g_loc, delta=g_loc - g_mod),
        dict(kind="gate", rule="max |delta median r| <= 1e-4 and |delta(Random - BP)| <= 1e-4",
             delta=max_d, passed=passed),
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    C.stamp(pd.concat([w, summ], ignore_index=True)).to_csv(OUT / "g4_equivalence.csv", index=False)
    pd.set_option("display.width", 200)
    print(w.to_string(index=False))
    print(summ.to_string(index=False))
    print("GATE", "PASSED" if passed else "FAILED")


if __name__ == "__main__":
    main()

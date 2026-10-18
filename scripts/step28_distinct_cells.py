"""
step28_distinct_cells.py
========================
Step 28, block D (amendments/2026-10-18_step28-predictions.md): distinct analyses among
the 36 multiverse cells. EXPLORATORY, descriptive; procedure fixed in the amendment.

Two cells are equivalent if their V1 RSA values agree to 1e-9 for all 30 own-CNN
checkpoints, all 3 subjects and NATIVE 32, NATIVE 224 and UPSAMPLED 224. The values are
recomputed with the step 7 point-estimate function (step7_multiverse._pts_one) and checked
against results/step7/multiverse.csv. Classes are the connected components of the
equivalence graph; every pair inside a class must agree (else STOP). The V1 gap at NATIVE
224 and its CI per cell are read from results/step7/multiverse_sign_table.csv (the step 7
bootstrap); members of one class must carry the same classification (else STOP).

Also checked: the expectation that the 12 euclidean cells form 4 classes.

Output: results/step28/distinct_cells.csv, results/step28/distinct_summary.csv
"""
import step3_common as C          # noqa: I001  (pins BLAS threads)

import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

import step7_multiverse as S7
from common_up_pairs import pair_set, ranks, run_labels
from v12_lift import RULES

CONDS = [("NATIVE", 32), ("NATIVE", 224), ("UPSAMPLED", 224)]
TOL = 1e-9
OUT = C.REPO / "results" / "step28"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, keep = pair_set(None, run_labels())
    brain_rank = {}
    for b in S7.BS:
        for d in S7.DS:
            for s in S7.SUBJ:
                v = pdist(S7.betas(s, "V1", b), d)
                assert np.isfinite(v).all()
                brain_rank[S7.brain_key(b, d, "V1", s)] = ranks(v[keep])

    t0 = time.time()
    jobs = [(f, d, arm, px, rule, si, "V1") for f in S7.FS for d in S7.DS for arm, px in CONDS
            for rule in RULES for si in C.SEED_IDX]
    with ProcessPoolExecutor(max_workers=8, initializer=S7._init_pts, initargs=(keep, brain_rank)) as ex:
        pts = pd.DataFrame([r for ch in ex.map(S7._pts_one, jobs, chunksize=16) for r in ch])
    print(f"point estimates: {len(jobs)} models ({time.time() - t0:.0f}s)", flush=True)

    # recomputation check against the stored step 7 values
    s7 = pd.read_csv(C.REPO / "results" / "step7" / "multiverse.csv", low_memory=False)
    s7 = s7[(s7.kind == "rho") & (s7.roi == "V1")]
    key = ["F", "D", "B", "arm", "res", "rule", "seed_idx"]
    subj = [f"rho_{s}" for s in S7.SUBJ]
    m = pts.merge(s7[key + subj], on=key, suffixes=("", "_s7"), how="left")
    assert len(m) == len(pts) and m[[c + "_s7" for c in subj]].notna().all().all()
    rec_dev = float(max((m[c] - m[c + "_s7"]).abs().max() for c in subj))
    print(f"recomputed vs results/step7: max |d| {rec_dev:.2e}", flush=True)
    if not rec_dev <= TOL:
        raise SystemExit("STOP: recomputation does not reproduce results/step7")

    cells = [(f, d, b) for f in S7.FS for d in S7.DS for b in S7.BS]
    idx = ["arm", "res", "rule", "seed_idx"]
    vec = {}
    for f, d, b in cells:
        q = pts[(pts.F == f) & (pts.D == d) & (pts.B == b)].sort_values(idx)
        assert len(q) == len(CONDS) * len(RULES) * len(C.SEED_IDX)
        vec[(f, d, b)] = q[subj].to_numpy().ravel()
    n_values = len(next(iter(vec.values())))

    dmax = {(a, c): float(np.abs(vec[a] - vec[c]).max()) for a, c in combinations(cells, 2)}
    parent = {c: c for c in cells}

    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c

    for (a, c), v in dmax.items():
        if v <= TOL:
            parent[find(c)] = find(a)
    roots = sorted({find(c) for c in cells}, key=cells.index)
    cls = {c: roots.index(find(c)) + 1 for c in cells}
    for (a, c), v in dmax.items():
        if cls[a] == cls[c] and v > TOL:
            raise SystemExit(f"STOP: class not transitive ({a} vs {c}: {v:.2e})")
    within = max([v for (a, c), v in dmax.items() if cls[a] == cls[c]] or [0.0])
    between = min(v for (a, c), v in dmax.items() if cls[a] != cls[c])

    sign = pd.read_csv(C.REPO / "results" / "step7" / "multiverse_sign_table.csv").set_index(["F", "D", "B"])
    rows = []
    for c in cells:
        g = sign.loc[c]
        where = "above 0" if g.ci_lo > 0 else ("below 0" if g.ci_hi < 0 else "includes 0")
        rows.append(dict(kind="cell", F=c[0], D=c[1], B=c[2], cls=cls[c], class_size=sum(v == cls[c] for v in cls.values()),
                         gap_native224=float(g.value), ci_lo=float(g.ci_lo), ci_hi=float(g.ci_hi), ci_class=where))
    cdf = pd.DataFrame(rows)
    per_class = cdf.groupby("cls").ci_class.agg(lambda s: sorted(set(s)))
    if (per_class.map(len) > 1).any():
        raise SystemExit("STOP: members of one class carry different CI classifications")
    ci_dev = float(cdf.groupby("cls")[["gap_native224", "ci_lo", "ci_hi"]].agg(lambda s: s.max() - s.min()).max().max())
    first = cdf.drop_duplicates("cls")
    eu = cdf[cdf.D == "euclidean"]
    n_eu = int(eu.cls.nunique())
    non_eu = cdf[cdf.D != "euclidean"]
    summ = [
        dict(kind="summary", quantity="cells", value=len(cells)),
        dict(kind="summary", quantity="distinct analyses", value=int(cdf.cls.nunique())),
        dict(kind="summary", quantity="distinct analyses, CI above 0", value=int((first.ci_class == "above 0").sum())),
        dict(kind="summary", quantity="distinct analyses, CI below 0", value=int((first.ci_class == "below 0").sum())),
        dict(kind="summary", quantity="distinct analyses, CI includes 0", value=int((first.ci_class == "includes 0").sum())),
        dict(kind="summary", quantity="cells, CI above 0", value=int((cdf.ci_class == "above 0").sum())),
        dict(kind="summary", quantity="cells, CI below 0", value=int((cdf.ci_class == "below 0").sum())),
        dict(kind="summary", quantity="cells, CI includes 0", value=int((cdf.ci_class == "includes 0").sum())),
        dict(kind="summary", quantity="distinct analyses, euclidean cells", value=n_eu),
        dict(kind="summary", quantity="distinct analyses, correlation and cosine cells", value=int(non_eu.cls.nunique())),
        dict(kind="check", quantity="expectation: 12 euclidean cells comprise 4 classes", value=n_eu, passed=bool(n_eu == 4)),
        dict(kind="check", quantity="expectation: 24 correlation and cosine cells distinct",
             value=int(non_eu.cls.nunique()), passed=bool(non_eu.cls.nunique() == 24 and non_eu.class_size.max() == 1)),
        dict(kind="check", quantity="recomputed vs results/step7 max |d|", value=rec_dev),
        dict(kind="check", quantity="max |d| within classes", value=within),
        dict(kind="check", quantity="min max|d| between classes", value=between),
        dict(kind="check", quantity="max spread of gap and CI within classes", value=ci_dev),
        dict(kind="check", quantity="values compared per cell", value=n_values),
    ]
    sdf = pd.DataFrame(summ)
    C.stamp(cdf).to_csv(OUT / "distinct_cells.csv", index=False)
    C.stamp(sdf).to_csv(OUT / "distinct_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(cdf[["F", "D", "B", "cls", "class_size", "gap_native224", "ci_lo", "ci_hi", "ci_class"]].to_string(index=False))
    print(sdf[["quantity", "value", "passed"]].to_string(index=False))


if __name__ == "__main__":
    main()

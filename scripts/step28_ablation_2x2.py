"""
step28_ablation_2x2.py
======================
Step 28, block X (amendments/2026-10-18_step28-predictions.md): mean-profile ablation on
each side. EXPLORATORY.

Own CNN, V1 (Conv1), NATIVE 224; raw GAP; correlation distance; B = paper. Model RDM in
{full, ortho} x brain RDM in {full, ortho}, both from step25_mean_profile.decompose.
V1 gap = mean over seeds of Random Weights - Backprop, mean over subjects; all cells and
the X1 contrast on the same 1000 bootstrap draws (step6_common.bootstrap).
X1 (rule): gap[full, full] - gap[ortho, ortho] has a CI above 0.
X2 (descriptive): the two mixed cells; ImageNet models (data/step8_gap, 224 px):
untrained (mean of 3) - trained in the two mixed cells, counts of CIs above / below 0.
Interpretation rule as written in the amendment.

Gate: [full, full] reproduces results/step7 (raw/correlation/paper, V1 gap NATIVE 224:
value and CI) and [ortho, ortho] reproduces results/step25 (ortho contrast, own CNN,
NATIVE 224) to 1e-9, else STOP.

Output: results/step28/ablation_2x2.csv
"""
import step3_common as C          # noqa: I001  (pins BLAS threads)

import os
import time

import numpy as np
import pandas as pd

from common_up_pairs import pair_set, ranks, run_labels
from step6_common import bootstrap, ci, roi_betas
from step25_mean_profile import IMN, UNTR, decompose, src_X

R_, BP = "Random Weights", "Backprop"
SUBJ = ["sub-01", "sub-02", "sub-03"]
SIDES = ["full", "ortho"]
CELLS = [(m, b) for m in SIDES for b in SIDES]
MIXED = [("ortho", "full"), ("full", "ortho")]
TOL = 1e-9
OUT = C.REPO / "results" / "step28"
CACHE = C.REPO / "data" / "step28_ablation.npy"


def own(rule, si):
    return ("own", "NATIVE", 224, rule, si, "V1")


def imn(m, c):
    return ("imn", m, c, 224, "V1")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, keep = pair_set(None, run_labels())
    brain_tri = {}
    for s in SUBJ:
        comp = decompose(roi_betas(s, "V1"))
        for k in SIDES:
            brain_tri[(k, s)] = comp[k]
    brains = {k: [brain_tri[(k, s)] for s in SUBJ] for k in SIDES}

    items = [(mk, bk, own(rule, si)) for mk, bk in CELLS for rule in (R_, BP) for si in C.SEED_IDX]
    items += [(mk, bk, imn(m, c)) for mk, bk in MIXED for m in IMN for c in ["trained"] + UNTR]
    t1 = time.time()
    _ = [ranks(v[keep]) for v in list(brain_tri.values())[:6]]
    t_rank = (time.time() - t1) / 6
    est = (len(items) + len(brains) * 3) * t_rank * 1001 / 8 * 1.5 / 3600
    print(f"bootstrap runtime estimate: {est:.2f} h ({len(items)} model vectors)", flush=True)
    if est > 3:
        raise SystemExit("STOP: estimate above 3 h; report and restrict to the own CNN")
    t2 = time.time()
    G = bootstrap([(lambda s=src, k=mk: decompose(src_X(s))[k]) for mk, bk, src in items],
                  [bk for mk, bk, src in items], brains, CACHE)
    os.remove(CACHE)
    print(f"bootstrap done ({time.time() - t2:.0f}s)", flush=True)
    col = {it: j for j, it in enumerate(items)}

    def gap(mk, bk):
        return np.mean([G[:, col[(mk, bk, own(R_, si))]] - G[:, col[(mk, bk, own(BP, si))]]
                        for si in C.SEED_IDX], axis=0)

    def pos(mk, bk):
        return int(sum(G[0, col[(mk, bk, own(R_, si))]] > G[0, col[(mk, bk, own(BP, si))]] for si in C.SEED_IDX))

    def imn_gap(mk, bk, m):
        u = np.mean([G[:, col[(mk, bk, imn(m, c))]] for c in UNTR], axis=0)
        return u - G[:, col[(mk, bk, imn(m, "trained"))]]

    rows = []
    for mk, bk in CELLS:
        p, lo, hi = ci(gap(mk, bk))
        rows.append(dict(kind="cell", model="own CNN", model_side=mk, brain_side=bk, res="NATIVE_224",
                         quantity="Random Weights - Backprop", value=p, ci_lo=lo, ci_hi=hi, seeds_positive=pos(mk, bk)))
    for mk, bk in MIXED:
        for m in IMN:
            p, lo, hi = ci(imn_gap(mk, bk, m))
            rows.append(dict(kind="cell", model=m, model_side=mk, brain_side=bk, res="R224",
                             quantity="untrained - trained", value=p, ci_lo=lo, ci_hi=hi))
    res = pd.DataFrame(rows)

    def cell(mk, bk, model="own CNN"):
        q = res[(res.model == model) & (res.model_side == mk) & (res.brain_side == bk)]
        assert len(q) == 1
        return q.iloc[0]

    # ── gate ──
    s7 = pd.read_csv(C.REPO / "results" / "step7" / "multiverse.csv", low_memory=False)
    s7 = s7[(s7.kind == "cell") & (s7.F == "raw") & (s7.D == "correlation") & (s7.B == "paper") & (s7.roi == "V1")
            & (s7.quantity == f"gap NATIVE 224: {R_} - {BP}")]
    s25 = pd.read_csv(C.REPO / "results" / "step25" / "mean_profile.csv", low_memory=False)
    s25 = s25[(s25.kind == "contrast") & (s25.model == "own CNN") & (s25.variant == "ortho") & (s25.roi == "V1")
              & (s25.res == "NATIVE_224")]
    assert len(s7) == 1 and len(s25) == 1
    ff, oo = cell("full", "full"), cell("ortho", "ortho")
    g7 = max(abs(ff[c] - s7.iloc[0][c]) for c in ("value", "ci_lo", "ci_hi"))
    g25 = max(abs(oo[c] - s25.iloc[0][c]) for c in ("value", "ci_lo", "ci_hi"))
    print(f"gate: [full, full] vs step7 {g7:.2e}; [ortho, ortho] vs step25 {g25:.2e}", flush=True)
    if not (g7 <= TOL and g25 <= TOL):
        raise SystemExit("STOP: reproduction gate failed")

    # ── X1 ──
    p, lo, hi = ci(gap("full", "full") - gap("ortho", "ortho"))
    x1 = bool(lo > 0)
    rows = [dict(kind="contrast", model="own CNN", res="NATIVE_224",
                 quantity="gap[full, full] - gap[ortho, ortho]", value=p, ci_lo=lo, ci_hi=hi)]

    # ── X2 counts ──
    for mk, bk in MIXED:
        q = res[(res.model != "own CNN") & (res.model_side == mk) & (res.brain_side == bk)]
        rows.append(dict(kind="count", model="ImageNet", model_side=mk, brain_side=bk, res="R224",
                         quantity="models with CI above 0", value=int((q.ci_lo > 0).sum())))
        rows.append(dict(kind="count", model="ImageNet", model_side=mk, brain_side=bk, res="R224",
                         quantity="models with CI below 0", value=int((q.ci_hi < 0).sum())))

    # ── interpretation rule ──
    keeps = {c: bool(cell(*c).ci_lo > 0 and cell(*c).value >= 0.5 * ff.value) for c in MIXED}
    incl = {c: bool(cell(*c).ci_lo <= 0 <= cell(*c).ci_hi) for c in MIXED}
    att = [c for c in MIXED if keeps[c] and incl[[o for o in MIXED if o != c][0]]]
    if len(att) == 1 and sum(keeps.values()) == 1:
        mk, bk = att[0]
        other = "brain" if mk == "ortho" else "model"
        interp = f"the intervention on the {other} side removes the advantage"
    else:
        interp = "no attribution to a side"
    rules = pd.DataFrame(rows + [
        dict(kind="rule", quantity="X1 gap[full, full] - gap[ortho, ortho] CI above 0", value=lo, passed=x1),
        dict(kind="interpretation", quantity="interpretation rule X",
             note=f"{interp}; mixed cells CI above 0 with value >= 0.5 gap[full, full]: "
                  + ", ".join(f"model {m} / brain {b}: {keeps[(m, b)]}" for m, b in MIXED)
                  + "; CI includes 0: " + ", ".join(f"model {m} / brain {b}: {incl[(m, b)]}" for m, b in MIXED)),
        dict(kind="check", quantity="gate [full, full] vs results/step7 max |d|", value=g7),
        dict(kind="check", quantity="gate [ortho, ortho] vs results/step25 max |d|", value=g25),
    ])
    out = C.stamp(pd.concat([res, rules], ignore_index=True))
    out.to_csv(OUT / "ablation_2x2.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.drop(columns=["config_hash", "git_head"]).to_string(index=False))


if __name__ == "__main__":
    main()

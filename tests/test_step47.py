"""
test_step47.py
==============
Checks for step 47 (amendments/2026-11-21_step47-predictions.md, config_step47.yaml).
All must pass before the step 47 runs.

  (i)   run centring (step47_runs.run_centre) on synthetic data with run offsets removes the offsets
        exactly (result with offsets = result without, 1e-12; run means of the centred trials 0)
  (ii)  without centring the code path reproduces the paper cell (+0.030446, results/step5) to 1e-12,
        the brain RDMs equal outputs_720 (0), the step 7 multiverse rho (36 cells, V1, NATIVE 32,
        NATIVE 224, UPSAMPLED 224, 30 checkpoints, 3 subjects) to 1e-9, the step 7 bootstrap interval
        of raw / correlation / paper to 1e-9, and with the step 7 intervals the 28 distinct analyses
        (5 above, 9 below, 14 include 0) of step 28
  (iii) RC3: run_var on synthetic data with known run means equals their variance; perm_null draws
        10 runs of 72; percentile on a known null
  (iv)  NSD ridge code path (step47_nsd_ridge.fit_one) on THINGS sub-01 V1 reproduces
        results/prereg/primary.csv (Random Weights s0 and Backprop s0, g = 4) exactly
  (v)   reliability (step47_nsd_ridge.reliability) on synthetic data: identical presentations -> 1;
        signal + noise with known single-presentation reliability (n = 40,000 images) within 0.01;
        Spearman-Brown relations exact

Run: python tests/test_step47.py [--only i,ii,iii,iv,v]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import nsd_common as C                                       # noqa: E402,F401
import paths                                                 # noqa: E402
import step47_runs as A                                      # noqa: E402
import step47_nsd_ridge as B                                 # noqa: E402

FAILS, N = [], 0


def check(name, ok, info=""):
    global N
    N += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)


def test_run_centre():
    rng = np.random.default_rng(47)
    lab = np.repeat(np.arange(10), 82)
    rng.shuffle(lab)
    S = rng.normal(size=(820, 30))
    off = rng.normal(scale=5, size=(10, 30))
    rows = np.sort(rng.choice(820, 720, replace=False))
    a = A.run_centre(S + off[lab], lab, rows)
    b = A.run_centre(S, lab, rows)
    allc = A.run_centre(S + off[lab], lab, np.arange(820))
    m = max(float(np.abs(allc[lab == r].mean(axis=0)).max()) for r in range(10))
    check("(i) run centring removes run offsets exactly", float(np.abs(a - b).max()) <= 1e-12 and m <= 1e-12,
          f"max |d| {np.abs(a - b).max():.1e}; run means after centring {m:.1e}")


def test_uncentred():
    import nsd_step43 as NS
    import step44_why as W
    import step7_multiverse as S7
    from common_up_pairs import pair_set, run_labels
    from v12_lift import load_fmri_tri
    br = A.brain_rdms("raw")
    rdm_dev = max(float(np.abs(a - b).max()) for a, b in zip(br["V1"], load_fmri_tri("V1")))
    d, _ = W.delta_on(W.things720_models("V1"), br["V1"], W.crossrun_index())
    check("(ii) uncentred path reproduces the paper cell", abs(d - 0.0304459379832895) <= 1e-12 and rdm_dev == 0,
          f"{d:.9f}; brain RDM vs outputs_720 {rdm_dev:.1e}")
    _, _, keep = pair_set(None, run_labels())
    brains = A.mv_brain_vectors("raw")
    pts = A.mv_points(brains, keep)
    ref = pd.read_csv(REPO / "results" / "step7" / "multiverse.csv", low_memory=False)
    ref = ref[(ref.kind == "rho") & (ref.roi == "V1")].set_index(["F", "D", "B", "arm", "res", "rule", "seed_idx"])
    worst = 0.0
    for r in pts.itertuples():
        worst = max(worst, abs(r.rho - float(ref.loc[(r.F, r.D, r.B, r.arm, float(r.res), r.rule, float(r.seed_idx)), f"rho_{r.subject}"])))
    check("(ii) uncentred multiverse rho reproduce step 7", worst <= 1e-9 and len(pts) == 36 * 3 * 30 * 3,
          f"{len(pts)} values, max |d| {worst:.1e}")
    cells = pd.read_csv(REPO / "results" / "step7" / "multiverse.csv", low_memory=False)
    cells = cells[(cells.kind == "cell") & (cells.roi == "V1") & (cells.quantity == "gap NATIVE 224: Random Weights - Backprop")]
    cells = cells.set_index(["F", "D", "B"])
    dr = A.mv_bootstrap(brains, [("raw", "correlation", "paper")], 8, "test")
    p, lo, hi = A.ci_of(dr[("raw", "correlation", "paper")])
    r7 = cells.loc[("raw", "correlation", "paper")]
    dev = max(abs(p - r7.value), abs(lo - r7.ci_lo), abs(hi - r7.ci_hi))
    check("(ii) bootstrap interval reproduces step 7 (raw / correlation / paper)", dev <= 1e-9,
          f"{p:+.6f} [{lo:+.6f}, {hi:+.6f}], max |d| {dev:.1e}")
    vec, cls_of = {}, {}
    for (f, d_, b), r in cells.iterrows():
        cls_of[(f, d_, b)] = A.ci_class(r.ci_lo, r.ci_hi)
        vec[(f, d_, b)] = pts[(pts.F == f) & (pts.D == d_) & (pts.B == b)].sort_values(
            ["arm", "res", "rule", "seed_idx", "subject"]).rho.to_numpy()
    cls, _, _ = NS.distinct_classes(vec, cls_of)
    first = {}
    for k, v in cls.items():
        first.setdefault(v, cls_of[k])
    cnt = [sum(x == c for x in first.values()) for c in ("above 0", "below 0", "includes 0")]
    check("(ii) class rule with the step 7 intervals gives 28 analyses, 5 / 9 / 14", len(first) == 28 and cnt == [5, 9, 14],
          f"{len(first)} analyses, above / below / include {cnt}")


def test_rc3():
    rng = np.random.default_rng(3)
    lab = np.repeat(np.arange(10), 72)
    rng.shuffle(lab)
    mu = rng.normal(size=10)
    noise = rng.normal(size=720)
    for r in range(10):
        noise[lab == r] -= noise[lab == r].mean()
    L = mu[lab] + noise
    v = A.run_var(L, lab)
    null = A.perm_null(L, 200, 1)
    lab0 = np.empty(720, int)
    lab0[np.random.default_rng(1).permutation(720)] = np.repeat(np.arange(10), 72)
    first_ok = abs(null[0] - A.run_var(L, lab0)) <= 1e-15
    pct = A.percentile(np.arange(100.0), 49.5)
    sorted_lab = np.repeat(np.arange(10), 72)
    top = A.percentile(A.perm_null(np.sort(L), 500, 2), A.run_var(np.sort(L), sorted_lab))
    check("(iii) RC3 statistic, permutation and percentile", abs(v - np.var(mu)) <= 1e-12 and first_ok and pct == 50.0 and top == 100.0,
          f"run_var {v:.6f} vs {np.var(mu):.6f}; sorted assignment percentile {top}")


def test_prereg():
    import paths as P
    d = P.ridge_data_dir()
    a, b = np.load(d / "arm_b_out" / "sub-01_arma.npz"), np.load(d / "arm_b_out" / "sub-01_armb.npz")
    v1 = a["local_V1"]
    Y = a["betas"][:, v1].astype(np.float64)
    rel = b["reliability"][v1] > 0.2
    ref = pd.read_csv(REPO / "results" / "prereg" / "primary.csv")
    worst, n = 0.0, 0
    for key, lab in (("random", "Random Weights s0"), ("backprop", "Backprop s0")):
        X = np.load(d / "pilot_features" / f"model_{key}_seed0_g4.npy").astype(np.float64)
        out = B.fit_one(X, Y, rel, f"test {lab} g4")
        r = ref[(ref.subject == "sub-01") & (ref.label == lab) & (ref.grid == 4)].iloc[0]
        worst = max(worst, abs(out["r_med_reliable"] - r.r_med_reliable), abs(out["r_med_all"] - r.r_med_all))
        n += int(out["n_low_edge_hits"] == r.n_low_edge_hits and out["gate"] == r.gate)
    check("(iv) ridge code path reproduces results/prereg on THINGS sub-01 (g = 4)", worst <= 1e-12 and n == 2,
          f"max |d r| {worst:.1e}")


def test_reliability():
    rng = np.random.default_rng(5)
    s = rng.normal(size=(40000, 1, 6))
    same = np.repeat(s, 3, axis=1)
    sb3, sb2, r = B.reliability(same)
    ok1 = np.allclose([sb3, sb2, r], 1.0, atol=1e-12)
    sig2 = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 9.0])
    P = s + rng.normal(size=(40000, 3, 6)) * np.sqrt(sig2)
    sb3, sb2, r = B.reliability(P)
    true_r = 1 / (1 + sig2)
    dev = float(np.abs(r - true_r).max())
    rel_ok = np.allclose(sb3, 3 * r / (1 + 2 * r), rtol=0, atol=1e-15) and np.allclose(sb2, 2 * r / (1 + r), rtol=0, atol=1e-15)
    check("(v) reliability on synthetic data", ok1 and dev <= 0.01 and rel_ok,
          f"single-presentation r vs truth max |d| {dev:.4f}; SB3 at r = 0.5: {3 * 0.5 / 2:.3f}")


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    print("step 47 checks")
    for tag, fn in (("i", test_run_centre), ("iii", test_rc3), ("v", test_reliability), ("iv", test_prereg), ("ii", test_uncentred)):
        if only is None or tag in only:
            fn()
    if FAILS:
        print(f"  {len(FAILS)} failure(s) of {N} checks")
        sys.exit(1)
    print(f"  all {N} checks passed")


if __name__ == "__main__":
    main()

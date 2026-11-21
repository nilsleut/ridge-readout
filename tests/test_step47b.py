"""
test_step47b.py
===============
Checks for step 47b (amendments/2026-11-21_step47b-predictions.md, config_step47b.yaml).
All must pass before the step 47b runs.

  (i)   run RDMs on synthetic data with known drift: X3 statistic (Spearman with |r - s|) near 1 for
        a slow linear drift and equal to scipy on the hand-built RDMs; permutation null: draw k of
        null_delta_a equals delta_a on draw k of perm_labels; a brain run RDM built from the Random
        model's run means gives A(Random) = 1 and a percentile >= 99
  (ii)  perm_labels keeps 10 runs of 82 in every draw; with 720 images / 72 per run it reproduces the
        step47_runs.perm_null draws exactly
  (iii) without run centring the robustness paths reproduce the existing values exactly: THINGS ridge
        results/prereg/primary.csv (g = 4, 3 subjects, 10 checkpoints), NSD P1 / P2 of step 42 (rho per
        subject and seed, group means), NSD ridge of step 47 (subj01 V1 g = 4, 11 fits); and the
        multiverse draw path: identity draw = step 43 rho (1e-9), draws 1-2 = per-subject rho_pairs (1e-12)
  (iv)  with run centring the THINGS RSA path (rc720 of step47b_robustness) reproduces the step 47
        run-centred V1 Delta (+0.0070) to 1e-12
  (v)   NSD run means on synthetic sessions: nsd_fetch.process on a synthetic int16 NIfTI with all 750
        positions, nsd_runmeans.run_means equal to hand-computed run means of the hand z-score; centre
        subtracts the run mean of each presentation's (session, run); centred trials have run mean 0

Run: python tests/test_step47b.py [--only i,ii,iii,iv,v]
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import nsd_common as C                                       # noqa: E402,F401
import paths                                                 # noqa: E402
import step47b_mechanism as M                                # noqa: E402
import step47b_robustness as R                               # noqa: E402
import nsd_runmeans as RM                                    # noqa: E402

FAILS, N = [], 0


def check(name, ok, info=""):
    global N
    N += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)


def test_run_rdms():
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    rng = np.random.default_rng(1)
    v = 200
    base, drift = rng.normal(size=v), rng.normal(size=v)
    lab = np.repeat(np.arange(10), 82)
    rng.shuffle(lab)
    t = lab.astype(float)
    Y = base + 0.02 * t[:, None] * drift + rng.normal(scale=0.01, size=(len(lab), v))
    rdm = M.run_rdm(Y, lab)
    hand = pdist(np.stack([Y[lab == r].mean(axis=0) for r in range(10)]), "correlation")
    x3 = M.spear(rdm, M.run_index_rdm())
    ridx = np.array([abs(a - b) for a in range(10) for b in range(a + 1, 10)], float)
    ok1 = np.allclose(rdm, hand, atol=1e-15) and abs(x3 - spearmanr(hand, ridx).statistic) < 1e-15 and x3 > 0.9
    Y0 = base + rng.normal(scale=0.01, size=(len(lab), v))
    x0 = M.spear(M.run_rdm(Y0, lab), M.run_index_rdm())
    check("(i) run RDM and X3 statistic on synthetic drift", ok1 and abs(x0) < 0.5,
          f"drift {x3:+.3f}, no drift {x0:+.3f}")
    feats = {(lab_, si): rng.normal(size=(820, 32)) + (2.0 if lab_ == "Random" else 0.0) * rng.normal(size=(1, 32))
             for lab_ in ("Random", "BP") for si in range(5)}
    lab0 = np.repeat(np.arange(10), 82)
    brain = M.model_run_rdms(feats, lab0)["Random"]
    a_r, a_b, d = M.delta_a(brain, M.model_run_rdms(feats, lab0))
    null = M.null_delta_a([brain], feats, 300, 7)
    labs = list(M.perm_labels(300, 7))
    k_ok = all(abs(null[k, 0] - M.delta_a(brain, M.model_run_rdms(feats, labs[k]))[2]) < 1e-15 for k in (0, 17, 299))
    pct = float(100 * (null[:, 0] < d).mean())
    check("(i) permutation null and percentile", k_ok and abs(a_r - 1) < 1e-12 and pct >= 99,
          f"A(Random) {a_r:.3f}, Delta A {d:+.3f}, percentile {pct:.1f}")


def test_perm():
    import step47_runs as A
    ok = all((np.bincount(lab, minlength=10) == 82).all() and len(lab) == 820 for lab in M.perm_labels(500, 20261122))
    L = np.random.default_rng(4).normal(size=720)
    ref = A.perm_null(L, 50, 11)
    mine = np.array([A.run_var(L, lab) for lab in M.perm_labels(50, 11, n_img=720, per_run=72)])
    check("(ii) permutation keeps 10 x 82 and follows the step 47 scheme", ok and np.array_equal(ref, mine),
          f"max |d| vs perm_null {np.abs(ref - mine).max():.1e}")


def test_reproduce():
    from concurrent.futures import ProcessPoolExecutor
    ref = pd.read_csv(REPO / "results" / "prereg" / "primary.csv")
    jobs = [(s, "raw", 4, n) for s in R.SUBJ for n in R.names()]
    with ProcessPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(R.things_job, jobs))
    worst, same = 0.0, 0
    for o in out:
        rule = "Random Weights" if o["condition"] == "Random" else "Backprop"
        r = ref[(ref.subject == o["subject"]) & (ref.rule == rule) & (ref.seed_idx == o["seed"]) & (ref.grid == 4)].iloc[0]
        worst = max(worst, abs(o["r_med_reliable"] - r.r_med_reliable), abs(o["r_med_all"] - r.r_med_all))
        same += int(o["n_low_edge_hits"] == r.n_low_edge_hits and o["gate"] == r.gate and o["n_reliable"] == r.n_reliable)
    check("(iii) THINGS ridge path reproduces results/prereg (g = 4, 30 fits)", worst <= 1e-12 and same == 30,
          f"max |d r| {worst:.1e}, {same}/30 with equal hits, gate and voxel count")

    from nsd_rsa import load_features, load_patterns, model_vectors, rsa_subject
    models = model_vectors(load_features())
    rho = []
    for s in C.cfg()["subjects"]:
        P, session, run, rois, _ = load_patterns(s)
        r = rsa_subject(P, session, run, rois, models, k1=False)
        r["variant"], r["subject"] = "raw", s
        rho.append(r)
    rho = pd.concat(rho, ignore_index=True)
    r42 = pd.read_csv(REPO / "results" / "step42" / "subjects.csv", low_memory=False)
    r42 = r42[(r42.kind == "rho") & (r42.k == 3)].set_index(["subject", "roi", "condition", "seed", "model_units", "distance"]).rho
    dev = max(abs(r.rho - float(r42.loc[(r.subject, r.roi, r.condition, r.seed, r.model_units, r.distance)])) for r in rho.itertuples())
    _, per = R.nsd_rsa_rows(rho)
    summ = pd.read_csv(REPO / "results" / "step42" / "summary.csv").set_index("rule").value
    d1 = abs(per[("raw", "V1", "raw", "correlation")].mean() - summ["P1"])
    d2 = abs(per[("raw", "V1", "z", "correlation")].mean() - summ["P2"])
    check("(iii) NSD RSA path reproduces step 42 (rho, P1, P2)", dev <= 1e-12 and d1 <= 1e-12 and d2 <= 1e-12,
          f"rho max |d| {dev:.1e} ({len(rho)} values); P1 {per[('raw', 'V1', 'raw', 'correlation')].mean():+.6f} "
          f"(|d| {d1:.1e}); P2 {per[('raw', 'V1', 'z', 'correlation')].mean():+.6f} (|d| {d2:.1e})")

    r47 = pd.read_csv(REPO / "results" / "step47" / "nsd_ridge.csv", low_memory=False)
    r47 = r47[(r47.kind == "fit") & (r47.subject == "subj01") & (r47.roi == "V1") & (r47.grid == 4)]
    jobs = [("raw", "subj01", "V1", 4, n) for n in R.names() + ["control_lumcontrast"]]
    with ProcessPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(R.nsd_job, jobs))
    worst, same = 0.0, 0
    for o in out:
        x = r47[(r47.condition == o["condition"]) & ((r47.seed == o["seed"]) if o["seed"] >= 0 else r47.seed.isna())].iloc[0]
        worst = max(worst, abs(o["r_med_reliable"] - x.r_med_reliable), abs(o["mse_skill"] - x.mse_skill))
        same += int(o["n_low_edge_hits"] == x.n_low_edge_hits and o["n_reliable"] == x.n_reliable)
    check("(iii) NSD ridge path reproduces step 47 (subj01 V1 g = 4, 11 fits)", worst <= 1e-12 and same == 11,
          f"max |d| {worst:.1e}, {same}/11 equal hits and voxel counts")

    import nsd_step43 as NS
    mkeys, Mm = NS.mv_models(load_features())
    Bs, bkeys = [], None
    for s in C.cfg()["subjects"]:
        bkeys, b = NS.mv_brains(NS.brain_X(s)["V1"])
        Bs.append(b)
    Bs = np.stack(Bs)
    pairs, lab = NS.mv_pairs(mkeys, bkeys)
    dr = R.mv_draws(Mm, Bs, pairs, 3, n_resamples=2)
    from nsd_rsa import boot_indices
    bc = NS.cfg()["inference"]["bootstrap"]
    it = list(boot_indices(NS.N_IMG, bc["n_resamples"], bc["seed"]))
    dd = 0.0
    for b in (1, 2):
        ci = NS.draw_index(it[b - 1])
        for s_i in range(Bs.shape[0]):
            dd = max(dd, float(np.abs(dr[b, s_i] - NS.rho_pairs(Mm, Bs[s_i], pairs, ci)).max()))
    r43 = pd.read_csv(REPO / "results" / "step43" / "multiverse_nsd.csv", low_memory=False)
    r43 = r43[r43.kind == "rho"].set_index(["F", "D", "B", "condition", "seed", "subject"]).rho
    subs = C.cfg()["subjects"]
    d43 = max(abs(dr[0, s_i, j] - float(r43.loc[(r.F, r.D, r.B, r.condition, float(r.seed), s)]))
              for j, r in lab.iterrows() for s_i, s in enumerate(subs))
    check("(iii) multiverse draws: identity = step 43 rho, resamples = per-subject rho_pairs", d43 <= 1e-9 and dd <= 1e-12,
          f"identity max |d| {d43:.1e}; resamples max |d| {dd:.1e}")


def test_things_rc():
    d, per = R.things_rsa_720()
    ref = pd.read_csv(REPO / "results" / "step47" / "run_centring.csv")
    x = float(ref[(ref.kind == "delta") & (ref.roi == "V1") & (ref.variant == "run-centred")].delta.iloc[0])
    check("(iv) run-centred THINGS RSA path reproduces step 47", abs(d - x) <= 1e-12, f"{d:+.9f} vs {x:+.9f}")


def test_nsd_runmeans():
    import nibabel as nib
    import nsd_fetch as F
    rng = np.random.default_rng(9)
    run_of = np.sort(np.r_[np.repeat(np.arange(12), 62), rng.choice(12, 6, replace=False)])
    vol = rng.integers(-3000, 3000, size=(4, 3, 2, 750)).astype(np.int16)
    vol[..., :] += (rng.integers(-500, 500, size=12)[run_of]).astype(np.int16)
    vol[0, 0, 0, :] = 7                                       # zero-SD voxel
    union = np.ones((4, 3, 2), bool)
    union[3, 2, 1] = False
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "syn.nii.gz"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
        Z, valid, _, _ = F.process(p, union, np.arange(750), 300)
    X = vol[union].astype(np.float64) / 300
    mu, sd = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
    hand = np.where(sd > 0, (X - mu) / np.where(sd > 0, sd, 1), 0).T
    Mn, n = RM.run_means(Z, run_of)
    Mh = np.stack([hand[run_of == r].mean(axis=0) for r in range(12)])
    ok1 = Z.shape == (750, int(union.sum())) and np.abs(Z - hand).max() < 1e-6 and np.abs(Mn - Mh).max() < 1e-6 \
        and (n == np.bincount(run_of)).all() and not valid[0]
    # centring: 500 images x 3 presentations spread over two sessions
    order = rng.permutation(1500)
    sess = np.where(order < 750, 1, 2).reshape(500, 3)
    pos = (order % 750).reshape(500, 3)
    Zs = {1: Z.astype(np.float64), 2: Z[::-1].astype(np.float64)}
    runs = run_of[pos]
    P = np.stack([np.stack([Zs[sess[i, r]][pos[i, r]] for r in range(3)]) for i in range(500)])
    RMs = {s: RM.run_means(Zs[s], run_of if s == 1 else run_of[::-1])[0].astype(np.float64) for s in (1, 2)}
    rr = np.where(sess == 1, runs, run_of[::-1][pos])
    Pc = RM.centre(P, sess, rr, RMs)
    hand_c = np.stack([np.stack([P[i, r] - RMs[sess[i, r]][rr[i, r]] for r in range(3)]) for i in range(500)])
    allc = RM.centre(Z[None].transpose(1, 0, 2).repeat(1, axis=1), np.ones((750, 1), int), run_of[:, None], {1: RM.run_means(Z, run_of)[0].astype(np.float64)})
    m0 = max(float(np.abs(allc[run_of == r, 0].mean(axis=0)).max()) for r in range(12))
    check("(v) NSD run means and centring on synthetic sessions", ok1 and np.array_equal(Pc, hand_c) and m0 < 1e-6,
          f"Z vs hand z-score {np.abs(Z - hand).max():.1e}; run means {np.abs(Mn - Mh).max():.1e}; "
          f"centred run means {m0:.1e}")


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    print("step 47b checks")
    for tag, fn in (("i", test_run_rdms), ("ii", test_perm), ("v", test_nsd_runmeans), ("iv", test_things_rc),
                    ("iii", test_reproduce)):
        if only is None or tag in only:
            fn()
    if FAILS:
        print(f"  {len(FAILS)} failure(s) of {N} checks")
        sys.exit(1)
    print(f"  all {N} checks passed")


if __name__ == "__main__":
    main()

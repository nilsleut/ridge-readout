"""
test_nsd.py
===========
Checks for the step 42 NSD pipeline (amendments/2026-11-17_step42-predictions.md).
All must pass before any NSD betas are loaded.

  (i)   extraction gate: the NSD builder of scripts/nsd_features.py (extract_pilot with the
        explicit image-path list) on the 20 Arm A images of the step 40 gate passes the
        downstream-equivalence gate against pilot_features (|delta median r| <= 1e-4)
  (ii)  synthetic data: k = 1 with run exclusion removes exactly the pairs whose chosen
        presentations share session and run; k = 3 averages the presentations correctly;
        k = 1 rho equals Spearman over the kept pairs; sign-flip p on known cases
  (iii) nsd_rsa.rsa_subject on the THINGS test images (12 presentations averaged) reproduces
        the step 41 k = 12 rho values (correlation vs b1, Euclidean vs b2; raw and z units;
        3 subjects, V1 and LOC, 10 models) to 1e-9
  (iv)  the commit check of rule (c) finds a test file deliberately placed under NSD_DIR
        (inside a temporary repository) and the pre-commit hook refuses the commit
  (v)   step 42b: nsd_rsa.rsa_components (N and A via step25_weighted_angle.nwa) on the THINGS
        data (720 stimuli, cross-run pairs) reproduces the per-subject component rho values of
        results/step9/separation.csv (own CNN, NATIVE 224, Random Weights and Backprop, 5 seeds,
        V1 and LOC, F = center, norm and angle) to 1e-9; the TRI of step25_weighted_angle is
        restored after every call

Run: python tests/test_nsd.py [--skip-gate]
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import nsd_common as C                                                              # noqa: E402,F401
import nsd_commit_check as CC                                                       # noqa: E402
from nsd_rsa import brain_vectors, model_vectors, rsa_subject, same_run_pairs, sign_flip_p   # noqa: E402
from cvrsa import spearman                                                          # noqa: E402

FAILS, N = [], 0


def check(name, ok, info=""):
    global N
    N += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAILS.append(name)


def test_synthetic():
    rng = np.random.default_rng(7)
    n, v = 12, 30
    P = rng.normal(size=(n, 3, v))
    session = rng.integers(1, 3, size=(n, 3))
    run = rng.integers(0, 2, size=(n, 3))
    a, b = np.triu_indices(n, 1)
    ok = True
    for r in range(3):
        brute = np.array([session[i, r] == session[j, r] and run[i, r] == run[j, r] for i, j in zip(a, b)])
        ok &= np.array_equal(same_run_pairs(session, run, r), brute) and brute.any() and (~brute).any()
    check("(ii) same-run pairs match brute force", bool(ok))
    bv = brain_vectors(P, "correlation")
    d = float(np.abs(bv["k3"] - pdist(P.mean(axis=1), "correlation")).max())
    d1 = max(float(np.abs(bv[f"k1_{r}"] - pdist(P[:, r], "correlation")).max()) for r in range(3))
    check("(ii) k = 3 averages the presentations, k = 1 keeps them", d < 1e-12 and d1 < 1e-12,
          f"max |d| {max(d, d1):.1e}")
    feats = {("Random", 0): rng.normal(size=(n, 20)), ("BP", 0): rng.normal(size=(n, 20))}
    models = model_vectors(feats)
    rois = {"V1": np.ones(v, bool)}
    rho = rsa_subject(P, session, run, rois, models)
    worst, npairs_ok = 0.0, True
    for r in range(3):
        keep = ~same_run_pairs(session, run, r)
        for (lab, si, u, dist), m in models.items():
            q = rho[(rho.k == 1) & (rho.presentation == r + 1) & (rho.condition == lab) & (rho.model_units == u)
                    & (rho.distance == dist)]
            ref = spearman(m[keep], pdist(P[:, r], dist)[keep])
            worst = max(worst, abs(float(q.rho.iloc[0]) - ref))
            npairs_ok &= int(q.n_pairs.iloc[0]) == int(keep.sum())
    check("(ii) k = 1 rho uses exactly the kept pairs", worst < 1e-12 and npairs_ok, f"max |d rho| {worst:.1e}")
    p_all_neg = sign_flip_p(-np.arange(1, 9, dtype=float))
    p_mixed = sign_flip_p(np.array([-1.0, 1.0] * 4))
    check("(ii) sign-flip p", abs(p_all_neg - 1 / 256) < 1e-12 and p_mixed > 0.5,
          f"all negative {p_all_neg:.5f}, symmetric {p_mixed:.3f}")


def test_things_reproduction():
    import step40_common as S40
    ref = pd.read_csv(REPO / "results" / "step41" / "noise_curve.csv")
    ref = ref[(ref.kind == "rho") & (ref.k == 12)]
    feats = {}
    for key, lab in S40.KEYS.items():
        for si in range(5):
            feats[(lab, si)] = np.load(S40.feature_dir() / "test100" / f"model_{key}_seed{si}_g1.npy")
    models = model_vectors(feats)
    worst, n = 0.0, 0
    for s in ["sub-01", "sub-02", "sub-03"]:
        reps, rois = S40.load_reps(s)
        masks = {}
        for roi, idx in rois.items():
            m = np.zeros(reps.shape[2], bool)
            m[idx] = True
            masks[roi] = m
        dummy = np.zeros((reps.shape[0], reps.shape[1]), int)
        rho = rsa_subject(reps, dummy, dummy, masks, models, k1=False)
        for row in rho.itertuples():
            b = "b1" if row.distance == "correlation" else "b2"
            q = ref[(ref.subject == s) & (ref.roi == row.roi) & (ref.condition == row.condition)
                    & (ref.seed == row.seed) & (ref.model_units == row.model_units)
                    & (ref.model_distance == row.distance) & (ref.brain_rdm == b)]
            assert len(q) == 1
            worst = max(worst, abs(row.rho - float(q.rho.iloc[0])))
            n += 1
    check("(iii) reproduces step 41 k = 12 rho on THINGS", worst <= 1e-9 and n == 3 * 2 * 10 * 2 * 2,
          f"{n} values, max |d rho| {worst:.1e}")


def test_components_reproduction():
    import step25_weighted_angle as W25
    from common_up_pairs import pair_set, run_labels
    from nsd_rsa import rsa_components
    from step6_common import roi_betas
    from step9_separation import own_X
    tri0 = W25.TRI
    ref = pd.read_csv(REPO / "results" / "step9" / "separation.csv")
    ref = ref[(ref.kind == "rho") & (ref.source == "own") & (ref.res == "NATIVE_224") & (ref.F == "center")]
    _, _, keep = pair_set(None, run_labels())
    lab = {"Random Weights": "Random", "Backprop": "BP"}
    comp = {"N": "norm", "A": "angle"}
    worst, n = 0.0, 0
    for roi in ("V1", "LOC"):
        feats = {(lab[rule], si): own_X("NATIVE", 224, rule, si, roi) for rule in lab for si in range(5)}
        for s in ("sub-01", "sub-02", "sub-03"):
            B = roi_betas(s, roi)
            rho = rsa_components(B, {roi: np.ones(B.shape[1], bool)}, feats, keep=keep)
            for row in rho[rho.distance.isin(["N", "A"])].itertuples():
                rule = [k for k, v in lab.items() if v == row.condition][0]
                q = ref[(ref.cond == rule) & (ref.seed_idx == row.seed) & (ref.roi == roi)
                        & (ref.component == comp[row.distance])]
                assert len(q) == 1
                worst = max(worst, abs(row.rho - float(q[f"rho_{s}"].iloc[0])))
                n += 1
    check("(v) N and A reproduce step 9 on THINGS", worst <= 1e-9 and n == 120 and W25.TRI is tri0,
          f"{n} values, max |d rho| {worst:.1e}")


def test_commit_check():
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "repo"
        repo.mkdir()
        g = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)  # noqa: E731
        g("init", "-q")
        g("config", "user.email", "test@example.invalid")
        g("config", "user.name", "test")
        nsd = repo / "nsd_data"
        nsd.mkdir()
        (repo / "ok.txt").write_text("fine\n")
        g("add", "ok.txt")
        _, clean = CC.check(repo, nsd)
        (nsd / "betas_test.npz").write_bytes(b"not real data")
        (repo / "copy.bin").write_bytes(b"not real data")
        g("add", "-f", "nsd_data/betas_test.npz", "copy.bin")
        _, probs = CC.check(repo, nsd)
        found = any("under NSD_DIR" in p for p in probs) and any("identical" in p for p in probs) \
            and any("data extension" in p for p in probs)
        # hook: copy the checker into the temporary repository and point NSD_DIR at nsd_data
        (repo / "scripts").mkdir()
        src = (REPO / "scripts" / "nsd_commit_check.py").read_text(encoding="utf-8")
        (repo / "scripts" / "nsd_commit_check.py").write_text(src, encoding="utf-8")
        (repo / "src" / "ridge").mkdir(parents=True)
        (repo / "src" / "ridge" / "paths.py").write_text((REPO / "src" / "ridge" / "paths.py").read_text(encoding="utf-8"),
                                                         encoding="utf-8")
        subprocess.run([sys.executable, str(repo / "scripts" / "nsd_commit_check.py"), "--repo", str(repo),
                        "--install-hook"], capture_output=True, check=True)
        env = dict(os.environ, NSD_DIR=str(nsd))
        r = subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "test"], capture_output=True, text=True, env=env)
        refused = r.returncode != 0 and "STOP" in (r.stdout + r.stderr)
        no_commit = g("rev-parse", "--verify", "-q", "HEAD").returncode != 0
        check("(iv) commit check finds the file under NSD_DIR and the hook refuses the commit",
              not clean and found and refused and no_commit,
              f"clean case {len(clean)} problems; test case {len(probs)} problems; commit exit {r.returncode}")


def main():
    print("nsd pipeline checks")
    test_synthetic()
    test_commit_check()
    test_things_reproduction()
    test_components_reproduction()
    if "--skip-gate" not in sys.argv:
        import nsd_features as F
        ok = F.run_gate(extract=True)
        check("(i) extraction gate on 20 THINGS images", ok)
    if FAILS:
        print(f"  {len(FAILS)} failure(s) of {N} checks")
        sys.exit(1)
    print(f"  all {N} checks passed")


if __name__ == "__main__":
    main()

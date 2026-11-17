"""
test_nsd43.py
=============
Checks for step 43 (amendments/2026-11-17_step43-predictions.md, config_step43.yaml).
All must pass before the step 43 runs.

  (i)   the NSD multiverse path (nsd_step43.mv_models / mv_brains / mv_pairs / rho_pairs, with
        step7_multiverse.normalise and .betas) on THINGS reproduces results/step7/multiverse.csv:
        36 cells, V1 (Conv1) and LOC (Conv3), NATIVE 224, Random Weights and Backprop x 5 seeds,
        per subject, to 1e-9; step7_multiverse.roi_betas is restored after every call
  (ii)  the NSD ImageNet extraction (nsd_step43.imn_extract via step8_extract_pretrained) on the
        20 step 40 gate images at R = 224 through the downstream-equivalence gate against
        data/step8_gap: rows replaced, RSA per subject (raw and z units, correlation, V1 and LOC,
        all 6 models x 4 conditions); |d rho| <= 1e-4 and |d Delta| <= 1e-4
  (iii) the NSD resolution path (nsd_step43.res_extract via the v12 extraction, res_models,
        rho_pairs) re-extracts the 720 THINGS images at NATIVE 32 and 224 px and reproduces the
        rho of results/step5/resolution_rsa.csv (Conv1 -> V1, raw and z, Random Weights and
        Backprop x 5 seeds, per subject) to 1e-9
  (iv)  nsd_step43.loso_lower (nili_bounds of learning-rules-rsa) on THINGS V1, cross-run pairs,
        reproduces lower|cr|V1 of learning-rules-rsa results/crossrun/step2_summary.csv to 1e-9

Run: python tests/test_nsd43.py [--only i,ii,iii,iv]
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import nsd_common as C                                       # noqa: E402,F401
import nsd_step43 as NS                                      # noqa: E402
import paths                                                 # noqa: E402
from cvrsa import model_rdm                                  # noqa: E402

FAILS, N = [], 0
R_, BP = "Random Weights", "Backprop"


def check(name, ok, info=""):
    global N
    N += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)


def crossrun_ci():
    from common_up_pairs import pair_set, run_labels
    _, _, keep = pair_set(None, run_labels())
    return np.flatnonzero(keep)


def test_multiverse():
    import step7_multiverse as S7
    from step6_common import roi_betas
    ci = crossrun_ci()
    ref = pd.read_csv(REPO / "results" / "step7" / "multiverse.csv", low_memory=False)
    ref = ref[(ref.kind == "rho") & (ref.arm == "NATIVE") & (ref.res == 224)]
    ref = ref.set_index(["F", "D", "B", "roi", "rule", "seed_idx"])
    old = S7.roi_betas
    worst, n = 0.0, 0
    for roi, layer in S7.ROIS.items():
        feats = {(rule, si): S7.feat("NATIVE", 224, rule, si, layer) for rule in (R_, BP) for si in range(5)}
        mkeys, M = NS.mv_models(feats)
        for s in S7.SUBJ:
            bkeys, B = NS.mv_brains(roi_betas(s, roi))
            pairs, lab = NS.mv_pairs(mkeys, bkeys)
            rho = NS.rho_pairs(M, B, pairs, ci)
            for r, v in zip(lab.itertuples(), rho):
                worst = max(worst, abs(v - float(ref.loc[(r.F, r.D, r.B, roi, r.condition, float(r.seed)), f"rho_{s}"])))
                n += 1
    check("(i) multiverse path reproduces step 7 on THINGS", worst <= 1e-9 and n == 2 * 3 * 36 * 10
          and S7.roi_betas is old, f"{n} values, max |d rho| {worst:.1e}")


def test_imagenet():
    import step7_multiverse as S7
    import step40_extract
    from v12_lift import namespaces
    ns12, _ = namespaces()
    idx, names = step40_extract.gate_images()
    imgs = [ns12["find_img"](n) for n in names]
    assert all(p is not None for p in imgs)
    ci = crossrun_ci()
    s8 = REPO / "data" / "step8_gap"
    with tempfile.TemporaryDirectory() as td:
        NS.imn_extract(imgs, Path(td))
        fdev = 0.0
        new = {}
        for m in NS.IMN_MODELS:
            for c in NS.IMN_CONDS:
                for layer in ("V1", "LOC"):
                    X = NS.imn_feat(s8, m, c, layer)
                    rows = NS.imn_feat(Path(td), m, c, layer)
                    fdev = max(fdev, float(np.abs(X[idx] - rows).max()))
                    Xr = X.copy()
                    Xr[idx] = rows
                    new[(m, c, layer)] = Xr
    worst, worst_d, n = 0.0, 0.0, 0
    for roi in ("V1", "LOC"):
        for s in S7.SUBJ:
            b = pdist(S7.betas(s, roi, "paper"), "correlation")[None]
            for u in ("raw", "z"):
                rho = {}
                for tag, get in (("stored", lambda m, c: NS.imn_feat(s8, m, c, roi)), ("nsd", lambda m, c: new[(m, c, roi)])):
                    for m in NS.IMN_MODELS:
                        for c in NS.IMN_CONDS:
                            v = model_rdm(get(m, c), u, "correlation")[None]
                            rho[(tag, m, c)] = float(NS.rho_pairs(v, b, np.array([[0, 0]]), ci)[0])
                for m in NS.IMN_MODELS:
                    for c in NS.IMN_CONDS:
                        worst = max(worst, abs(rho[("nsd", m, c)] - rho[("stored", m, c)]))
                        n += 1
                    dl = {t: np.mean([rho[(t, m, c)] for c in NS.IMN_CONDS[1:]]) - rho[(t, m, "trained")] for t in ("stored", "nsd")}
                    worst_d = max(worst_d, abs(dl["nsd"] - dl["stored"]))
    check("(ii) ImageNet extraction passes the downstream-equivalence gate", worst <= 1e-4 and worst_d <= 1e-4 and n == 288,
          f"{n} rho, max |d rho| {worst:.1e}, max |d Delta| {worst_d:.1e}, max |d feature| {fdev:.1e}")


def test_resolution():
    from common_up_pairs import stim_order
    from v12_lift import namespaces, load_fmri_tri
    import step5_resolution_rsa as S5
    ns12, _ = namespaces()
    imgs = [ns12["find_img"](s) for s in stim_order()]
    assert all(p is not None for p in imgs)
    ci = crossrun_ci()
    ref = pd.read_csv(REPO / "results" / "step5" / "resolution_rsa.csv", low_memory=False)
    ref = ref[(ref.kind == "rho") & (ref.arm == "NATIVE") & (ref.layer == "Conv1") & (ref.roi == "V1")]
    ref = ref.set_index(["res", "rule", "seed_idx", "variant"])
    brains = load_fmri_tri("V1")
    subj = ["sub-01", "sub-02", "sub-03"]
    with tempfile.TemporaryDirectory() as td:
        NS.res_extract(imgs, Path(td), pxs=[32, 224])
        fdev = max(float(np.abs(np.load(NS.res_path(Path(td), px, rule, si))
                                - np.load(S5.gap_path("NATIVE", px, si, rule, "Conv1"))).max())
                   for px in (32, 224) for rule in (R_, BP) for si in range(5))
        mkeys, M = NS.res_models(lambda px, rule, si: np.load(NS.res_path(Path(td), px, rule, si)).astype(np.float64),
                                 pxs=[32, 224])
    worst, n = 0.0, 0
    pairs = np.array([(i, 0) for i in range(len(mkeys))])
    for s, b in zip(subj, brains):
        rho = NS.rho_pairs(M, b[None], pairs, ci)
        for (px, lab, si, u), v in zip(mkeys, rho):
            worst = max(worst, abs(v - float(ref.loc[(float(px), NS.RES_RULES[lab], float(si), u), f"rho_{s}"])))
            n += 1
    check("(iii) resolution path reproduces step 5 on THINGS (32 and 224 px)", worst <= 1e-9 and n == 120,
          f"{n} values, max |d rho| {worst:.1e}, max |d feature| {fdev:.1e}")


def test_loso():
    from v12_lift import load_fmri_tri
    ci = crossrun_ci()
    ref = pd.read_csv(paths.crossrun_repo() / "results" / "crossrun" / "step2_summary.csv").set_index("key")
    target = float(ref.loc["lower|cr|V1", "point"])
    v = NS.loso_lower([b[ci] for b in load_fmri_tri("V1")])
    check("(iv) LOSO lower bound reproduces THINGS V1 (cross-run)", abs(v - target) <= 1e-9,
          f"{v:.6f} vs {target:.6f}, |d| {abs(v - target):.1e}")


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    print("step 43 checks")
    for tag, fn in (("iv", test_loso), ("i", test_multiverse), ("iii", test_resolution), ("ii", test_imagenet)):
        if only is None or tag in only:
            fn()
    if FAILS:
        print(f"  {len(FAILS)} failure(s) of {N} checks")
        sys.exit(1)
    print(f"  all {N} checks passed")


if __name__ == "__main__":
    main()

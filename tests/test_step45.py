"""
test_step45.py
==============
Checks for step 45 (amendments/2026-11-21_step45-predictions.md, config_step45.yaml).
All must pass before the step 45 run.

  (i)   gate: only_0-14 images (bandpass_filters.bands / condition, vendored verbatim from
        evaluation-resolution-rsa code/bandpass/bandpass_extract.py; sha256 of the source checked)
        of the 20 step 40 gate images, through v12 extract_features, reproduce the stored step 6
        GAP features (STEP5_GAP_DIR/bandpass/only_0-14), the evaluation-resolution-rsa RDM entries
        of the 190 pairs (RIDGE_DATA_DIR/data/bandpass/rdms/only_0-14, Conv1, where stored) and
        rho against THINGS-720 V1 with the 20 rows replaced (cross-run, per subject); all to 1e-4
  (ii)  LP-RDM without filter and without downsampling equals the Euclidean distance of the grey
        images (20 images, 1e-9 relative)
  (iii) the step 45 partial path with L and C alone reproduces step 44 (+0.0163) at THINGS-720 V1
        to 1e-12; partial_many equals step44_why.partial_rho

Run: python tests/test_step45.py [--only i,ii,iii]
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import nsd_common as C                                       # noqa: E402,F401
import paths                                                 # noqa: E402
import step44_why as W                                       # noqa: E402
import step45_lowpass as LP                                  # noqa: E402

FAILS, N = [], 0
SHA = "b35e679965ff4bac1fc942debbb5aedd318ee5bb5aa4cc873cbefac32788ded6"


def check(name, ok, info=""):
    global N
    N += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)


def test_gate():
    import torch
    import torchvision.transforms as T
    from PIL import Image
    import nsd_step43 as NS
    import step40_extract
    from bandpass_filters import PX, bands, condition
    from common_up_pairs import stim_order
    from cvrsa import condensed_index, model_rdm
    from v12_lift import RULES, SEEDS, rule_key, namespaces, transform, load_v12_model
    src = W.ERR / "code" / "bandpass" / "bandpass_extract.py"
    sha_ok = hashlib.sha256(src.read_bytes()).hexdigest() == SHA
    torch.set_num_threads(12)
    ns12, ns11 = namespaces()
    idx, names = step40_extract.gate_images()
    idx = np.asarray(idx)
    order_ok = [stim_order()[i] for i in idx] == list(names)
    imgs = [Image.open(ns12["find_img"](n)).convert("RGB") for n in names]
    pre = T.Compose([T.Resize(PX), T.CenterCrop(PX), T.ToTensor()])
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    X0 = np.stack([pre(im).numpy() for im in imgs]).astype(np.float32)
    tf_ok = bool(torch.equal(norm(torch.from_numpy(X0[0])), transform(ns12, ns11, "NATIVE", PX)(imgs[0])))
    X = norm(torch.from_numpy(np.stack([condition(x, bands(x), "only_0-14") for x in X0])))
    paper = paths.ridge_data_dir() / "data" / "bandpass" / "rdms" / "only_0-14"
    gap = paths.step5_gap_dir() / "bandpass" / "only_0-14"
    ci = W.crossrun_index()
    B = np.stack(W.things720_brains("V1"))
    a, b = np.triu_indices(20, 1)
    sel = condensed_index(idx[a], idx[b], W.N720)
    fdev = rdev = rhodev = 0.0
    n_rdm = 0
    for rule in RULES:
        for si in range(len(SEEDS)):
            torch.manual_seed(SEEDS[si])
            m = load_v12_model(ns12, rule, si)
            f = np.asarray(ns12["extract_features"](m, list(range(20)), X)[0], np.float64)
            stored = np.load(gap / f"seed_{si}" / f"gap_{rule_key(rule)}_Conv1.npy").astype(np.float64)
            fdev = max(fdev, float(np.abs(stored[idx] - f).max()))
            pr = paper / f"seed_{si}" / f"rdm_{rule.lower().replace(' ', '_')}_Conv1.npy"
            if pr.exists():
                rdev = max(rdev, float(np.abs(np.load(pr)[sel] - ns12["compute_rdm"](f)[a, b]).max()))
                n_rdm += 1
            Xr = stored.copy()
            Xr[idx] = f
            pairs = np.array([[0, k] for k in range(len(B))])
            r0 = NS.rho_pairs(model_rdm(stored, "raw", "correlation")[None], B, pairs, ci)
            r1 = NS.rho_pairs(model_rdm(Xr, "raw", "correlation")[None], B, pairs, ci)
            rhodev = max(rhodev, float(np.abs(r0 - r1).max()))
    check("(i) only_0-14 gate: features, paper RDM entries, rho with replaced rows",
          sha_ok and order_ok and tf_ok and n_rdm == 10 and max(fdev, rdev, rhodev) <= 1e-4,
          f"source sha256 {'ok' if sha_ok else 'differs'}; features {fdev:.1e}, RDM entries ({n_rdm} RDMs x 190) {rdev:.1e}, "
          f"rho {rhodev:.1e}")


def test_identity():
    imgs = W.things720_images()[:20]
    v = LP.lp_rdm(imgs, filt=False, down=False)
    ref = pdist(np.stack([LP.grey(p).astype(np.float64).ravel() for p in imgs]), "euclidean")
    d = float(np.abs(v - ref).max() / np.abs(ref).max())
    v2 = np.stack([LP.lp_vector(LP.grey(p)) for p in imgs[:2]])
    check("(ii) LP-RDM without filter and downsampling = Euclidean distance of the grey images",
          d <= 1e-9 and v2.shape == (2, LP.GRID ** 2), f"max rel |d| {d:.1e}; LP vector length {v2.shape[1]}")


def test_step44():
    imgs = W.things720_images()
    L, Cn = W.lum_contrast(imgs)
    ci = W.crossrun_index()
    cs = [W.abs_vec(L)[ci], W.abs_vec(Cn)[ci]]
    models = {k: v[ci] for k, v in W.things720_models("V1").items()}
    brains = [b[ci] for b in W.things720_brains("V1")]
    rho = LP.partial_many(models, brains, cs)
    d = float(LP.delta_from(rho).mean())
    p44 = pd.read_csv(REPO / "results" / "step44" / "partial.csv")
    ref = float(p44[(p44.kind == "delta") & (p44.roi == "V1") & (p44.covariates == "L+C")].delta.iloc[0])
    one = W.partial_rho(models[("Random", 0)], brains[0], cs)
    check("(iii) L + C partial reproduces step 44", abs(d - ref) <= 1e-12 and abs(one - rho[("Random", 0)][0]) <= 1e-15,
          f"{d:+.6f} vs {ref:+.6f}, |d| {abs(d - ref):.1e}")


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    print("step 45 checks")
    for tag, fn in (("ii", test_identity), ("iii", test_step44), ("i", test_gate)):
        if only is None or tag in only:
            fn()
    if FAILS:
        print(f"  {len(FAILS)} failure(s) of {N} checks")
        sys.exit(1)
    print(f"  all {N} checks passed")


if __name__ == "__main__":
    main()

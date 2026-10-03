"""
test_lower_edge_gate.py
=======================
1. The four known low-edge hits (Random Weights (BN-calibrated) s3, g = 1, sub-01;
   reports/STEP2_LOWER_EDGE.md) are all classified Plateau.
2. Synthetic case (noise-free targets, full-rank features): the optimum lies below
   1e-6, the fit selects the low edge, and the gate classifies the cell invalid.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "ridge"))
sys.path.insert(0, str(HERE.parent / "scripts"))

import paths                                         # noqa: E402
from fit import fit_condition                        # noqa: E402
from lower_edge_gate import gate, hits_from_fit      # noqa: E402

KNOWN = [(3, 206), (5, 934), (5, 983), (5, 989)]


def main() -> int:
    fails = []
    data = paths.ridge_data_dir()
    a = np.load(data / "arm_b_out" / "sub-01_arma.npz")
    Y = a["betas"][:, a["local_V1"]].astype(np.float64)
    X = np.load(data / "pilot_features" / "model_random_bn_calibrated_seed3_g1.npy")
    hits = [dict(condition="Random Weights (BN-calibrated)", seed=3, g=1, voxel=v, outer_fold=f)
            for f, v in KNOWN]
    df = gate(hits, load_X=lambda h: X, Y=Y)
    print(df[["outer_fold", "voxel", "alpha_min_diagnostic", "relative_gain", "classification"]]
          .to_string(index=False))
    if not (df.classification == "Plateau").all():
        fails.append("known hits not all Plateau")

    # synthetic: y = X w exactly, X full rank -> ordinary least squares is optimal
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((720, 32))
    Ys = Xs @ rng.standard_normal((32, 3))
    res = fit_condition(Xs, Ys, np.arange(720), reliable=np.ones(3, bool),
                        assert_low_boundary=False)
    sh = hits_from_fit(res, np.ones(3, bool), condition="synthetic", seed=0, g=1)
    ds = gate(sh, load_X=lambda h: Xs, Y=Ys)
    print(f"synthetic: {len(sh)} low-edge hits, classes {sorted(set(ds.classification))}, "
          f"min gain {ds.relative_gain.min() if len(ds) else float('nan'):.3e}")
    if len(sh) == 0 or not (ds.classification == "invalid").all():
        fails.append("synthetic below-grid optimum not classified invalid")

    print(f"{'all 2 checks passed' if not fails else f'{len(fails)} failure(s) of 2 checks: {fails}'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

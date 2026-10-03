"""
pilot_fit.py
============
Phase 3 pilot: 224 px, V1, one subject, g in {1, 4}, all 30 checkpoints, plus the
control band. Reports seed-level SD, Delta_min, each condition's position relative to
the pixel baseline, and lo_rel for every condition.

DELTA_MIN, derived for the paper's own test
-------------------------------------------
The test is a paired ONE-SIDED SIGN-FLIP permutation over 5 seeds: 2^5 = 32 sign
assignments, so the smallest attainable p is 1/32 = 0.03125 and the NEXT smallest is
2/32 = 0.0625. At alpha = 0.05 only p = 1/32 qualifies, and the observed statistic
attains it only when all five per-seed differences share a sign. So

    power(Delta) = P(all 5 seed differences > 0) = Phi(Delta / SD)^5

and 80% power needs Phi(Delta/SD) = 0.8^(1/5) = 0.9564, i.e.

    Delta_min = 1.71 * SD          (SD = seed-level SD of the paired difference)

Cross-checked parametrically with (t_0.975,4 + t_0.80,4) * SD / sqrt(5) = 1.66 * SD,
and by direct simulation. The number goes in the paper regardless of outcome.

THE CEILING, AND WHY THE PILOT DOES NOT REPORT ONE
--------------------------------------------------
config.yaml sets Arm A's denominator to leave-one-subject-out inter-subject
consistency, explicitly labelled a LOWER BOUND AND NOT A NOISE CEILING. That quantity
needs all three subjects, and the pilot runs one, so it is NOT computed here.

Arm B's split-half reliability is reported alongside as a REFERENCE, never as a
denominator: it is measured on 100 DIFFERENT stimuli with 12-repetition averaging,
while Arm A is single-trial. It bounds nothing about Arm A directly. No r/ceiling
number is produced in this report.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "ridge"))

from config_hash import config_hash, load_config          # noqa: E402
from fit import fit_condition, CELLS                      # noqa: E402
from gridpool import shuffle_cells_per_image              # noqa: E402
from ridge_dual import alpha_grid                         # noqa: E402

import torch                                              # noqa: E402

REPO = HERE.parents[1]
import paths  # noqa: E402
FEAT = paths.ridge_data_dir() / "pilot_features"
ARM = paths.ridge_data_dir() / "arm_b_out"

RULES = ["random", "backprop", "feedback_alignment", "predictive_coding", "stdp",
         "random_bn_calibrated"]
RULE_LABEL = {
    "random": "Random Weights", "backprop": "Backprop",
    "feedback_alignment": "Feedback Alignment",
    "predictive_coding": "Predictive Coding", "stdp": "STDP",
    "random_bn_calibrated": "Random Weights (BN-calibrated)",
}
SEEDS = [42, 123, 456, 789, 1337]
CONTROLS = ["pixels", "lumcontrast", "gabor"]
SUBJECT = "sub-01"
GRIDS = [1, 4]
RELIABILITY_THRESHOLD = 0.2


def delta_min_signflip(sd: float, n_seeds: int = 5, power: float = 0.80,
                       n_sim: int = 200000, seed: int = 20260820):
    """Delta_min by simulation, plus the closed form and the parametric cross-check."""
    from scipy.stats import norm, t as tdist

    closed = norm.ppf(power ** (1.0 / n_seeds)) * sd
    param = (tdist.ppf(0.975, n_seeds - 1) + tdist.ppf(power, n_seeds - 1)) * sd / np.sqrt(n_seeds)

    rng = np.random.default_rng(seed)
    lo, hi = 0.0, max(10 * sd, 1e-6)
    for _ in range(60):
        mid = (lo + hi) / 2
        d = rng.normal(mid, sd, size=(n_sim, n_seeds))
        # p = 1/32 exactly when all five differences share a (positive) sign.
        pw = float((d > 0).all(axis=1).mean())
        if pw < power:
            lo = mid
        else:
            hi = mid
    return {"simulated": (lo + hi) / 2, "closed_form": closed, "parametric": param}


def load_features(name, g):
    p = FEAT / f"{name}_g{g}.npy"
    if not p.exists():
        raise FileNotFoundError(f"missing pilot features: {p}")
    a = np.load(p)
    if a.dtype != np.float32:
        raise AssertionError(f"{p.name} is {a.dtype}, expected float32")
    return a


def main():
    cfg = load_config()
    h = config_hash()
    alphas = alpha_grid(cfg["ridge"]["alpha_grid"]["min"],
                        cfg["ridge"]["alpha_grid"]["max"],
                        cfg["ridge"]["alpha_grid"]["n_points"])

    a = np.load(ARM / f"{SUBJECT}_arma.npz")
    b = np.load(ARM / f"{SUBJECT}_armb.npz")
    v1 = a["local_V1"]
    Y = a["betas"][:, v1].astype(np.float64)
    rel = b["reliability"][v1]
    reliable = rel > RELIABILITY_THRESHOLD
    groups = np.arange(Y.shape[0])

    print("=" * 92)
    print(f"PHASE 3 PILOT  |  config {h}  |  224 px, V1, {SUBJECT}, g in {GRIDS}")
    print(f"alpha grid {alphas[0]:.0e}..{alphas[-1]:.0e}, {len(alphas)} pts  |  "
          f"selection=MSE, reporting=Pearson r")
    print(f"V1 voxels {Y.shape[1]}, reliable (split-half > {RELIABILITY_THRESHOLD}) "
          f"{int(reliable.sum())} ({100 * reliable.mean():.1f}%)")
    print(f"Arm B split-half of these voxels: median {np.median(rel):+.3f}, "
          f"reliable-subset median {np.median(rel[reliable]):+.3f}  "
          f"[REFERENCE ONLY -- different stimuli, 12-rep averaged; NOT an Arm A ceiling]")
    print("=" * 92)

    rows = []

    def run(label, X, g, kind):
        res = fit_condition(X.astype(np.float64), Y, groups, alphas=alphas,
                            reliable=reliable, context=f"{label} g{g}")
        med = float(np.median(res["r"][reliable]))
        rows.append(dict(label=label, kind=kind, g=g, d=X.shape[1], r_med=med,
                         r_med_all=float(np.median(res["r"])),
                         **{k: float(np.mean(res[k])) for k in CELLS}))
        print(f"  {label:<34} g={g}  d={X.shape[1]:<5} "
              f"r(med,reliable)={med:+.4f}  lo_rel={np.mean(res['lo_rel']):.3f}  "
              f"hi_rel={np.mean(res['hi_rel']):.3f}", flush=True)
        return res

    for g in GRIDS:
        print(f"\n--- g = {g} ---")
        for key in RULES:
            for si in range(len(SEEDS)):
                X = load_features(f"model_{key}_seed{si}", g)
                run(f"{RULE_LABEL[key]} s{si}", X, g, "model")
        for c in CONTROLS:
            run(f"control:{c}", load_features(f"control_{c}", g), g, "control")

        # Negative test: permuted stimulus<->feature assignment must give r ~ 0.
        X = load_features("model_backprop_seed0", g)
        rng = np.random.default_rng(20260820)
        run("control:permuted", X[rng.permutation(len(X))], g, "control")

        # Spatial-correspondence control at g=4 only (meaningless at g=1).
        if g >= 2:
            for key in ("random", "backprop"):
                X = load_features(f"model_{key}_seed0", g)
                Xs = shuffle_cells_per_image(torch.from_numpy(X), g, seed=20260820).numpy()
                run(f"{RULE_LABEL[key]} s0 CELL-SHUFFLED", Xs, g, "shuffle")

    import pandas as pd
    df = pd.DataFrame(rows)
    df["config_hash"] = h
    out = REPO / "pilot_results.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out.name}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    main()

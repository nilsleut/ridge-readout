"""
step4_std_rsa.py
================
S1 (amendments/2026-10-03_step4-predictions.md). EXPLORATORY, sub-01 only.

RSA on transformed features: sub-01, V1, NATIVE 224 px, Conv1, g = 1, all 30
checkpoints from pilot_features/, 501 reliable voxels. Pair set (210,205 cross-run
pairs) and statistic exactly as step 2C: tri_rdm (scripts/step2_rsa_501.py) and
ranks / pair_set / run_labels (src/vendor/common_up_pairs.py), called unchanged;
rho = ranks(model[P]) @ ranks(fmri[P]).

Feature transformation before pdist(..., "correlation"), per channel over the 720
stimuli (population SD, as ridge_dual.standardize):
  raw     unchanged (must reproduce results/step2/rsa_501.csv to 1e-12, else STOP)
  z       (x - mean) / SD          <- primary
  center  x - mean
  scale   x / SD
Dead channels (SD < 1e-8, the ridge_dual.standardize eps) are set to 0, counted and
reported; no STOP.

Descriptive (no rule): coefficient of variation of the 32 channel SDs per checkpoint.

S1, fixed before the run: D_std = Random - BP under z, mean over 5 seeds.
  D_std <= 0.010 -> H_scale supported; D_std >= 0.020 -> not supported; else ambiguous.
  Control: |Random - BN-cal| under z <= 0.010; if not met, offsets before the ReLU
  contribute.

Output: results/step4/std_rsa.csv, results/step4/channel_scale.csv
"""
import step3_common as C          # noqa: I001

import numpy as np
import pandas as pd

from common_up_pairs import pair_set, ranks, run_labels
from step2_rsa_501 import tri_rdm

VARIANTS = ["raw", "z", "center", "scale"]
EPS = 1e-8
R, BP, BN = "Random Weights", "Backprop", "Random Weights (BN-calibrated)"
DIFFS = [(R, BP), (BN, BP), (R, BN), ("Feedback Alignment", BP), ("Predictive Coding", BP),
         ("STDP", BP)]
OUT = C.REPO / "results" / "step4"


def transform(X, variant):
    """(transformed X, number of dead channels)."""
    X = np.asarray(X, dtype=np.float64)
    if variant == "raw":
        return X, 0
    mu, sd = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    dead = sd[0] < EPS
    sd = np.where(dead[None, :], 1.0, sd)
    out = {"z": (X - mu) / sd, "center": X - mu, "scale": X / sd}[variant]
    out[:, dead] = 0.0
    return out, int(dead.sum())


def rsa_rows(Y, reliable, keep, subject, load):
    """One row per (rule, seed_idx, variant). load(key, si) -> g=1 features."""
    fr = ranks(tri_rdm(Y[:, reliable])[keep])
    rows = []
    for key, label in C.RULES.items():
        for si in C.SEED_IDX:
            X = load(key, si)
            for v in VARIANTS:
                Xt, nd = transform(X, v)
                m = tri_rdm(Xt)
                nn = int(np.isnan(m).sum())
                rows.append(dict(subject=subject, rule=label, seed_idx=si, variant=v,
                                 dead_channels=nd, nan_pairs=nn,
                                 rho=float(ranks(m[keep]) @ fr) if nn == 0 else np.nan))
    return pd.DataFrame(rows)


def summarise(df, by=("variant",)):
    out = []
    for keys, grp in df.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        tag = dict(zip(by, keys))
        p = grp.pivot(index="seed_idx", columns="rule", values="rho")
        for rule in p.columns:
            out.append(dict(kind="condition", **tag, quantity=rule, mean=p[rule].mean(),
                            sd=p[rule].std(ddof=1), n_seeds=int(p[rule].notna().sum())))
        for a, b in DIFFS:
            d = (p[a] - p[b]).dropna()
            out.append(dict(kind="difference", **tag, quantity=f"{a} - {b}", mean=d.mean(),
                            sd=d.std(ddof=1), n_seeds=len(d), seeds_positive=int((d > 0).sum())))
    return pd.DataFrame(out)


def main():
    _, _, keep = pair_set(None, run_labels())
    assert int(keep.sum()) == 210205
    Y, rel = C.brain()
    assert int(rel.sum()) == 501
    per = rsa_rows(Y, rel, keep, C.SUBJECT, lambda k, s: C.load_X(k, s, 1))

    stored = pd.read_csv(C.REPO / "results" / "step2" / "rsa_501.csv")
    stored = stored[stored.voxel_set == "reliable"].set_index(["rule", "seed_idx"]).rho
    raw = per[per.variant == "raw"].set_index(["rule", "seed_idx"]).rho
    dev = float((raw - stored.reindex(raw.index)).abs().max())
    print(f"raw vs rsa_501.csv: max |d| {dev:.2e}")
    if dev > 1e-12:
        raise SystemExit("STOP: raw variant does not reproduce rsa_501.csv")
    if per.nan_pairs.sum():
        print(f"NOTE: {int(per.nan_pairs.sum())} NaN pairs")

    scale = []
    for key, label in C.RULES.items():
        for si in C.SEED_IDX:
            sd = C.load_X(key, si, 1).astype(np.float64).std(axis=0)
            scale.append(dict(subject=C.SUBJECT, rule=label, seed_idx=si,
                              cv_channel_sd=float(sd.std() / sd.mean()), sd_min=float(sd.min()),
                              sd_max=float(sd.max()), dead_channels=int((sd < EPS).sum())))
    scale = pd.DataFrame(scale)
    sc_sum = scale.groupby("rule")[["cv_channel_sd", "sd_min", "sd_max"]].mean().reset_index()
    sc_sum["kind"] = "condition_mean"
    scale["kind"] = "checkpoint"

    summ = summarise(per)
    per["kind"] = "checkpoint"
    OUT.mkdir(parents=True, exist_ok=True)
    C.stamp(pd.concat([per, summ], ignore_index=True)).to_csv(OUT / "std_rsa.csv", index=False)
    C.stamp(pd.concat([scale, sc_sum], ignore_index=True)).to_csv(OUT / "channel_scale.csv", index=False)

    pd.set_option("display.width", 220)
    print(summ.to_string(index=False))
    print("\ndead channels per variant:", per.groupby("variant").dead_channels.sum().to_dict())
    print(sc_sum.to_string(index=False))
    d = summ[(summ.kind == "difference") & (summ.variant == "z")].set_index("quantity")["mean"]
    ds, dc = float(d[f"{R} - {BP}"]), float(d[f"{R} - {BN}"])
    verdict = ("H_scale supported" if ds <= 0.010 else "H_scale not supported" if ds >= 0.020
               else "ambiguous")
    print(f"\nS1: D_std = {ds:+.4f} -> {verdict}; control |Random - BN-cal| = {abs(dc):.4f} -> "
          f"{'met' if abs(dc) <= 0.010 else 'not met (offsets before the ReLU contribute)'}")


if __name__ == "__main__":
    main()

"""
step3_e4_encoding_rsa.py
========================
E4: encoding-RSA at g = 1. EXPLORATORY, sub-01 only.

For both fold variants (random = default folds; run = run-grouped folds of E3) and
all 30 checkpoints: refit at g = 1 (fit_condition, D1 counting mode, lower-edge gate
afterwards) and take the OOF predictions of the 501 reliable V1 voxels.

  primary: per outer fold, subtract per voxel the mean of that fold's predictions
           (removes intercept differences between folds)
  info:    uncentred

Model RDM = pdist(pred, "correlation") over the 720 stimuli. fMRI RDM (501 reliable
voxels), pair set (210,205 cross-run pairs) and statistic exactly as step 2C: the
functions tri_rdm (scripts/step2_rsa_501.py) and ranks / pair_set / run_labels
(src/vendor/common_up_pairs.py) are called unchanged; rho = ranks(m[P]) @ ranks(f[P]).
NaN pairs (constant prediction vector of a stimulus) are counted; any -> STOP.
Wiring check: the raw-feature RSA of every checkpoint recomputed here must equal
results/step2/rsa_501.csv (reliable) to 1e-12.

Rule, fixed before the run, with D_raw = +0.0278 (RSA, 501 voxels, g = 1) and D_enc
(centred, per fold variant):
  D_enc <= 0                                 -> the flip arises in the readout
  D_enc > 0 and |D_enc - D_raw| <= 0.010     -> encoding-RSA follows the RSA; the flip
                                                lies in the score (RSA vs voxelwise
                                                predictability)
  otherwise                                  -> ambiguous

Output: results/step3/encoding_rsa.csv, results/step3/lower_edge_gate_e4.csv
"""
import step3_common as C          # noqa: I001

import numpy as np
import pandas as pd

from common_up_pairs import pair_set, ranks, run_labels
from step2_rsa_501 import tri_rdm

R, BP = "Random Weights", "Backprop"
D_RAW = 0.0278


def centre_per_fold(pred, fold_of):
    out = pred.copy()
    for f in np.unique(fold_of):
        out[fold_of == f] -= pred[fold_of == f].mean(axis=0, keepdims=True)
    return out


def main():
    _, _, keep = pair_set(None, run_labels())
    assert int(keep.sum()) == 210205
    Y, rel = C.brain()
    fr = ranks(tri_rdm(Y[:, rel])[keep])

    stored = pd.read_csv(C.REPO / "results" / "step2" / "rsa_501.csv")
    stored = stored[stored.voxel_set == "reliable"].set_index(["rule", "seed_idx"]).rho
    dev = max(abs(float(ranks(tri_rdm(C.load_X(C.LABEL_KEY[r], s, 1))[keep]) @ fr) - v)
              for (r, s), v in stored.items())
    print(f"wiring check vs rsa_501.csv: max |d| {dev:.2e}")
    if dev > 1e-12:
        raise SystemExit("STOP: wiring check failed")

    run_f = C.run_folds()
    res = C.pmap(C.fit_one, [(k, s, 1, sch) for sch in ("random", "run")
                             for k in C.RULES for s in C.SEED_IDX])
    gate = C.stamp(C.run_gate(res, {"run": run_f}))
    C.OUT.mkdir(parents=True, exist_ok=True)
    gate.to_csv(C.OUT / "lower_edge_gate_e4.csv", index=False)
    invalid = {(r.condition, r.seed, r.fold_scheme) for r in gate.itertuples() if r.classification == "invalid"}

    rows, n_nan = [], 0
    for r in res:
        for centring in ("centred", "uncentred"):
            pred = centre_per_fold(r["pred_rel"], r["fold_of"]) if centring == "centred" else r["pred_rel"]
            m = tri_rdm(pred)
            nn = int(np.isnan(m).sum())
            n_nan += nn
            rho = float(ranks(m[keep]) @ fr) if nn == 0 else np.nan
            rows.append(dict(kind="checkpoint", rule=r["rule"], seed_idx=r["seed_idx"], grid=1,
                             fold_scheme=r["fold_scheme"], centring=centring, n_voxels=int(rel.sum()),
                             n_pairs=int(keep.sum()), nan_pairs=nn, rho=rho,
                             valid=(r["rule"], r["seed_idx"], r["fold_scheme"]) not in invalid))
    if n_nan:
        pd.DataFrame(rows).pipe(C.stamp).to_csv(C.OUT / "encoding_rsa.csv", index=False)
        raise SystemExit(f"STOP: {n_nan} NaN pairs in model RDMs")
    per = pd.DataFrame(rows)

    summ, verdicts = [], {}
    v = per[per.valid]
    for (sch, cen), grp in v.groupby(["fold_scheme", "centring"]):
        p = grp.pivot(index="seed_idx", columns="rule", values="rho")
        for rule in p.columns:
            summ.append(dict(kind="condition", fold_scheme=sch, centring=cen, rule=rule,
                             rho=p[rule].mean(), sd=p[rule].std(ddof=1), n_seeds=int(p[rule].notna().sum())))
        d = (p[R] - p[BP]).dropna()
        summ.append(dict(kind="difference", fold_scheme=sch, centring=cen, rule=f"{R} - {BP}",
                         rho=d.mean(), sd=d.std(ddof=1), n_seeds=len(d), seeds_positive=int((d > 0).sum())))
        if cen == "centred":
            de = d.mean()
            verdicts[sch] = ("flip arises in the readout" if de <= 0 else
                             "encoding-RSA follows the RSA; flip lies in the score"
                             if abs(de - D_RAW) <= 0.010 else "ambiguous")
    s = pd.DataFrame(summ)
    C.stamp(pd.concat([per, s], ignore_index=True)).to_csv(C.OUT / "encoding_rsa.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    print(f"\nNaN pairs: {n_nan}; low-edge hits: {len(gate)}; classes: "
          f"{gate.classification.value_counts().to_dict() if len(gate) else {}}; invalid: {sorted(invalid)}")
    for sch, vd in verdicts.items():
        de = s[(s.kind == "difference") & (s.fold_scheme == sch) & (s.centring == "centred")].rho.iloc[0]
        print(f"rule [{sch} folds]: D_enc {de:+.4f}, D_raw {D_RAW:+.4f}, |diff| {abs(de - D_RAW):.4f} -> {vd}")


if __name__ == "__main__":
    main()

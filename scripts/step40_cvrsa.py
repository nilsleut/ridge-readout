"""
step40_cvrsa.py
===============
Step 40, part C (amendments/2026-11-16_step40-predictions.md): RSA on the 100 repeated
test images with cross-validated brain RDMs. EXPLORATORY.

Brain: single presentations from step40_common.load_reps (12 sessions per image), V1 and
LOC (all voxels of the ROI), per subject; RDMs b1, b2, b3 from src/ridge/cvrsa.py.
Models: global average pooled Conv1 features (g = 1 rung of extract_pilot, 224 px NATIVE)
of Random Weights and Backprop, seeds 0-4 (RIDGE_DATA_DIR/step40_features/test100, made by
step40_extract.py test); raw or standardised over the 100 stimuli; correlation or Euclidean.
Cells: correlation vs b1, Euclidean vs b2, Euclidean vs b3, each for raw and z units.
Spearman over all 4950 pairs per subject; mean over subjects and seeds; Random - BP with a
95% stimulus-bootstrap interval (1000 resamples, seed 20261002, self-pairs dropped, the same
draws for every comparison).
C1: raw, correlation vs b1: Random - BP > 0. C2: z, correlation vs b1: Random - BP <= 0.
C3: Euclidean: sign of Random - BP the same against b2 and b3 (raw and z each).
All three on point estimates; intervals reported.

Output: results/step40/cvrsa.csv, results/step40/cvrsa_summary.csv
"""
import step40_common as S     # noqa: I001

import time

import numpy as np
import pandas as pd

from cvrsa import boot_draws, boot_spearman, brain_rdms, model_rdm

SUBJ = ["sub-01", "sub-02", "sub-03"]
CELLS = [("correlation", "b1"), ("euclidean", "b2"), ("euclidean", "b3")]


def main():
    c = S.cfg()["cvrsa"]
    t0 = time.time()
    draws = boot_draws(c["n_test_images"], c["bootstrap"]["n_resamples"], c["bootstrap"]["seed"])
    brains, counts = {}, []
    for s in SUBJ:
        reps, rois = S.load_reps(s)
        counts.append(dict(subject=s, images=reps.shape[0], presentations_per_image=reps.shape[1]))
        for roi in c["rois"]:
            brains[(s, roi)] = brain_rdms(reps[:, :, rois[roi]])
    fdir = S.feature_dir() / "test100"
    rows, G = [], {}
    for key, lab in S.KEYS.items():
        for si in S.cfg()["seed_idx"]:
            X = np.load(fdir / f"model_{key}_seed{si}_g1.npy")
            if X.shape[0] != c["n_test_images"]:
                raise SystemExit(f"STOP: {key} seed {si} has {X.shape[0]} rows")
            for norm in c["model_normalisation"]:
                for dist, b in CELLS:
                    m = model_rdm(X, norm, dist)
                    for s in SUBJ:
                        for roi in c["rois"]:
                            vals = boot_spearman(m, brains[(s, roi)][b], draws)
                            G[(lab, si, norm, dist, b, s, roi)] = vals
                            rows.append(dict(kind="rho", subject=s, roi=roi, condition=lab, seed=si,
                                             model_units=norm, model_distance=dist, brain_rdm=b, rho=vals[0]))
    print(f"comparisons done ({time.time() - t0:.0f} s)", flush=True)
    out = []
    for roi in c["rois"]:
        for norm in c["model_normalisation"]:
            for dist, b in CELLS:
                def mean_draws(lab):
                    return np.mean([G[(lab, si, norm, dist, b, s, roi)] for si in S.cfg()["seed_idx"]
                                    for s in SUBJ], axis=0)
                diff = mean_draws("Random") - mean_draws("BP")
                out.append(dict(kind="contrast", roi=roi, model_units=norm, model_distance=dist, brain_rdm=b,
                                rho_random=float(mean_draws("Random")[0]), rho_bp=float(mean_draws("BP")[0]),
                                value=float(diff[0]), ci_lo=float(np.quantile(diff[1:], .025)),
                                ci_hi=float(np.quantile(diff[1:], .975))))
    res = pd.DataFrame(out)

    def get(roi, norm, dist, b):
        q = res[(res.roi == roi) & (res.model_units == norm) & (res.model_distance == dist) & (res.brain_rdm == b)]
        assert len(q) == 1
        return float(q.value.iloc[0])

    rules = []
    for roi in c["rois"]:
        note = "V1 (primary)" if roi == "V1" else "LOC (secondary, descriptive)"
        c1 = get(roi, "raw", "correlation", "b1")
        c2 = get(roi, "z", "correlation", "b1")
        rules.append(dict(kind="rule", roi=roi, quantity="C1 raw, correlation vs b1: Random - BP > 0",
                          value=c1, passed=bool(c1 > 0), note=note))
        rules.append(dict(kind="rule", roi=roi, quantity="C2 z, correlation vs b1: Random - BP <= 0",
                          value=c2, passed=bool(c2 <= 0), note=note))
        for norm in ("raw", "z"):
            e2, e3 = get(roi, norm, "euclidean", "b2"), get(roi, norm, "euclidean", "b3")
            rules.append(dict(kind="rule", roi=roi, quantity=f"C3 {norm}, Euclidean: same sign vs b2 and b3",
                              value=e3, passed=bool(np.sign(e2) == np.sign(e3)),
                              note=f"{note}; b2 {e2:+.4f}, b3 {e3:+.4f}"))
    summ = pd.concat([res, pd.DataFrame(rules), pd.DataFrame(counts).assign(kind="presentations")],
                     ignore_index=True)
    S.OUT.mkdir(parents=True, exist_ok=True)
    rho = pd.DataFrame(rows)
    rho["seed"] = rho["seed"].astype("Int64")
    S.stamp(rho).to_csv(S.OUT / "cvrsa.csv", index=False)
    summ["seed"] = pd.array([pd.NA] * len(summ), dtype="Int64")
    S.stamp(summ).to_csv(S.OUT / "cvrsa_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.drop(columns=["config_hash", "git_head"]).to_string(index=False))


if __name__ == "__main__":
    main()

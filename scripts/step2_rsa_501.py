"""
step2_rsa_501.py
================
EXPLORATORY, sub-01 only. Does the voxel set separate the sign change between RSA
(paper: Random - Backprop > 0 at V1, 224 px) and the ridge readout (< 0 on the 501
reliable V1 voxels)?

Setup: sub-01, V1, NATIVE 224 px, Conv1, g = 1 (GAP), all 30 checkpoints from
pilot_features/. RSA exactly as the paper (v3): model RDM = pdist(features,
"correlation") as v12 compute_rdm; Spearman over the cross-run pair set P as
ranks(model[P]) @ ranks(fmri[P]) (common_up / step2_rsa.py, per-subject convention).

Reproduction checks, criteria fixed before the run (any failure stops the script):
  (i)   fMRI RDM from all 1049 V1 voxels vs OUTPUTS_720_DIR/fmri_rdm_V1_sub-01.npy:
        Pearson of the triangles >= 0.999
  (ii)  model RDMs from pilot_features (g = 1) vs the stored v12 NATIVE 224 Conv1 RDM
        triangles (data/upsampling/rdms in RIDGE_DATA_DIR): Pearson >= 0.999, all 30
  (iii) per-subject V1 RSA for sub-01 at NATIVE 224, cross-run (results/upsampling/
        rsa_seed{n}.csv, column rho_sub-01): |mine on 1049 voxels - stored| <= 1e-4

Interpretation rule, fixed before the run (D_all on 1049 voxels, D_rel on 501):
  D_rel > 0 and |D_rel - D_all| <= 0.010 -> the voxel set is NOT the cause
  D_rel <= 0                              -> reliability selection alone reproduces
                                             the direction of the ridge result
  otherwise                               -> ambiguous

Output: results/step2/rsa_501.csv (+ rsa_501_checks.csv), with config_hash and git_head.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src" / "ridge"))
sys.path.insert(0, str(REPO / "src" / "vendor"))

import paths                                                      # noqa: E402
from config_hash import config_hash                               # noqa: E402
from common_up_pairs import TRI, ranks, run_labels, pair_set      # noqa: E402

SUBJECT = "sub-01"
RES = 224
RELIABILITY_THRESHOLD = 0.2
RULES = {  # pilot_features key -> v12 label
    "random": "Random Weights", "backprop": "Backprop",
    "feedback_alignment": "Feedback Alignment", "predictive_coding": "Predictive Coding",
    "stdp": "STDP", "random_bn_calibrated": "Random Weights (BN-calibrated)",
}
SEEDS = [42, 123, 456, 789, 1337]
RANDOM_PAPER = "Random Weights"   # reports/STEP2_RANDOM_VARIANT.md, question 2
OUT = REPO / "results" / "step2"


def git_head():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def tri_rdm(X):
    """Upper triangle (TRI order) of pdist(X, 'correlation'); pdist's condensed order
    is the row-major strict upper triangle, i.e. TRI order."""
    return pdist(np.asarray(X, dtype=np.float64), metric="correlation")


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def stop(msg):
    raise SystemExit(f"STOP: {msg}")


def main():
    h, gh = config_hash(), git_head()
    data = paths.ridge_data_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []

    # ── pair set ────────────────────────────────────────────────────────────
    runs = run_labels()
    _, _, keep = pair_set(None, runs)
    n_all, n_cr = len(TRI[0]), int(keep.sum())
    _, _, keep_s1 = pair_set(None, {SUBJECT: runs[SUBJECT]})
    print(f"pairs: all {n_all}, cross-run (every subject) {n_cr}, "
          f"cross-run (sub-01 runs only) {int(keep_s1.sum())}")
    checks.append(dict(check="pairs_all", value=n_all, criterion="== 258840", passed=n_all == 258840))
    checks.append(dict(check="pairs_crossrun", value=n_cr, criterion="== 210205", passed=n_cr == 210205))
    checks.append(dict(check="pairs_crossrun_sub01_only", value=int(keep_s1.sum()),
                       criterion="reported only", passed=True))
    if n_all != 258840 or n_cr != 210205:
        stop(f"pair counts {n_all}/{n_cr}")

    # ── brain side ──────────────────────────────────────────────────────────
    a = np.load(data / "arm_b_out" / f"{SUBJECT}_arma.npz")
    b = np.load(data / "arm_b_out" / f"{SUBJECT}_armb.npz")
    v1 = a["local_V1"]
    betas = a["betas"]
    rel = b["reliability"][v1]
    sets = {"all": v1, "reliable": v1[rel > RELIABILITY_THRESHOLD],
            "unreliable": v1[~(rel > RELIABILITY_THRESHOLD)]}
    print({k: len(v) for k, v in sets.items()})
    if [len(sets[k]) for k in ("all", "reliable", "unreliable")] != [1049, 501, 548]:
        stop("voxel-set sizes differ from 1049 / 501 / 548")
    fmri = {k: tri_rdm(betas[:, idx]) for k, idx in sets.items()}

    ref = np.load(paths.outputs_720_dir() / f"fmri_rdm_V1_{SUBJECT}.npy")[TRI].astype(np.float64)
    r_i = pearson(fmri["all"], ref)
    checks.append(dict(check="(i) fmri_rdm_V1_sub-01 vs own 1049-voxel RDM", value=r_i,
                       criterion=">= 0.999", passed=r_i >= 0.999,
                       max_abs_diff=float(np.abs(fmri["all"] - ref).max())))
    print(f"(i)  Pearson {r_i:.8f}, max|d| {np.abs(fmri['all'] - ref).max():.2e}")
    if r_i < 0.999:
        stop("check (i)")
    fr = {k: ranks(v[keep]) for k, v in fmri.items()}

    # ── model side ──────────────────────────────────────────────────────────
    stored_rdm = data / "data" / "upsampling" / "rdms" / "NATIVE" / f"res{RES}"
    model, r_ii = {}, []
    for key, label in RULES.items():
        for si in range(len(SEEDS)):
            X = np.load(data / "pilot_features" / f"model_{key}_seed{si}_g1.npy")
            m = tri_rdm(X)
            s = np.load(stored_rdm / f"seed_{si}" /
                        f"rdm_{label.lower().replace(' ', '_')}_Conv1.npy").astype(np.float64)
            rr = pearson(m, s)
            r_ii.append(rr)
            checks.append(dict(check=f"(ii) model RDM {label} seed_idx {si}", value=rr,
                               criterion=">= 0.999", passed=rr >= 0.999,
                               max_abs_diff=float(np.abs(m - s).max())))
            model[(label, si)] = (m, s)
    print(f"(ii) Pearson min {min(r_ii):.8f} over {len(r_ii)}")
    if min(r_ii) < 0.999:
        stop("check (ii)")

    # ── (iii) stored per-subject values ─────────────────────────────────────
    stored = pd.concat([pd.read_csv(data / "results" / "upsampling" / f"rsa_seed{si}.csv")
                        for si in range(len(SEEDS))])
    stored = stored[(stored.arm == "NATIVE") & (stored.res == RES) & (stored.layer == "Conv1")
                    & (stored.roi == "V1") & (stored.variant == "cr")
                    & (stored.convention == "persub")]
    if len(stored) != 30:
        stop(f"expected 30 stored rows, found {len(stored)}")
    dev_iii, dev_pipe = [], []
    for _, row in stored.iterrows():
        m, s = model[(row.rule, row.seed_idx)]
        mine = float(ranks(m[keep]) @ fr["all"])
        # Diagnostic only: stored model RDM against the stored fMRI RDM.
        pipe = float(ranks(s[keep]) @ ranks(ref[keep]))
        dev_iii.append(abs(mine - row["rho_sub-01"]))
        dev_pipe.append(abs(pipe - row["rho_sub-01"]))
    checks.append(dict(check="(iii) RSA 1049 voxels vs stored rho_sub-01 (max |d|)",
                       value=max(dev_iii), criterion="<= 1e-4", passed=max(dev_iii) <= 1e-4))
    checks.append(dict(check="(iii-diag) stored RDMs through this code vs stored rho_sub-01",
                       value=max(dev_pipe), criterion="reported only", passed=True))
    print(f"(iii) max|d| {max(dev_iii):.2e}  (stored RDMs through this code: {max(dev_pipe):.2e})")
    ck = pd.DataFrame(checks)
    ck["config_hash"], ck["git_head"] = h, gh
    ck.to_csv(OUT / "rsa_501_checks.csv", index=False)
    if max(dev_iii) > 1e-4:
        stop("check (iii)")

    # ── RSA on three voxel sets ─────────────────────────────────────────────
    rows = []
    for (label, si), (m, _) in model.items():
        mr = ranks(m[keep])
        for vs, f in fr.items():
            rows.append(dict(subject=SUBJECT, roi="V1", layer="Conv1", grid=1, arm="NATIVE",
                             resolution=RES, pair_set="crossrun", n_pairs=n_cr,
                             voxel_set=vs, n_voxels=len(sets[vs]), rule=label,
                             seed=SEEDS[si], seed_idx=si, rho=float(mr @ f)))
    df = pd.DataFrame(rows)
    df["config_hash"], df["git_head"] = h, gh
    df.to_csv(OUT / "rsa_501.csv", index=False)

    print("\nvoxel set   n     BP mean   RAND mean  RAND-BP (SD)        RANDBN-BP (SD)")
    summ = {}
    for vs in sets:
        p = df[df.voxel_set == vs].pivot(index="seed_idx", columns="rule", values="rho")
        d = p[RANDOM_PAPER] - p["Backprop"]
        dc = p["Random Weights (BN-calibrated)"] - p["Backprop"]
        summ[vs] = d.mean()
        print(f"{vs:<11} {len(sets[vs]):<5} {p['Backprop'].mean():+.4f}   "
              f"{p[RANDOM_PAPER].mean():+.4f}    {d.mean():+.4f} ({d.std(ddof=1):.4f})   "
              f"{dc.mean():+.4f} ({dc.std(ddof=1):.4f})   per seed {np.round(d.values, 4)}")
    d_all, d_rel = summ["all"], summ["reliable"]
    if d_rel > 0 and abs(d_rel - d_all) <= 0.010:
        verdict = "voxel set is NOT the cause of the sign change"
    elif d_rel <= 0:
        verdict = "reliability selection alone reproduces the direction of the ridge result"
    else:
        verdict = "ambiguous"
    print(f"\nD_all {d_all:+.4f}  D_rel {d_rel:+.4f}  |D_rel-D_all| {abs(d_rel - d_all):.4f}"
          f"  -> {verdict}")


if __name__ == "__main__":
    main()

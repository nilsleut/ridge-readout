"""
step45_lowpass.py
=================
Step 45 (amendments/2026-11-21_step45-predictions.md, config_step45.yaml): coarse spatial luminance.

  LP-RDM  Rec. 709 grey image (load_rgb, 224) -> only_0-14 band of arXiv:2608.12408 (incl. mean)
          -> area-downsampled to 28 x 28 -> Euclidean distance
  W6      THINGS-720 V1: Delta after partialling L, C and LP together <= +0.010
  W7      rho(V1, LP) / LOSO bound: THINGS-720 > test images (k = 1) and > NSD (k = 1)

Imported unchanged:
  step44_why          partial-Spearman definitions (rank_resid, pearson of compute_lowlevel.py),
                      pair sets (crossrun_index, nsd_k1_index), datasets (things720_*, test_set),
                      L and C (lum_contrast), positions (T), delta_on, abs_vec, REC709, stamp logic
  bandpass_filters    bands, condition (vendored verbatim from bandpass_extract.py)
  colour_statistic_control  load_rgb (via step44_why.colour_module)

Per-image LP vectors are not written; NSD values are aggregated only.

Usage:
  python scripts/step45_lowpass.py run
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads, sets the import paths)

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from config_hash import config_hash, load_config
import step44_why as W

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config_step45.yaml"
OUT = REPO / "results" / "step45"
GRID = 28


def cfg():
    return load_config(CONFIG)


def stamp(df):
    df["config_hash"] = config_hash(cfg())
    df["config_hash_44"] = config_hash(W.cfg())
    df["git_head"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
    if "seed" not in df:
        df["seed"] = np.nan
    df["seed"] = df["seed"].astype("Int64")
    return df


# ── LP covariate ────────────────────────────────────────────────────────────
def grey(path):
    CS = W.colour_module()
    return (CS.load_rgb(path, CS.REF_PX) @ W.REC709).astype(np.float32)          # (224, 224)


def lp_vector(g, filt=True, down=True):
    """only_0-14 band (bandpass_filters.condition on a (1, H, W) array), then area-downsampling."""
    import torch
    import torch.nn.functional as F
    from bandpass_filters import bands, condition
    x = g[None]
    if filt:
        x = condition(x, bands(x), "only_0-14")
    if down:
        x = F.interpolate(torch.from_numpy(np.ascontiguousarray(x, np.float32))[None], size=(GRID, GRID),
                          mode="area")[0].numpy()
    return np.asarray(x, np.float64).ravel()


def lp_rdm(img_paths, filt=True, down=True):
    V = np.stack([lp_vector(grey(p), filt, down) for p in img_paths])
    return pdist(V, "euclidean")


# ── partial Spearman with shared model residuals ────────────────────────────
def partial_many(models, brains, covs):
    """{key: rho per brain} for every model; the step 44 definitions (rank_resid, pearson), model
    residuals computed once for all brains on the same pair set."""
    if not W._LL:
        W._LL.update(W.lowlevel_defs())
    rr, pe = W._LL["rank_resid"], W._LL["pearson"]
    rb = [rr(b, covs) for b in brains]
    return {k: np.array([float(pe(rr(m, covs), r)) for r in rb]) for k, m in models.items()}


def delta_from(rho):
    """rho {(lab, seed): per-subject array} -> per-subject Delta (mean over seeds)."""
    seeds = sorted({k[1] for k in rho})
    return np.mean([rho[("Random", s)] - rho[("BP", s)] for s in seeds], axis=0)


def covsets(names, cov):
    out = {"none": []}
    for n in names:
        if n != "none":
            out[n] = [cov[c] for c in n.split("+")]
    return out


# ── datasets ────────────────────────────────────────────────────────────────
def things720_part(c, rows):
    imgs = W.things720_images()
    L, Cn = W.lum_contrast(imgs)
    LP = lp_rdm(imgs)
    pos = W.positions()
    ci = W.crossrun_index()
    base = {"L": W.abs_vec(L)[ci], "C": W.abs_vec(Cn)[ci], "LP": LP[ci]}
    out = {}
    for roi in ("V1", "LOC"):
        models = {k: v[ci] for k, v in W.things720_models(roi).items()}
        brains = [b[ci] for b in W.things720_brains(roi)]
        for name in c["partial_sets"]["things720"]:
            per = []
            for s_i, s in enumerate(W.SUBJ):           # T is subject-specific
                cov = dict(base, T=W.abs_vec(pos[s])[ci])
                cs = covsets([name], cov)[name]
                per.append({k: v[0] for k, v in partial_many(models, [brains[s_i]], cs).items()})
            rho = {k: np.array([p[k] for p in per]) for k in models}
            d = delta_from(rho)
            out[(roi, name)] = float(d.mean())
            rows += [dict(kind="subject_delta", dataset="things720", roi=roi, covariates=name, subject=s, delta=float(v))
                     for s, v in zip(W.SUBJ, d)]
            rows.append(dict(kind="delta", dataset="things720", roi=roi, covariates=name, delta=float(d.mean()),
                             rho_random=float(np.mean([rho[("Random", s)].mean() for s in W.seeds()])),
                             rho_bp=float(np.mean([rho[("BP", s)].mean() for s in W.seeds()])),
                             seeds_positive=int(sum((rho[("Random", s)] - rho[("BP", s)]).mean() > 0 for s in W.seeds()))))
        lp_rows("things720", roi, models, brains, base["LP"], rows)
    return out


def lp_rows(dataset, roi, models, brains, lp, rows, extra=None):
    """rho(model, LP) per seed and rho(brain, LP) per subject on the given pair set."""
    for (lab, si), m in models.items():
        rows.append(dict(kind="model_lp", dataset=dataset, roi=roi, condition=lab, seed=si, rho=W.spear(m, lp)))
    for j, b in enumerate(brains):
        rows.append(dict(kind="brain_lp_subject", dataset=dataset, roi=roi, subject_idx=j, rho=W.spear(b, lp), **(extra or {})))


def test_part(c, rows):
    imgs, tbrains, tmodels, n_draws = W.test_set()
    L, Cn = W.lum_contrast(imgs)
    cov = {"L": W.abs_vec(L), "C": W.abs_vec(Cn), "LP": lp_rdm(imgs)}
    for roi in ("V1", "LOC"):
        for name, cs in covsets(c["partial_sets"]["things_test"], cov).items():
            ds = []
            for d in range(n_draws):
                ds.append(delta_from(partial_many(tmodels, [tbrains[(s, roi)][d] for s in W.SUBJ], cs)))
            d = np.mean(ds, axis=0)
            rows.append(dict(kind="delta", dataset="things_test", roi=roi, covariates=name, delta=float(d.mean()),
                             **{f"delta_{s}": float(v) for s, v in zip(W.SUBJ, d)}))
        for (lab, si), m in tmodels.items():
            rows.append(dict(kind="model_lp", dataset="things_test", roi=roi, condition=lab, seed=si, rho=W.spear(m, cov["LP"])))
        for j, s in enumerate(W.SUBJ):
            rows.append(dict(kind="brain_lp_subject", dataset="things_test", roi=roi, subject_idx=j,
                             rho=float(np.mean([W.spear(tbrains[(s, roi)][d], cov["LP"]) for d in range(n_draws)]))))


def nsd_part(c, rows):
    import nsd_features as F
    from cvrsa import model_rdm
    from nsd_rsa import load_features
    imgs = F.image_paths()
    L, Cn = W.lum_contrast(imgs)
    full = {"L": W.abs_vec(L), "C": W.abs_vec(Cn), "LP": lp_rdm(imgs)}
    nmodels = {k: model_rdm(np.asarray(X, np.float64), "raw", "correlation") for k, X in load_features().items()}
    subs, z, k1idx = W.nsd_k1_index()
    for roi in ("V1", "LO-obj"):
        for name in c["partial_sets"]["nsd"]:
            per_r = []
            for r, idx in enumerate(k1idx):
                cs = covsets([name], {k: v[idx] for k, v in full.items()})[name]
                brains = [z[s][f"{roi}|correlation|k1_{r}"][idx] for s in subs]
                per_r.append(delta_from(partial_many({k: v[idx] for k, v in nmodels.items()}, brains, cs)))
            d = np.mean(per_r, axis=0)
            rows.append(dict(kind="delta", dataset="nsd", roi=roi, covariates=name, delta=float(d.mean()),
                             n_negative=int((d < 0).sum()), n_subjects=len(subs)))
        for (lab, si), m in nmodels.items():
            rows.append(dict(kind="model_lp", dataset="nsd", roi=roi, condition=lab, seed=si, rho=W.spear(m, full["LP"])))
        for j, s in enumerate(subs):
            rows.append(dict(kind="brain_lp_subject", dataset="nsd", roi=roi, subject_idx=j,
                             rho=float(np.mean([W.spear(z[s][f"{roi}|correlation|k1_{r}"][idx], full["LP"][idx])
                                                for r, idx in enumerate(k1idx)]))))


def run():
    c = cfg()
    C.require_free(c["disk"]["min_free_gb"])
    OUT.mkdir(parents=True, exist_ok=True)
    t0, times, rows = time.time(), {}, []
    t = time.time()
    d720 = things720_part(c, rows)
    times["things720"] = time.time() - t
    t = time.time()
    test_part(c, rows)
    times["things_test"] = time.time() - t
    t = time.time()
    nsd_part(c, rows)
    times["nsd"] = time.time() - t
    df = pd.DataFrame(rows)

    # check: L + C at THINGS-720 reproduces step 44
    p44 = pd.read_csv(REPO / "results" / "step44" / "partial.csv")
    ref = float(p44[(p44.kind == "delta") & (p44.roi == "V1") & (p44.covariates == "L+C")].delta.iloc[0])
    if abs(d720[("V1", "L+C")] - ref) > 1e-12:
        raise SystemExit(f"STOP: L + C partial {d720[('V1', 'L+C')]} does not reproduce step 44 {ref}")
    n41 = pd.read_csv(REPO / "results" / "step41" / "noise_curve.csv", low_memory=False)
    r41 = float(n41[(n41.kind == "contrast") & (n41.roi == "V1") & (n41.k == 1) & (n41.model_units == "raw")
                    & (n41.model_distance == "correlation") & (n41.brain_rdm == "b1")].delta.iloc[0])
    t_none = float(df[(df.kind == "delta") & (df.dataset == "things_test") & (df.roi == "V1")
                      & (df.covariates == "none")].delta.iloc[0])
    if abs(t_none - r41) > 1e-9:
        raise SystemExit(f"STOP: test-image Delta without covariates {t_none} does not reproduce step 41 {r41}")
    part = df[df.kind.isin(["delta", "subject_delta"])].copy()
    part = pd.concat([part, pd.DataFrame([dict(kind="check", quantity="THINGS-720 V1 L+C vs step 44",
                                               delta=d720[("V1", "L+C")], ref=ref),
                                          dict(kind="check", quantity="test images V1 no covariate vs step 41",
                                               delta=t_none, ref=r41)])], ignore_index=True)
    stamp(part).to_csv(OUT / "partial_lp.csv", index=False)

    # LP by dataset: brain / bound ratio and model rho
    l44 = pd.read_csv(REPO / "results" / "step44" / "luminance_by_dataset.csv")
    bound = l44[l44.kind == "brain_lum"].set_index(["dataset", "roi"]).loso_lower
    lp = []
    for (ds, roi), g in df[df.kind == "brain_lp_subject"].groupby(["dataset", "roi"], sort=False):
        m = df[(df.kind == "model_lp") & (df.dataset == ds) & (df.roi == roi)]
        mr = m.pivot_table(index="seed", columns="condition", values="rho")
        lp.append(dict(kind="lp", dataset=ds, roi=roi, rho_brain_lp=float(g.rho.mean()), sd=float(g.rho.std(ddof=1)),
                       n_subjects=len(g), loso_lower=float(bound.loc[(ds, roi)]),
                       ratio=float(g.rho.mean()) / float(bound.loc[(ds, roi)]),
                       rho_random_lp=float(mr["Random"].mean()), rho_bp_lp=float(mr["BP"].mean()),
                       seeds_random_larger=int((mr["Random"] > mr["BP"]).sum())))
    lpd = pd.concat([pd.DataFrame(lp), df[df.kind.isin(["model_lp", "brain_lp_subject"])]], ignore_index=True)
    stamp(lpd).to_csv(OUT / "lp_by_dataset.csv", index=False)
    times["total"] = time.time() - t0

    # summary
    s44 = pd.read_csv(REPO / "results" / "step44" / "summary.csv").set_index("rule")
    w3, w4 = bool(s44.loc["W3", "passed"]), bool(s44.loc["W4", "passed"])
    lpi = pd.DataFrame(lp).set_index(["dataset", "roi"]).ratio
    rat = {d: float(lpi.loc[(d, "V1")]) for d in ("things720", "things_test", "nsd")}
    w6v = d720[("V1", "L+C+LP")]
    met = {"W6": bool(w6v <= 0.010), "W7": bool(rat["things720"] > rat["things_test"] and rat["things720"] > rat["nsd"])}
    frac = 1 - w6v / d720[("V1", "none")]
    if met["W6"] and met["W7"] and w3 and w4:
        interp = ("coarse luminance: the THINGS-720 advantage is attributed to coarse luminance structure dominating its "
                  "V1 RDM, combined with the untrained network's greater sensitivity to it")
    else:
        interp = f"not identified; explained fraction {frac:.3f}"
    summ = [dict(rule="W6", quantity="THINGS-720 V1 Delta after partialling L, C and LP (<= +0.010)", value=w6v,
                 passed=met["W6"], note=f"Delta without {d720[('V1', 'none')]:+.4f}; explained fraction {frac:.3f}"),
            dict(rule="W7", quantity="rho(V1, LP) / LOSO bound: THINGS-720 > test (k = 1) and > NSD (k = 1)",
                 value=rat["things720"], passed=met["W7"], note="; ".join(f"{k} {v:.3f}" for k, v in rat.items())),
            dict(rule="W3 (step 44)", quantity="from results/step44/summary.csv", passed=w3),
            dict(rule="W4 (step 44)", quantity="from results/step44/summary.csv", passed=w4),
            dict(rule="interpretation", quantity="amendment rule", value=frac, note=interp)]
    summ += [dict(rule="runtime", quantity=f"{k} seconds", value=v) for k, v in times.items()]
    sm = pd.DataFrame(summ)
    stamp(sm).to_csv(OUT / "summary.csv", index=False)
    print(sm[["rule", "quantity", "value", "passed", "note"]].to_string(index=False))
    return sm


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "") == "run":
        run()
    else:
        raise SystemExit("usage: step45_lowpass.py run")

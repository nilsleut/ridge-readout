"""
step44_why.py
=============
Step 44 (amendments/2026-11-19_step44-predictions.md, config_step44.yaml): why does the THINGS-720
set show an untrained advantage?

  W1  1000 random subsets of 100 of the 720 images: Delta > -0.0002 in >= 95 % of subsets
  W2  THINGS-720: partialling L-RDM and C-RDM reduces Delta to <= +0.010
  W3  Spearman(model RDM, L-RDM) Random > BP in 5/5 seeds in THINGS-720, THINGS test, NSD
  W4  rho(V1, L-RDM) / LOSO lower bound: THINGS-720 > THINGS test (k = 1) and > NSD (k = 1)
  W5  THINGS-720: partialling T-RDM changes Delta by < 0.005

Existing functions are imported, not re-implemented:
  evaluation-resolution-rsa code/colour_statistic_control.py   load_rgb, build_references (MEANLUM), abs_rdm
  evaluation-resolution-rsa code/paper_v3/compute_lowlevel.py  rank_resid, pearson (lifted by AST:
                                                               the module imports the full v3 pipeline)
  common_up_pairs   stim_order, run_labels, pair_set (cross-run pairs, also for subsets)
  v12_lift          namespaces (find_img), load_fmri_tri
  step7_multiverse  feat (GAP features, NATIVE 224, Conv1 and Conv3)
  step40_common     load_reps, test_stimuli, feature_dir (test images, step 40 features)
  step41_noise_curve  session_draws, brain_for (k = 1 brains of the test images)
  nsd_rsa           load_features (step 42 features); nsd_step43: rho_pairs, rdm_cache, loso_lower
  nsd_features      image_paths; cvrsa: model_rdm, condensed_index

Per-image L and C are not written; NSD quantities are aggregated only.

Usage:
  python scripts/step44_why.py estimate     runtime estimate (STOP above 4 h)
  python scripts/step44_why.py run          W1-W5 -> results/step44
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads, sets the import paths)

import ast
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import paths
from config_hash import config_hash, load_config
from cvrsa import condensed_index, model_rdm

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config_step44.yaml"
OUT = REPO / "results" / "step44"
ERR = paths.repo_root().parent / "evaluation-resolution-rsa"
SUBJ = ["sub-01", "sub-02", "sub-03"]
RULES = {"Random": "Random Weights", "BP": "Backprop"}
LAYER = {"V1": "Conv1", "LOC": "Conv3"}
N720 = 720
REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)   # colour_statistic_control.build_references


def cfg():
    return load_config(CONFIG)


def seeds():
    return list(cfg()["rsa"]["seeds"])


def stamp(df):
    df["config_hash"] = config_hash(cfg())
    df["git_head"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
    if "seed" not in df:
        df["seed"] = np.nan
    df["seed"] = df["seed"].astype("Int64")
    return df


# ── imported definitions from evaluation-resolution-rsa ────────────────────
def colour_module():
    sys.path.insert(0, str(ERR / "code"))
    import colour_statistic_control as CS
    return CS


def lowlevel_defs(names=("rank_resid", "pearson")):
    """Top-level functions of compute_lowlevel.py, compiled from its source unchanged (the module
    itself imports compute_numbers, which loads the whole v3 pipeline)."""
    path = ERR / "code" / "paper_v3" / "compute_lowlevel.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in body} != set(names):
        raise SystemExit(f"STOP: {sorted(set(names) - {n.name for n in body})} not found in {path}")
    ns = {"np": np, "rankdata": rankdata}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


_LL = {}


def partial_rho(m, b, covs):
    """Partial Spearman of the amendment: ranks over the pair set, model and brain ranks regressed
    on the covariate ranks (OLS with intercept), Pearson of the residuals."""
    if not _LL:
        _LL.update(lowlevel_defs())
    return float(_LL["pearson"](_LL["rank_resid"](m, covs), _LL["rank_resid"](b, covs)))


def spear(a, b):
    return float(spearmanr(a, b).correlation)


# ── covariates ──────────────────────────────────────────────────────────────
def lum_contrast(img_paths):
    """(L, C) per image: L = MEANLUM of build_references; C = SD of the Rec. 709 grey values of the
    same load_rgb image."""
    CS = colour_module()
    _, _, lum = CS.build_references(img_paths)
    con = np.array([float((CS.load_rgb(p, CS.REF_PX) @ REC709).std()) for p in img_paths])
    return np.asarray(lum, np.float64), con


def abs_vec(v):
    i, j = np.triu_indices(len(v), 1)
    return np.abs(v[i] - v[j]).astype(np.float64)


def positions():
    """{subject: position of each of the 720 images within its session, in trials (metadata row
    order within the session)}; checks one session per subject."""
    from common_up_pairs import stim_order
    order = stim_order()
    out = {}
    for s in SUBJ:
        st = pd.read_csv(paths.things_fmri_dir() / f"{s}_task-things_stimulus-metadata.csv")
        if not (st.trial_id.values == np.arange(len(st))).all():
            raise SystemExit(f"STOP: {s} trial_id is not the row index")
        st["pos"] = st.groupby("session").cumcount()
        m = st.set_index("stimulus").loc[order]
        if m.session.nunique() != 1:
            raise SystemExit(f"STOP: {s}: the 720 images are not in one session")
        out[s] = m.pos.to_numpy().astype(np.float64)
    return out


# ── datasets ────────────────────────────────────────────────────────────────
def things720_images():
    from common_up_pairs import stim_order
    from v12_lift import namespaces
    ns12, _ = namespaces()
    imgs = [ns12["find_img"](s) for s in stim_order()]
    if any(p is None for p in imgs):
        raise SystemExit("STOP: THINGS-720 image missing")
    return imgs


def crossrun_index(stims=None):
    """Condensed 720-index of the cross-run pairs of a stimulus sample (pair_set)."""
    from common_up_pairs import pair_set, run_labels
    a, b, _ = pair_set(stims, run_labels())
    return condensed_index(a, b, N720)


def things720_models(roi):
    import step7_multiverse as S7
    return {(lab, si): model_rdm(S7.feat("NATIVE", 224, rule, si, LAYER[roi]), "raw", "correlation")
            for lab, rule in RULES.items() for si in seeds()}


def things720_brains(roi):
    from v12_lift import load_fmri_tri
    return load_fmri_tri(roi)


def delta_on(models, brains, ci):
    """Paper-cell Delta over pair index ci: per subject Random - BP (mean over seeds), mean over
    subjects. Returns (Delta, per-subject Delta)."""
    import nsd_step43 as NS
    keys = list(models)
    M = np.stack([models[k] for k in keys])
    B = np.stack(brains)
    pairs = np.array([(i, j) for j in range(len(brains)) for i in range(len(keys))])
    rho = NS.rho_pairs(M, B, pairs, ci).reshape(len(brains), len(keys))
    w = np.array([1 / 5 if k[0] == "Random" else -1 / 5 for k in keys])
    per = rho @ w
    return float(per.mean()), per


def test_set():
    import step40_common as S40
    import step41_noise_curve as S41
    from v12_lift import namespaces
    ns12, _ = namespaces()
    names = S40.test_stimuli()
    imgs = [ns12["find_img"](n) for n in names]
    if any(p is None for p in imgs):
        raise SystemExit("STOP: test image missing")
    draws = S41.session_draws()[1]
    brains = {}
    for s in SUBJ:
        reps, rois = S40.load_reps(s)
        for roi in ("V1", "LOC"):
            R = reps[:, :, rois[roi]]
            brains[(s, roi)] = [S41.brain_for(R, d)["b1"] for d in draws]
    models = {}
    for key, lab in {"random": "Random", "backprop": "BP"}.items():
        for si in seeds():
            X = np.load(S40.feature_dir() / "test100" / f"model_{key}_seed{si}_g1.npy")
            models[(lab, si)] = model_rdm(X, "raw", "correlation")
    return imgs, brains, models, len(draws)


def nsd_k1_index():
    """Per presentation r: condensed 515-index of the pairs where no subject has the two
    presentations in the same session and run (the step 43 NC rule)."""
    import nsd_step43 as NS
    subs = cfg()["datasets"]["nsd"]["subjects"]
    z = {s: NS.rdm_cache(s) for s in subs}
    n = NS.N_IMG
    a, b = np.triu_indices(n, 1)
    out = []
    for r in range(3):
        ok = np.ones(len(a), bool)
        for s in subs:
            key = z[s]["session"][:, r].astype(np.int64) * 100 + z[s]["run"][:, r]
            ok &= key[a] != key[b]
        out.append(np.flatnonzero(ok))
    return subs, z, out


# ── parts ───────────────────────────────────────────────────────────────────
def run_w1(c, models, brains, rois=("V1", "LOC")):
    w = c["W1"]
    rng = np.random.default_rng(w["seed"])
    subsets = [np.sort(rng.choice(N720, w["subset_size"], replace=False)) for _ in range(w["n_subsets"])]
    rows = []
    for k, stims in enumerate(subsets):
        ci = crossrun_index(stims)
        for roi in rois:
            d, per = delta_on(models[roi], brains[roi], ci)
            rows.append(dict(kind="subset", subset=k, roi=roi, n_pairs=len(ci), delta=d,
                             **{f"delta_{s}": float(v) for s, v in zip(SUBJ, per)}))
    df = pd.DataFrame(rows)
    summ = []
    for roi in rois:
        x = df[df.roi == roi].delta.to_numpy()
        summ.append(dict(kind="summary", roi=roi, quantity="fraction of subsets with Delta > threshold",
                         threshold=w["threshold"], value=float((x > w["threshold"]).mean()),
                         mean=float(x.mean()), sd=float(x.std(ddof=1)), q025=float(np.quantile(x, .025)),
                         q05=float(np.quantile(x, .05)), q50=float(np.median(x)), q975=float(np.quantile(x, .975))))
    return pd.concat([df, pd.DataFrame(summ)], ignore_index=True)


def run_partial(models, brains, ci, cov):
    """THINGS-720 per subject, ROI and covariate set: rho per model, Delta."""
    sets = {"none": [], "L+C": ["L", "C"], "T": ["T"], "L+C+T": ["L", "C", "T"]}
    rows = []
    for roi in models:
        for name, cs in sets.items():
            for s_i, s in enumerate(SUBJ):
                Z = [cov[(c_, s)][ci] for c_ in cs]
                b = brains[roi][s_i][ci]
                for (lab, si), m in models[roi].items():
                    rows.append(dict(kind="rho", roi=roi, covariates=name, subject=s, condition=lab, seed=si,
                                     rho=partial_rho(m[ci], b, Z)))
    df = pd.DataFrame(rows)
    g = df.groupby(["roi", "covariates", "subject", "condition"]).rho.mean().unstack()
    per = (g["Random"] - g["BP"]).rename("delta").reset_index()
    per["kind"] = "subject_delta"
    per[["rho_random", "rho_bp"]] = g[["Random", "BP"]].to_numpy()
    mean = per.groupby(["roi", "covariates"])[["delta", "rho_random", "rho_bp"]].mean().reset_index()
    mean["kind"] = "delta"
    seed_d = df.pivot_table(index=["roi", "covariates", "seed"], columns="condition", values="rho").reset_index()
    seed_d["delta"] = seed_d["Random"] - seed_d["BP"]
    mean["seeds_positive"] = [int((seed_d[(seed_d.roi == r.roi) & (seed_d.covariates == r.covariates)].delta > 0).sum())
                              for r in mean.itertuples()]
    return pd.concat([df, per, mean], ignore_index=True)


def model_lum(models, L, idx):
    """Spearman(model RDM, L-RDM) per model over the pair index idx."""
    return {k: spear(m[idx], L[idx]) for k, m in models.items()}


def lum_rows(dataset, roi, layer, ml, brain_l, bound, extra=None):
    rows = [dict(kind="model_lum", dataset=dataset, roi=roi, layer=layer, condition=lab, seed=si, rho=v)
            for (lab, si), v in ml.items()]
    win = sum(ml[("Random", s)] > ml[("BP", s)] for s in seeds())
    rows.append(dict(kind="model_lum_summary", dataset=dataset, roi=roi, layer=layer,
                     rho_random=float(np.mean([ml[("Random", s)] for s in seeds()])),
                     rho_bp=float(np.mean([ml[("BP", s)] for s in seeds()])), seeds_random_larger=int(win)))
    rows.append(dict(kind="brain_lum", dataset=dataset, roi=roi, rho=float(np.mean(brain_l)), sd=float(np.std(brain_l, ddof=1)),
                     n_subjects=len(brain_l), loso_lower=bound, ratio=float(np.mean(brain_l)) / bound, **(extra or {})))
    return rows


def run_order(pos, cov, models, brains, ci, L):
    rows = []
    for s_i, s in enumerate(SUBJ):
        rows.append(dict(kind="position_vs_L", subject=s, rho=spear(pos[s], L), n=N720))
        T = cov[("T", s)][ci]
        for roi in models:
            rows.append(dict(kind="T_vs_brain", subject=s, roi=roi, rho=spear(T, brains[roi][s_i][ci])))
            for (lab, si), m in models[roi].items():
                rows.append(dict(kind="T_vs_model", subject=s, roi=roi, layer=LAYER[roi], condition=lab, seed=si,
                                 rho=spear(T, m[ci])))
    return pd.DataFrame(rows)


def interpret(met):
    out = []
    if met["W2"] and met["W3"] and met["W4"]:
        out.append("luminance: the advantage is attributed to luminance dominance of the THINGS-720 V1 RDM "
                   "combined with the untrained network's greater luminance sensitivity")
    if not met["W5"]:
        out.append("presentation order contributes")
    if not met["W1"]:
        out.append("the test-image difference is compatible with sampling variation of 100 images")
    return out or ["not identified"]


def run(verbose=True):
    import nsd_step43 as NS
    c = cfg()
    C.require_free(c["disk"]["min_free_gb"])
    OUT.mkdir(parents=True, exist_ok=True)
    t0, times = time.time(), {}

    # THINGS-720
    imgs = things720_images()
    L720, C720 = lum_contrast(imgs)
    pos = positions()
    cov = {}
    for s in SUBJ:
        cov[("L", s)], cov[("C", s)], cov[("T", s)] = abs_vec(L720), abs_vec(C720), abs_vec(pos[s])
    ci = crossrun_index()
    models = {roi: things720_models(roi) for roi in ("V1", "LOC")}
    brains = {roi: things720_brains(roi) for roi in ("V1", "LOC")}
    d0, _ = delta_on(models["V1"], brains["V1"], ci)
    ref = pd.read_csv(REPO / "results" / "step5" / "resolution_rsa.csv", low_memory=False)
    ref = float(ref[(ref.kind == "gap_point") & (ref.arm == "NATIVE") & (ref.res == 224) & (ref.roi == "V1")
                    & (ref.layer == "Conv1") & (ref.variant == "raw")].value.iloc[0])
    if abs(d0 - ref) > 1e-12:
        raise SystemExit(f"STOP: paper cell {d0} does not reproduce step 5 {ref}")
    times["setup"] = time.time() - t0

    t = time.time()
    sub = run_w1(c, models, brains)
    stamp(sub).to_csv(OUT / "subsets.csv", index=False)
    times["W1 subsets"] = time.time() - t

    t = time.time()
    par = run_partial(models, brains, ci, cov)
    stamp(par).to_csv(OUT / "partial.csv", index=False)
    times["W2/W5 partial"] = time.time() - t

    # luminance by dataset (W3, W4)
    t = time.time()
    rows = []
    Lv = abs_vec(L720)
    for roi in ("V1", "LOC"):
        bl = [spear(b[ci], Lv[ci]) for b in brains[roi]]
        rows += lum_rows("things720", roi, LAYER[roi], model_lum(models[roi], Lv, ci), bl,
                         NS.loso_lower([b[ci] for b in brains[roi]]), dict(pairs="cross-run P", n_pairs=len(ci)))
    timgs, tbrains, tmodels, n_draws = test_set()
    Lt, _ = lum_contrast(timgs)
    Ltv = abs_vec(Lt)
    allt = np.arange(len(Ltv))
    # step 41 check: k = 1 Delta (mean over draws, subjects, seeds) reproduces noise_curve.csv
    dk1 = np.mean([delta_on(tmodels, [tbrains[(s, "V1")][d] for s in SUBJ], allt)[0] for d in range(n_draws)])
    n41 = pd.read_csv(REPO / "results" / "step41" / "noise_curve.csv", low_memory=False)
    r41 = float(n41[(n41.kind == "contrast") & (n41.roi == "V1") & (n41.k == 1) & (n41.model_units == "raw")
                    & (n41.model_distance == "correlation") & (n41.brain_rdm == "b1")].delta.iloc[0])
    if abs(dk1 - r41) > 1e-9:
        raise SystemExit(f"STOP: test-image k = 1 Delta {dk1} does not reproduce step 41 {r41}")
    for roi in ("V1", "LOC"):
        bl = [np.mean([spear(tbrains[(s, roi)][d], Ltv) for d in range(n_draws)]) for s in SUBJ]
        bound = float(np.mean([NS.loso_lower([tbrains[(s, roi)][d] for s in SUBJ]) for d in range(n_draws)]))
        rows += lum_rows("things_test", roi, "Conv1", model_lum(tmodels, Ltv, allt), bl, bound,
                         dict(pairs="all", n_pairs=len(allt), k=1, n_draws=n_draws))
    import nsd_features as F
    from nsd_rsa import load_features
    Ln, _ = lum_contrast(F.image_paths())
    Lnv = abs_vec(Ln)
    nmodels = {k: model_rdm(np.asarray(X, np.float64), "raw", "correlation") for k, X in load_features().items()}
    subs, z, k1idx = nsd_k1_index()
    ceil = pd.read_csv(REPO / "results" / "step43" / "ceiling_nsd.csv")
    ml_n = model_lum(nmodels, Lnv, np.arange(len(Lnv)))
    for roi in ("V1", "LO-obj"):
        bl = [np.mean([spear(z[s][f"{roi}|correlation|k1_{r}"][k1idx[r]], Lnv[k1idx[r]]) for r in range(3)]) for s in subs]
        bound = float(ceil[(ceil.kind == "ceiling") & (ceil.roi == roi) & (ceil.k == 1)].value.iloc[0])
        rows += lum_rows("nsd", roi, "Conv1", ml_n, bl, bound,
                         dict(pairs="k = 1, no subject same session and run", n_pairs=int(np.mean([len(i) for i in k1idx])), k=1))
    lum = pd.DataFrame(rows)
    lum = pd.concat([lum, pd.DataFrame([dict(kind="check", quantity="test images k = 1 Delta vs step 41", value=dk1, ref=r41),
                                        dict(kind="check", quantity="THINGS-720 paper cell vs step 5", value=d0, ref=ref)])],
                    ignore_index=True)
    stamp(lum).to_csv(OUT / "luminance_by_dataset.csv", index=False)
    times["W3/W4 luminance"] = time.time() - t

    t = time.time()
    order = run_order(pos, cov, models, brains, ci, L720)
    stamp(order).to_csv(OUT / "order.csv", index=False)
    times["order"] = time.time() - t

    # summary
    s1 = sub[(sub.kind == "summary") & (sub.roi == "V1")].iloc[0]
    pv = par[(par.kind == "delta") & (par.roi == "V1")].set_index("covariates").delta
    ms = lum[lum.kind == "model_lum_summary"].set_index(["dataset", "roi"])
    bs = lum[lum.kind == "brain_lum"].set_index(["dataset", "roi"])
    w3_n = {d: int(ms.loc[(d, "V1"), "seeds_random_larger"]) for d in ("things720", "things_test", "nsd")}
    rat = {d: float(bs.loc[(d, "V1"), "ratio"]) for d in ("things720", "things_test", "nsd")}
    met = {"W1": bool(s1.value >= 0.95), "W2": bool(pv["L+C"] <= 0.010),
           "W3": all(v == 5 for v in w3_n.values()),
           "W4": bool(rat["things720"] > rat["things_test"] and rat["things720"] > rat["nsd"]),
           "W5": bool(abs(pv["T"] - pv["none"]) < 0.005)}
    summ = [dict(rule="W1", quantity="fraction of subsets with Delta > -0.0002 (>= 0.95)", value=float(s1.value), passed=met["W1"],
                 note=f"mean {s1['mean']:+.4f}, 2.5-97.5 % [{s1.q025:+.4f}, {s1.q975:+.4f}]"),
            dict(rule="W2", quantity="THINGS-720 V1 Delta after partialling L and C (<= +0.010)", value=float(pv["L+C"]),
                 passed=met["W2"], note=f"Delta without {pv['none']:+.4f}"),
            dict(rule="W3", quantity="seeds with Spearman(model, L) Random > BP (5 of 5 in each dataset)",
                 value=float(min(w3_n.values())), passed=met["W3"], note="; ".join(f"{k} {v}/5" for k, v in w3_n.items())),
            dict(rule="W4", quantity="rho(V1, L) / LOSO bound: THINGS-720 > test (k = 1) and > NSD (k = 1)",
                 value=rat["things720"], passed=met["W4"], note="; ".join(f"{k} {v:.3f}" for k, v in rat.items())),
            dict(rule="W5", quantity="|Delta(T partialled) - Delta| (< 0.005)", value=float(abs(pv["T"] - pv["none"])),
                 passed=met["W5"], note=f"Delta {pv['none']:+.4f} -> {pv['T']:+.4f}")]
    summ += [dict(rule="interpretation", quantity="amendment rule", note=s) for s in interpret(met)]
    summ += [dict(rule="runtime", quantity=f"{k} seconds", value=v) for k, v in times.items()]
    summ += [dict(rule="runtime", quantity="total seconds", value=time.time() - t0)]
    sm = pd.DataFrame(summ)
    stamp(sm).to_csv(OUT / "summary.csv", index=False)
    if verbose:
        print(sm[["rule", "quantity", "value", "passed", "note"]].to_string(index=False))
    return sm


def estimate():
    """Time one subset, one partial call and one model-luminance call; extrapolate."""
    models = {roi: things720_models(roi) for roi in ("V1", "LOC")}
    brains = {roi: things720_brains(roi) for roi in ("V1", "LOC")}
    ci = crossrun_index()
    t = time.time()
    rng = np.random.default_rng(0)
    for _ in range(5):
        stims = np.sort(rng.choice(N720, 100, replace=False))
        delta_on(models["V1"], brains["V1"], crossrun_index(stims))
    t_sub = (time.time() - t) / 5
    Z = [np.random.default_rng(1).random(len(ci)) for _ in range(3)]
    t = time.time()
    partial_rho(models["V1"][("Random", 0)][ci], brains["V1"][0][ci], Z)
    t_par = time.time() - t
    est = {"W1 (1000 subsets x 2 ROIs)": 2000 * t_sub, "partial (4 sets x 3 subj x 2 ROIs x 10 models)": 240 * t_par,
           "images (720 + 100 + 515, luminance and contrast)": 1335 * 0.15, "fixed (features, brains, NSD cache)": 300}
    est = {k: v * 1.5 for k, v in est.items()}
    for k, v in est.items():
        print(f"  {k:55s} {v / 60:6.1f} min")
    h = sum(est.values()) / 3600
    print(f"total x 1.5: {h:.2f} h")
    return h


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "estimate":
        h = estimate()
        if h > cfg()["runtime"]["stop_hours"]:
            raise SystemExit(f"STOP: estimate {h:.2f} h above {cfg()['runtime']['stop_hours']} h")
    elif which == "run":
        run()
    else:
        raise SystemExit("usage: step44_why.py {estimate|run}")

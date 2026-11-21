"""
step47_nsd_ridge.py
===================
Step 47, part B (amendments/2026-11-21_step47-predictions.md, config_step47.yaml): the
pre-specified primary ridge analysis of arXiv:2611.15533 applied to NSD.

  8 subjects, 515 images, k = 3 targets (mean of the three z-scored presentations), V1 (LO-obj
  reported); reliable voxels: Spearman-Brown corrected mean pairwise presentation correlation > 0.2;
  fit.fit_condition (dual ridge, 8 x 5 nested CV, seed 20260818, alpha grid of config.yaml, min-MSE
  selection) with assert_low_boundary False and the D1 plateau gate of scripts/lower_edge_gate.py;
  pooled out-of-fold r, median over reliable voxels; MSE skill as step25_ridge.scores.
  Features: nsd_features.build (the prereg extraction code), Conv1 after max-pool, 224 px,
  g in {1, 4, 8}; Random and BP x seeds 0-4; luminance-and-contrast bank (control_lumcontrast).

  RD1  g = 4: Random - BP < 0 in >= 6 of 8 subjects and one-sided sign-flip p <= 0.05
  RD2  the same at g = 1 and at g = 8

Only aggregated values are written (NSD rule a/b).

Usage:
  python scripts/step47_nsd_ridge.py extract       g = 4 and 8 features -> NSD_DIR/features
  python scripts/step47_nsd_ridge.py estimate
  python scripts/step47_nsd_ridge.py run [workers]
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads, sets the import paths)

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from config_hash import config_hash, load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config_step47.yaml"
OUT = REPO / "results" / "step47"
GRIDS = [1, 4, 8]
KEYS = {"random": "Random", "backprop": "BP"}
ROIS = ["V1", "LO-obj"]


def cfg():
    return load_config(CONFIG)


def stamp(df):
    df["config_hash"] = config_hash(cfg())
    df["git_head"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
    if "seed" not in df:
        df["seed"] = np.nan
    df["seed"] = df["seed"].astype("Int64")
    return df


def alphas():
    from ridge_dual import alpha_grid
    a = load_config()["ridge"]["alpha_grid"]
    return alpha_grid(a["min"], a["max"], a["n_points"])


# ── reliability and targets ─────────────────────────────────────────────────
def reliability(P):
    """P (n_images, 3, v): mean over the 3 presentation pairs of the voxel-wise Pearson r across
    images; returns (r_SB3 = 3r / (1 + 2r), r_SB2 = 2r / (1 + r), r)."""
    from ridge_dual import pearson_columns
    P = np.asarray(P, np.float64)
    r = np.mean([pearson_columns(P[:, a], P[:, b]) for a, b in ((0, 1), (0, 2), (1, 2))], axis=0)
    return 3 * r / (1 + 2 * r), 2 * r / (1 + r), r


_T = {}


def targets(subject, roi):
    """(Y (515, v) mean of the three presentations, reliable mask, SB2 count); cached per process."""
    if (subject, roi) not in _T:
        from nsd_rsa import load_patterns
        P, _, _, rois, _ = load_patterns(subject)
        Pm = np.asarray(P[:, :, rois[roi]], np.float64)
        sb3, sb2, _ = reliability(Pm)
        thr = cfg()["B"]["reliability"]["threshold"]
        _T[(subject, roi)] = (Pm.mean(axis=1), sb3 > thr, int((sb2 > thr).sum()))
    return _T[(subject, roi)]


def features(name, g):
    import nsd_features as F
    X = np.load(F.feature_dir() / f"{name}_g{g}.npy")
    if X.dtype != np.float32 or X.shape[0] != 515:
        raise SystemExit(f"STOP: {name}_g{g} is {X.dtype} {X.shape}")
    return X.astype(np.float64)


# ── one fit, as prereg_primary.run_subject.recording_fit ────────────────────
def fit_one(X, Y, reliable, context):
    """fit_condition with assert_low_boundary False, the plateau gate on low-edge hits, scores."""
    from fit import fit_condition
    from lower_edge_gate import gate, hits_from_fit
    from step25_ridge import scores
    res = fit_condition(X, Y, np.arange(Y.shape[0]), alphas=alphas(), reliable=reliable,
                        assert_low_boundary=False, context=context)
    hits = hits_from_fit(res, reliable, condition=context)
    cls = "none"
    if hits:
        gd = gate(hits, load_X=lambda h: X, Y=Y)
        cls = "invalid" if (gd.classification == "invalid").any() else "plateau"
    sc = scores(res, Y, reliable)
    return dict(d=int(res["d"]), r_med_reliable=sc["r_pooled"], r_med_all=float(np.median(res["r"])),
                mse_skill=sc["mse_skill"], n_low_edge_hits=len(hits), gate=cls, valid=cls != "invalid",
                **{k: float(np.mean(res[k])) for k in ("lo_rel", "hi_rel", "lo_unrel", "hi_unrel")})


def _job(job):
    subject, roi, g, name = job
    Y, rel, n_sb2 = targets(subject, roi)
    out = fit_one(features(name, g), Y, rel, f"{subject} {roi} {name} g{g}")
    return dict(subject=subject, roi=roi, grid=g, name=name, n_voxels=Y.shape[1], n_reliable=int(rel.sum()),
                n_reliable_sb2=n_sb2, **out)


def jobs():
    subs = cfg()["B"]["subjects"]
    names = [f"model_{k}_seed{si}" for k in KEYS for si in range(5)] + ["control_lumcontrast"]
    return [(s, roi, g, n) for roi in ROIS for s in subs for g in GRIDS for n in names]


def extract():
    import nsd_features as F
    todo = [g for g in (4, 8) if not all((F.feature_dir() / f"model_{k}_seed{si}_g{g}.npy").exists()
                                         for k in KEYS for si in range(5))]
    if todo:
        t = time.time()
        F.build(F.feature_dir(), F.image_paths(), grids=todo)()
        print(f"extraction g = {todo}: {time.time() - t:.0f} s", flush=True)


def run(workers=8):
    from nsd_rsa import sign_flip_p
    C.require_free(cfg()["disk"]["min_free_gb"])
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    js = jobs()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(_job, js), 1):
            rows.append(r)
            if k % 24 == 0:
                print(f"  {k}/{len(js)} fits ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df["kind"] = "fit"
    df["condition"] = df.name.map(lambda n: "bank" if n.startswith("control") else KEYS[n.split("_")[1]])
    df["seed"] = df.name.map(lambda n: int(n[-1]) if n.startswith("model") else -1)
    df.loc[df.seed < 0, "seed"] = np.nan
    subs = cfg()["B"]["subjects"]
    agg, rules = [], {}
    for roi in ROIS:
        for g in GRIDS:
            q = df[(df.roi == roi) & (df.grid == g)]
            per = []
            for s in subs:
                x = q[q.subject == s]
                m = x[x.condition.isin(["Random", "BP"])]
                n_invalid = int((~m.valid).sum())
                mv = m[m.valid]
                r_r = float(mv[mv.condition == "Random"].r_med_reliable.mean())
                r_b = float(mv[mv.condition == "BP"].r_med_reliable.mean())
                bank = x[x.condition == "bank"].iloc[0]
                per.append(r_r - r_b)
                agg.append(dict(kind="subject", roi=roi, grid=g, subject=s, r_random=r_r, r_bp=r_b, delta=r_r - r_b,
                                skill_random=float(mv[mv.condition == "Random"].mse_skill.mean()),
                                skill_bp=float(mv[mv.condition == "BP"].mse_skill.mean()),
                                r_bank=float(bank.r_med_reliable), skill_bank=float(bank.mse_skill), bank_valid=bool(bank.valid),
                                n_reliable=int(x.n_reliable.iloc[0]), n_reliable_sb2=int(x.n_reliable_sb2.iloc[0]),
                                n_voxels=int(x.n_voxels.iloc[0]), n_invalid_model_cells=n_invalid,
                                n_low_edge_hits=int(x.n_low_edge_hits.sum()),
                                seeds_random_below_bp=int(sum(mv[(mv.condition == "Random") & (mv.seed == si)].r_med_reliable.mean()
                                                              < mv[(mv.condition == "BP") & (mv.seed == si)].r_med_reliable.mean()
                                                              for si in range(5)))))
            per = np.array(per)
            p = sign_flip_p(per, "negative")
            n_neg = int((per < 0).sum())
            rules[(roi, g)] = (n_neg, p)
            agg.append(dict(kind="group", roi=roi, grid=g, delta=float(per.mean()), n_negative=n_neg, p_negative=p,
                            n_subjects=len(per)))
    out = pd.concat([df.drop(columns=["name"]), pd.DataFrame(agg)], ignore_index=True)
    stamp(out).to_csv(OUT / "nsd_ridge.csv", index=False)

    def ok(g):
        n, p = rules[("V1", g)]
        return n >= 6 and p <= 0.05

    b_rows = [dict(rule="RD1", quantity="g = 4: Random - BP < 0 in >= 6/8 subjects and p <= 0.05",
                   value=rules[("V1", 4)][0], passed=ok(4), note=f"{rules[('V1', 4)][0]}/8 negative, p {rules[('V1', 4)][1]:.4f}"),
              dict(rule="RD2", quantity="the same at g = 1 and g = 8", value=min(rules[("V1", 1)][0], rules[("V1", 8)][0]),
                   passed=ok(1) and ok(8), note="; ".join(f"g = {g}: {rules[('V1', g)][0]}/8, p {rules[('V1', g)][1]:.4f}" for g in (1, 8))),
              dict(rule="runtime", quantity="B fits seconds", value=time.time() - t0)]
    sp = OUT / "summary.csv"
    prev = pd.read_csv(sp) if sp.exists() else pd.DataFrame()
    if len(prev):
        prev = prev[~prev.rule.isin(["RD1", "RD2"]) & ~((prev.rule == "runtime") & prev.quantity.str.startswith("B "))]
        prev = prev.drop(columns=[c for c in ("config_hash", "git_head", "seed") if c in prev])
    sm = pd.concat([prev, pd.DataFrame(b_rows)], ignore_index=True)
    stamp(sm).to_csv(sp, index=False)
    print(pd.DataFrame(agg)[lambda x: x.kind == "group"].to_string(index=False))
    print(pd.DataFrame(b_rows)[["rule", "value", "passed", "note"]].to_string(index=False))


def estimate(workers=8):
    t = time.time()
    r = _job(("subj01", "V1", 4, "model_random_seed0"))
    t_v1 = time.time() - t
    t = time.time()
    _job(("subj01", "LO-obj", 4, "model_random_seed0"))
    t_lo = time.time() - t
    n = len(jobs()) // 2
    h = n * (t_v1 + t_lo) / workers * 1.5 / 3600
    print(f"one fit: V1 {t_v1:.1f} s ({r['n_voxels']} voxels), LO-obj {t_lo:.1f} s; {2 * n} fits / {workers} workers x 1.5: {h:.2f} h "
          f"(single-thread fits in parallel workers may be slower than measured here)")
    return h


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if which == "extract":
        extract()
    elif which == "estimate":
        h = estimate(w)
        if h > cfg()["runtime"]["stop_hours_B"]:
            raise SystemExit(f"STOP: estimate {h:.2f} h above {cfg()['runtime']['stop_hours_B']} h")
    elif which == "run":
        run(w)
    else:
        raise SystemExit("usage: step47_nsd_ridge.py {extract|estimate|run} [workers]")

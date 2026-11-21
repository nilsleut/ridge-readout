"""
step47b_mechanism.py
====================
Step 47b, part X (amendments/2026-11-21_step47b-predictions.md, config_step47b.yaml): mechanism of
the run structure in the THINGS-720 session, V1, per subject.

  Brain run RDM   correlation distance between the 10 run-mean V1 patterns (mean over all 82 trials
                  of the run of the z-scored responses of step47_runs.load_resp)
  Model run RDM   correlation distance between the 10 run-mean GAP Conv1 vectors (raw units; mean over
                  the 82 images of the run), per seed, then mean over seeds 0-4 of the 45 entries
  A(model)        Spearman(brain run RDM, model run RDM); Delta A = A(Random) - A(BP)
  Null            10,000 reassignments of the 820 images to 10 runs of 82 (seed 20261122), model side only

  X1  Delta A > 0 in sub-01 and sub-02
  X2  observed Delta A of sub-01 and sub-02 within the central 95 % of its null (2.5 <= pct <= 97.5)
  X3  Spearman(brain run RDM, |r - s|) > 0 in >= 2 of 3 subjects

Imported: step47_runs (load_resp, percentile), step7_multiverse.feat, step40_common (feature_dir,
test_stimuli), common_up_pairs.stim_order. Only aggregated values are written.

Usage:
  python scripts/step47b_mechanism.py estimate
  python scripts/step47b_mechanism.py run
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads, sets the import paths)

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

import paths
from config_hash import config_hash, load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config_step47b.yaml"
OUT = REPO / "results" / "step47b"
SUBJ = ["sub-01", "sub-02", "sub-03"]
KEYS = {"random": ("Random", "Random Weights"), "backprop": ("BP", "Backprop")}
N_RUNS, PER_RUN = 10, 82


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


# ── images, runs, features ──────────────────────────────────────────────────
def images820():
    import step40_common as S40
    from common_up_pairs import stim_order
    return list(stim_order()) + list(S40.test_stimuli())


def session_runs(subject, names=None):
    """(run index 0..9 per image of images820, metadata rows of the session in image order).
    Runs ordered by run number; STOP unless every image occurs exactly once in the session."""
    names = names or images820()
    st = pd.read_csv(paths.things_fmri_dir() / f"{subject}_task-things_stimulus-metadata.csv")
    ses = st.set_index("stimulus").loc[names[:720]].session.unique()
    if len(ses) != 1:
        raise SystemExit(f"STOP: {subject}: the 720 are not in one session")
    S_ = st[st.session == ses[0]]
    if len(S_) != N_RUNS * PER_RUN or S_.stimulus.nunique() != len(S_) or set(S_.stimulus) != set(names):
        raise SystemExit(f"STOP: {subject}: session {ses[0]} does not hold exactly the 820 images once")
    runs = sorted(S_.run.unique())
    pos = {r: i for i, r in enumerate(runs)}
    by = S_.set_index("stimulus")
    lab = np.array([pos[by.loc[n, "run"]] for n in names])
    if not (np.bincount(lab, minlength=N_RUNS) == PER_RUN).all():
        raise SystemExit(f"STOP: {subject}: runs do not have 82 images each")
    rows = np.array([S_.index[S_.stimulus == n][0] for n in names])
    return lab, rows


def features820():
    """{(label, seed): (820, 32) float64}: step7_multiverse.feat (720) stacked on test100 g = 1."""
    import step40_common as S40
    import step7_multiverse as S7
    out = {}
    for key, (lab, rule) in KEYS.items():
        for si in cfg_seeds():
            a = S7.feat("NATIVE", 224, rule, si, "Conv1")
            b = np.load(S40.feature_dir() / "test100" / f"model_{key}_seed{si}_g1.npy").astype(np.float64)
            out[(lab, si)] = np.vstack([a, b])
    return out


def cfg_seeds():
    return [0, 1, 2, 3, 4]


# ── run RDMs and alignment ──────────────────────────────────────────────────
def run_means(X, lab, n_runs=N_RUNS):
    X = np.asarray(X, np.float64)
    return np.stack([X[lab == r].mean(axis=0) for r in range(n_runs)])


def run_rdm(X, lab, n_runs=N_RUNS):
    return pdist(run_means(X, lab, n_runs), "correlation")


def run_index_rdm(n_runs=N_RUNS):
    return pdist(np.arange(n_runs, dtype=np.float64)[:, None], "cityblock")


def spear(a, b):
    return float(spearmanr(a, b).statistic)


def model_run_rdms(feats, lab):
    """{label: mean over seeds of the run RDM}."""
    out = {}
    for lab_name in ("Random", "BP"):
        out[lab_name] = np.mean([run_rdm(X, lab) for (l, _), X in feats.items() if l == lab_name], axis=0)
    return out


def delta_a(brain, models):
    a_r, a_b = spear(brain, models["Random"]), spear(brain, models["BP"])
    return a_r, a_b, a_r - a_b


def perm_labels(n_draws, seed, n_img=N_RUNS * PER_RUN, n_runs=N_RUNS, per_run=PER_RUN):
    """Run labels per draw: default_rng(seed).permutation(n_img) cut into n_runs of per_run
    (the step47_runs.perm_null scheme)."""
    rng = np.random.default_rng(seed)
    base = np.repeat(np.arange(n_runs), per_run)
    for _ in range(n_draws):
        lab = np.empty(n_img, int)
        lab[rng.permutation(n_img)] = base
        yield lab


def null_delta_a(brains, feats, n_draws, seed):
    """(n_draws, n_subjects) null Delta A; model side permuted, brains fixed."""
    out = np.empty((n_draws, len(brains)))
    for k, lab in enumerate(perm_labels(n_draws, seed)):
        m = model_run_rdms(feats, lab)
        for j, b in enumerate(brains):
            out[k, j] = delta_a(b, m)[2]
    return out


def brain_run_rdm(subject):
    import step47_runs as A
    resp, a, st = A.load_resp(subject)
    lab, rows = session_runs(subject)
    v1 = a["local_V1"]
    R = resp[rows][:, v1]
    return run_rdm(R, lab), lab


# ── driver ──────────────────────────────────────────────────────────────────
def run():
    import step47_runs as A
    c = cfg()["X"]
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    feats = features820()
    brains, labs = {}, {}
    for s in SUBJ:
        brains[s], labs[s] = brain_run_rdm(s)
    if not np.array_equal(labs["sub-01"], labs["sub-02"]):
        raise SystemExit("STOP: sub-01 and sub-02 run assignments differ")
    obs = {s: delta_a(brains[s], model_run_rdms(feats, labs[s])) for s in SUBJ}
    ridx = run_index_rdm()
    x3 = {s: spear(brains[s], ridx) for s in SUBJ}
    t_null = time.time()
    null = null_delta_a([brains[s] for s in SUBJ], feats, c["permutation"]["n"], c["permutation"]["seed"])
    t_null = time.time() - t_null
    pct = {s: A.percentile(null[:, j], obs[s][2]) for j, s in enumerate(SUBJ)}
    rc = pd.read_csv(REPO / "results" / "step47" / "run_centring.csv")
    rc = rc[(rc.kind == "subject_delta") & (rc.roi == "V1")].set_index(["variant", "subject"]).delta
    rows = []
    for j, s in enumerate(SUBJ):
        a_r, a_b, d = obs[s]
        rows.append(dict(kind="subject", subject=s, A_random=a_r, A_bp=a_b, delta_A=d, percentile=pct[s],
                         rho_run_index=x3[s], delta_raw=float(rc[("raw", s)]), delta_runcentred=float(rc[("run-centred", s)]),
                         run_level_part=float(rc[("raw", s)] - rc[("run-centred", s)])))
    for a, b in (("sub-01", "sub-02"), ("sub-01", "sub-03"), ("sub-02", "sub-03")):
        rows.append(dict(kind="between_subjects", subject=f"{a} vs {b}", rho_brain_run_rdm=spear(brains[a], brains[b])))
    stamp(pd.DataFrame(rows)).to_csv(OUT / "mechanism.csv", index=False)
    ps = [dict(subject=s, n=len(null), seed=c["permutation"]["seed"], mean=float(null[:, j].mean()),
               sd=float(null[:, j].std(ddof=1)), q025=float(np.quantile(null[:, j], .025)),
               q50=float(np.median(null[:, j])), q975=float(np.quantile(null[:, j], .975)),
               observed=obs[s][2], percentile=pct[s]) for j, s in enumerate(SUBJ)]
    stamp(pd.DataFrame(ps)).to_csv(OUT / "permutation_summary.csv", index=False)

    met = {"X1": obs["sub-01"][2] > 0 and obs["sub-02"][2] > 0,
           "X2": all(2.5 <= pct[s] <= 97.5 for s in ("sub-01", "sub-02")),
           "X3": sum(x3[s] > 0 for s in SUBJ) >= 2}
    fail2 = [s for s in ("sub-01", "sub-02") if not 2.5 <= pct[s] <= 97.5]
    if met["X1"] and met["X2"] and met["X3"]:
        interp = ("the run-level part of the THINGS-720 advantage is attributed to drift across runs together with a chance "
                  "alignment of the untrained network's run-mean features in the run assignment shared by sub-01 and sub-02")
    elif met["X1"] and not met["X2"] and all(pct[s] > 97.5 for s in fail2):
        interp = "attributed to stimulus-driven run-level structure"
    else:
        interp = "not identified"
    summ = [dict(rule="X1", quantity="Delta A > 0 in sub-01 and sub-02", value=min(obs["sub-01"][2], obs["sub-02"][2]),
                 passed=met["X1"], note="; ".join(f"{s} {obs[s][2]:+.4f}" for s in SUBJ)),
            dict(rule="X2", quantity="observed Delta A within the central 95 % of the null (sub-01, sub-02)",
                 value=max(abs(pct["sub-01"] - 50), abs(pct["sub-02"] - 50)), passed=met["X2"],
                 note="; ".join(f"{s} pct {pct[s]:.1f}" for s in SUBJ)),
            dict(rule="X3", quantity="Spearman(brain run RDM, run-index RDM) > 0 in >= 2 of 3",
                 value=sum(x3[s] > 0 for s in SUBJ), passed=met["X3"], note="; ".join(f"{s} {x3[s]:+.3f}" for s in SUBJ)),
            dict(rule="interpretation", quantity="amendment rule (part X)", note=interp),
            dict(rule="runtime", quantity="X null seconds", value=t_null),
            dict(rule="runtime", quantity="X total seconds", value=time.time() - t0)]
    write_summary(summ, "X")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(summ)[["rule", "value", "passed", "note"]].to_string(index=False))


def write_summary(rows, part):
    """Merge rows of one part into results/step47b/summary.csv (rules of the part replaced)."""
    sp = OUT / "summary.csv"
    new = pd.DataFrame(rows)
    new["part"] = part
    if sp.exists():
        prev = pd.read_csv(sp).drop(columns=["config_hash", "git_head", "seed"], errors="ignore")
        prev = prev[prev.part != part]
        new = pd.concat([prev, new], ignore_index=True)
    order = {"X": 0, "Y-THINGS": 1, "Y-NSD": 2}
    new = new.sort_values("part", key=lambda s: s.map(order), kind="stable")
    stamp(new).to_csv(sp, index=False)


def estimate():
    feats = features820()
    lab = next(perm_labels(1, 0))
    b = run_index_rdm()
    t = time.time()
    for _ in range(200):
        m = model_run_rdms(feats, lab)
        for _ in range(3):
            delta_a(b, m)
    per = (time.time() - t) / 200
    h = (per * cfg()["X"]["permutation"]["n"] + 3 * 5) * 1.5 / 3600
    print(f"one null draw {per * 1e3:.1f} ms; part X x 1.5: {h:.3f} h")
    return h


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "estimate":
        h = estimate()
        if h > cfg()["runtime"]["stop_hours_per_part"]:
            raise SystemExit(f"STOP: estimate {h:.2f} h above {cfg()['runtime']['stop_hours_per_part']} h")
    elif which == "run":
        run()
    else:
        raise SystemExit("usage: step47b_mechanism.py {estimate|run}")

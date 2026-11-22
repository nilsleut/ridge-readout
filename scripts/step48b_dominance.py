"""
step48b_dominance.py
====================
Step 48b (amendments/2026-11-22_step48b-predictions.md, config_step48b.yaml): amplitude dominance
of the RDMs. No new downloads; the step 48 data (RIDGE_DATA_DIR/step48, NSD_DIR/raw48).

Per unit (dataset, convention, ROI, subject, k, presentation or draw), paper cell (raw units,
correlation distance):
  brain RDM b, brain gain g (step48_amplitude.gain), amplitude RDM A (sum form 2 - g_i - g_j; the
  difference form |g_i - g_j| for comparison)
  per model (condition, seed): model RDM m, g^M, A^M
  rho          Spearman(m, b) as Pearson of the ranks over the pair set
  rho_partial  two-sided: rank_resid(m, [A^M]) vs rank_resid(b, [A]) (step 44 functions)
  dominance    Spearman(b, A) (brain) and Spearman(m, A^M) (model)
Delta = Random - BP (mean over seeds, presentations / draws), then mean over subjects.

Datasets: THINGS-720 (cross-run pairs; Conv1 V1, Conv3 LOC), THINGS test images (all pairs,
k = 1: 20 draws, k = 12), NSD-515 (k = 1 per presentation without same-session-and-run pairs,
k = 3 presentation mean; V1, LO-obj, lateral).

gate: recomputes Delta(T-raw, 720, V1), Delta(N-raw, k = 1, V1) (step 48 conventions.csv) and
brain_gain, brain_model_gain, model_gain_stat (step 48 amplitude.csv); no dominance or partial.

Usage:
  python scripts/step48b_dominance.py gate
  python scripts/step48b_dominance.py run
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads, sets the import paths)

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata

from config_hash import config_hash, load_config
import step44_why as W
import step48_amplitude as AM
import step48_conventions as S48

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config_step48b.yaml"
OUT = REPO / "results" / "step48b"
R48 = REPO / "results" / "step48"
SUBJ = S48.SUBJ
SEEDS = range(5)
FORMS = ("sum", "diff")
TOL = 1e-12
_LL = {}


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


def ll():
    if not _LL:
        _LL.update(W.lowlevel_defs())
    return _LL


# ── core ────────────────────────────────────────────────────────────────────
def amp_rdm(g, form):
    """Condensed amplitude RDM of the gains g: 'sum' 2 - g_i - g_j, 'diff' |g_i - g_j|."""
    g = np.asarray(g, np.float64)
    i, j = np.triu_indices(len(g), 1)
    if form == "sum":
        return 2.0 - g[i] - g[j]
    if form == "diff":
        return np.abs(g[i] - g[j])
    raise ValueError(form)


def rho_rank(m, b):
    """Spearman as Pearson of the ranks (average ties)."""
    return ll()["pearson"](rankdata(m), rankdata(b))


def partial2(m, b, am, ab):
    """Two-sided partial: model ranks residualised on A^M, brain ranks on A, Pearson."""
    return ll()["pearson"](ll()["rank_resid"](m, [am]), ll()["rank_resid"](b, [ab]))


class Models:
    """Model RDMs and gains of one dataset / layer, with cached model-side residuals per pair set."""

    def __init__(self, feats):
        from cvrsa import model_rdm
        self.keys = list(feats)
        self.rdm = {k: model_rdm(X, "raw", "correlation") for k, X in feats.items()}
        self.gm = {k: AM.gain(X) for k, X in feats.items()}
        self.amp = {(k, f): amp_rdm(self.gm[k], f) for k in self.keys for f in FORMS}
        self.feats = feats
        self._res = {}

    def resid(self, k, form, sel_key, sel):
        key = (k, form, sel_key)
        if key not in self._res:
            m = self.rdm[k] if sel is None else self.rdm[k][sel]
            a = self.amp[(k, form)] if sel is None else self.amp[(k, form)][sel]
            self._res[key] = ll()["rank_resid"](m, [a])
        return self._res[key]


def unit_rows(models, P, sel, sel_key, info, full=True):
    """Rows for one brain unit: P (n, v) analysed patterns; sel = pair index (None = all pairs)."""
    b = pdist(np.asarray(P, np.float64), "correlation")
    g = AM.gain(P)
    bs = b if sel is None else b[sel]
    rows = []
    amp_b = {f: (amp_rdm(g, f) if sel is None else amp_rdm(g, f)[sel]) for f in FORMS}
    res_b = {f: ll()["rank_resid"](bs, [amp_b[f]]) for f in FORMS} if full else {}
    if full:
        rows += [dict(kind="brain_dominance", form=f, rho=AM.spear(bs, amp_b[f]), **info) for f in FORMS]
    for k in models.keys:
        m = models.rdm[k] if sel is None else models.rdm[k][sel]
        rows.append(dict(kind="rho", form="none", condition=k[0], seed=k[1], rho=rho_rank(m, bs), **info))
        if full:
            for f in FORMS:
                rows.append(dict(kind="rho", form=f, condition=k[0], seed=k[1],
                                 rho=ll()["pearson"](models.resid(k, f, sel_key, sel), res_b[f]), **info))
    return rows, g


def gain_rows(g_list, gm, info):
    """The step 48 brain_gain / brain_model_gain quantities (for the gate)."""
    rows = [dict(kind="brain_gain", mean=float(np.mean([g.mean() for g in g_list])),
                 sd=float(np.mean([g.std() for g in g_list])), **info)]
    rows += [dict(kind="brain_model_gain", condition=k[0], seed=k[1], rho=float(np.mean([AM.spear(g, v) for g in g_list])),
                  **info) for k, v in gm.items()]
    return rows


def model_dominance_rows(models, dataset, layer, sel):
    import step6_common as S6
    rows = []
    for k in models.keys:
        for f in FORMS:
            m = models.rdm[k] if sel is None else models.rdm[k][sel]
            a = models.amp[(k, f)] if sel is None else models.amp[(k, f)][sel]
            rows.append(dict(kind="model_dominance", dataset=dataset, layer=layer, form=f, condition=k[0], seed=k[1],
                             rho=AM.spear(m, a)))
        rows.append(dict(kind="moment_ratio", dataset=dataset, layer=layer, condition=k[0], seed=k[1],
                         value=S6.moment_ratio(models.feats[k])))
    return rows


# ── datasets ────────────────────────────────────────────────────────────────
def things720(full):
    import step7_multiverse as S7
    ci = W.crossrun_index()
    data = {s: S48.things_data(s) for s in SUBJ}
    loc = {s: S48.local_rois(s) for s in SUBJ}
    rows, gains = [], []
    for roi, layer in W.LAYER.items():
        models = Models({(lab, si): S7.feat("NATIVE", 224, rule, si, layer) for lab, rule in W.RULES.items() for si in SEEDS})
        if full:
            rows += model_dominance_rows(models, "things720", layer, ci)
        for s in SUBJ:
            for cv in S48.T_CONV:
                info = dict(dataset="things720", convention=cv, roi=roi, subject=s, k=1, rep=0)
                r, g = unit_rows(models, data[s][f"p720_{cv}"][:, loc[s][roi]], ci, "ci", info, full)
                rows += r
                gains += gain_rows([g], models.gm, info)
        gains += model_stat_rows(models, "things720", layer)
    return rows, gains


def things_test(full):
    import step40_common as S40
    import step41_noise_curve as S41
    draws = S41.session_draws()
    feats = {(lab, si): np.load(S40.feature_dir() / "test100" / f"model_{key}_seed{si}_g1.npy")
             for key, lab in {"random": "Random", "backprop": "BP"}.items() for si in SEEDS}
    models = Models(feats)
    rows = model_dominance_rows(models, "things_test", "Conv1", None) if full else []
    gains = model_stat_rows(models, "things_test", "Conv1")
    for s in SUBJ:
        t = S48.things_data(s)
        _, rois = S40.load_reps(s)
        for cv in S48.T_CONV:
            for roi in ("V1", "LOC"):
                R = t[f"reps_{cv}"][:, :, rois[roi]]
                for k in (1, 12):
                    gl = []
                    for di, sess in enumerate(draws[k]):
                        info = dict(dataset="things_test", convention=cv, roi=roi, subject=s, k=k, rep=di)
                        r, g = unit_rows(models, R[:, sess].mean(axis=1), None, "all", info, full)
                        rows += r
                        gl.append(g)
                    if k == 12:
                        gains += gain_rows(gl, models.gm, dict(dataset="things_test", convention=cv, roi=roi, subject=s, k=12))
    return rows, gains


def nsd(full):
    import nsd_raw_fetch as RF
    from nsd_rsa import load_features, same_run_pairs
    models = Models({k: np.asarray(v, np.float64) for k, v in load_features().items()})
    rows = model_dominance_rows(models, "nsd515", "Conv1", None) if full else []
    gains = model_stat_rows(models, "nsd515", "Conv1")
    t0 = time.time()
    for s in C.cfg()["subjects"]:
        conv, session, run, rois, _ = RF.conventions(s)
        keep = {r: np.flatnonzero(~same_run_pairs(session, run, r)) for r in range(3)}
        for cv in S48.N_CONV:
            for roi in ("V1", "LO-obj", "lateral"):
                P = np.asarray(conv[cv][:, :, rois[roi]], np.float64)
                gl = []
                for r in range(3):
                    info = dict(dataset="nsd515", convention=cv, roi=roi, subject=s, k=1, rep=r)
                    rr, g = unit_rows(models, P[:, r], keep[r], f"{s}_{r}", info, full)
                    rows += rr
                    gl.append(g)
                gains += gain_rows(gl, models.gm, dict(dataset="nsd515", convention=cv, roi=roi, subject=s, k=1))
                info = dict(dataset="nsd515", convention=cv, roi=roi, subject=s, k=3, rep=0)
                rr, g = unit_rows(models, P.mean(axis=1), None, "all", info, full)
                rows += rr
                gains += gain_rows([g], models.gm, dict(info))
        print(f"  {s} ({time.time() - t0:.0f} s)", flush=True)
    return rows, gains


def model_stat_rows(models, dataset, layer):
    st = AM.dataset_stats(dataset)
    import image_stats as IS
    return [dict(kind="model_gain_stat", dataset=dataset, layer=layer, condition=k[0], seed=k[1], stat=x,
                 rho=AM.spear(models.gm[k], st[x])) for k in models.keys for x in IS.STATS]


# ── aggregation ─────────────────────────────────────────────────────────────
def deltas(rows):
    """Per subject and group Delta per (dataset, convention, roi, k, form)."""
    from nsd_rsa import sign_flip_p
    q = pd.DataFrame([r for r in rows if r["kind"] == "rho"])
    keys = ["dataset", "convention", "roi", "k", "form", "subject"]
    per = q.groupby(keys + ["condition"]).rho.mean().unstack("condition").reset_index()
    per["delta"] = per["Random"] - per["BP"]
    out = [dict(kind="subject_delta", **{c: r[c] for c in keys}, rho_random=r.Random, rho_bp=r.BP, delta=r.delta)
           for _, r in per.iterrows()]
    for key, g in per.groupby(keys[:-1]):
        x = g.delta.to_numpy()
        out.append(dict(kind="delta", **dict(zip(keys[:-1], key)), delta=float(x.mean()), n_subjects=len(x),
                        n_positive=int((x > 0).sum()), n_le_0020=int((x <= 0.020).sum()),
                        p_positive=sign_flip_p(x, "positive") if len(x) == 8 else np.nan,
                        rho_random=float(g.Random.mean()), rho_bp=float(g.BP.mean())))
    return out


# ── gate ────────────────────────────────────────────────────────────────────
def keyed(df, cols):
    """df indexed by cols as strings (k and seed as integers), for matching against step 48."""
    df = df.copy()
    for c in cols:
        if c in ("k", "seed"):
            df[c] = pd.to_numeric(df[c]).astype(int).astype(str)
        else:
            df[c] = df[c].astype(str)
    return df.set_index(cols)


def gate():
    t0 = time.time()
    rows, gains = [], []
    for fn in (things720, things_test, nsd):
        r, g = fn(full=False)
        rows += r
        gains += g
    d = pd.DataFrame(deltas(rows))
    c48 = pd.read_csv(R48 / "conventions.csv", low_memory=False, float_precision="round_trip")
    a48 = pd.read_csv(R48 / "amplitude.csv", low_memory=False, float_precision="round_trip")
    res = {}
    # Delta T-raw THINGS-720 V1 and N-raw k = 1 V1, per subject and mean
    for ds, cv, k, kind48, ksub in (("things720", "T-raw", 1, "things720", "things720_subject"),
                                     ("nsd515", "N-raw", 1, "nsd", "nsd_subject")):
        sel = (c48.convention == cv) & (c48.roi == "V1") & (c48.units == "raw") & (c48.distance == "correlation")
        if ds == "nsd515":
            sel &= c48.k == k
        ref_m = float(c48[sel & (c48.kind == kind48)].delta.iloc[0])
        ref_s = c48[sel & (c48.kind == ksub)].set_index("subject").delta
        q = d[(d.dataset == ds) & (d.convention == cv) & (d.roi == "V1") & (d.k == k) & (d.form == "none")]
        mine_m = float(q[q.kind == "delta"].delta.iloc[0])
        mine_s = q[q.kind == "subject_delta"].set_index("subject").delta.loc[ref_s.index]
        res[f"Delta {cv} {ds} V1 k={k}"] = (mine_m, ref_m, max(abs(mine_m - ref_m), float((mine_s - ref_s).abs().max())))
    # g and g^M
    g = pd.DataFrame(gains)
    for kind, cols, val in (("brain_gain", ["dataset", "convention", "roi", "subject", "k"], ["mean", "sd"]),
                            ("brain_model_gain", ["dataset", "convention", "roi", "subject", "k", "condition", "seed"], ["rho"]),
                            ("model_gain_stat", ["dataset", "layer", "condition", "seed", "stat"], ["rho"])):
        mine = keyed(g[g.kind == kind], cols)[val]
        ref = a48[a48.kind == kind].copy()
        if "seed" in cols:
            # step 48 write_part dropped the seed column of the THINGS part when the NSD part was
            # merged; the rows are in generation order (seeds 0-4 within each condition)
            miss = ref.seed.isna()
            other = [c for c in cols if c != "seed"]
            ref.loc[miss, "seed"] = ref[miss].fillna({"k": -1}).groupby(other, sort=False).cumcount().to_numpy()
            n_rec = int(miss.sum())
        ref = keyed(ref, cols)[val]
        if len(mine) != len(ref) or set(mine.index) != set(ref.index):
            raise SystemExit(f"STOP: {kind}: {len(mine)} rows vs {len(ref)} in step 48")
        dev = float((mine - ref.loc[mine.index]).abs().to_numpy().max())
        same = (mine.to_numpy() == ref.loc[mine.index].to_numpy()).all(axis=1)
        rec = f", seed from row order in {n_rec}" if "seed" in cols else ""
        res[f"{kind} ({len(mine)} rows{rec}), bitwise identical rows"] = (f"{int(same.sum())}/{len(same)}", None, dev)
    print(f"gate ({time.time() - t0:.0f} s; tolerance {TOL:g}: BLAS dot products differ by 1-2 ulp between runs)")
    ok = True
    for k, (a, b, dev) in res.items():
        print(f"  {k}: {a if b is None else f'{a:+.6f} vs step 48 {b:+.6f}'}  max |d| {dev:.1e}")
        ok &= dev <= TOL
    if not ok:
        raise SystemExit("STOP: gate failed")
    print("  gate passed")
    return res


# ── run ─────────────────────────────────────────────────────────────────────
def summary(dom, dl):
    q = dom[(dom.kind == "model_dominance") & (dom.form == "sum")]
    d1, d1_note = {}, []
    for ds, layer in (("things720", "Conv1"), ("things_test", "Conv1"), ("nsd515", "Conv1")):
        x = q[(q.dataset == ds) & (q.layer == layer)].pivot_table(index="seed", columns="condition", values="rho")
        d1[ds] = int((x.Random > x.BP).sum())
        d1_note.append(f"{ds}: {d1[ds]}/5 (R " + " ".join(f"{v:+.3f}" for v in x.Random) + "; BP "
                       + " ".join(f"{v:+.3f}" for v in x.BP) + ")")
    b = dom[(dom.kind == "brain_dominance") & (dom.form == "sum") & (dom.roi == "V1")]
    t = b[(b.dataset == "things720")].pivot_table(index="subject", columns="convention", values="rho")
    n = b[(b.dataset == "nsd515") & (b.k == 1)].groupby(["subject", "convention"]).rho.mean().unstack()
    d2t, d2n = int((t["T-raw"] > t["T-session"]).sum()), int((n["N-raw"] > n["N-session"]).sum())
    p = dl[(dl.dataset == "nsd515") & (dl.convention == "N-raw") & (dl.roi == "V1") & (dl.k == 1)]
    before = float(p[(p.kind == "delta") & (p.form == "none")].delta.iloc[0])
    after = p[(p.kind == "delta") & (p.form == "sum")].iloc[0]
    sub = p[(p.kind == "subject_delta") & (p.form == "sum")].set_index("subject").delta
    met = {"D1": all(v == 5 for v in d1.values()), "D2": d2t == 3 and d2n >= 7,
           "D3": bool(after.delta <= 0.020 and int((sub <= 0.020).sum()) >= 6)}
    if all(met.values()):
        interp = ("the untrained advantage under conventions that leave a common profile in both patterns is attributed to "
                  "amplitude structure dominating both RDMs, with the untrained network's RDM more amplitude-dominated; "
                  "not to a better match of per-image amplitude")
    else:
        interp = "not identified; all values are reported"
    rows = [dict(rule="D1", quantity="Spearman(model RDM, A^M) Random > BP in 5 of 5 seeds, in each dataset",
                 value=min(d1.values()), passed=met["D1"], note="; ".join(d1_note)),
            dict(rule="D2", quantity="Spearman(brain RDM, A): T-raw > T-session in 3 of 3; N-raw > N-session (k = 1) in >= 7 of 8",
                 value=d2n, passed=met["D2"],
                 note=f"THINGS {d2t}/3 (" + "; ".join(f"{s} {t.loc[s, 'T-raw']:+.3f} vs {t.loc[s, 'T-session']:+.3f}" for s in t.index)
                 + f"); NSD {d2n}/8 (" + "; ".join(f"{s} {n.loc[s, 'N-raw']:+.3f} vs {n.loc[s, 'N-session']:+.3f}" for s in n.index) + ")"),
            dict(rule="D3", quantity="NSD N-raw k = 1 V1: two-sided partial Delta <= +0.020 and <= +0.020 in >= 6 of 8",
                 value=float(after.delta), passed=met["D3"],
                 note=f"before {before:+.4f}, after {after.delta:+.4f}; {int((sub <= 0.020).sum())}/8 <= +0.020; per subject "
                      + "; ".join(f"{s} {v:+.4f}" for s, v in sub.items())),
            dict(rule="interpretation", quantity="amendment rule", note=interp)]
    return pd.DataFrame(rows)


def run():
    C.require_free(cfg()["disk"]["min_free_gb"])
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows, times = [], {}
    for name, fn in (("THINGS-720", things720), ("THINGS test", things_test), ("NSD", nsd)):
        t = time.time()
        r, _ = fn(full=True)
        rows += r
        times[name] = time.time() - t
        print(f"{name} done ({times[name]:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    dom = df[df.kind.isin(["brain_dominance", "model_dominance", "moment_ratio"])].dropna(axis=1, how="all").copy()
    # Spearman of model dominance with the moment ratio across the 10 checkpoints
    md = dom[dom.kind == "model_dominance"]
    mr = dom[dom.kind == "moment_ratio"].set_index(["dataset", "layer", "condition", "seed"]).value
    extra = []
    for (ds, ly, f), g in md.groupby(["dataset", "layer", "form"]):
        g = g.set_index(["dataset", "layer", "condition", "seed"]).rho
        extra.append(dict(kind="dominance_vs_moment_ratio", dataset=ds, layer=ly, form=f, n=len(g),
                          rho=AM.spear(g.to_numpy(), mr.loc[g.index].to_numpy())))
    # brain dominance aggregated over draws / presentations per subject
    bd = dom[dom.kind == "brain_dominance"].groupby(["dataset", "convention", "roi", "k", "form", "subject"]).rho.mean().reset_index()
    bd["kind"] = "brain_dominance_subject"
    dom = pd.concat([dom, bd, pd.DataFrame(extra)], ignore_index=True)
    dl = pd.DataFrame(deltas(rows))
    rho = df[df.kind == "rho"].groupby(["dataset", "convention", "roi", "k", "form", "subject", "condition", "seed"]).rho.mean().reset_index()
    rho["kind"] = "rho_subject_seed"
    part = pd.concat([dl, rho], ignore_index=True)
    sm = summary(dom, dl)
    times["total"] = time.time() - t0
    sm = pd.concat([sm, pd.DataFrame([dict(rule="runtime", quantity=f"{k} seconds", value=v) for k, v in times.items()])],
                   ignore_index=True)
    stamp(dom).to_csv(OUT / "dominance.csv", index=False)
    stamp(part).to_csv(OUT / "partial.csv", index=False)
    stamp(sm).to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(sm[["rule", "value", "passed", "note"]].to_string(index=False))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "gate":
        gate()
    elif which == "run":
        run()
    else:
        raise SystemExit("usage: step48b_dominance.py {gate|run}")

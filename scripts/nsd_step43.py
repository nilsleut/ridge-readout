"""
nsd_step43.py
=============
Step 43 (amendments/2026-11-17_step43-predictions.md, config_step43.yaml): NSD extensions.

  M   multiverse F x D x B at V1 (own CNN, step 42 features), distinct analyses, M1 and M2
  I   six ImageNet models, trained vs untrained (s0-s2), paper cell, z/correlation, N, A, W;
      I1 and I2
  R   own CNN at NATIVE 32 ... 224 px, raw and z units, correlation, V1; slope S = Delta(224) - Delta(32)
  NC  leave-one-subject-out lower bound of the noise ceiling, V1 and LO-obj, k = 3 and k = 1

Existing functions are imported, not re-implemented:
  step7_multiverse      FS, DS, BS, betas (B transform; its roi_betas is pointed at the NSD
                        patterns for the duration of the call), normalise (F transform)
  step28_distinct_cells the class rule (re-stated in distinct_classes, same tolerance, same checks)
  step8_extract_pretrained  build, layers, pool, tf, extract (ImageNet models, layers, pooling)
  v12_lift              namespaces, transform, load_v12_model, extract_features (step 5 extraction)
  nsd_rsa               rank_z, component_vectors (step 25 nwa), sign_flip_p, boot_indices,
                        load_patterns, load_features, same_run_pairs
  learning-rules-rsa    scripts/noise_ceiling_v2/common.py::nili_bounds (LOSO lower bound)

Point estimates are draw 0 (the identity draw) of the stimulus bootstrap; draws 1..1000 are
the step 42 resamples (seed 20261002, self-pairs dropped).

Usage:
  python scripts/nsd_step43.py estimate          runtime estimate (STOP above 10 h)
  python scripts/nsd_step43.py extract           ImageNet and resolution features -> NSD_DIR
  python scripts/nsd_step43.py run [workers]     all four parts -> results/step43
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads)

import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

import paths
from cvrsa import condensed_index, model_rdm
from nsd_rsa import (ROIS, COMPS, rank_z, component_vectors, sign_flip_p, boot_indices,
                     load_patterns, load_features, same_run_pairs)

CONFIG = C.REPO / "config_step43.yaml"
OUT = C.REPO / "results" / "step43"
N_IMG = 515
LABS = ["Random", "BP"]
SEEDS = [0, 1, 2, 3, 4]
IMN_MODELS = ["alexnet", "vgg11_bn", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
IMN_CONDS = ["trained", "untrained_s0", "untrained_s1", "untrained_s2"]
IMN_R = 224
IMN_CELLS = [("raw", "correlation"), ("z", "correlation"), ("center", "N"), ("center", "A"), ("center", "W")]
IMN_ROI_LAYER = {"V1": "V1", "LO-obj": "LOC", "lateral": "LOC"}
RES_PX = [32, 64, 96, 128, 160, 224]
RES_RULES = {"Random": "Random Weights", "BP": "Backprop"}
RES_UNITS = ["raw", "z"]
TOL = 1e-9


def cfg():
    from config_hash import load_config
    return load_config(CONFIG)


def stamp(df):
    from config_hash import config_hash
    C.stamp(df)
    df["config_hash_42"] = df["config_hash"]
    df["config_hash"] = config_hash(cfg())
    return df


def subjects():
    return cfg()["subjects"]


# ── shared core ─────────────────────────────────────────────────────────────
def rho_pairs(M, B, pairs, ci):
    """Spearman rho for every (model row, brain row) in pairs, over the pair index ci.
    Only the rows that occur in pairs are ranked."""
    mi, bi = np.unique(pairs[:, 0]), np.unique(pairs[:, 1])
    Zm = dict(zip(mi, rank_z(np.asarray(M[mi][:, ci], np.float64))))
    Zb = dict(zip(bi, rank_z(np.asarray(B[bi][:, ci], np.float64))))
    return np.array([float(Zm[a] @ Zb[b]) for a, b in pairs])


def draw_index(idx, n=N_IMG):
    """Condensed pair index of a resample (self-pairs dropped); idx None = identity."""
    if idx is None:
        return np.arange(n * (n - 1) // 2)
    a, b = np.triu_indices(n, 1)
    si, sj = idx[a], idx[b]
    keep = si != sj
    return condensed_index(si[keep], sj[keep], n)


_W = {}


def _init(mpath, bpath, pairs):
    _W.update(M=np.load(mpath, mmap_mode="r"), B=np.load(bpath, mmap_mode="r"), pairs=pairs)


def _chunk(chunk):
    out = np.zeros((len(chunk), _W["B"].shape[0], len(_W["pairs"])))
    for j, (_, idx) in enumerate(chunk):
        ci = draw_index(idx)
        for s in range(_W["B"].shape[0]):
            out[j, s] = rho_pairs(_W["M"], _W["B"][s], _W["pairs"], ci)
    return [b for b, _ in chunk], out


def bootstrap(name, M, B, pairs, workers):
    """rho draws (1001, n_subjects, n_pairs); draw 0 = identity. M (n_models, P), B (n_sub, n_brains, P)."""
    tmp = C.nsd_dir("tmp43")
    mpath, bpath = tmp / f"{name}_M.npy", tmp / f"{name}_B.npy"
    np.save(mpath, np.asarray(M, np.float64))
    np.save(bpath, np.asarray(B, np.float64))
    bc = cfg()["inference"]["bootstrap"]
    items = [(0, None)] + [(b + 1, idx) for b, idx in enumerate(boot_indices(N_IMG, bc["n_resamples"], bc["seed"]))]
    n_chunks = workers * 6
    chunks = [items[i::n_chunks] for i in range(n_chunks)]
    acc = np.zeros((len(items), B.shape[0], len(pairs)))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(mpath, bpath, pairs)) as ex:
        for done, (bs, out) in enumerate(ex.map(_chunk, chunks), 1):
            acc[bs] = out
            if done % workers == 0 or done == len(chunks):
                print(f"  {name}: chunk {done}/{len(chunks)} ({time.time() - t0:.0f} s)", flush=True)
    os.remove(mpath)
    os.remove(bpath)
    return acc


def summarise(draws, weights, direction="negative"):
    """draws (1001, n_sub, n_pairs); weights (n_pairs,) -> per-subject Delta and group statistics."""
    d = draws @ weights                     # (1001, n_sub)
    per = d[0]
    m = d.mean(axis=1)
    return dict(mean_delta=float(m[0]), ci_lo=float(np.quantile(m[1:], .025)), ci_hi=float(np.quantile(m[1:], .975)),
                n_negative=int((per < 0).sum()), n_positive=int((per > 0).sum()),
                p_negative=sign_flip_p(per, "negative"), p_positive=sign_flip_p(per, "positive")), per


def ci_class(lo, hi):
    return "above 0" if lo > 0 else ("below 0" if hi < 0 else "includes 0")


def brain_X(subject):
    """{roi: (515, v) float64 presentation average} as in step 42 (nsd_rsa.brain_vectors)."""
    P, _, _, rois, _ = load_patterns(subject)
    return {r: np.asarray(P[:, :, m], np.float64).mean(axis=1) for r, m in rois.items()}


def rdm_cache(subject):
    return np.load(C.nsd_dir("rdm") / f"{subject}.npz")


# ── M: multiverse ───────────────────────────────────────────────────────────
def mv_brain(X, b):
    """step7_multiverse.betas(.., b) on the matrix X: its roi_betas is pointed at X during the call."""
    import step7_multiverse as S7
    old = S7.roi_betas
    S7.roi_betas = lambda *_: np.asarray(X, np.float64)
    try:
        return S7.betas(None, None, b)
    finally:
        S7.roi_betas = old


def mv_models(feats):
    """keys [(F, D, label, seed)], matrix of condensed model RDMs (step7_multiverse.normalise)."""
    import step7_multiverse as S7
    keys, rows = [], []
    for f in S7.FS:
        for d in S7.DS:
            for k, X in feats.items():
                v = pdist(S7.normalise(X, f), d)
                if not np.isfinite(v).all():
                    raise SystemExit(f"STOP: non-finite model RDM {f} {d} {k}")
                keys.append((f, d) + tuple(k))
                rows.append(v)
    return keys, np.stack(rows)


def mv_brains(X):
    import step7_multiverse as S7
    keys, rows = [], []
    for b in S7.BS:
        Xb = mv_brain(X, b)
        for d in S7.DS:
            v = pdist(Xb, d)
            if not np.isfinite(v).all():
                raise SystemExit(f"STOP: non-finite brain RDM {b} {d}")
            keys.append((b, d))
            rows.append(v)
    return keys, np.stack(rows)


def mv_pairs(mkeys, bkeys):
    pairs, labels = [], []
    for bi, (b, d) in enumerate(bkeys):
        for mi, (f, dm, lab, si) in enumerate(mkeys):
            if dm == d:
                pairs.append((mi, bi))
                labels.append(dict(F=f, D=d, B=b, condition=lab, seed=si))
    return np.array(pairs), pd.DataFrame(labels)


def distinct_classes(vec, cls_of):
    """The step 28 rule: union of cells whose value vectors agree to TOL; every pair inside a class
    must agree and members must share the interval class (else STOP)."""
    from itertools import combinations
    cells = list(vec)
    dmax = {(a, c): float(np.abs(vec[a] - vec[c]).max()) for a, c in combinations(cells, 2)}
    parent = {c: c for c in cells}

    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c

    for (a, c), v in dmax.items():
        if v <= TOL:
            parent[find(c)] = find(a)
    roots = sorted({find(c) for c in cells}, key=cells.index)
    cls = {c: roots.index(find(c)) + 1 for c in cells}
    for (a, c), v in dmax.items():
        if cls[a] == cls[c] and v > TOL:
            raise SystemExit(f"STOP: class not transitive ({a} vs {c}: {v:.2e})")
        if cls[a] == cls[c] and cls_of[a] != cls_of[c]:
            raise SystemExit(f"STOP: members of one class carry different interval classes ({a}, {c})")
    within = max([v for (a, c), v in dmax.items() if cls[a] == cls[c]] or [0.0])
    between = min(v for (a, c), v in dmax.items() if cls[a] != cls[c])
    return cls, within, between


def run_multiverse(workers):
    import step7_multiverse as S7
    t0 = time.time()
    subs = subjects()
    feats = load_features()
    mkeys, M = mv_models(feats)
    B, bkeys = [], None
    for s in subs:
        bkeys, Bs = mv_brains(brain_X(s)["V1"])
        B.append(Bs)
    B = np.stack(B)
    pairs, lab = mv_pairs(mkeys, bkeys)
    draws = bootstrap("multiverse", M, B, pairs, workers)
    # check: B = paper, correlation, F = raw / z equals the step 42 V1 rho (k = 3)
    dev = 0.0
    for s_i, s in enumerate(subs):
        r42 = pd.read_csv(C.nsd_dir("rsa") / f"{s}.csv")
        r42 = r42[(r42.kind == "rho") & (r42.roi == "V1") & (r42.k == 3) & (r42.distance == "correlation")]
        for row in r42.itertuples():
            j = lab.index[(lab.F == row.model_units) & (lab.D == "correlation") & (lab.B == "paper")
                          & (lab.condition == row.condition) & (lab.seed == row.seed)][0]
            dev = max(dev, abs(draws[0, s_i, j] - row.rho))
    print(f"multiverse: paper cells vs step 42 rho max |d| {dev:.1e}", flush=True)
    if dev > TOL:
        raise SystemExit("STOP: multiverse paper cell does not reproduce step 42")
    rows, per_rows, vec, cls_of = [], [], {}, {}
    for f in S7.FS:
        for d in S7.DS:
            for b in S7.BS:
                sel = (lab.F == f) & (lab.D == d) & (lab.B == b)
                w = np.where(sel & (lab.condition == "Random"), 1 / 5, 0.0) - np.where(sel & (lab.condition == "BP"), 1 / 5, 0.0)
                st, per = summarise(draws, w, "negative")
                cell = (f, d, b)
                order = lab[sel].sort_values(["condition", "seed"]).index
                vec[cell] = draws[0][:, order].ravel()
                cls_of[cell] = ci_class(st["ci_lo"], st["ci_hi"])
                rows.append(dict(kind="cell", F=f, D=d, B=b, roi="V1", **st, ci_class=cls_of[cell],
                                 mean_rho_random=float(draws[0][:, sel & (lab.condition == "Random")].mean()),
                                 mean_rho_bp=float(draws[0][:, sel & (lab.condition == "BP")].mean())))
                per_rows += [dict(kind="subject_delta", F=f, D=d, B=b, roi="V1", subject=s, delta=float(v))
                             for s, v in zip(subs, per)]
    cls, within, between = distinct_classes(vec, cls_of)
    cells = pd.DataFrame(rows)
    cells["cls"] = [cls[(r.F, r.D, r.B)] for r in cells.itertuples()]
    cells["class_size"] = cells.cls.map(cells.cls.value_counts())
    first = cells.drop_duplicates("cls")
    n_cls = len(first)
    above, below = int((first.ci_class == "above 0").sum()), int((first.ci_class == "below 0").sum())
    rho_rows = []
    for j, r in lab.iterrows():
        rho_rows += [dict(kind="rho", F=r.F, D=r.D, B=r.B, roi="V1", condition=r.condition, seed=r.seed,
                          subject=s, rho=float(draws[0, s_i, j])) for s_i, s in enumerate(subs)]
    summ = [dict(kind="summary", quantity="distinct analyses", value=n_cls),
            dict(kind="summary", quantity="distinct analyses, interval above 0", value=above),
            dict(kind="summary", quantity="distinct analyses, interval below 0", value=below),
            dict(kind="summary", quantity="distinct analyses, interval includes 0", value=n_cls - above - below),
            dict(kind="check", quantity="paper cell vs step 42 rho max |d|", value=dev),
            dict(kind="check", quantity="max |d| within classes", value=within),
            dict(kind="check", quantity="min max|d| between classes", value=between),
            dict(kind="summary", quantity="runtime s", value=time.time() - t0)]
    out = pd.concat([cells, pd.DataFrame(per_rows), pd.DataFrame(rho_rows), pd.DataFrame(summ)], ignore_index=True)
    stamp(out).to_csv(OUT / "multiverse_nsd.csv", index=False)
    rules = [dict(rule="M1", quantity="distinct analyses with the interval above 0 <= 2", value=above,
                  passed=bool(above <= 2), note=f"{above} above, {below} below, {n_cls - above - below} include 0 of {n_cls}"),
             dict(rule="M2", quantity="distinct analyses with the interval below 0 >= half", value=below,
                  passed=bool(below >= n_cls / 2), note=f"{below} of {n_cls} (half = {n_cls / 2:g})")]
    print(cells[["F", "D", "B", "cls", "mean_delta", "ci_lo", "ci_hi", "n_negative", "ci_class"]].to_string(index=False))
    return rules


# ── I: ImageNet models ──────────────────────────────────────────────────────
def imn_dir():
    return C.nsd_dir("features_imagenet")


def imn_extract(img_paths, out_root, models=IMN_MODELS, conds=IMN_CONDS, R=IMN_R):
    """step8_extract_pretrained.build / extract / tf on an explicit image list -> out_root/{m}/{c}/R{R}/{V1,LOC}.npy."""
    import torch
    from PIL import Image
    import step8_extract_pretrained as S8
    torch.set_num_threads(os.cpu_count())
    X = torch.stack([S8.tf(R)(Image.open(p).convert("RGB")) for p in img_paths])
    for m in models:
        S8.layers(m, S8.build(m, "untrained_s0"))
    for m in models:
        if R not in S8.resolutions(m, [R]):
            continue
        for c in conds:
            d = out_root / m / c / f"R{R}"
            if all((d / f"{k}.npy").exists() for k in ("V1", "LOC")):
                continue
            feats = S8.extract(m, S8.build(m, c), X)
            d.mkdir(parents=True, exist_ok=True)
            for k, v in feats.items():
                assert v.shape[0] == len(img_paths) and np.isfinite(v).all(), (m, c, k)
                np.save(d / f"{k}.npy", v.astype(np.float32))
            print(f"  imagenet {m} {c} done", flush=True)


def imn_feat(root, m, c, layer, R=IMN_R):
    return np.load(root / m / c / f"R{R}" / f"{layer}.npy").astype(np.float64)


def imn_models(get):
    """keys [(model, cond, layer, units, cell)], matrix; get(m, c, layer) -> features."""
    keys, rows = [], []
    for m in IMN_MODELS:
        for c in IMN_CONDS:
            for layer in ("V1", "LOC"):
                X = get(m, c, layer)
                comp = component_vectors(X)
                for u, cell in IMN_CELLS:
                    v = model_rdm(X, u, cell) if cell == "correlation" else comp[cell]
                    keys.append((m, c, layer, u, cell))
                    rows.append(v)
    return keys, np.stack(rows)


def imn_brain_keys():
    return [(r, u, cell) for r in ROIS for u, cell in IMN_CELLS]


def imn_brains(subject):
    z = rdm_cache(subject)
    rows = []
    for r, u, cell in imn_brain_keys():
        rows.append(z[f"{r}|correlation|k3"] if cell == "correlation" else z[f"{r}|{cell}|k3"])
    return np.stack(rows)


def imn_pairs(mkeys, bkeys):
    pairs, labels = [], []
    for bi, (r, u, cell) in enumerate(bkeys):
        for mi, (m, c, layer, mu, mc) in enumerate(mkeys):
            if layer == IMN_ROI_LAYER[r] and (mu, mc) == (u, cell):
                pairs.append((mi, bi))
                labels.append(dict(model=m, cond=c, roi=r, model_units=u, distance=cell))
    return np.array(pairs), pd.DataFrame(labels)


def imn_weights(lab, m, r, u, cell, untrained=("untrained_s0", "untrained_s1", "untrained_s2")):
    sel = (lab.model == m) & (lab.roi == r) & (lab.model_units == u) & (lab.distance == cell)
    w = np.where(sel & lab.cond.isin(untrained), 1 / len(untrained), 0.0)
    return w - np.where(sel & (lab.cond == "trained"), 1.0, 0.0)


def run_imagenet(workers):
    t0 = time.time()
    subs = subjects()
    root = imn_dir()
    mkeys, M = imn_models(lambda m, c, layer: imn_feat(root, m, c, layer))
    bkeys = imn_brain_keys()
    B = np.stack([imn_brains(s) for s in subs])
    pairs, lab = imn_pairs(mkeys, bkeys)
    draws = bootstrap("imagenet", M, B, pairs, workers)
    rows, per_rows = [], []
    for m in IMN_MODELS:
        for r in ROIS:
            for u, cell in IMN_CELLS:
                for variant, untr in (("mean s0-s2", ("untrained_s0", "untrained_s1", "untrained_s2")),
                                      ("s0 only", ("untrained_s0",))):
                    st, per = summarise(draws, imn_weights(lab, m, r, u, cell, untr))
                    sel = (lab.model == m) & (lab.roi == r) & (lab.model_units == u) & (lab.distance == cell)
                    rows.append(dict(kind="cell", model=m, roi=r, model_units=u, distance=cell, untrained=variant, **st,
                                     mean_rho_trained=float(draws[0][:, sel & (lab.cond == "trained")].mean()),
                                     mean_rho_untrained=float(draws[0][:, sel & lab.cond.isin(untr)].mean())))
                    if variant == "mean s0-s2":
                        per_rows += [dict(kind="subject_delta", model=m, roi=r, model_units=u, distance=cell,
                                          subject=s, delta=float(v)) for s, v in zip(subs, per)]
    rho_rows = []
    for j, r in lab.iterrows():
        rho_rows += [dict(kind="rho", model=r.model, cond=r.cond, roi=r.roi, model_units=r.model_units,
                          distance=r.distance, subject=s, rho=float(draws[0, s_i, j])) for s_i, s in enumerate(subs)]
    cells = pd.DataFrame(rows)
    out = pd.concat([cells, pd.DataFrame(per_rows), pd.DataFrame(rho_rows),
                     pd.DataFrame([dict(kind="summary", quantity="runtime s", value=time.time() - t0)])], ignore_index=True)
    stamp(out).to_csv(OUT / "imagenet_nsd.csv", index=False)
    main = cells[cells.untrained == "mean s0-s2"]
    a = main[(main.roi == "V1") & (main.distance == "A")]
    p = main[(main.roi == "V1") & (main.distance == "correlation") & (main.model_units == "raw")]
    n_a, n_p = int((a.mean_delta < 0).sum()), int((p.mean_delta <= 0).sum())
    print(main[main.roi != "lateral"][["model", "roi", "model_units", "distance", "mean_delta", "ci_lo", "ci_hi",
                                       "n_negative", "p_negative"]].to_string(index=False))
    return [dict(rule="I1", quantity="V1, A: 8-subject mean Delta < 0 in >= 5 of 6 architectures", value=n_a,
                 passed=bool(n_a >= 5), note="; ".join(f"{r.model} {r.mean_delta:+.4f}" for r in a.itertuples())),
            dict(rule="I2", quantity="V1, raw, correlation: 8-subject mean Delta <= 0 in >= 4 of 6 architectures",
                 value=n_p, passed=bool(n_p >= 4), note="; ".join(f"{r.model} {r.mean_delta:+.4f}" for r in p.itertuples()))]


# ── R: resolution ───────────────────────────────────────────────────────────
def res_dir():
    return C.nsd_dir("features_resolution")


def res_path(root, px, rule, si):
    from v12_lift import rule_key
    return root / "NATIVE" / f"res{px}" / f"seed_{si}" / f"gap_{rule_key(rule)}_Conv1.npy"


def res_extract(img_paths, root, pxs=RES_PX, rules=tuple(RES_RULES.values()), seeds=SEEDS):
    """The step 5 extraction (scripts/step5_extract_gap.extract) on an explicit image list:
    v12 _tf (NATIVE), load_v12_model, extract_features; Conv1 GAP features are kept."""
    import torch
    from PIL import Image
    from v12_lift import LAYERS, SEEDS as V12_SEEDS, namespaces, transform, load_v12_model
    torch.set_num_threads(12)
    ns12, ns11 = namespaces()
    li = LAYERS.index("Conv1")
    for px in pxs:
        todo = [(r, si) for r in rules for si in seeds if not res_path(root, px, r, si).exists()]
        if not todo:
            continue
        tf = transform(ns12, ns11, "NATIVE", px)
        X = torch.stack([tf(Image.open(p).convert("RGB")) for p in img_paths])
        for rule, si in todo:
            torch.manual_seed(V12_SEEDS[si])
            m = load_v12_model(ns12, rule, si)
            f = ns12["extract_features"](m, list(range(len(img_paths))), X)[li]
            p = res_path(root, px, rule, si)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, np.asarray(f, np.float32))
        print(f"  resolution {px} px done", flush=True)


def res_models(get, pxs=RES_PX):
    """keys [(px, label, seed, units)], matrix of correlation-distance model RDMs (cvrsa.model_rdm)."""
    keys, rows = [], []
    for px in pxs:
        for lab, rule in RES_RULES.items():
            for si in SEEDS:
                X = get(px, rule, si)
                for u in RES_UNITS:
                    keys.append((px, lab, si, u))
                    rows.append(model_rdm(X, u, "correlation"))
    return keys, np.stack(rows)


def run_resolution(workers):
    t0 = time.time()
    subs = subjects()
    root = res_dir()
    mkeys, M = res_models(lambda px, rule, si: np.load(res_path(root, px, rule, si)).astype(np.float64))
    B = np.stack([rdm_cache(s)["V1|correlation|k3"][None] for s in subs])
    pairs = np.array([(i, 0) for i in range(len(mkeys))])
    lab = pd.DataFrame(mkeys, columns=["px", "condition", "seed", "model_units"])
    draws = bootstrap("resolution", M, B, pairs, workers)
    # check: 224 px through the step 5 extraction vs the step 42 features (same images)
    f42 = load_features()
    fdev = max(float(np.abs(np.load(res_path(root, 224, RES_RULES[l], si)) - f42[(l, si)]).max())
               for l in LABS for si in SEEDS)
    rows, per_rows = [], []

    def w_of(px, u):
        sel = (lab.px == px) & (lab.model_units == u)
        return np.where(sel & (lab.condition == "Random"), 1 / 5, 0.0) - np.where(sel & (lab.condition == "BP"), 1 / 5, 0.0)

    for u in RES_UNITS:
        for px in RES_PX:
            st, per = summarise(draws, w_of(px, u))
            sel = (lab.px == px) & (lab.model_units == u)
            rows.append(dict(kind="delta", roi="V1", model_units=u, distance="correlation", px=px, **st,
                             mean_rho_random=float(draws[0][:, sel & (lab.condition == "Random")].mean()),
                             mean_rho_bp=float(draws[0][:, sel & (lab.condition == "BP")].mean())))
            per_rows += [dict(kind="subject_delta", roi="V1", model_units=u, px=px, subject=s, delta=float(v))
                         for s, v in zip(subs, per)]
        st, per = summarise(draws, w_of(224, u) - w_of(32, u))
        rows.append(dict(kind="slope", roi="V1", model_units=u, distance="correlation",
                         quantity="S = Delta(224) - Delta(32)", **st))
        per_rows += [dict(kind="subject_slope", roi="V1", model_units=u, subject=s, delta=float(v)) for s, v in zip(subs, per)]
    out = pd.concat([pd.DataFrame(rows), pd.DataFrame(per_rows),
                     pd.DataFrame([dict(kind="check", quantity="224 px features (step 5 path) vs step 42 features max |d|", value=fdev),
                                   dict(kind="summary", quantity="runtime s", value=time.time() - t0)])], ignore_index=True)
    stamp(out).to_csv(OUT / "resolution_nsd.csv", index=False)
    print(pd.DataFrame(rows)[["kind", "model_units", "px", "mean_delta", "ci_lo", "ci_hi", "n_negative"]].to_string(index=False))
    print(f"224 px features, step 5 path vs step 42: max |d| {fdev:.2e}")


# ── NC: noise ceiling ───────────────────────────────────────────────────────
def nili_bounds():
    p = paths.crossrun_repo() / "scripts" / "noise_ceiling_v2" / "common.py"
    spec = importlib.util.spec_from_file_location("nc_v2_common", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.nili_bounds


def loso_lower(vecs):
    return float(nili_bounds()(vecs)["lower"])


def _nc_init(subs):
    keys = ["session", "run"] + [f"{r}|correlation|{k}" for r in ("V1", "LO-obj") for k in ("k3", "k1_0", "k1_1", "k1_2")]
    _W.update(nb=nili_bounds(), subs=subs, z={s: {k: rdm_cache(s)[k] for k in keys} for s in subs})


def _nc_draw(item):
    b, idx = item
    nb, z, subs = _W["nb"], _W["z"], _W["subs"]
    a_, b_ = np.triu_indices(N_IMG, 1)
    si, sj = (a_, b_) if idx is None else (idx[a_], idx[b_])
    keep = si != sj
    si, sj = si[keep], sj[keep]
    ci = condensed_index(si, sj, N_IMG)
    out = {}
    for roi in ("V1", "LO-obj"):
        out[(roi, 3)] = nb([z[s][f"{roi}|correlation|k3"][ci] for s in subs])["lower"]
        lows = []
        for r in range(3):
            ok = np.ones(len(ci), bool)
            for s in subs:
                key = z[s]["session"][:, r].astype(np.int64) * 100 + z[s]["run"][:, r]
                ok &= key[si] != key[sj]
            lows.append(nb([z[s][f"{roi}|correlation|k1_{r}"][ci][ok] for s in subs])["lower"])
        out[(roi, 1)] = float(np.mean(lows))
    return b, out


def run_ceiling(workers):
    t0 = time.time()
    subs = subjects()
    bc = cfg()["inference"]["bootstrap"]
    items = [(0, None)] + [(b + 1, idx) for b, idx in enumerate(boot_indices(N_IMG, bc["n_resamples"], bc["seed"]))]
    res = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_nc_init, initargs=(subs,)) as ex:
        for b, out in ex.map(_nc_draw, items, chunksize=8):
            res[b] = out
    # pair counts at k = 1 (identity), and the model rho for comparison (step 42 group, raw units)
    z = {s: rdm_cache(s) for s in subs}
    n1 = []
    for r in range(3):
        ok = np.ones(N_IMG * (N_IMG - 1) // 2, bool)
        for s in subs:
            ok &= ~same_run_pairs(z[s]["session"], z[s]["run"], r)
        n1.append(int(ok.sum()))
    g42 = pd.read_csv(C.OUT / "group.csv")
    rows = []
    for roi in ("V1", "LO-obj"):
        for k in (3, 1):
            v = np.array([res[b][(roi, k)] for b in range(len(items))])
            ref = g42[(g42.roi == roi) & (g42.k == k) & (g42.model_units == "raw") & (g42.distance == "correlation")
                      & (g42.subjects == "all 8")].iloc[0]
            rows.append(dict(kind="ceiling", roi=roi, k=k, distance="correlation", quantity="LOSO lower bound",
                             value=float(v[0]), ci_lo=float(np.quantile(v[1:], .025)), ci_hi=float(np.quantile(v[1:], .975)),
                             n_pairs=(N_IMG * (N_IMG - 1) // 2) if k == 3 else int(np.mean(n1)),
                             mean_rho_random_raw=float(ref.mean_rho_random), mean_rho_bp_raw=float(ref.mean_rho_bp)))
    out = pd.DataFrame(rows)
    out = pd.concat([out, pd.DataFrame([dict(kind="summary", quantity="runtime s", value=time.time() - t0)])], ignore_index=True)
    stamp(out).to_csv(OUT / "ceiling_nsd.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


# ── estimate, extraction, driver ────────────────────────────────────────────
def estimate(workers=8):
    """Measured: one ranking of a 132,355-pair vector, ImageNet and step 5 extraction per image."""
    import torch
    from PIL import Image
    import step8_extract_pretrained as S8
    import nsd_features as F
    t = time.time()
    v = np.random.default_rng(0).normal(size=(10, N_IMG * (N_IMG - 1) // 2))
    rank_z(v)
    t_rank = (time.time() - t) / 10
    imgs = F.image_paths()[:8]
    X = torch.stack([S8.tf(IMN_R)(Image.open(p).convert("RGB")) for p in imgs])
    t_imn = 0.0
    for m in IMN_MODELS:
        t = time.time()
        S8.extract(m, S8.build(m, "trained"), X, batch=8)
        t_imn += (time.time() - t) / len(imgs)
    n_sub = len(subjects())
    rank_per_draw = {
        "multiverse": 4 * 3 * 10 + n_sub * 9,
        "imagenet": len(IMN_MODELS) * len(IMN_CONDS) * 2 * len(IMN_CELLS) + n_sub * len(ROIS) * len(IMN_CELLS),
        "resolution": len(RES_PX) * 10 * 2 + n_sub,
        "ceiling": 2 * (2 * n_sub) * 4,
    }
    est = {k: v * t_rank * 1001 / workers for k, v in rank_per_draw.items()}
    est["imagenet extraction"] = t_imn * N_IMG * len(IMN_CONDS)
    est["resolution extraction"] = 1385 / 30 * 10 * N_IMG / 720       # step 5 CPU log, 10 of 30 models
    est = {k: v * 1.5 for k, v in est.items()}
    total = sum(est.values()) / 3600
    for k, v in est.items():
        print(f"  {k:24s} {v / 60:7.1f} min")
    print(f"ranking {t_rank * 1e3:.0f} ms; ImageNet {t_imn * 1e3:.0f} ms per image (all 6 models); total x 1.5: {total:.2f} h")
    return total


def extract_all():
    import nsd_features as F
    ps = F.image_paths()
    if not all(p.exists() for p in ps):
        raise SystemExit("STOP: NSD images missing")
    t0 = time.time()
    imn_extract(ps, imn_dir())
    print(f"imagenet extraction {time.time() - t0:.0f} s", flush=True)
    t1 = time.time()
    res_extract(ps, res_dir())
    print(f"resolution extraction {time.time() - t1:.0f} s", flush=True)


def run_all(workers=8):
    OUT.mkdir(parents=True, exist_ok=True)
    C.require_free(cfg()["disk"]["min_free_gb"])
    t0 = time.time()
    times, rules = {}, []
    for name, fn in (("multiverse", run_multiverse), ("imagenet", run_imagenet),
                     ("resolution", run_resolution), ("ceiling", run_ceiling)):
        t = time.time()
        r = fn(workers)
        if r:
            rules += r
        times[name] = time.time() - t
        print(f"== {name} done in {times[name]:.0f} s", flush=True)
    summ = pd.DataFrame(rules + [dict(rule="runtime", quantity=f"{k} wall seconds", value=v) for k, v in times.items()]
                        + [dict(rule="runtime", quantity="total wall seconds", value=time.time() - t0)])
    stamp(summ).to_csv(OUT / "summary.csv", index=False)
    print(summ[["rule", "quantity", "value", "passed", "note"]].to_string(index=False))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if which == "estimate":
        h = estimate(w)
        if h > cfg()["runtime"]["stop_hours"]:
            raise SystemExit(f"STOP: estimate {h:.2f} h above {cfg()['runtime']['stop_hours']} h")
    elif which == "extract":
        extract_all()
    elif which == "run":
        run_all(w)
    elif which in ("multiverse", "imagenet", "resolution", "ceiling"):
        OUT.mkdir(parents=True, exist_ok=True)
        print(globals()[f"run_{which}"](w))
    else:
        raise SystemExit("usage: nsd_step43.py {estimate|extract|run|multiverse|imagenet|resolution|ceiling} [workers]")

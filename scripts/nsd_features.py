"""
nsd_features.py
===============
Step 42 (amendments/2026-11-17_step42-predictions.md): Conv1 features of the 515 NSD images.

The EXISTING extraction function scripts/pilot_features_modal.py::extract_pilot, built and
run on the CPU as in step 40 (scripts/step40_extract.py) with the same substitutions
(device, data roots, stimulus list, output folder, workers, no volume commit), plus one:
the image lookup by THINGS concept folder is replaced by an explicit list of file paths
  paths = [find_img(s) for s in stim]  ->  paths = [Path(p) for p in IMG_PATHS]
so that the NSD images (PNG, 425 x 425) are read from NSD_DIR/images. The transform is
unchanged (Resize(224), CenterCrop(224), ToTensor, Normalize), Conv1 after max-pool,
Random Weights and Backprop x 5 seeds, global average pooling (g = 1).

Usage:
  python scripts/nsd_features.py gate      # 20 THINGS Arm A images through the same builder,
                                           # downstream-equivalence gate of step 40
                                           # -> results/step42/feature_gate.csv (STOP on failure)
  python scripts/nsd_features.py images    # download the 515 shared PNGs to NSD_DIR/images
  python scripts/nsd_features.py extract   # features of the 515 images -> NSD_DIR/features
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads)

import ast
import sys
import time

import numpy as np
import pandas as pd

import paths

SRC = C.REPO / "scripts" / "pilot_features_modal.py"
RULES = {"Random Weights": "random", "Backprop": "backprop"}


def build(out_dir, img_paths, grids):
    text = SRC.read_text(encoding="utf-8")
    tree = ast.parse(text)
    consts, fn = {}, None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("SEEDS", "PILOT_RES"):
            consts[node.targets[0].id] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == "extract_pilot":
            fn = node
    assert fn is not None and set(consts) == {"SEEDS", "PILOT_RES"}
    body = ast.get_source_segment(text, fn)
    subs = [
        ('dev = torch.device("cuda")', 'dev = torch.device("cpu")', 1),
        ('Path("/data/outputs_720")', f'Path(r"{paths.outputs_720_dir()}")', 1),
        ('Path("/data/object_images")', f'Path(r"{paths.things_images_dir()}")', 1),
        ('Path("/results/outputs_allseeds/checkpoints")', f'Path(r"{paths.ckpt_v12_dir()}")', 1),
        ('Path("/results/pilot_features")', f'Path(r"{out_dir}")', 1),
        ('stim = [l.strip() for l in open(FMRI / "stim_order_sub-01.txt") if l.strip()]',
         'stim = list(STIM_LIST)', 1),
        ('paths = [find_img(s) for s in stim]', 'paths = [Path(p) for p in IMG_PATHS]', 1),
        ("num_workers=4", "num_workers=0", 2),
        ("results_vol.commit()", "None", 1),
    ]
    for old, new, n in subs:
        cnt = body.count(old)
        if cnt != n:
            raise SystemExit(f"STOP: substitution '{old}' found {cnt} times, expected {n}")
        body = body.replace(old, new)
    from pathlib import Path
    img_paths = [Path(p) for p in img_paths]
    ns = dict(consts, RULES=RULES, GRIDS=list(grids), STIM_LIST=[p.name for p in img_paths],
              IMG_PATHS=[str(p) for p in img_paths])
    exec(compile(body, str(SRC), "exec"), ns)
    return lambda: ns["extract_pilot"](grids=list(grids))


# ── gate on 20 THINGS images ───────────────────────────────────────────────
def gate_dir():
    d = paths.ridge_data_dir() / "step42_features" / "gate20"
    d.mkdir(parents=True, exist_ok=True)
    return d


def gate_images():
    """The 20 Arm A images of the step 40 gate (config_step40.yaml extraction_gate)."""
    import step40_extract
    idx, names = step40_extract.gate_images()
    root = paths.things_images_dir()
    out = []
    for n in names:
        hits = sorted(root.glob(f"*/{n}"))
        if len(hits) != 1:
            raise SystemExit(f"STOP: {n} resolves to {len(hits)} files")
        out.append(hits[0])
    return idx, out


def run_gate(extract=True):
    """Downstream-equivalence gate of step 40 with the features of this builder."""
    import step40_common as S40
    from config_hash import load_config
    from fit import fit_condition
    from ridge_dual import alpha_grid
    c = S40.cfg()["extraction_gate"]
    idx, img = gate_images()
    if extract:
        build(gate_dir(), img, grids=[1, 4])()
    ref_dir = paths.ridge_data_dir() / "pilot_features"
    a = np.load(paths.ridge_data_dir() / "arm_b_out" / f"{c['subject']}_arma.npz")
    b = np.load(paths.ridge_data_dir() / "arm_b_out" / f"{c['subject']}_armb.npz")
    v1 = a["local_V1"]
    Y = a["betas"][:, v1].astype(np.float64)
    rel = b["reliability"][v1] > 0.2
    ag = load_config()["ridge"]["alpha_grid"]
    alphas = alpha_grid(ag["min"], ag["max"], ag["n_points"])
    rows = []
    for key in RULES.values():
        for si in range(5):
            name = f"model_{key}_seed{si}_g{c['grid']}.npy"
            Xm = np.load(ref_dir / name).astype(np.float64)
            Xl_rows = np.load(gate_dir() / name).astype(np.float64)
            Xl = Xm.copy()
            Xl[idx] = Xl_rows
            el = float(np.abs(Xm[idx] - Xl_rows).max())
            r = {}
            for tag, X in (("modal", Xm), ("local", Xl)):
                res = fit_condition(X, Y, np.arange(len(Y)), alphas=alphas, reliable=rel,
                                    assert_low_boundary=False, context=f"nsd gate {key} s{si}")
                r[tag] = float(np.median(res["r"][rel]))
            rows.append(dict(kind="checkpoint", condition=key, seed=si, median_r_modal=r["modal"],
                             median_r_local=r["local"], delta=r["local"] - r["modal"],
                             max_abs_feature_dev=el))
    df = pd.DataFrame(rows)
    piv = df.pivot(index="seed", columns="condition", values="delta")
    d_con = float((piv["random"] - piv["backprop"]).mean())
    thr = c["threshold_median_r"]
    ok = bool(df.delta.abs().max() <= thr and abs(d_con) <= thr)
    df = pd.concat([df, pd.DataFrame([dict(kind="gate", delta=d_con, passed=ok,
                                           note=f"max |delta median r| {df.delta.abs().max():.2e}; "
                                                f"|delta (Random - BP)| {abs(d_con):.2e}")])], ignore_index=True)
    C.OUT.mkdir(parents=True, exist_ok=True)
    C.stamp(df).to_csv(C.OUT / "feature_gate.csv", index=False)
    print(df.drop(columns=["config_hash", "git_head"]).to_string(index=False))
    return ok


# ── NSD images and features ────────────────────────────────────────────────
def image_paths(subject=None):
    d = C.design(subject or C.cfg()["pilot_subject"])
    root = C.nsd_dir("images")
    return [root / f"shared{s:04d}_nsd{i:05d}.png" for s, i in zip(d["shared"], d["images"])]


def fetch_images():
    ps = image_paths()
    for p in ps:
        if not p.exists():
            C.download(f"nsddata/stimuli/nsd/shared1000/{p.name}", p)
    from PIL import Image
    sizes = {Image.open(p).size for p in ps}
    modes = {Image.open(p).mode for p in ps}
    print(f"{len(ps)} images; sizes {sizes}; modes {modes}")


def feature_dir():
    return C.nsd_dir("features")


def extract():
    ps = image_paths()
    if not all(p.exists() for p in ps):
        raise SystemExit("STOP: images missing; run 'images' first")
    build(feature_dir(), ps, grids=[1])()
    n = 0
    for key in RULES.values():
        for si in C.cfg()["seed_idx"]:
            X = np.load(feature_dir() / f"model_{key}_seed{si}_g1.npy")
            assert X.shape[0] == len(ps) and np.isfinite(X).all()
            n += 1
    print(f"{n} models x {len(ps)} images extracted; units {X.shape[1]}")


if __name__ == "__main__":
    t0 = time.time()
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "gate":
        if not run_gate():
            raise SystemExit("STOP: feature gate failed")
    elif which == "images":
        fetch_images()
    elif which == "extract":
        extract()
    else:
        raise SystemExit("usage: nsd_features.py {gate|images|extract}")
    print(f"done in {time.time() - t0:.0f} s")

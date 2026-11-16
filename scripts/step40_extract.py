"""
step40_extract.py
=================
Step 40, part C (amendments/2026-11-16_step40-predictions.md). EXPLORATORY.

Conv1 features (after max-pooling, 224 px NATIVE) of the Random Weights and Backprop
checkpoints (5 seeds each) for image sets other than the 720 Arm A stimuli, with the
EXISTING extraction code: scripts/pilot_features_modal.py::extract_pilot, built and run on
the CPU exactly as scripts/prereg_extract.py does, plus two substitutions:
  - the stimulus list:  stim = [... stim_order_sub-01.txt ...]  ->  stim = list(STIM_LIST)
  - the output folder:  RIDGE_DATA_DIR/step40_features/<set>  (never inside the repository)
and RULES restricted to Random Weights and Backprop. Grids g = 1 (global average pooling,
used for the RSA) and g = 4 (used by the gate).

Usage:
  python scripts/step40_extract.py gate   # 20 Arm A images, then the reproduction gate
  python scripts/step40_extract.py test   # the 100 repeated test images

Reproduction gate (STOP on failure), the downstream-equivalence gate of
amendments/2026-10-13_g2-local-extraction.md applied to 20 images: the 20 locally extracted
g = 4 rows replace the same rows of the Modal features in pilot_features; ridge at g = 4,
sub-01, unchanged fit configuration (config.yaml grid), all 10 checkpoints, fitted once with
each version. Pass: |delta median r| <= 1e-4 for every checkpoint and |delta (Random - BP)|
<= 1e-4 (mean over seeds). Element-wise deviations are reported, not gated.

Output: results/step40/extraction_gate.csv
"""
import step40_common as S     # noqa: I001  (pins BLAS threads)

import ast
import sys
import time

import numpy as np
import pandas as pd

import paths
from config_hash import load_config
from fit import fit_condition
from ridge_dual import alpha_grid

SRC = S.REPO / "scripts" / "pilot_features_modal.py"
RULES = {"Random Weights": "random", "Backprop": "backprop"}


def build(out_dir, stim_list):
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
        ("num_workers=4", "num_workers=0", 2),
        ("results_vol.commit()", "None", 1),
    ]
    for old, new, n in subs:
        c = body.count(old)
        if c != n:
            raise SystemExit(f"STOP: substitution '{old}' found {c} times, expected {n}")
        body = body.replace(old, new)
    ns = dict(consts, RULES=RULES, GRIDS=[1, 4], STIM_LIST=list(stim_list))
    exec(compile(body, str(SRC), "exec"), ns)
    return ns["extract_pilot"]


def gate_images():
    c = S.cfg()["extraction_gate"]
    order = [l.strip() for l in open(paths.outputs_720_dir() / "stim_order_sub-01.txt") if l.strip()]
    idx = np.sort(np.random.default_rng(c["image_seed"]).choice(len(order), c["n_images"], replace=False))
    return idx, [order[i] for i in idx]


def run_gate():
    c = S.cfg()["extraction_gate"]
    idx, _ = gate_images()
    ref_dir = paths.ridge_data_dir() / "pilot_features"
    loc_dir = S.feature_dir() / "gate20"
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
            Xl_rows = np.load(loc_dir / name).astype(np.float64)
            Xl = Xm.copy()
            Xl[idx] = Xl_rows
            el = float(np.abs(Xm[idx] - Xl_rows).max())
            rel_el = el / float(np.abs(Xm[idx]).max())
            r = {}
            for tag, X in (("modal", Xm), ("local", Xl)):
                res = fit_condition(X, Y, np.arange(len(Y)), alphas=alphas, reliable=rel,
                                    assert_low_boundary=False, context=f"gate {key} s{si}")
                r[tag] = float(np.median(res["r"][rel]))
            rows.append(dict(kind="checkpoint", condition=key, seed=si, median_r_modal=r["modal"],
                             median_r_local=r["local"], delta=r["local"] - r["modal"],
                             max_abs_feature_dev=el, max_rel_feature_dev=rel_el))
    df = pd.DataFrame(rows)
    piv = df.pivot(index="seed", columns="condition", values="delta")
    d_contrast = float((piv["random"] - piv["backprop"]).mean())
    ok = bool(df.delta.abs().max() <= c["threshold_median_r"] and abs(d_contrast) <= c["threshold_median_r"])
    df = pd.concat([df, pd.DataFrame([dict(kind="gate", delta=d_contrast, passed=ok,
                                           note=f"max |delta median r| {df.delta.abs().max():.2e}; "
                                                f"|delta (Random - BP)| {abs(d_contrast):.2e}")])],
                   ignore_index=True)
    df["seed"] = df["seed"].astype("Int64")
    S.OUT.mkdir(parents=True, exist_ok=True)
    S.stamp(df).to_csv(S.OUT / "extraction_gate.csv", index=False)
    print(df.to_string(index=False))
    if not ok:
        raise SystemExit("STOP: extraction reproduction gate failed")


def main(which):
    t0 = time.time()
    if which == "gate":
        _, names = gate_images()
        build(S.feature_dir() / "gate20", names)(grids=[1, 4])
        print(f"extraction {time.time() - t0:.0f}s", flush=True)
        run_gate()
    elif which == "test":
        build(S.feature_dir() / "test100", S.test_stimuli())(grids=[1, 4])
    else:
        raise SystemExit("usage: step40_extract.py {gate|test}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")

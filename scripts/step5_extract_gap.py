"""
step5_extract_gap.py
====================
GAP features for step 5 (amendments/2026-10-08_step5-resolution-predictions.md),
local CPU. EXPLORATORY.

Exactly the paper's extraction (code/upsampling/step1_extract.py): v12's own
`extract_features` (model.eval() + assert_bn_eval + get_features, batches of 128) and
`_tf`, lifted unchanged via src/vendor/v12_lift.py; the transform is applied once per
(arm, px) and the tensor stack reused for all 30 models. get_features returns, per
image, the channel means of Conv1, Conv2, Conv3 (x.mean([2, 3])) and the FC1 vector:
these are stored, no full maps. Layer -> ROI (v12 LAYER_ROI_FIXED): Conv1 -> V1, V2;
Conv2 -> V1, V2; Conv3 -> LOC; FC1 -> IT. All four are stored at no extra cost.

Conditions: NATIVE at 32, 64, 96, 128, 160, 224 px, and UPSAMPLED at 224 px; all 30
checkpoints (CKPT_V12_DIR).

Runtime estimate before start, from the paper's own CPU extraction log
(data/upsampling_extract.log): NATIVE, all six resolutions, 1385 s; UPSAMPLED 224 px,
~448 s. Times 1.5. Above 3 h only NATIVE 32, NATIVE 224 and UPSAMPLED 224 are run.

Output: STEP5_GAP_DIR/{arm}/res{px}/seed_{i}/gap_{rule_key}_{layer}.npy (float32,
720 x C). Resumable.

Gate (STOP on failure): compute_rdm (raw) of the new features against the stored paper
RDM triangles (evaluation-resolution-rsa data/upsampling/rdms) for every cell computed
here: Pearson >= 0.999.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "vendor"))
sys.path.insert(0, str(REPO / "src" / "ridge"))

import paths                                                        # noqa: E402
from common_up_pairs import stim_order                              # noqa: E402
from v12_lift import (RULES, SEEDS, LAYERS, TRI, RES, rule_key,     # noqa: E402
                      namespaces, transform, load_v12_model)

EST_S = {("NATIVE", "all6"): 1385, ("UPSAMPLED", 224): 448}
PAPER_RDMS = paths.ridge_data_dir() / "data" / "upsampling" / "rdms"


def out_path(arm, px, si, rule, layer):
    return paths.step5_gap_dir() / arm / f"res{px}" / f"seed_{si}" / f"gap_{rule_key(rule)}_{layer}.npy"


def conditions():
    est = (EST_S[("NATIVE", "all6")] + EST_S[("UPSAMPLED", 224)]) * 1.5
    print(f"runtime estimate: {est / 60:.1f} min (paper CPU log x 1.5)", flush=True)
    if est > 3 * 3600:
        print("estimate above 3 h: NATIVE 32, NATIVE 224, UPSAMPLED 224 only")
        return [("NATIVE", 32), ("NATIVE", 224), ("UPSAMPLED", 224)]
    return [("NATIVE", px) for px in RES] + [("UPSAMPLED", 224)]


def extract(conds):
    torch.set_num_threads(12)
    ns12, ns11 = namespaces()
    from PIL import Image
    imgs = [ns12["find_img"](s) for s in stim_order()]
    assert all(p is not None for p in imgs)
    t0 = time.time()
    for arm, px in conds:
        todo = [(r, si) for r in RULES for si in range(len(SEEDS))
                if not all(out_path(arm, px, si, r, l).exists() for l in LAYERS)]
        if not todo:
            continue
        tf = transform(ns12, ns11, arm, px)
        X = torch.stack([tf(Image.open(p).convert("RGB")) for p in imgs])
        print(f"[{time.time() - t0:6.0f}s] {arm} {px}px: {tuple(X.shape)}, {len(todo)} models", flush=True)
        for rule, si in todo:
            torch.manual_seed(SEEDS[si])
            m = load_v12_model(ns12, rule, si)
            feats = ns12["extract_features"](m, list(range(len(imgs))), X)
            for layer, f in zip(LAYERS, feats):
                p = out_path(arm, px, si, rule, layer)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, np.asarray(f, np.float32))
        del X
    print(f"extraction done {time.time() - t0:.0f}s", flush=True)
    return ns12


def gate(ns12, conds):
    worst, n = 1.0, 0
    for arm, px in conds:
        for rule in RULES:
            for si in range(len(SEEDS)):
                for layer in LAYERS:
                    ref = PAPER_RDMS / arm / f"res{px}" / f"seed_{si}" / f"rdm_{rule_key(rule)}_{layer}.npy"
                    if not ref.exists():
                        continue
                    mine = ns12["compute_rdm"](np.load(out_path(arm, px, si, rule, layer)))[TRI]
                    r = float(np.corrcoef(mine, np.load(ref))[0, 1])
                    worst, n = min(worst, r), n + 1
                    if not r >= 0.999:
                        raise SystemExit(f"STOP: {arm} {px} {rule} s{si} {layer}: Pearson {r:.6f}")
    print(f"gate: {n} cells compared, min Pearson {worst:.8f}")


def main():
    conds = conditions()
    ns12 = extract(conds)
    gate(ns12, conds)


if __name__ == "__main__":
    main()

"""
step6_extract_bandpass.py
=========================
Band-pass GAP features for P_mech2 (amendments/2026-10-09_step6-predictions.md).
EXPLORATORY, local CPU.

Inputs: NATIVE 224, conditions only_0-14 and notch_0-14, built exactly as
code/bandpass/bandpass_extract.py does (vendored sigma / bands / condition, same
pre-transform Resize(224) -> CenterCrop(224) -> ToTensor, filter, then Normalize with the
CIFAR-10 statistics; check: Normalize(pre(img)) == v12 NATIVE _tf(224)). All 30 v12
checkpoints, v12 extract_features (lifted unchanged). All four GAP layers are stored
(Conv1 -> V1 and Conv3 -> LOC are the ones used).

Output: STEP5_GAP_DIR/bandpass/{cond}/seed_{i}/gap_{rule_key}_{layer}.npy (float32).

Gate (STOP on failure): raw Conv1 RDMs (v12 compute_rdm) against
evaluation-resolution-rsa data/bandpass/rdms for the same conditions and seeds
(Random Weights and Backprop, which are the ones stored there): Pearson >= 0.999.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "vendor"))
sys.path.insert(0, str(REPO / "src" / "ridge"))

import paths                                                         # noqa: E402
from bandpass_filters import PX, bands, condition                    # noqa: E402
from common_up_pairs import stim_order                               # noqa: E402
from v12_lift import RULES, SEEDS, LAYERS, TRI, rule_key, namespaces, transform, load_v12_model  # noqa: E402

CONDS = ["only_0-14", "notch_0-14"]
PAPER = paths.ridge_data_dir() / "data" / "bandpass" / "rdms"


def out_path(cond, si, rule, layer):
    return paths.step5_gap_dir() / "bandpass" / cond / f"seed_{si}" / f"gap_{rule_key(rule)}_{layer}.npy"


def main():
    torch.set_num_threads(12)
    import torchvision.transforms as T
    from PIL import Image
    ns12, ns11 = namespaces()
    imgs = [Image.open(ns12["find_img"](s)).convert("RGB") for s in stim_order()]
    pre = T.Compose([T.Resize(PX), T.CenterCrop(PX), T.ToTensor()])
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    nat = transform(ns12, ns11, "NATIVE", PX)
    X0 = np.stack([pre(im).numpy() for im in imgs]).astype(np.float32)
    assert torch.equal(norm(torch.from_numpy(X0[0])), nat(imgs[0]))
    print(f"runtime estimate: 2 conditions x ~448 s (paper CPU log, 224 px) x 1.5 = {2 * 448 * 1.5 / 60:.0f} min", flush=True)
    t0 = time.time()
    for cond in CONDS:
        X = norm(torch.from_numpy(np.stack([condition(x, bands(x), cond) for x in X0])))
        print(f"[{time.time() - t0:5.0f}s] {cond}", flush=True)
        for rule in RULES:
            for si in range(len(SEEDS)):
                if all(out_path(cond, si, rule, l).exists() for l in LAYERS):
                    continue
                torch.manual_seed(SEEDS[si])
                m = load_v12_model(ns12, rule, si)
                feats = ns12["extract_features"](m, list(range(len(imgs))), X)
                for layer, f in zip(LAYERS, feats):
                    p = out_path(cond, si, rule, layer)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.save(p, np.asarray(f, np.float32))
    worst, n = 1.0, 0
    for cond in CONDS:
        for rule in RULES:
            for si in range(len(SEEDS)):
                ref = PAPER / cond / f"seed_{si}" / f"rdm_{rule.lower().replace(' ', '_')}_Conv1.npy"
                if not ref.exists():
                    continue
                r = float(np.corrcoef(ns12["compute_rdm"](np.load(out_path(cond, si, rule, "Conv1")))[TRI], np.load(ref))[0, 1])
                worst, n = min(worst, r), n + 1
                if not r >= 0.999:
                    raise SystemExit(f"STOP: {cond} {rule} s{si}: Pearson {r:.6f}")
    print(f"gate: {n} cells, min Pearson {worst:.8f}; done {time.time() - t0:.0f}s")
    if n == 0:
        raise SystemExit("STOP: no reference RDMs found")


if __name__ == "__main__":
    main()

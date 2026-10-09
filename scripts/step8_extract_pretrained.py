"""
step8_extract_pretrained.py
===========================
GAP features of six torchvision models for step 8
(amendments/2026-10-09_step8-generalization-predictions.md). EXPLORATORY, local CPU.

Models: alexnet, vgg11_bn, resnet18, resnet50, convnext_tiny, vit_b_16.
Conditions: trained (IMAGENET1K_V1) and untrained s0, s1, s2 (torch.manual_seed(s)
immediately before construction, weights=None, BN statistics left at defaults). Eval mode.
Layers (fixed; STOP if one does not exist, never substituted):
  V1 : alexnet features[1]; vgg11_bn features[2]; resnet18/50 maxpool;
       convnext_tiny features[0]; vit_b_16 encoder.layers[0]
  LOC: alexnet features[12]; vgg11_bn last MaxPool2d in features; resnet18/50 layer4;
       convnext_tiny features[7]; vit_b_16 encoder.layers[-1]
Pooling: mean over the spatial dimensions; for ViT mean over tokens without the class
token. Forward hooks clone the module output (alexnet/vgg11_bn V1 layers are in-place ReLUs).
Input: RGB, Resize(R) of the shorter side, CenterCrop(R), ToTensor, ImageNet Normalize.
R in {112, 224, 448}; vit_b_16 at R = 224 only (operationalisation note 1).

Runtime estimate before the main loop (measured per model at R = 224 on 16 images,
scaled by (R/224)^2, x 1.5): above 6 h R = 448 is dropped; if still above 6 h,
vit_b_16 is dropped too.

Sanity check (no criterion): top-1 ImageNet class of the trained models on the first 20
stimuli of the stimulus order at R = 224 -> results/step8/sanity_top1.csv.

Output: data/step8_gap/{model}/{condition}/R{R}/{V1,LOC}.npy (float32, 720 x C).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models as M
import torchvision.transforms as T

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "vendor"))
sys.path.insert(0, str(REPO / "src" / "ridge"))

from common_up_pairs import stim_order          # noqa: E402
from config_hash import config_hash             # noqa: E402

MODELS = ["alexnet", "vgg11_bn", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
CONDS = ["trained", "untrained_s0", "untrained_s1", "untrained_s2"]
RES = [112, 224, 448]
OUT = REPO / "data" / "step8_gap"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build(name, cond):
    if cond == "trained":
        m = getattr(M, name)(weights=M.get_model_weights(name).IMAGENET1K_V1)
    else:
        torch.manual_seed(int(cond[-1]))
        m = getattr(M, name)(weights=None)
    return m.eval()


def layers(name, m):
    try:
        if name == "alexnet":
            v1, loc = m.features[1], m.features[12]
        elif name == "vgg11_bn":
            pools = [l for l in m.features if isinstance(l, torch.nn.MaxPool2d)]
            v1, loc = m.features[2], pools[-1]
        elif name in ("resnet18", "resnet50"):
            v1, loc = m.maxpool, m.layer4
        elif name == "convnext_tiny":
            v1, loc = m.features[0], m.features[7]
        elif name == "vit_b_16":
            v1, loc = m.encoder.layers[0], m.encoder.layers[-1]
        else:
            raise KeyError(name)
    except (AttributeError, IndexError, KeyError) as e:
        raise SystemExit(f"STOP: layer for {name} not found ({e})")
    return {"V1": v1, "LOC": loc}


def pool(name, t):
    if name == "vit_b_16":
        return t[:, 1:, :].mean(dim=1)
    return t.mean(dim=[2, 3])


def resolutions(name, res):
    return [224] if name == "vit_b_16" else list(res)


def tf(R):
    return T.Compose([T.Resize(R), T.CenterCrop(R), T.ToTensor(), T.Normalize(MEAN, STD)])


def images():
    from PIL import Image
    from v12_lift import namespaces
    ns12, _ = namespaces()
    paths = [ns12["find_img"](s) for s in stim_order()]
    assert all(p is not None for p in paths)
    return [Image.open(p).convert("RGB") for p in paths], stim_order()


def extract(name, m, X, batch=32):
    got = {}
    hs = [mod.register_forward_hook(lambda _m, _i, o, k=k: got.__setitem__(k, o.detach().clone()))
          for k, mod in layers(name, m).items()]
    out = {"V1": [], "LOC": []}
    with torch.no_grad():
        for i in range(0, len(X), batch):
            m(X[i:i + batch])
            for k in out:
                out[k].append(pool(name, got[k]).float().numpy())
    for h in hs:
        h.remove()
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    torch.set_num_threads(os.cpu_count())
    imgs, names = images()
    for name in MODELS:                                  # layer check before anything else
        layers(name, build(name, "untrained_s0"))
    print("layers: all present", flush=True)

    X224 = torch.stack([tf(224)(im) for im in imgs[:16]])
    per_img = {}
    for name in MODELS:
        m = build(name, "trained")
        t0 = time.time()
        extract(name, m, X224, batch=16)
        per_img[name] = (time.time() - t0) / 16
    res, models = list(RES), list(MODELS)

    def estimate():
        return sum(per_img[n] * (R / 224) ** 2 * 720 * len(CONDS) for n in models for R in resolutions(n, res)) * 1.5 / 3600
    print(f"runtime estimate: {estimate():.2f} h", flush=True)
    if estimate() > 6:
        res = [112, 224]
        print(f"above 6 h: R = 448 dropped (GEN4 not testable); new estimate {estimate():.2f} h")
        if estimate() > 6:
            models.remove("vit_b_16")
            print(f"still above 6 h: vit_b_16 dropped; new estimate {estimate():.2f} h")

    # sanity check
    X = torch.stack([tf(224)(im) for im in imgs[:20]])
    rows = []
    for name in models:
        m = build(name, "trained")
        cats = M.get_model_weights(name).IMAGENET1K_V1.meta["categories"]
        with torch.no_grad():
            top = m(X).argmax(dim=1).tolist()
        rows += [dict(model=name, stimulus=names[i], top1_index=t, top1_label=cats[t]) for i, t in enumerate(top)]
    sd = pd.DataFrame(rows)
    sd["config_hash"] = config_hash()
    sd["git_head"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
    (REPO / "results" / "step8").mkdir(parents=True, exist_ok=True)
    sd.to_csv(REPO / "results" / "step8" / "sanity_top1.csv", index=False)
    print(sd.pivot(index="stimulus", columns="model", values="top1_label").to_string(), flush=True)

    t0 = time.time()
    for R in res:
        XR = torch.stack([tf(R)(im) for im in imgs])
        for name in models:
            if R not in resolutions(name, res):
                continue
            for cond in CONDS:
                d = OUT / name / cond / f"R{R}"
                if all((d / f"{k}.npy").exists() for k in ("V1", "LOC")):
                    continue
                feats = extract(name, build(name, cond), XR)
                d.mkdir(parents=True, exist_ok=True)
                for k, v in feats.items():
                    assert v.shape[0] == 720 and np.isfinite(v).all(), (name, cond, R, k)
                    np.save(d / f"{k}.npy", v.astype(np.float32))
            print(f"[{time.time() - t0:6.0f}s] R={R} {name} done", flush=True)
        del XR
    print("extraction done", flush=True)


if __name__ == "__main__":
    main()

"""
pilot_features_modal.py
=======================
Phase 3 pilot feature extraction: 224 px, Conv1, g in {1, 4}, all 30 checkpoints,
plus the control band. Saves float32 features to the results volume; fitting runs on
CPU from those arrays.

CONTROL BAND (config.yaml control_band). Every control goes through the IDENTICAL
readout at the IDENTICAL g, so the only thing that differs is the feature bank:

  pixels              3 channels  -> 3*g^2    raw RGB, same resize/crop as the models
  luminance+contrast  2 channels  -> 2*g^2    Rec.709 luminance + local RMS contrast
  gabor              12 channels  -> 12*g^2   4 orientations x 3 spatial frequencies,
                                              energy: the dimensionality-matched
                                              low-level bank

Under a FITTED readout a single luminance scalar is no longer the right trivial
baseline -- one feature can gain nothing from ridge except scaling -- which is why the
band is banks rather than scalars.

Two controls need no extraction and are applied at fit time:
  * permuted stimulus<->feature assignment (a permutation of the pairing);
  * the per-image cell shuffle, built from the g=4 features.
"""

import modal

app = modal.App("ridge-pilot-features")

data_vol = modal.Volume.from_name("burstprop-data", create_if_missing=True)
results_vol = modal.Volume.from_name("learning-rules-rsa", create_if_missing=True)

import sys as _sys
from pathlib import Path as _Path

REPO = str(_Path(__file__).resolve().parents[1])
_sys.path.insert(0, REPO + "/src/vendor")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "numpy", "scipy", "pandas", "Pillow", "pyyaml")
    .add_local_python_source("bn_guard")
    .add_local_dir(REPO + "/src/ridge", remote_path="/root/ridge")
    .add_local_file(REPO + "/src/vendor/models_v2.py",
                    remote_path="/root/ridge/models_v2.py")
)

RULES = {
    "Random Weights": "random",
    "Backprop": "backprop",
    "Feedback Alignment": "feedback_alignment",
    "Predictive Coding": "predictive_coding",
    "STDP": "stdp",
    "Random Weights (BN-calibrated)": "random_bn_calibrated",
}
SEEDS = [42, 123, 456, 789, 1337]
PILOT_RES = 224
GRIDS = [1, 4]


@app.function(image=image, gpu="T4", timeout=60 * 60 * 4,
              volumes={"/data": data_vol, "/results": results_vol})
def extract_pilot(grids=None):
    import sys
    sys.path.insert(0, "/root/ridge")
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from gridpool import pool_grid, assert_g1_is_original, assert_float32
    from models_v2 import load_model
    from bn_guard import assert_bn_eval

    global GRIDS
    GRIDS = list(grids) if grids else GRIDS
    dev = torch.device("cuda")
    FMRI = Path("/data/outputs_720")
    THINGS = Path("/data/object_images")
    CKPT = Path("/results/outputs_allseeds/checkpoints")
    OUT = Path("/results/pilot_features")
    OUT.mkdir(parents=True, exist_ok=True)

    assert_g1_is_original()

    stim = [l.strip() for l in open(FMRI / "stim_order_sub-01.txt") if l.strip()]

    def find_img(s):
        name = s.replace(".jpg", "")
        parts = name.split("_")
        last = parts[-1]
        concept = "_".join(parts[:-1]) if (
            len(parts) > 1 and len(last) <= 4 and any(c.isdigit() for c in last)) else name
        for pat in [concept + "/" + name + ".jpg", concept + "/*.jpg"]:
            h = sorted(THINGS.glob(pat))
            if h:
                return h[0]
        return None

    paths = [find_img(s) for s in stim]
    bad = [s for s, p in zip(stim, paths) if p is None or p.name != s]
    if bad:
        raise RuntimeError(
            str(len(bad)) + " stimuli did not resolve by exact filename: " + str(bad[:3]))
    print("stimuli: " + str(len(paths)) + ", all exact-filename matches", flush=True)

    tf = T.Compose([T.Resize(PILOT_RES), T.CenterCrop(PILOT_RES), T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    tf_raw = T.Compose([T.Resize(PILOT_RES), T.CenterCrop(PILOT_RES), T.ToTensor()])

    class DS(Dataset):
        def __init__(self, ps, t):
            self.ps, self.t = ps, t

        def __len__(self):
            return len(self.ps)

        def __getitem__(self, i):
            return self.t(Image.open(self.ps[i]).convert("RGB")), i

    def conv1_map(model, x):
        cls = type(model).__name__
        if cls in ("BP_CNN", "FA_CNN", "Random_CNN"):
            return model.conv1(x)
        if cls == "PC_CNN":
            # The inference loop's r1, NOT a feedforward pass (models_v2.py:290-294).
            r1, _, _ = model.infer(x)
            return r1
        if cls == "STDP_CNN":
            return model.pool(F.relu(model.bn1(
                model.L1.forward(x, False, x.mean([2, 3])))))
        raise RuntimeError("no Conv1 extractor for " + cls)

    def save(name, feats_by_g):
        for g, arr in feats_by_g.items():
            assert_float32(arr, name + " g=" + str(g))
            np.save(OUT / (name + "_g" + str(g) + ".npy"), arr.numpy())
        print("  saved " + name + ": " + ", ".join(
            "g" + str(g) + "=" + str(tuple(a.shape)) for g, a in feats_by_g.items()),
            flush=True)

    for rule_name, key in RULES.items():
        for si in range(len(SEEDS)):
            model = load_model(key, weights_dir=CKPT, seed_idx=si)
            model = model.to(dev) if hasattr(model, "to") else model.to_device(dev)
            model.eval()
            assert_bn_eval(model, context="pilot/" + key)
            loader = DataLoader(DS(paths, tf), batch_size=64, shuffle=False, num_workers=4)
            acc = {g: [] for g in GRIDS}
            with torch.no_grad():
                for imgs, _ in loader:
                    m = conv1_map(model, imgs.to(dev))
                    for g in GRIDS:
                        acc[g].append(pool_grid(m, g).cpu())
            save("model_" + key + "_seed" + str(si),
                 {g: torch.cat(v) for g, v in acc.items()})
            del model
            torch.cuda.empty_cache()

    # ── control band ─────────────────────────────────────────────────────────
    def gabor_bank():
        ks = 21
        banks = []
        yy, xx = torch.meshgrid(torch.arange(ks) - ks // 2,
                                torch.arange(ks) - ks // 2, indexing="ij")
        xx = xx.float()
        yy = yy.float()
        for sf in (0.10, 0.20, 0.40):
            for th in (0.0, 45.0, 90.0, 135.0):
                t = np.deg2rad(th)
                xr = xx * np.cos(t) + yy * np.sin(t)
                yr = -xx * np.sin(t) + yy * np.cos(t)
                sig = 1.0 / (3.0 * sf)
                env = torch.exp(-(xr ** 2 + yr ** 2) / (2 * sig ** 2))
                banks.append((env * torch.cos(2 * np.pi * sf * xr)).float())
        return torch.stack(banks).unsqueeze(1)

    gb = gabor_bank().to(dev)
    loader = DataLoader(DS(paths, tf_raw), batch_size=64, shuffle=False, num_workers=4)
    acc = {n: {g: [] for g in GRIDS} for n in ("pixels", "lumcontrast", "gabor")}
    with torch.no_grad():
        for imgs, _ in loader:
            x = imgs.to(dev)
            lum = (0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]).unsqueeze(1)
            mu = F.avg_pool2d(lum, 9, stride=1, padding=4)
            mu2 = F.avg_pool2d(lum ** 2, 9, stride=1, padding=4)
            con = (mu2 - mu ** 2).clamp(min=0).sqrt()
            gab = F.conv2d(lum, gb, padding=gb.shape[-1] // 2).abs()
            for g in GRIDS:
                acc["pixels"][g].append(pool_grid(x, g).cpu())
                acc["lumcontrast"][g].append(pool_grid(torch.cat([lum, con], 1), g).cpu())
                acc["gabor"][g].append(pool_grid(gab, g).cpu())
    for n, d in acc.items():
        save("control_" + n, {g: torch.cat(v) for g, v in d.items()})

    results_vol.commit()
    files = sorted(p.name for p in OUT.glob("*.npy"))
    print("\nPILOT FEATURES COMPLETE: " + str(len(files)) + " arrays", flush=True)
    return {"n_files": len(files), "res": PILOT_RES, "grids": GRIDS}

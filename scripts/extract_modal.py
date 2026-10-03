"""
extract_modal.py
================
Conv1 feature extraction with the grid ladder, plus the BRIDGE CONTROL.

THE CORRECTNESS RISK THIS FILE CARRIES
--------------------------------------
Every other rule reads its Conv1 map off a single `nn.Sequential`. Predictive coding
does not: its features are the representations produced by the T=10 inference loop, not
a feedforward pass, and the paper says explicitly that using the feedforward pass
instead gives a different model. STDP is a plain class whose first block is assembled by
hand. So `conv1_map()` mirrors each class's own `get_features()` up to the point just
before pooling, per class, rather than assuming a common structure.

That is exactly what the bridge control checks. At g = 1 the pooled map must reproduce
the evaluation this project already ran on THE SAME WEIGHTS, so any error in
`conv1_map()` -- a missing ReLU, a feedforward PC path, a wrong BN -- shows up as a
mismatch rather than as a plausible-looking number.

BRIDGE CONTROL, precisely
-------------------------
Target: outputs_allseeds/rsa_resolution_sweep.csv, produced by the ORIGINAL evaluation
path (learning_rules_v10's run_rsa / compute_rdm / rsa_score) on these same 30
checkpoints. Comparing against that rather than against the published bnfix_sweep.csv
is deliberate: identical weights on both sides, so the comparison isolates the new
extraction code. Feature extraction involves no convolution backward, so it is
deterministic and the requirement is EXACT agreement for all six rules -- not the ~1e-3
tolerance the trained-weight comparison needed.

g = 1 pooling goes through gridpool.pool_grid, which calls the paper's own
`x.mean([2, 3])` expression verbatim (asserted at runtime by assert_g1_is_original).
"""

import modal

app = modal.App("ridge-extract")

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
    .add_local_dir(
        REPO + "/src/ridge",
        remote_path="/root/ridge",
    )
    .add_local_file(
        REPO + "/src/vendor/models_v2.py",
        remote_path="/root/ridge/models_v2.py",
    )
)

SUBJECTS = ["sub-01", "sub-02", "sub-03"]
RESOLUTIONS = [32, 64, 96, 128, 160, 224]
GRIDS_A = [1, 2, 4, 8]
RULES = {
    "Random Weights": "random",
    "Backprop": "backprop",
    "Feedback Alignment": "feedback_alignment",
    "Predictive Coding": "predictive_coding",
    "STDP": "stdp",
    "Random Weights (BN-calibrated)": "random_bn_calibrated",
}
SEEDS = [42, 123, 456, 789, 1337]


@app.function(image=image, gpu="T4", timeout=60 * 60 * 6,
              volumes={"/data": data_vol, "/results": results_vol})
def bridge_control(only_rule: str = None, only_seed: int = None):
    """Extract Conv1 at g=1 through the NEW path, rebuild the RDM and rho, and compare
    against the original evaluation on the same weights. Exact agreement required."""
    import sys
    sys.path.insert(0, "/root/ridge")

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
    from torch.utils.data import DataLoader, Dataset

    from gridpool import pool_grid, assert_g1_is_original, assert_float32
    from models_v2 import load_model
    from bn_guard import assert_bn_eval

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FMRI = Path("/data/outputs_720")
    THINGS = Path("/data/object_images")
    CKPT = Path("/results/outputs_allseeds/checkpoints")

    # The g=1 rung must BE the paper's expression, on this device.
    assert_g1_is_original()
    print("assert_g1_is_original: pool_grid(x,1) is bit-identical to x.mean([2,3])")

    # ── stimuli, exactly as the sweep resolves them ──────────────────────────
    stim = [l.strip() for l in open(FMRI / "stim_order_sub-01.txt") if l.strip()]

    def find_img(s):
        name = s.replace(".jpg", ""); parts = name.split("_"); last = parts[-1]
        concept = "_".join(parts[:-1]) if (len(parts) > 1 and len(last) <= 4
                                           and any(c.isdigit() for c in last)) else name
        for pat in [f"{concept}/{name}.jpg", f"{concept}/*.jpg"]:
            hits = sorted(THINGS.glob(pat))
            if hits:
                return hits[0]
        for folder in THINGS.iterdir():
            if folder.name.lower() == concept.lower():
                imgs = sorted(folder.glob("*.jpg"))
                if imgs:
                    return imgs[0]
        return None

    paths = [find_img(s) for s in stim]
    if any(p is None for p in paths):
        raise RuntimeError(f"{sum(p is None for p in paths)} stimuli did not resolve")
    # Phase 1 verified all 720 resolve by EXACT filename; assert it so a future data
    # move that silently substitutes a different exemplar cannot pass unnoticed.
    subs = [(s, p.name) for s, p in zip(stim, paths) if p.name != s]
    if subs:
        raise RuntimeError(f"{len(subs)} stimuli resolved to a DIFFERENT file, e.g. {subs[:3]}")
    print(f"stimuli: {len(paths)} resolved, all by exact filename")

    class DS(Dataset):
        def __init__(self, ps, t): self.ps, self.t = ps, t
        def __len__(self): return len(self.ps)
        def __getitem__(self, i): return self.t(Image.open(self.ps[i]).convert("RGB")), i

    def tf(px):
        return T.Compose([T.Resize(px), T.CenterCrop(px), T.ToTensor(),
                          T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # ── per-class Conv1 map, mirroring each get_features() ───────────────────
    def conv1_map(model, x):
        cls = type(model).__name__
        if cls in ("BP_CNN", "FA_CNN", "Random_CNN"):
            return model.conv1(x)
        if cls == "PC_CNN":
            # The inference loop's r1, NOT a feedforward pass. models_v2.py:290-294.
            r1, _, _ = model.infer(x)
            return r1
        if cls == "STDP_CNN":
            # models_v2.py:386, first line of _forward.
            return model.pool(F.relu(model.bn1(model.L1.forward(x, False, x.mean([2, 3])))))
        raise RuntimeError(f"no Conv1 extractor for {cls}")

    def rdm(f):
        return squareform(pdist(np.asarray(f), metric="correlation"))

    def rho(a, b):
        n = min(a.shape[0], b.shape[0]); i = np.triu_indices(n, k=1)
        return float(spearmanr(a[:n, :n][i], b[:n, :n][i])[0])

    brain = {}
    for roi in ["V1", "V2"]:
        rs = [np.load(FMRI / f"fmri_rdm_{roi}_{s}.npy") for s in SUBJECTS]
        brain[roi] = np.mean(rs, axis=0)

    target = pd.read_csv("/results/outputs_allseeds/rsa_resolution_sweep.csv")
    target = target[target.layer == "Conv1"]

    rows = []
    for rule_name, rule_key in RULES.items():
        if only_rule and rule_name != only_rule:
            continue
        for si, seed in enumerate(SEEDS):
            if only_seed is not None and si != only_seed:
                continue
            model = load_model(rule_key, weights_dir=CKPT, seed_idx=si)
            model = model.to(dev) if hasattr(model, "to") else model.to_device(dev)
            model.eval()
            assert_bn_eval(model, context=f"bridge/{rule_key}")
            if hasattr(model, "training"):
                assert not model.training, "model is still in training mode"

            for px in RESOLUTIONS:
                loader = DataLoader(DS(paths, tf(px)), batch_size=64,
                                    shuffle=False, num_workers=4)
                feats = []
                with torch.no_grad():
                    for imgs, _ in loader:
                        m = conv1_map(model, imgs.to(dev))
                        feats.append(pool_grid(m, 1).cpu())
                X = torch.cat(feats)
                assert_float32(X, "Conv1 g=1 features")
                Xn = X.numpy()
                R = rdm(Xn)
                for roi in ["V1", "V2"]:
                    got = rho(R, brain[roi])
                    want = target[(target.rule == rule_name) & (target.seed_idx == si)
                                  & (target.res == px) & (target.roi == roi)]["rho"]
                    want = float(want.iloc[0]) if len(want) else float("nan")
                    # The target CSV stores rho ROUNDED to 6 decimals
                    # (learning_rules_v10_sweep_modal.py:430), so the comparison must
                    # be at that precision. Differencing an unrounded value against a
                    # rounded one manufactures a spurious residual of up to 5e-7 that
                    # is storage precision, not disagreement.
                    got_r = round(got, 6)
                    rows.append({"rule": rule_name, "seed_idx": si, "res": px,
                                 "roi": roi, "rho_new": got_r,
                                 "rho_new_full": got,
                                 "rho_original": want,
                                 "delta": round(got_r - want, 9)})
                print(f"  {rule_name:<32} s{si} @{px:>3}px  "
                      f"V1 new={rows[-2]['rho_new']:.6f} orig={rows[-2]['rho_original']:.6f} "
                      f"d={rows[-2]['delta']:+.2e}", flush=True)
            del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = Path("/results/outputs_allseeds/bridge_control.csv")
    df.to_csv(out, index=False)
    results_vol.commit()

    worst = df["delta"].abs().max()
    n_bad = int((df["delta"].abs() > 0).sum())
    print("\n" + "=" * 70)
    print(f"BRIDGE CONTROL: {len(df)} cells, max|delta| = {worst:.3e}, "
          f"{n_bad} non-exact")
    if n_bad:
        print(df.reindex(df["delta"].abs().sort_values(ascending=False).index).head(12)
              .to_string(index=False))
        print("\nFAILED: extraction is deterministic, so exact agreement was required.")
    else:
        print("PASSED: the new g=1 path reproduces the original evaluation exactly.")
    print("=" * 70)
    return {"n": len(df), "max_abs_delta": float(worst), "n_non_exact": n_bad}

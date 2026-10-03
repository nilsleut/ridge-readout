"""
determinism_probe_modal.py
==========================
Precondition P3: does `torch.use_deterministic_algorithms(True)` make Backprop
training bit-reproducible at seed 42?

DESIGN -- 2x2, not 1x1
----------------------
P3 asks for one deterministic run. One run cannot answer the question: if it
reproduced the published checkpoint we would learn nothing (the published one was
made nondeterministically, so it should NOT match), and a single run has nothing to
be compared against. So the probe runs a 2x2:

    mode = deterministic     : train twice, compare the two   -> expect IDENTICAL
    mode = default (control) : train twice, compare the two   -> expect DIFFERENT

The control is the negative test that makes the positive result mean anything. If
two DEFAULT runs came out identical, the probe would be measuring nothing at all --
the machine would simply be reproducible -- and the deterministic arm's success
would be vacuous. That is the same "an assertion that cannot fail proves nothing"
rule applied to an experiment.

WHAT IT WOULD ESTABLISH
-----------------------
Deterministic identical + default different  -> the nondeterminism is in the
    algorithm selection that `use_deterministic_algorithms` controls, i.e. the
    nondeterministic cuDNN convolution-backward kernels. Turns an open question into
    a methods sentence.
Deterministic ALSO different -> the source is elsewhere (data loading order, worker
    seeding, atomics not covered by the flag) and the account in RETRAIN_REPORT.md
    must be withdrawn, not merely softened.

Only Backprop is probed: it is the rule with the large divergence, and it is the one
whose account is load-bearing. Nothing here touches the 30-checkpoint retrain.

Run:
    python -m modal deploy code/ridge/determinism_probe_modal.py
    python -c "import modal; print(modal.Function.from_name('determinism-probe','probe').remote())"
"""

import modal

app = modal.App("determinism-probe")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "numpy")
    # CUBLAS_WORKSPACE_CONFIG is REQUIRED by use_deterministic_algorithms for
    # cuBLAS GEMM reproducibility; without it PyTorch raises at the first matmul.
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
)

results_vol = modal.Volume.from_name("learning-rules-rsa", create_if_missing=True)

# Exactly the v10 Backprop hyperparameters (learning_rules_v10_sweep_modal.py:74-90, 504-505).
SEED, N_EPOCHS, BATCH, LR, N_CIFAR = 42, 40, 128, 1e-3, 8000
C1, C2, C3, FC1_DIM, N_CLS, FEAT_SIZE = 32, 64, 128, 512, 10, 4
FC1_IN = C3 * FEAT_SIZE * FEAT_SIZE


@app.function(image=image, gpu="T4", timeout=60 * 90,
              volumes={"/results": results_vol})
def probe():
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset

    def build_and_train(deterministic: bool):
        """One full Backprop training run, mirroring the v10 path exactly."""
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False   # held fixed across both arms,
            # so the only thing varying is the deterministic-algorithm flag.

        dev = torch.device("cuda")
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2))

        class BP_CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = blk(3, C1); self.conv2 = blk(C1, C2); self.conv3 = blk(C2, C3)
                self.fc1 = nn.Linear(FC1_IN, FC1_DIM); self.fc2 = nn.Linear(FC1_DIM, N_CLS)
                self.drop = nn.Dropout(0.3)
            def forward(self, x):
                x = self.conv3(self.conv2(self.conv1(x)))
                return self.fc2(self.drop(F.relu(self.fc1(x.view(x.size(0), -1)))))

        model = BP_CNN().to(dev)

        tf = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                        T.ToTensor(),
                        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        full = torchvision.datasets.CIFAR10("/results/cifar_data", train=True,
                                            download=True, transform=tf)
        idx = torch.randperm(len(full))[:N_CIFAR].tolist()
        # num_workers=0: worker seeding is itself a nondeterminism source, and the
        # question here is about the conv kernels, not the loader. Held fixed
        # across both arms so it cannot explain a difference between them.
        loader = DataLoader(Subset(full, idx), batch_size=BATCH, shuffle=True,
                            num_workers=0, drop_last=True)

        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)

        model.train()
        for _ in range(N_EPOCHS):
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad()
                F.cross_entropy(model(x), y).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
        model.eval()
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def compare(a, b):
        assert set(a) == set(b), "key sets differ"
        diffs = {}
        for k in sorted(a):
            x, y = a[k], b[k]
            if torch.equal(x, y):
                continue
            xf, yf = x.double(), y.double()
            n = xf.norm().item()
            diffs[k] = {
                "max_abs": (xf - yf).abs().max().item(),
                "rel_norm": ((xf - yf).norm().item() / n) if n > 0 else float("inf"),
                "n_diff": int((xf != yf).sum().item()),
                "n": x.numel(),
            }
        return diffs

    out = {}
    for mode, det in [("deterministic", True), ("default", False)]:
        print(f"\n=== {mode}: run 1 ===", flush=True)
        a = build_and_train(det)
        print(f"=== {mode}: run 2 ===", flush=True)
        b = build_and_train(det)
        d = compare(a, b)
        out[mode] = {
            "identical": not d,
            "n_tensors": len(a),
            "n_differing": len(d),
            "conv1_0_weight": d.get("conv1.0.weight"),
            "fc1_weight": d.get("fc1.weight"),
            "worst_rel_norm": max([v["rel_norm"] for v in d.values()], default=0.0),
        }
        print(f"{mode}: identical={not d}  differing={len(d)}/{len(a)}", flush=True)

    # ── Verdict ──────────────────────────────────────────────────────────────
    det_ok = out["deterministic"]["identical"]
    ctl_differs = not out["default"]["identical"]
    if det_ok and ctl_differs:
        verdict = ("CONFIRMED: two default runs differ, two deterministic runs are "
                   "bit-identical. The nondeterminism is in the algorithm selection "
                   "that use_deterministic_algorithms(True) controls.")
    elif det_ok and not ctl_differs:
        verdict = ("VACUOUS: the default control also reproduced, so this machine "
                   "was reproducible anyway and the deterministic arm shows nothing. "
                   "The probe did not test what it was built to test.")
    elif not det_ok and ctl_differs:
        verdict = ("REFUTED: deterministic runs still differ. The source is NOT (only) "
                   "the flagged algorithms -- withdraw the cuDNN account rather than "
                   "softening it.")
    else:
        verdict = "INCOHERENT: deterministic differs, default identical. Investigate."
    out["verdict"] = verdict
    print("\n" + "=" * 70 + f"\n{verdict}\n" + "=" * 70, flush=True)
    return out

# Phase 2 preconditions — findings note

Config `8163d8c4a146`. Sources: the Phase 1 gate verdict
(`outputs_allseeds/gate_verdict.json`), the retrain CSV, the published `results/`, and the
THINGS voxel metadata. No new claims beyond what these support.

---

## P1 — Per-tensor divergence: the depth gradient, and a correction

> **SCOPE (set 2026-08-19, Phase 2b Task 4). This is a methods footnote at most, not a
> paper finding.** It rests on one re-run pair per rule, it concerns only
> nondeterminism-induced divergence, and it is a by-product of a reproducibility check —
> not a claim about weight space and function in general. It does not enter the paper as
> a finding, and no further compute is spent on it. The FA-under-Adam run that would
> separate the optimiser account is **out of scope and will not be run.**


**The 63.6% figure was cited in the wrong context.** It is `fc1.weight`, which sits
*downstream* of Conv1 and therefore cannot affect the measured Conv1 features at all.
Quoting it while discussing the Conv1→V1 claim was a category error on my part.

### Backprop, seed 42, all differing tensors

| Tensor | group | max \|Δ\| | rel-norm | elements differing |
|---|---|---|---|---|
| `conv1.0.weight` | conv | 2.82e−2 | **4.90e−2** | 864/864 |
| `conv1.1.weight` (γ) | bn | 1.53e−2 | 5.87e−3 | 32/32 |
| `conv1.1.bias` (β) | bn | 2.08e−2 | 1.05e−1 | 32/32 |
| `conv1.1.running_mean` | bn | 4.67e−2 | **1.54e−1** | 32/32 |
| `conv1.1.running_var` | bn | 2.58e−1 | **1.39e−1** | 32/32 |
| `conv2.0.weight` | conv | 7.58e−2 | 2.30e−1 | 18432/18432 |
| `conv2.1.running_mean` | bn | 5.78e−1 | 1.61e−1 | 64/64 |
| `conv3.0.weight` | conv | 7.32e−2 | 4.15e−1 | 73728/73728 |
| `conv3.1.running_mean` | bn | 5.70e−1 | 2.14e−1 | 128/128 |
| `fc1.weight` | fc | 1.23e−1 | 6.36e−1 | 851808/1048576 |
| `fc2.weight` | fc | 9.66e−2 | 4.08e−1 | 4155/5120 |

Bit-identical: `conv1.1.num_batches_tracked`, `conv2.1.num_batches_tracked`,
`conv3.1.num_batches_tracked` (3/22) — counters, so this is expected and not informative.

**Divergence grows monotonically with depth: 4.9% → 23.0% → 41.5% → 63.6%.**

### The answer to P1's binary

Neither branch cleanly. Conv1 **does** genuinely diverge — every one of its 864 weights,
max |Δ| 2.8e−2, and the BatchNorm statistics that normalise its output diverge by
13.9–15.4%, considerably more than the weights themselves. But it diverges **13× less than
`fc1`**, so "Conv1 diverges by a comparable fraction" is false.

The accurate statement: *divergence grows monotonically with depth, and the measured layer
sits at the shallow end of that gradient.*

### What the ρ data adds

| Layer→ROI | weight rel-norm | max \|Δρ\| (seed 42, 6 res) |
|---|---|---|
| Conv1→V1 | 4.9% (BN ~15%) | 1.30e−3 |
| Conv1→V2 | " | 9.00e−4 |
| Conv2→V1 | 23.0% | 2.23e−3 |
| Conv2→V2 | " | 1.20e−3 |
| Conv3→LOC | 41.5% | 5.78e−4 |
| FC1→IT | 63.6% | 6.54e−4 |

ρ agreement is ~1e−3 everywhere, **with no depth trend**: FC1→IT, with the largest weight
divergence, agrees *better* than Conv2→V1, which has a quarter of it.

**So "gross weight-space divergence, near-identical representational geometry" is
supportable — but the evidence is at the deep layers, not at Conv1.** At Conv1 the
divergence is modest and the near-identity is unremarkable. The sentence is written
nowhere in this form, and where it appears in `RETRAIN_REPORT.md` it is now scoped to the
layers that support it.

**Feedback Alignment:** ~1e−5 to 1e−4 throughout. One reading caveat: `conv{n}.0.b` shows
rel-norm up to 1.41 at max |Δ| ≈ 2e−10 — the biases initialise to zero and barely move, so
the denominator is ~0 and **relative norm is meaningless for those tensors**. Read absolute
magnitude there.

---

## P2 — Nondeterminism magnitude, restated descriptively

**Withdrawn:** "about a sixth of the reported seed-level SEM". One re-run per rule is a
single realization of |Δ|, not a variance estimate, and a ratio of one |Δ| to an SEM is not
a variance decomposition.

**What was measured, and all of it:**

- A single re-run produced ρ deviations of up to **4.6e−3** across 1,080 cells.
- At Conv1→V1, Backprop: **77/180 cells exceed 1e−3**, mean **1.1–1.3e−3**, max 2.3e−3.
- Across all 180 Backprop cells: median 8.7e−4, p95 3.0e−3, max 4.6e−3.
- Random Weights, Random (BN-calibrated), Predictive Coding, STDP: **exactly 0**.

A variance component would need repeated Backprop runs at a fixed seed; three more would
support one. **Deferred, and no variance claim is made.**

---

## P3 — Mechanism

### No compute: optimizers and backward structure

| Rule | Optimizer | Backward reaches a conv? | Conv weights updated by |
|---|---|---|---|
| Backprop | **Adam** (lr 1e−3, wd 1e−4, cosine) — v10:504 | yes | autograd |
| Feedback Alignment | **SGD** (lr 5e−4, momentum 0.9, wd 1e−4, cosine) — v10:507 | yes (fixed random feedback) | autograd |
| Predictive Coding | Adam on `fc1`+`fc2` only — v10:226 | **no** — `r3.detach()` at v10:257 | hand-computed local rule |
| STDP | Adam on `fc1`,`fc2`,BN affine — v10:295 | **yes** (no detach at v10:319; BN params need it) | hand-computed STDP rule |
| Random / Random (BN-cal) | none | no | not updated |

**Backprop uses Adam and FA uses SGD+momentum, as P3 anticipated.** Adam's per-parameter
√v normalisation amplifies small divergences far more aggressively than SGD over 2,480
steps, which is consistent with the ~600× gap in weight divergence between the two. That
is consistency, not proof.

**A refinement P3's framing does not cover.** The proposed partition — "the four
bit-identical rules are exactly those with no autograd backward pass" — is not quite right.
PC detaches before the conv, so it has none. But **STDP does backpropagate through
convolutions**: there is no detach at v10:319, and its BN affine parameters are in the
optimizer, which requires gradient flow back through `L2.conv` and `L3.conv`. Yet STDP is
bit-identical.

The partition that actually fits the data is narrower: **rules whose stored parameters are
updated from convolution-*weight* gradients** (Backprop, FA) versus rules where they are
not (PC, STDP — STDP's conv weights are set by `weight.data +=` from a hand-computed
kernel, and any autograd conv-weight gradient it computes is discarded). This maps onto the
known PyTorch situation in which `conv2d_weight` backward uses atomics and is
nondeterministic while `conv2d_input` backward is not.

That is a hypothesis consistent with all six rules. The probe below is what tests it.

### Compute: the 2×2 probe

`code/ridge/determinism_probe_modal.py`. P3 asked for one deterministic run; one run cannot
answer the question, because a single run has nothing to compare against and should *not*
match the published checkpoint (that one was made nondeterministically). So:

| arm | two runs, seed 42, 40 epochs | expectation |
|---|---|---|
| `use_deterministic_algorithms(True)` | compare to each other | identical |
| default (negative control) | compare to each other | different |

The control is what makes the result mean anything: if two *default* runs came out
identical, the machine would simply be reproducible and the deterministic arm would be
vacuous. `cudnn.benchmark=False` and `num_workers=0` are held fixed across both arms, so
the deterministic-algorithm flag is the only thing varying.

**Result: see the appendix at the end of this file.**

---

## P4 — Contamination audit

### P4.1 — Did any published number flow through the `models_v2` loader with random weights?

**No. Both loader-dependent artifacts are clean, by direct evidence.**

Every script traced:

| Result file | Model source | Loader risk |
|---|---|---|
| `bnfix_sweep.csv` | v10, trains inline | none |
| `rsa_resolution_sweep_upsample.csv` | v11, trains inline | none |
| `training_dynamics_results.csv` | trains inline | none |
| `bncal_rows.csv` | trains inline | none |
| `colour_statistic_control.csv` | trains inline | none |
| `colour_within_filter.csv` | RDMs from inline run | none |
| `rsa_resolution_sweep_resnet.csv` / `_swin.csv` | torchvision | none |
| **`rsa_lowlevel_control.csv`** | **`models_v2.load_model`** | **audited below** |
| **`rsa_resolution_control_macaque.csv`** | **`models_v2.load_model`** | **audited below** |

**`rsa_lowlevel_control.csv` — clean by exact identity.** Its `cnn_backprop` and
`cnn_random` V1 alignments reproduce `bnfix_sweep.csv` seed 42 **exactly, to all six
decimal places, at all six resolutions** (Δ = 0.0 in 12/12 comparisons). A
randomly-initialised fallback could not reproduce trained-Backprop values to six decimals.
Real weights were loaded.

**`rsa_resolution_control_macaque.csv` — clean by a preflight guard plus cross-run
stability.** The script already defends against this exact failure: `_resolve_weights_dir()`
(`run_resolution_control_macaque_v2.py:52`) followed by a preflight check at `:90-105` that
aborts if any checkpoint is missing, logging *"load_model would use RANDOM init for these
-> results would be invalid."* Empirically, `random` reproduces between the pre-fix and
post-fix runs to **1.0e−6** (mean 1.25e−7). Two separate runs falling back to random
initialisation would produce two *different* random networks and differ by O(0.1), not
1e−6. The other rules move as documented: backprop 2.8e−3, FA 3.4e−3 (nondeterminism),
PC 3.2e−2 and STDP 5.8e−2 (the BatchNorm repair).

**The `Random (BN-calibrated)` omission affected nothing published.** Neither
loader-dependent file contains that condition: the macaque file's rules are
`{backprop, feedback_alignment, predictive_coding, random, stdp}` and the low-level file's
models are `{cnn_random, cnn_backprop, resnet50, GABOR, PIXEL}`. The condition appears only
in `bnfix_sweep.csv` and `bncal_rows.csv`, both from scripts that train inline. The omission
blocked Phase 2 and nothing else.

**No artifact requires correction or retraction on this account.**

### P4.2 — The `nc_testset` threshold in the sibling project

**Scoped only, not fixed, as instructed.**

The sibling *Dendritic Encoding Project* passes `--nc_threshold 0.05` in its
`reproduce_all.sh:100` and in both documented commands (`README.md:140, 143`);
`analysis/permutation_null_test.py:459` defaults to `0.05`. `nc_testset` is on a **0–100
percent** scale, so a 0.05 threshold is being read as though it were a 0–1 fraction.

**A correction to my own Phase 1 statement.** I said this "selects essentially every
voxel". Measured, it does not — a substantial share of voxels sit at or near zero:

| Threshold | V1 retained (3 subjects) | IT retained (3 subjects) |
|---|---|---|
| `>= 0.05` (as run) | 70.2 – 78.7% | 67.6 – 75.7% |
| `>= 5` (a true 5%) | 60.4 – 72.3% | 57.2 – 67.4% |
| `>= 10` (a true 10%) | 49.5 – 62.8% | 44.3 – 58.0% |

So the filter is **weaker than intended by roughly 7–11 percentage points**, not absent.
The retained voxel sets overlap heavily, so the consequence is a mild dilution of an
already-inclusive selection rather than a qualitative change. "Essentially every voxel"
overstated it and is withdrawn.

**Does anything published depend on it?** Within *this* repo, no — nothing in
`evaluation-resolution-rsa` reads `nc_testset`. Within the sibling project, yes: its
documented reproduction path and its permutation null test both run at 0.05. Whether that
project's outputs are published anywhere is outside what I can determine from these files.

**This has no bearing on the ridge study.** `config.yaml` deliberately does not threshold
on `nc_testset`; voxel selection uses a split-half reliability computed on Arm B, whose
definition we wrote.

---

## Appendix — P3 probe result: **CONFIRMED**

Four Backprop training runs at seed 42, 40 epochs each, on one T4.

| Arm | Two runs compared | Result |
|---|---|---|
| `use_deterministic_algorithms(True)` | to each other | **bit-identical, 0/22 tensors** |
| default (negative control) | to each other | **differ, 19/22 tensors** |

Control-arm magnitudes: `conv1.0.weight` rel-norm **4.53e−2**, `fc1.weight` **5.91e−1**,
worst 5.91e−1.

**Two things follow.**

1. **The source is identified.** Two default runs diverge; two deterministic runs are
   bit-identical. The nondeterminism lies entirely in the algorithm selection that
   `use_deterministic_algorithms(True)` controls — the nondeterministic cuDNN
   convolution-backward kernels. The negative control is what licenses this: had the
   default arm also reproduced, the machine would simply have been reproducible and the
   deterministic arm would have shown nothing.

2. **The P1 depth gradient replicates independently.** The control arm's divergence —
   conv1 4.53%, fc1 59.1% — closely reproduces the retrain-vs-published figures of 4.90%
   and 63.6%, from a completely separate pair of runs. The depth gradient is a
   reproducible property of this training setup, not a one-off.

**What can now be said in methods, and what still cannot.** It is established that the
nondeterminism is confined to the flagged algorithms and that it produces a
depth-increasing weight divergence. It is *not* established that Adam's √v normalisation
is the amplifier — that remains the plausible account fitting the FA (SGD) contrast, and
separating it would need FA re-run under Adam, which is not part of this study.

Deterministic mode is **not** adopted for the main runs: it would break bit-identity with
the published checkpoints, which is the comparison the gate depends on.

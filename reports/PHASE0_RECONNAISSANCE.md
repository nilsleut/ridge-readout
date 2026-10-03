# Phase 0 — Reconnaissance report

Ridge readout on full feature maps across the resolution sweep.
Repo: `evaluation-resolution-rsa` @ `6cf64f8`. Date: 2026-08-17.

Every claim below cites the file and line it came from. Paths outside the repo are
marked **[external]** — the repo alone does not answer A, B, or C, and the report says
so where that is the case.

**Two headline findings that change the design:**

1. **The 720 evaluation stimuli have zero within-subject repeats** — each is a single
   presentation. No split-half noise ceiling is possible on them. (§A)
2. **Only 6 of the 25 checkpoints exist**, one per rule at seed 42. The sweep trains
   inline and saves `seed_idx == 0` only. Phase 3 as written ("all 25 checkpoints")
   cannot run without retraining. (§F)

---

## Locations

| What | Where |
|---|---|
| Repo | `c:/Users/nilsl/Desktop/Projekte_2/evaluation-resolution-rsa` |
| **[external]** Upstream fMRI→RDM script | `Projekte_1/learning-rules-rsa/extract_fmri_rdms_720.py` |
| **[external]** Derived RDMs + stim order | `Projekte_1/learning-rules-rsa/outputs_720/` |
| **[external]** Raw THINGS-fMRI (47 GB) | `Projekte_1/RSA/Datensatz/` |
| **[external]** Checkpoints | `Projekte_1/learning-rules-rsa/learning_rules_outputs_bnfix/checkpoints/` |
| **[external]** THINGS images | `Projekte_1/RSA/Datensatz/images_THINGS/object_images` |

The raw voxel-wise responses **are present locally** — `sub-0{1,2,3}_task-things_voxel-wise-responses.h5`,
15.7 / 16.9 / 14.1 GB. Ridge on voxels is therefore feasible in principle. Nothing in
the repo itself contains voxel data; `code/cross_species/data_loader.py:177-215`
(`load_things_fmri`) reads **precomputed RDMs only**, so no code path in this repo
currently touches a voxel.

---

## A. Repeat structure — **no repeats on the evaluation set**

Determined from **[external]** `sub-0N_task-things_stimulus-metadata.csv` (9,840 rows
per subject) cross-referenced against `outputs_720/stim_order_sub-01.txt` (720 lines).

Per subject, the design splits into:

| `trial_type` | Trials | Unique images | Presentations each |
|---|---|---|---|
| `train` | 8,640 | 8,640 | **1** |
| `test` | 1,200 | 100 | **12** |

**All 720 evaluation stimuli are `train` trials, one presentation each, in all three
subjects.** Overlap between the 720 and the 100-image test set is **exactly zero**. The
100 test images are identical across all three subjects and cover 100 concepts.

Consequences:

- **No split-half reliability is available on the evaluation stimuli.** Per the working
  rules, the ceiling falls back to **leave-one-subject-out inter-subject consistency**,
  which is a lower bound with a different interpretation and must be labelled as such
  everywhere it appears — including in figure axes, not only in prose.
- The averaging step at `extract_fmri_rdms_720.py:108-113` (`responses_all[idx].mean(axis=0)`)
  is a **no-op** for these 720: `idx` has length 1 for every one of them.
- Therefore **the published RDMs are built from single-trial betas**, not trial averages.
  This contradicts the paper: `paper/Evaluation_Resolution_Confounds_paper.tex:223`
  states "fMRI RDMs were computed with correlation distance on trial-averaged BOLD
  responses." That sentence is accurate about the code path but not about what the code
  path did to these stimuli. **This is a correction the paper needs regardless of how
  the ridge work turns out.**

  *(WITHDRAWN 2026-08-18, per the Phase 1 brief: this bullet originally continued "and it
  is the most likely reason absolute ρ is so low". That was plausible but not established,
  and it is exactly what Arm B exists to measure. Not asserted anywhere until measured.)*

**A usable instrument exists anyway, with a caveat.** The voxel metadata ships four
per-voxel reliability columns (`sub-0N_task-things_voxel-metadata.csv`, columns 6-9):
`nc_singletrial`, `nc_testset`, `splithalf_uncorrected`, `splithalf_corrected`. In V1:

| Subject | V1 voxels | median `splithalf_corrected` | fraction > 0.2 |
|---|---|---|---|
| sub-01 | 1,049 | 0.176 | 0.478 |
| sub-02 | 1,104 | 0.179 | 0.469 |
| sub-03 | 839 | **0.108** | 0.341 |

sub-03 is quantitatively the weak subject flagged in Limitations item 3 (tex:192).

These columns almost certainly derive from the 12-repeat **test** set, i.e. reliability
measured over *different stimuli* than the 720. That is a genuine advantage for §1.4 —
voxel selection becomes structurally independent of the data being fitted, so it cannot
leak — but it is **not** a ceiling for the 720 and must not be used as the denominator
in r/ceiling. I could not verify the estimator behind these columns from anything in
this repo (see "Could not determine", item 2).

> **UPDATE 2026-08-20 — confirmed, and `splithalf_corrected` is now identified.** The
> "almost certainly" above is discharged: computing split-half reliability ourselves from
> the 12 repetitions (all 462 balanced splits, Spearman-Brown) reproduces the shipped
> `splithalf_corrected` column **exactly** — r = 1.000000, max |diff| = 0.00000, over all
> 21,578 ROI voxels in three subjects. The three V1 medians quoted in the table above
> (0.176 / 0.179 / 0.108) are therefore that quantity. The "not a ceiling for the 720"
> caveat is unaffected and still binds.

## B. Concept/exemplar structure — **one exemplar per concept, grouping is a no-op**

`extract_fmri_rdms_720.py:89-98` builds the stimulus order by taking, for each concept
in sorted order, `sorted(...)[:1]` — **exactly one stimulus per concept**, truncated to
720.

Verified against the data: the 720 entries in `stim_order_sub-01.txt` are 720 distinct
stimuli spanning **720 distinct concepts**, 1 exemplar each. The three subjects' stim
order files are byte-identical (sha256 `e6e2125077df5291…` for all three).

**Consequence for §1.3: concept-grouped CV folds are identical to image-level folds.**
The requirement is satisfied trivially and needs no special fold construction — but the
assertion "no concept crosses a fold boundary" should still be written and run as a
regression control, because it is cheap and it will catch a future change to the
stimulus selection. Residual risk not verified: THINGS concepts are distinct nouns, but
semantic near-neighbours (e.g. two similar animals) are not excluded by this design.

## C. Voxel counts per ROI per subject, and upstream selection

Counted **[external]** from the ROI mask columns, using the same expressions as
`extract_fmri_rdms_720.py:55-62` (note `V4 ← hV4`, `LOC ← lLOC | rLOC`):

| Subject | Total voxels | V1 | V2 | V3 | V4 | LOC | IT |
|---|---|---|---|---|---|---|---|
| sub-01 | 211,339 | 1,049 | 774 | 762 | 613 | 2,700 | 4,145 |
| sub-02 | 226,950 | 1,104 | 988 | 830 | 577 | 1,175 | 3,720 |
| sub-03 | 189,164 | 839 | 660 | 663 | 472 | 423 | 3,022 |

**Upstream selection applied:** ROI membership only (`extract_fmri_rdms_720.py:55-67`).
Plus a per-voxel z-score across all 9,840 trials before stimulus selection
(`:77-78`). **No reliability threshold, no outlier removal, no voxel selection of any
kind** — consistent with `tex:223`. So §1.4's fixed-threshold selection is a genuinely
new step, not a re-derivation of something already applied.

## D. Layer for V1 — Conv1 for the headline; the code also computes Conv2

`learning_rules_v10_sweep_modal.py:96-101`:

```python
LAYER_ROI_FIXED = {
    "Conv1": (0, ["V1", "V2"]),
    "Conv2": (1, ["V1", "V2"]),
    "Conv3": (2, ["LOC"]),
    "FC1":   (3, ["IT"]),
}
```

So **six** layer→ROI pairs are computed, and **two** of them target V1 (Conv1 and Conv2).
The headline claim is Conv1→V1 throughout (`README.md:13`, `:22`; `results/README.md:55`).

Two documentation defects found:

- `paper/…tex:55` states the mapping as "Conv1→V1/V2, Conv3→LOC, FC1→IT" — **it omits
  Conv2 entirely**, though Conv2→V1/V2 rows exist in the shipped CSV.
- `results/README.md:177` and `:209` both state "(4 layer→ROI pairs)". The correct count
  is 6. Arithmetic check: 6 rules × 5 seeds × 6 resolutions × 6 pairs = 1,080 = the
  actual row count; with 4 it would be 720. Verified directly against
  `results/bnfix_sweep.csv`: 180 rows in each of Conv1/V1, Conv1/V2, Conv2/V1, Conv2/V2,
  Conv3/LOC, FC1/IT.

**Extraction point and hook: there is no hook.** Activations are computed inline in each
model's `get_features()` (`v10:165-169` BP, `:207-211` FA, `:249-253` PC, `:309-313`
STDP). "Conv1" is the output of the whole first block — `Conv2d → BatchNorm2d → ReLU →
MaxPool2d(2)` (`v10:148-150`) — i.e. **post-pool**, not post-conv. That matters for the
grid ladder: the spatial map already carries one factor-of-2 downsample.

Per §1.6 the a-priori layer for the ridge work is **Conv1**, matching the paper.

## E. The exact GAP call

**`x.mean([2, 3])`** — a tensor-method mean over dims 2 and 3.

- `learning_rules_v10_sweep_modal.py:169, 211, 253, 313` (one per rule class)
- `code/cross_species/models_v2.py:127, 154, 232, 294, 395`

Note the FC1 path uses a *different* pooling call:
`F.adaptive_avg_pool2d(c3, FEAT_SIZE)` with `FEAT_SIZE = 4` (`v10:152-153`,
`models_v2.py:99-101`). So both idioms exist in the codebase and they are used for
different layers. For the g-ladder, the FC1 path is already effectively g = 4 on Conv3.

**On bit-identity.** I tested `x.mean([2,3])` against `F.adaptive_avg_pool2d(x,1).flatten(1)`
on this machine (torch 2.10.0+cpu, float32, batch 128) at all six Conv1 shapes —
16², 32², 48², 64², 80², 112² — and they were **bit-identical in all six cases**. This
does not generalize: I could not test on a T4, where reduction kernels differ and the
result may not hold. The design rule stands either way — **the bridge control must call
the original expression, not a reimplementation of it** (working rule 3), so this
question never needs to be settled.

## F. Checkpoint loading

**Key naming schemes differ per rule.** Read directly from the six shipped `.pt` files
**[external]**, not inferred:

| Rule | Conv key scheme | Other |
|---|---|---|
| Backprop, Random, Random (BN-cal) | `conv{1,2,3}.0.weight` | BN at `conv{n}.1.{weight,bias,running_mean,running_var,num_batches_tracked}`; `fc1`, `fc2`. 22 keys |
| Feedback Alignment | `conv{n}.0.W`, `conv{n}.0.b`, `conv{n}.0.B_feedback` | same BN scheme; 28 keys |
| Predictive Coding | `W{1,2,3}.weight` | `bn{n}.*`, `P2.weight`, `P3.weight`, `fc1`, `fc2`. 24 keys |
| STDP | `conv{1,2,3}.weight` (flat) | `bn{n}.{weight,bias,running_mean,running_var}` — **no `num_batches_tracked`** — `fc1`, `fc2`. 19 keys |

STDP is the odd one out because `STDP_CNN` is a plain class with no `state_dict()`; the
dict is assembled by hand at `v10:564-577`. Loader: `models_v2.py:441-500`, with the
STDP branch at `:457-489`. Note `:460` prefers a `_seed0.pt` variant that **does not
exist** on disk, so it silently falls back to `model_weights_stdp.pt`. That file does
contain the full 19 keys, so the fallback is currently benign — but it is a silent
fallback of exactly the kind that hid the original defect, and `models_v2.py:497-498`
has the same shape: a missing file **warns and proceeds with random init** rather than
raising. Any new pipeline should convert both to hard failures.

**How eval-mode BN is guaranteed** — four mechanisms, all verified present:

1. `assert_bn_eval(model)` called inside feature extraction, after `model.eval()`:
   `v10:401-402`; `models_v2.py:541-545`.
2. `iter_bn()` **raises if it finds zero BatchNorm layers** (`bn_guard.py:66-71`) — the
   anti-vacuous-pass guard. This is the specific check that would have caught the
   original defect.
3. `STDP_CNN.train(mode)` drives `bn1/bn2/bn3` by hand and `eval()` delegates to it
   (`v10:298-303`; `models_v2.py:407-418`). `PC_CNN.eval()` delegates to `super().eval()`
   (`models_v2.py:306-307`).
4. Mode is set explicitly at each phase of the main loop rather than inherited from
   construction (`v10:494-495` comment, `:514`, `:530`, `:540`).

**Independent verification of the stored BN state** (not just the code claim):

| Checkpoint | `num_batches_tracked` | Conv1 BN `running_mean` max abs | `running_var` range |
|---|---|---|---|
| `random_weights` | 0 | 0.0000 | [1.0000, 1.0000] |
| `random_weights_(bn-calibrated)` | 50 | 0.3286 | [0.0680, 2.0577] |
| `backprop` | 2,480 | 0.3106 | [0.0265, 1.7904] |

The random checkpoint is exactly identity BN as documented at `results/README.md:89`;
the calibrated one matches `N_BN_CALIB = 50` (`v10:73`); backprop's 2,480 = 40 epochs ×
62 batches (8,000 images / 128, `drop_last=True`). All three are internally consistent.

### The blocker: only 6 checkpoints exist, not 25

`v10:554` — `if seed_idx == 0:` — the sweep saves checkpoints **for seed 42 only**. The
local checkpoint directory contains exactly six `.pt` files, one per rule, with no seed
suffix. The sweep trains models inline inside the Modal job and discards them
(`v10:594`).

**Phase 3 as specified ("all 25 checkpoints") cannot be run from existing artifacts.**
Either the pilot runs at seed 42 only — which forfeits the seed-level SD that Phase 3
exists to produce, and therefore forfeits Δ_min — or the four remaining seeds per rule
must be retrained first. §1.3 makes the seed the unit of inference, so this is not a
detail that can be deferred; it needs a decision before Phase 1 is instantiated.

Retraining is cheap for BP/FA (≤ 1 h per shard, `spawn_runs.py:37-38`) and expensive for
PC/STDP (4 h windows, `:39-40`). A modified sweep that drops the `seed_idx == 0`
condition would produce all 25 in one pass.

## G. Feature-map shapes at Conv1 (C × H × W), all six resolutions

Computed by instantiating the architecture from `v10:148-150` and running one forward
pass per resolution. Conv1 output is post-`MaxPool2d(2)`, so H = W = px/2.

| px | Conv1 | d = C·H·W | Conv2 | Conv3 |
|---|---|---|---|---|
| 32 | 32 × 16 × 16 | 8,192 | 64 × 8 × 8 | 128 × 4 × 4 |
| 64 | 32 × 32 × 32 | 32,768 | 64 × 16 × 16 | 128 × 8 × 8 |
| 96 | 32 × 48 × 48 | 73,728 | 64 × 24 × 24 | 128 × 12 × 12 |
| 128 | 32 × 64 × 64 | 131,072 | 64 × 32 × 32 | 128 × 16 × 16 |
| 160 | 32 × 80 × 80 | 204,800 | 64 × 40 × 40 | 128 × 20 × 20 |
| 224 | 32 × 112 × 112 | **401,408** | 64 × 56 × 56 | 128 × 28 × 28 |

Three design consequences:

- **d ≫ n = 720 at every resolution, by a factor of 11 to 558.** The dual form (§1.2) is
  not an optimization, it is the only tractable option, and it is uniformly applicable —
  no condition needs a different solver.
- **The capacity confound is a 49× span** in native dimensionality from 32 px to 224 px.
  §1.1's fixed grid is doing real work.
- **The g-ladder is feasible at every resolution without upsampling.** The smallest
  Conv1 map is 16 × 16, so g = 8 is a true 2 × 2 average even at 32 px. Fixed
  dimensionality per g: **g=1 → 32, g=2 → 128, g=4 → 512, g=8 → 2,048.** All four remain
  ≤ n = 720 except g = 8, which sits just above it — worth noting, since g = 8 is then
  the only rung where d > n and the interpolation regime is reachable.

## H. Modal pattern

Defined in `learning_rules_v10_sweep_modal.py`:

- App `learning-rules-rsa` (`:52`). Volumes: `burstprop-data` → `/data` (read),
  `learning-rules-rsa` → `/results` (write) (`:54-55`, `:106`).
- Image: `debian_slim(python_version="3.11")`, apt `git`, pip torch/torchvision/numpy/
  scipy/pandas/Pillow/matplotlib, plus `.add_local_python_source("bn_guard")` (`:57-62`).
- **`gpu="T4"`, `timeout=60*60*24`** (`:105`).
- Outputs under `/results/outputs_bnfix/` (`:138`): `rdms/res{px}/seed_{i}/rdm_{rule}_{layer}.npy`
  (`:546-549`), `checkpoints/` (`:140`), `shards/rows_{rule}_seed{n}.csv` (`:585`),
  merged `rsa_resolution_sweep.csv` (`:605`, `:692`).

**The pattern in use is deploy + spawn, confirmed:** `spawn_runs.py:47` uses
`modal.Function.from_name("learning-rules-rsa", "run_resolution_control")` and `:62`
calls `.spawn(only_rule=…, only_seed=…)`, 30 shards, one per (rule, seed). Per-rule
timeouts via `.with_options(timeout=…)` (`:60`, table at `:34-41`). Call IDs are written
to `spawned_calls.json` (`:69`) and polled with `--poll` (`:74-90`). Rationale at
`:6-8`: `modal run --detach` did not survive the host entering Modern Standby — both
jobs died 2 s after it, 2 h in.

Two patterns worth carrying over verbatim: the shard-per-(rule,seed) write isolation
(`:110-121`), and the **merge-time duplicate check that raises rather than collapses**
(`v10:665-689`) — this is already the "duplicate checking on merge" control required by
working rule 3, and it verifies every numeric column at `atol=1e-12`.

## I. Brain-Score integration surface available locally

Installed and importable on this machine:

| Package | Version |
|---|---|
| `brainscore-vision` | 2.3.18 |
| `brainscore-core` | 2.3.5 |
| `result-caching` | 0.1.0 |
| `scikit-learn` | 1.7.2 |
| `torch` / `torchvision` | 2.10.0+**cpu** / 0.25.0+cpu |
| `modal` | 1.4.3 |
| `h5py` | 3.16.0 |

`brainscore_vision.metrics.regression_correlation` exposes exactly what §1.5(e) needs:
`pls_regression`, `ridge_regression`, `ridge_cv_regression`, `linear_regression`,
`single_regression`, `pearsonr_correlation`, `CrossRegressedCorrelation`,
`SpanTimeCrossRegressedCorrelation`, `TrainTestSplitCorrelation`. Both factories take
`(regression_kwargs=None, xarray_kwargs=None)`. `brainscore_vision.metrics.internal_consistency`
exposes `InternalConsistency` — the standard Brain-Score ceiling, which requires a
`repetition` coordinate and therefore **cannot be applied to the 720** (see §A).

`metric_registry` is empty on import — Brain-Score loads plugins lazily, so registry
population needs an explicit plugin load rather than a bare import.

**Torch here is CPU-only.** Feature extraction for the pilot has to run on Modal, or
accept CPU wall time locally.

**PR #2352: I could not verify anything about it.** The string does not appear anywhere
in the repo, and I did not query GitHub. Everything above is what the *installed*
2.3.18 provides; whether that predates or postdates #2352, and what #2352 changed, is
unknown to me. Per working rule 2, I am not going to assume.

---

## Could not determine

1. **What Brain-Score PR #2352 merged.** No reference in the repo. The available surface
   in §I is measured from the installed package, independent of that PR.
2. **~~The estimator behind `nc_singletrial`, `nc_testset`, `splithalf_uncorrected`,
   `splithalf_corrected`.~~ RESOLVED 2026-08-20.**

   `nc_testset` / `nc_singletrial`: percent of explainable variance, 0-100, from the
   12-repetition test set (Hebart et al. 2023 reports up to 80% in early visual cortex;
   the measured global max is 83.67, attained in V1).

   **`splithalf_corrected`: identified by exact reconstruction.** Computing split-half
   reliability ourselves — all 462 balanced splits of the 12 repetitions, averaged, then
   Spearman-Brown corrected — reproduces the shipped column **exactly**: r = 1.000000,
   median |diff| = 0.00000, max |diff| = 0.00000, across all 21,578 ROI voxels in three
   subjects. So the column is that quantity, and the refusal to threshold on it is
   discharged. (The operative rule still computes its own, which is the identical
   quantity; `config.yaml`'s `status: UNVERIFIED` line for it is now stale and is
   queued for the next schema bump rather than edited in isolation.)

   `splithalf_uncorrected` remains unverified, though the obvious reading — the same
   quantity before the Spearman-Brown step — is now strongly implied.

   *(original text follows)*

   **The estimator behind `nc_singletrial`, `nc_testset`, `splithalf_uncorrected`,
   `splithalf_corrected`.** These are THINGS-pipeline columns; nothing in this repo
   documents or computes them. Their scale is heterogeneous — the two `nc_*` columns run
   to 88, so they are not correlations — and I will not build a voxel-selection threshold
   on a column whose definition I have not read. Resolving this needs the THINGS-fMRI
   data documentation, not this repo.
3. **Whether the other 19 checkpoints exist on the Modal `learning-rules-rsa` volume.**
   `v10:554` says only seed 0 is ever written, so almost certainly not — but I did not
   query the volume, which would need `modal volume ls`.
4. **ROI mask provenance.** The metadata carries both plain `V1`/`V2`/`V3` columns and
   pRF-suffixed ones (`LO1 (prf)`, `LO2 (prf)`), plus `prf-eccentricity`, `prf-polarangle`,
   `prf-size`, `prf-rsquared`. Whether the V1 mask is atlas- or pRF-derived is not
   documented here. Relevant because retinotopic structure is the stated motivation for
   the whole ridge exercise — and because `prf-*` would allow a retinotopy-aware version
   of the g-ladder that the current design does not contemplate.
5. **Units and exact definition of `ResponseData/block0_values` in the H5.** I inferred
   single-trial betas from the row count matching the 9,840 metadata rows exactly; the
   repo does not document it.
6. **Stimulus visual subtense during acquisition.** Absent, as the paper itself notes
   (tex:196). Not needed for this work, but it is the missing input for the
   receptive-field-matching criterion the paper declines to establish.

## Two corrections the repo needs regardless of this work

- **`paper/…tex:223`** claims trial-averaged BOLD responses. The 720 evaluation stimuli
  are single-presentation, so no averaging occurs for any of them (§A).
- **`results/README.md:177`, `:209`** state 4 layer→ROI pairs; the actual number is 6,
  and 6 is what makes the stated 1,080-row grid arithmetic work (§D). `paper/…tex:55`
  omits Conv2 from the mapping.

Neither changes a number in the paper. Both are the kind of documentation drift that
makes a re-derivation silently disagree with the source.

---

## Stopping here

Phase 0 ends per working rule 1. Before Phase 1 is instantiated, one decision is
genuinely yours and cannot be defaulted:

**The 19 missing checkpoints.** §1.3 makes the seed the unit of inference and Phase 3
exists to produce a seed-level SD and Δ_min. With one seed, neither is estimable, and
"the method cannot resolve it" becomes unfalsifiable — the exact failure mode §1.5(d)
is designed to prevent. The options are to retrain the four remaining seeds per rule
before the pilot (a modified sweep dropping the `seed_idx == 0` guard, and one pass),
or to accept a single-seed pilot that can position conditions against the pixel baseline
and the ceiling but cannot deliver Δ_min.

Everything else in Phases 1-2 is answerable from what is above.

# Step 7 — offsets, the distance multiverse, and the gain axis

**Exploratory.**
- Predictions and rules: `amendments/2026-10-09_step7-offset-predictions.md` (commit
  `084b22c`, before any step 7 script). Its "operationalisation notes" fix three points
  the task left open; the main one is that D is applied to both the model features and
  the fMRI betas.
- Config hash `206fa875cca5`.
- No new fits and no new extraction: features come from `STEP5_GAP_DIR` (incl.
  `bandpass/`), betas from `arm_b_out/`.
- Convention as in step 5: per subject on all ROI voxels, cross-run pairs, 1000 paired
  stimulus-bootstrap draws.
- N = 3 subjects. Inference over stimuli. The V1 sign is not read as a statement about
  learning rules.

## Run notes

- The first run of `step7_multiverse.py` (commit `a468946`) computed point estimates
  serially and needed about 70 min per feature normalisation. It was stopped after
  three of four, before writing any result.
- The point estimates were then parallelised (commit `be4a850`; identical
  computation) and the script was re-run from the start: 71 s for
  5,040 model RDMs.
- Bootstrap estimate 1.02 h, below the 4 h limit, so **all 36 cells are bootstrapped**
  (no z720 restriction).
- Identity draw vs point estimates: max |Δ| 1.3e-16.
- Per-cell draws: `data/step7_draws/cell_{F}_{D}_{B}.npz` (1001 × 5 per-seed gaps per
  condition; gitignored, not in the manifest).

## O1 — mathematical control

Under D = euclidean:
- models, F = raw vs F = center: max |Δ| = **1.1e-14**;
- fMRI, B = paper vs B = center720 (the betas differ by per-voxel offsets only):
  max |Δ| = **4.3e-14**.

**O1 met.** By construction, euclidean also equates scale with z and center720 with
paper; the tables show these identities.

## O2 — resolution slope S across the multiverse (`results/step7/multiverse.csv`)

S = Gap(NATIVE 224) − Gap(NATIVE 32), Gap = Random − BP at V1. "Strong" means a CI
above 0 and S ≥ 0.5 · S_paper = +0.0156. Allowed cells (where strong slopes are
permitted): D ∈ {correlation, cosine}, F ∈ {raw, scale}, B = paper.

| F | D | S, B = paper | S, B = center720 | S, B = z720 |
|---|---|---|---|---|
| raw | correlation | **+0.0311 [+0.0239, +0.0384]** (strong, allowed) | +0.0061 [+0.0008, +0.0114] | +0.0060 [+0.0008, +0.0113] |
| raw | cosine | +0.0020 [−0.0014, +0.0053] | −0.0015 [−0.0036, +0.0005] | −0.0015 [−0.0036, +0.0005] |
| raw | euclidean | +0.0053 [−0.0031, +0.0139] | = paper | +0.0051 [−0.0031, +0.0138] |
| center | correlation | +0.0062 [+0.0026, +0.0097] | +0.0064 [+0.0033, +0.0097] | +0.0063 [+0.0032, +0.0096] |
| center | cosine | +0.0034 [−0.0020, +0.0090] | +0.0009 [−0.0043, +0.0060] | +0.0009 [−0.0042, +0.0059] |
| center | euclidean | = raw/euclidean | = raw/euclidean | = raw/euclidean |
| scale | correlation | +0.0012 [−0.0038, +0.0062] | −0.0027 [−0.0057, −0.0001] | −0.0028 [−0.0057, −0.0002] |
| scale | cosine | −0.0040 [−0.0076, −0.0006] | −0.0031 [−0.0054, −0.0011] | −0.0031 [−0.0054, −0.0011] |
| scale | euclidean | +0.0044 [−0.0076, +0.0170] | = paper | +0.0043 [−0.0078, +0.0167] |
| z | correlation | +0.0016 [−0.0038, +0.0070] | +0.0001 [−0.0047, +0.0049] | +0.0001 [−0.0046, +0.0049] |
| z | cosine | −0.0024 [−0.0096, +0.0053] | −0.0076 [−0.0147, −0.0004] | −0.0076 [−0.0146, −0.0004] |
| z | euclidean | = scale/euclidean | = scale/euclidean | = scale/euclidean |

**Applying O2:**
- 36 of 36 cells tested; **0 violations** (≤ 2 allowed): **O2 met.**
- The only strong slope is in raw / correlation / paper, the paper's own convention.
- The other three allowed cells (raw / cosine, scale / correlation, scale / cosine,
  each with B = paper) do not show one either.
- Several non-allowed cells have CIs above 0, but every such slope is small (≤ +0.0064).

## O3 — LOC, BP − Random at NATIVE 224

| F | D | B = paper | B = center720 | B = z720 |
|---|---|---|---|---|
| raw | correlation | +0.0113 [+0.0025, +0.0210] | +0.0044 [−0.0015, +0.0101] | +0.0044 [−0.0015, +0.0101] |
| raw | cosine | +0.0082 [+0.0005, +0.0164] | +0.0045 [−0.0008, +0.0097] | +0.0045 [−0.0007, +0.0095] |
| raw / center | euclidean | +0.0117 [−0.0102, +0.0333] | same | +0.0118 [−0.0098, +0.0332] |
| center | correlation | +0.0062 [+0.0007, +0.0111] | +0.0071 [+0.0021, +0.0119] | +0.0070 [+0.0020, +0.0118] |
| center | cosine | +0.0063 [+0.0011, +0.0116] | +0.0067 [+0.0016, +0.0117] | +0.0067 [+0.0016, +0.0116] |
| scale | correlation | +0.0104 [+0.0029, +0.0183] | +0.0052 [+0.0005, +0.0102] | +0.0052 [+0.0005, +0.0103] |
| scale | cosine | +0.0076 [+0.0012, +0.0138] | +0.0047 [+0.0004, +0.0090] | +0.0047 [+0.0004, +0.0089] |
| scale / z | euclidean | +0.0107 [−0.0096, +0.0309] | same | +0.0107 [−0.0090, +0.0307] |
| z | correlation | +0.0066 [+0.0024, +0.0109] | +0.0070 [+0.0029, +0.0112] | +0.0069 [+0.0028, +0.0112] |
| z | cosine | +0.0058 [+0.0013, +0.0106] | +0.0058 [+0.0016, +0.0102] | +0.0058 [+0.0015, +0.0102] |

**Applying O3:** positive in **36/36** cells (≥ 30 required), CI above 0 in **20/36**
(≥ 18 required): **O3 met.** The CI includes 0 in all 12 euclidean cells and in the
four raw cells with B = center720 or z720.

## V1 gap at NATIVE 224 — 4 × 3 × 3 sign table (`results/step7/multiverse_sign_table.csv`)

`+` / `−` = sign of the point estimate; `*` = CI excludes 0.

| F \ (B, D) | paper corr | paper cos | paper eucl | c720 corr | c720 cos | c720 eucl | z720 corr | z720 cos | z720 eucl |
|---|---|---|---|---|---|---|---|---|---|
| raw | **+\*** (+0.030) | + | **+\*** (+0.023) | + | + | **+\*** | + | + | **+\*** |
| center | + | − | **+\*** | + | **−\*** | **+\*** | + | **−\*** | **+\*** |
| scale | − | **−\*** | **+\*** (+0.016) | − | − | **+\*** | − | − | **+\*** |
| z | **−\*** (−0.011) | **−\*** | **+\*** | **−\*** | **−\*** | **+\*** | **−\*** | **−\*** | **+\*** |

**Counts (36 cells):**
- positive 21, negative 15;
- **positive with CI excluding 0: 13** (raw / correlation / paper, plus all 12 euclidean
  cells);
- **negative with CI excluding 0: 9** (z / correlation × 3, z / cosine × 3,
  scale / cosine / paper, center / cosine / center720, center / cosine / z720).

With euclidean distance on both sides the gap is +0.016 to +0.023 at 224 px, but its
slope is small and not resolved (S ≈ +0.004 to +0.005, CI includes 0).

## G1 — gain axis and raw RSA (`results/step7/gain.csv`)

c_gain = Pearson over the 720 stimuli between model gain (Conv1) and voxel gain (V1),
mean over subjects; mean ± SD over seeds:

| Condition | 32 | 64 | 96 | 128 | 160 | 224 | UP224 | only_0-14 | notch_0-14 |
|---|---|---|---|---|---|---|---|---|---|
| Random Weights | 0.182 ± 0.049 | 0.198 | 0.209 | 0.215 | 0.220 | 0.226 ± 0.046 | 0.209 | 0.199 | 0.057 |
| Backprop | 0.120 ± 0.053 | 0.064 | 0.028 | 0.006 | −0.004 | −0.019 ± 0.026 | −0.019 | −0.019 | 0.012 |

**Applying G1:** Spearman(c_gain, raw RSA V1) over the 70 predefined cells = **0.975**
(≥ 0.6): **G1 met.**
- Without a rule: within Random 0.988; within BP 0.945; including the band-pass cells
  (n = 90) 0.920.
- c_gain reproduces the paper's pattern. Random's alignment with the brain's gain axis
  rises with resolution; BP's falls to zero. Removing the coarse band (notch) removes
  most of it for both.

**Gain and stimulus statistics** (Pearson over 720 stimuli; descriptive):

| Gain | vs mean luminance | vs RMS contrast |
|---|---|---|
| voxel gain V1 (sub-01 / 02 / 03; 224 px stimuli) | −0.28 / −0.27 / −0.23 | +0.13 / +0.20 / +0.13 |
| model gain, Random (32 / 224) | −0.46 / −0.44 | +0.39 / +0.45 |
| model gain, Backprop (32 / 224) | +0.24 / +0.40 | +0.00 / +0.00 |
| model gain, FA / PC / BN-cal / STDP (224) | +0.70 / +0.68 / +0.69 / +0.56 | ≈ 0 |

The brain's gain axis correlates negatively with luminance and positively with
contrast. Random's gain axis has the same signs; every trained condition, and the
BN-calibrated untrained network, has the opposite luminance sign.

## G2 — partial RSA on the gain RDMs (`results/step7/partial_rsa.csv`)

F = raw, D = correlation, B = paper. Check: the plain RSA computed in the same draws
equals step 5 (identity draw, 1e-12).

| Quantity | plain [CI] | partial [CI] |
|---|---|---|
| V1 gap NATIVE 32 | −0.0007 [−0.0109, +0.0090] | −0.0122 [−0.0199, −0.0052] (0/5) |
| V1 gap NATIVE 224 | +0.0304 [+0.0181, +0.0423] | **+0.0044 [−0.0045, +0.0135]** (4/5) |
| V1 gap UPSAMPLED 224 | +0.0347 [+0.0229, +0.0464] | +0.0097 [+0.0017, +0.0179] |
| S (224 − 32) | +0.0311 [+0.0239, +0.0384] | **+0.0166 [+0.0105, +0.0225]** |
| LOC BP − Random, NATIVE 224 | +0.0113 [+0.0025, +0.0210] | +0.0107 [+0.0038, +0.0179] |

**Applying G2:** the upper CI bound of the partial V1 gap at 224 px is 0.0135 > 0.010,
and S_partial = +0.0166 > 0.5 · S_paper = +0.0156. **G2 not met** on both parts. The
second part misses narrowly.

Partialling the gain RDMs:
- removes most of the V1 gap at 224 px (+0.030 → +0.004, CI includes 0);
- removes about half of the slope (+0.031 → +0.017, CI still above 0);
- leaves the LOC effect unchanged.

The gain axis carries a large share of the V1 effect but not all of the resolution
dependence.

## Summary of rules

| Rule | Result | Verdict |
|---|---|---|
| O1 | max \|Δ\| 1.1e-14 (models), 4.3e-14 (fMRI) | **met** |
| O2 | 0 violations in 36 cells; strong slope only in raw / correlation / paper | **met** |
| O3 | LOC positive 36/36, CI > 0 in 20/36 | **met** |
| G1 | Spearman(c_gain, raw RSA) = 0.975 (n = 70) | **met** |
| G2 | partial gap 224 +0.0044 [−0.0045, +0.0135]; S_partial +0.0166 [+0.0105, +0.0225] | **not met** |

## Limits

- N = 3 subjects; inference over stimuli; seeds averaged inside each draw.
- The multiverse is one set of 36 related analyses on the same data, not 36 independent
  tests; nothing is corrected for multiplicity.
- D is applied to both sides (operationalisation note 1). With euclidean distance on
  the model side only, step 6 gave the same sign at 224 px (+0.018 to +0.029).
- The gain axis is a one-dimensional summary. G1 shows that it orders the raw RSA
  values; G2 shows that removing it does not remove the whole slope.
- Only GAP features, Conv1 → V1 and Conv3 → LOC; one architecture and training set.
- The first multiverse run was stopped before producing results; only the second run's
  outputs exist.

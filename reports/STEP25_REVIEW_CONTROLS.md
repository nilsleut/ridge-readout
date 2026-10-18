# Step 25 — controls prompted by an external critique

**Exploratory.**
- Predictions and rules: `amendments/2026-10-16_step25-predictions.md`, commit `74853dc`
  (2026-10-18T11:30:57+02:00), before any step 25 script (`dbc4662`). The file name carries
  the date 2026-10-16; the commit timestamp is authoritative.
- Config hash `206fa875cca5`; every CSV in `results/step25/` carries `config_hash` and
  `git_head` (`dbc4662`).
- RSA convention as in steps 7 to 9 (functions imported): per subject on all ROI voxels,
  cross-run pairs, mean over subjects; the paper's 1000 stimulus resamples (seed 20261002)
  with paired differences. Ridge: `fit.fit_condition` unchanged, pre-specified
  configuration, D1 in counting mode with the plateau gate.
- N = 3 subjects; inference over stimuli (RSA) or seeds (ridge), not over the population.

## B1 — normalisation statistics in the bootstrap (code reading)

All statistics are computed once on the 720 stimuli; the draws only resample pairs of fixed
RDMs (`v12_lift.boot_pairs`):

| Quantity | Computed once at |
|---|---|
| channel z / center / scale (step 5) | `scripts/step5_resolution_rsa.py:66` (`transform`), RDM cached at `:104`; draws `:119` |
| F transformations (steps 6-8) | `scripts/step6_common.py:47-51` (`normalise`), used at `scripts/step7_multiverse.py:113`, `:173`; `scripts/step8_generalization.py:83`, `:134` |
| voxel normalisation B | `scripts/step7_multiverse.py:61-68` (`betas`), RDMs at `:141`; `scripts/step8_generalization.py:117` |
| mean profile / gain axis | `scripts/step7_gain.py:58` (`gain`), evaluated at `:133-135`, RDMs at `:191-194` |
| norms and centring (step 9) | `scripts/step9_separation.py:72` (`normf`), components at `:139`, `:211`, `:281` |
| draws | `boot_pairs` in `step5_resolution_rsa.py:119`, `step6_common.py:79`, `step7_multiverse.py:101`, `step7_gain.py:87`, `step8_generalization.py:101`, `step9_separation.py:159` |

Hence B2 was run.

## B2 / BN1 — statistics re-estimated inside each draw (`bootnorm.csv`)

First 200 resamples; V1, own CNN, NATIVE 224, Random − BP.

| Cell (F, D, B) | re-estimated: value [95% CI] | fixed (step 7, 1000 draws) | class |
|---|---|---|---|
| z, correlation, paper | −0.011 [−0.017, −0.005] | −0.011 [−0.017, −0.005] | below / below |
| center, euclidean, paper | +0.023 [+0.002, +0.039] | +0.023 [+0.004, +0.041] | above / above |
| center, correlation, center720 | +0.001 [−0.005, +0.007] | +0.001 [−0.005, +0.007] | includes / includes |

**BN1: met.** The classification is unchanged in all three cells. With euclidean distance
the re-estimated interval equals the fixed one over the same 200 draws, as expected:
centring does not change euclidean distances.

## M — mean-profile ablation (`mean_profile.csv`)

Gate: "full" reproduces the stored raw/correlation/paper cells of steps 7 and 8, max |Δ|
9.9e-17 over 224 matrices (≤ 1e-9: passed). Identity draw vs point estimates: 0.

V1, own CNN, Random − BP (value [95% CI], positive seeds of 5):

| Variant | NATIVE 32 | NATIVE 224 | UPSAMPLED 224 | slope S (224 − 32) |
|---|---|---|---|---|
| full | −0.001 [−0.011, +0.009] (3) | +0.030 [+0.018, +0.042] (5) | +0.035 [+0.023, +0.046] (5) | +0.031 [+0.024, +0.038] |
| ortho | −0.002 [−0.006, +0.002] (1) | −0.002 [−0.007, +0.003] (1) | +0.004 [−0.001, +0.009] (5) | +0.000 [−0.003, +0.003] |
| par | +0.006 [−0.016, +0.028] (3) | +0.012 [−0.011, +0.035] (3) | +0.000 [−0.023, +0.024] (2) | +0.006 [−0.007, +0.020] |

LOC, own CNN, BP − Random at NATIVE 224: full +0.011 [+0.003, +0.021]; ortho +0.005
[+0.001, +0.010]; par −0.002 [−0.020, +0.017].

ImageNet, V1 untrained − trained at 224 (ortho): alexnet −0.001 [−0.008, +0.005], vgg11_bn
−0.005 [−0.013, +0.002], resnet18 −0.009 [−0.014, −0.005], resnet50 −0.010 [−0.015, −0.005],
convnext_tiny +0.001 [−0.005, +0.006], vit_b_16 −0.025 [−0.036, −0.014]. The ortho slope
(448 − 224) has a CI above 0 in 4 of 5 models (alexnet +0.003, vgg11_bn +0.005, resnet18
+0.003, resnet50 +0.004; convnext_tiny +0.001 [−0.000, +0.002]). The par component varies
widely between models (from −0.067 to +0.059 at 224).

| Rule | Criterion | Result | Verdict |
|---|---|---|---|
| M1 | own ortho V1 gap NATIVE 224: upper CI ≤ 0.010 | upper bound +0.003 | **met** |
| M2 | S_ortho ≤ 0.5 · S_paper = 0.01555 | +0.0004 | **met** |
| M3 | own par V1 gap NATIVE 224 with CI above 0, and par slope CI above 0 | +0.012 [−0.011, +0.035]; slope [−0.007, +0.020] | **not met** |
| M4 | ImageNet ortho V1 gap 224: upper CI ≤ 0.010 in ≥ 4 of 6 | 6 of 6 | **met** |

**Interpretation rule, applied as written:** "M3 not met: it is not sufficient." The
orthogonal remainder carries neither the advantage nor the slope, but the parallel
component alone does not reproduce them with a resolved interval. No biological claim
follows.

## W — weighted angular term and component fit (`weighted_angle.csv`, descriptive)

RSA of W = 2 nᵢ nⱼ (1 − cos θᵢⱼ) with the brain W (F = center), next to N and A of step 9:

| Model | V1 W (untrained − trained) | V1 N (step 9) | V1 A (step 9) | LOC W (trained − untrained) |
|---|---|---|---|---|
| own CNN | +0.013 [−0.003, +0.030] | +0.017 [−0.004, +0.036] | −0.009 [−0.017, −0.002] | +0.011 [−0.009, +0.031] |
| alexnet | +0.016 [−0.006, +0.038] | +0.025 [+0.001, +0.051] | −0.020 [−0.031, −0.010] | +0.010 [−0.020, +0.045] |
| vgg11_bn | +0.026 [+0.004, +0.046] | +0.027 [+0.004, +0.050] | −0.019 [−0.029, −0.009] | +0.003 [−0.028, +0.038] |
| resnet18 | +0.007 [−0.009, +0.023] | +0.019 [+0.001, +0.037] | −0.028 [−0.037, −0.019] | +0.005 [−0.025, +0.036] |
| resnet50 | +0.004 [−0.011, +0.019] | +0.017 [−0.001, +0.034] | −0.026 [−0.036, −0.017] | +0.009 [−0.021, +0.040] |
| convnext_tiny | +0.005 [−0.008, +0.016] | +0.009 [−0.010, +0.026] | −0.003 [−0.007, +0.002] | −0.021 [−0.059, +0.015] |
| vit_b_16 | +0.016 [−0.009, +0.038] | +0.028 [+0.001, +0.053] | −0.032 [−0.046, −0.019] | +0.006 [−0.022, +0.036] |

The weighted angular term points the same way as the norm term (untrained ahead at V1) and
not as the unweighted angle; its CI excludes 0 in one model (vgg11_bn).

Rank correlations over the cross-run pairs (V1): own CNN, mean of 30 checkpoints at NATIVE
224: N–A 0.09, N–W 0.30, A–W 0.75. fMRI: N–A 0.00 / 0.01 / 0.01, N–W 0.34 / 0.35 / 0.36,
A–W 0.63 / 0.68 / 0.67 (sub-01 / 02 / 03).

Component fit (own CNN, V1, NATIVE 224; fMRI RDM = correlation distance on the B = paper
betas, the standard fMRI RDM; operationalisation fixed in the script before the run):

| R² | untrained | Backprop | untrained − Backprop [95% CI] |
|---|---|---|---|
| N | 0.0008 | 0.0001 | +0.0007 [−0.0003, +0.0017] |
| A | 0.0010 | 0.0009 | +0.0001 [−0.0005, +0.0008] |
| N + A | 0.0017 | 0.0012 | +0.0006 [−0.0006, +0.0018] |

All R² values are below 0.002, and no contrast has a CI excluding 0.

## F — ridge robustness

### F1 floor and F2 skills (`ridge_skill.csv`)

Gate: pooled median r reproduces `results/prereg/primary.csv`, max |Δ| 1.8e-13 (≤ 1e-9:
passed).

F1, intercept-only predictions (outer training mean): pooled median r −0.091 / −0.088 /
−0.097 (sub-01 / 02 / 03), mean −0.092. Per-fold r is undefined for a prediction that is
constant within each fold. MSE skill is 0 by construction. For comparison, the
permuted-assignment control of `results/prereg` (mean over subjects): −0.078, −0.075,
−0.074, −0.075 at g = 1, 2, 4, 8.

Random − BP, mean over subjects, mean ± SD over seeds, positive seeds of 5:

| g | MSE skill | per-fold r | pooled r |
|---|---|---|---|
| 1 | −0.0041 ± 0.0008 (0) | −0.0166 ± 0.0051 (0) | −0.0209 ± 0.0056 (0) |
| 2 | −0.0055 ± 0.0010 (0) | −0.0220 ± 0.0053 (0) | −0.0347 ± 0.0064 (0) |
| 4 | −0.0069 ± 0.0013 (0) | −0.0281 ± 0.0068 (0) | −0.0425 ± 0.0096 (0) |
| 8 | −0.0074 ± 0.0013 (0) | −0.0311 ± 0.0067 (0) | −0.0449 ± 0.0092 (0) |

**F-R1 (MSE skill): met. F-R2 (per-fold r): met.**

### F4 joint readout (`ridge_joint.csv`)

Median r, mean over subjects and seeds; bank alone from `results/prereg/
instruments_control_band.csv` (0.081 at g = 1, 0.106 at g = 4):

| Condition | joint g = 1 | joint − bank g = 1 | joint g = 4 | joint − bank g = 4 |
|---|---|---|---|---|
| Backprop | 0.068 | −0.013 | 0.090 | −0.015 |
| Random Weights | 0.060 | −0.021 | 0.064 | −0.042 |
| Random (BN-cal) | 0.059 | −0.022 | 0.063 | −0.043 |
| FA | 0.058 | −0.023 | 0.059 | −0.047 |
| PC | 0.061 | −0.020 | 0.064 | −0.041 |
| STDP | 0.059 | −0.022 | 0.063 | −0.043 |

**F-R4: met** (Backprop −0.013 and −0.015 ≤ 0.010). In every condition the joint readout
scored below the bank alone; with a single regularisation value per voxel the added network
features lowered the score.

### F3 permutation null (`ridge_permnull.csv`)

Pooled median r, 50 permutations of the feature rows, seed-0 checkpoints:

| Subject | g | condition | null mean [2.5%, 97.5%] | real |
|---|---|---|---|---|
| sub-01 | 1 | Random | −0.070 [−0.083, −0.040] | +0.052 |
| sub-01 | 1 | Backprop | −0.072 [−0.083, −0.046] | +0.088 |
| sub-01 | 4 | Random | −0.073 [−0.084, −0.041] | +0.060 |
| sub-01 | 4 | Backprop | −0.072 [−0.086, −0.046] | +0.118 |
| sub-02 | 1 | Random | −0.068 [−0.078, −0.056] | +0.073 |
| sub-02 | 1 | Backprop | −0.069 [−0.079, −0.062] | +0.096 |
| sub-02 | 4 | Random | −0.071 [−0.081, −0.060] | +0.071 |
| sub-02 | 4 | Backprop | −0.069 [−0.080, −0.055] | +0.120 |
| sub-03 | 1 | Random | −0.074 [−0.091, −0.038] | −0.014 |
| sub-03 | 1 | Backprop | −0.077 [−0.093, −0.048] | −0.004 |
| sub-03 | 4 | Random | −0.075 [−0.094, −0.037] | −0.021 |
| sub-03 | 4 | Backprop | −0.076 [−0.093, −0.048] | +0.008 |

**F-R3: met** (sub-01 and sub-02). In sub-03 (descriptive) the real values are also above
the 97.5th percentile of their nulls, although near or below zero.

### Lower-edge gate (`ridge_lower_edge.csv`)

Skill fits 4 hits, joint fits 3, permutation fits 3; all 10 Plateau, no fit excluded. The
primary analysis recorded 6 hits on the same model fits; the difference comes from near-ties
under single-threaded BLAS (as in step 3), and the pooled scores agree to 1.8e-13.

## Summary of rules

| Rule | Verdict |
|---|---|
| M1 | met |
| M2 | met |
| M3 | **not met** |
| M4 | met |
| Interpretation M | "M3 not met: it is not sufficient" |
| F-R1 | met |
| F-R2 | met |
| F-R3 | met |
| F-R4 | met |
| BN1 | met |

## Limits and run notes

- N = 3 subjects; nothing here is corrected for multiplicity; the rules of this step were
  written after all earlier results were known.
- The decompositions are definitions of components of the RDMs, not biological quantities.
- W and the component fit are descriptive; the R² values are small throughout.
- F4 uses a single regularisation value for the concatenated features, as in the
  pre-specified readout; a separate penalty per feature block was not part of the plan.
- In `ridge_permnull.csv` the null rows have `kind = null`, which pandas reads as missing;
  the macro filters in `scripts/export_macros.py` account for this.
- Runtime: B2 4 min, M 39 min, W 18 min, F4 11 min, F3 32 min. The F1/F2 block took 5.4 h
  instead of the estimated 0.2 h because single fits stalled (up to 4.9 h, consistent with
  the machine pausing). A second, parallel start of that block was stopped before it wrote
  any output; the results come from the first run only.
- `\FailedPredictions` in the paper keeps counting the analyses of steps 2-9 (5); M3 is
  counted in the new macro `\FailedPredictionsAll` (6). The paper text was not changed.

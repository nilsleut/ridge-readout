# Step 9 — separating gain magnitude from geometry

**Exploratory.**
- Predictions: `amendments/2026-10-10_step9-separation-predictions.md` (commit
  `6ef68f2`, before any step 9 script). Its operationalisation notes fix that F is
  applied to both the model and the fMRI side and define the (d) and (e) references.
- Config hash `206fa875cca5`.
- No new extraction: own CNN from `STEP5_GAP_DIR`, ImageNet models from
  `data/step8_gap`, fMRI from `arm_b_out`.
- RSA convention as in steps 7 and 8.
- N = 3 subjects; inference over stimuli. "Norm = gain, angle = geometry" is a
  definition, not a biological claim.

## Decomposition

Let Xc be the stimulus × unit matrix after centring every unit over the 720 stimuli
(F = center; for F = z each unit is also scaled to SD 1). For rows i and j with norms
nᵢ = ‖Xc_i‖ and cos θᵢⱼ = ⟨Xc_i, Xc_j⟩ / (nᵢ nⱼ):

‖Xc_i − Xc_j‖² = nᵢ² + nⱼ² − 2⟨Xc_i, Xc_j⟩
              = nᵢ² + nⱼ² − 2 nᵢ nⱼ cos θᵢⱼ
              = (nᵢ − nⱼ)² + 2 nᵢ nⱼ (1 − cos θᵢⱼ).

So the euclidean RDM on Xc is an exact function of two component RDMs: the
**norm RDM** Nᵢⱼ = |nᵢ − nⱼ| (how much stronger one stimulus drives the population than
the other) and the **angle RDM** Aᵢⱼ = 1 − cos θᵢⱼ (how differently the population
responds, independent of magnitude). The same decomposition is applied to the fMRI
betas.

## Checks

- **SEP1 (gate):** across 628 matrices (own CNN, ImageNet, fMRI; both F) the identity
  holds with a max relative error of **3.4e-12** (< 1e-8). No row has zero norm.
  **SEP1 met.**
- **(e):** the RSA of the full euclidean RDM on Xc equals the stored center / euclidean
  / center720 cells of steps 7 and 8 to **9.9e-17** (308 rows).
- Identity draw vs point estimates: 9.7e-17.

## Run notes

- The first run (commit `8d17349`) stopped at check (e): the step 8 resolution column
  is stored as text. Commit `f2145b3` fixed the lookup key, and the script was re-run
  from the start.
- The bootstrap estimate was 0.54 h; the run took about 1.5 h. The estimate timed only
  brain vectors, while each draw re-ranks 416 vectors. Both F were bootstrapped (below
  the 4 h limit either way).
- The controlling shell session was interrupted during the bootstrap; the process itself ran to
  completion (exit 0).

## Contrasts at the primary resolution (`results/step9/separation.csv`)

Own CNN: V1 Random − BP, LOC BP − Random, NATIVE 224. ImageNet: V1 untrained − trained,
LOC trained − untrained, 224. Values with 95% CI.

### F = center (primary)

| Model | V1 norm | V1 angle | LOC angle | LOC norm |
|---|---|---|---|---|
| own CNN | +0.017 [−0.004, +0.036] | **−0.009 [−0.017, −0.002]** | **+0.007 [+0.002, +0.012]** | −0.002 [−0.024, +0.021] |
| alexnet | **+0.025 [+0.001, +0.051]** | **−0.020 [−0.031, −0.010]** | **+0.025 [+0.019, +0.032]** | +0.018 [−0.004, +0.042] |
| vgg11_bn | **+0.027 [+0.004, +0.050]** | **−0.019 [−0.029, −0.009]** | **+0.027 [+0.020, +0.034]** | +0.008 [−0.015, +0.030] |
| resnet18 | **+0.019 [+0.001, +0.037]** | **−0.028 [−0.037, −0.019]** | **+0.031 [+0.024, +0.039]** | −0.012 [−0.034, +0.008] |
| resnet50 | +0.017 [−0.001, +0.034] | **−0.026 [−0.036, −0.017]** | **+0.032 [+0.024, +0.040]** | +0.005 [−0.017, +0.027] |
| convnext_tiny | +0.009 [−0.010, +0.026] | −0.003 [−0.007, +0.002] | **+0.011 [+0.004, +0.018]** | −0.001 [−0.024, +0.022] |
| vit_b_16 | **+0.028 [+0.001, +0.053]** | **−0.032 [−0.046, −0.019]** | **+0.017 [+0.009, +0.027]** | −0.006 [−0.027, +0.016] |

### F = z (descriptive point a)

| Model | V1 norm | V1 angle | LOC angle |
|---|---|---|---|
| own CNN | **+0.015 [+0.001, +0.029]** | **−0.022 [−0.030, −0.014]** | **+0.006 [+0.002, +0.010]** |
| alexnet | +0.019 [−0.005, +0.040] | **−0.036 [−0.049, −0.024]** | **+0.026 [+0.019, +0.033]** |
| vgg11_bn | +0.014 [−0.003, +0.031] | **−0.029 [−0.039, −0.018]** | **+0.027 [+0.020, +0.035]** |
| resnet18 | +0.010 [−0.003, +0.022] | **−0.023 [−0.030, −0.016]** | **+0.030 [+0.023, +0.038]** |
| resnet50 | +0.012 [−0.002, +0.027] | **−0.023 [−0.032, −0.015]** | **+0.032 [+0.024, +0.039]** |
| convnext_tiny | +0.007 [−0.011, +0.026] | −0.004 [−0.009, +0.001] | **+0.014 [+0.007, +0.021]** |
| vit_b_16 | +0.011 [−0.009, +0.032] | **−0.030 [−0.040, −0.020]** | **+0.018 [+0.010, +0.027]** |

## Rules

| Rule | Result | Verdict |
|---|---|---|
| SEP1 | max relative error 3.4e-12 | **met** |
| SEP2 | V1 norm (center) > 0 with CI > 0: own CNN **no** (+0.017, CI includes 0); ImageNet 4/6 (alexnet, vgg11_bn, resnet18, vit_b_16) | **not met** (own-CNN condition fails) |
| SEP3 | V1 angle (center) < 0 with CI < 0: own CNN yes; ImageNet 5/6 (all but convnext_tiny) | **met** |
| SEP4 | LOC angle (center) > 0 with CI > 0: own CNN yes; ImageNet 6/6 | **met** |
| Interpretation | SEP2 and SEP3 are not both met | **the V1 ranking remains unidentifiable, without attribution to gain or geometry** |

**Reading within the rule.**
- The rule forbids an attribution, and none is made.
- Descriptively:
  - the norm contrast points the same way (untrained ahead) in all seven models, with
    a CI excluding 0 in four of the ImageNet models;
  - for the own CNN that CI includes 0 under F = center (it excludes 0 under F = z,
    which is descriptive only);
  - the angle contrast (trained ahead at V1) has a CI excluding 0 in six of seven
    models;
  - at LOC the angle contrast favours trained networks in all seven, while the norm
    component carries no LOC effect anywhere.

## Resolution slopes of the V1 contrast (descriptive point b; `results/step9/separation_slopes.csv`)

Own CNN: contrast(224) − contrast(32). ImageNet: contrast(448) − contrast(224); vit_b_16
excluded.

| Model | norm, center | angle, center | norm, z | angle, z |
|---|---|---|---|---|
| own CNN | −0.004 [−0.013, +0.005] | +0.001 [−0.004, +0.006] | −0.002 [−0.015, +0.009] | **−0.008 [−0.015, −0.000]** |
| alexnet | −0.001 [−0.007, +0.003] | **+0.004 [+0.001, +0.006]** | +0.002 [−0.007, +0.010] | **+0.005 [+0.001, +0.009]** |
| vgg11_bn | +0.002 [−0.003, +0.007] | **+0.008 [+0.004, +0.011]** | +0.002 [−0.006, +0.009] | **+0.007 [+0.004, +0.010]** |
| resnet18 | +0.001 [−0.006, +0.008] | **+0.009 [+0.006, +0.013]** | −0.003 [−0.010, +0.004] | **+0.007 [+0.004, +0.009]** |
| resnet50 | +0.001 [−0.007, +0.008] | **+0.010 [+0.007, +0.014]** | −0.002 [−0.010, +0.007] | **+0.009 [+0.006, +0.013]** |
| convnext_tiny | −0.000 [−0.003, +0.002] | +0.001 [−0.000, +0.002] | −0.001 [−0.012, +0.010] | +0.001 [−0.001, +0.002] |

The slope of the ImageNet models above their training resolution (GEN4, step 8) sits
in the **angle** component: in four of five models the angle contrast grows from 224 to
448 with a CI above 0, under both F, while no norm slope excludes 0. The trained models'
geometric advantage at V1 shrinks as the input is enlarged beyond 224 px.

For the own CNN neither component has a slope under F = center, matching step 7, where
the centred cells showed no strong slope.

**|S_gen| from step 8 (descriptive point c), z / correlation / paper:**

| Model | \|S_gen\| z | [CI] | \|S_gen\| raw |
|---|---|---|---|
| alexnet | 0.0055 | [+0.0029, +0.0080] | 0.0085 |
| vgg11_bn | 0.0099 | [+0.0065, +0.0137] | 0.0078 |
| resnet18 | 0.0062 | [+0.0042, +0.0084] | 0.0072 |
| resnet50 | 0.0075 | [+0.0052, +0.0101] | 0.0057 |
| convnext_tiny | 0.0010 | [−0.0011, +0.0030] | 0.0018 |

## Norm and stimulus statistics (descriptive point d; `results/step9/norm_stimulus_stats.csv`)

Pearson over the 720 stimuli between nᵢ (V1 layer, F = center) and stimulus mean
luminance / RMS contrast; mean over seeds, conditions and resolutions.

| Source | luminance | RMS contrast |
|---|---|---|
| own CNN, Random | +0.09 | −0.23 |
| own CNN, Backprop | −0.01 | −0.24 |
| own CNN, FA / PC / STDP / BN-cal | +0.12 to +0.20 | −0.21 to −0.27 |
| ImageNet, untrained (5 conv models) | +0.09 to +0.27 | −0.23 to −0.55 |
| ImageNet, trained | −0.08 to +0.02 | −0.15 to −0.35 |
| fMRI V1, sub-01 / sub-02 / sub-03 | +0.01 / +0.03 / −0.01 | +0.02 / −0.07 / −0.02 |

The model norms are negatively related to RMS contrast in every model and condition.
The voxel norms are essentially unrelated to either statistic (|r| ≤ 0.07). The brain's
gain magnitude in this sense is not a luminance or contrast effect. That the norm
contrast still favours untrained networks is therefore not explained by these two
statistics.

## Limits

- N = 3 subjects; inference over stimuli; no multiplicity correction across models,
  components and F.
- The decomposition is exact for euclidean distance on Xc only. It does not decompose
  correlation-distance RSA (the paper's convention), which normalises each row before
  comparing.
- The angle RDM equals cosine distance on Xc, i.e. correlation distance across
  stimuli computed on unit-centred features. It is not the same as the row-centred
  correlation distance used in the paper.
- vit_b_16 at 224 only; no slope for it.
- One set of trained weights per ImageNet model.

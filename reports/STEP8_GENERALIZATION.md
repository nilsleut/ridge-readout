# Step 8 — generalization to ImageNet models, and paper infrastructure

**Exploratory.**
- Predictions: `amendments/2026-10-09_step8-generalization-predictions.md` (commit
  `10d302b`, before any step 8 script).
- Operationalisation notes there (fixed before runs): vit_b_16 runs at R = 224 only;
  layer access and hooks; untrained = uncalibrated BN.
- Config hash `206fa875cca5`.
- RSA convention and the F, D, B, gain and c_gain definitions as in step 7 (functions
  reused).
- N = 3 subjects; inference over stimuli; one set of trained weights per model.

## Run notes

- **Extraction** (`scripts/step8_extract_pretrained.py`).
  - All fixed layers present.
  - Runtime estimate 2.21 h (below 6 h), so all resolutions were run.
  - Actual time: 78 min.
  - 128 feature files: 5 models × 4 conditions × 3 R × 2 ROIs, plus vit_b_16 at 224 ×
    4 × 2.
- **vit_b_16 at R = 224 only.** torchvision's ViT asserts a 224 px input (note 1).
  Hence GEN1 uses n = 64, not 72, and vit_b_16 has no S_gen.
- **Sanity check** (`results/step8/sanity_top1.csv`): the trained models' top-1 classes
  on the first 20 stimuli are plausible. Acorn, ant, American alligator and artichoke
  are correct for all six models; Granny Smith, impala and aircraft carrier for five or
  six. Weights and normalisation are therefore in order. No criterion attached.
- **Analysis** (`scripts/step8_generalization.py`): 1,536 model RDMs for the point
  estimates; bootstrap over 352 model and 18 brain vectors; identity draw vs point
  estimates max |Δ| 1.2e-16.

## Bootstrap cells: V1 gap (untrained − trained), LOC (trained − untrained), S_gen

Values with 95% CI.

| Model | Cell | V1 gap 224 | S_gen (448 − 224) | LOC 224 |
|---|---|---|---|---|
| alexnet | raw / corr / paper | +0.005 [−0.005, +0.015] | **+0.008 [+0.006, +0.011]** | +0.028 [+0.019, +0.038] |
| | z / corr / paper | **−0.019 [−0.030, −0.007]** | +0.005 [+0.003, +0.008] | +0.028 [+0.022, +0.035] |
| | raw / eucl / paper | **+0.031 [+0.004, +0.056]** | +0.004 [−0.001, +0.009] | +0.009 [−0.024, +0.043] |
| | center / corr / center720 | +0.001 [−0.007, +0.010] | +0.003 [+0.001, +0.004] | +0.030 [+0.023, +0.036] |
| vgg11_bn | raw / corr / paper | **+0.019 [+0.008, +0.031]** | **+0.008 [+0.006, +0.010]** | +0.030 [+0.021, +0.040] |
| | z / corr / paper | **−0.015 [−0.026, −0.005]** | +0.010 [+0.006, +0.014] | +0.032 [+0.025, +0.039] |
| | raw / eucl / paper | **+0.043 [+0.017, +0.067]** | +0.008 [+0.002, +0.015] | +0.005 [−0.027, +0.040] |
| | center / corr / center720 | **−0.012 [−0.022, −0.001]** | +0.009 [+0.006, +0.012] | +0.033 [+0.027, +0.040] |
| resnet18 | raw / corr / paper | **−0.013 [−0.024, −0.002]** | **+0.007 [+0.005, +0.009]** | +0.032 [+0.023, +0.042] |
| | z / corr / paper | **−0.016 [−0.023, −0.009]** | +0.006 [+0.004, +0.008] | +0.030 [+0.023, +0.037] |
| | raw / eucl / paper | **+0.020 [+0.000, +0.037]** | +0.008 [+0.002, +0.014] | +0.001 [−0.030, +0.034] |
| | center / corr / center720 | **−0.022 [−0.030, −0.015]** | +0.008 [+0.006, +0.011] | +0.033 [+0.026, +0.040] |
| resnet50 | raw / corr / paper | −0.002 [−0.013, +0.008] | **+0.006 [+0.003, +0.008]** | +0.035 [+0.026, +0.044] |
| | z / corr / paper | **−0.015 [−0.023, −0.008]** | +0.008 [+0.005, +0.010] | +0.034 [+0.027, +0.041] |
| | raw / eucl / paper | +0.017 [−0.002, +0.034] | +0.008 [+0.001, +0.015] | +0.015 [−0.017, +0.046] |
| | center / corr / center720 | **−0.016 [−0.023, −0.009]** | +0.007 [+0.005, +0.009] | +0.035 [+0.027, +0.042] |
| convnext_tiny | raw / corr / paper | **+0.016 [+0.007, +0.024]** | **+0.002 [+0.001, +0.003]** | +0.006 [−0.004, +0.016] |
| | z / corr / paper | +0.001 [−0.005, +0.006] | +0.001 [−0.001, +0.003] | +0.017 [+0.010, +0.023] |
| | raw / eucl / paper | +0.013 [−0.002, +0.026] | +0.003 [+0.000, +0.005] | −0.023 [−0.061, +0.013] |
| | center / corr / center720 | −0.002 [−0.007, +0.004] | +0.002 [+0.000, +0.003] | +0.013 [+0.006, +0.020] |
| vit_b_16 | raw / corr / paper | −0.010 [−0.025, +0.005] | — | +0.015 [+0.005, +0.025] |
| | z / corr / paper | **−0.034 [−0.049, −0.020]** | — | +0.018 [+0.010, +0.026] |
| | raw / eucl / paper | **+0.033 [+0.005, +0.059]** | — | −0.000 [−0.029, +0.029] |
| | center / corr / center720 | **−0.039 [−0.055, −0.024]** | — | +0.020 [+0.012, +0.029] |

## Rules

**GEN1 — met.** Spearman(c_gain, raw RSA V1) over 64 cells = **0.79** (≥ 0.6).
Mean c_gain per condition:

| Model | trained | untrained s0 | untrained s1 | untrained s2 |
|---|---|---|---|---|
| alexnet | −0.083 | −0.094 | −0.015 | −0.075 |
| vgg11_bn | −0.035 | +0.025 | +0.068 | −0.025 |
| resnet18 | +0.091 | +0.161 | −0.036 | −0.045 |
| resnet50 | +0.024 | +0.099 | −0.043 | −0.011 |
| convnext_tiny | −0.075 | −0.048 | −0.010 | −0.037 |
| vit_b_16 | −0.081 | −0.110 | −0.097 | −0.109 |

The ImageNet models are far less aligned with the brain's gain axis than the custom
CNN's untrained network (step 7: +0.18 to +0.23). Across cells, c_gain still orders raw
RSA.

**GEN2 — met (4/6).** At 224, two bootstrap cells exclude 0 in opposite directions
for:
- alexnet: euclidean + vs z −;
- vgg11_bn: raw + and euclidean + vs z − and center720 −;
- resnet18: euclidean + vs raw, z and center720 −;
- vit_b_16: euclidean + vs z and center720 −.

Not for resnet50 (no positive CI) or convnext_tiny (no negative CI).

**GEN3 — not met (0/6).** LOC trained − untrained at 224 has a CI above 0 in all four
cells for no model. The failing cell is the same everywhere: **raw / euclidean / paper**,
where the CI includes 0 for all six models (as for the custom CNN in steps 6 and 7).
convnext_tiny also fails in raw / correlation / paper.
- In the three correlation-type cells, LOC is positive with CI above 0 for all models,
  except convnext_tiny in raw / correlation / paper (+0.006, CI includes 0).

**GEN4 — not met.**
- Five models qualify (S_gen in raw / correlation / paper with a CI excluding 0; all
  positive).
- In none of them is |S_gen(z)| ≤ 0.5 · |S_gen(raw)|:

| Model | S_gen raw | S_gen z |
|---|---|---|
| alexnet | +0.0085 | +0.0055 |
| vgg11_bn | +0.0078 | +0.0099 |
| resnet18 | +0.0072 | +0.0062 |
| resnet50 | +0.0057 | +0.0075 |
| convnext_tiny | +0.0018 | +0.0010 (CI includes 0) |

- Beyond the training resolution, the untrained − trained gap grows under z as well,
  and also under center720 and euclidean (all positive). Unlike the custom CNN (step 7,
  O2), the slope of these models is **not** offset-specific.

## Sign tables at 224 (`results/step8/generalization_signs.csv`)

The 4 bootstrap cells are classified by CI (+ / − / 0), the other 32 by the sign of the
point estimate.

| Model | + | − | 0 | raw/corr | z/corr | raw/eucl | center/corr/c720 |
|---|---|---|---|---|---|---|---|
| alexnet | 17 | 17 | 2 | 0 | − | + | 0 |
| vgg11_bn | 22 | 14 | 0 | + | − | + | − |
| resnet18 | 14 | 22 | 0 | − | − | + | − |
| resnet50 | 13 | 21 | 2 | 0 | − | 0 | − |
| convnext_tiny | 25 | 8 | 3 | + | 0 | 0 | 0 |
| vit_b_16 | 14 | 21 | 1 | 0 | − | + | − |

## Gain axis and stimulus statistics (`results/step8/generalization_gain.csv`)

Pearson over 720 stimuli, R = 224.

| Model | trained: luminance / contrast | untrained (mean of 3): luminance / contrast |
|---|---|---|
| alexnet | +0.13 / +0.09 | +0.34 / +0.48 |
| vgg11_bn | +0.06 / +0.21 | +0.22 / +0.51 |
| resnet18 | −0.10 / +0.25 | +0.08 / +0.51 |
| resnet50 | −0.10 / +0.21 | +0.14 / +0.54 |
| convnext_tiny | +0.49 / −0.06 | +0.48 / −0.12 |
| vit_b_16 | +0.13 / +0.05 | +0.71 / +0.03 |
| brain V1 (sub-01/02/03) | −0.28 / −0.27 / −0.23 (luminance) | +0.13 / +0.20 / +0.13 (contrast) |

The untrained conv nets track RMS contrast (+0.48 to +0.54) much more strongly than
the trained ones. The luminance sign varies across seeds for the untrained ResNets. No
model reproduces the brain's negative luminance correlation consistently.

## Paper infrastructure

- `scripts/export_macros.py` writes `paper/numbers.tex` (**83 macros**, letters-only
  names, one comment per macro with source, filter, column and rounding) and
  `paper/numbers_map.csv`.
- It reads only `results/` and fails if a source is missing or a value filter does not
  match exactly one row. Cell counts use explicit count filters.
- `paper/main.tex` is a skeleton (sections as specified, no prose, AI-disclosure
  placeholder) and compiles with pdflatex (MiKTeX).
- Build artefacts are gitignored.

## Summary

| Rule | Result | Verdict |
|---|---|---|
| GEN1 | Spearman 0.79, n = 64 | **met** |
| GEN2 | 4/6 models with opposite CI-excluding V1 gaps at 224 | **met** |
| GEN3 | 0/6 models with LOC CI > 0 in all 4 cells (euclidean fails for all) | **not met** |
| GEN4 | 5 qualifying models, 0 with \|S_z\| ≤ 0.5 \|S_raw\| | **not met** |

## Limits

- One set of trained weights per model; three untrained inits.
- vit_b_16 at 224 only.
- Layer choices fixed a priori, not optimised.
- ImageNet models were trained at 224 px. Their "slope" (448 − 224) is therefore
  measured above the training resolution, not across it as for the 32 px custom CNN.
- N = 3 subjects; inference over stimuli; no multiplicity correction across models
  and cells.

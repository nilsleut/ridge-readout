# Step 8 generalization predictions (fixed before runs)

Date: 2026-10-09. Committed before any step 8 script exists or runs. Exploratory.

RSA convention as in step 7: per subject, then averaged; all voxels of the ROI;
cross-run pairs; stimulus bootstrap with 1000 resamples, seed 20261002, paired
differences; distance D applied to both the model and the fMRI side. Definitions of F,
D, B, gain and c_gain are taken verbatim from
`amendments/2026-10-09_step7-offset-predictions.md` and `scripts/step7_*.py`.

## Models and conditions

- **Models** (torchvision, weights IMAGENET1K_V1): alexnet, vgg11_bn, resnet18,
  resnet50, convnext_tiny, vit_b_16.
- **Conditions:**
  - trained: ImageNet weights;
  - untrained: weights=None, `torch.manual_seed(s)` immediately before construction,
    s ∈ {0, 1, 2}; BN running statistics left at their defaults, i.e. uncalibrated.
- Eval mode.

## Layers, fixed before the run

GAP over the spatial dimensions, or mean over tokens without the class token.

| Model | V1 layer | LOC layer |
|---|---|---|
| alexnet | features[1] | features[12] |
| vgg11_bn | features[2] | last MaxPool in features |
| resnet18 | maxpool | layer4 |
| resnet50 | maxpool | layer4 |
| convnext_tiny | features[0] | features[7] |
| vit_b_16 | encoder.layers[0] | encoder.layers[-1] |

## Input and resolutions

- RGB; resize of the shorter side to R; CenterCrop R; ImageNet normalisation.
- Resolutions R ∈ {112, 224, 448}; training resolution 224.
- Slope S_gen = Gap(448) − Gap(224), with Gap = untrained − trained (untrained averaged
  over the 3 init seeds).

## Cells

- Bootstrap cells (F, D, B): (raw, correlation, paper), (z, correlation, paper),
  (raw, euclidean, paper), (center, correlation, center720).
- All 36 cells as point estimates.

## Predictions

**GEN1 (mechanism).** Spearman(c_gain, raw RSA in cell raw/correlation/paper) over all
cells model × {trained, untrained s0–s2} × R at V1 (n = 72) ≥ 0.6.

**GEN2 (V1 not identifiable).** In at least 4 of 6 models, at R = 224, two of the 4
bootstrap cells have gap CIs that exclude 0 in opposite directions.

**GEN3 (LOC robust).** trained − untrained at LOC, R = 224, is > 0 with a CI above 0 in
all 4 bootstrap cells, for at least 5 of 6 models.

**GEN4 (slope offset-specific).**
- Qualifying models: those whose S_gen in cell raw/correlation/paper has a CI excluding
  0.
- For each qualifying model: |S_gen| in z/correlation/paper ≤ 0.5 · |S_gen| in
  raw/correlation/paper.
- Met if this holds for all qualifying models; if no model qualifies, GEN4 is not
  testable.

## Limits

One set of trained weights per model; inference over stimuli; N = 3 subjects.

---

## Operationalisation notes (added by the executor before any run)

Fixed before any step 8 script exists, after checking the models only (no stimuli, no
brain data).

1. **vit_b_16 runs at R = 224 only.** torchvision's vit_b_16 asserts a 224 × 224 input
   ("Wrong image height! Expected 224"). Interpolating its position embeddings would
   change the model, so R = 112 and R = 448 are not run for it. Consequences:
   - GEN1 is evaluated on n = 64 cells, not 72;
   - S_gen does not exist for vit_b_16, so it cannot qualify for GEN4;
   - GEN2 and GEN3 (R = 224) are unaffected.
2. **Layer access.** Layers are addressed by index or attribute exactly as written above
   (vit_b_16's `encoder.layers[0]` is the module named `encoder_layer_0`). A forward
   hook takes the module output, cloned, because the alexnet and vgg11_bn V1 layers are
   in-place ReLUs.
3. **Untrained is uncalibrated.** BN running statistics are left at their defaults (mean
   0, variance 1); convnext_tiny and vit_b_16 use LayerNorm, which has no running
   statistics.
4. **Trained weights.** resnet50 trained = IMAGENET1K_V1 (not V2); the same version
   label for all models, as specified.
5. **Stimulus statistics.** Luminance and RMS contrast for the descriptive gain
   correlations are computed as in step 7 (Y = 0.299 R + 0.587 G + 0.114 B on the
   pre-normalisation tensor), here with the step 8 transform at R = 224.

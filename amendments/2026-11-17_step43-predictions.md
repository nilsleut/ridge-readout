# Step 43: NSD extensions, predictions (fixed before scripts and runs)
Date: 2026-11-17. Configuration: config_step43.yaml, config hash e0a00a888b48.
Data and conventions as in step 42: 8 subjects, 515 images, k = 3, V1 primary, LO-obj secondary; Δ = untrained − trained, mean over seeds (own CNN: 5 seeds; ImageNet: one weight set each); group inference by stimulus bootstrap of the 8-subject mean (1000 resamples, seed 20261002) and one-sided sign-flip test over the 8 subjects (256 assignments).
Note: the step 42 results for 8 cells were known when these predictions were written.

M. Multiverse on NSD (own CNN, V1): the same 36 cells as in arXiv:2611.15533 (F ∈ {raw, center, scale, z} × D ∈ {correlation, cosine, Euclidean} × B ∈ {as in step 42, voxels centred over the 515 images, voxels z-scored over the 515 images}), counted as distinct analyses with the same rule as in the paper.
M1: the interval of the 8-subject mean lies above 0 in at most 2 of the distinct analyses.
M2: the interval lies below 0 in at least half of the distinct analyses.

I. ImageNet models on NSD: alexnet, vgg11_bn, resnet18, resnet50, convnext_tiny, vit_b_16, trained versus untrained (same architectures, layers and untrained initialisation as in arXiv:2611.15533).
I1 (V1, angle component A): Δ < 0 for at least 5 of 6 architectures (8-subject mean).
I2 (V1, paper cell: raw, correlation): Δ ≤ 0 for at least 4 of 6 architectures (8-subject mean).

R. Resolution on NSD (own CNN, paper cell and standardised cell, V1): evaluation at the six resolutions of arXiv:2608.12408, slope defined as in arXiv:2611.15533. Reported without prediction.

NC. Leave-one-subject-out lower bound of the noise ceiling (as in the paper) for V1 and LO-obj, correlation distance, k = 3 and k = 1. Descriptive.

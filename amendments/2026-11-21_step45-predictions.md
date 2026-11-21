# Step 45: coarse spatial luminance, predictions (fixed before scripts and runs)
Date: 2026-11-21. Configuration: config_step45.yaml, config hash 347f9a0110a8.
Motivation: step 44, W2 not met (global L and C removed 46 % of the THINGS-720 V1 advantage); arXiv:2608.12408 located the V1 gap in the coarse band (only_0-14, including the mean). This is the last analysis before the manuscript revision; its outcome is reported whatever it is.
Covariate LP-RDM: for each image, the Rec. 709 grey image after Resize(224)/CenterCrop(224) (as in step 44), filtered to the only_0-14 band of arXiv:2608.12408 including the mean, area-downsampled to 28 × 28; LP-RDM = Euclidean distance between these vectors.
Partial Spearman as in step 44, on the same pair sets. Δ = Random − BP, paper cell, mean over seeds 0–4, then subjects.
W6: in THINGS-720 at V1, partialling L-RDM, C-RDM and LP-RDM together reduces Δ to ≤ +0.010.
W7: in THINGS-720 at V1, the correlation of the V1 RDM with LP-RDM, divided by the LOSO lower bound, is larger than the same ratio in THINGS test images (k = 1) and in NSD (k = 1).
Interpretation fixed in advance: W6 and W7 met, together with W3 and W4 of step 44 → the THINGS-720 advantage is attributed to coarse luminance structure dominating its V1 RDM, combined with the untrained network's greater sensitivity to it. Otherwise → not identified; the explained fraction is reported.
Reported without prediction: Δ after LP-RDM alone; the same partialling at LOC, in THINGS test images and in NSD; ρ(Random, LP-RDM) and ρ(BP, LP-RDM) per dataset; per-subject values in THINGS-720; Δ after L + C + LP + T.

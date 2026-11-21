# Step 44: why does the THINGS main set show an untrained advantage? Predictions (fixed before scripts and runs)
Date: 2026-11-21. Configuration: config_step44.yaml, config hash 451099bd833e.
Facts known before: THINGS-720 paper cell Δ = +0.030 (Random − BP, raw units, correlation distance, cross-run pairs); THINGS test images Δ ≈ 0 (k = 1: −0.0002) and NSD Δ < 0 (k = 1 and 3). Luminance RDM at THINGS-720 V1: ρ = 0.054; LOSO lower bound 0.051. The 720 images were shown in one session per subject; sub-01 and sub-02 share the presentation order.
Definitions: L-RDM = |L_i − L_j| (mean luminance, definition of arXiv:2608.12408); C-RDM = |C_i − C_j| (RMS contrast); T-RDM = |p_i − p_j| (presentation position within the session, in trials, per subject). Partial Spearman: rank-transform all RDM vectors over the pair set, regress model ranks and brain ranks on the covariate ranks, correlate the residuals. Δ as in the paper cell; mean over seeds 0–4, then subjects.

W1 (sampling): over 1000 random subsets of 100 of the 720 images (seed 20261119), Δ exceeds −0.0002 (the THINGS test-image value at k = 1) in at least 95 % of subsets.
W2 (luminance): in THINGS-720, partialling L-RDM and C-RDM reduces Δ to ≤ +0.010.
W3 (model property): Spearman between model RDM and L-RDM is larger for Random than for BP in 5 of 5 seeds, in THINGS-720, THINGS test images and NSD.
W4 (data property): the V1 correlation with L-RDM, divided by the dataset's LOSO lower bound, is larger in THINGS-720 than in THINGS test images (k = 1) and than in NSD (k = 1).
W5 (order): partialling T-RDM changes Δ in THINGS-720 by less than 0.005 in absolute value.
Interpretation fixed in advance: W2, W3 and W4 met → the advantage is attributed to luminance dominance of the THINGS-720 V1 RDM combined with the untrained network's greater luminance sensitivity. W5 not met → presentation order contributes. W1 not met → the test-image difference is compatible with sampling variation of 100 images. Any other pattern → reported as not identified.
Reported without prediction: per-subject Δ in THINGS-720 (sub-01, sub-02 vs sub-03), with and without each partialling; Spearman between presentation position and L per subject; Spearman of T-RDM with the V1 RDM and with each model RDM; all W quantities for LOC and LO-obj.

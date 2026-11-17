# Step 42: NSD replication, predictions (fixed before scripts, data loading and runs)
Date: 2026-11-17. Configuration: config_step42.yaml, config hash da1cbb33b6d7. Setup: reports/STEP42_NSD_SETUP.md.

Data: NSD, 8 subjects, the 515 images seen three times by every subject; prepared single-trial betas (func1pt8mm, betas_fithrf_GLMdenoise_RR), scaled as in the manual and z-scored per voxel within session. Only sessions 1–30 are loaded (all 1545 trials of the 515 images lie there, for all 8 subjects). ROIs: V1 (prf-visualrois V1v + V1d), primary; LO-obj (streams 'lateral' ∩ floc_objectstval > 0, threshold fixed at t > 0 following the manual's convention for the floc ROIs), secondary; 'lateral' (streams label 6 alone), reported as a sensitivity variant without prediction. All ROI voxels.
LO-obj is constructed here because the manual provides no object-selective floc ROI; it is fixed before any RSA. The contrast underlying floc_objectstval is the objects domain (categories car and instrument) against all other stimuli of the floc experiment (the characters, bodies, faces and places domains: word, number, body, limb, adult, child, corridor and house).
Models: own CNN, Random (untrained, uncalibrated BN) and BP, seeds 0–4, Conv1 after max-pool, 224 px with the same transform as for THINGS, global average pooling as in the paper cell.
Brain patterns: k = 3 (average of the three presentations, primary) and k = 1 (each presentation separately, the three RSA values averaged; pairs whose chosen presentations fall in the same run are excluded).
Cells: model units raw or standardised over the 515 images; correlation or Euclidean distance on both sides (brain side: patterns as defined above, correlation or Euclidean distance). Spearman over all pairs.
Per subject: mean over seeds. Contrast Δ = Random − BP.
Inference: across subjects, one-sided sign-flip test over the 8 subject means (256 assignments) in the predicted direction; plus a stimulus bootstrap (1000 resamples, seed 20261002) of the 8-subject mean.
P1 (paper cell: raw, correlation, V1, k = 3): mean Δ ≤ 0 (point estimate).
P2 (standardised, correlation, V1, k = 3): Δ < 0 in at least 6 of 8 subjects, sign-flip p ≤ 0.05, and the bootstrap interval of the mean excludes 0.
P3 (standardised, correlation, LO-obj, k = 3): Δ < 0 in at least 6 of 8 subjects and sign-flip p ≤ 0.05.
P4 (raw, correlation, V1): mean Δ(k = 3) ≤ mean Δ(k = 1).
All other cells (Euclidean, raw and standardised; k = 1 for every cell; higher ROI for every cell) are reported without predictions.
The pilot (subj01) is part of this analysis; no prediction is changed after the pilot.

# Step 25 predictions (fixed before runs)
Motivation: an external critique of the draft. All analyses are exploratory. Conventions as in steps 7 to 9: per subject, then mean over subjects; all voxels of the ROI for RSA; cross-run pairs; stimulus bootstrap with 1000 resamples (seed 20261002) and paired differences; normalisation statistics estimated once on the 720 stimuli and held fixed across draws unless stated otherwise. Contrasts: own CNN untrained (uncalibrated BN) minus Backprop at V1; ImageNet models untrained (mean over 3 init seeds) minus trained at V1. S_paper = +0.0311 (resolution slope, raw/correlation/paper, own CNN).

## M: mean-profile ablation (RSA)
For raw GAP features X (720 x units): mu = unit means over the 720 stimuli, mu_c = mu minus its mean over units, u = mu_c / ||mu_c||; x~_i = row i centred across units; c_i = <x~_i, u>; r_i = x~_i - c_i u.
Variants: "full" = correlation distance on x_i (the paper cell, must reproduce results/step7 on 1e-9, else stop); "ortho" = correlation distance on r_i; "par" = RDM |c_i - c_j|. The same decomposition is applied to the beta patterns (voxel means over the 720 stimuli, B = paper), and each model component is compared with the matching brain component.
Evaluation: own CNN at NATIVE 32, NATIVE 224 and UPSAMPLED 224, 30 checkpoints, V1 (primary) and LOC (descriptive, Backprop minus untrained); ImageNet models at 224 and 448 (vit_b_16 at 224 only). Bootstrap CIs for all contrasts and slopes (S = gap(224) - gap(32) for the own CNN; gap(448) - gap(224) for ImageNet).
M1: own CNN, ortho, V1 gap at NATIVE 224: upper CI bound <= 0.010.
M2: own CNN, ortho, slope S_ortho <= 0.5 * S_paper.
M3: own CNN, par, V1 gap at NATIVE 224 > 0 with CI above 0, and the par slope has a CI above 0.
M4: ImageNet, ortho, V1 gap at 224 with upper CI bound <= 0.010 in at least 4 of 6 models.
Interpretation rule: M1, M2 and M3 met: in these data the raw-RSA advantage and its slope are carried by the component parallel to the mean profile and absent from the orthogonal remainder. M1 or M2 not met: the parallel component is not necessary. M3 not met: it is not sufficient. No biological claim follows. Prior expectation: uncertain, since the rank-regression test G2 failed.

## W: weighted angular term and component fit (descriptive, no decision rule, fixed in advance)
With F = center: n_i = ||X_c,i||, A_ij = 1 - cos(theta_ij), N_ij = |n_i - n_j|, W_ij = 2 n_i n_j (1 - cos(theta_ij)), the angular term of the identity d^2 = (n_i - n_j)^2 + W_ij. Report the V1 contrast (own CNN at NATIVE 224; ImageNet at 224) and the LOC contrast (Backprop/trained minus untrained) for the RSA of W with the matching brain W, with CIs, next to the existing N and A values of step 9. Report rank correlations between the N, A and W RDMs over cross-run pairs (own CNN: mean over checkpoints; fMRI: per subject). Component fit: for each checkpoint, R^2 of an ordinary least squares regression of the ranks of the fMRI RDM (cross-run pairs) on the ranks of the model N RDM alone, A RDM alone, and N and A together; report the untrained-minus-Backprop contrast of each R^2 with CIs (own CNN, V1, NATIVE 224).

## F: ridge robustness (sub-01, sub-02, sub-03; Arm A; Conv1; NATIVE 224; reliable V1 voxels; pre-specified configuration; fit_condition unchanged; D1 plateau gate in counting mode; default random folds)
F1 floor (descriptive): intercept-only predictions (mean of the outer training fold): pooled r and per-fold r (median over reliable voxels, mean over subjects), reported next to the permuted-assignment control values of results/prereg.
F2 skills: for every fit, MSE skill per voxel = 1 - sum_i (y_i - yhat_i)^2 / sum_i (y_i - ybar_train(fold(i)))^2 over pooled out-of-fold predictions, and per-fold r (mean over outer folds, then median over voxels). All 30 checkpoints, 3 subjects, g in {1, 2, 4, 8}. Reproduction gate: the pooled median r must reproduce results/prereg/primary.csv to 1e-9, else stop.
F-R1: untrained minus Backprop in median MSE skill is <= 0 at every g with at most 1 of 5 seeds positive at each g (mean over subjects).
F-R2: the same for per-fold r.
F3 permutation null: 50 permutations (seeds 1 to 50) of the stimulus assignment of the feature rows (same permutation for all voxels), for the seed-0 untrained and Backprop checkpoints, g in {1, 4}, each subject: null mean and 2.5th/97.5th percentile of the pooled median r.
F-R3: at g = 1 and g = 4 the real pooled median r of both conditions exceeds the 97.5th percentile of its own null in sub-01 and sub-02. sub-03: descriptive.
F4 joint readout: features = network (g x g pooled, standardised per training split) concatenated with the luminance-and-contrast bank (g x g pooled, same standardisation); same CV; all 6 conditions x 5 seeds, g in {1, 4}, 3 subjects.
F-R4: for Backprop, mean over subjects and seeds, joint median r minus bank-alone median r is <= 0.010 at g = 1 and g = 4 (the network adds little beyond the bank); otherwise it adds. The same quantity for the untrained network is reported without a rule.

## B: normalisation statistics inside the bootstrap
B1: read the bootstrap code of steps 5 to 9 and report whether unit means, SDs, feature transformations F, voxel normalisations B, the mean profile and the norms are computed once on the 720 stimuli or inside each draw (file and line).
B2 (if they are fixed): recompute, with the statistics re-estimated from the resampled rows inside each of the first 200 draws (duplicates kept, self-pairs dropped), the V1 gap at NATIVE 224 (own CNN) in the cells (z, correlation, paper), (center, euclidean, paper) and (center, correlation, center720). Cells without normalisation statistics (raw) are unaffected by construction.
BN1: for all three cells the classification of the 95% CI relative to 0 (above, below, includes) is unchanged compared with the fixed-normalisation intervals of step 7.

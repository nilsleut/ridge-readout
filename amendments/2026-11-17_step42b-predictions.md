# Step 42b: predictions for subjects 2–8 (fixed after the subj01 pilot, before loading any data of subjects 2–8)
Date: 2026-11-17. Configuration: config_step42b.yaml, config hash 02d0ae4e72de. config_step42.yaml (da1cbb33b6d7) is unchanged.

Motivation: in the subj01 pilot (commit 49fc44b), the untrained network was ahead under Euclidean distance and behind under correlation distance at V1. Correlation distance discards pattern norms; Euclidean distance keeps them. This motivates a test of the norm–angle decomposition of arXiv:2611.15533 (Eq. 2; prediction SEP2 there was not met) on held-out subjects.
The predictions of step 42 (P1–P4, all 8 subjects) remain unchanged and are evaluated as committed.
Subjects for Q1–Q4: subj02–subj08 only (n = 7); subj01 is excluded because it motivated these predictions. Inference: one-sided sign-flip test over the 7 subject means (128 assignments) in the predicted direction; bootstrap interval of the 7-subject mean (1000 resamples, seed 20261002). Δ = Random − BP, mean over seeds 0–4, k = 3.
Norm and angle components: as defined in arXiv:2611.15533, Section 3.5 and Eq. 2, on units centred over the 515 images (model side) and voxels centred over the 515 images (brain side); N_ij = (n_i − n_j)^2, A_ij = 1 − cos θ_ij (unweighted angular term). Model N is compared with brain N, model A with brain A (Spearman over all pairs). Implementation: the existing functions from the step 6/7 scripts of this repository, unchanged.
Q1 (V1, raw units, Euclidean distance): Δ > 0 in at least 5 of 7 subjects and sign-flip p ≤ 0.05.
Q2 (V1, N component): Δ > 0 in at least 5 of 7 subjects and sign-flip p ≤ 0.05.
Q3 (V1, A component): Δ < 0 in at least 5 of 7 subjects and sign-flip p ≤ 0.05.
Q4 (LO-obj, A component): Δ < 0 in at least 5 of 7 subjects and sign-flip p ≤ 0.05.
N and A for subj01 and for the 'lateral' variant are reported without prediction. The weighted term W is reported without prediction.

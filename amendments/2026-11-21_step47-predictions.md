# Step 47: run structure in THINGS-720 and ridge replication in NSD. Predictions (fixed before scripts and runs)
Date: 2026-11-21. Configuration: config_step47.yaml, config hash aa9c06baf1a1.
Facts known before: THINGS-720 betas are z-scored per voxel over all 9840 trials, not per run; the 720 images of each subject lie in one session of 10 runs; sub-01 and sub-02 share the run assignment and order, sub-03 does not; the THINGS-720 V1 advantage (+0.030) comes mainly from sub-01 (+0.033) and sub-02 (+0.051), sub-03 +0.007; global L and C remove 46 % of it (step 44); the V1 RDM's luminance correlation reaches the LOSO bound only in THINGS-720.
Hypothesis: run-level response offsets, which are not removed by the normalisation and are shared by subjects with the same run assignment, carry luminance structure that the untrained network matches.

A. Run structure (THINGS-720, own CNN, paper cell unless stated; cross-run pairs P; Δ = Random − BP, mean over seeds 0–4, then subjects).
Run centring: after the existing normalisation, subtract from each voxel its mean over all trials of the run (82: 72 main + 10 test) of the same run.
RC1: with run centring, V1 Δ ≤ +0.010.
RC2: with run centring, ρ(V1, L-RDM) / LOSO lower bound (bound recomputed on run-centred data) < 0.5.
RC3: the across-run variance of the run-mean luminance of the 720 images is larger for the sub-01/02 assignment than for the sub-03 assignment.
RC4: with run centring, the THINGS-720 V1 multiverse (36 cells, 28 distinct analyses, rule of the paper) has at most 2 distinct analyses with the interval above 0.
CO1: without run centring, partialling the predecessor-luminance RDM |L_prev(i) − L_prev(j)| (pairs without a predecessor excluded) changes V1 Δ by at least 0.005 in absolute value.
Interpretation fixed in advance: RC1, RC2 and RC3 met → the THINGS-720 advantage is attributed to run-level structure shared by subjects with the same run assignment, combined with the untrained network's greater luminance sensitivity. RC1 not met → run structure is not the explanation; the explained fraction is reported. CO1 met → carry-over from the preceding image contributes.
Reported without prediction: per-subject values for all quantities; the LOSO lower bound at V1 with and without run centring; percentile of RC3 among 10,000 random assignments of the 720 images to 10 runs of 72 (seed 20261121), for both assignments; run-centred Δ after L + C; LOC; ridge of the step 40 type is not rerun.

B. Ridge replication in NSD (pre-specified primary analysis of arXiv:2611.15533 applied to NSD).
8 subjects, 515 images, k = 3 responses as targets; V1; reliable voxels: split-half reliability over the three presentations (mean of the three pairwise voxel-wise correlations across the 515 images, Spearman–Brown corrected) > 0.2; dual ridge, nested CV 8 × 5 (seed 20260818), α grid 1e-6…1e8 (29 points), min-MSE selection, pooled out-of-fold r, median over reliable voxels; lower-edge gate D1 with the plateau rule of the paper; own CNN Conv1 after max-pool, 224 px, g ∈ {1, 4, 8}; Random and BP, seeds 0–4.
RD1: at g = 4, Random − BP < 0 in at least 6 of 8 subjects (seed means) and one-sided sign-flip p ≤ 0.05.
RD2: the same holds at g = 1 and at g = 8.
Reported without prediction: MSE skill; the luminance-and-contrast bank (2g² features) alone; LO-obj; number of reliable voxels per subject.

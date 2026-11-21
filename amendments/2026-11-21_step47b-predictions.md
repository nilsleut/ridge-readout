# Step 47b: mechanism of the run structure and robustness to run centring. Predictions (fixed before scripts, data loading and runs)
Date: 2026-11-21. Configuration: config_step47b.yaml, config hash 97ffa8ef4507.
Facts known before: step 47 (run centring removes 77 % of the THINGS-720 V1 advantage; RC3 not met and reversed); step 44 (sub-02 V1 RDM correlates with presentation order, ρ = 0.139); sub-01 and sub-02 share run assignment and order.
Hypothesis: drift of the brain response across runs, combined with a chance alignment between this drift and the run-mean features of the untrained network in the run assignment shared by sub-01 and sub-02.

X. Mechanism (THINGS-720 session, V1, per subject).
Brain run RDM: correlation distance between the 10 run-mean voxel patterns (mean over all 82 trials of a run, after the existing z-score). Model run RDM: correlation distance between the 10 run-mean GAP Conv1 feature vectors (mean over the features of the 82 images of the run, raw units), mean over seeds 0–4 of the RDM entries. Alignment A(model) = Spearman between brain and model run RDM over the 45 run pairs. ΔA = A(Random) − A(BP). Run-index RDM: |r − s|.
Null: 10,000 random reassignments of the 820 images of the session to 10 runs of 82 (seed 20261122), model side only; percentile of the observed ΔA.
X1: ΔA > 0 in sub-01 and in sub-02.
X2: the observed ΔA of sub-01 and of sub-02 lies within the central 95 % of its null distribution.
X3: the brain run RDM correlates positively with the run-index RDM in at least 2 of the 3 subjects.
Interpretation fixed in advance: X1, X2 and X3 met → the run-level part of the THINGS-720 advantage is attributed to drift across runs together with a chance alignment of the untrained network's run-mean features in the run assignment shared by sub-01 and sub-02. X1 met and X2 not met (ΔA above the 97.5th percentile) → attributed to stimulus-driven run-level structure. Otherwise → not identified.
Reported without prediction: A(Random), A(BP) and percentiles per subject; Spearman between the brain run RDMs of sub-01 and sub-02 and between each of them and sub-03 (by run index); the run-level part of Δ (Δ raw − Δ run-centred) per subject.

Y. Robustness to run centring (run centring = subtract the voxel mean over all trials of the same run, after the existing normalisation; in NSD per run within session, before averaging the presentations).
Y1 (THINGS ridge, primary analysis): with run centring, g = 4, Random − BP < 0 in 5 of 5 seeds (mean over subjects).
Y2 (NSD RSA): with run centring, V1 paper cell and V1 standardised cell: Δ < 0 in at least 6 of 8 subjects each.
Y3 (NSD ridge): with run centring, g = 4, V1: Random − BP < 0 in at least 6 of 8 subjects.
Y4 (NSD ridge, all voxels): without run centring, g = 4, all V1 voxels (no reliability selection): Random − BP < 0 in at least 6 of 8 subjects.
Reported without prediction: THINGS test images with run centring (k = 1 and k = 12, paper cell and standardised cell); NSD multiverse class counts with run centring; THINGS ridge g = 1 and 8 with run centring; LO-obj.

# Step 48: one mechanism across both datasets. Predictions (fixed before scripts, data loading and runs)
Date: 2026-11-22. Configuration: config_step48.yaml, config hash 159c5c78e854.
Facts known before: step 47c (removing the session-mean voxel profile reduces the THINGS-720 V1 paper-cell Δ from +0.030 to +0.005; trial-wise cosine with that profile correlates with luminance, ρ −0.16 to −0.22); W3 of step 44 (the untrained network is more luminance-sensitive in all datasets); NSD per-session z-scoring sets voxel session means to 0.
Hypothesis: when a common voxel response profile is left in the patterns, correlation-distance RSA is driven by each trial's response amplitude along that profile; in V1 this amplitude tracks low-level image energy; the untrained network's mean-feature axis tracks the same statistics; hence an untrained advantage.

Conventions (paper cell: raw model units, correlation distance; Δ = Random − BP, mean over seeds 0–4, then subjects; V1 primary):
THINGS-720 (cross-run pairs): T-raw = h5 values without our z-score (unit: single-trial GLMsingle betas as stored in the THINGS-fMRI h5 files, not normalised by the dataset; per-voxel mean about +0.01 and SD about 0.07 over all trials and within sessions; absolute unit not documented locally); T-global = z per voxel over all 9840 trials (= paper); T-session = z per voxel within the session of the 720.
NSD (k = 1 and k = 3; step 42/43 pair rules): N-raw = betas in % signal change, no z-score; N-global = z per voxel over all trials of sessions 1–30; N-session = z per voxel within session (= step 42).

NB1: in NSD, Δ(N-raw) > Δ(N-session) at k = 1, in at least 6 of 8 subjects.
NB2: in NSD, Δ(N-raw) > 0 at k = 1 in at least 6 of 8 subjects, one-sided sign-flip p ≤ 0.05.
NB3: in THINGS-720, Δ(T-session) ≤ +0.010.
Response amplitude: per trial, g = cosine of the voxel pattern with the mean voxel pattern of the trials of that convention (brain gain); g^M = cosine of the model's raw GAP feature vector with the model's mean feature vector over the images (model gain).
G1: in THINGS-720 (T-global), |Spearman(g, CE)| > |Spearman(g, L)| in at least 2 of 3 subjects.
G2: in THINGS-720 (T-global), Spearman(g, g^M Random) > Spearman(g, g^M BP) in all 3 subjects (mean over seeds).
G3: in NSD (N-raw, k = 1), Spearman(g, g^M Random) > Spearman(g, g^M BP) in at least 6 of 8 subjects.
Interpretation fixed in advance:
  NB1, NB2, G2 and G3 met → the untrained advantage is attributed to a common voxel response profile left in by the normalisation, through per-image response amplitude that the untrained network tracks better; the attribution holds in both datasets.
  NB1 met, NB2 not met → the normalisation convention moves NSD in the predicted direction but does not create an untrained advantage there.
  NB1 not met → the mechanism is not supported in NSD.
Reported without prediction: all conventions for every cell of the paper (z/correlation, raw and z Euclidean), k = 3, LO-obj, THINGS test images under T-raw / T-global / T-session; Spearman of g with L, C, CE, DD, OFF, ON per dataset and subject; Spearman of g^M with the same statistics per model; the NSD multiverse class counts under N-raw (if runtime allows; otherwise not computed).

# Literals in Methods 2.1-2.6 (`paper/main.tex`)

Every number in 2.1-2.6 that is not a macro from `paper/numbers.tex` is a fixed
parameter or a design count. Each is listed with the source it was checked against
(file, line). Macros used: `\NReliableSubOne`, `\NReliableSubTwo`, `\NReliableSubThree`
(added to `scripts/export_macros.py` for this draft; source `results/prereg/primary.csv`,
column `n_reliable`).

## 2.1 Data

| Literal | Source |
|---|---|
| 3 subjects | `config.yaml:136` |
| V1; LOC = lLOC ∪ rLOC | `config.yaml:127`; `src/ridge/arm_b_betas.py:50-51` |
| z-score per voxel across all 9,840 trials, before stimulus selection | `config.yaml:124-129`; `src/ridge/arm_b_betas.py:19-23` |
| 720 stimuli, one exemplar of each of 720 distinct concepts | `config.yaml:82`; `reports/PHASE0_RECONNAISSANCE.md:107-108` |
| same stimulus order in all subjects | `reports/PHASE0_RECONNAISSANCE.md:108-109`; `reports/STEP4_SCALE_SUBJECTS.md:59` |
| single-presentation `train` trials | `reports/PHASE0_RECONNAISSANCE.md:51`; `PREREGISTRATION.md:108-109` |
| one session per subject | `reports/SESSIONS_CHECK.md` (sessions 3 / 2 / 5 for sub-01 / sub-02 / sub-03); `src/vendor/common_up_pairs.py:49` (assertion) |
| 10 runs of 72 stimuli in every subject | `reports/SESSIONS_CHECK.md` |
| sub-01 and sub-02 identical run order; same-run pairs share a run component | `evaluation-resolution-rsa` @ `arxiv-v3`, `paper/arxiv_upload_v3/Evaluation_Resolution_Confounds_paper_v3.tex:31`, `:67`; `learning-rules-rsa` @ `arxiv-v4`, `paper/arxiv_upload_learning_rules_v3/learning_rules_rsa_paper_v3.tex:211`; run assignment checked in `reports/SESSIONS_CHECK.md` |
| 100 `test` images, 12 repetitions, identical across subjects, disjoint from the 720 | `config.yaml:104-107`; `PREREGISTRATION.md:108-110` |
| halves of 6 repetitions, all 462 balanced splits, mean over splits, Spearman-Brown 2r/(1+r) | `config.yaml:117-120`; `src/ridge/arm_b_betas.py:27-33` |
| threshold r_SB > 0.2, fixed in advance, identical for all conditions | `config.yaml:156`; `PREREGISTRATION.md:133-135`; `reports/PREREG_PRIMARY.md:22` |
| selection cannot leak (other images) | `config.yaml:159-160`; `PREREGISTRATION.md:137-138` |
| ridge on reliable V1 voxels, RSA on all ROI voxels | `reports/STEP6_RIDGE_MECHANISM_ROBUSTNESS.md:8-9` |
| cross-run pairs: distinct stimuli in different runs in every subject | `src/vendor/common_up_pairs.py:9`, `:60-62` |
| 210,205 of 258,840 pairs | `src/vendor/common_up_pairs.py:10`; `src/vendor/v12_lift.py:51`; `amendments/2026-10-08_step5-resolution-predictions.md:16` |

## 2.2 Models

| Literal | Source |
|---|---|
| 30 checkpoints, 6 conditions × 5 seeds | `config.yaml:58-60` |
| seeds 42, 123, 456, 789, 1337 | `config.yaml:57` |
| trained on CIFAR at 32 px, the training resolution | `PREREGISTRATION.md:20`; `src/vendor/modal_src/learning_rules_v12_allseeds_modal.py:402-404` |
| condition names (Backprop, FA, PC, STDP, two untrained) | `config.yaml:50-56` |
| BN-calibrated: 50 forward-only CIFAR batches at 32 px, weights unchanged | `reports/STEP2_RANDOM_VARIANT.md:28-31`; `config.yaml:58-59` |
| "untrained" alone = uncalibrated variant, as in the published headline | `reports/STEP2_RANDOM_VARIANT.md:33-47` |
| Conv1 = whole first block, 2 × 2 max pooling, 32 channels, map side = input/2 | `config.yaml:42-48` |
| LOC mapped to Conv3 | `scripts/step7_multiverse.py:45`; `scripts/step5_resolution_rsa.py:19` |
| NATIVE p ∈ {32, 64, 96, 128, 160, 224} | `config.yaml:62`; `scripts/step7_multiverse.py:46` |
| UPSAMPLED: 32 px, then p; used at p = 224 | `src/vendor/modal_src/learning_rules_v11_upsample_modal.py:13-14`, `:110`, `:511-521`; `scripts/step7_multiverse.py:46` |
| CIFAR channel normalisation | `src/vendor/modal_src/learning_rules_v12_allseeds_modal.py:404`, `:528-530` |
| six torchvision models, IMAGENET1K_V1 | `amendments/2026-10-09_step8-generalization-predictions.md:13-14` |
| untrained: no weights, `torch.manual_seed(s)`, s ∈ {0, 1, 2}; BN defaults mean 0, variance 1; LayerNorm for convnext_tiny and vit_b_16 | same file `:17-18`, `:87-89` |
| eval mode | same file `:19` |
| RGB, shorter side to R, centre crop R, ImageNet normalisation; R ∈ {112, 224, 448}; training resolution 224 | same file `:36-37` |
| vit_b_16 at 224 only | same file `:77-79` |
| layer table | same file `:25-32` |
| GAP, or mean over tokens without the class token | same file `:23` |

## 2.3 RSA convention

| Literal | Source |
|---|---|
| correlation distance for model and fMRI RDMs | `src/vendor/modal_src/learning_rules_v12_allseeds_modal.py:432-434`; `scripts/step5_resolution_rsa.py:14`; `amendments/2026-10-09_step7-offset-predictions.md:21` (B = paper) |
| Spearman over cross-run pairs, per subject, then mean | `scripts/step5_resolution_rsa.py:6-9`; `amendments/2026-10-08_step5-resolution-predictions.md:14-16` |
| contrast = mean over seeds of per-seed differences | `reports/STEP5_RESOLUTION_MOMENTS.md:18-19` |
| ImageNet contrast: untrained (mean over 3 init seeds) − trained | `amendments/2026-10-09_step8-generalization-predictions.md:38-39`; `amendments/2026-10-10_step9-separation-predictions.md:32` |
| 1000 resamples, seed 20261002, with replacement | `src/vendor/v12_lift.py:53`, `:79`; `amendments/2026-10-08_step5-resolution-predictions.md:17-18` |
| pair set recomputed per draw, self-pairs dropped | `scripts/step5_resolution_rsa.py:10-11`; `src/vendor/common_up_pairs.py:60` |
| paired differences within draw | `amendments/2026-10-08_step5-resolution-predictions.md:17-18`; `reports/STEP5_RESOLUTION_MOMENTS.md:20-21` |
| 95% percentile intervals | `amendments/2026-10-08_step5-resolution-predictions.md:19` |
| inference over stimuli in 3 subjects | `amendments/2026-10-08_step5-resolution-predictions.md:56-58` |

## 2.4 Analysis multiverse

| Literal | Source |
|---|---|
| F ∈ {raw, center, scale, z}, per unit over the 720 | `amendments/2026-10-09_step7-offset-predictions.md:18`; definitions `scripts/step4_std_rsa.py:51-54` |
| SD < 1e-8 → 0 | `scripts/step4_std_rsa.py:39`, `:52-55`; `scripts/step6_common.py:28` |
| D ∈ {correlation, cosine, euclidean}, on both sides | `amendments/2026-10-09_step7-offset-predictions.md:19`, `:80-82` |
| B ∈ {paper, center720, z720} | same file `:20-22`; `scripts/step7_multiverse.py:60-69` |
| paper cell = (raw, correlation, paper) | same file `:57-58` |
| 4 × 3 × 3 = 36 cells | same file `:42` |
| V1 Conv1 untrained − Backprop; LOC Conv3 Backprop − untrained | same file `:23`, `:50`; `scripts/step7_multiverse.py:45` |
| S = Gap(NATIVE 224) − Gap(NATIVE 32) | same file `:23` |
| euclidean raw vs center check (O1) | same file `:34-39` |
| ImageNet: four bootstrap cells, all 36 as point estimates | `amendments/2026-10-09_step8-generalization-predictions.md:43-45` |
| S_gen = Gap(448) − Gap(224) | same file `:38` |

## 2.5 Ridge readout

| Literal | Source |
|---|---|
| dual ridge, K = XXᵀ eigendecomposed once per split | `config.yaml:209-210` |
| adaptive average pooling, g ∈ {1, 2, 4, 8}, 32g² features | `config.yaml:87`, `:323-324`; `src/ridge/gridpool.py:18-22` |
| g = 1 is GAP | `config.yaml:331-337` |
| standardisation from training rows of each split only | `config.yaml:212-218`; `src/ridge/fit.py:16-17` |
| 8 outer, 5 inner folds, seed 20260818 | `config.yaml:286-288` |
| one exemplar per concept, folds concept-grouped | `config.yaml:290-294` |
| 29 log-spaced α, 1e-6 … 1e8 | `config.yaml:227-230`; `PREREGISTRATION.md:360-361` (D3) |
| min inner-fold MSE, refit on outer training set | `config.yaml:236`; `amendments/2026-10-03_tolerance-rule.md:5`; `src/ridge/fit.py:9-13` |
| selection vs reporting statement | `PREREGISTRATION.md:347-353` (D2) |
| pooled out-of-fold predictions, one Pearson r per voxel | `src/ridge/fit.py:146-150` |
| median over reliable voxels, mean over subjects, contrast within seed | `config.yaml:199-201`; `PREREGISTRATION.md:35-38` |
| low-edge gate: 1e-7, 1e-8, 1e-9; relative gain < 1e-3; score-neutral | `amendments/2026-10-03_d1-plateau-gate.md:5-19` |
| preregistered run: 720 single-trial betas, NATIVE 224 px, 30 checkpoints, 3 subjects, all g | `reports/PREREG_PRIMARY.md:20-24` |
| g = 2 on CPU, downstream equivalence gate | `amendments/2026-10-13_g2-local-extraction.md:8-14`, `:27-36` |
| exploratory g = 1 fit at all NATIVE resolutions and UPSAMPLED 224 | `amendments/2026-10-09_step6-predictions.md:10-12`; `reports/STEP6_RIDGE_MECHANISM_ROBUSTNESS.md:14-17` |
| pixels; Rec. 709 luminance with local RMS contrast, 9 × 9 window | `scripts/pilot_features_modal.py:11-12`, `:187-190` |
| Gabor bank, absolute responses, 4 orientations × 3 spatial frequencies | `config.yaml:365-366`; `scripts/pilot_features_modal.py:13`, `:171-172`, `:191` |
| negative test: Backprop seed-0 features, permuted assignment | `config.yaml:367-371`; `scripts/pilot_fit.py:152-155` |
| cell shuffle per image, seed 20260820 | `PREREGISTRATION.md:161-166`; `scripts/pilot_fit.py:154`, `:161` |
| cell shuffle run: seed-0 untrained and Backprop, g ∈ {2, 4, 8} | `scripts/pilot_fit.py:158-162`; `reports/PREREG_PRIMARY.md:200-201` |

## 2.6 Gain axis and decomposition

| Literal | Source |
|---|---|
| gain_i, μ_c, x̃_i, gain RDM | `amendments/2026-10-09_step7-offset-predictions.md:24-30` |
| model gain on raw features; voxel gain on all ROI voxels under B = paper | same file `:30`, `:87-88` |
| c_gain, Pearson over the 720, mean over subjects | same file `:55-56` |
| partial RSA | same file `:60-63`, `:83-86` |
| X_c, n_i, cos θ_ij, identity | `amendments/2026-10-10_step9-separation-predictions.md:11-18` |
| norm RDM, angle RDM | same file `:20-22` |
| F = center primary, F = z secondary (SD 1); fMRI side the same | same file `:24-27`, `:76-80` |
| identity checked numerically (SEP1) | same file `:37-38` |
| "gain"/"geometry" are definitions, not biological claims | same file `:69-70`; `reports/STEP9_GAIN_GEOMETRY.md:11-12` |

## Points where the sources are unclear

1. **Rationale for the cross-run restriction.** Resolved: see "Cross-run rationale
   (verbatim)" below; 2.1 now states it with `\cite{evalres}`.
2. **Two luminance definitions.** The control band uses Rec. 709 weights
   (`scripts/pilot_features_modal.py:187`); the descriptive stimulus statistics of steps
   7-9 use Rec. 601 weights (`amendments/2026-10-09_step7-offset-predictions.md:90`).
   Both are as implemented; only the control band appears in 2.1-2.6.
3. **Cell-shuffle grids.** Preregistered for g = 4 and 8 at all resolutions
   (`PREREGISTRATION.md:166`, `config.yaml:388-389`); run at g ∈ {2, 4, 8}, 224 px only,
   seed 0 (`scripts/pilot_fit.py:158`). The draft describes what was run.
4. **"One session".** Resolved: `reports/SESSIONS_CHECK.md` (`scripts/check_sessions.py`)
   shows exactly one session per subject (IDs 3 / 2 / 5), 10 runs of 72 stimuli each.

## Cross-run rationale (verbatim)

Read with `git show <tag>:<path>`; line numbers refer to the file at the tag.

`evaluation-resolution-rsa`, tag `arxiv-v3`,
`paper/arxiv_upload_v3/Evaluation_Resolution_Confounds_paper_v3.tex`, line 67 (Methods,
"RSA"):

> using only the $210{,}205$ of $258{,}840$ stimulus pairs whose two stimuli lie in
> different fMRI runs for all three subjects (two subjects saw the stimuli in identical
> run order, so same-run pairs share a run component).

Same file, line 31 (note on version 3):

> on the $210{,}205$ of $258{,}840$ stimulus pairs whose two stimuli lie in different fMRI
> runs for all three subjects. Two subjects saw the stimuli in identical run order, and
> same-run pairs carry a run component they share (see the revised endpoint study,
> \citealt{leutenegger2026}).

`learning-rules-rsa`, tag `arxiv-v4`,
`paper/arxiv_upload_learning_rules_v3/learning_rules_rsa_paper_v3.tex`, line 211
(Methods, "Stimulus pairs (cross-run)"):

> Each subject saw the 720 stimuli in ten runs, and sub-01 and sub-02 saw them in
> identical order. Responses within a run can share run-specific components (baseline
> drifts, run-wise noise), so for two subjects with the same order the dissimilarities of
> same-run pairs share a component that has nothing to do with the stimuli. This inflated
> the agreement between sub-01 and sub-02 (at LOC $\NV{agree.0102.LOC.all}$ on all pairs
> vs.\ $\NV{agree.0102.LOC.cr}$ on cross-run pairs, while sub-01 vs.\ sub-03 is
> $\NV{agree.0103.LOC.all}$ and $\NV{agree.0103.LOC.cr}$) and thereby the lower bound at
> LOC and IT (appendix, Table~\ref{tab:sens}). We therefore use only stimulus pairs that
> lie in different runs for all three subjects: $\NV{pairs.cr}$ of $\NV{pairs.total}$
> pairs; $\NV{pairs.excl.pct}\,\%$ are excluded.

The identical run order of sub-01 and sub-02 (and the different order of sub-03) is
confirmed from the stimulus metadata in `reports/SESSIONS_CHECK.md`.

## Word counts (current draft)

Counted on the LaTeX source without comments; each inline formula, equation and macro
counts as one word, citations as none.

| Subsection | Words |
|---|---|
| 2.1 Data | 230 |
| 2.2 Models | 281 |
| 2.3 RSA convention | 147 |
| 2.4 Analysis multiverse | 155 |
| 2.5 Ridge readout | 378 |
| 2.6 Gain axis and decomposition | 227 |
| Total | 1418 |

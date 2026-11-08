# Correction notes for earlier preprints

Status: final text, not posted. "[arXiv ID of the new paper]" stands for the arXiv ID of
*Analysis choices decide whether untrained networks beat trained ones at V1: a multiverse
analysis of model–brain RSA*, to be filled in once it is announced. The affected claims
are those listed in `reports/AUDIT_RAW_RSA_CLAIMS.md`. Every number below comes from the
new paper or its `results/` files; the source of each number is given in the source
notes under each revision note and in the last column of each table.

Form: a notice on page 1 of the replaced PDF (1a), a short prefix for the arXiv abstract
field (1b) and a line in the arXiv Comments field (1c). Titles and main texts stay
unchanged.

Status labels in the tables:
- **not robust to the analysis choices**: the claim does not hold under at least one of
  the analysis choices examined in the new paper (unit standardisation over the evaluated
  stimuli, centring of units and voxels, other distance measures, a fitted ridge
  readout). "(point estimate)" marks claims for which only a point estimate exists.
- **holds under centring**: the claim also holds when model units are standardised over
  the evaluated stimuli (this includes centring; `results/step5/`) and, where it was
  computed, when units and voxels are centred over the evaluated stimuli
  (`results/step7/multiverse.csv`, F = center).
- **not tested here**: the new paper contains no analysis of this claim.

Throughout, "standardised" means channel z-scoring of the model features over the 720
evaluated stimuli, as in Section 3.2 of the new paper.

---

## 2608.12408 (v3 → v4)

*Evaluation Resolution Confounds Learning-Rule Comparisons in Model–Brain RSA of Early
Visual Cortex*

### (1a) Notice for page 1 of the replaced PDF

> Revision note (v4). The resolution dependence of the V1 difference between the untrained network and backpropagation reported here is not robust to the analysis choices examined in "Analysis choices decide whether untrained networks beat trained ones at V1: a multiverse analysis of model–brain RSA" [arXiv ID of the new paper]. With standardised model units, the slope from 32 to 224 px is +0.002 [−0.004, 0.007] instead of +0.031 [0.024, 0.038], and the V1 difference at 224 px is −0.011 [−0.017, −0.005]. Under Euclidean distance the V1 advantage persists without a resolution dependence; in ImageNet models a smaller resolution dependence remained. The backpropagation advantage at LOC holds with standardised units. The text below is unchanged.

### (1b) Prefix for the arXiv abstract field

> Not used: the abstract field has 1871 characters (measured 2026-11-07 on the arXiv listing text) and the full prefix would exceed the limit of 1920. Option, only if the length check at submission time allows it: 'Revision note (v4): see Comments. ' (34 characters). Otherwise the abstract field stays unchanged; the notice is on page 1 of the PDF and in the Comments field.

### (1c) Comments field

> v4: revision note on the V1 resolution result, see arXiv:[ID]; main text unchanged.

Source notes:
- +0.031 [0.024, 0.038]: `\SlopeRaw`, `\SlopeRawLo`, `\SlopeRawHi` (`results/step5/paper_comparisons_raw_vs_z.csv`).
- +0.002 [−0.004, 0.007]: `\SlopeZ`, `\SlopeZLo`, `\SlopeZHi` (same file).
- −0.011 [−0.017, −0.005]: `\VoneGapZTwoTwoFour`, `Lo`, `Hi` (same file).
- Euclidean persistence: in all 12 euclidean cells the V1 difference at NATIVE 224 is positive with a CI above 0 (+0.016 to +0.023), and every slope CI includes 0 (largest |S| 0.005); `results/step7/multiverse.csv`; example cell `\MultiEuclidGap`, `Lo`, `Hi`.
- ImageNet: with standardised units, the slope from 224 to 448 px has a CI above 0 in 4 of 5 models (+0.005 to +0.010; convnext_tiny +0.001 [−0.001, 0.003]); `results/step8/generalization.csv` (kind boot, S_gen, F = z); in the angular component `\SepSlopeAngleCiPositive`.
- LOC with standardised units +0.007 [0.002, 0.011]: `\LocGapZ`, `Lo`, `Hi`.
- Not in the note, for reference: with units and voxels centred, the slope is +0.006 [0.003, 0.010] and the V1 difference at 224 px +0.001 [−0.005, 0.007]; LOC +0.007 [0.002, 0.012] (`results/step7/multiverse.csv`, F = center, D = correlation, B = center720). Across the 28 distinct analyses of the multiverse, the V1 interval at 224 px lay above 0 in 5, below 0 in 9 and included 0 in 14 (`\MultiDistinct`, `\MultiDistinctPos`, `\MultiDistinctNeg`, `\MultiDistinctIncl`).

### (3) Affected claims

| # | Line | Claim | Status | Source |
|---|---|---|---|---|
| 1 | 13–14 | Title: evaluation resolution confounds learning-rule comparisons at V1 | not robust to the analysis choices | `\SlopeRaw`, `\SlopeZ` |
| 2 | 26, 47, 94 | Untrained exceeds backpropagation at V1 at 224 px | not robust to the analysis choices | `\VoneGapRawTwoTwoFour`, `\VoneGapZTwoTwoFour` |
| 3 | 26, 94 | V1 gap is about zero at 32 px | not robust to the analysis choices | `\VoneGapRawThreeTwo`, `\VoneGapZThreeTwo` |
| 4 | 26, 94 | V1 gap grows monotonically across six resolutions | not robust to the analysis choices (point estimate) (CSV only, not in the new paper) | `results/step5/resolution_rsa.csv` |
| 5 | 26, 96 | Best-layer gap at 32 px and its growth | not tested here | — |
| 6 | 26, 49 | Holds across conditions, macaque, training, ResNet-50, Swin-Tiny | not tested here | — |
| 7 | 26, 102–111 | Calibrated untrained network still exceeds backpropagation | not tested here | — |
| 8 | 26, 158 | V1 gap at 224 px with content limited to 32 px | not robust to the analysis choices | `\VoneGapRawUp`, `\VoneGapZUp` |
| 9 | 26, 158, 196, 216 | Ratio R of content-limited to native gap is 1.14 | not robust to the analysis choices (CSV only, not in the new paper) | `results/step5/paper_comparisons_raw_vs_z.csv` ("R V1") |
| 10 | 26, 158, 160 | Limiting content widens the V1 gap | direction holds under standardisation; no untrained advantage remains (V1 difference −0.011 native, +0.002 content-limited) | `results/step5/paper_comparisons_raw_vs_z.csv` (variant z, "diff V1"); `\VoneGapZTwoTwoFour`, `\VoneGapZUp` |
| 11 | 26, 162 | LOC backpropagation advantage disappears under upsampling | holds under centring (CSV only, not in the new paper) | `results/step5/paper_comparisons_raw_vs_z.csv` (variant z, "diff LOC") |
| 12 | 26, 139 | Luminance scalar matches the untrained network at V1 | not tested here | — |
| 13 | 26, 51, 179 | LOC: backpropagation above untrained at every resolution | holds at 224 px under standardisation and centring; other resolutions not in the new paper | `\LocGapZ`; `results/step7/multiverse.csv` (F = center, LOC) |
| 14 | 85 | Every trained condition peaks at 32 px and falls off | not tested here | — |
| 15 | 85 | Untrained network's V1 alignment climbs with resolution | not robust to the analysis choices (point estimate) (CSV only, not in the new paper) | `results/step5/resolution_rsa.csv` |
| 16 | 96 | Best V1 layer: Conv2 for untrained, Conv1 for backpropagation | not tested here | — |
| 17 | 117 | ResNet-50 and Swin-Tiny V1 alignment falls toward 224 px | not tested here | — |
| 18 | 128–137 | Gabor and pixel similarity do not track V1 alignment | not tested here | — |
| 19 | 139–143 | Partialling luminance halves untrained alignment | not tested here | — |
| 20 | 160 | With content limited, backpropagation's V1 alignment falls | holds under centring (point estimate) (CSV only, not in the new paper) | `results/step5/resolution_rsa.csv` |
| 21 | 168 | Training dynamics of V1 alignment | not tested here | — |
| 22 | 179 | IT: backpropagation above untrained | not tested here | — |
| 23 | 190 | Rivalling backpropagation at V1 is largely a function of resolution | not robust to the analysis choices | `\SlopeZ`, `\VoneGapZTwoTwoFour` |
| 24 | 194, 196 | Gap appears in full once input exceeds training resolution | not robust to the analysis choices | `\VoneGapZUp`; `results/step5/paper_comparisons_raw_vs_z.csv` ("R V1") |

---

## 2604.16875 (v4 → v5)

*Untrained CNNs Exceed Backpropagation in V1 Alignment at High Evaluation Resolution*

### (1a) Notice for page 1 of the replaced PDF

> Revision note (v5). The central result of this paper, that the untrained network exceeds backpropagation in V1 alignment at 224 px, is not robust to the analysis choices examined in "Analysis choices decide whether untrained networks beat trained ones at V1: a multiverse analysis of model–brain RSA" [arXiv ID of the new paper]. In the model set re-tested there, the V1 difference is −0.011 [−0.017, −0.005] with standardised model units, against +0.030 [0.018, 0.042] without, and a pre-specified ridge readout places backpropagation above the untrained network; under Euclidean distance the untrained advantage persists. This affects the title, the comparison of the untrained network with backpropagation at V1 and the conclusion on architecture. The backpropagation advantage at LOC holds with standardised units. The other analyses of this paper were not re-tested. The text below is unchanged.

### (1b) Prefix for the arXiv abstract field

> Revision note (v5): the V1 result reported here is not robust to the analysis choices examined in arXiv:[ID]; see page 1. Text below unchanged.

Fits: the abstract field has 1569 characters (measured 2026-11-07 on the arXiv listing text); with this prefix about 1713 of 1920. The existing sentence 'Revised version (v4): ...' stays after the new prefix.

### (1c) Comments field

> v5: revision note on the V1 result, see arXiv:[ID]; main text unchanged.

Source notes:
- +0.030 [0.018, 0.042]: `\VoneGapRawTwoTwoFour`, `Lo`, `Hi` (`results/step5/paper_comparisons_raw_vs_z.csv`).
- −0.011 [−0.017, −0.005]: `\VoneGapZTwoTwoFour`, `Lo`, `Hi` (same file).
- "In the model set re-tested there": the new paper re-tests the 30-checkpoint retrain of the same configuration (`results/step5/paper_comparisons_raw_vs_z.csv`); the abstract of 2604.16875 v4 states that its +0.030 comes from the five-seed model set of the resolution study.
- Ridge: untrained minus backpropagation median r −0.043 at g = 4, 0 of 5 seeds positive (`\PreregRandBpGFour`, `\PreregRandBpGFourPos`; `results/prereg/signflip.csv`).
- Euclidean persistence: all 12 euclidean cells positive with a CI above 0 at NATIVE 224 (`results/step7/multiverse.csv`; `\MultiEuclidGap`, `Lo`, `Hi`).
- LOC with standardised units +0.007 [0.002, 0.011]: `\LocGapZ`, `Lo`, `Hi`.

### (3) Affected claims

| # | Line | Claim | Status | Source |
|---|---|---|---|---|
| 1 | 17 | Title: untrained CNNs exceed backpropagation in V1 alignment at high resolution | not robust to the analysis choices | `\VoneGapZTwoTwoFour` |
| 2 | 55 | At V1 the untrained baseline exceeds backpropagation | not robust to the analysis choices | `\VoneGapRawTwoTwoFour`, `\VoneGapZTwoTwoFour`, `\PreregRandBpGFour` |
| 3 | 55, 252 | The same holds at V2 | not tested here (V2: point estimates in the repository only) | `results/step5/resolution_rsa.csv` |
| 4 | 55, 252, 346, 364 | At 32 px the gap vanishes | not robust to the analysis choices | `\VoneGapZThreeTwo` |
| 5 | 55, 254 | At LOC backpropagation exceeds the random baseline | holds under centring | `\LocGapZ`; `results/step7/multiverse.csv` (F = center, LOC) |
| 6 | 254 | PC and STDP exceed the random baseline at LOC | not tested here | — |
| 7 | 55, 256 | At IT no two conditions can be distinguished | not tested here | — |
| 8 | 55, 258, 352 | FA produces the lowest V1 alignment | not tested here | — |
| 9 | 55, 248, 364 | Only the untrained network reaches the lower noise bound at V1 | not robust to the analysis choices (point estimate) (CSV only, not in the new paper) | `results/step5/resolution_rsa.csv` |
| 10 | 55, 364 | Early visual alignment driven more by architecture than learning rule | not robust to the analysis choices | `\VoneGapZTwoTwoFour` |
| 11 | 106 | Contribution: untrained exceeds backpropagation at V1 and V2 | not tested here (V2: point estimates in the repository only) | `\VoneGapZTwoTwoFour` |
| 12 | 252 | V1 order Random > STDP > BP > PC > FA | not robust to the analysis choices (point estimate) (CSV only, not in the new paper) | `results/step5/resolution_rsa.csv` |
| 13 | 252 | All ten pairwise V1 differences are significant | not tested here | — |
| 14 | 252, 346 | At 32 px backpropagation and untrained do not differ | not robust to the analysis choices | `\VoneGapZThreeTwo` |
| 15 | 278 | Untrained highest at V1 and V2, lowest at LOC and IT | not tested here (V2: point estimates in the repository only) | `results/step5/resolution_rsa.csv` |
| 16 | 293 | Per subject, untrained exceeds backpropagation at V1 | not robust to the analysis choices (point estimate) (CSV only, not in the new paper) | `results/step5/resolution_rsa.csv` |
| 17 | 293 | LOC difference significant only in sub-02 | not tested here | — |
| 18 | 304 | Seed SD small relative to the V1 difference | not tested here | — |
| 19 | 308 | Best layer: FC1 highest for untrained at V1 | not tested here | — |
| 20 | 321 | Partial RSA: untrained still exceeds backpropagation at V1 and V2 | not tested here | — |
| 21 | 348 | Calibrated baseline still exceeds backpropagation at V1 | not tested here | — |
| 22 | 354 | Accuracy and V1 alignment dissociated | not robust to the analysis choices | `\VoneGapZTwoTwoFour` |
| 23 | 356 | Architecture confound essentially complete at V1 | not robust to the analysis choices | `\VoneGapZTwoTwoFour` |
| 24 | 364 | Conclusion: V1 gap at 224 px, vanishing at 32 px | not robust to the analysis choices | `\VoneGapZTwoTwoFour`, `\VoneGapZThreeTwo` |
| 25 | 401–408 | Primary family of comparisons significant after correction | not tested here | — |

---

## 2605.22401 (v2 → v3)

*Cross-Species RSA Reveals Conserved Early Visual Alignment but Divergent Higher-Area
Rankings Across Human fMRI and Macaque Electrophysiology*

### (1a) Notice for page 1 of the replaced PDF

> Revision note (v3). The human V1 results of this paper, the untrained network above backpropagation and the human V1 ranking of learning rules, are not robust to the analysis choices examined in "Analysis choices decide whether untrained networks beat trained ones at V1: a multiverse analysis of model–brain RSA" [arXiv ID of the new paper]. In the model set re-tested there, the untrained-minus-backpropagation difference at human V1 is −0.011 [−0.017, −0.005] with standardised model units. The macaque comparisons and the cross-species conclusions were not re-tested. The text below is unchanged.

### (1b) Prefix for the arXiv abstract field

> Not used: the abstract field (v2) has 1857 characters (measured 2026-11-07 on the listing text, including the August correction note), and the full prefix would exceed the limit of 1920. Option, only if the length check at submission time allows it: 'Revision note (v3): see Comments. ' (34 characters, about 1891 in total). Otherwise the abstract field stays unchanged.

### (1c) Comments field

> v3: revision note on the human V1 results, see arXiv:[ID]; main text unchanged.

Source notes:
- −0.011 [−0.017, −0.005]: `\VoneGapZTwoTwoFour`, `Lo`, `Hi` (`results/step5/paper_comparisons_raw_vs_z.csv`).
- "In the model set re-tested there": the new paper re-tests the 30-checkpoint retrain of the same configuration (`results/step5/paper_comparisons_raw_vs_z.csv`); the abstract of 2604.16875 v4 states that its +0.030 comes from the five-seed model set of the resolution study.

### (3) Affected claims

Only the claims the audit lists as affected; the macaque data were not re-tested.

| # | Line | Claim | Status | Source |
|---|---|---|---|---|
| 3 | 42, 193, 254, 292 | Local rules outperform backpropagation at V1/V2 (human side) | not robust to the analysis choices (point estimate; already withdrawn by the August note) (CSV only, not in the new paper) | `results/step5/resolution_rsa.csv` |
| 6 | 99 | Untrained exceeds backpropagation at human V1 | not robust to the analysis choices | `\VoneGapZTwoTwoFour` |
| 7 | 194 | Human V1 order Random > STDP > PC > BP > FA | not robust to the analysis choices (point estimate) (CSV only, not in the new paper) | `results/step5/resolution_rsa.csv` |

---

## 2605.30556 (v2 → v3)

*Supervised Training Rapidly Degrades Early Visual Cortex Alignment Across Biologically
Plausible Learning Rules*

### (1a) Notice for page 1 of the replaced PDF

> Revision note (v3). The degradation of V1 alignment with training reported here is not robust to the analysis choices examined in "Analysis choices decide whether untrained networks beat trained ones at V1: a multiverse analysis of model–brain RSA" [arXiv ID of the new paper]. In the model set re-tested there, with standardised model units, a fixed Conv1 layer and 224 px input, backpropagation after training scores above the untrained network at V1 (+0.011 [0.005, 0.017]). This covers the endpoint only; the trajectory and the best-layer analyses were not re-tested. The statement of the August note that the degradation of V1 alignment survives for every rule tested is likewise not robust to these choices at the endpoint. The text below is unchanged.

### (1b) Prefix for the arXiv abstract field

> Revision note (v3): the V1 degradation result is not robust to the analysis choices examined in arXiv:[ID] (endpoint only). Text below unchanged.

### (1c) Comments field

> v3: revision note on the V1 degradation result, see arXiv:[ID]; main text unchanged.

Source notes:
- +0.011 [0.005, 0.017]: `\VoneGapZTwoTwoFour`, `Lo`, `Hi` with the sign reversed (backpropagation minus untrained; `results/step5/paper_comparisons_raw_vs_z.csv`).
- "In the model set re-tested there": the new paper re-tests the 30-checkpoint retrain of the same configuration (`results/step5/paper_comparisons_raw_vs_z.csv`); the abstract of 2604.16875 v4 states that its +0.030 comes from the five-seed model set of the resolution study.
- The August note of v2 states that the degradation survives for every rule tested; the new paper's endpoint result (backpropagation above the untrained network under standardisation, Conv1, 224 px) is the basis of the added sentence.

### (3) Affected claims

Only the claims the audit lists as affected, at the training endpoint with a fixed layer;
the trajectory and best-layer analyses were not re-tested.

| # | Line | Claim | Status | Source |
|---|---|---|---|---|
| 1 | 19–20 | Title: supervised training degrades early visual cortex alignment | not robust to the analysis choices (endpoint, fixed layer) | `\VoneGapZTwoTwoFour` (sign reversed) |
| 2 | 44–47 | Training degrades V1 alignment for every rule tested | not robust to the analysis choices (point estimate) (CSV only, not in the new paper) | `results/step5/resolution_rsa.csv` |
| 5 | 52, 189 | LOC: backpropagation gains alignment with training | holds under centring (endpoint, fixed layer) | `\LocGapZ`; `results/step7/multiverse.csv` (F = center, LOC) |
| 6 | 172 | All trained conditions below untrained at V1 at epoch 40 | not robust to the analysis choices (point estimate) (CSV only, not in the new paper) | `results/step5/resolution_rsa.csv` |
| 7 | 176 | Degradation pattern extends to V2 | not robust to the analysis choices (point estimate) (CSV only, not in the new paper) | `results/step5/resolution_rsa.csv` |

---

## Checklist

### (a) Sources to fetch from arXiv (by the author; not downloaded here)

- [ ] 2608.12408v3 (source tarball)
- [ ] 2604.16875v4 (source tarball)
- [ ] 2605.22401v2 (source tarball): needed to establish the posted version; the
      repository has no tag and the TeX is untracked.
- [ ] 2605.30556v2 (source tarball): needed to establish the posted version; the folder
      is not a git repository.
- [ ] After downloading each source: measure the length of the arXiv abstract field (plain
      text, with markup as in the metadata field); the limit is 1920 characters. If (1b)
      does not fit, the abstract field stays unchanged, and the notice appears only in the
      Comments field and on page 1 of the PDF. Remeasure in the arXiv form at submission time; the figures above come from the listing text.
- [ ] Categories: 2605.22401 listing: cs.LG primary, q-bio.NC and cs.NE cross-lists; 2604.16875 and 2605.30556: cs.LG, q-bio.NC; 2608.12408: q-bio.NC primary, cs.LG cross-list. Keep categories unchanged on replacement.

### (b) Files to change per preprint

| Preprint | Repository | File | Title line | PDF notice (page 1) | Abstract | Abstract length check | Version | Comments |
|---|---|---|---|---|---|---|---|---|
| 2608.12408 v4 | `Projekte_2/evaluation-resolution-rsa` (tag `arxiv-v3`) | `paper/arxiv_upload_v3/Evaluation_Resolution_Confounds_paper_v3.tex` (copy to a v4 folder) | unchanged | (1a) at the top of page 1 | (1b) at the start of the abstract field, if the length check passes | measured 1871; (1b) does not fit, see (1b) | v3 → v4 where the TeX states a version | (1c); `paper/arxiv_upload_v3/ARXIV_COMMENT.txt` (v4 copy) and arXiv form |
| 2604.16875 v5 | `Projekte_1/learning-rules-rsa` (tag `arxiv-v4`) | `paper/arxiv_upload_learning_rules_v3/learning_rules_rsa_paper_v3.tex` (copy to a v5 folder) | unchanged | (1a) at the top of page 1 | (1b) at the start of the abstract field, if the length check passes | measured 1569; (1b) fits | v4 → v5 where the TeX states a version | (1c); arXiv form |
| 2605.22401 v3 | `Projekte_2/Cross_Species_RSA` (commit and tag the posted v2 first) | `paper/Cross_species_rsa_paper_final_version/Cross_species_rsa_paper_final.tex` | unchanged | (1a) at the top of page 1, before the August note | (1b) at the start of the abstract field, if the length check passes | measured 1857; (1b) does not fit, see (1b) | v2 → v3 where the TeX states a version | (1c); arXiv form |
| 2605.30556 v3 | `Projekte_2/training_dynamics` (initialise git, commit and tag the posted v2 first) | `training_dynamics_arxiv/training_dynamics.tex` | unchanged | (1a) at the top of page 1, before the August note | (1b) at the start of the abstract field, if the length check passes | to be measured. The abstract field of v2 contains the August correction paragraph; measure the field text on arxiv.org/abs/2605.30556 (the PDF text of the abstract is about 2012 characters and is therefore not the field text). | v2 → v3 where the TeX states a version | (1c); arXiv form |

### (c) Order after the announcement of the new paper

1. Day 0: the new paper is announced. Insert its arXiv ID for "[arXiv ID of the new
   paper]" in all four notes; re-check every number against the source notes.
2. Day 0: submit the replacement of 2608.12408 (v4), the paper the new one re-tests directly.
3. Day 1: submit the replacement of 2604.16875 (v5).
4. Day 1: submit the replacements of 2605.22401 (v3) and 2605.30556 (v3), after their
   posted versions have been identified and tagged.
5. By day 2 at the latest: all four replacements announced; tag each repository at the
   new version (existing tags are never moved).
6. Titles stay unchanged (citability); this is a deliberate decision.

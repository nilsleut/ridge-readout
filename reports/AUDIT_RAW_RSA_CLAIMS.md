# Audit: published claims that rest on RSA with non-standardized model features

Read-only audit. No numbers were recomputed for this report; the z-variant values are
looked up in `results/step5/` (commit `77e801a`). Exploratory, like everything in this
repository.

## Sources

| arXiv | Repository | Version read | TeX file |
|---|---|---|---|
| 2604.16875 | `Projekte_1/learning-rules-rsa` | tag `arxiv-v4` (`b8d677b`), via `git show` | `paper/arxiv_upload_learning_rules_v3/learning_rules_rsa_paper_v3.tex`, values from `numbers.tex` and `tab_*.tex` of the same tag |
| 2608.12408 | `Projekte_2/evaluation-resolution-rsa` | tag `arxiv-v3` (`7a9b374`), via `git show` | `paper/arxiv_upload_v3/Evaluation_Resolution_Confounds_paper_v3.tex`, values from `numbers_v3.tex` |
| 2605.22401 | `Projekte_2/Cross_Species_RSA` | **no tag**; the paper is untracked in that repository; working tree read | `paper/Cross_species_rsa_paper_final_version/Cross_species_rsa_paper_final.tex` |
| 2605.30556 | `Projekte_2/training_dynamics` | **not a git repository**; working tree read | `training_dynamics_arxiv/training_dynamics.tex` |

Neither 2605.22401 nor 2605.30556 could be read at an arXiv tag, so it is not verified
that the files read are the versions currently on arXiv. Both carry an August 2026
correction note in the abstract, as described for the current arXiv versions in 2608.12408.

## Scope

Every claim in the title, abstract, results or discussion that states a number or a
direction resting on RSA with non-standardized model features (GAP features, correlation
distance, Spearman). Excluded:
- brain-only quantities (lower bounds, luminance-scalar RSA, which has no channels);
- weight-based measures (Gabor-peakedness, weight displacement);
- accuracy;
- claims the papers already withdraw in their own correction notes. Those are listed
  once per paper, not item by item.

## Coverage and categories

**Coverage by `results/step5/`:**
- **yes**: the same quantity has a stimulus-bootstrap CI under z. Step 5 covers
  Conv1 → V1 and Conv3 → LOC, Random − BP, at NATIVE 32, NATIVE 224 and UPSAMPLED 224
  (`paper_comparisons_raw_vs_z.csv`).
- **partial**: only point estimates exist (`resolution_rsa.csv`: all rules, all layer →
  ROI pairs of v12, six NATIVE resolutions and UPSAMPLED 224), or the step-5 cell is the
  nearest equivalent rather than the same cell (fixed Conv1 instead of best layer;
  endpoint instead of trajectory).
- **no**: not computed (other model sets, best-layer, calibration variants,
  ResNet/Swin, partial RSA, training trajectory, macaque data).

**Model set.** 2604.16875 uses the "repaired five-seed re-run"; step 5 uses the v12
30-checkpoint retrain of the same configuration. 2608.12408 states that the two differ at
the headline cell by +0.031 vs +0.030, less than the seed spread. Coverage of 2604
claims is therefore via an equivalent, not identical, model set.

**Categories.** `sign flips under z`, `CI claim lost under z`, `survives under z`,
`not yet testable`. With partial coverage, a category is followed by "(point
estimate)": there is no CI under z, and the category describes the direction only.

z = channel-wise z-score of the GAP features over the 720 stimuli (step 5).

---

## 2604.16875 — Untrained CNNs Exceed Backpropagation in V1 Alignment at High Evaluation Resolution

Features: GAP of Conv1 (V1, V2), Conv3 (LOC), FC1 (IT); 224 px; repaired five-seed set;
per subject, cross-run pairs. Line numbers refer to the tagged TeX file.

| # | Line | Claim (≤ 20 words) | Number (raw) | Coverage | Under z | Category |
|---|---|---|---|---|---|---|
| 1 | 17 | Title: "Untrained CNNs Exceed Backpropagation in V1 Alignment at High Evaluation Resolution" | Random − BP V1 224 +0.031 | yes (step 5 V1 NAT224) | −0.0107 [−0.0171, −0.0046] | **sign flips under z** |
| 2 | 55 | "At V1, the untrained baseline exceeds backpropagation" | 0.053 vs 0.022; +0.031 [0.019, 0.043] | yes | −0.0107 [−0.0171, −0.0046]; 0/5 seeds | **sign flips under z** |
| 3 | 55, 252 | "and the same holds at V2" | +0.017 [0.007, 0.026] | partial | −0.0066 (point) | sign flips under z (point estimate) |
| 4 | 55, 252, 346, 364 | At 32 px "the gap vanishes" | −0.001 [−0.011, 0.009] | yes (V1 NAT32) | −0.0123 [−0.0182, −0.0067] | **CI claim lost under z** (BP > Random at 32 px) |
| 5 | 55, 254 | At LOC "BP, PC and STDP exceed the random baseline" | BP − Random +0.011 [0.002, 0.021] | yes for BP | +0.0066 [+0.0024, +0.0109] | **survives under z** (BP) |
| 6 | 254 | PC − Random, STDP − Random at LOC | +0.007 [0.001, 0.014] each | partial | +0.0014 / +0.0015 (point) | not yet testable (point estimate: same sign, ~5× smaller) |
| 7 | 55, 256 | "At IT, no two conditions can be distinguished" | range 0.004–0.009 | partial | 0.005–0.012 (point) | not yet testable |
| 8 | 55, 258, 352 | "FA produces the lowest alignment at V1" | FA 0.009; all FA differences significant | partial | FA 0.0268, still lowest of the five (point) | not yet testable (point estimate: survives) |
| 9 | 55, 248, 364 | "Only the untrained network reaches the lower noise bound at V1" | Random − bound +0.002 [−0.014, 0.017] | partial | Random 0.029 vs bound 0.051 (point) | sign flips under z (point estimate) |
| 10 | 55, 364 | Early visual alignment "driven more by the architecture than by the learning rule" | rests on #2 | yes (via #2) | BP > Random | **sign flips under z** |
| 11 | 106 | Contribution (2): untrained exceeds BP at V1 and V2 | +0.031 [0.019, 0.043] | yes / partial | see #2, #3 | **sign flips under z** |
| 12 | 252 | "The order at V1 is Random > STDP > BP > PC > FA" | — | partial | BP > STDP > PC > Random > FA (point) | sign flips under z (point estimate) |
| 13 | 252 | "all ten pairwise differences at V1 are significant" | Table pairwise | partial | Random − FA +0.0024, Random − PC −0.0006, Random − STDP −0.0017 (point) | not yet testable |
| 14 | 252, 346 | 32 px: "BP and the untrained network do not differ" | 0.046 vs 0.045 | yes | 0.042 vs 0.030; −0.0123 [−0.0182, −0.0067] | **CI claim lost under z** |
| 15 | 278 | Random "highest at V1/V2 … lowest at LOC and IT" | — | partial | not highest at V1/V2; lowest at LOC (0.0029) and IT (0.0051) (point) | sign flips under z (point estimate) for V1/V2 |
| 16 | 293 | Per subject: Random − BP sub-01 +0.033, sub-02 +0.052 | [0.016, 0.051], [0.026, 0.076] | partial | −0.0166, −0.0094 (point) | sign flips under z (point estimate) |
| 17 | 293 | LOC Random − BP significant only in sub-02 | −0.024 [−0.047, −0.001] | partial | −0.0074 (point) | not yet testable |
| 18 | 304 | Seed SD small relative to the Random − BP difference | +0.031 | yes (via #2) | difference reversed | not yet testable (statement about seed spread) |
| 19 | 308 | Best layer: FC1 highest for Random at V1 | 0.102 vs 0.053 | no | — | not yet testable |
| 20 | 321 | Partial RSA: Random still exceeds BP at V1 / V2 | +0.034 [0.021, 0.046]; +0.018 [0.008, 0.027] | no | — | not yet testable |
| 21 | 348 | Calibrated baseline still exceeds BP at V1 | +0.041 / +0.033 | no (resolution-matched calibration not computed) | — | not yet testable |
| 22 | 354 | Untrained (10 % accuracy) exceeds BP (75 %) at V1: accuracy and alignment dissociated | rests on #2 | yes (via #2) | BP > Random | **sign flips under z** |
| 23 | 356 | Architecture confound "essentially complete at V1" | rests on #2 | yes (via #2) | — | **sign flips under z** |
| 24 | 364 | Conclusion: gap +0.0304 [0.0181, 0.0423] at 224 px, vanishes at 32 px | — | yes | see #2, #4 | **sign flips under z** / **CI claim lost under z** |
| 25 | 401–408 | Primary family (7, earlier 11), all significant after BH and Holm | Table `tab:fdr` | partial | see section "11 primary comparisons" | not yet testable |

Already withdrawn by the paper itself: the v1–v3 values (0.076 vs 0.034, etc.), the
"exhaust most of the available signal" claim, and PC exceeding BP at V1 (revision note,
lines 64–92).

**Title affected: yes** (#1, sign flips under z).

---

## 2608.12408 — Evaluation Resolution Confounds Learning-Rule Comparisons in Model–Brain RSA of Early Visual Cortex

Features: GAP of Conv1 (V1), Conv3 (LOC), FC1 (IT), plus all layers for best-layer;
6 resolutions + UPSAMPLED; v12 30-checkpoint retrain; per subject, cross-run pairs.

| # | Line | Claim (≤ 20 words) | Number (raw) | Coverage | Under z | Category |
|---|---|---|---|---|---|---|
| 1 | 13–14 | Title: "Evaluation Resolution Confounds Learning-Rule Comparisons in Model–Brain RSA of Early Visual Cortex" | slope S = +0.0311 [0.0239, 0.0384] | yes (S, step 5) | S_z +0.0016 [−0.0038, +0.0070] | **CI claim lost under z** (for the Random − BP comparison) |
| 2 | 26, 47, 94 | V1 gap at 224 px | +0.030 [0.018, 0.042], 5/5 | yes | −0.0107 [−0.0171, −0.0046], 0/5 | **sign flips under z** |
| 3 | 26, 94 | V1 gap at 32 px | −0.001 [−0.011, 0.009], 3/5 | yes | −0.0123 [−0.0182, −0.0067], 0/5 | **CI claim lost under z** ("≈ 0" becomes BP > Random) |
| 4 | 26, 94 | Gap "growing monotonically across six resolutions" | −0.001 … +0.030 | partial (CIs at 32 and 224 only) | −0.012, −0.016, −0.016, −0.015, −0.013, −0.011 (not monotone) | CI claim lost under z (point estimate for the shape; S has a CI, see #1) |
| 5 | 26, 96 | Best-layer gap at 32 px and its growth | +0.011 → +0.042 | no | — | not yet testable |
| 6 | 26, 49 | Holds across conditions, macaque, training trajectory, ResNet-50, Swin-Tiny | — | no | — | not yet testable |
| 7 | 26, 102–111 | 2×2 calibration: untrained still exceeds BP | +0.027 / +0.022 | no | — | not yet testable |
| 8 | 26, 158 | Content limited: gap at UP224 | +0.035 [0.023, 0.046] | yes | +0.0017 [−0.0013, +0.0048] | **CI claim lost under z** |
| 9 | 26, 158, 196, 216 | R = Gap_UP / Gap_NAT | 1.14 [1.03, 1.34] | yes | −0.16 [−0.86, +0.09] | **CI claim lost under z** (not meaningful with a negative denominator) |
| 10 | 26, 158, 160 | Limiting content widens the gap ("content … narrows it") | +0.004 [0.001, 0.008] | yes | +0.0124 [+0.0078, +0.0182] | **survives under z** |
| 11 | 26, 162 | LOC backprop advantage disappears under upsampling | UP224 +0.003 [−0.004, 0.009]; diff −0.009 [−0.017, −0.001] | yes | +0.0020 [−0.0019, +0.0058]; diff −0.0046 [−0.0077, −0.0016] | **survives under z** |
| 12 | 26, 139 | Luminance scalar 0.054 "essentially matching the untrained network's 0.053" | Random 0.053 | partial | Random 0.029 (point) | not yet testable (point estimate: no longer matching) |
| 13 | 26, 51, 179 | LOC: BP above untrained at every resolution | +0.012 (32), +0.011 (224), 5/5 | yes at 32 / 224; partial elsewhere | +0.0070 [+0.0027, +0.0117]; +0.0066 [+0.0024, +0.0109]; 5/5 | **survives under z** |
| 14 | 85 | Every trained condition peaks at 32 px and falls off: BP 0.046 → 0.023, FA, PC, STDP | — | partial | BP 0.042 → 0.040 (peak at 64), FA 0.027 → 0.027, PC 0.030 → 0.030, STDP 0.032 → 0.031 | not yet testable (point estimate: decline almost gone) |
| 15 | 85 | Untrained "climbing from 0.045 to 0.053" | +0.008 | partial | 0.030 → 0.029 | sign flips under z (point estimate) |
| 16 | 96 | Untrained best V1 layer is Conv2, backprop's Conv1 | — | no | — | not yet testable |
| 17 | 117 | ResNet-50 and Swin-Tiny V1 alignment falls toward 224 px | 0.031 → 0.023; 0.056 → 0.037 | no | — | not yet testable |
| 18 | 128–137 | Gabor and pixel similarity of models do not track V1 alignment | — | no | — | not yet testable |
| 19 | 139–143 | Partialling luminance halves untrained alignment; brightness convergence orders conditions (ρ = 0.94 / 1.00) | 0.053 → 0.026 | no | — | not yet testable |
| 20 | 160 | With content limited, backprop falls from 0.046 to 0.014–0.016 | — | partial | 0.042 → 0.025 at UP224 (point) | survives under z (point estimate) |
| 21 | 168 | Training dynamics: BP drop in first epoch and recovery; PC/FA lose at 32 px | −0.030 ± 0.005 etc. | no | — | not yet testable |
| 22 | 179 | IT: BP above untrained (+0.012 at 32 px, +0.006 at 224 px, 5/5) | — | partial | +0.009 / +0.007 (point) | not yet testable (point estimate: same sign) |
| 23 | 190 | Untrained/local rules rivalling BP at V1 is "largely a function of the evaluation resolution" | rests on #1–#3 | yes | — | **CI claim lost under z** |
| 24 | 194, 196 | Gap appears in full once input exceeds training resolution (R = 1.14) | — | yes | see #8, #9 | **CI claim lost under z** |

**Title affected: yes** (#1). Under z the Random − BP gap no longer depends on
resolution (S_z CI includes 0). It is negative at every resolution, and BP's own V1
alignment still drops under upsampling (#20).

---

## 2605.22401 — Cross-Species RSA Reveals Conserved Early Visual Alignment but Divergent Higher-Area Rankings Across Human fMRI and Macaque Electrophysiology

Features: GAP, Conv1 (V1/V2), Conv2 (V4), FC1 (IT); 224 px; original (pre-repair) model
set; macaque data via Brain-Score. Human values are quoted from 2604.16875 v2.

| # | Line (file) | Claim (≤ 20 words) | Number (raw) | Coverage | Under z | Category |
|---|---|---|---|---|---|---|
| 1 | 11 | Title: "Conserved Early Visual Alignment … Divergent Higher-Area Rankings" across species | — | no (no macaque data here) | — | not yet testable |
| 2 | 41, 189 | Macaque ρ 0.15–0.30 at V1/V2 vs human 0.01–0.08 | — | no | — | not yet testable |
| 3 | 42, 193, 254, 292 | STDP and PC highest macaque V1/V2 alignment; local rules outperform BP in both species | ≈ 0.30, 0.28 | human side partial | human V1 under z: STDP 0.031, PC 0.030 < BP 0.040 (point) | human side: sign flips under z (point estimate); macaque side not yet testable. **Already withdrawn by the paper's correction note** |
| 4 | 43, 212 | IT rankings uncorrelated across species (τ = 0.00) | — | no | — | not yet testable |
| 5 | 44, 236 | ResNet-50 at macaque IT 0.25 vs custom CNN 0.07–0.14 | — | no | — | not yet testable |
| 6 | 99 | Untrained exceeds BP at human V1 (0.076 vs 0.034) | old v2 values | yes (repaired equivalent, 2604 #2) | −0.0107 [−0.0171, −0.0046] | **sign flips under z** |
| 7 | 194 | Human V1 order Random > STDP > PC > BP > FA | old values | partial | BP > STDP > PC > Random > FA (point) | sign flips under z (point estimate) |
| 8 | 211–213 | V1 τ = 0.40, V4 τ = 0.20, V2 τ = −0.20 | — | no | — | not yet testable |
| 9 | 219–221 | Human V1 Δρ = 0.064; macaque 0.147 | — | partial (human: max − min of 5 rules) | human z range ≈ 0.013 (point) | not yet testable |
| 10 | 227–228 | Species × rule interaction: STDP −0.138 (V1), −0.124 (V2); BP +0.035 (IT) | — | no | — | not yet testable |
| 11 | 244–248 | Stimulus-control τ (V1 0.40, IT −0.40) | — | no | — | not yet testable |

**Title affected: not testable here** (macaque data not available in this repository).

---

## 2605.30556 — Supervised Training Rapidly Degrades Early Visual Cortex Alignment Across Biologically Plausible Learning Rules

Features: GAP, **best layer per ROI and epoch**, 224 px, ImageNet normalization (as
stated in the paper), eight checkpoints (epochs 0–40), original model set. The
correction note (lines 41–101) withdraws the PC/STDP preservation claim and states that
"training degrades V1 alignment for every rule tested" survives.

| # | Line | Claim (≤ 20 words) | Number (raw) | Coverage | Under z | Category |
|---|---|---|---|---|---|---|
| 1 | 19–20 | Title: "Supervised Training Rapidly Degrades Early Visual Cortex Alignment" | — | partial (endpoint only, fixed Conv1, v12 checkpoints; BP epoch 0 = Random of the same seed) | BP (epoch 40) − Random (epoch 0) at V1 224: +0.0107 [+0.0046, +0.0171] | sign flips under z (point estimate of the endpoint only; "rapidly" and the best-layer trajectory not testable) |
| 2 | 44–47 | Correction note: "training degrades V1 alignment for every rule tested does survive" | BP −0.031, PC −0.040, STDP −0.018 | partial (endpoint vs Random, fixed Conv1) | trained − Random at V1 224, z: BP +0.011, STDP +0.002, PC +0.001, FA −0.002 (point) | sign flips under z (point estimate) for BP, STDP, PC |
| 3 | 52, 160 | One epoch reduces V1 alignment by 25–90 % | — | no | — | not yet testable |
| 4 | 52, 172 | BP degrades most (Δr = −0.080); PC, STDP preserve more | — | no | — | not yet testable. **Already withdrawn by the correction note** |
| 5 | 52, 189 | LOC: BP shows the largest increase during training (+0.011) | — | partial (endpoint fixed Conv3: BP − Random) | +0.0066 [+0.0024, +0.0109] | survives under z (point estimate of the endpoint only) |
| 6 | 172 | Epoch 40 order PC > STDP > BP ≈ FA; all trained below untrained at V1 | — | partial | BP > STDP > PC > Random > FA at fixed Conv1 (point) | sign flips under z (point estimate) |
| 7 | 176 | Degradation pattern extends to V2 and V3 | — | partial for V2 only | V2 BP − Random +0.0066 (point) | sign flips under z (point estimate) |

**Title affected: yes, at the endpoint** (point estimate, fixed layer). The trajectory
and best-layer claims are not testable here.

---

## 11 primary comparisons / FDR–Holm correction

**2608.12408 contains no such list.** Its TeX (tag `arxiv-v3`) contains no FDR,
Benjamini–Hochberg or Holm correction and no list of primary comparisons. It reports one
headline CI (V1 gap) and the pre-specified content-control criterion R.

The list exists in **2604.16875** (tag `arxiv-v4`), Methods line 215 and Appendix lines
401–408 (`tab_fdr.tex`). The current primary family has 7 comparisons; the earlier
family had **11**. Verbatim from `tab_fdr.tex` (Δρ_subj [95% CI], per subject,
cross-run, repaired set):

| # | Comparison | ROI | Δρ_subj [95% CI] | Family |
|---|---|---|---|---|
| 1 | Random − BP | V1 | +0.0308 [0.0186, 0.0428] | 7 and 11 |
| 2 | Random − BP | V2 | +0.0165 [0.0072, 0.0261] | 7 and 11 |
| 3 | Random − BP | LOC | −0.0110 [−0.0206, −0.0023] | 7 and 11 |
| 4 | Random − FA | V1 | +0.0446 [0.0313, 0.0580] | 7 and 11 |
| 5 | BP − FA | V1 | +0.0138 [0.0092, 0.0186] | 7 and 11 |
| 6 | FA − PC | V1 | −0.0033 [−0.0054, −0.0011] | 7 and 11 |
| 7 | FA − STDP | V1 | −0.0179 [−0.0229, −0.0130] | 7 and 11 |
| 8 | Random − FA | V2 | +0.0252 [0.0150, 0.0361] | 11 only |
| 9 | BP − FA | V2 | +0.0087 [0.0053, 0.0125] | 11 only |
| 10 | FA − PC | V2 | −0.0021 [−0.0037, −0.0005] | 11 only |
| 11 | FA − STDP | V2 | −0.0110 [−0.0146, −0.0077] | 11 only |

Methods, line 215 (verbatim, shortened): "the untrained network vs. BP at V1 and V2, BP
vs. the untrained network at LOC, and FA vs. each other condition at V1 (7 of the 40
pairwise comparisons). They were designated in this revision, not before the original
analysis."

Under z, from step 5 point estimates (v12 set):

| # | Comparison | z, point |
|---|---|---|
| 1 | Random − BP, V1 | −0.0107 (CI [−0.0171, −0.0046]; sign flips) |
| 2 | Random − BP, V2 | −0.0066 (sign flips) |
| 3 | Random − BP, LOC | −0.0066 (CI [−0.0109, −0.0024]; survives) |
| 4 | Random − FA, V1 | +0.0024 |
| 5 | BP − FA, V1 | +0.0131 |
| 6 | FA − PC, V1 | −0.0030 |
| 7 | FA − STDP, V1 | −0.0041 |
| 8 | Random − FA, V2 | +0.0018 |
| 9 | BP − FA, V2 | +0.0084 |
| 10 | FA − PC, V2 | −0.0017 |
| 11 | FA − STDP, V2 | −0.0023 |

Only #1 and #3 have a CI under z. The step-5 comparison list (8 + 3) was taken from the
2608.12408 manifest, not from this family.

## Summary counts

Counts per category (each claim row once; #24 of 2604 counted as sign flip); partial-coverage entries are counted under their point-estimate
category.

| Paper | sign flips | CI claim lost | survives | not yet testable | Title |
|---|---|---|---|---|---|
| 2604.16875 | 12 (7 with CI) | 2 | 1 | 10 | affected (sign flips) |
| 2608.12408 | 2 (1 with CI) | 7 | 4 (3 with CI) | 11 | affected (CI claim lost) |
| 2605.22401 | 3 (1 with CI) | 0 | 0 | 8 | not testable (macaque) |
| 2605.30556 | 4 (all point estimates) | 0 | 1 (point estimate) | 2 | affected at the endpoint (point estimate) |

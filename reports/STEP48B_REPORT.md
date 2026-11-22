# Step 48b report: amplitude dominance of the RDMs

Amendment `amendments/2026-11-22_step48b-predictions.md`, configuration `config_step48b.yaml` (hash 73c5a25219bb).
`config.yaml` (206fa875cca5) unchanged. Gate and choice of the amplitude RDM: `reports/STEP48B_SETUP.md`.
Results: `results/step48b/` (`dominance.csv`, `partial.csv`, `summary.csv`); every CSV carries
`config_hash`, `git_head` (4a4dc3e) and `seed`. No new downloads; the step 48 data were used.

## 1. Timeline

| Commit | Time (local) | Content |
|---|---|---|
| — | 2026-11-22 before 16:00 | simulation with synthetic patterns (no real data); decision for the sum form |
| — | 16:02 – 16:20 | gate runs (no dominance or partial values computed) |
| 56e49d1 | 16:20:57 | config_step48b.yaml, amendment, STEP48B_SETUP.md |
| 4a4dc3e | 16:29:14 | `scripts/step48b_dominance.py`, `tests/test_step48b.py` (5/5) |
| — | 16:29:15 – 16:33:36 | run (261 s) |

## 2. Gate

Δ(T-raw, THINGS-720, V1) +0.099551 and Δ(N-raw, k = 1, V1) +0.116497 reproduce step 48 to 8.3e-17
and 2.8e-17 (mean and per subject). g and g^M: brain_gain (180 rows), brain_model_gain (1800) and
model_gain_stat (240) of `results/step48/amplitude.csv` reproduce to ≤ 1e-16. Most rows are bitwise
identical; the rest differ by 1–2 ulp (BLAS rounding of the cosine). Passed (tolerance 1e-12). The
test run of the gate gave max |d| 2.2e-16. The moment ratio m of the THINGS-720 Conv1 checkpoints
equals `results/step5/moment_ratio.csv` (NATIVE 224).

## 3. Predictions (sum form A = 2 − g_i − g_j, A^M = 2 − g^M_i − g^M_j)

| Rule | Value | Met |
|---|---|---|
| D1 model side: Spearman(model RDM, A^M) Random > BP in 5/5 seeds, each dataset | THINGS-720 5/5 (R +0.588 +0.333 +0.508 +0.521 +0.381; BP +0.258 +0.236 +0.320 +0.242 +0.207); test images 5/5 (R +0.545 +0.376 +0.454 +0.463 +0.403; BP +0.348 +0.355 +0.404 +0.330 +0.280); NSD 5/5 (R +0.708 +0.404 +0.616 +0.634 +0.459; BP +0.175 +0.194 +0.238 +0.224 +0.192) | yes |
| D2 brain side: Spearman(brain RDM, A) T-raw > T-session in 3/3; N-raw > N-session (k = 1) in ≥ 7/8 | THINGS-720 3/3 (sub-01 +0.802 vs +0.019; sub-02 +0.829 vs +0.022; sub-03 +0.502 vs +0.029); NSD 8/8 (N-raw +0.846 to +0.911 vs N-session +0.042 to +0.115) | yes |
| D3 bridge: NSD N-raw k = 1 V1, two-sided partial Δ ≤ +0.020 and ≤ +0.020 in ≥ 6/8 | +0.1165 → **−0.0092**; 8/8 ≤ +0.020 (subj01 −0.0104, subj02 −0.0100, subj03 −0.0092, subj04 −0.0096, subj05 −0.0057, subj06 −0.0164, subj07 −0.0125, subj08 −0.0002) | yes |

**Interpretation by the amendment rule:** the untrained advantage under conventions that leave a common
profile in both patterns is attributed to amplitude structure dominating both RDMs, with the untrained
network's RDM more amplitude-dominated; not to a better match of per-image amplitude.

## 4. Results

Mean over seeds 0–4 (and draws or presentations) per subject, then over subjects. NSD per-subject
values for all cells and the per-seed ρ: `results/step48b/partial.csv`.

### Delta before and after the two-sided partial (paper cell)

none = plain Spearman; sum = partial on A = 2 - g_i - g_j (D3); diff = partial on |g_i - g_j| (comparison). n<=.020: subjects with Delta <= +0.020 (sum).

| dataset | ROI | k | convention | Delta none | n > 0 | Delta sum | n > 0 | n <= .020 | Delta diff | n > 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| things720 | V1 | 1 | T-raw | +0.0996 | 3/3 | **+0.0207** | 3/3 | 1/3 | +0.0487 | 3/3 |
| things720 | V1 | 1 | T-global | +0.0304 | 3/3 | **+0.0071** | 3/3 | 3/3 | +0.0138 | 3/3 |
| things720 | V1 | 1 | T-session | +0.0052 | 3/3 | **+0.0038** | 3/3 | 3/3 | +0.0014 | 1/3 |
| things720 | LOC | 1 | T-raw | -0.0318 | 0/3 | **-0.0045** | 1/3 | 3/3 | -0.0192 | 1/3 |
| things720 | LOC | 1 | T-global | -0.0113 | 0/3 | **-0.0054** | 0/3 | 3/3 | -0.0135 | 0/3 |
| things720 | LOC | 1 | T-session | -0.0044 | 0/3 | **-0.0046** | 0/3 | 3/3 | -0.0076 | 0/3 |
| things_test | V1 | 1 | T-raw | +0.0784 | 3/3 | **+0.0138** | 2/3 | 2/3 | +0.0370 | 3/3 |
| things_test | V1 | 12 | T-raw | +0.1651 | 3/3 | **+0.0513** | 2/3 | 1/3 | +0.0712 | 3/3 |
| things_test | V1 | 1 | T-global | -0.0002 | 1/3 | **-0.0047** | 1/3 | 3/3 | -0.0055 | 1/3 |
| things_test | V1 | 12 | T-global | -0.0178 | 1/3 | **-0.0160** | 1/3 | 3/3 | -0.0273 | 1/3 |
| things_test | V1 | 1 | T-session | -0.0066 | 0/3 | **-0.0052** | 0/3 | 3/3 | -0.0054 | 1/3 |
| things_test | V1 | 12 | T-session | -0.0177 | 1/3 | **-0.0161** | 1/3 | 3/3 | -0.0273 | 1/3 |
| things_test | LOC | 1 | T-raw | +0.0098 | 2/3 | **-0.0022** | 1/3 | 3/3 | +0.0108 | 2/3 |
| things_test | LOC | 12 | T-raw | +0.0011 | 2/3 | **+0.0044** | 2/3 | 2/3 | +0.0053 | 2/3 |
| things_test | LOC | 1 | T-global | -0.0004 | 1/3 | **-0.0028** | 1/3 | 3/3 | -0.0000 | 1/3 |
| things_test | LOC | 12 | T-global | -0.0161 | 0/3 | **-0.0113** | 1/3 | 3/3 | -0.0193 | 0/3 |
| things_test | LOC | 1 | T-session | -0.0028 | 1/3 | **-0.0030** | 1/3 | 3/3 | -0.0023 | 1/3 |
| things_test | LOC | 12 | T-session | -0.0160 | 0/3 | **-0.0112** | 1/3 | 3/3 | -0.0193 | 0/3 |
| nsd515 | V1 | 1 | N-raw | +0.1165 | 8/8 | **-0.0092** | 0/8 | 8/8 | +0.0483 | 8/8 |
| nsd515 | V1 | 3 | N-raw | +0.1656 | 8/8 | **-0.0144** | 0/8 | 8/8 | +0.0682 | 8/8 |
| nsd515 | V1 | 1 | N-global | -0.0122 | 0/8 | **-0.0010** | 2/8 | 8/8 | -0.0129 | 0/8 |
| nsd515 | V1 | 3 | N-global | -0.0187 | 0/8 | **-0.0028** | 2/8 | 8/8 | -0.0192 | 0/8 |
| nsd515 | V1 | 1 | N-session | -0.0120 | 0/8 | **-0.0020** | 2/8 | 8/8 | -0.0131 | 0/8 |
| nsd515 | V1 | 3 | N-session | -0.0169 | 0/8 | **-0.0026** | 2/8 | 8/8 | -0.0178 | 0/8 |
| nsd515 | LO-obj | 1 | N-raw | +0.0068 | 6/8 | **-0.0046** | 1/8 | 8/8 | +0.0038 | 5/8 |
| nsd515 | LO-obj | 3 | N-raw | +0.0126 | 6/8 | **-0.0105** | 0/8 | 8/8 | +0.0053 | 5/8 |
| nsd515 | LO-obj | 1 | N-global | -0.0074 | 0/8 | **-0.0048** | 0/8 | 8/8 | -0.0078 | 0/8 |
| nsd515 | LO-obj | 3 | N-global | -0.0131 | 0/8 | **-0.0099** | 0/8 | 8/8 | -0.0126 | 0/8 |
| nsd515 | LO-obj | 1 | N-session | -0.0081 | 0/8 | **-0.0049** | 0/8 | 8/8 | -0.0084 | 0/8 |
| nsd515 | LO-obj | 3 | N-session | -0.0139 | 0/8 | **-0.0103** | 0/8 | 8/8 | -0.0132 | 0/8 |
| nsd515 | lateral | 1 | N-raw | -0.0007 | 5/8 | **-0.0074** | 0/8 | 8/8 | -0.0020 | 3/8 |
| nsd515 | lateral | 3 | N-raw | -0.0003 | 6/8 | **-0.0150** | 0/8 | 8/8 | -0.0021 | 4/8 |
| nsd515 | lateral | 1 | N-global | -0.0091 | 0/8 | **-0.0057** | 0/8 | 8/8 | -0.0101 | 0/8 |
| nsd515 | lateral | 3 | N-global | -0.0148 | 0/8 | **-0.0106** | 0/8 | 8/8 | -0.0152 | 0/8 |
| nsd515 | lateral | 1 | N-session | -0.0097 | 0/8 | **-0.0056** | 0/8 | 8/8 | -0.0104 | 0/8 |
| nsd515 | lateral | 3 | N-session | -0.0156 | 0/8 | **-0.0109** | 0/8 | 8/8 | -0.0157 | 0/8 |

### Per subject, V1, k = 1 (THINGS-720, NSD) : Delta none -> sum

| dataset | convention | subject | none | sum | diff |
|---|---|---|---|---|---|
| nsd515 | N-global | subj01 | -0.0144 | -0.0038 | -0.0166 |
| nsd515 | N-global | subj02 | -0.0052 | +0.0045 | -0.0101 |
| nsd515 | N-global | subj03 | -0.0065 | +0.0008 | -0.0077 |
| nsd515 | N-global | subj04 | -0.0209 | -0.0021 | -0.0196 |
| nsd515 | N-global | subj05 | -0.0154 | -0.0003 | -0.0158 |
| nsd515 | N-global | subj06 | -0.0149 | -0.0034 | -0.0159 |
| nsd515 | N-global | subj07 | -0.0093 | -0.0028 | -0.0075 |
| nsd515 | N-global | subj08 | -0.0111 | -0.0012 | -0.0101 |
| nsd515 | N-raw | subj01 | +0.1401 | -0.0104 | +0.0620 |
| nsd515 | N-raw | subj02 | +0.1427 | -0.0100 | +0.0618 |
| nsd515 | N-raw | subj03 | +0.1039 | -0.0092 | +0.0443 |
| nsd515 | N-raw | subj04 | +0.1247 | -0.0096 | +0.0413 |
| nsd515 | N-raw | subj05 | +0.1012 | -0.0057 | +0.0461 |
| nsd515 | N-raw | subj06 | +0.1329 | -0.0164 | +0.0550 |
| nsd515 | N-raw | subj07 | +0.0814 | -0.0125 | +0.0308 |
| nsd515 | N-raw | subj08 | +0.1052 | -0.0002 | +0.0453 |
| nsd515 | N-session | subj01 | -0.0140 | -0.0052 | -0.0163 |
| nsd515 | N-session | subj02 | -0.0073 | +0.0024 | -0.0109 |
| nsd515 | N-session | subj03 | -0.0070 | -0.0018 | -0.0075 |
| nsd515 | N-session | subj04 | -0.0200 | -0.0030 | -0.0200 |
| nsd515 | N-session | subj05 | -0.0142 | +0.0011 | -0.0169 |
| nsd515 | N-session | subj06 | -0.0139 | -0.0050 | -0.0165 |
| nsd515 | N-session | subj07 | -0.0101 | -0.0036 | -0.0088 |
| nsd515 | N-session | subj08 | -0.0097 | -0.0013 | -0.0081 |
| things720 | T-global | sub-01 | +0.0330 | +0.0067 | +0.0132 |
| things720 | T-global | sub-02 | +0.0509 | +0.0144 | +0.0259 |
| things720 | T-global | sub-03 | +0.0074 | +0.0001 | +0.0022 |
| things720 | T-raw | sub-01 | +0.1150 | +0.0217 | +0.0571 |
| things720 | T-raw | sub-02 | +0.1111 | +0.0283 | +0.0519 |
| things720 | T-raw | sub-03 | +0.0725 | +0.0119 | +0.0372 |
| things720 | T-session | sub-01 | +0.0043 | +0.0026 | -0.0009 |
| things720 | T-session | sub-02 | +0.0100 | +0.0084 | +0.0055 |
| things720 | T-session | sub-03 | +0.0013 | +0.0005 | -0.0004 |

### Model dominance Spearman(model RDM, A^M), per seed

| dataset | layer | form | model | seed 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| things720 | Conv1 | sum | Random | +0.588 | +0.333 | +0.508 | +0.521 | +0.381 | +0.466 |
| things720 | Conv1 | diff | Random | +0.550 | +0.207 | +0.446 | +0.481 | +0.279 | +0.393 |
| things720 | Conv1 | sum | BP | +0.258 | +0.236 | +0.320 | +0.242 | +0.207 | +0.253 |
| things720 | Conv1 | diff | BP | +0.149 | +0.130 | +0.191 | +0.153 | +0.104 | +0.145 |
| things720 | Conv3 | sum | Random | +0.725 | +0.595 | +0.676 | +0.696 | +0.632 | +0.664 |
| things720 | Conv3 | diff | Random | +0.642 | +0.487 | +0.539 | +0.606 | +0.554 | +0.566 |
| things720 | Conv3 | sum | BP | +0.441 | +0.534 | +0.623 | +0.488 | +0.532 | +0.524 |
| things720 | Conv3 | diff | BP | +0.307 | +0.390 | +0.470 | +0.345 | +0.376 | +0.378 |
| things_test | Conv1 | sum | Random | +0.545 | +0.376 | +0.454 | +0.463 | +0.403 | +0.448 |
| things_test | Conv1 | diff | Random | +0.524 | +0.294 | +0.407 | +0.428 | +0.325 | +0.396 |
| things_test | Conv1 | sum | BP | +0.348 | +0.355 | +0.404 | +0.330 | +0.280 | +0.344 |
| things_test | Conv1 | diff | BP | +0.202 | +0.218 | +0.245 | +0.211 | +0.135 | +0.202 |
| nsd515 | Conv1 | sum | Random | +0.708 | +0.404 | +0.616 | +0.634 | +0.459 | +0.564 |
| nsd515 | Conv1 | diff | Random | +0.641 | +0.311 | +0.568 | +0.597 | +0.396 | +0.503 |
| nsd515 | Conv1 | sum | BP | +0.175 | +0.194 | +0.238 | +0.224 | +0.192 | +0.205 |
| nsd515 | Conv1 | diff | BP | +0.103 | +0.119 | +0.148 | +0.126 | +0.117 | +0.123 |

### Brain dominance Spearman(brain RDM, A), mean over subjects (range)

| dataset | ROI | k | convention | sum | diff |
|---|---|---|---|---|---|
| things720 | V1 | 1 | T-raw | +0.711 [+0.50, +0.83] | +0.407 [+0.33, +0.47] |
| things720 | V1 | 1 | T-global | +0.225 [+0.07, +0.38] | +0.302 [+0.20, +0.37] |
| things720 | V1 | 1 | T-session | +0.023 [+0.02, +0.03] | +0.268 [+0.20, +0.35] |
| things720 | LOC | 1 | T-raw | +0.690 [+0.52, +0.84] | +0.368 [+0.30, +0.41] |
| things720 | LOC | 1 | T-global | +0.149 [+0.06, +0.32] | +0.172 [+0.09, +0.33] |
| things720 | LOC | 1 | T-session | +0.014 [+0.01, +0.02] | +0.111 [+0.07, +0.14] |
| things_test | V1 | 1 | T-raw | +0.666 [+0.54, +0.73] | +0.366 [+0.35, +0.38] |
| things_test | V1 | 12 | T-raw | +0.907 [+0.85, +0.94] | +0.590 [+0.49, +0.66] |
| things_test | V1 | 1 | T-global | +0.189 [+0.14, +0.22] | +0.307 [+0.28, +0.33] |
| things_test | V1 | 12 | T-global | +0.080 [+0.03, +0.13] | +0.352 [+0.20, +0.49] |
| things_test | V1 | 1 | T-session | +0.122 [+0.10, +0.13] | +0.294 [+0.23, +0.34] |
| things_test | V1 | 12 | T-session | +0.080 [+0.03, +0.13] | +0.353 [+0.20, +0.49] |
| things_test | LOC | 1 | T-raw | +0.624 [+0.45, +0.75] | +0.374 [+0.31, +0.46] |
| things_test | LOC | 12 | T-raw | +0.854 [+0.83, +0.90] | +0.522 [+0.44, +0.58] |
| things_test | LOC | 1 | T-global | +0.174 [+0.14, +0.21] | +0.211 [+0.12, +0.26] |
| things_test | LOC | 12 | T-global | +0.300 [+0.22, +0.40] | +0.459 [+0.40, +0.49] |
| things_test | LOC | 1 | T-session | +0.138 [+0.12, +0.16] | +0.214 [+0.18, +0.24] |
| things_test | LOC | 12 | T-session | +0.300 [+0.22, +0.40] | +0.462 [+0.40, +0.49] |
| nsd515 | V1 | 1 | N-raw | +0.883 [+0.85, +0.91] | +0.601 [+0.51, +0.67] |
| nsd515 | V1 | 3 | N-raw | +0.889 [+0.86, +0.90] | +0.581 [+0.53, +0.62] |
| nsd515 | V1 | 1 | N-global | +0.090 [+0.07, +0.13] | +0.220 [+0.17, +0.26] |
| nsd515 | V1 | 3 | N-global | +0.080 [+0.04, +0.13] | +0.268 [+0.20, +0.33] |
| nsd515 | V1 | 1 | N-session | +0.073 [+0.04, +0.11] | +0.231 [+0.18, +0.28] |
| nsd515 | V1 | 3 | N-session | +0.078 [+0.04, +0.13] | +0.282 [+0.22, +0.34] |
| nsd515 | LO-obj | 1 | N-raw | +0.759 [+0.35, +0.92] | +0.494 [+0.36, +0.63] |
| nsd515 | LO-obj | 3 | N-raw | +0.827 [+0.53, +0.91] | +0.474 [+0.36, +0.55] |
| nsd515 | LO-obj | 1 | N-global | +0.178 [+0.10, +0.25] | +0.362 [+0.22, +0.53] |
| nsd515 | LO-obj | 3 | N-global | +0.163 [+0.14, +0.24] | +0.336 [+0.24, +0.49] |
| nsd515 | LO-obj | 1 | N-session | +0.158 [+0.09, +0.22] | +0.373 [+0.24, +0.55] |
| nsd515 | LO-obj | 3 | N-session | +0.165 [+0.14, +0.22] | +0.342 [+0.27, +0.52] |
| nsd515 | lateral | 1 | N-raw | +0.778 [+0.54, +0.88] | +0.462 [+0.26, +0.56] |
| nsd515 | lateral | 3 | N-raw | +0.803 [+0.73, +0.87] | +0.450 [+0.31, +0.58] |
| nsd515 | lateral | 1 | N-global | +0.177 [+0.11, +0.25] | +0.342 [+0.23, +0.48] |
| nsd515 | lateral | 3 | N-global | +0.179 [+0.15, +0.26] | +0.361 [+0.27, +0.43] |
| nsd515 | lateral | 1 | N-session | +0.161 [+0.10, +0.22] | +0.352 [+0.23, +0.48] |
| nsd515 | lateral | 3 | N-session | +0.183 [+0.15, +0.26] | +0.365 [+0.30, +0.42] |

### Model dominance vs moment ratio m across the 10 checkpoints (Spearman)

| dataset | layer | form | rho | m Random (mean) | m BP (mean) |
|---|---|---|---|---|---|
| nsd515 | Conv1 | diff | +0.842 | 0.759 | 0.732 |
| nsd515 | Conv1 | sum | +0.842 | 0.759 | 0.732 |
| things720 | Conv1 | diff | +0.855 | 0.707 | 0.638 |
| things720 | Conv1 | sum | +0.879 | 0.707 | 0.638 |
| things720 | Conv3 | diff | +0.721 | 0.878 | 0.718 |
| things720 | Conv3 | sum | +0.648 | 0.878 | 0.718 |
| things_test | Conv1 | diff | +0.600 | 0.720 | 0.697 |
| things_test | Conv1 | sum | +0.673 | 0.720 | 0.697 |


## 5. Runtimes

| Part | Estimate (× 1.5) | Actual |
|---|---|---|
| Gate | — | 207 s (test run 4 min) |
| THINGS-720 | | 20 s |
| THINGS test images | | 9 s |
| NSD | | 232 s |
| Run total | 0.26 h | 261 s |

## 6. Open points

1. **Choice of the amplitude RDM.** The draft defined A as |g_i − g_j|. A simulation on synthetic
   patterns before the amendment showed that this difference form does not capture the amplitude
   structure of correlation-distance RDMs under a common profile. The amendment uses the sum form for
   D1–D3 and reports the difference form without prediction. Test (i) documents the difference form
   without requiring it: no monotone rise, and the partial does not remove a pure amplitude effect. On
   the real data the difference-form partial reduces the NSD N-raw V1 Δ only to +0.048 (8/8 positive),
   THINGS-720 T-raw to +0.049.
2. **THINGS is not reduced as far as NSD (no prediction).** After the sum-form partial, THINGS-720
   T-raw is +0.0207 (3/3 positive, 1/3 ≤ +0.020) and T-global +0.0071 (3/3 positive). On the test
   images T-raw is +0.0138 at k = 1 and +0.0513 at k = 12. NSD N-raw V1 becomes negative (−0.0092 at
   k = 1, −0.0144 at k = 3). In THINGS part of the untrained advantage under T-raw and T-global stays
   after removing amplitude structure on both sides.
3. **"Spearman of the residuals":** the amendment says Spearman of the residuals; the implementation
   is the step 44 one it refers to ("rank-based, as in step 44"): ranks residualised on the ranks of
   A or A^M (OLS with intercept), then the Pearson correlation of the rank residuals. This is stated
   in `config_step48b.yaml`.
4. **Pair set of the model-side dominance (D1):** THINGS-720 the cross-run pairs, test images and
   NSD all pairs. The NSD k = 1 pair sets depend on the subject, so the model side is not tied to
   them; the brain side and the partial use the step 48 pair sets.
5. **Defect in the step 48 results (not changed):** the `seed` column of the THINGS rows of
   `results/step48/amplitude.csv` is empty (`step48_conventions.write_part` drops the seed column
   of the existing part when merging). The values are complete; the gate reconstructs the seed from
   the row order. `conventions.csv` is not affected (its THINGS rows have no seed). The step 48b
   outputs do not use `write_part`.
6. **Gate at 1–2 ulp instead of bitwise:** g and g^M agree to ≤ 2.2e-16; bitwise identity holds for
   most but not all rows (BLAS rounding depending on memory alignment).
7. Brain dominance under the difference form stays at +0.2 to +0.4 under T-session and N-session,
   where the sum form is near 0 (V1: +0.023, +0.073). The difference form measures something other
   than the profile-driven amplitude structure.
8. Model dominance and the moment ratio m (arXiv:2611.15533) across the 10 checkpoints: Spearman +0.84
   (NSD), +0.88 (THINGS-720 Conv1), +0.67 (test images), +0.65 (THINGS-720 Conv3); m is larger for
   Random than for BP in every dataset.

# Step 48 report: mechanism across both datasets, normalisation convention and response amplitude

Amendment `amendments/2026-11-22_step48-predictions.md`, configuration `config_step48.yaml` (hash 159c5c78e854).
`config.yaml` (206fa875cca5) unchanged. Setup, unit of the THINGS values, gates and estimates:
`reports/STEP48_SETUP.md`. Results: `results/step48/` (`conventions.csv`, `amplitude.csv`,
`image_stats_summary.csv`, `summary.csv`, `nsd_raw_log.csv`). Every CSV carries `config_hash`, `git_head`
(830f471) and `seed` (empty for rows aggregated over seeds or without seeds, as in earlier steps).

## 1. Timeline

| Commit | Time (local) | Content |
|---|---|---|
| 01e3790 | 2026-11-22 11:45:46 | config_step48.yaml, amendment, STEP48_SETUP.md sections 1–4 |
| — | 11:46:30 (10:46:30 UTC) | first NSD download (gate subj01 session 1), after the amendment commit |
| 830f471 | 11:57:53 | scripts, tests (15/15), STEP48_SETUP.md sections 5–6 |
| — | 11:59:50 – 14:14 | NSD reload, 240 sessions (8073 s) |
| — | 14:18:31 – 14:40:40 | NSD conventions (RSA 390 s, multiverse 931 s) |
| — | 14:41 | NSD amplitude (48 s), summary |

THINGS parts were run before the NSD reload (conventions 40 s, amplitude 56 s).

## 2. Gates

| Gate | Result |
|---|---|
| THINGS: our z-score of the h5 values = arm_a | exact (`np.array_equal`), 3/3 subjects |
| T-global V1 paper cell = step 47 | \|d\| 4.2e-17 (+0.030446) |
| T-global test-image Δ = step 41 | max \|d\| 9.4e-17 |
| NSD reload: z from raw betas and session statistics = step 42 trials | exact in **240/240** sessions (8 subjects × 30); 6 voxels with sd = 0 in total, `valid` identical |
| N-session ρ = step 42 `subjects.csv` (k = 1 and 3) | max \|d\| 1.0e-16 |
| Multiverse N-raw paper cell = N-raw RSA | max \|d\| 2.5e-16; within classes 0, between classes ≥ 0.012 |
| Image statistics L, C = step 44 | max \|d\| 0 (all three sets) |

Free disk space never below 105 GB. NSD raw data are stored only under `NSD_DIR/raw48/` and in
`RIDGE_DATA_DIR/step48/`; the repository receives only aggregated values.

## 3. Rules of the amendment

| Rule | Quantity | Value | Met |
|---|---|---|---|
| NB1 | NSD V1 paper cell k = 1: Δ(N-raw) > Δ(N-session) in ≥ 6 of 8 | 8/8 (mean N-raw +0.1165, N-global −0.0122, N-session −0.0120) | yes |
| NB2 | NSD V1 paper cell k = 1: Δ(N-raw) > 0 in ≥ 6 of 8 and sign-flip p ≤ 0.05 | 8/8, p 0.0039 | yes |
| NB3 | THINGS-720 V1 paper cell: Δ(T-session) ≤ +0.010 | +0.0052 (T-raw +0.0996, T-global +0.0304) | yes |
| G1 | THINGS-720 T-global V1: \|ρ(g, CE)\| > \|ρ(g, L)\| in ≥ 2 of 3 | 3/3 (CE +0.254 / +0.278 / +0.216; L −0.197 / −0.151 / −0.183) | yes |
| G2 | THINGS-720 T-global V1: ρ(g, g^M Random) > ρ(g, g^M BP) in 3 of 3 | 1/3 (R +0.131 / +0.156 / +0.098; BP +0.158 / +0.160 / +0.054) | no |
| G3 | NSD N-raw k = 1 V1: ρ(g, g^M Random) > ρ(g, g^M BP) in ≥ 6 of 8 | 5/8 (mean R +0.189, BP +0.187) | no |

**Interpretation by the amendment rule:** no statement of the rule applies

NB1 and NB2 are met, G2 and G3 are not. The rule has no case for this pattern (see open points).

## 4. Δ (Random − BP) per convention, dataset, k and cell

Mean over seeds 0–4 per subject; "mean" is the group value. NSD p: one-sided sign-flip test (positive).

### THINGS-720 (cross-run pairs)

| ROI | cell | convention | sub-01 | sub-02 | sub-03 | mean | n > 0 |
|---|---|---|---|---|---|---|---|
| V1 | raw / correlation | T-raw | +0.1150 | +0.1111 | +0.0725 | **+0.0996** | 3/3 |
| V1 | raw / correlation | T-global | +0.0330 | +0.0509 | +0.0074 | **+0.0304** | 3/3 |
| V1 | raw / correlation | T-session | +0.0043 | +0.0100 | +0.0013 | **+0.0052** | 3/3 |
| V1 | raw / euclidean | T-raw | +0.0125 | +0.0416 | +0.0148 | **+0.0230** | 3/3 |
| V1 | raw / euclidean | T-global | +0.0124 | +0.0430 | +0.0141 | **+0.0232** | 3/3 |
| V1 | raw / euclidean | T-session | +0.0120 | +0.0435 | +0.0146 | **+0.0234** | 3/3 |
| V1 | z / correlation | T-raw | +0.0006 | +0.0058 | +0.0004 | **+0.0023** | 3/3 |
| V1 | z / correlation | T-global | -0.0166 | -0.0094 | -0.0062 | **-0.0107** | 0/3 |
| V1 | z / correlation | T-session | -0.0193 | -0.0156 | -0.0068 | **-0.0139** | 0/3 |
| V1 | z / euclidean | T-raw | +0.0054 | +0.0202 | +0.0223 | **+0.0160** | 3/3 |
| V1 | z / euclidean | T-global | +0.0057 | +0.0211 | +0.0220 | **+0.0162** | 3/3 |
| V1 | z / euclidean | T-session | +0.0051 | +0.0216 | +0.0223 | **+0.0163** | 3/3 |
| LOC | raw / correlation | T-raw | -0.0093 | -0.0311 | -0.0550 | **-0.0318** | 0/3 |
| LOC | raw / correlation | T-global | -0.0016 | -0.0242 | -0.0080 | **-0.0113** | 0/3 |
| LOC | raw / correlation | T-session | -0.0034 | -0.0058 | -0.0040 | **-0.0044** | 0/3 |
| LOC | raw / euclidean | T-raw | -0.0307 | +0.0098 | -0.0099 | **-0.0103** | 1/3 |
| LOC | raw / euclidean | T-global | -0.0300 | +0.0069 | -0.0121 | **-0.0117** | 1/3 |
| LOC | raw / euclidean | T-session | -0.0298 | +0.0071 | -0.0125 | **-0.0117** | 1/3 |
| LOC | z / correlation | T-raw | -0.0024 | -0.0008 | -0.0093 | **-0.0042** | 0/3 |
| LOC | z / correlation | T-global | -0.0057 | -0.0074 | -0.0067 | **-0.0066** | 0/3 |
| LOC | z / correlation | T-session | -0.0060 | -0.0080 | -0.0066 | **-0.0069** | 0/3 |
| LOC | z / euclidean | T-raw | -0.0331 | +0.0088 | -0.0053 | **-0.0098** | 1/3 |
| LOC | z / euclidean | T-global | -0.0331 | +0.0071 | -0.0062 | **-0.0107** | 1/3 |
| LOC | z / euclidean | T-session | -0.0332 | +0.0075 | -0.0064 | **-0.0107** | 1/3 |

### THINGS test images

| ROI | cell | k | convention | sub-01 | sub-02 | sub-03 | mean | n > 0 |
|---|---|---|---|---|---|---|---|---|
| V1 | raw / correlation | 1 | T-global | -0.0138 | +0.0145 | -0.0011 | **-0.0002** | 1/3 |
| V1 | raw / correlation | 1 | T-raw | +0.0731 | +0.1290 | +0.0330 | **+0.0784** | 3/3 |
| V1 | raw / correlation | 1 | T-session | -0.0152 | -0.0027 | -0.0020 | **-0.0066** | 0/3 |
| V1 | raw / correlation | 12 | T-global | -0.0360 | +0.0068 | -0.0241 | **-0.0178** | 1/3 |
| V1 | raw / correlation | 12 | T-raw | +0.1759 | +0.2025 | +0.1168 | **+0.1651** | 3/3 |
| V1 | raw / correlation | 12 | T-session | -0.0361 | +0.0072 | -0.0243 | **-0.0177** | 1/3 |
| V1 | raw / euclidean | 1 | T-global | -0.0039 | +0.0196 | -0.0166 | **-0.0003** | 1/3 |
| V1 | raw / euclidean | 1 | T-raw | -0.0037 | +0.0203 | -0.0156 | **+0.0003** | 1/3 |
| V1 | raw / euclidean | 1 | T-session | -0.0038 | +0.0195 | -0.0166 | **-0.0003** | 1/3 |
| V1 | raw / euclidean | 12 | T-global | -0.0258 | +0.0286 | -0.0694 | **-0.0222** | 1/3 |
| V1 | raw / euclidean | 12 | T-raw | -0.0259 | +0.0244 | -0.0708 | **-0.0241** | 1/3 |
| V1 | raw / euclidean | 12 | T-session | -0.0249 | +0.0294 | -0.0693 | **-0.0216** | 1/3 |
| V1 | z / correlation | 1 | T-global | -0.0208 | -0.0211 | -0.0060 | **-0.0159** | 0/3 |
| V1 | z / correlation | 1 | T-raw | -0.0067 | +0.0002 | +0.0000 | **-0.0022** | 2/3 |
| V1 | z / correlation | 1 | T-session | -0.0223 | -0.0235 | -0.0062 | **-0.0173** | 0/3 |
| V1 | z / correlation | 12 | T-global | -0.0844 | -0.0817 | -0.0522 | **-0.0728** | 0/3 |
| V1 | z / correlation | 12 | T-raw | -0.0153 | -0.0046 | +0.0025 | **-0.0058** | 1/3 |
| V1 | z / correlation | 12 | T-session | -0.0846 | -0.0814 | -0.0524 | **-0.0728** | 0/3 |
| V1 | z / euclidean | 1 | T-global | -0.0052 | -0.0005 | +0.0024 | **-0.0011** | 1/3 |
| V1 | z / euclidean | 1 | T-raw | -0.0055 | -0.0004 | +0.0035 | **-0.0008** | 1/3 |
| V1 | z / euclidean | 1 | T-session | -0.0051 | -0.0007 | +0.0026 | **-0.0011** | 1/3 |
| V1 | z / euclidean | 12 | T-global | -0.0396 | -0.0168 | -0.0341 | **-0.0302** | 0/3 |
| V1 | z / euclidean | 12 | T-raw | -0.0398 | -0.0210 | -0.0350 | **-0.0319** | 0/3 |
| V1 | z / euclidean | 12 | T-session | -0.0392 | -0.0162 | -0.0340 | **-0.0298** | 0/3 |
| LOC | raw / correlation | 1 | T-global | -0.0036 | -0.0064 | +0.0089 | **-0.0004** | 1/3 |
| LOC | raw / correlation | 1 | T-raw | +0.0103 | +0.0262 | -0.0072 | **+0.0098** | 2/3 |
| LOC | raw / correlation | 1 | T-session | -0.0064 | -0.0084 | +0.0064 | **-0.0028** | 1/3 |
| LOC | raw / correlation | 12 | T-global | -0.0111 | -0.0202 | -0.0171 | **-0.0161** | 0/3 |
| LOC | raw / correlation | 12 | T-raw | +0.0275 | +0.0505 | -0.0748 | **+0.0011** | 2/3 |
| LOC | raw / correlation | 12 | T-session | -0.0109 | -0.0201 | -0.0171 | **-0.0160** | 0/3 |
| LOC | raw / euclidean | 1 | T-global | +0.0029 | -0.0052 | +0.0126 | **+0.0034** | 2/3 |
| LOC | raw / euclidean | 1 | T-raw | +0.0036 | -0.0041 | +0.0144 | **+0.0046** | 2/3 |
| LOC | raw / euclidean | 1 | T-session | +0.0027 | -0.0059 | +0.0115 | **+0.0028** | 2/3 |
| LOC | raw / euclidean | 12 | T-global | +0.0143 | +0.0430 | -0.0198 | **+0.0125** | 2/3 |
| LOC | raw / euclidean | 12 | T-raw | +0.0149 | +0.0379 | -0.0138 | **+0.0130** | 2/3 |
| LOC | raw / euclidean | 12 | T-session | +0.0137 | +0.0428 | -0.0203 | **+0.0121** | 2/3 |
| LOC | z / correlation | 1 | T-global | -0.0048 | -0.0025 | +0.0024 | **-0.0016** | 1/3 |
| LOC | z / correlation | 1 | T-raw | -0.0064 | -0.0002 | +0.0010 | **-0.0019** | 1/3 |
| LOC | z / correlation | 1 | T-session | -0.0047 | -0.0032 | +0.0021 | **-0.0019** | 1/3 |
| LOC | z / correlation | 12 | T-global | -0.0146 | -0.0021 | -0.0032 | **-0.0066** | 0/3 |
| LOC | z / correlation | 12 | T-raw | -0.0087 | +0.0022 | -0.0029 | **-0.0031** | 1/3 |
| LOC | z / correlation | 12 | T-session | -0.0145 | -0.0018 | -0.0033 | **-0.0065** | 0/3 |
| LOC | z / euclidean | 1 | T-global | +0.0003 | -0.0104 | +0.0102 | **+0.0000** | 2/3 |
| LOC | z / euclidean | 1 | T-raw | -0.0015 | -0.0093 | +0.0106 | **-0.0001** | 1/3 |
| LOC | z / euclidean | 1 | T-session | +0.0002 | -0.0107 | +0.0090 | **-0.0005** | 2/3 |
| LOC | z / euclidean | 12 | T-global | -0.0102 | +0.0250 | -0.0043 | **+0.0035** | 1/3 |
| LOC | z / euclidean | 12 | T-raw | -0.0116 | +0.0227 | -0.0015 | **+0.0032** | 1/3 |
| LOC | z / euclidean | 12 | T-session | -0.0109 | +0.0249 | -0.0045 | **+0.0031** | 1/3 |

### NSD-515

| ROI | cell | k | convention | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | mean | n > 0 | p (>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 | raw / correlation | 1 | N-global | -0.014 | -0.005 | -0.007 | -0.021 | -0.015 | -0.015 | -0.009 | -0.011 | **-0.0122** | 0/8 | 1.0000 |
| V1 | raw / correlation | 1 | N-raw | +0.140 | +0.143 | +0.104 | +0.125 | +0.101 | +0.133 | +0.081 | +0.105 | **+0.1165** | 8/8 | 0.0039 |
| V1 | raw / correlation | 1 | N-session | -0.014 | -0.007 | -0.007 | -0.020 | -0.014 | -0.014 | -0.010 | -0.010 | **-0.0120** | 0/8 | 1.0000 |
| V1 | raw / correlation | 3 | N-global | -0.020 | -0.005 | -0.015 | -0.031 | -0.023 | -0.022 | -0.015 | -0.019 | **-0.0187** | 0/8 | 1.0000 |
| V1 | raw / correlation | 3 | N-raw | +0.189 | +0.184 | +0.162 | +0.166 | +0.151 | +0.187 | +0.128 | +0.157 | **+0.1656** | 8/8 | 0.0039 |
| V1 | raw / correlation | 3 | N-session | -0.017 | -0.006 | -0.014 | -0.027 | -0.020 | -0.021 | -0.015 | -0.016 | **-0.0169** | 0/8 | 1.0000 |
| V1 | raw / euclidean | 1 | N-global | +0.013 | -0.005 | +0.001 | -0.012 | -0.005 | -0.014 | -0.000 | -0.017 | **-0.0047** | 2/8 | 0.9141 |
| V1 | raw / euclidean | 1 | N-raw | +0.011 | +0.003 | -0.009 | -0.009 | -0.009 | -0.012 | -0.004 | -0.013 | **-0.0052** | 2/8 | 0.9375 |
| V1 | raw / euclidean | 1 | N-session | +0.010 | -0.006 | -0.001 | -0.010 | -0.008 | -0.004 | +0.004 | -0.007 | **-0.0027** | 2/8 | 0.8672 |
| V1 | raw / euclidean | 3 | N-global | +0.020 | -0.005 | -0.015 | -0.011 | +0.002 | -0.025 | -0.020 | -0.013 | **-0.0084** | 2/8 | 0.9258 |
| V1 | raw / euclidean | 3 | N-raw | +0.014 | +0.006 | -0.027 | -0.010 | -0.002 | -0.023 | -0.024 | -0.011 | **-0.0098** | 2/8 | 0.9453 |
| V1 | raw / euclidean | 3 | N-session | +0.013 | -0.012 | -0.017 | -0.007 | -0.002 | -0.020 | -0.020 | +0.000 | **-0.0081** | 2/8 | 0.9570 |
| V1 | z / correlation | 1 | N-global | -0.015 | -0.022 | -0.012 | -0.012 | -0.019 | -0.010 | -0.009 | -0.021 | **-0.0148** | 0/8 | 1.0000 |
| V1 | z / correlation | 1 | N-raw | -0.004 | -0.009 | -0.004 | -0.006 | -0.004 | -0.002 | -0.002 | -0.003 | **-0.0043** | 0/8 | 1.0000 |
| V1 | z / correlation | 1 | N-session | -0.016 | -0.022 | -0.014 | -0.012 | -0.020 | -0.011 | -0.011 | -0.020 | **-0.0159** | 0/8 | 1.0000 |
| V1 | z / correlation | 3 | N-global | -0.024 | -0.033 | -0.022 | -0.018 | -0.030 | -0.015 | -0.014 | -0.040 | **-0.0246** | 0/8 | 1.0000 |
| V1 | z / correlation | 3 | N-raw | -0.008 | -0.013 | -0.010 | -0.012 | -0.006 | -0.005 | -0.006 | -0.007 | **-0.0083** | 0/8 | 1.0000 |
| V1 | z / correlation | 3 | N-session | -0.025 | -0.034 | -0.026 | -0.019 | -0.033 | -0.017 | -0.018 | -0.039 | **-0.0262** | 0/8 | 1.0000 |
| V1 | z / euclidean | 1 | N-global | +0.015 | -0.007 | -0.005 | -0.009 | -0.024 | -0.007 | +0.003 | -0.031 | **-0.0081** | 2/8 | 0.9180 |
| V1 | z / euclidean | 1 | N-raw | +0.007 | -0.013 | -0.016 | -0.011 | -0.025 | -0.012 | -0.000 | -0.023 | **-0.0117** | 1/8 | 0.9922 |
| V1 | z / euclidean | 1 | N-session | +0.016 | -0.002 | -0.005 | -0.007 | -0.019 | +0.002 | +0.011 | -0.020 | **-0.0028** | 3/8 | 0.7227 |
| V1 | z / euclidean | 3 | N-global | +0.028 | -0.005 | -0.027 | -0.006 | -0.032 | -0.013 | -0.010 | -0.047 | **-0.0140** | 1/8 | 0.9414 |
| V1 | z / euclidean | 3 | N-raw | +0.016 | -0.014 | -0.040 | -0.012 | -0.034 | -0.018 | -0.016 | -0.039 | **-0.0197** | 1/8 | 0.9883 |
| V1 | z / euclidean | 3 | N-session | +0.023 | -0.001 | -0.026 | +0.002 | -0.026 | -0.008 | -0.003 | -0.035 | **-0.0094** | 2/8 | 0.9297 |
| LO-obj | raw / correlation | 1 | N-global | -0.012 | -0.010 | -0.005 | -0.009 | -0.008 | -0.006 | -0.007 | -0.003 | **-0.0074** | 0/8 | 1.0000 |
| LO-obj | raw / correlation | 1 | N-raw | +0.016 | +0.017 | +0.019 | +0.005 | -0.006 | +0.005 | +0.007 | -0.008 | **+0.0068** | 6/8 | 0.0586 |
| LO-obj | raw / correlation | 1 | N-session | -0.014 | -0.012 | -0.006 | -0.009 | -0.008 | -0.006 | -0.007 | -0.002 | **-0.0081** | 0/8 | 1.0000 |
| LO-obj | raw / correlation | 3 | N-global | -0.026 | -0.016 | -0.010 | -0.012 | -0.013 | -0.012 | -0.008 | -0.007 | **-0.0131** | 0/8 | 1.0000 |
| LO-obj | raw / correlation | 3 | N-raw | +0.031 | +0.030 | +0.025 | +0.007 | -0.007 | +0.015 | +0.017 | -0.017 | **+0.0126** | 6/8 | 0.0391 |
| LO-obj | raw / correlation | 3 | N-session | -0.027 | -0.018 | -0.013 | -0.012 | -0.015 | -0.012 | -0.008 | -0.006 | **-0.0139** | 0/8 | 1.0000 |
| LO-obj | raw / euclidean | 1 | N-global | -0.008 | -0.005 | -0.010 | -0.006 | -0.001 | -0.011 | +0.011 | -0.006 | **-0.0044** | 1/8 | 0.9453 |
| LO-obj | raw / euclidean | 1 | N-raw | -0.011 | -0.010 | -0.013 | -0.005 | -0.002 | -0.003 | +0.013 | -0.010 | **-0.0050** | 1/8 | 0.9336 |
| LO-obj | raw / euclidean | 1 | N-session | -0.014 | -0.007 | -0.013 | -0.001 | -0.006 | -0.008 | +0.016 | -0.005 | **-0.0048** | 1/8 | 0.9023 |
| LO-obj | raw / euclidean | 3 | N-global | -0.001 | -0.017 | -0.029 | -0.007 | -0.005 | -0.020 | +0.002 | -0.019 | **-0.0119** | 1/8 | 0.9922 |
| LO-obj | raw / euclidean | 3 | N-raw | -0.004 | -0.023 | -0.024 | -0.006 | -0.003 | -0.013 | +0.011 | -0.022 | **-0.0104** | 1/8 | 0.9727 |
| LO-obj | raw / euclidean | 3 | N-session | -0.006 | -0.019 | -0.036 | -0.001 | -0.005 | -0.016 | +0.012 | -0.016 | **-0.0111** | 1/8 | 0.9727 |
| LO-obj | z / correlation | 1 | N-global | -0.005 | -0.002 | -0.002 | -0.002 | -0.004 | -0.002 | -0.003 | -0.001 | **-0.0028** | 0/8 | 1.0000 |
| LO-obj | z / correlation | 1 | N-raw | -0.001 | +0.001 | +0.000 | -0.002 | -0.001 | -0.000 | -0.001 | -0.001 | **-0.0008** | 2/8 | 0.9805 |
| LO-obj | z / correlation | 1 | N-session | -0.005 | -0.002 | -0.003 | -0.003 | -0.004 | -0.002 | -0.004 | -0.001 | **-0.0029** | 0/8 | 1.0000 |
| LO-obj | z / correlation | 3 | N-global | -0.010 | -0.007 | -0.004 | -0.004 | -0.007 | -0.005 | -0.005 | -0.005 | **-0.0057** | 0/8 | 1.0000 |
| LO-obj | z / correlation | 3 | N-raw | -0.003 | +0.000 | -0.001 | -0.006 | -0.001 | -0.000 | -0.002 | -0.004 | **-0.0021** | 1/8 | 0.9961 |
| LO-obj | z / correlation | 3 | N-session | -0.010 | -0.007 | -0.004 | -0.004 | -0.008 | -0.004 | -0.006 | -0.004 | **-0.0059** | 0/8 | 1.0000 |
| LO-obj | z / euclidean | 1 | N-global | -0.003 | -0.002 | +0.003 | -0.009 | -0.001 | -0.009 | +0.019 | -0.012 | **-0.0016** | 2/8 | 0.6836 |
| LO-obj | z / euclidean | 1 | N-raw | -0.003 | -0.005 | +0.001 | -0.010 | -0.002 | -0.003 | +0.022 | -0.013 | **-0.0017** | 2/8 | 0.6641 |
| LO-obj | z / euclidean | 1 | N-session | -0.009 | -0.004 | -0.000 | -0.004 | -0.004 | -0.004 | +0.027 | -0.013 | **-0.0014** | 1/8 | 0.5938 |
| LO-obj | z / euclidean | 3 | N-global | +0.006 | +0.002 | -0.008 | -0.010 | -0.010 | -0.014 | +0.020 | -0.019 | **-0.0041** | 3/8 | 0.8164 |
| LO-obj | z / euclidean | 3 | N-raw | +0.007 | +0.001 | -0.006 | -0.012 | -0.011 | -0.008 | +0.027 | -0.018 | **-0.0025** | 3/8 | 0.6836 |
| LO-obj | z / euclidean | 3 | N-session | +0.000 | -0.004 | -0.014 | +0.002 | -0.009 | -0.005 | +0.034 | -0.024 | **-0.0024** | 3/8 | 0.6602 |
| lateral | raw / correlation | 1 | N-global | -0.014 | -0.013 | -0.006 | -0.013 | -0.008 | -0.007 | -0.007 | -0.004 | **-0.0091** | 0/8 | 1.0000 |
| lateral | raw / correlation | 1 | N-raw | +0.006 | +0.002 | +0.008 | -0.001 | -0.015 | +0.003 | +0.004 | -0.013 | **-0.0007** | 5/8 | 0.6016 |
| lateral | raw / correlation | 1 | N-session | -0.015 | -0.015 | -0.007 | -0.013 | -0.009 | -0.007 | -0.007 | -0.004 | **-0.0097** | 0/8 | 1.0000 |
| lateral | raw / correlation | 3 | N-global | -0.027 | -0.018 | -0.013 | -0.014 | -0.011 | -0.016 | -0.010 | -0.009 | **-0.0148** | 0/8 | 1.0000 |
| lateral | raw / correlation | 3 | N-raw | +0.006 | +0.005 | +0.004 | +0.002 | -0.017 | +0.007 | +0.013 | -0.022 | **-0.0003** | 6/8 | 0.5273 |
| lateral | raw / correlation | 3 | N-session | -0.028 | -0.018 | -0.015 | -0.016 | -0.013 | -0.016 | -0.010 | -0.009 | **-0.0156** | 0/8 | 1.0000 |
| lateral | raw / euclidean | 1 | N-global | -0.005 | -0.006 | -0.012 | -0.011 | -0.008 | -0.009 | +0.013 | -0.007 | **-0.0056** | 1/8 | 0.9609 |
| lateral | raw / euclidean | 1 | N-raw | -0.007 | -0.009 | -0.018 | -0.011 | -0.006 | -0.004 | +0.012 | -0.008 | **-0.0062** | 1/8 | 0.9609 |
| lateral | raw / euclidean | 1 | N-session | -0.011 | -0.005 | -0.015 | -0.006 | -0.014 | -0.007 | +0.018 | -0.007 | **-0.0059** | 1/8 | 0.9375 |
| lateral | raw / euclidean | 3 | N-global | +0.006 | -0.018 | -0.027 | -0.006 | -0.014 | -0.018 | +0.007 | -0.021 | **-0.0111** | 2/8 | 0.9766 |
| lateral | raw / euclidean | 3 | N-raw | +0.002 | -0.020 | -0.030 | -0.003 | -0.009 | -0.013 | +0.013 | -0.021 | **-0.0101** | 2/8 | 0.9609 |
| lateral | raw / euclidean | 3 | N-session | +0.000 | -0.015 | -0.031 | +0.000 | -0.017 | -0.016 | +0.016 | -0.019 | **-0.0103** | 3/8 | 0.9531 |
| lateral | z / correlation | 1 | N-global | -0.005 | -0.004 | -0.003 | -0.004 | -0.004 | -0.002 | -0.004 | -0.002 | **-0.0036** | 0/8 | 1.0000 |
| lateral | z / correlation | 1 | N-raw | -0.003 | -0.002 | -0.002 | -0.003 | -0.003 | -0.001 | -0.002 | -0.002 | **-0.0023** | 0/8 | 1.0000 |
| lateral | z / correlation | 1 | N-session | -0.005 | -0.004 | -0.003 | -0.004 | -0.005 | -0.002 | -0.004 | -0.002 | **-0.0036** | 0/8 | 1.0000 |
| lateral | z / correlation | 3 | N-global | -0.010 | -0.008 | -0.005 | -0.006 | -0.007 | -0.006 | -0.006 | -0.006 | **-0.0067** | 0/8 | 1.0000 |
| lateral | z / correlation | 3 | N-raw | -0.006 | -0.005 | -0.004 | -0.007 | -0.005 | -0.002 | -0.003 | -0.004 | **-0.0046** | 0/8 | 1.0000 |
| lateral | z / correlation | 3 | N-session | -0.009 | -0.008 | -0.005 | -0.006 | -0.008 | -0.006 | -0.007 | -0.006 | **-0.0069** | 0/8 | 1.0000 |
| lateral | z / euclidean | 1 | N-global | +0.000 | -0.001 | +0.000 | -0.014 | -0.004 | -0.007 | +0.020 | -0.012 | **-0.0022** | 3/8 | 0.7070 |
| lateral | z / euclidean | 1 | N-raw | -0.000 | -0.001 | -0.002 | -0.017 | -0.002 | -0.003 | +0.021 | -0.011 | **-0.0020** | 1/8 | 0.6875 |
| lateral | z / euclidean | 1 | N-session | -0.005 | -0.002 | -0.003 | -0.007 | -0.008 | -0.003 | +0.028 | -0.013 | **-0.0016** | 1/8 | 0.6094 |
| lateral | z / euclidean | 3 | N-global | +0.011 | +0.001 | -0.010 | -0.013 | -0.013 | -0.009 | +0.023 | -0.020 | **-0.0038** | 3/8 | 0.7461 |
| lateral | z / euclidean | 3 | N-raw | +0.009 | +0.002 | -0.012 | -0.015 | -0.007 | -0.006 | +0.029 | -0.018 | **-0.0022** | 3/8 | 0.6523 |
| lateral | z / euclidean | 3 | N-session | +0.006 | -0.001 | -0.013 | +0.001 | -0.013 | -0.003 | +0.036 | -0.024 | **-0.0014** | 3/8 | 0.5898 |

### NSD multiverse with N-raw

- distinct analyses: 28
- distinct analyses, interval above 0: 1
- distinct analyses, interval below 0: 10
- distinct analyses, interval includes 0: 17

| ROI | F | D | B | Delta | 95 % CI | n < 0 | class |
|---|---|---|---|---|---|---|---|
| V1 | raw | correlation | paper | +0.1656 | [+0.1251, +0.2034] | 0 | above 0 |
| V1 | raw | correlation | center720 | +0.0037 | [-0.0112, +0.0209] | 2 | includes 0 |
| V1 | raw | correlation | z720 | +0.0012 | [-0.0108, +0.0146] | 4 | includes 0 |
| V1 | raw | cosine | paper | +0.0206 | [-0.0016, +0.0436] | 0 | includes 0 |
| V1 | raw | cosine | center720 | -0.0077 | [-0.0165, +0.0021] | 8 | includes 0 |
| V1 | raw | cosine | z720 | -0.0077 | [-0.0163, +0.0018] | 8 | includes 0 |
| V1 | raw | euclidean | paper | -0.0098 | [-0.0322, +0.0101] | 6 | includes 0 |
| V1 | raw | euclidean | center720 | -0.0098 | [-0.0322, +0.0101] | 6 | includes 0 |
| V1 | raw | euclidean | z720 | -0.0088 | [-0.0322, +0.0122] | 6 | includes 0 |
| V1 | center | correlation | paper | -0.0088 | [-0.0243, +0.0070] | 8 | includes 0 |
| V1 | center | correlation | center720 | +0.0039 | [-0.0064, +0.0156] | 2 | includes 0 |
| V1 | center | correlation | z720 | -0.0004 | [-0.0089, +0.0090] | 5 | includes 0 |
| V1 | center | cosine | paper | +0.0002 | [-0.0184, +0.0195] | 3 | includes 0 |
| V1 | center | cosine | center720 | -0.0148 | [-0.0294, +0.0006] | 8 | includes 0 |
| V1 | center | cosine | z720 | -0.0162 | [-0.0304, -0.0012] | 8 | below 0 |
| V1 | center | euclidean | paper | -0.0098 | [-0.0322, +0.0101] | 6 | includes 0 |
| V1 | center | euclidean | center720 | -0.0098 | [-0.0322, +0.0101] | 6 | includes 0 |
| V1 | center | euclidean | z720 | -0.0088 | [-0.0322, +0.0122] | 6 | includes 0 |
| V1 | scale | correlation | paper | -0.0574 | [-0.0835, -0.0323] | 8 | below 0 |
| V1 | scale | correlation | center720 | -0.0046 | [-0.0129, +0.0039] | 6 | includes 0 |
| V1 | scale | correlation | z720 | -0.0043 | [-0.0113, +0.0029] | 7 | includes 0 |
| V1 | scale | cosine | paper | -0.0427 | [-0.0611, -0.0241] | 8 | below 0 |
| V1 | scale | cosine | center720 | -0.0092 | [-0.0162, -0.0025] | 8 | below 0 |
| V1 | scale | cosine | z720 | -0.0101 | [-0.0169, -0.0035] | 8 | below 0 |
| V1 | scale | euclidean | paper | -0.0197 | [-0.0383, -0.0008] | 7 | below 0 |
| V1 | scale | euclidean | center720 | -0.0197 | [-0.0383, -0.0008] | 7 | below 0 |
| V1 | scale | euclidean | z720 | -0.0146 | [-0.0342, +0.0054] | 7 | includes 0 |
| V1 | z | correlation | paper | -0.0083 | [-0.0243, +0.0060] | 8 | includes 0 |
| V1 | z | correlation | center720 | -0.0274 | [-0.0384, -0.0177] | 8 | below 0 |
| V1 | z | correlation | z720 | -0.0216 | [-0.0301, -0.0142] | 8 | below 0 |
| V1 | z | cosine | paper | +0.0067 | [-0.0176, +0.0301] | 0 | includes 0 |
| V1 | z | cosine | center720 | -0.0524 | [-0.0684, -0.0354] | 8 | below 0 |
| V1 | z | cosine | z720 | -0.0532 | [-0.0692, -0.0373] | 8 | below 0 |
| V1 | z | euclidean | paper | -0.0197 | [-0.0383, -0.0008] | 7 | below 0 |
| V1 | z | euclidean | center720 | -0.0197 | [-0.0383, -0.0008] | 7 | below 0 |
| V1 | z | euclidean | z720 | -0.0146 | [-0.0342, +0.0054] | 7 | includes 0 |

## 5. Spearman of g and g^M with image statistics

Brain g (mean over subjects; per-subject range in brackets):

| dataset | convention | ROI | k | L | C | CE | DD |
|---|---|---|---|---|---|---|---|
| nsd515 | N-raw | V1 | 1 | -0.189 [-0.24, -0.14] | +0.161 [+0.07, +0.22] | +0.257 [+0.18, +0.32] | +0.180 [+0.14, +0.23] |
| nsd515 | N-global | V1 | 1 | +0.188 [+0.12, +0.26] | -0.081 [-0.15, -0.01] | -0.221 [-0.31, -0.12] | -0.178 [-0.25, -0.12] |
| nsd515 | N-session | V1 | 1 | +0.218 [+0.13, +0.29] | -0.104 [-0.18, -0.02] | -0.253 [-0.36, -0.14] | -0.205 [-0.28, -0.13] |
| nsd515 | N-raw | V1 | 3 | -0.273 [-0.33, -0.22] | +0.186 [+0.08, +0.25] | +0.321 [+0.23, +0.39] | +0.259 [+0.21, +0.31] |
| nsd515 | N-global | V1 | 3 | +0.317 [+0.21, +0.39] | -0.172 [-0.30, -0.01] | -0.403 [-0.50, -0.24] | -0.301 [-0.38, -0.19] |
| nsd515 | N-session | V1 | 3 | +0.323 [+0.18, +0.40] | -0.175 [-0.31, -0.01] | -0.407 [-0.50, -0.22] | -0.304 [-0.39, -0.16] |
| nsd515 | N-raw | LO-obj | 1 | -0.020 [-0.08, +0.03] | +0.013 [-0.02, +0.05] | +0.033 [-0.02, +0.10] | +0.022 [-0.03, +0.09] |
| nsd515 | N-global | LO-obj | 1 | +0.023 [-0.02, +0.06] | +0.008 [-0.01, +0.06] | -0.011 [-0.04, +0.03] | -0.021 [-0.06, +0.03] |
| nsd515 | N-session | LO-obj | 1 | +0.032 [-0.02, +0.07] | +0.007 [-0.02, +0.06] | -0.016 [-0.05, +0.03] | -0.030 [-0.07, +0.02] |
| nsd515 | N-raw | LO-obj | 3 | -0.033 [-0.10, +0.04] | +0.016 [-0.04, +0.07] | +0.050 [-0.02, +0.13] | +0.034 [-0.04, +0.10] |
| nsd515 | N-global | LO-obj | 3 | +0.035 [-0.02, +0.09] | -0.003 [-0.05, +0.13] | -0.036 [-0.10, +0.04] | -0.037 [-0.08, +0.03] |
| nsd515 | N-session | LO-obj | 3 | +0.043 [-0.02, +0.10] | +0.003 [-0.05, +0.13] | -0.039 [-0.11, +0.04] | -0.045 [-0.10, +0.03] |
| nsd515 | N-raw | lateral | 1 | -0.000 [-0.05, +0.04] | -0.011 [-0.03, +0.00] | -0.005 [-0.04, +0.04] | +0.003 [-0.03, +0.05] |
| nsd515 | N-global | lateral | 1 | +0.024 [-0.01, +0.06] | +0.015 [-0.01, +0.07] | -0.005 [-0.04, +0.04] | -0.022 [-0.06, +0.01] |
| nsd515 | N-session | lateral | 1 | +0.033 [-0.01, +0.07] | +0.017 [-0.02, +0.08] | -0.009 [-0.03, +0.04] | -0.030 [-0.07, +0.01] |
| nsd515 | N-raw | lateral | 3 | -0.006 [-0.06, +0.05] | -0.024 [-0.06, -0.00] | -0.013 [-0.07, +0.05] | +0.008 [-0.04, +0.05] |
| nsd515 | N-global | lateral | 3 | +0.027 [-0.04, +0.07] | +0.025 [-0.03, +0.13] | +0.003 [-0.05, +0.06] | -0.028 [-0.07, +0.03] |
| nsd515 | N-session | lateral | 3 | +0.035 [-0.03, +0.08] | +0.029 [-0.02, +0.13] | -0.003 [-0.07, +0.05] | -0.036 [-0.08, +0.02] |
| things720 | T-raw | V1 | 1 | -0.194 [-0.21, -0.18] | +0.134 [+0.09, +0.18] | +0.282 [+0.26, +0.31] | +0.187 [+0.17, +0.21] |
| things720 | T-global | V1 | 1 | -0.177 [-0.20, -0.15] | +0.116 [+0.09, +0.16] | +0.249 [+0.22, +0.28] | +0.174 [+0.14, +0.20] |
| things720 | T-session | V1 | 1 | -0.021 [-0.13, +0.07] | +0.069 [+0.04, +0.10] | +0.085 [+0.04, +0.15] | +0.017 [-0.06, +0.12] |
| things720 | T-raw | LOC | 1 | +0.021 [-0.03, +0.07] | +0.000 [-0.03, +0.03] | -0.033 [-0.04, -0.02] | -0.021 [-0.07, +0.04] |
| things720 | T-global | LOC | 1 | -0.003 [-0.06, +0.04] | -0.031 [-0.06, -0.01] | -0.032 [-0.03, -0.03] | +0.004 [-0.04, +0.07] |
| things720 | T-session | LOC | 1 | +0.021 [-0.04, +0.06] | -0.026 [-0.06, +0.01] | -0.007 [-0.04, +0.03] | -0.029 [-0.07, +0.03] |
| things_test | T-raw | V1 | 12 | -0.237 [-0.26, -0.20] | +0.280 [+0.17, +0.36] | +0.469 [+0.42, +0.51] | +0.252 [+0.21, +0.28] |
| things_test | T-global | V1 | 12 | +0.154 [+0.10, +0.20] | -0.170 [-0.33, -0.02] | -0.304 [-0.46, -0.17] | -0.177 [-0.22, -0.12] |
| things_test | T-session | V1 | 12 | +0.152 [+0.10, +0.20] | -0.167 [-0.33, -0.02] | -0.304 [-0.46, -0.17] | -0.174 [-0.21, -0.13] |
| things_test | T-raw | LOC | 12 | -0.094 [-0.20, -0.03] | +0.087 [+0.03, +0.15] | -0.050 [-0.09, +0.01] | +0.069 [-0.01, +0.20] |
| things_test | T-global | LOC | 12 | +0.091 [-0.02, +0.16] | -0.064 [-0.14, -0.01] | +0.039 [-0.03, +0.09] | -0.062 [-0.17, +0.07] |
| things_test | T-session | LOC | 12 | +0.090 [-0.02, +0.16] | -0.063 [-0.14, -0.01] | +0.041 [-0.03, +0.09] | -0.061 [-0.16, +0.07] |

Model g^M (Conv1; mean over seeds 0-4, range in brackets; LOC rows use Conv3):

| dataset | layer | model | L | C | CE | DD |
|---|---|---|---|---|---|---|
| things720 | Conv1 | Random | -0.213 [-0.35, -0.02] | +0.462 [+0.44, +0.48] | +0.380 [+0.37, +0.38] | +0.198 [-0.00, +0.34] |
| things720 | Conv1 | BP | -0.019 [-0.08, +0.01] | +0.348 [+0.32, +0.36] | +0.327 [+0.31, +0.34] | +0.012 [-0.03, +0.08] |
| things720 | Conv3 | Random | -0.171 [-0.26, -0.03] | +0.443 [+0.43, +0.47] | +0.403 [+0.40, +0.41] | +0.162 [+0.01, +0.27] |
| things720 | Conv3 | BP | +0.079 [+0.03, +0.13] | +0.376 [+0.36, +0.42] | +0.297 [+0.25, +0.35] | -0.080 [-0.13, -0.03] |
| things_test | Conv1 | Random | -0.101 [-0.26, +0.09] | +0.521 [+0.51, +0.53] | +0.306 [+0.28, +0.32] | +0.098 [-0.10, +0.26] |
| things_test | Conv1 | BP | +0.070 [-0.00, +0.12] | +0.416 [+0.40, +0.43] | +0.280 [+0.26, +0.29] | -0.071 [-0.12, +0.00] |
| nsd515 | Conv1 | Random | -0.194 [-0.32, -0.04] | +0.487 [+0.47, +0.51] | +0.385 [+0.38, +0.40] | +0.185 [+0.03, +0.31] |
| nsd515 | Conv1 | BP | -0.011 [-0.04, +0.04] | +0.467 [+0.46, +0.48] | +0.438 [+0.42, +0.46] | +0.007 [-0.05, +0.04] |

### Brain g vs model g^M per convention (V1, mean over seeds)

| dataset | convention | k | subject | Random | BP |
|---|---|---|---|---|---|
| things720 | T-raw | 1 | mean | +0.149 | +0.142 (R > BP in 2/3) |
| things_test | T-raw | 12 | mean | +0.278 | +0.282 (R > BP in 2/3) |
| things720 | T-global | 1 | mean | +0.128 | +0.124 (R > BP in 1/3) |
| things_test | T-global | 12 | mean | -0.204 | -0.200 (R > BP in 1/3) |
| things720 | T-session | 1 | mean | +0.041 | +0.054 (R > BP in 2/3) |
| things_test | T-session | 12 | mean | -0.206 | -0.201 (R > BP in 1/3) |
| nsd515 | N-raw | 1 | mean | +0.189 | +0.187 (R > BP in 5/8) |
| nsd515 | N-raw | 3 | mean | +0.249 | +0.243 (R > BP in 6/8) |
| nsd515 | N-global | 1 | mean | -0.108 | -0.083 (R > BP in 0/8) |
| nsd515 | N-global | 3 | mean | -0.210 | -0.181 (R > BP in 0/8) |
| nsd515 | N-session | 1 | mean | -0.142 | -0.110 (R > BP in 0/8) |
| nsd515 | N-session | 3 | mean | -0.214 | -0.183 (R > BP in 0/8) |


g: cosine of each brain pattern with the mean of the analysed patterns (NSD k = 1: one g per presentation,
Spearman averaged over presentations). g^M: cosine of each raw GAP feature vector with the mean feature
vector. Full rows per subject and seed: `results/step48/amplitude.csv`.

## 6. Runtimes

| Part | Estimate (× 1.5) | Actual |
|---|---|---|
| THINGS conventions | < 0.1 h | 40 s |
| THINGS amplitude incl. image statistics | < 0.1 h | 56 s (image statistics cached from setup) |
| NSD reload (240 sessions) | 3.5 h | 8073 s = 2.24 h |
| NSD conventions RSA | 0.4 h (0.51 h re-estimated) | 390 s |
| NSD multiverse N-raw | 0.4 h | 931 s |
| NSD amplitude | < 0.1 h | 48 s |

## 7. Open points

1. **Gap in the interpretation rule.** The rule of the amendment has no case for "NB1 and NB2 met, G2 or
   G3 not met". This pattern occurred (G2 1/3, G3 5/8). As instructed, the result is reported literally as
   "no statement of the rule applies", without an intermediate wording. The amendment and
   `config_step48.yaml` were not changed; the config's fallback line still carries the parenthesis
   "(NB1 and NB2 met, G2 or G3 not met)", which is not used in the results.
2. **Fallback text in `scripts/step48_amplitude.py`** was changed after the scripts commit 830f471 (before
   the summary was run): from "no statement of the rule applies (NB1 and NB2 met, G2 or G3 not met)" to
   exactly "no statement of the rule applies". No computation changed; the change is in this commit.
3. **Unit of the THINGS h5 values** is not documented locally or found on the web: stored GLMsingle betas,
   not normalised (per-voxel means +0.008 to +0.021, SD ≈ 0.07). T-raw uses them as stored.
4. **OFF / ON reference:** with the image's own mean OFF = ON identically (DD = 0.5); the reference is the
   mean grey over all images of the dataset. DD is then close to a mirror of L (Spearman −0.98 to −0.99),
   so DD carries little information beyond L.
5. **Gabor bank:** the existing control bank has 4 × 3 filters; it was extended to 4 × 4 by one octave
   below (0.05 cycles / px), kernels 41 × 41, zero-mean, reflect padding.
6. Under N-global and N-session, and under T-global / T-session on the THINGS test images, ρ(g, g^M) is
   negative for both models (brain g rises with L, model g^M falls with L), while under N-raw and T-raw it
   is positive. The sign of the brain gain–image statistic relation therefore depends on the
   normalisation; G2 and G3 are defined on single conventions only.
7. Step 47b multiverse with N-raw instead of run-centred: of 28 distinct analyses 1 has an interval above
   0 (the paper cell), 10 below 0, 17 include 0.

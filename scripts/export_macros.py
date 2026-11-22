"""
export_macros.py
================
Write paper/numbers.tex (one LaTeX macro per number used in the paper) and
paper/numbers_map.csv (macro, source, filter, column, value), reading ONLY the CSVs in
results/ (steps 2-8).

Each spec is (macro, source CSV, pandas query, column, format):
  format "rsa"   RSA values, ridge r and CIs, 3 decimals
         "corr"  correlations (Spearman / Pearson), 2 decimals
         "int"   integers
         "sci"   numerical-check magnitudes (e.g. relative errors), 1-digit scientific
         "sci2"  the same with 2 significant digits
         "sd"    the pilot seed SD, 4 decimals (as reported in PILOT_REPORT.md)
A value spec fails unless its query matches exactly one row. A "count" spec counts the
rows its query matches (used for cell counts); it fails if the source is missing or the
query matches no row at all. A "mean3" spec averages a column over exactly three rows
(one per subject; fails otherwise). A "shufflewin" spec counts subjects in which, at every
listed g, the intact grid beats its per-image cell shuffle for both Random Weights s0 and
Backprop s0 (intact from prereg/primary.csv, shuffled from
prereg/instruments_cell_shuffle.csv; fails unless every needed row exists exactly once).
A "config" spec reads one value of the frozen config.yaml by dotted key (used only for
the reference effect's SEM, which no CSV carries). A "failedrules" spec counts the
distinct pre-stated predictions with a rule row (kind == 'rule') whose passed is False,
over all CSVs in results/. A "max" spec takes the maximum of a column over the rows its
query matches (fails if none match). A "count0" spec is a "count" that allows zero rows.
A "mean5" spec averages a column over exactly 5 rows; a "mean15" spec over exactly 15 rows
from 3 subjects (5 seeds each). A "seeddiff" spec reads
prereg/primary.csv, pairs Random Weights and Backprop by seed (exactly 5 pairs) and
returns the mean, SD (ddof = 1) or number of positive seeds ("mean", "sd", "pos" as the
column) of the paired difference in median r.
Macro names contain letters only.

Run:  python scripts/export_macros.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
PAPER = REPO / "paper"

RB = "`quantity` == 'Random Weights - Backprop'"
RN = "`quantity` == 'Random Weights - Random Weights (BN-calibrated)'"
G224 = "quantity == 'gap V1 NATIVE 224: Random - BP, Conv1 -> V1'"
G32 = "quantity == 'gap V1 NATIVE 32: Random - BP, Conv1 -> V1'"
GUP = "quantity == 'gap V1 UPSAMPLED 224: Random - BP, Conv1 -> V1'"
L224 = "quantity == 'gap LOC NATIVE 224: BP - Random, Conv3 -> LOC'"
SV1 = "quantity == 'S V1 = gap NAT224 - gap NAT32'"


def ci3(prefix, src, q):
    """value, lower and upper CI bound of one row."""
    return [(prefix, src, q, "value", "rsa"), (prefix + "Lo", src, q, "ci_lo", "rsa"),
            (prefix + "Hi", src, q, "ci_hi", "rsa")]


SPECS = [
    # step 2/3: the sign flip
    ("RsaGapReliableGOne", "step3/ridge_vs_rsa.csv", f"kind == 'difference' and metric == 'rsa_501_g1' and {RB}", "mean", "rsa"),
    ("RidgeGapGOne", "step3/ridge_vs_rsa.csv", f"kind == 'difference' and metric == 'ridge_g1' and {RB}", "mean", "rsa"),
    ("RidgeGapGFour", "step3/ridge_vs_rsa.csv", f"kind == 'difference' and metric == 'ridge_g4' and {RB}", "mean", "rsa"),
    ("RidgeGapGEight", "step3/ridge_vs_rsa.csv", f"kind == 'difference' and metric == 'ridge_g8' and {RB}", "mean", "rsa"),
    ("POneRidgeRandBncal", "step3/ridge_vs_rsa.csv", "kind == 'P1'", "mean", "rsa"),
    ("POneRsaRandBncal", "step3/ridge_vs_rsa.csv", f"kind == 'difference' and metric == 'rsa_501_g1' and {RN}", "mean", "rsa"),
    ("ETwoPerFoldGap", "step3/collapse_rpath_g1.csv", f"kind == 'difference' and metric == 'r_perfold' and {RB}", "mean", "rsa"),
    ("ETwoHiFracDiff", "step3/collapse_rpath_g1.csv", f"kind == 'difference' and metric == 'hi_frac' and {RB}", "mean", "rsa"),
    ("EThreeRunFoldGap", "step3/runfolds.csv", f"kind == 'difference' and grid == 1 and metric == 'r_pooled' and {RB}", "mean", "rsa"),
    ("EFourEncGapRandom", "step3/encoding_rsa.csv", "kind == 'difference' and fold_scheme == 'random' and centring == 'centred'", "rho", "rsa"),
    ("EFourEncGapRun", "step3/encoding_rsa.csv", "kind == 'difference' and fold_scheme == 'run' and centring == 'centred'", "rho", "rsa"),
    ("EFourEncPosSeedsRandom", "step3/encoding_rsa.csv",
     "kind == 'difference' and fold_scheme == 'random' and centring == 'centred'", "seeds_positive", "int"),
    # step 4
    ("SOneStdGap", "step4/std_rsa.csv", f"kind == 'difference' and variant == 'z' and {RB}", "mean", "rsa"),
    ("SOneStdRandBncal", "step4/std_rsa.csv", f"kind == 'difference' and variant == 'z' and {RN}", "mean", "rsa"),
    ("STwoSubTwoRsa", "step4/subjects.csv", f"kind == 'difference' and subject == 'sub-02' and metric == 'rsa_raw' and {RB}", "mean", "rsa"),
    ("STwoSubTwoRidge", "step4/subjects.csv", f"kind == 'difference' and subject == 'sub-02' and metric == 'ridge' and {RB}", "mean", "rsa"),
    ("STwoSubThreeRsa", "step4/subjects.csv", f"kind == 'difference' and subject == 'sub-03' and metric == 'rsa_raw' and {RB}", "mean", "rsa"),
    ("STwoSubThreeRidge", "step4/subjects.csv", f"kind == 'difference' and subject == 'sub-03' and metric == 'ridge' and {RB}", "mean", "rsa"),
    # step 5
    *ci3("VoneGapRawTwoTwoFour", "step5/paper_comparisons_raw_vs_z.csv", f"kind == 'comparison' and variant == 'raw' and {G224}"),
    *ci3("VoneGapRawThreeTwo", "step5/paper_comparisons_raw_vs_z.csv", f"kind == 'comparison' and variant == 'raw' and {G32}"),
    *ci3("VoneGapZTwoTwoFour", "step5/paper_comparisons_raw_vs_z.csv", f"kind == 'comparison' and variant == 'z' and {G224}"),
    *ci3("VoneGapZThreeTwo", "step5/paper_comparisons_raw_vs_z.csv", f"kind == 'comparison' and variant == 'z' and {G32}"),
    *ci3("VoneGapRawUp", "step5/paper_comparisons_raw_vs_z.csv", f"kind == 'comparison' and variant == 'raw' and {GUP}"),
    *ci3("VoneGapZUp", "step5/paper_comparisons_raw_vs_z.csv", f"kind == 'comparison' and variant == 'z' and {GUP}"),
    *ci3("SlopeRaw", "step5/paper_comparisons_raw_vs_z.csv", f"kind == 'comparison' and variant == 'raw' and {SV1}"),
    *ci3("SlopeZ", "step5/paper_comparisons_raw_vs_z.csv", f"kind == 'comparison' and variant == 'z' and {SV1}"),
    *ci3("LocGapRaw", "step5/paper_comparisons_raw_vs_z.csv", f"kind == 'comparison' and variant == 'raw' and {L224}"),
    *ci3("LocGapZ", "step5/paper_comparisons_raw_vs_z.csv", f"kind == 'comparison' and variant == 'z' and {L224}"),
    ("MOneSpearman", "step5/paper_comparisons_raw_vs_z.csv", "kind == 'rule' and quantity.str.startswith('M1 Spearman')", "value", "corr"),
    # step 6
    ("PRidgeGapThreeTwo", "step6/ridge_resolution.csv", f"kind == 'difference' and subject == 'mean sub-01/02' and arm == 'NATIVE' and res == 32 and {RB}", "mean", "rsa"),
    ("PRidgeGapTwoTwoFour", "step6/ridge_resolution.csv", f"kind == 'difference' and subject == 'mean sub-01/02' and arm == 'NATIVE' and res == 224 and {RB}", "mean", "rsa"),
    ("PRidgeSlope", "step6/ridge_resolution.csv", "kind == 'P_ridge'", "mean", "rsa"),
    ("PMechOneRandRange", "step6/mechanism_bandpass.csv", "kind == 'rule' and quantity == 'P_mech1 Random range < 0.5 x BP range'", "value", "rsa"),
    ("PMechOneBpSeeds", "step6/mechanism_bandpass.csv", "kind == 'rule' and quantity == 'P_mech1 BP m(UP224) < m(NAT224) < m(NAT32), seeds'", "value", "int"),
    ("PMechTwoNotchSeeds", "step6/mechanism_bandpass.csv", "kind == 'rule' and quantity == 'P_mech2 m_BP(notch) > m_BP(only), seeds'", "value", "int"),
    ("PMechTwoSpearman", "step6/mechanism_bandpass.csv", "kind == 'rule' and quantity == 'P_mech2 Spearman(m, d), Random+BP x 5 conditions'", "value", "corr"),
    *ci3("RobustEuclidZTwoTwoFour", "step6/robustness.csv", "roi == 'V1' and res == 224 and feature_norm == 'z' and distance == 'euclidean' and fmri_zscore == 'all-trials'"),
    *ci3("RobustFmriRawTwoTwoFour", "step6/robustness.csv", "roi == 'V1' and res == 224 and feature_norm == 'raw' and distance == 'correlation' and fmri_zscore == '720-stimuli'"),
    # step 7
    *ci3("MultiSlopePaper", "step7/multiverse.csv", "kind == 'cell' and F == 'raw' and D == 'correlation' and B == 'paper' and quantity == 'S = gap NAT224 - gap NAT32'"),
    *ci3("MultiEuclidGap", "step7/multiverse.csv", "kind == 'cell' and F == 'raw' and D == 'euclidean' and B == 'paper' and roi == 'V1' and quantity == 'gap NATIVE 224: Random Weights - Backprop'"),
    ("MultiOTwoViolations", "step7/multiverse.csv", "kind == 'rule' and quantity.str.startswith('O2')", "value", "int"),
    ("MultiLocPositive", "count:step7/multiverse.csv", "kind == 'cell' and roi == 'LOC' and value > 0", None, "int"),
    ("MultiLocCiPositive", "count:step7/multiverse.csv", "kind == 'cell' and roi == 'LOC' and ci_lo > 0", None, "int"),
    ("MultiPosCiCells", "count:step7/multiverse_sign_table.csv", "value > 0 and ci_excludes_0", None, "int"),
    ("MultiNegCiCells", "count:step7/multiverse_sign_table.csv", "value < 0 and ci_excludes_0", None, "int"),
    ("GOneSpearman", "step7/gain.csv", "kind == 'rule' and quantity.str.startswith('G1')", "value", "corr"),
    *ci3("GTwoPartialGap", "step7/partial_rsa.csv", "kind == 'partial' and roi == 'V1' and cond == 'NATIVE_224'"),
    *ci3("GTwoPartialSlope", "step7/partial_rsa.csv", "kind == 'partial' and cond == 'slope'"),
    # step 8
    ("GenOneSpearman", "step8/generalization.csv", "kind == 'rule' and quantity.str.startswith('GEN1')", "value", "corr"),
    ("GenTwoModels", "step8/generalization.csv", "kind == 'rule' and quantity.str.startswith('GEN2')", "value", "int"),
    ("GenThreeModels", "step8/generalization.csv", "kind == 'rule' and quantity.str.startswith('GEN3')", "value", "int"),
    ("GenFourQualified", "step8/generalization.csv", "kind == 'rule' and quantity.str.startswith('GEN4')", "value", "int"),
]


# ── step 8 per model and step 9 contrasts / slopes, generated from name tables ──
MODEL_NAMES = {"alexnet": "Alexnet", "vgg11_bn": "Vggeleven", "resnet18": "Resneteighteen",
               "resnet50": "Resnetfifty", "convnext_tiny": "Convnexttiny", "vit_b_16": "Vitbsixteen"}
CELL_NAMES = {("raw", "correlation", "paper"): "RawCorr", ("z", "correlation", "paper"): "ZCorr",
              ("raw", "euclidean", "paper"): "RawEucl", ("center", "correlation", "center720"): "CentCorr"}
for _m, _mn in MODEL_NAMES.items():
    for (_f, _d, _b), _cn in CELL_NAMES.items():
        _base = f"kind == 'boot' and F == '{_f}' and D == '{_d}' and B == '{_b}' and model == '{_m}'"
        SPECS += ci3(f"Gen{_mn}Vone{_cn}", "step8/generalization.csv",
                     f"{_base} and roi == 'V1' and R == '224' and quantity == 'untrained - trained'")
        SPECS += ci3(f"Gen{_mn}Loc{_cn}", "step8/generalization.csv",
                     f"{_base} and roi == 'LOC' and R == '224' and quantity == 'trained - untrained'")
        if _m != "vit_b_16" and _cn in ("RawCorr", "ZCorr"):
            SPECS += ci3(f"Gen{_mn}Slope{_cn}", "step8/generalization.csv",
                         f"{_base} and quantity == 'S_gen'")

SEP_MODELS = {"own CNN": "Own", **MODEL_NAMES}
for _m, _mn in SEP_MODELS.items():
    for _f, _fn in (("center", "Center"), ("z", "Zscore")):
        for _k, _kn in (("norm", "Norm"), ("angle", "Angle")):
            for _roi, _rn in (("V1", "Vone"), ("LOC", "Loc")):
                SPECS += ci3(f"Sep{_mn}{_rn}{_kn}{_fn}", "step9/separation.csv",
                             f"kind == 'contrast' and model == '{_m}' and roi == '{_roi}' and F == '{_f}' and component == '{_k}'")
            if _m != "vit_b_16":
                SPECS += ci3(f"SepSlope{_mn}{_kn}{_fn}", "step9/separation_slopes.csv",
                             f"kind == 'slope' and model == '{_m}' and F == '{_f}' and component == '{_k}'")
SPECS += [
    ("SepOneMaxRelError", "step9/separation.csv", "kind == 'rule' and quantity == 'SEP1 identity max relative error'", "value", "sci"),
    ("SepTwoModels", "step9/separation.csv", "kind == 'rule' and quantity.str.startswith('SEP2')", "value", "int"),
    ("SepThreeModels", "step9/separation.csv", "kind == 'rule' and quantity.str.startswith('SEP3')", "value", "int"),
    ("SepFourModels", "step9/separation.csv", "kind == 'rule' and quantity.str.startswith('SEP4')", "value", "int"),
    ("SepCheckEMaxDev", "step9/separation.csv", "kind == 'check_e'", "value", "sci"),
    ("SepSlopeAngleCiPositive", "count:step9/separation_slopes.csv",
     "kind == 'slope' and F == 'center' and component == 'angle' and model != 'own CNN' and ci_lo > 0", None, "int"),
]

# ── preregistered primary analysis (results/prereg, results/prereg_gate) ──────
GRID_NAMES = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}
CONTRAST_NAMES = {"Random Weights": "RandBp", "Random Weights (BN-calibrated)": "BncalBp",
                  "Feedback Alignment": "FaBp", "Predictive Coding": "PcBp", "STDP": "StdpBp"}
RULE_NAMES = {"Random Weights": "Rand", "Backprop": "Bp", "Random Weights (BN-calibrated)": "Bncal",
              "Feedback Alignment": "Fa", "Predictive Coding": "Pc", "STDP": "Stdp"}
SUBJ_NAMES = {"sub-01": "SubOne", "sub-02": "SubTwo", "sub-03": "SubThree"}
for _g, _gn in GRID_NAMES.items():
    for _r, _cn in CONTRAST_NAMES.items():
        _q = f"kind == 'contrast' and grid == {_g} and quantity == '{_r} - Backprop'"
        SPECS += [(f"Prereg{_cn}G{_gn}", "prereg/signflip.csv", _q, "mean", "rsa"),
                  (f"Prereg{_cn}G{_gn}Sd", "prereg/signflip.csv", _q, "sd", "rsa"),
                  (f"Prereg{_cn}G{_gn}Pos", "prereg/signflip.csv", _q, "seeds_positive", "int"),
                  (f"Prereg{_cn}G{_gn}P", "prereg/signflip.csv", _q, "p_one_sided", "rsa"),
                  (f"Prereg{_cn}G{_gn}Dmin", "prereg/signflip.csv", _q, "delta_min", "rsa")]
    for _r, _rn in RULE_NAMES.items():
        SPECS.append((f"PreregCond{_rn}G{_gn}", "prereg/signflip.csv",
                      f"kind == 'condition' and grid == {_g} and quantity == '{_r}'", "mean", "rsa"))
    for _s, _sn in SUBJ_NAMES.items():
        for _c in ("pixels", "lumcontrast", "gabor", "permuted"):
            SPECS.append((f"PreregCtl{_c.capitalize()}{_sn}G{_gn}", "prereg/instruments_control_band.csv",
                          f"subject == '{_s}' and grid == {_g} and label == 'control:{_c}'", "r_med_reliable", "rsa"))
        if _g > 1:
            for _r, _rn in (("Random Weights", "Rand"), ("Backprop", "Bp")):
                SPECS.append((f"PreregShuf{_rn}{_sn}G{_gn}", "prereg/instruments_cell_shuffle.csv",
                              f"subject == '{_s}' and grid == {_g} and label == '{_r} s0 CELL-SHUFFLED'",
                              "r_med_reliable", "rsa"))
SPECS += [
    ("PreregReferenceEffect", "prereg/signflip.csv", "kind == 'verdict'", "reference_effect", "rsa"),
    ("PreregLowEdgeHits", "count:prereg/lower_edge.csv", "classification == classification", None, "int"),
    ("PreregLowEdgePlateau", "count:prereg/lower_edge.csv", "classification == 'Plateau'", None, "int"),
    ("GtwoGateMaxDelta", "prereg_gate/g4_equivalence.csv", "kind == 'gate'", "delta", "sci2"),
    ("GtwoGateContrastDelta", "prereg_gate/g4_equivalence.csv", "kind == 'contrast'", "delta", "sci"),
    # preregistered Delta_min (pilot SD; identical for every g, taken from the g = 1 row)
    ("PreregPilotSd", "prereg/delta_min_prereg.csv", "g == 1", "sd_value", "sd"),
    ("PreregDminSim", "prereg/delta_min_prereg.csv", "g == 1", "delta_min_sim", "rsa"),
    ("PreregDminClosed", "prereg/delta_min_prereg.csv", "g == 1", "delta_min_closed", "rsa"),
    ("PreregDminParam", "prereg/delta_min_prereg.csv", "g == 1", "delta_min_param", "rsa"),
    ("PreregShuffleWinSubjects", "shufflewin:prereg/instruments_cell_shuffle.csv", "(4, 8)", None, "int"),
    # reliable V1 voxels per subject (split-half > 0.2), as used by the preregistered fits
    ("NReliableSubOne", "prereg/primary.csv", "subject == 'sub-01' and grid == 1 and label == 'Backprop s0'",
     "n_reliable", "int"),
    ("NReliableSubTwo", "prereg/primary.csv", "subject == 'sub-02' and grid == 1 and label == 'Backprop s0'",
     "n_reliable", "int"),
    ("NReliableSubThree", "prereg/primary.csv", "subject == 'sub-03' and grid == 1 and label == 'Backprop s0'",
     "n_reliable", "int"),
    # Rec. 709 (control band) vs Rec. 601 (stimulus statistics) image-mean luminance, 720 stimuli at 224 px
    ("LumDefPearson", "checks/luminance_defs.csv", "quantity == 'pearson'", "value", "rsa"),
    ("LumDefSpearman", "checks/luminance_defs.csv", "quantity == 'spearman'", "value", "rsa"),
    # reference effect SEM (paper v1) from the frozen config; number of failed pre-stated predictions
    ("PreregReferenceSem", "config:config.yaml", "provenance.reference_effect.sem_at_224px", None, "rsa"),
        # all pre-stated predictions not met (every step); checked against the rows of Tab. tab:failed
    # secondary pre-specified contrasts (BN-cal, FA, PC, STDP minus Backprop), all g: largest mean and seeds positive
    ("PreregSecondaryMaxMean", "max:prereg/signflip.csv", "kind == 'contrast' and role == 'secondary'", "mean", "rsa"),
    ("PreregSecondaryMaxPos", "max:prereg/signflip.csv", "kind == 'contrast' and role == 'secondary'",
     "seeds_positive", "int"),
    # step 7 multiverse: O1 check and sign counts of the V1 gap at NATIVE 224 over the 36 cells
    ("MultiOOneMaxDelta", "step7/multiverse.csv", "kind == 'rule' and quantity.str.startswith('O1')", "value", "sci"),
    ("MultiSignPositive", "count:step7/multiverse_sign_table.csv", "sign == '+'", None, "int"),
    ("MultiSignNegative", "count:step7/multiverse_sign_table.csv", "sign == '-'", None, "int"),
]
_SC = "kind == 'stimulus_corr' and res == 224 and "
for _s, _sn in (("sub-01", "SubOne"), ("sub-02", "SubTwo"), ("sub-03", "SubThree")):
    SPECS += [(f"VoxGainLum{_sn}", "step7/gain.csv", _SC + f"what == 'voxel gain V1 {_s}' and stat == 'mean luminance'",
               "pearson", "corr"),
              (f"VoxGainCon{_sn}", "step7/gain.csv", _SC + f"what == 'voxel gain V1 {_s}' and stat == 'RMS contrast'",
               "pearson", "corr")]
for _r, _rn in (("Random Weights", "Rand"), ("Backprop", "Bp")):
    SPECS += [(f"ModelGain{_rn}Lum", "step7/gain.csv", _SC + f"what == 'model gain Conv1 {_r}' and stat == 'mean luminance'",
               "pearson", "corr"),
              (f"ModelGain{_rn}Con", "step7/gain.csv", _SC + f"what == 'model gain Conv1 {_r}' and stat == 'RMS contrast'",
               "pearson", "corr")]
# ── appendices ───────────────────────────────────────────────────────────────
# lower-edge gate per analysis (hits, plateaus, excluded cells, largest relative gain)
for _tag, _f, _all, _plat, _exc in (
        ("Pilot", "step2/lower_edge_diagnosis.csv", "relative_gain == relative_gain", "relative_gain < 0.001", "relative_gain >= 0.001"),
        ("Etwo", "step3/lower_edge_gate_e2.csv", "classification == classification", "classification == 'Plateau'", "classification != 'Plateau'"),
        ("Ethree", "step3/lower_edge_gate_e3.csv", "classification == classification", "classification == 'Plateau'", "classification != 'Plateau'"),
        ("Efour", "step3/lower_edge_gate_e4.csv", "classification == classification", "classification == 'Plateau'", "classification != 'Plateau'"),
        ("Subjects", "step4/lower_edge_gate_subjects.csv", "classification == classification", "classification == 'Plateau'", "classification != 'Plateau'"),
        ("SubThree", "step5/sub03_diagnosis.csv", "classification == classification", "classification == 'Plateau'", "classification != 'Plateau' and classification == classification"),
        ("Ridge", "step6/lower_edge_gate_ridge.csv", "classification == classification", "classification == 'Plateau'", "classification != 'Plateau'"),
        ("Primary", "prereg/lower_edge.csv", "classification == classification", "classification == 'Plateau'", "classification != 'Plateau'")):
    SPECS += [(f"LowEdge{_tag}Hits", f"count:{_f}", _all, None, "int"),
              (f"LowEdge{_tag}Plateau", f"count:{_f}", _plat, None, "int"),
              (f"LowEdge{_tag}Excluded", f"count0:{_f}", _exc, None, "int"),
              (f"LowEdge{_tag}MaxGain", f"max:{_f}", _all, "relative_gain", "sci2")]
# run-grouped folds (sub-01): Random - BP, mean / SD over 5 seeds, seeds positive
for _g, _gn in ((1, "One"), (4, "Four"), (8, "Eight")):
    for _m, _mn in (("r_pooled", "Pooled"), ("r_perfold", "Perfold"), ("r_pooled_randomfolds", "PooledRand")):
        _q = f"kind == 'difference' and grid == {_g} and metric == '{_m}'"
        SPECS += [(f"RunFold{_mn}G{_gn}Mean", "step3/runfolds.csv", _q, "mean", "rsa"),
                  (f"RunFold{_mn}G{_gn}Sd", "step3/runfolds.csv", _q, "sd", "rsa"),
                  (f"RunFold{_mn}G{_gn}Pos", "step3/runfolds.csv", _q, "seeds_positive", "int")]
    SPECS.append((f"RunFoldLevelG{_gn}", "step3/runfolds.csv", f"kind == 'level_change' and grid == {_g}", "mean", "rsa"))
for _fs, _fn in (("random", "Random"), ("run", "Run")):
    for _cs, _cn in (("centred", "Centred"), ("uncentred", "Uncentred")):
        _q = f"kind == 'difference' and fold_scheme == '{_fs}' and centring == '{_cs}'"
        SPECS += [(f"EncRsa{_fn}{_cn}Mean", "step3/encoding_rsa.csv", _q, "rho", "rsa"),
                  (f"EncRsa{_fn}{_cn}Sd", "step3/encoding_rsa.csv", _q, "sd", "rsa"),
                  (f"EncRsa{_fn}{_cn}Pos", "step3/encoding_rsa.csv", _q, "seeds_positive", "int")]
# band-pass inputs (NATIVE 224): gaps with CI, moment ratio m
for _c, _cn in (("only_0-14", "Only"), ("notch_0-14", "Notch")):
    for _roi, _rn in (("V1", "Vone"), ("LOC", "Loc")):
        for _v, _vn in (("raw", "Raw"), ("z", "Z")):
            _q = f"kind == 'gap' and cond == '{_c}' and roi == '{_roi}' and variant == '{_v}'"
            SPECS += [(f"Band{_cn}{_rn}{_vn}", "step6/mechanism_bandpass.csv", _q, "value", "rsa"),
                      (f"Band{_cn}{_rn}{_vn}Lo", "step6/mechanism_bandpass.csv", _q, "ci_lo", "rsa"),
                      (f"Band{_cn}{_rn}{_vn}Hi", "step6/mechanism_bandpass.csv", _q, "ci_hi", "rsa")]
for _c, _cn in (("only_0-14", "Only"), ("notch_0-14", "Notch"), ("NATIVE_224", "Native")):
    for _r, _rn in (("Random Weights", "Rand"), ("Backprop", "Bp")):
        SPECS.append((f"BandM{_cn}{_rn}", "step6/moment_ratio_bandpass.csv",
                      f"kind == 'condition_summary' and cond == '{_c}' and rule == '{_r}'", "m_mean", "rsa"))
# per-subject results of the primary analysis
for _s, _sn in SUBJ_NAMES.items():
    for _g, _gn in GRID_NAMES.items():
        _q = f"subject == '{_s}' and grid == {_g}"
        SPECS += [(f"PerSubDiff{_sn}G{_gn}Mean", "seeddiff:prereg/primary.csv", _q, "mean", "rsa"),
                  (f"PerSubDiff{_sn}G{_gn}Sd", "seeddiff:prereg/primary.csv", _q, "sd", "rsa"),
                  (f"PerSubDiff{_sn}G{_gn}Pos", "seeddiff:prereg/primary.csv", _q, "pos", "int"),
                  (f"PerSubBp{_sn}G{_gn}", "mean5:prereg/primary.csv", _q + " and rule == 'Backprop'", "r_med_reliable", "rsa")]
# ── step 25 (review controls) ───────────────────────────────────────────────
_IMN = {"alexnet": "Alexnet", "vgg11_bn": "Vggeleven", "resnet18": "Resneteighteen", "resnet50": "Resnetfifty",
        "convnext_tiny": "Convnexttiny", "vit_b_16": "Vitbsixteen"}
_MP = "step25/mean_profile.csv"


def _ci(name, src, q, how="rsa"):
    return [(name, src, q, "value", how), (name + "Lo", src, q, "ci_lo", how), (name + "Hi", src, q, "ci_hi", how)]


for _v, _vn in (("full", "Full"), ("ortho", "Ortho"), ("par", "Par")):
    for _r, _rn in (("NATIVE_32", "NatThreeTwo"), ("NATIVE_224", "NatTwoTwoFour"), ("UPSAMPLED_224", "UpTwoTwoFour")):
        _q = f"kind == 'contrast' and model == 'own CNN' and roi == 'V1' and variant == '{_v}' and res == '{_r}'"
        SPECS += _ci(f"Mp{_vn}OwnVone{_rn}", _MP, _q)
        SPECS.append((f"Mp{_vn}OwnVone{_rn}Pos", _MP, _q, "seeds_positive", "int"))
    SPECS += _ci(f"Mp{_vn}OwnLoc", _MP, f"kind == 'contrast' and model == 'own CNN' and roi == 'LOC' and variant == '{_v}'")
    SPECS += _ci(f"Mp{_vn}OwnSlope", _MP, f"kind == 'slope' and model == 'own CNN' and variant == '{_v}'")
    for _m, _mn in _IMN.items():
        for _R, _Rn in (("R224", "RTwoTwoFour"), ("R448", "RFourFourEight")):
            if _m == "vit_b_16" and _R == "R448":
                continue
            SPECS += _ci(f"Mp{_vn}{_mn}{_Rn}", _MP,
                         f"kind == 'contrast' and model == '{_m}' and variant == '{_v}' and res == '{_R}'")
        if _m != "vit_b_16":
            SPECS += _ci(f"Mp{_vn}{_mn}Slope", _MP, f"kind == 'slope' and model == '{_m}' and variant == '{_v}'")
SPECS += [("MpMFourModels", _MP, "kind == 'rule' and quantity.str.startswith('M4')", "value", "int"),
          ("MpGateMaxDelta", _MP, "kind == 'check' and quantity.str.startswith('gate')", "value", "sci2")]
_DS = "step28/distinct_summary.csv"
for _n, _q in (("MultiDistinct", "distinct analyses"), ("MultiDistinctPos", "distinct analyses, CI above 0"),
               ("MultiDistinctNeg", "distinct analyses, CI below 0"),
               ("MultiDistinctIncl", "distinct analyses, CI includes 0"),
               ("MultiDistinctEucl", "distinct analyses, euclidean cells")):
    SPECS.append((_n, _DS, f"kind == 'summary' and quantity == '{_q}'", "value", "int"))
_AB = "step28/ablation_2x2.csv"
for _mk, _mkn in (("full", "Full"), ("ortho", "Ortho")):
    for _bk, _bkn in (("full", "Full"), ("ortho", "Ortho")):
        SPECS += _ci(f"AbTwoOwn{_mkn}{_bkn}", _AB,
                     f"kind == 'cell' and model == 'own CNN' and model_side == '{_mk}' and brain_side == '{_bk}'")
        if _mk == _bk:
            continue
        for _m, _mn in _IMN.items():
            SPECS += _ci(f"AbTwo{_mn}{_mkn}{_bkn}", _AB,
                         f"kind == 'cell' and model == '{_m}' and model_side == '{_mk}' and brain_side == '{_bk}'")
        for _d, _dn in (("above", "Pos"), ("below", "Neg")):
            SPECS.append((f"AbTwoImn{_mkn}{_bkn}{_dn}", _AB,
                          f"kind == 'count' and model_side == '{_mk}' and brain_side == '{_bk}' and quantity == 'models with CI {_d} 0'",
                          "value", "int"))
SPECS += _ci("AbTwoXOne", _AB, "kind == 'contrast'")
_WA = "step25/weighted_angle.csv"
for _roi, _rn in (("V1", "Vone"), ("LOC", "Loc")):
    SPECS += _ci(f"WAngOwn{_rn}", _WA, f"kind == 'contrast' and model == 'own CNN' and roi == '{_roi}'")
    for _m, _mn in _IMN.items():
        SPECS += _ci(f"WAng{_mn}{_rn}", _WA, f"kind == 'contrast' and model == '{_m}' and roi == '{_roi}'")
SPECS.append(("WAngImnPosModels", f"count0:{_WA}",
              "kind == 'contrast' and roi == 'V1' and model != 'own CNN' and ci_lo > 0", None, "int"))
for _p, _pn in (("N-A", "NA"), ("N-W", "NW"), ("A-W", "AW")):
    SPECS.append((f"WRank{_pn}Own", _WA, f"kind == 'rankcorr' and model == 'own CNN' and quantity == 'Spearman {_p}'",
                  "value", "corr"))
    for _s, _sn in SUBJ_NAMES.items():
        SPECS.append((f"WRank{_pn}{_sn}", _WA,
                      f"kind == 'rankcorr' and model == 'fMRI' and subject == '{_s}' and quantity == 'Spearman {_p}'",
                      "value", "corr"))
for _c, _cn in (("R2 N", "N"), ("R2 A", "A"), ("R2 N+A", "NA")):
    for _r, _rn in (("Random Weights", "Rand"), ("Backprop", "Bp")):
        SPECS.append((f"WRtwo{_cn}{_rn}", _WA, f"kind == 'r2' and component == '{_c}' and quantity == '{_r}'",
                      "value", "sd"))
    SPECS += _ci(f"WRtwo{_cn}Diff", _WA, f"kind == 'r2_contrast' and component == '{_c}'", "sd")
_RS = "step25/ridge_skill.csv"
for _m, _mn in (("mse_skill", "MseSkill"), ("r_perfold", "Perfold"), ("r_pooled", "Pooled")):
    for _g, _gn in GRID_NAMES.items():
        _q = f"kind == 'contrast' and quantity == 'Random Weights - Backprop, {_m}' and grid == {_g}"
        SPECS += [(f"FSkill{_mn}G{_gn}Mean", _RS, _q, "value", "sd"),
                  (f"FSkill{_mn}G{_gn}Sd", _RS, _q, "sd", "sd"),
                  (f"FSkill{_mn}G{_gn}Pos", _RS, _q, "seeds_positive", "int")]
for _r, _rn in (("Backprop", "Bp"), ("Random Weights", "Rand")):
    for _g, _gn in ((1, "One"), (4, "Four"), (8, "Eight")):
        SPECS.append((f"FSkillAbs{_rn}G{_gn}", f"mean15:{_RS}",
                      f"kind == 'fit' and grid == {_g} and label.str.startswith('{_r} s')", "mse_skill", "sd"))
for _s, _sn in SUBJ_NAMES.items():
    SPECS.append((f"FFloor{_sn}", _RS, f"kind == 'floor' and subject == '{_s}'", "r_pooled", "rsa"))
SPECS += [("FFloorMean", f"mean3:{_RS}", "kind == 'floor'", "r_pooled", "rsa"),
          ("FSkillGateMaxDelta", _RS, "kind == 'check'", "value", "sci2"),
          ("FSkillLowEdgeHits", "count:step25/ridge_lower_edge.csv", "block == 'skill'", None, "int"),
          ("FJointLowEdgeHits", "count:step25/ridge_lower_edge.csv", "block == 'joint'", None, "int"),
          ("FPermLowEdgeHits", "count:step25/ridge_lower_edge.csv", "block == 'perm'", None, "int"),
          ("FLowEdgePlateau", "count:step25/ridge_lower_edge.csv", "classification == 'Plateau'", None, "int"),
          ("LowEdgePrimaryModelHits", "count:prereg/lower_edge.csv", "~condition.str.startswith('control')", None, "int")]
for _b, _bn in (("skill", "Skill"), ("joint", "Joint"), ("perm", "Perm")):
    _f = "step25/ridge_lower_edge.csv"
    SPECS += [(f"LowEdgeStep{_bn}Hits", f"count:{_f}", f"block == '{_b}'", None, "int"),
              (f"LowEdgeStep{_bn}Plateau", f"count:{_f}", f"block == '{_b}' and classification == 'Plateau'", None, "int"),
              (f"LowEdgeStep{_bn}Excluded", f"count0:{_f}", f"block == '{_b}' and classification != 'Plateau'", None, "int"),
              (f"LowEdgeStep{_bn}MaxGain", f"max:{_f}", f"block == '{_b}'", "relative_gain", "sci2")]
_RJ = "step25/ridge_joint.csv"
for _r, _rn in RULE_NAMES.items():
    for _g, _gn in ((1, "One"), (4, "Four")):
        _q = f"kind == 'joint_minus_bank' and quantity == '{_r}' and grid == {_g}"
        SPECS += [(f"FJoint{_rn}G{_gn}Diff", _RJ, _q, "value", "rsa"),
                  (f"FJoint{_rn}G{_gn}Joint", _RJ, _q, "joint", "rsa"),
                  (f"FJoint{_rn}G{_gn}Bank", _RJ, _q, "bank", "rsa")]
_RP = "step25/ridge_permnull.csv"
for _s, _sn in SUBJ_NAMES.items():
    for _g, _gn in ((1, "One"), (4, "Four")):
        for _r, _rn in (("Random Weights s0", "Rand"), ("Backprop s0", "Bp")):
            # kind 'null' is read as NA by pandas, hence "kind != kind" (NA) selects these rows
            _q = f"kind != kind and subject == '{_s}' and grid == {_g} and quantity == '{_r}'"
            SPECS += [(f"FPerm{_sn}{_rn}G{_gn}Mean", _RP, _q, "null_mean", "rsa"),
                      (f"FPerm{_sn}{_rn}G{_gn}Lo", _RP, _q, "null_p025", "rsa"),
                      (f"FPerm{_sn}{_rn}G{_gn}Hi", _RP, _q, "null_p975", "rsa"),
                      (f"FPerm{_sn}{_rn}G{_gn}Real", _RP, _q, "real", "rsa")]
_NULL = "kind != kind"            # kind 'null' rows of ridge_permnull.csv (read as NA)
SPECS += [("FPermPNineSevenFiveMin", f"min:{_RP}", _NULL, "null_p975", "rsa"),
          ("FPermPNineSevenFiveMax", f"max:{_RP}", _NULL, "null_p975", "rsa"),
          ("FPermRealMinMain", f"min:{_RP}", _NULL + " and subject in ('sub-01', 'sub-02')", "real", "rsa"),
          ("FPermRealMaxMain", f"max:{_RP}", _NULL + " and subject in ('sub-01', 'sub-02')", "real", "rsa"),
          ("FPermRealMinSubThree", f"min:{_RP}", _NULL + " and subject == 'sub-03'", "real", "rsa"),
          ("FPermRealMaxSubThree", f"max:{_RP}", _NULL + " and subject == 'sub-03'", "real", "rsa"),
          ("FJointMinDiff", f"min:{_RJ}", "kind == 'joint_minus_bank'", "value", "rsa"),
          ("FJointMaxDiff", f"max:{_RJ}", "kind == 'joint_minus_bank'", "value", "rsa")]
# absolute RSA at V1 in the paper cell (raw/correlation/paper, NATIVE 224): range of the six condition means
_PC = ("kind == 'rho' and F == 'raw' and D == 'correlation' and B == 'paper' and roi == 'V1' "
       "and arm == 'NATIVE' and res == 224")
SPECS += [("AbsRsaPaperMin", "grpmin:step7/multiverse.csv", _PC, "rho@rule", "rsa"),
          ("AbsRsaPaperMax", "grpmax:step7/multiverse.csv", _PC, "rho@rule", "rsa")]
# leave-one-subject-out lower bound at V1, cross-run pairs (learning-rules-rsa, tag arxiv-v4)
for _c, _cn in (("point", ""), ("ci_lo", "Lo"), ("ci_hi", "Hi")):
    SPECS.append((f"LosoBoundVone{_cn}", "lrtag:arxiv-v4:results/crossrun/step2_summary.csv",
                  "key == 'lower|cr|V1'", _c, "rsa"))
_BN = "step25/bootnorm.csv"
for (_f, _d, _b), _cn in ((("z", "correlation", "paper"), "ZCorrPaper"), (("center", "euclidean", "paper"), "CentEuclPaper"),
                          (("center", "correlation", "center720"), "CentCorrCent")):
    _q = f"kind == 'cell' and F == '{_f}' and D == '{_d}' and B == '{_b}'"
    SPECS += _ci(f"BNorm{_cn}", _BN, _q) + [(f"BNorm{_cn}FixedLo", _BN, _q, "fixed_ci_lo", "rsa"),
                                             (f"BNorm{_cn}FixedHi", _BN, _q, "fixed_ci_hi", "rsa")]
for _g, _gn in GRID_NAMES.items():
    SPECS += [(f"PreregDminG{_gn}Sim", "prereg/delta_min_prereg.csv", f"g == {_g}", "delta_min_sim", "rsa"),
              (f"PreregDminG{_gn}Closed", "prereg/delta_min_prereg.csv", f"g == {_g}", "delta_min_closed", "rsa"),
              (f"PreregDminG{_gn}Param", "prereg/delta_min_prereg.csv", f"g == {_g}", "delta_min_param", "rsa"),
              (f"PreregDminG{_gn}Sd", "prereg/delta_min_prereg.csv", f"g == {_g}", "sd_value", "sd")]
    for _c in ("lumcontrast", "pixels", "gabor", "permuted"):
        SPECS.append((f"PreregCtl{_c.capitalize()}MeanG{_gn}", "mean3:prereg/instruments_control_band.csv",
                      f"grid == {_g} and label == 'control:{_c}'", "r_med_reliable", "rsa"))


ROUNDING = {"rsa": "3 decimals", "corr": "2 decimals", "int": "integer", "sci": "1-digit scientific",
            "sci2": "2-digit scientific", "sd": "4 decimals", "s2": "2 decimals, signed", "s3": "3 decimals, signed",
            "s4": "4 decimals, signed", "f1": "1 decimal", "f2": "2 decimals", "f3": "3 decimals", "f4": "4 decimals",
            "p4": "4 decimals (p-value)", "pct": "percent, integer", "utc": "timestamp (UTC)"}


def fmt(v, how):
    if how == "utc":
        d, t = str(v).split("+")[0].split("T")
        return f"{d} {t} UTC"
    if how in ("s2", "s3", "s4"):
        nd = int(how[1])
        t = f"{float(v):.{nd}f}"
        if float(t) == 0:
            return f"{0:.{nd}f}"
        return t if t.startswith("-") else "+" + t
    if how in ("f1", "f2", "f3", "f4", "p4"):
        nd = 4 if how == "p4" else int(how[1])
        t = f"{float(v):.{nd}f}"
        return f"{0:.{nd}f}" if float(t) == 0 else t
    if how == "pct":
        return str(int(round(100 * float(v))))
    if how == "int":
        return str(int(round(float(v))))
    if how == "sci2":
        m, e = f"{float(v):.1e}".split("e")
        return "\\ensuremath{" + f"{m}" + "\\times 10^{" + f"{int(e)}" + "}}"
    if how == "sci":
        m, e = f"{float(v):.0e}".split("e")
        return "\\ensuremath{" + f"{m}" + "\\times 10^{" + f"{int(e)}" + "}}"
    if how == "sd":
        return f"{float(v):.4f}"
    return f"{float(v):.{3 if how == 'rsa' else 2}f}"


def tex_minus(text):
    """Typographic minus for negative values in text and math mode; the value itself is unchanged."""
    return "\\ensuremath{-}" + text[1:] if text.startswith("-") else text


def shuffle_wins(grids):
    """Subjects in which intact > shuffled for Random Weights s0 and Backprop s0 at every g in grids."""
    prim = pd.read_csv(RES / "prereg" / "primary.csv")
    shuf = pd.read_csv(RES / "prereg" / "instruments_cell_shuffle.csv")
    if not grids:
        raise SystemExit("shuffle_wins: empty grid list")
    n = 0
    for s in sorted(prim.subject.unique()):
        ok = True
        for g in grids:
            for lab in ("Random Weights s0", "Backprop s0"):
                a = prim[(prim.subject == s) & (prim.grid == g) & (prim.label == lab)]
                b = shuf[(shuf.subject == s) & (shuf.grid == g) & (shuf.label == lab + " CELL-SHUFFLED")]
                if len(a) != 1 or len(b) != 1:
                    raise SystemExit(f"shuffle_wins: missing row for {s} g={g} {lab}")
                ok &= bool(a.r_med_reliable.iloc[0] > b.r_med_reliable.iloc[0])
        n += int(ok)
    return n


def config_value(key):
    """A value of the frozen config.yaml, addressed by a dotted key."""
    import yaml
    node = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    for k in key.split("."):
        if not isinstance(node, dict) or k not in node:
            raise SystemExit(f"config_value: missing key {key}")
        node = node[k]
    return float(node), "config.yaml", "config value"


def failed_rules(exclude=()):
    """Number of distinct pre-stated predictions with a failed rule row in results/,
    skipping the result directories named in `exclude` (e.g. step25, added after the
    paper text was written).

    A prediction is the first token of `quantity` in rows with kind == 'rule'; it counts
    as failed if any of its rule rows has passed == False."""
    failed = set()
    for f in sorted(RES.rglob("*.csv")):
        if f.relative_to(RES).parts[0] in exclude:
            continue
        d = pd.read_csv(f, low_memory=False)
        if not {"kind", "passed", "quantity"} <= set(d.columns):
            continue
        r = d[d.kind == "rule"]
        failed |= set(r[r.passed.astype(str) == "False"].quantity.str.split().str[0])
    if not failed:
        raise SystemExit("failed_rules: no failed rule row found")
    src = "results/**/*.csv" + (f" except {', '.join(exclude)}" if exclude else "")
    return len(failed), src, f"distinct predictions with passed == False ({', '.join(sorted(failed))})"


def main():
    import tables_v2
    cache, lines, rows = {}, [], []
    names = set()
    for name, src, q, col, how in SPECS + tables_v2.build():
        if name in names:
            raise SystemExit(f"duplicate macro {name}")
        names.add(name)
        if src.startswith("ledger:"):
            led = pd.read_csv(REPO / src.split(":", 1)[1])
            hit = led.query(q, engine="python")
            val = len(hit) if col == "count" else hit.amendment_file.nunique()
            if name == "FailedPredictions":
                tab = (PAPER / "drafts" / "app_failed.tex").read_text(encoding="utf-8")
                bodies = re.findall(r"\\midrule(.*?)\\bottomrule", tab, flags=re.S)
                n_rows = sum(b.count("\\\\") for b in bodies)
                if n_rows != val:
                    raise SystemExit(f"FailedPredictions = {val}, but the tables of app_failed.tex have {n_rows} rows")
            text = fmt(val, how)
            colname = "row count" if col == "count" else "distinct amendment files"
            lines.append(f"% {name}: {src.split(':', 1)[1]} | {q} | {colname} | integer")
            lines.append(f"\\newcommand{{\\{name}}}{{{tex_minus(text)}}}")
            rows.append(dict(macro=name, source=src.split(":", 1)[1], filter=q, column=colname, rounding="integer", value=text))
            continue
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise SystemExit(f"invalid macro name {name}")
        kind = src.split(":", 1)[0] if ":" in src else "value"
        if kind in ("config", "failedrules"):
            val, source, colname = (config_value(q) if kind == "config"
                                    else failed_rules(tuple(src.split(":", 1)[1].split(",")[1:])))
            if name == "FailedPredictions":
                tab = (PAPER / "drafts" / "app_failed.tex").read_text(encoding="utf-8")
                body = tab.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
                n_rows = body.count("\\\\")
                if n_rows != val:
                    raise SystemExit(f"FailedPredictions = {val}, but Tab. tab:failed has {n_rows} rows")
            text = fmt(val, how)
            rounding = {"rsa": "3 decimals", "int": "integer"}[how]
            lines.append(f"% {name}: {source} | {q} | {colname} | {rounding}")
            lines.append(f"\\newcommand{{\\{name}}}{{{tex_minus(text)}}}")
            rows.append(dict(macro=name, source=source, filter=q, column=colname, rounding=rounding, value=text))
            continue
        if kind == "lrtag":
            # a CSV of learning-rules-rsa (CROSSRUN_REPO) read at a fixed tag: "lrtag:<tag>:<path>"
            _, tag, rel = src.split(":", 2)
            import io
            import subprocess
            sys.path.insert(0, str(REPO / "src" / "ridge"))
            import paths
            blob = subprocess.run(["git", "-C", str(paths.crossrun_repo()), "show", f"{tag}:{rel}"],
                                  capture_output=True, text=True, check=True).stdout
            hit = pd.read_csv(io.StringIO(blob)).query(q, engine="python").drop_duplicates()
            if len(hit) != 1:
                raise SystemExit(f"{name}: filter matches {len(hit)} rows, expected 1: {q}")
            text = fmt(hit.iloc[0][col], how)
            source = f"learning-rules-rsa@{tag}:{rel}"
            lines.append(f"% {name}: {source} | {q} | {col} | 3 decimals")
            lines.append(f"\\newcommand{{\\{name}}}{{{tex_minus(text)}}}")
            rows.append(dict(macro=name, source=source, filter=q, column=col, rounding="3 decimals", value=text))
            continue
        path = RES / src.split(":", 1)[-1]
        if not path.exists():
            raise SystemExit(f"missing source {path}")
        if path not in cache:
            if path == RES / "step48" / "amplitude.csv":
                cache[path] = tables_v2.amplitude48()     # seeds of the THINGS rows reconstructed as in step 48b
            else:
                cache[path] = pd.read_csv(path, low_memory=False, float_precision="round_trip")
        if kind == "shufflewin":
            val, colname = shuffle_wins([int(x) for x in re.findall(r"\d+", q)]), "subjects (derived)"
            text = fmt(val, how)
            lines.append(f"% {name}: results/prereg/primary.csv + {path.relative_to(REPO).as_posix()} | g in {q} | "
                         f"{colname} | integer")
            lines.append(f"\\newcommand{{\\{name}}}{{{tex_minus(text)}}}")
            rows.append(dict(macro=name, source=path.relative_to(REPO).as_posix(), filter=f"g in {q}",
                             column=colname, rounding="integer", value=text))
            continue
        hit = cache[path].query(q, engine="python")
        if kind == "mean3":
            if len(hit) != 3 or hit.subject.nunique() != 3:
                raise SystemExit(f"{name}: mean3 filter matches {len(hit)} rows, expected 3 subjects: {q}")
            val, colname = hit[col].mean(), f"mean of {col} over 3 subjects"
        elif kind == "seeddiff":
            a = hit[hit.rule == "Random Weights"].set_index("seed_idx")["r_med_reliable"]
            b = hit[hit.rule == "Backprop"].set_index("seed_idx")["r_med_reliable"]
            if len(a) != 5 or len(b) != 5 or set(a.index) != set(b.index):
                raise SystemExit(f"{name}: seeddiff needs 5 paired seeds, got {len(a)} and {len(b)}: {q}")
            d = a - b
            val = {"mean": d.mean(), "sd": d.std(ddof=1), "pos": int((d > 0).sum())}[col]
            colname = f"{col} over 5 seeds of Random Weights - Backprop (median r)"
        elif kind == "mean5":
            if len(hit) != 5:
                raise SystemExit(f"{name}: mean5 filter matches {len(hit)} rows, expected 5: {q}")
            val, colname = hit[col].mean(), f"mean of {col} over 5 rows"
        elif kind == "mean15":
            if len(hit) != 15 or hit.subject.nunique() != 3:
                raise SystemExit(f"{name}: mean15 filter matches {len(hit)} rows, expected 3 subjects x 5 seeds: {q}")
            val, colname = hit[col].mean(), f"mean of {col} over 3 subjects x 5 seeds"
        elif kind == "count0":
            val, colname = len(hit), "row count (zero allowed)"
        elif kind in ("max", "min"):
            if len(hit) == 0:
                raise SystemExit(f"{name}: {kind} filter matches no row: {q}")
            val = hit[col].max() if kind == "max" else hit[col].min()
            colname = f"{kind} of {col} over {len(hit)} rows"
        elif kind in ("grpmax", "grpmin"):
            vcol, gcol = col.split("@")
            if len(hit) == 0:
                raise SystemExit(f"{name}: {kind} filter matches no row: {q}")
            m = hit.groupby(gcol)[vcol].mean()
            val = m.max() if kind == "grpmax" else m.min()
            colname = f"{kind[3:]} over {gcol} of the mean of {vcol} ({len(m)} groups)"
        elif kind == "avg":
            if len(hit) == 0:
                raise SystemExit(f"{name}: avg filter matches no row: {q}")
            val, colname = hit[col].mean(), f"mean of {col} over {len(hit)} rows"
        elif kind in ("expr", "exprmin", "exprmax"):
            if len(hit) == 0 or (kind == "expr" and len(hit) != 1):
                raise SystemExit(f"{name}: {kind} filter matches {len(hit)} rows: {q}")
            v = hit.eval(col, engine="python")
            val = v.iloc[0] if kind == "expr" else (v.min() if kind == "exprmin" else v.max())
            colname = f"{kind[4:] or 'value'} of {col} over {len(hit)} rows"
        elif kind == "count":
            if len(hit) == 0:
                raise SystemExit(f"{name}: count filter matches no row: {q}")
            val, colname = len(hit), "row count"
        else:
            if len(hit) != 1:
                raise SystemExit(f"{name}: filter matches {len(hit)} rows, expected 1: {q}")
            val, colname = hit.iloc[0][col], col
        text = fmt(val, how)
        rounding = ROUNDING[how]
        lines.append(f"% {name}: {path.relative_to(REPO).as_posix()} | {q} | {colname} | {rounding}")
        lines.append(f"\\newcommand{{\\{name}}}{{{tex_minus(text)}}}")
        rows.append(dict(macro=name, source=path.relative_to(REPO).as_posix(), filter=q, column=colname,
                         rounding=rounding, value=text))
    PAPER.mkdir(exist_ok=True)
    (PAPER / "numbers.tex").write_text("% generated by scripts/export_macros.py -- do not edit\n" + "\n".join(lines) + "\n",
                                       encoding="utf-8", newline="\n")
    pd.DataFrame(rows).to_csv(PAPER / "numbers_map.csv", index=False, lineterminator="\n")
    print(f"wrote {len(rows)} macros")


if __name__ == "__main__":
    sys.exit(main())

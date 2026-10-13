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
         "sd"    the pilot seed SD, 4 decimals (as reported in PILOT_REPORT.md)
A value spec fails unless its query matches exactly one row. A "count" spec counts the
rows its query matches (used for cell counts); it fails if the source is missing or the
query matches no row at all. A "mean3" spec averages a column over exactly three rows
(one per subject; fails otherwise). A "shufflewin" spec counts subjects in which, at every
listed g, the intact grid beats its per-image cell shuffle for both Random Weights s0 and
Backprop s0 (intact from prereg/primary.csv, shuffled from
prereg/instruments_cell_shuffle.csv; fails unless every needed row exists exactly once).
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
    ("GtwoGateMaxDelta", "prereg_gate/g4_equivalence.csv", "kind == 'gate'", "delta", "sci"),
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
]
for _g, _gn in GRID_NAMES.items():
    SPECS += [(f"PreregDminG{_gn}Sim", "prereg/delta_min_prereg.csv", f"g == {_g}", "delta_min_sim", "rsa"),
              (f"PreregDminG{_gn}Closed", "prereg/delta_min_prereg.csv", f"g == {_g}", "delta_min_closed", "rsa"),
              (f"PreregDminG{_gn}Param", "prereg/delta_min_prereg.csv", f"g == {_g}", "delta_min_param", "rsa"),
              (f"PreregDminG{_gn}Sd", "prereg/delta_min_prereg.csv", f"g == {_g}", "sd_value", "sd")]
    for _c in ("lumcontrast", "pixels", "gabor"):
        SPECS.append((f"PreregCtl{_c.capitalize()}MeanG{_gn}", "mean3:prereg/instruments_control_band.csv",
                      f"grid == {_g} and label == 'control:{_c}'", "r_med_reliable", "rsa"))


def fmt(v, how):
    if how == "int":
        return str(int(round(float(v))))
    if how == "sci":
        m, e = f"{float(v):.0e}".split("e")
        return "\\ensuremath{" + f"{m}" + "\\times 10^{" + f"{int(e)}" + "}}"
    if how == "sd":
        return f"{float(v):.4f}"
    return f"{float(v):.{3 if how == 'rsa' else 2}f}"


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


def main():
    cache, lines, rows = {}, [], []
    for name, src, q, col, how in SPECS:
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise SystemExit(f"invalid macro name {name}")
        kind = src.split(":", 1)[0] if ":" in src else "value"
        path = RES / src.split(":", 1)[-1]
        if not path.exists():
            raise SystemExit(f"missing source {path}")
        if path not in cache:
            cache[path] = pd.read_csv(path, low_memory=False)
        if kind == "shufflewin":
            val, colname = shuffle_wins([int(x) for x in re.findall(r"\d+", q)]), "subjects (derived)"
            text = fmt(val, how)
            lines.append(f"% {name}: results/prereg/primary.csv + {path.relative_to(REPO).as_posix()} | g in {q} | "
                         f"{colname} | integer")
            lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
            rows.append(dict(macro=name, source=path.relative_to(REPO).as_posix(), filter=f"g in {q}",
                             column=colname, rounding="integer", value=text))
            continue
        hit = cache[path].query(q, engine="python")
        if kind == "mean3":
            if len(hit) != 3 or hit.subject.nunique() != 3:
                raise SystemExit(f"{name}: mean3 filter matches {len(hit)} rows, expected 3 subjects: {q}")
            val, colname = hit[col].mean(), f"mean of {col} over 3 subjects"
        elif kind == "count":
            if len(hit) == 0:
                raise SystemExit(f"{name}: count filter matches no row: {q}")
            val, colname = len(hit), "row count"
        else:
            if len(hit) != 1:
                raise SystemExit(f"{name}: filter matches {len(hit)} rows, expected 1: {q}")
            val, colname = hit.iloc[0][col], col
        text = fmt(val, how)
        rounding = {"rsa": "3 decimals", "corr": "2 decimals", "int": "integer", "sci": "1-digit scientific",
                    "sd": "4 decimals"}[how]
        lines.append(f"% {name}: {path.relative_to(REPO).as_posix()} | {q} | {colname} | {rounding}")
        lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
        rows.append(dict(macro=name, source=path.relative_to(REPO).as_posix(), filter=q, column=colname,
                         rounding=rounding, value=text))
    PAPER.mkdir(exist_ok=True)
    (PAPER / "numbers.tex").write_text("% generated by scripts/export_macros.py -- do not edit\n" + "\n".join(lines) + "\n",
                                       encoding="utf-8")
    pd.DataFrame(rows).to_csv(PAPER / "numbers_map.csv", index=False)
    print(f"wrote {len(rows)} macros")


if __name__ == "__main__":
    sys.exit(main())

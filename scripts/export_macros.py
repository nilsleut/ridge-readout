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
A value spec fails unless its query matches exactly one row. A "count" spec counts the
rows its query matches (used for cell counts); it fails if the source is missing or the
query matches no row at all. Macro names contain letters only.

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


def fmt(v, how):
    if how == "int":
        return str(int(round(float(v))))
    return f"{float(v):.{3 if how == 'rsa' else 2}f}"


def main():
    cache, lines, rows = {}, [], []
    for name, src, q, col, how in SPECS:
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise SystemExit(f"invalid macro name {name}")
        count = src.startswith("count:")
        path = RES / src.split(":", 1)[-1]
        if not path.exists():
            raise SystemExit(f"missing source {path}")
        if path not in cache:
            cache[path] = pd.read_csv(path, low_memory=False)
        hit = cache[path].query(q, engine="python")
        if count:
            if len(hit) == 0:
                raise SystemExit(f"{name}: count filter matches no row: {q}")
            val, colname = len(hit), "row count"
        else:
            if len(hit) != 1:
                raise SystemExit(f"{name}: filter matches {len(hit)} rows, expected 1: {q}")
            val, colname = hit.iloc[0][col], col
        text = fmt(val, how)
        rounding = {"rsa": "3 decimals", "corr": "2 decimals", "int": "integer"}[how]
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

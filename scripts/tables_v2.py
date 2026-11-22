"""
tables_v2.py
============
Macro specs and generated tables for manuscript v2 (follow-up steps 40-48b and the predictions
ledger). Imported by export_macros.py: build() returns the spec tuples
(macro, source, filter, column, format) and writes the tables to paper/drafts/v2tab/*.tex.
Every number in those tables is a macro whose value export_macros.py reads from a stored CSV
(results/ or paper/PREDICTIONS_LEDGER.csv); the table layout (rows, cells, labels) is fixed
here, and the set of rows of the multiverse tables (class representatives) is read from
results/step43/multiverse_nsd.csv.

Formats (export_macros.fmt): s2 / s3 / s4 signed with 2 / 3 / 4 decimals; f2 / f3 / f4 unsigned;
p4 p-value with 4 decimals; pct a fraction as an integer percent; int; corr (2 decimals);
sci2 scientific with 2 significant digits.

results/step48/amplitude.csv has no seed in the THINGS rows (step48_conventions.write_part
drops the seed column of the existing part on merge); the values are in generation order, and
the "amp48" source reconstructs the seed exactly as scripts/step48b_dominance.py does
(cumcount over seeds 0-4 within each condition). Macros here average over seeds, so they do not
depend on the reconstruction; figures_v2 uses the same loader.
"""
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
TABDIR = REPO / "paper" / "drafts" / "v2tab"

SPECS = []
_NAMES = set()

WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
         "8": "Eight", "9": "Nine"}


def nm(*parts):
    s = ""
    for p in parts:
        p = str(p)
        p = re.sub(r"\d+", lambda m: "".join(WORDS[c] for c in m.group(0)), p)
        s += "".join(w[0].upper() + w[1:] for w in re.split(r"[^A-Za-z]+", p) if w)
    return s


def M(name):
    return "\\" + name


def S(name, src, q, col, fmt):
    """Register a spec; return the macro call."""
    if name in _NAMES:
        return "\\" + name
    _NAMES.add(name)
    SPECS.append((name, src, q, col, fmt))
    return "\\" + name


def csv(rel):
    return pd.read_csv(RES / rel, low_memory=False)


CONV_T = {"T-raw": "Traw", "T-global": "Tglobal", "T-session": "Tsession"}
CONV_N = {"N-raw": "Nraw", "N-global": "Nglobal", "N-session": "Nsession"}
CELLS = {("raw", "correlation"): "RawCorr", ("z", "correlation"): "ZCorr", ("raw", "euclidean"): "RawEucl",
         ("z", "euclidean"): "ZEucl"}
ROIN = {"V1": "Vone", "LOC": "Loc", "LO-obj": "Loobj", "lateral": "Lat"}
SUBJ3 = ["sub-01", "sub-02", "sub-03"]
SUBJ8 = [f"subj0{i}" for i in range(1, 9)]
SUBN = {s: nm("Sub", s[-1]) for s in SUBJ3} | {s: nm("Subj", s[-1]) for s in SUBJ8}


# ── ledger ──────────────────────────────────────────────────────────────────
def ledger_specs():
    L = "ledger:paper/PREDICTIONS_LEDGER.csv"
    pred = "kind in ('prediction', 'sanity')"
    for name, q in (("FailedPredictions", f"{pred} and result == 'not met'"),
                    ("FailedPredictionsVone", f"{pred} and result == 'not met' and stage == 'v1'"),
                    ("FailedPredictionsFollowup", f"{pred} and result == 'not met' and stage == 'follow-up'"),
                    ("TotalPredictions", f"{pred} and result in ('met', 'not met')"),
                    ("TotalPredictionsVone", f"{pred} and result in ('met', 'not met') and stage == 'v1'"),
                    ("TotalPredictionsFollowup", f"{pred} and result in ('met', 'not met') and stage == 'follow-up'"),
                    ("NotInterpretedPredictions", f"{pred} and result == 'not interpreted'"),
                    ("InterpretationRules", "kind in ('interpretation', 'decision')"),
                    ("AmendmentsFollowup", "stage == 'follow-up' and kind in ('prediction', 'sanity', 'interpretation', 'decision')")):
        S(name, L, q, "count" if name != "AmendmentsFollowup" else "amendments", "int")


# ── step 48 conventions ─────────────────────────────────────────────────────
C48 = "step48/conventions.csv"


def conv_specs():
    for cv, cn in CONV_T.items():
        for roi in ("V1", "LOC"):
            for (u, d), cc in CELLS.items():
                q = f"kind == 'things720' and convention == '{cv}' and roi == '{roi}' and units == '{u}' and distance == '{d}'"
                S(nm("Cv", "Tsev", cn, ROIN[roi], cc), C48, q, "delta", "s3")
                S(nm("Cv", "Tsev", cn, ROIN[roi], cc, "Pos"), C48, q, "n_subjects_positive", "int")
                for s in SUBJ3:
                    qs = q.replace("kind == 'things720'", "kind == 'things720_subject'") + f" and subject == '{s}'"
                    S(nm("Cv", "Tsev", cn, ROIN[roi], cc, SUBN[s]), C48, qs, "delta", "s3")
                for k in (1, 12):
                    q = (f"kind == 'things_test' and convention == '{cv}' and roi == '{roi}' and units == '{u}' "
                         f"and distance == '{d}' and k == {k}")
                    S(nm("Cv", "Ttest", cn, ROIN[roi], cc, "K", k), C48, q, "delta", "s3")
                    S(nm("Cv", "Ttest", cn, ROIN[roi], cc, "K", k, "Pos"), C48, q, "n_subjects_positive", "int")
    for cv, cn in CONV_N.items():
        for roi in ("V1", "LO-obj", "lateral"):
            for (u, d), cc in CELLS.items():
                for k in (1, 3):
                    q = (f"kind == 'nsd' and convention == '{cv}' and roi == '{roi}' and units == '{u}' and distance == '{d}' "
                         f"and k == {k}")
                    base = nm("Cv", "Nsd", cn, ROIN[roi], cc, "K", k)
                    S(base, C48, q, "delta", "s3")
                    S(base + "Pos", C48, q, "n_subjects_positive", "int")
                    S(base + "P", C48, q, "p_positive", "p4")
    for q, n in (("distinct analyses", "Distinct"), ("distinct analyses, interval above 0", "Above"),
                 ("distinct analyses, interval below 0", "Below"), ("distinct analyses, interval includes 0", "Incl")):
        S(nm("Mv", "Nraw", n), C48, f"kind == 'nsd_multiverse_summary' and quantity == '{q}'", "value", "int")
        S(nm("Mv", "Nsession", n), "step43/multiverse_nsd.csv", f"kind == 'summary' and quantity == '{q}'", "value", "int")
        S(nm("Mv", "Nrc", n), "step47b/robustness.csv", f"kind == 'nsd_multiverse_summary' and quantity == '{q}'", "value", "int")
    # per-subject NSD paper cell under N-raw, k = 1 (abstract: 8 of 8)
    for s in SUBJ8:
        S(nm("Cv", "Nsd", "Nraw", "Vone", "RawCorr", "K", 1, SUBN[s]), C48,
          f"kind == 'nsd_subject' and convention == 'N-raw' and roi == 'V1' and units == 'raw' and distance == 'correlation' "
          f"and k == 1 and subject == '{s}'", "delta", "s3")


def conv_table():
    """Main-text table: Delta of three cells under the three conventions in each dataset."""
    rows = []
    for dataset, label in (("Tsev", "THINGS-720"), ("TtestOne", "THINGS test, $k = 1$"), ("TtestTwelve", "THINGS test, $k = 12$"),
                           ("NsdOne", "NSD, $k = 1$"), ("NsdThree", "NSD, $k = 3$")):
        convs = CONV_T if dataset.startswith("T") else CONV_N
        for cv, cn in convs.items():
            cells = []
            for cc, nsub in (("RawCorr", None), ("ZCorr", None), ("RawEucl", None)):
                if dataset == "Tsev":
                    m, pos, n = nm("Cv", "Tsev", cn, "Vone", cc), nm("Cv", "Tsev", cn, "Vone", cc, "Pos"), 3
                elif dataset.startswith("Ttest"):
                    k = 1 if dataset.endswith("One") else 12
                    m, pos, n = nm("Cv", "Ttest", cn, "Vone", cc, "K", k), nm("Cv", "Ttest", cn, "Vone", cc, "K", k, "Pos"), 3
                else:
                    k = 1 if dataset.endswith("One") else 3
                    m, pos, n = nm("Cv", "Nsd", cn, "Vone", cc, "K", k), nm("Cv", "Nsd", cn, "Vone", cc, "K", k, "Pos"), 8
                cells.append(f"${M(m)}$ ({M(pos)}/{n})")
            rows.append(f"{label} & {cv} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    body = "\n".join(rows[:-1])
    return (r"""\begin{table}[t]
\centering
\small
\begin{tabular}{@{}llrrr@{}}
\toprule
Data & Convention & raw / correlation & z / correlation & raw / Euclidean \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Untrained minus Backprop ($\Delta$) at V1 under the three voxel normalisation conventions
(Section~\ref{sec:convmethods}), for the paper cell (raw model units, correlation distance), the
standardised cell (model units z-scored over the images) and raw units with Euclidean distance.
Mean over seeds 0--4, then subjects; in brackets the number of subjects with $\Delta > 0$.
T-global and N-session are the conventions of the respective data sets (THINGS: the paper; NSD:
the NSD standard).}
\label{tab:conventions}
\end{table}
""")


# ── step 47c / 47 / 44 / 45 / 47b ───────────────────────────────────────────
def why_specs():
    for roi in ("V1", "LOC"):
        for v in ("V0", "V1", "V2", "V3", "V4", "V5"):
            S(nm("Dec", ROIN[roi], v), "step47c/decomposition.csv", f"kind == 'delta' and roi == '{roi}' and variant == '{v}'",
              "delta", "s4")
        for s in SUBJ3:
            for v in ("V0", "V1", "V2", "V3"):
                S(nm("Dec", ROIN[roi], v, SUBN[s]), "step47c/decomposition.csv",
                  f"kind == 'subject_delta' and roi == '{roi}' and variant == '{v}' and subject == '{s}'", "delta", "s4")
    for v in ("V0", "V1", "V3"):
        S(nm("DecThree", "Vone", v), "step47c/decomposition.csv", f"kind == 'delta' and roi == 'V1' and variant == '{v}'",
          "delta", "s3")
    for s in SUBJ3:
        q = f"kind == 'run_deviation' and roi == 'V1' and subject == '{s}'"
        S(nm("Dec", "Sd", SUBN[s]), "step47c/decomposition.csv", q, "sd_d_dot_mhat", "f2")
        S(nm("Dec", "Share", SUBN[s]), "step47c/decomposition.csv", q, "share_along_mhat", "pct")
        S(nm("Dec", "CosLum", SUBN[s]), "step47c/decomposition.csv", q, "rho_cos_lum_without", "s2")
        S(nm("Dec", "CosLumRc", SUBN[s]), "step47c/decomposition.csv", q, "rho_cos_lum_with", "s2")
    S("CenterSevenTwentyCell", "step7/multiverse.csv",
      "kind == 'cell' and F == 'raw' and D == 'correlation' and B == 'center720' and roi == 'V1' and "
      "quantity == 'gap NATIVE 224: Random Weights - Backprop'", "value", "s4")
    S("CenterSevenTwentyCellLo", "step7/multiverse.csv",
      "kind == 'cell' and F == 'raw' and D == 'correlation' and B == 'center720' and roi == 'V1' and "
      "quantity == 'gap NATIVE 224: Random Weights - Backprop'", "ci_lo", "s4")
    S("CenterSevenTwentyCellHi", "step7/multiverse.csv",
      "kind == 'cell' and F == 'raw' and D == 'correlation' and B == 'center720' and roi == 'V1' and "
      "quantity == 'gap NATIVE 224: Random Weights - Backprop'", "ci_hi", "s4")
    for v, n in (("raw", "Raw"), ("run-centred", "Rc"), ("run-centred, L + C", "RcLc")):
        for roi in ("V1", "LOC"):
            S(nm("RunCent", ROIN[roi], n), "step47/run_centring.csv", f"kind == 'delta' and roi == '{roi}' and variant == '{v}'",
              "delta", "s4")
    for v, n in (("raw", "Raw"), ("run-centred", "Rc")):
        S(nm("LumBound", n), "step47/run_centring.csv", f"kind == 'lum_bound' and roi == 'V1' and variant == '{v}'", "ratio", "f2")
    for cov, n in (("none", "None"), ("L+C", "Lc"), ("T", "T"), ("L+C+T", "Lct")):
        for roi in ("V1", "LOC"):
            S(nm("Part", "Ff", ROIN[roi], n), "step44/partial.csv", f"kind == 'delta' and roi == '{roi}' and covariates == '{cov}'",
              "delta", "s4")
    for cov, n in (("LP", "Lp"), ("L+C+LP", "Lclp"), ("L+C+LP+T", "Lclpt")):
        for roi in ("V1", "LOC"):
            S(nm("Part", "Fff", ROIN[roi], n), "step45/partial_lp.csv",
              f"kind == 'delta' and dataset == 'things720' and roi == '{roi}' and covariates == '{cov}'", "delta", "s4")
    for ds, dn in (("things_test", "Test"), ("nsd", "Nsd")):
        for cov, n in (("none", "None"), ("L+C", "Lc"), ("LP", "Lp"), ("L+C+LP", "Lclp")):
            for roi in (("V1", "LOC") if ds == "things_test" else ("V1", "LO-obj")):
                S(nm("Part", dn, ROIN[roi], n), "step45/partial_lp.csv",
                  f"kind == 'delta' and dataset == '{ds}' and roi == '{roi}' and covariates == '{cov}'", "delta", "s4")
    S("OrderRhoSubTwo", "step44/order.csv", "kind == 'T_vs_brain' and subject == 'sub-02' and roi == 'V1'", "rho", "s3")
    for s in SUBJ3:
        S(nm("OrderRho", SUBN[s]), "step44/order.csv", f"kind == 'T_vs_brain' and subject == '{s}' and roi == 'V1'", "rho", "s3")
    for rid, col, fmt in (("W1", "value", "pct"), ("W2", "value", "s3"), ("W5", "value", "f4"), ("W4", "value", "f2")):
        S(nm("Rule", rid), "step44/summary.csv", f"rule == '{rid}'", col, fmt)
    for ds, n in (("things720", "Tsev"), ("things_test", "Test"), ("nsd", "Nsd")):
        S(nm("LumRatio", n), "step44/luminance_by_dataset.csv", f"kind == 'brain_lum' and dataset == '{ds}' and roi == 'V1'",
          "ratio", "f2")
        S(nm("LpRatio", n), "step45/lp_by_dataset.csv", f"kind == 'lp' and dataset == '{ds}' and roi == 'V1'", "ratio", "f2")
    S("RuleWSix", "step45/summary.csv", "rule == 'W6'", "value", "s3")
    S("RuleWSixFraction", "step45/summary.csv", "rule == 'interpretation'", "value", "pct")
    S("RuleRcOne", "step47/summary.csv", "rule == 'RC1'", "value", "s3")
    S("RuleRcTwo", "step47/summary.csv", "rule == 'RC2'", "value", "f2")
    S("RuleRcFour", "step47/summary.csv", "rule == 'RC4'", "value", "int")
    S("RuleCoOne", "step47/summary.csv", "rule == 'CO1'", "value", "sci2")
    for a, n in (("sub-01/02", "SubOneTwo"), ("sub-03", "SubThree")):
        S(nm("RunLumPct", n), "step47/run_luminance.csv", f"kind == 'assignment' and assignment == '{a}'", "percentile", "f1")
    for s in SUBJ3:
        q = f"kind == 'subject' and subject == '{s}'"
        for col, n, f in (("delta_A", "DeltaA", "s3"), ("percentile", "Pct", "f1"), ("rho_run_index", "RunIdx", "s2"),
                          ("run_level_part", "RunPart", "s3")):
            S(nm("Mech", n, SUBN[s]), "step47b/mechanism.csv", q, col, f)


def why_tables():
    t1 = [r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
          r"Variant (Section~\ref{sec:convmethods}) & sub-01 & sub-02 & sub-03 & mean V1 & mean LOC \\", r"\midrule"]
    labels = {"V0": "V0 nothing removed (paper cell)", "V1": "V1 session-mean profile $m$ removed",
              "V2": "V2 run deviations $d_r$ removed", "V3": "V3 both removed (run centring)",
              "V4": r"V4 = V3 + $d_{r\parallel}$ re-added", "V5": r"V5 = V3 + $d_{r\perp}$ re-added"}
    for v, lab in labels.items():
        subs = " & ".join((f"${M(nm('Dec', 'Vone', v, SUBN[s]))}$" if v in ("V0", "V1", "V2", "V3") else "") for s in SUBJ3)
        t1.append(f"{lab} & {subs} & ${M(nm('Dec', 'Vone', v))}$ & ${M(nm('Dec', 'Loc', v))}$ \\\\")
    t1 += [r"\bottomrule", r"\end{tabular}",
           r"\caption{Decomposition of run centring in THINGS-720 (step 47c): $\Delta$ (untrained minus Backprop, paper cell, cross-run pairs) after removing parts of the voxel means. Per subject for V0--V3, V1 ROI.}",
           r"\label{tab:decomp}", r"\end{table}"]
    t2 = [r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
          r"Covariates partialled (THINGS-720, paper cell) & V1 & LOC \\", r"\midrule"]
    for lab, mv, ml in (("none", "PartFfVoneNone", "PartFfLocNone"), ("$L$, $C$ (global luminance, RMS contrast)", "PartFfVoneLc", "PartFfLocLc"),
                        ("$T$ (presentation position)", "PartFfVoneT", "PartFfLocT"), ("$L$, $C$, $T$", "PartFfVoneLct", "PartFfLocLct"),
                        ("LP (coarse luminance image)", "PartFffVoneLp", "PartFffLocLp"), ("$L$, $C$, LP", "PartFffVoneLclp", "PartFffLocLclp"),
                        ("$L$, $C$, LP, $T$", "PartFffVoneLclpt", "PartFffLocLclpt")):
        t2.append(f"{lab} & ${M(mv)}$ & ${M(ml)}$ \\\\")
    t2 += [r"\midrule", r"run centring (step 47) & $\RunCentVoneRc$ & $\RunCentLocRc$ \\",
           r"run centring, then $L$, $C$ & $\RunCentVoneRcLc$ & $\RunCentLocRcLc$ \\", r"\bottomrule", r"\end{tabular}",
           r"\caption{Partial RSA in THINGS-720 (steps 44, 45, 47): $\Delta$ after rank-regressing both RDMs on covariate RDMs. THINGS test images and NSD: Table~\ref{tab:partialother}.}",
           r"\label{tab:partial}", r"\end{table}"]
    t3 = [r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
          r"Covariates & test V1 ($k = 1$) & test LOC ($k = 1$) & NSD V1 ($k = 1$) & NSD LO-obj ($k = 1$) \\", r"\midrule"]
    for cov, n in (("none", "None"), ("$L$, $C$", "Lc"), ("LP", "Lp"), ("$L$, $C$, LP", "Lclp")):
        t3.append(f"{cov} & ${M(nm('Part', 'Test', 'Vone', n))}$ & ${M(nm('Part', 'Test', 'Loc', n))}$ & "
                  f"${M(nm('Part', 'Nsd', 'Vone', n))}$ & ${M(nm('Part', 'Nsd', 'Loobj', n))}$ \\\\")
    t3 += [r"\bottomrule", r"\end{tabular}", r"\caption{Partial RSA in the THINGS test images and in NSD (step 45), paper cell, standard normalisation of each data set.}",
           r"\label{tab:partialother}", r"\end{table}"]
    t4 = [r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
          r"Subject & $\Delta A$ & percentile & $\rho$(run RDM, run index) & run-level part of $\Delta$ & SD of $d_r\!\cdot\!\hat m$ & share along $\hat m$ & order $\rho$ \\",
          r"\midrule"]
    for s in SUBJ3:
        t4.append(f"{s} & ${M(nm('Mech', 'DeltaA', SUBN[s]))}$ & ${M(nm('Mech', 'Pct', SUBN[s]))}$ & "
                  f"${M(nm('Mech', 'RunIdx', SUBN[s]))}$ & ${M(nm('Mech', 'RunPart', SUBN[s]))}$ & "
                  f"${M(nm('Dec', 'Sd', SUBN[s]))}$ & ${M(nm('Dec', 'Share', SUBN[s]))}\\%$ & ${M(nm('OrderRho', SUBN[s]))}$ \\\\")
    t4 += [r"\bottomrule", r"\end{tabular}",
           r"\caption{Run-level quantities per subject (steps 44, 47b, 47c). $\Delta A$: alignment of the brain run RDM with the untrained minus the Backprop run RDM, and its percentile in the null of random run assignments; run-level part: $\Delta$ without minus with run centring; SD and share: the projection of the run deviations on the session-mean profile; order $\rho$: Spearman between the V1 RDM and the presentation-position RDM.}",
           r"\label{tab:runlevel}", r"\end{table}"]
    return {"why_decomp": "\n".join(t1), "why_partial": "\n".join(t2), "why_partialother": "\n".join(t3),
            "why_runlevel": "\n".join(t4)}


# ── steps 40 / 41: banded ridge and test images ─────────────────────────────
def banded_specs():
    B = "step40/banded_summary.csv"
    S("BandedBOneMinGFour", B, "kind == 'rule' and g == 4 and quantity.str.startswith('B1')", "value", "s4")
    S("BandedBOneMinGOne", B, "kind == 'rule' and g == 1 and quantity.str.startswith('B1')", "value", "s4")
    S("BandedBOneMinGEight", B, "kind == 'rule' and g == 8 and quantity.str.startswith('B1')", "value", "s4")
    for g in (1, 4, 8):
        q = f"kind == 'U_mean' and g == {g}"
        for col, n in (("U_BP", "UBp"), ("U_Random", "URand"), ("U_skill_BP", "USkillBp"), ("U_skill_Random", "USkillRand")):
            S(nm("Banded", n, "G", g), B, q, col, "s4")
        for col, n in (("bank", "Bank"), ("net_BP", "NetBp"), ("net_Random", "NetRand")):
            S(nm("Banded", n, "G", g), B, q, col, "f3")
    Q = "step41/summary.csv"
    for col, n in (("U_BP", "UBp"), ("U_Random", "URand"), ("U_skill_BP", "USkillBp"), ("U_skill_Random", "USkillRand")):
        S(nm("Cond", n), Q, "kind == 'U_conditional_mean'", col, "s4")
    S("DAOneDev", Q, "kind == 'rule' and quantity.str.startswith('D-A1')", "value", "sci2")
    S("DATwoMin", Q, "kind == 'rule' and quantity.str.startswith('D-A2')", "value", "s4")
    S("NTwoSpearman", Q, "kind == 'rule' and quantity.str.startswith('N2')", "value", "s2")
    C = "step40/cvrsa_summary.csv"
    for roi in ("V1", "LOC"):
        for u in ("raw", "z"):
            for d, b in (("correlation", "b1"), ("euclidean", "b2"), ("euclidean", "b3")):
                q = (f"kind == 'contrast' and roi == '{roi}' and model_units == '{u}' and model_distance == '{d}' and brain_rdm == '{b}'")
                base = nm("Cv", "Rsa", ROIN[roi], u, b)
                S(base, C, q, "value", "s3")
                S(base + "Lo", C, q, "ci_lo", "s3")
                S(base + "Hi", C, q, "ci_hi", "s3")
    N = "step41/noise_curve.csv"
    for roi in ("V1", "LOC"):
        for u in ("raw", "z"):
            for k in (1, 2, 3, 4, 6, 12):
                q = (f"kind == 'contrast' and roi == '{roi}' and model_units == '{u}' and model_distance == 'correlation' "
                     f"and brain_rdm == 'b1' and k == {k}")
                base = nm("Noise", ROIN[roi], u, "K", k)
                S(base, N, q, "delta", "s3")
                S(base + "Four", N, q, "delta", "s4")
                if k in (1, 12):
                    S(base + "Lo", N, q, "ci_lo", "s3")
                    S(base + "Hi", N, q, "ci_hi", "s3")


def banded_tables():
    t = [r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Readout & $g$ & bank & network & $U$ (median $r$) & $U$ (MSE skill) & min.\ joint $-$ bank \\", r"\midrule"]
    for g, gn in ((1, "One"), (4, "Four"), (8, "Eight")):
        for net, nn in (("Backprop", "Bp"), ("untrained", "Rand")):
            last = f"${M('BandedBOneMinG' + gn)}$" if nn == "Bp" else ""
            t.append(f"banded, 2D search, {net} & {g} & ${M('BandedBankG' + gn)}$ & ${M('BandedNet' + nn + 'G' + gn)}$ & "
                     f"${M('BandedU' + nn + 'G' + gn)}$ & ${M('BandedUSkill' + nn + 'G' + gn)}$ & {last} \\\\")
    t.append(r"banded, conditional search, Backprop & 4 & & & $\CondUBp$ & $\CondUSkillBp$ & $\DATwoMin$ \\")
    t.append(r"banded, conditional search, untrained & 4 & & & $\CondURand$ & $\CondUSkillRand$ & \\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Banded ridge in THINGS-720 (steps 40, 41; V1, reliable voxels; mean over subjects and seeds). $U$: joint readout minus bank alone. Minimum joint minus bank: over all fits (B1 and D-A2 required $\geq -0.002$). B2 and B3 are not interpreted because B1 was not met; the conditional estimator is not interpreted because D-A2 was not met.}",
          r"\label{tab:banded}", r"\end{table}"]
    n = [r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"ROI & units & $k = 1$ & 2 & 3 & 4 & 6 & 12 \\", r"\midrule"]
    for roi in ("V1", "LOC"):
        for u in ("raw", "z"):
            cells = []
            for k in (1, 2, 3, 4, 6, 12):
                b = nm("Noise", ROIN[roi], u, "K", k)
                cells.append(f"${M(b)}$" + (f" {{\\scriptsize [${M(b + 'Lo')}$, ${M(b + 'Hi')}$]}}" if k in (1, 12) else ""))
            n.append(f"{roi} & {u} & " + " & ".join(cells) + r" \\")
    n += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Noise curve on the 100 THINGS test images (step 41): $\Delta(k)$, untrained minus Backprop, correlation distance on the mean of $k$ of the 12 presentations (20 draws per $k < 12$). Intervals: 95\% stimulus bootstrap.}",
          r"\label{tab:noisecurve}", r"\end{table}"]
    return {"banded": "\n".join(t), "noisecurve": "\n".join(n)}


# ── NSD (steps 42, 43, 47, 47b) ─────────────────────────────────────────────
def nsd_specs():
    G = "step42/group.csv"
    for subs, sn in (("all 8", "All"), ("subj02-08", "Held")):
        for roi in ("V1", "LO-obj", "lateral"):
            for k in (1, 3):
                for u, d in (("raw", "correlation"), ("z", "correlation"), ("raw", "euclidean"), ("z", "euclidean"),
                             ("center", "N"), ("center", "A"), ("center", "W")):
                    if k == 1 and u == "center":
                        continue
                    q = (f"kind == 'group' and subjects == '{subs}' and roi == '{roi}' and k == {k} and model_units == '{u}' "
                         f"and distance == '{d}'")
                    base = nm("Nsd", sn, ROIN[roi], "K", k, u, d)
                    S(base, G, q, "mean_delta", "s3")
                    S(base + "Lo", G, q, "ci_lo", "s3")
                    S(base + "Hi", G, q, "ci_hi", "s3")
                    S(base + "Neg", G, q, "n_negative", "int")
                    S(base + "Pos", G, q, "n_positive", "int")
                    S(base + "Pn", G, q, "p_negative", "p4")
                    S(base + "Pp", G, q, "p_positive", "p4")
    P = "step42/pilot_subj01.csv"
    for roi in ("V1", "LO-obj"):
        for u, d in (("raw", "correlation"), ("z", "correlation"), ("raw", "euclidean"), ("z", "euclidean")):
            S(nm("Pilot", ROIN[roi], u, d), P,
              f"kind == 'contrast' and roi == '{roi}' and k == 3 and model_units == '{u}' and distance == '{d}'", "delta", "s3")
    R = "step47/nsd_ridge.csv"
    for roi in ("V1", "LO-obj"):
        for g in (1, 4, 8):
            q = f"kind == 'group' and roi == '{roi}' and grid == {g}"
            S(nm("NsdRidge", ROIN[roi], "G", g), R, q, "delta", "s3")
            S(nm("NsdRidge", ROIN[roi], "G", g, "Neg"), R, q, "n_negative", "int")
            S(nm("NsdRidge", ROIN[roi], "G", g, "P"), R, q, "p_negative", "p4")
            for c, cn in (("BP", "Bp"), ("Random", "Rand"), ("bank", "Bank")):
                S(nm("NsdRidge", ROIN[roi], "G", g, cn), "avg:" + R,
                  f"kind == 'fit' and roi == '{roi}' and grid == {g} and condition == '{c}'", "r_med_reliable", "f3")
            for s in SUBJ8:
                qs = f"kind == 'subject' and roi == '{roi}' and grid == {g} and subject == '{s}'"
                S(nm("NsdRidge", ROIN[roi], "G", g, SUBN[s]), R, qs, "delta", "s3")
                S(nm("NsdRidge", ROIN[roi], "G", g, SUBN[s], "Bank"), R, qs, "r_bank", "f3")
                S(nm("NsdRidge", ROIN[roi], "G", g, SUBN[s], "Bp"), R, qs, "r_bp", "f3")
                S(nm("NsdRidge", ROIN[roi], "G", g, SUBN[s], "Rand"), R, qs, "r_random", "f3")
                if g == 4 and roi == "V1":
                    S(nm("NsdReliable", SUBN[s]), R, qs, "n_reliable", "int")
                    S(nm("NsdVoxels", SUBN[s]), R, qs, "n_voxels", "int")
    for roi in ("V1", "LO-obj", "lateral"):
        for s in SUBJ8:
            S(nm("NsdVox", ROIN[roi], SUBN[s]), "step42/gates.csv", f"gate == 'G3' and quantity == 'voxels {roi}' and subject == '{s}'",
              "value", "int")
    Y = "step47b/robustness.csv"
    S("YOneThingsRc", Y, "kind == 'things_ridge' and variant == 'rc' and grid == 4", "delta", "s3")
    S("YOneThingsRaw", Y, "kind == 'things_ridge' and variant == 'raw (results/prereg)' and grid == 4", "delta", "s3")
    S("YThreeNsdRc", Y, "kind == 'nsd_ridge_group' and variant == 'rc' and grid == 4 and roi == 'V1'", "delta", "s3")
    S("YFourNsdAll", Y, "kind == 'nsd_ridge_group' and variant == 'raw_all' and grid == 4 and roi == 'V1'", "delta", "s3")
    for u, un in (("raw", "Raw"), ("z", "Z")):
        S(nm("YTwoNsdRc", un), Y, f"kind == 'nsd_rsa_group' and variant == 'run-centred' and roi == 'V1' and k == 3 and units == '{u}' "
          f"and distance == 'correlation'", "delta", "s3")
    I = "step43/imagenet_nsd.csv"
    for unt, un in (("mean s0-s2", ""), ("s0 only", "Szero")):
        for m in ("alexnet", "vgg11_bn", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"):
            for roi in ("V1", "LO-obj"):
                for u, d in (("raw", "correlation"), ("z", "correlation"), ("center", "N"), ("center", "A"), ("center", "W")):
                    q = (f"kind == 'cell' and untrained == '{unt}' and model == '{m}' and roi == '{roi}' and model_units == '{u}' "
                         f"and distance == '{d}'")
                    base = nm("Imn", un, m, ROIN[roi], u, d)
                    S(base, I, q, "mean_delta", "s3")
                    S(base + "Neg", I, q, "n_negative", "int")
    for d, dn in (("A", "A"), ("correlation", "RawCorr")):
        u = "center" if d == "A" else "raw"
        q = f"kind == 'cell' and untrained == 'mean s0-s2' and roi == 'V1' and model_units == '{u}' and distance == '{d}'"
        S(nm("Imn", "Vone", dn, "Min"), "min:" + I, q, "mean_delta", "s3")
        S(nm("Imn", "Vone", dn, "Max"), "max:" + I, q, "mean_delta", "s3")
    RS = "step43/resolution_nsd.csv"
    for u in ("raw", "z"):
        for px in (32, 64, 96, 128, 160, 224):
            q = f"kind == 'delta' and model_units == '{u}' and px == {px}"
            S(nm("Res", u, px), RS, q, "mean_delta", "s3")
            S(nm("Res", u, px, "Lo"), RS, q, "ci_lo", "s3")
            S(nm("Res", u, px, "Hi"), RS, q, "ci_hi", "s3")
        q = f"kind == 'slope' and model_units == '{u}'"
        S(nm("Res", u, "Slope"), RS, q, "mean_delta", "s3")
        S(nm("Res", u, "Slope", "Lo"), RS, q, "ci_lo", "s3")
        S(nm("Res", u, "Slope", "Hi"), RS, q, "ci_hi", "s3")
    CE = "step43/ceiling_nsd.csv"
    for roi in ("V1", "LO-obj"):
        for k in (1, 3):
            q = f"kind == 'ceiling' and roi == '{roi}' and k == {k}"
            base = nm("Ceil", ROIN[roi], "K", k)
            S(base, CE, q, "value", "f3")
            S(base + "Lo", CE, q, "ci_lo", "f3")
            S(base + "Hi", CE, q, "ci_hi", "f3")
            S(base + "Rand", CE, q, "mean_rho_random_raw", "f3")
            S(base + "Bp", CE, q, "mean_rho_bp_raw", "f3")
    S("CeilVoneFracMin", "exprmin:" + CE, "kind == 'ceiling' and roi == 'V1'", "mean_rho_random_raw / value", "pct")
    S("CeilVoneFracMax", "exprmax:" + CE, "kind == 'ceiling' and roi == 'V1'", "mean_rho_bp_raw / value", "pct")
    S("CeilThingsLoso", "step44/luminance_by_dataset.csv", "kind == 'brain_lum' and dataset == 'things720' and roi == 'V1'",
      "loso_lower", "f3")


def nsd_tables():
    out = {}
    t = [r"\begin{table}[H]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}l" + "r" * 8 + r"@{}}", r"\toprule", "& " + " & ".join(SUBJ8) + r" \\", r"\midrule"]
    for lab, fn in (("voxels V1", lambda s: nm("NsdVox", "Vone", SUBN[s])), ("voxels LO-obj", lambda s: nm("NsdVox", "Loobj", SUBN[s])),
                    ("voxels lateral", lambda s: nm("NsdVox", "Lat", SUBN[s])), ("reliable V1 voxels (ridge)", lambda s: nm("NsdReliable", SUBN[s])),
                    (r"ridge $g = 1$, $\Delta$", lambda s: nm("NsdRidge", "Vone", "G", 1, SUBN[s])),
                    (r"ridge $g = 4$, $\Delta$", lambda s: nm("NsdRidge", "Vone", "G", 4, SUBN[s])),
                    (r"ridge $g = 8$, $\Delta$", lambda s: nm("NsdRidge", "Vone", "G", 8, SUBN[s])),
                    (r"ridge $g = 4$, bank", lambda s: nm("NsdRidge", "Vone", "G", 4, SUBN[s], "Bank")),
                    (r"ridge $g = 4$, Backprop", lambda s: nm("NsdRidge", "Vone", "G", 4, SUBN[s], "Bp")),
                    (r"ridge $g = 4$, untrained", lambda s: nm("NsdRidge", "Vone", "G", 4, SUBN[s], "Rand")),
                    (r"ridge LO-obj $g = 4$, $\Delta$", lambda s: nm("NsdRidge", "Loobj", "G", 4, SUBN[s]))):
        t.append(lab + " & " + " & ".join(f"${M(fn(s))}$" for s in SUBJ8) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{NSD per subject: voxel counts (all voxels of each ROI that were valid in every session), reliable V1 voxels used by the ridge readout (split-half reliability over the three presentations, Spearman--Brown corrected, $> 0.2$), and ridge median $r$ (untrained minus Backprop, mean over seeds; bank: luminance-and-contrast bank).}",
          r"\label{tab:nsdsubjects}", r"\end{table}"]
    out["nsd_subjects"] = "\n".join(t)
    t = [r"\begin{table}[H]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
         r"ROI & $k$ & raw, corr. & z, corr. & raw, Eucl. & z, Eucl. & $N$ & $A$ & $W$ \\", r"\midrule"]
    for roi in ("V1", "LO-obj", "lateral"):
        for k in (3, 1):
            cells = []
            for u, d in (("raw", "correlation"), ("z", "correlation"), ("raw", "euclidean"), ("z", "euclidean"),
                         ("center", "N"), ("center", "A"), ("center", "W")):
                b = nm("Nsd", "All", ROIN[roi], "K", k, u, d)
                if b in _NAMES and (k == 3 or d in ("correlation", "euclidean")):
                    cells.append(f"${M(b)}$ ({M(b + 'Neg')})")
                else:
                    cells.append("")
            t.append(f"{roi} & {k} & " + " & ".join(cells) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{NSD, all 8 subjects, NSD standard normalisation (N-session): $\Delta$ (untrained minus Backprop, mean over seeds and subjects); in brackets the number of subjects with $\Delta < 0$. $N$, $A$, $W$: norm, angular and norm-weighted angular components on centred units and voxels ($k = 3$).}",
          r"\label{tab:nsdcells}", r"\end{table}"]
    out["nsd_cells"] = "\n".join(t)
    t = [r"\begin{table}[H]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule", r"Model & ROI & raw, corr. & z, corr. & $N$ & $A$ & $W$ \\", r"\midrule"]
    for unt, un, lab in (("mean s0-s2", "", ""), ("s0 only", "Szero", " (s0)")):
        for m in ("alexnet", "vgg11_bn", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"):
            for roi in ("V1", "LO-obj"):
                cells = [f"${M(nm('Imn', un, m, ROIN[roi], u, d))}$ ({M(nm('Imn', un, m, ROIN[roi], u, d) + 'Neg')})"
                         for u, d in (("raw", "correlation"), ("z", "correlation"), ("center", "N"), ("center", "A"), ("center", "W"))]
                t.append(f"\\texttt{{{m.replace('_', chr(92) + '_')}}}{lab} & {roi} & " + " & ".join(cells) + r" \\")
        if unt == "mean s0-s2":
            t.append(r"\midrule")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{ImageNet models on NSD ($k = 3$, 224\,px): untrained (mean over initialisation seeds 0--2, or seed 0 only) minus trained; in brackets the number of subjects with $\Delta < 0$ of 8.}",
          r"\label{tab:nsdimagenet}", r"\end{table}"]
    out["nsd_imagenet"] = "\n".join(t)
    t = [r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
         r"px & raw, correlation & z, correlation \\", r"\midrule"]
    for px in (32, 64, 96, 128, 160, 224):
        t.append(f"{px} & " + " & ".join(f"${M(nm('Res', u, px))}$ [${M(nm('Res', u, px, 'Lo'))}$, ${M(nm('Res', u, px, 'Hi'))}$]"
                                       for u in ("raw", "z")) + r" \\")
    t.append(r"$S = \Delta(224) - \Delta(32)$ & " + " & ".join(
        f"${M(nm('Res', u, 'Slope'))}$ [${M(nm('Res', u, 'Slope', 'Lo'))}$, ${M(nm('Res', u, 'Slope', 'Hi'))}$]" for u in ("raw", "z")) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Evaluation resolution on NSD (step 43, reported without prediction): $\Delta$ at V1, own CNN, NATIVE inputs; 95\% stimulus-bootstrap intervals of the 8-subject mean.}",
          r"\label{tab:nsdres}", r"\end{table}"]
    out["nsd_res"] = "\n".join(t)
    t = [r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
         r"ROI & $k$ & lower bound [95\% interval] & $\rho$ untrained & $\rho$ Backprop \\", r"\midrule"]
    for roi in ("V1", "LO-obj"):
        for k in (3, 1):
            b = nm("Ceil", ROIN[roi], "K", k)
            t.append(f"{roi} & {k} & ${M(b)}$ [${M(b + 'Lo')}$, ${M(b + 'Hi')}$] & ${M(b + 'Rand')}$ & ${M(b + 'Bp')}$ \\\\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Leave-one-subject-out lower bound of the noise ceiling in NSD (correlation distance) and the RSA of the own CNN (raw units, mean over seeds and subjects).}",
          r"\label{tab:nsdceil}", r"\end{table}"]
    out["nsd_ceil"] = "\n".join(t)
    t = [r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"ROI, cell (subj01, $k = 3$) & raw, corr. & z, corr. & raw, Eucl. & z, Eucl. \\", r"\midrule"]
    for roi in ("V1", "LO-obj"):
        t.append(f"{roi} & " + " & ".join(f"${M(nm('Pilot', ROIN[roi], u, d))}$" for u, d in
                                         (("raw", "correlation"), ("z", "correlation"), ("raw", "euclidean"), ("z", "euclidean"))) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{The subj01 pilot (step 42): $\Delta$, untrained minus Backprop, mean over seeds. The positive Euclidean values motivated Q1--Q4 (step 42b), which were tested on subj02--08.}",
          r"\label{tab:nsdpilot}", r"\end{table}"]
    out["nsd_pilot"] = "\n".join(t)
    return out


def multiverse_specs_tables():
    """28 classes: representative cell of each NSD class (step 43), with THINGS, NSD N-session,
    N-raw and run-centred."""
    mv = csv("step43/multiverse_nsd.csv")
    cells = mv[(mv.kind == "cell") & (mv.roi == "V1")]
    reps = cells.groupby("cls", sort=True).head(1)[["cls", "F", "D", "B"]]
    rows = []
    srcs = {"Things": ("step7/multiverse.csv", "kind == 'cell' and roi == 'V1' and quantity == 'gap NATIVE 224: Random Weights - Backprop'",
                       "value", "ci_lo", "ci_hi", "B"),
            "Nsession": ("step43/multiverse_nsd.csv", "kind == 'cell' and roi == 'V1'", "mean_delta", "ci_lo", "ci_hi", "B"),
            "Nraw": ("step48/conventions.csv", "kind == 'nsd_multiverse_cell' and roi == 'V1'", "delta", "ci_lo", "ci_hi", "B"),
            "Nrc": ("step47b/robustness.csv", "kind == 'nsd_multiverse_cell' and roi == 'V1'", "delta", "ci_lo", "ci_hi", "B")}
    for _, r in reps.iterrows():
        cn = nm("Cls", int(r.cls))
        line = [str(int(r.cls)), f"{r.F} / {r.D} / {r.B}"]
        for key, (src, base, vc, lc, hc, bcol) in srcs.items():
            q = f"{base} and F == '{r.F}' and D == '{r.D}' and {bcol} == '{r.B}'"
            m = S(nm("Mv", key, cn), src, q, vc, "s3")
            lo = S(nm("Mv", key, cn, "Lo"), src, q, lc, "s3")
            hi = S(nm("Mv", key, cn, "Hi"), src, q, hc, "s3")
            line.append(f"${m}$ {{\\scriptsize [${lo}$, ${hi}$]}}")
        rows.append(" & ".join(line) + r" \\")
    t = [r"\begin{table}[H]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{@{}rlrrrr@{}}", r"\toprule",
         r"Class & $F$ / $D$ / $B$ (representative) & THINGS-720 & NSD, N-session & NSD, N-raw & NSD, run-centred \\", r"\midrule"]
    t += rows
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{The 28 distinct analyses of the V1 multiverse (class numbering of step 43; one representative cell per class, $B$ = paper denotes the data set's own normalisation). THINGS-720: the paper's multiverse (NATIVE 224\,px); NSD under N-session (the NSD standard), N-raw and run centring. $\Delta$ with 95\% stimulus-bootstrap interval of the subject mean.}",
          r"\label{tab:multi28}", r"\end{table}"]
    return reps, "\n".join(t)


# ── step 48 amplitude, step 48b dominance ───────────────────────────────────
A48 = "amp48:step48/amplitude.csv"
DS = {"things720": "Tsev", "things_test": "Test", "nsd515": "Nsd"}


def amp_specs():
    for ds, dn in DS.items():
        convs = CONV_T if ds != "nsd515" else CONV_N
        rois = ("V1", "LOC") if ds != "nsd515" else ("V1", "LO-obj")
        ks = {"things720": (1,), "things_test": (12,), "nsd515": (1, 3)}[ds]
        for cv, cn in convs.items():
            for roi in rois:
                for k in ks:
                    for st in ("L", "C", "CE", "DD"):
                        q = (f"kind == 'brain_gain_stat' and dataset == '{ds}' and convention == '{cv}' and roi == '{roi}' and k == {k} "
                             f"and stat == '{st}'")
                        S(nm("Amp", dn, cn, ROIN[roi], "K", k, st), "avg:" + A48[6:], q, "rho", "s2")
                    for c, cc in (("Random", "Rand"), ("BP", "Bp")):
                        q = (f"kind == 'brain_model_gain' and dataset == '{ds}' and convention == '{cv}' and roi == '{roi}' and k == {k} "
                             f"and condition == '{c}'")
                        S(nm("AmpGm", dn, cn, ROIN[roi], "K", k, cc), "avg:" + A48[6:], q, "rho", "s3")
        for layer in (("Conv1", "Conv3") if ds == "things720" else ("Conv1",)):
            for c, cc in (("Random", "Rand"), ("BP", "Bp")):
                for st in ("L", "C", "CE", "DD"):
                    S(nm("AmpModel", dn, layer, cc, st), "avg:" + A48[6:],
                      f"kind == 'model_gain_stat' and dataset == '{ds}' and layer == '{layer}' and condition == '{c}' and stat == '{st}'",
                      "rho", "s2")
    for s in SUBJ3:
        for st in ("CE", "L"):
            S(nm("Amp", "Tsev", "Tglobal", "Vone", SUBN[s], st), A48[6:],
              f"kind == 'brain_gain_stat' and dataset == 'things720' and convention == 'T-global' and roi == 'V1' and subject == '{s}' "
              f"and stat == '{st}'", "rho", "s2")
        for c, cc in (("Random", "Rand"), ("BP", "Bp")):
            S(nm("AmpGm", "Tsev", "Tglobal", "Vone", SUBN[s], cc), "avg:" + A48[6:],
              f"kind == 'brain_model_gain' and dataset == 'things720' and convention == 'T-global' and roi == 'V1' and subject == '{s}' "
              f"and condition == '{c}'", "rho", "s3")
    for s in SUBJ8:
        for c, cc in (("Random", "Rand"), ("BP", "Bp")):
            S(nm("AmpGm", "Nsd", "Nraw", "Vone", "K", 1, SUBN[s], cc), "avg:" + A48[6:],
              f"kind == 'brain_model_gain' and dataset == 'nsd515' and convention == 'N-raw' and roi == 'V1' and k == 1 and subject == '{s}' "
              f"and condition == '{c}'", "rho", "s3")
    S("RuleGThree", "step48/summary.csv", "rule == 'G3'", "value", "int")
    S("RuleGTwo", "step48/summary.csv", "rule == 'G2'", "value", "int")
    S("RuleGOne", "step48/summary.csv", "rule == 'G1'", "value", "int")
    S("ImgSpearLDD", "step48/image_stats_summary.csv", "kind == 'image_stats_spearman' and dataset == 'things720' and stat == 'L-DD'",
      "value", "s2")
    S("ImgSpearLDDNsd", "step48/image_stats_summary.csv", "kind == 'image_stats_spearman' and dataset == 'nsd515' and stat == 'L-DD'",
      "value", "s2")
    S("ImgSpearCCE", "step48/image_stats_summary.csv", "kind == 'image_stats_spearman' and dataset == 'things720' and stat == 'C-CE'",
      "value", "s2")
    D = "step48b/dominance.csv"
    for ds, dn in DS.items():
        for layer in (("Conv1", "Conv3") if ds == "things720" else ("Conv1",)):
            for form in ("sum", "diff"):
                for c, cc in (("Random", "Rand"), ("BP", "Bp")):
                    q = f"kind == 'model_dominance' and dataset == '{ds}' and layer == '{layer}' and form == '{form}' and condition == '{c}'"
                    S(nm("Dom", dn, layer, form, cc), "avg:" + D, q, "rho", "s3")
                    for si in range(5):
                        S(nm("Dom", dn, layer, form, cc, "Seed", si), D, q + f" and seed == {si}", "rho", "s3")
                S(nm("DomMr", dn, layer, form), D,
                  f"kind == 'dominance_vs_moment_ratio' and dataset == '{ds}' and layer == '{layer}' and form == '{form}'", "rho", "s2")
            for c, cc in (("Random", "Rand"), ("BP", "Bp")):
                S(nm("Mratio", dn, layer, cc), "avg:" + D,
                  f"kind == 'moment_ratio' and dataset == '{ds}' and layer == '{layer}' and condition == '{c}'", "value", "f2")
        convs = CONV_T if ds != "nsd515" else CONV_N
        rois = ("V1", "LOC") if ds != "nsd515" else ("V1", "LO-obj", "lateral")
        ks = {"things720": (1,), "things_test": (1, 12), "nsd515": (1, 3)}[ds]
        for cv, cn in convs.items():
            for roi in rois:
                for k in ks:
                    for form in ("sum", "diff"):
                        S(nm("BDom", dn, cn, ROIN[roi], "K", k, form), "avg:" + D,
                          f"kind == 'brain_dominance_subject' and dataset == '{ds}' and convention == '{cv}' and roi == '{roi}' and k == {k} "
                          f"and form == '{form}'", "rho", "s2")
                    for form in ("none", "sum", "diff"):
                        q = (f"kind == 'delta' and dataset == '{ds}' and convention == '{cv}' and roi == '{roi}' and k == {k} "
                             f"and form == '{form}'")
                        base = nm("Pb", dn, cn, ROIN[roi], "K", k, form)
                        S(base, "step48b/partial.csv", q, "delta", "s3")
                        S(base + "Pos", "step48b/partial.csv", q, "n_positive", "int")
                        S(base + "Le", "step48b/partial.csv", q, "n_le_0020", "int")
    for s in SUBJ8:
        S(nm("Pb", "Nsd", "Nraw", "Vone", "K", 1, "sum", SUBN[s]), "step48b/partial.csv",
          f"kind == 'subject_delta' and dataset == 'nsd515' and convention == 'N-raw' and roi == 'V1' and k == 1 and form == 'sum' "
          f"and subject == '{s}'", "delta", "s3")
        for cv, cn in (("N-raw", "Nraw"), ("N-session", "Nsession")):
            S(nm("BDom", "Nsd", cn, "Vone", SUBN[s]), "avg:" + D,
              f"kind == 'brain_dominance' and dataset == 'nsd515' and convention == '{cv}' and roi == 'V1' and k == 1 and form == 'sum' "
              f"and subject == '{s}'", "rho", "s2")
    for s in SUBJ3:
        for cv, cn in (("T-raw", "Traw"), ("T-session", "Tsession")):
            S(nm("BDom", "Tsev", cn, "Vone", SUBN[s]), D,
              f"kind == 'brain_dominance_subject' and dataset == 'things720' and convention == '{cv}' and roi == 'V1' and k == 1 "
              f"and form == 'sum' and subject == '{s}'", "rho", "s2")
    S("RuleDThreeLe", "step48b/partial.csv",
      "kind == 'delta' and dataset == 'nsd515' and convention == 'N-raw' and roi == 'V1' and k == 1 and form == 'sum'", "n_le_0020", "int")
    S("RuleDTwoNsd", "step48b/summary.csv", "rule == 'D2'", "value", "int")


def amp_tables():
    out = {}
    t = [r"\begin{table}[H]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lllrrrrrrr@{}}", r"\toprule",
         r"Data & ROI & $k$ & Convention & $\Delta$ & ($n > 0$) & two-sided partial, sum form & ($n > 0$) & ($n \leq 0.020$) & difference form \\",
         r"\midrule"]
    for ds, dn, lab in (("things720", "Tsev", "THINGS-720"), ("things_test", "Test", "THINGS test"), ("nsd515", "Nsd", "NSD")):
        convs = CONV_T if ds != "nsd515" else CONV_N
        rois = ("V1", "LOC") if ds != "nsd515" else ("V1", "LO-obj", "lateral")
        ks = {"things720": (1,), "things_test": (1, 12), "nsd515": (1, 3)}[ds]
        for roi in rois:
            for k in ks:
                for cv, cn in convs.items():
                    b = lambda f: nm("Pb", dn, cn, ROIN[roi], "K", k, f)  # noqa: E731
                    t.append(f"{lab} & {roi} & {k} & {cv} & ${M(b('none'))}$ & ({M(b('none') + 'Pos')}) & ${M(b('sum'))}$ & "
                             f"({M(b('sum') + 'Pos')}) & ({M(b('sum') + 'Le')}) & ${M(b('diff'))}$ \\\\")
        t.append(r"\midrule")
    t = t[:-1]
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{$\Delta$ (paper cell) before and after the two-sided partial (step 48b): the brain RDM is rank-residualised on its amplitude RDM $A$, the model RDM on its own $A^M$, and the residuals are correlated. Sum form $A_{ij} = 2 - g_i - g_j$ (the form of D1--D3); difference form $|g_i - g_j|$, reported for comparison.}",
          r"\label{tab:partial48b}", r"\end{table}"]
    out["amp_partial"] = "\n".join(t)
    t = [r"\begin{table}[H]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
         r"Data & Convention & \multicolumn{4}{c}{brain $g$: Spearman with} & \multicolumn{2}{c}{$\rho(g, g^M)$} & \multicolumn{2}{c}{brain dominance} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r" & & $L$ & $C$ & CE & DD & untrained & Backprop & sum & diff. \\", r"\midrule"]
    for ds, dn, lab, k in (("things720", "Tsev", "THINGS-720 V1", 1), ("things_test", "Test", "THINGS test V1, $k = 12$", 12),
                           ("nsd515", "Nsd", "NSD V1, $k = 1$", 1), ("nsd515", "Nsd", "NSD V1, $k = 3$", 3)):
        convs = CONV_T if ds != "nsd515" else CONV_N
        for cv, cn in convs.items():
            stats = " & ".join(f"${M(nm('Amp', dn, cn, 'Vone', 'K', k, st))}$" for st in ("L", "C", "CE", "DD"))
            gm = " & ".join(f"${M(nm('AmpGm', dn, cn, 'Vone', 'K', k, cc))}$" for cc in ("Rand", "Bp"))
            bd = " & ".join(f"${M(nm('BDom', dn, cn, 'Vone', 'K', k, f))}$" for f in ("sum", "diff"))
            t.append(f"{lab} & {cv} & {stats} & {gm} & {bd} \\\\")
        t.append(r"\addlinespace")
    t = t[:-1]
    t += [r"\midrule", r"Model $g^M$ & & \multicolumn{4}{c}{Spearman with $L$, $C$, CE, DD} & \multicolumn{2}{c}{model dominance} & \multicolumn{2}{c}{$\rho$(dominance, $m$)} \\",
          r"\midrule"]
    for ds, dn, lab in (("things720", "Tsev", "THINGS-720"), ("things_test", "Test", "THINGS test"), ("nsd515", "Nsd", "NSD")):
        for cc, cl in (("Rand", "untrained"), ("Bp", "Backprop")):
            stats = " & ".join(f"${M(nm('AmpModel', dn, 'Conv1', cc, st))}$" for st in ("L", "C", "CE", "DD"))
            dom = f"${M(nm('Dom', dn, 'Conv1', 'sum', cc))}$ & ${M(nm('Dom', dn, 'Conv1', 'diff', cc))}$"
            mr = (f"${M(nm('DomMr', dn, 'Conv1', 'sum'))}$ & ${M(nm('DomMr', dn, 'Conv1', 'diff'))}$" if cc == "Rand" else " & ")
            t.append(f"{lab}, Conv1 & {cl} & {stats} & {dom} & {mr} \\\\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Response amplitude (steps 48, 48b). Brain $g$: cosine of each pattern with the mean pattern; Spearman with the image statistics (mean over subjects) and with the model $g^M$ (mean over seeds and subjects); brain dominance: Spearman(brain RDM, $A$) in the sum and the difference form. Model rows: Spearman of $g^M$ with the image statistics, model dominance Spearman(model RDM, $A^M$) (mean over seeds), and its Spearman with the moment ratio $m$ across the 10 checkpoints. DD is close to a mirror of $L$ (Section~\ref{sec:ampmethods}).}",
          r"\label{tab:amplitude}", r"\end{table}"]
    out["amp_stats"] = "\n".join(t)
    t = [r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Data & Model & seed 0 & 1 & 2 & 3 & 4 & mean \\", r"\midrule"]
    for ds, dn, lab in (("things720", "Tsev", "THINGS-720"), ("things_test", "Test", "THINGS test"), ("nsd515", "Nsd", "NSD")):
        for cc, cl in (("Rand", "untrained"), ("Bp", "Backprop")):
            t.append(f"{lab} & {cl} & " + " & ".join(f"${M(nm('Dom', dn, 'Conv1', 'sum', cc, 'Seed', si))}$" for si in range(5))
                     + f" & ${M(nm('Dom', dn, 'Conv1', 'sum', cc))}$ \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\caption{D1 (step 48b): Spearman(model RDM, $A^M$) per seed, sum form, Conv1.}",
          r"\label{tab:dom}", r"\end{table}"]
    out["amp_dom"] = "\n".join(t)
    return out


# ── evidence table (Discussion) ─────────────────────────────────────────────
def evidence_table():
    def c(m, pos, n):
        return f"${M(m)}$ ({M(pos)}/{n})"
    rows = [
        (r"Ridge, $g = 4$ (pre-specified; RD1)", r"$\PreregRandBpGFour$ (\PreregRandBpGFourPos/5 seeds $> 0$)", "--",
         r"$\NsdRidgeVoneGFour$ (\NsdRidgeVoneGFourNeg/8 $< 0$)"),
        ("RSA, paper cell, standard convention (T-global, N-session)", c("CvTsevTglobalVoneRawCorr", "CvTsevTglobalVoneRawCorrPos", 3),
         c("CvTtestTglobalVoneRawCorrKOne", "CvTtestTglobalVoneRawCorrKOnePos", 3), c("CvNsdNsessionVoneRawCorrKOne", "CvNsdNsessionVoneRawCorrKOnePos", 8)),
        ("RSA, paper cell, raw betas (T-raw, N-raw)", c("CvTsevTrawVoneRawCorr", "CvTsevTrawVoneRawCorrPos", 3),
         c("CvTtestTrawVoneRawCorrKOne", "CvTtestTrawVoneRawCorrKOnePos", 3), c("CvNsdNrawVoneRawCorrKOne", "CvNsdNrawVoneRawCorrKOnePos", 8)),
        ("RSA, paper cell, session-normalised (T-session, N-session)", c("CvTsevTsessionVoneRawCorr", "CvTsevTsessionVoneRawCorrPos", 3),
         c("CvTtestTsessionVoneRawCorrKOne", "CvTtestTsessionVoneRawCorrKOnePos", 3), c("CvNsdNsessionVoneRawCorrKOne", "CvNsdNsessionVoneRawCorrKOnePos", 8)),
        ("RSA, standardised units, raw betas", c("CvTsevTrawVoneZCorr", "CvTsevTrawVoneZCorrPos", 3),
         c("CvTtestTrawVoneZCorrKOne", "CvTtestTrawVoneZCorrKOnePos", 3), c("CvNsdNrawVoneZCorrKOne", "CvNsdNrawVoneZCorrKOnePos", 8)),
        ("RSA, raw units, Euclidean, standard convention", c("CvTsevTglobalVoneRawEucl", "CvTsevTglobalVoneRawEuclPos", 3),
         c("CvTtestTglobalVoneRawEuclKOne", "CvTtestTglobalVoneRawEuclKOnePos", 3), c("CvNsdNsessionVoneRawEuclKOne", "CvNsdNsessionVoneRawEuclKOnePos", 8)),
        ("RSA, paper cell, raw betas, amplitude structure removed on both sides (D3)",
         c("PbTsevTrawVoneKOneSum", "PbTsevTrawVoneKOneSumPos", 3), c("PbTestTrawVoneKOneSum", "PbTestTrawVoneKOneSumPos", 3),
         c("PbNsdNrawVoneKOneSum", "PbNsdNrawVoneKOneSumPos", 8)),
        ("Multiverse: distinct analyses above / below / including 0", r"\MultiDistinctPos{} / \MultiDistinctNeg{} / \MultiDistinctIncl", "--",
         r"N-session \MvNsessionAbove{} / \MvNsessionBelow{} / \MvNsessionIncl; N-raw \MvNrawAbove{} / \MvNrawBelow{} / \MvNrawIncl"),
    ]
    t = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.33\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}p{0.25\textwidth}@{}}",
         r"\toprule", r"Readout and convention (V1) & THINGS-720 & THINGS test images ($k = 1$) & NSD ($k = 1$; ridge $k = 3$) \\", r"\midrule"]
    for r in rows:
        t.append(" & ".join(r) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Evidence table: untrained minus Backprop at V1 under each readout and voxel normalisation convention, mean over seeds and subjects; in brackets the number of subjects with $\Delta > 0$ (ridge rows: the number of seeds or subjects with the sign stated in the cell). THINGS ridge is the pre-specified primary analysis (mean over the three subjects). The rows are not independent (same images, subjects and networks).}",
          r"\label{tab:evidence}", r"\end{table}"]
    return "\n".join(t)


def chronology_specs_table():
    """Follow-up amendments with commit and commit time (from the ledger, i.e. from git), plus the
    first data download after the NSD amendments (from the stored logs)."""
    S("FirstDownloadSubjTwo", "step42/fetch_subj02.csv", "kind == 'session' and session == 1", "download_started_utc", "utc")
    S("FirstDownloadRawFortyEight", "step48/nsd_raw_log.csv", "subject == 'subj01' and session == 1", "download_started_utc", "utc")
    led = pd.read_csv(REPO / "paper" / "PREDICTIONS_LEDGER.csv")
    fu = led[led.stage == "follow-up"].drop_duplicates("amendment_file")
    content = {"40": "banded ridge; cross-validated RSA on the repeated test images", "41": "diagnosis of B1; noise curve",
               "42": "NSD replication (P1--P4)", "42b": "norm and angle components, subj02--08 (Q1--Q4)",
               "43": "NSD multiverse, ImageNet models, resolution, noise ceiling", "44": "candidate explanations in THINGS (W1--W5)",
               "45": "coarse spatial luminance (W6, W7)", "47": "run centring in THINGS; ridge in NSD",
               "47b": "run-level mechanism; robustness of ridge and NSD RSA to run centring", "47c": "decomposition of run centring",
               "48": "voxel normalisation conventions; response amplitude", "48b": "amplitude dominance of the RDMs"}
    rows = []
    for _, r in fu.iterrows():
        step = re.search(r"step(\w+)-predictions", r.amendment_file).group(1)
        rows.append(f"{step} & \\texttt{{{r.amendment_commit}}} & {r.commit_time} & {content[step]} \\\\")
    t = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\begin{tabular}{@{}llll@{}}", r"\toprule",
         r"Step & Commit & Committed (local time) & Content \\", r"\midrule"] + rows + [
         r"\bottomrule", r"\end{tabular}",
         r"\caption{Amendments of the follow-up analyses, in commit order. Each amendment was committed before the scripts of its step; the commit timestamps, not the dates in the file names, are authoritative.}",
         r"\label{tab:chronology}", r"\end{table}"]
    return "\n".join(t)


def conv_full_table():
    """All cells and ROIs of step 48 (appendix)."""
    t = [r"\begin{table}[H]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}llllrrrr@{}}", r"\toprule",
         r"Data & ROI & $k$ & Convention & raw, corr. & z, corr. & raw, Eucl. & z, Eucl. \\", r"\midrule"]
    for dn, lab, rois, ks, convs, nsub in (("Tsev", "THINGS-720", ("V1", "LOC"), (None,), CONV_T, 3),
                                           ("Ttest", "THINGS test", ("V1", "LOC"), (1, 12), CONV_T, 3),
                                           ("Nsd", "NSD", ("V1", "LO-obj", "lateral"), (1, 3), CONV_N, 8)):
        for roi in rois:
            for k in ks:
                for cv, cn in convs.items():
                    cells = []
                    for cc in ("RawCorr", "ZCorr", "RawEucl", "ZEucl"):
                        b = nm("Cv", dn, cn, ROIN[roi], cc) if k is None else nm("Cv", dn, cn, ROIN[roi], cc, "K", k)
                        cells.append(f"${M(b)}$ ({M(b + 'Pos')}/{nsub})")
                    t.append(f"{lab} & {roi} & {'' if k is None else k} & {cv} & " + " & ".join(cells) + r" \\")
        t.append(r"\midrule")
    t = t[:-1] + [r"\bottomrule", r"\end{tabular}",
                  r"\caption{All cells of step 48: $\Delta$ (untrained minus Backprop) under the three conventions per data set, ROI and $k$; in brackets the number of subjects with $\Delta > 0$.}",
                  r"\label{tab:convfull}", r"\end{table}"]
    return "\n".join(t)


def build():
    """Register all v2 specs and write the generated tables; return the spec list."""
    SPECS.clear()
    _NAMES.clear()
    ledger_specs()
    conv_specs()
    why_specs()
    banded_specs()
    nsd_specs()
    amp_specs()
    tabs = {}
    tabs["conventions"] = conv_table()
    tabs.update(why_tables())
    tabs.update(banded_tables())
    tabs.update(nsd_tables())
    _, tabs["multi28"] = multiverse_specs_tables()
    tabs.update(amp_tables())
    tabs["evidence"] = evidence_table()
    tabs["chronology"] = chronology_specs_table()
    tabs["conv_full"] = conv_full_table()
    TABDIR.mkdir(parents=True, exist_ok=True)
    for k, v in tabs.items():
        if k != "evidence":
            v = v.replace(r"\begin{tabular}", r"\begin{adjustbox}{max width=\textwidth, max totalheight=0.82\textheight}" + "\n"
                          + r"\begin{tabular}").replace(r"\end{tabular}", r"\end{tabular}" + "\n" + r"\end{adjustbox}")
        (TABDIR / f"{k}.tex").write_text(f"% generated by scripts/tables_v2.py; every number is a macro of numbers.tex\n{v}\n",
                                         encoding="utf-8")
    return list(SPECS)


def amplitude48():
    """results/step48/amplitude.csv with the seeds of the THINGS rows reconstructed as in step 48b
    (rows in generation order, seeds 0-4 within each condition)."""
    a = pd.read_csv(RES / "step48" / "amplitude.csv", low_memory=False, float_precision="round_trip")
    for kind, other in (("brain_model_gain", ["dataset", "convention", "roi", "subject", "k", "condition"]),
                        ("model_gain_stat", ["dataset", "layer", "condition", "stat"])):
        sel = (a.kind == kind) & a.seed.isna()
        a.loc[sel, "seed"] = a[sel].fillna({"k": -1}).groupby(other, sort=False).cumcount().to_numpy()
    return a

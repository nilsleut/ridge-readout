"""
build_ledger.py
===============
Write paper/PREDICTIONS_LEDGER.csv: one row per pre-specified prediction of every amendment
(paper v1: PREREGISTRATION.md and steps 4-28; follow-up: steps 40-48b), one row per
interpretation rule (with its literal outcome) and per decision rule with labelled outcomes.

Columns: id, amendment_file, amendment_commit, commit_time, data, subjects, rule (wording of the
amendment), result, value, source_csv, stage (v1 / follow-up), kind (prediction / sanity /
interpretation / decision), note.

result: "met" / "not met" / "not interpreted" for predictions and sanity rules; the literal
outcome for interpretation and decision rules. Results and values are read from the stored CSVs
in results/ (never typed in); commit and time from git (the commit that added the amendment).

Counting (used by export_macros.py): predictions and sanity rules count as predictions; the
evaluated ones are those with result met or not met; "not interpreted" is counted separately.
Interpretation and decision rules are not counted.

Run: python scripts/build_ledger.py
"""
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = REPO / "paper" / "PREDICTIONS_LEDGER.csv"

_CACHE = {}


def csv(rel):
    if rel not in _CACHE:
        _CACHE[rel] = pd.read_csv(RES / rel, low_memory=False)
    return _CACHE[rel]


def commit_of(rel):
    out = subprocess.run(["git", "-C", str(REPO), "log", "--diff-filter=A", "--format=%h|%ad",
                          "--date=format:%Y-%m-%d %H:%M:%S", "--", rel], capture_output=True, text=True, check=True).stdout
    line = out.strip().splitlines()[-1]
    return line.split("|")


def num(v, nd=4):
    try:
        return f"{float(v):+.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def verdict(passed):
    vals = [str(p) for p in passed]
    if not vals or any(v not in ("True", "False") for v in vals):
        raise SystemExit(f"ledger: unexpected passed values {vals}")
    return "met" if all(v == "True" for v in vals) else "not met"


# ── getters: (result, value, source) ─────────────────────────────────────────
def kind_rule(rel, prefix, extra=None):
    """rows kind == 'rule' whose quantity starts with prefix (+ optional pandas filter)."""
    def get():
        d = csv(rel)
        r = d[(d.kind == "rule") & d.quantity.str.startswith(prefix)]
        if extra:
            r = r.query(extra, engine="python")
        if len(r) == 0:
            raise SystemExit(f"ledger: no rule row {prefix} in {rel}")
        vals = "; ".join(num(v) for v in r["value"]) if "value" in r else ""
        notes = "; ".join(str(n) for n in r["note"].dropna()) if "note" in r else ""
        return verdict(r.passed), (vals + (f" ({notes})" if notes else "")).strip(), rel
    return get


def summary_rule(rel, rid):
    def get():
        d = csv(rel)
        r = d[d.rule == rid]
        if len(r) != 1:
            raise SystemExit(f"ledger: {rid} matches {len(r)} rows in {rel}")
        r = r.iloc[0]
        return verdict([r.passed]), f"{num(r.value)} ({r.note})" if isinstance(r.note, str) else num(r.value), rel
    return get


def summary_interp(rel, quantity=None):
    def get():
        d = csv(rel)
        r = d[d.rule == "interpretation"]
        if quantity:
            r = r[r.quantity == quantity]
        if len(r) != 1:
            raise SystemExit(f"ledger: interpretation matches {len(r)} rows in {rel}")
        return str(r.note.iloc[0]), "", rel
    return get


def not_interpreted(rel, prefix=None, rid=None, why=""):
    def get():
        d = csv(rel)
        if rid:
            r = d[d.rule == rid]
        else:
            r = d[(d.kind == "rule") & d.quantity.str.startswith(prefix)]
        if len(r) == 0:
            raise SystemExit(f"ledger: no row for not-interpreted {prefix or rid} in {rel}")
        vals = "; ".join(num(v) for v in r["value"])
        return "not interpreted", f"{vals} (formally {'met' if all(str(p) == 'True' for p in r.passed) else 'not met'}; {why})", rel
    return get


def literal(text, rel, value_fn=None):
    def get():
        csv(rel)
        return text, (value_fn() if value_fn else ""), rel
    return get


def s1_decision():
    d = csv("step4/std_rsa.csv")
    r = d[(d.kind == "difference") & (d.variant == "z") & (d.quantity == "Random Weights - Backprop")]
    v = float(r["mean"].iloc[0])
    lab = "H_scale supported" if v <= 0.010 else ("H_scale not supported" if v >= 0.020 else "ambiguous")
    return lab, f"D_std {num(v)}", "step4/std_rsa.csv"


def s1_control():
    d = csv("step4/std_rsa.csv")
    r = d[(d.kind == "difference") & (d.variant == "z") & (d.quantity == "Random Weights - Random Weights (BN-calibrated)")]
    v = float(r["mean"].iloc[0])
    return ("met" if abs(v) <= 0.010 else "not met"), f"|Random - BN-cal| {abs(v):.4f}", "step4/std_rsa.csv"


def s2():
    d = csv("step4/subjects.csv")
    q = d[(d.kind == "difference") & (d.quantity == "Random Weights - Backprop")].set_index(["subject", "metric"])["mean"]
    ok = all(q[(s, "rsa_raw")] > 0 and q[(s, "ridge")] < 0 for s in ("sub-02", "sub-03"))
    val = "; ".join(f"{s}: RSA raw {num(q[(s, 'rsa_raw')])}, ridge {num(q[(s, 'ridge')])}" for s in ("sub-02", "sub-03"))
    return ("met" if ok else "not met"), val, "step4/subjects.csv"


def prereg_verdict():
    d = csv("prereg/signflip.csv")
    r = d[d.kind == "verdict"]
    if len(r) != 1:
        raise SystemExit("ledger: prereg verdict row")
    lab = str(r.quantity.iloc[0]).split(": ", 1)[1]
    prim = d[(d.kind == "contrast") & (d.role == "primary")]
    val = "; ".join(f"g = {int(g)}: {num(m)} (p {p:.2f})" for g, m, p in zip(prim.grid, prim["mean"], prim.p_one_sided))
    return f'"{lab[0].upper() + lab[1:]}."', val, "prereg/signflip.csv"


def p_ridge():
    d = csv("step6/ridge_resolution.csv")
    r = d[d.kind == "P_ridge"]
    return verdict(r.passed), f"{num(r['mean'].iloc[0])} ({r.note.iloc[0]})", "step6/ridge_resolution.csv"


def r2_decision():
    d = csv("step5/paper_comparisons_raw_vs_z.csv")
    r = d[(d.kind == "rule") & d.quantity.str.startswith("R2")].iloc[0]
    return str(r.note), f"{num(r.value)} [{num(r.ci_lo)}, {num(r.ci_hi)}]", "step5/paper_comparisons_raw_vs_z.csv"


def step41_interp():
    d = csv("step41/summary.csv")
    a = d[(d.kind == "rule") & d.quantity.str.startswith("D-A1")].passed.astype(str).iloc[0] == "True"
    b = d[(d.kind == "rule") & d.quantity.str.startswith("D-A2")].passed.astype(str).iloc[0] == "True"
    lab = ("the B1 violation of step 40 is attributed to selection variance of the two-dimensional search"
           if a and b else "not attributed (D-A1 and D-A2 not both met); the conditional estimator is not interpreted")
    return lab, "", "step41/summary.csv"


def step9_interp():
    d = csv("step9/separation.csv")
    r = d[d.kind == "rule"]
    s2_ = r[r.quantity.str.startswith("SEP2")].passed.astype(str).iloc[0] == "True"
    s3_ = r[r.quantity.str.startswith("SEP3")].passed.astype(str).iloc[0] == "True"
    lab = ("the V1 advantage of untrained networks under euclidean distance sits in the gain magnitude, and in geometry "
           "trained networks are ahead" if s2_ and s3_ else
           "Otherwise the V1 ranking remains unidentifiable, with no attribution to gain or geometry.")
    return lab, "", "step9/separation.csv"


def step25_interp():
    d = csv("step25/mean_profile.csv")
    r = d[d.kind == "rule"].set_index(d[d.kind == "rule"].quantity.str.split().str[0]).passed.astype(str) == "True"
    if r["M1"] and r["M2"] and r["M3"]:
        lab = "M1, M2 and M3 met: carried by the component parallel to the mean profile"
    elif not (r["M1"] and r["M2"]):
        lab = "M1 or M2 not met: the parallel component is not necessary."
    else:
        lab = "M3 not met: it is not sufficient."
    return lab, "", "step25/mean_profile.csv"


def step28_interp():
    d = csv("step28/ablation_2x2.csv")
    c = d[(d.kind == "cell") & (d.model == "own CNN")]
    full = float(c[(c.model_side == "full") & (c.brain_side == "full")].value.iloc[0])
    mixed = c[c.model_side != c.brain_side]
    above = mixed[(mixed.ci_lo > 0) & (mixed.value >= 0.5 * full)]
    incl = mixed[(mixed.ci_lo <= 0) & (mixed.ci_hi >= 0)]
    lab = ("the intervention on the other side removes the advantage" if len(above) == 1 and len(incl) == 1
           else "no attribution to a side (both mixed cells include 0)" if len(incl) == 2 else "no attribution to a side")
    val = "; ".join(f"model {a} / brain {b}: {num(v)} [{num(lo)}, {num(hi)}]"
                    for a, b, v, lo, hi in zip(mixed.model_side, mixed.brain_side, mixed.value, mixed.ci_lo, mixed.ci_hi))
    return lab, val, "step28/ablation_2x2.csv"


# ── the ledger ───────────────────────────────────────────────────────────────
T720 = "THINGS-fMRI, 720 single-presentation images (cross-run pairs)"
TALL = "THINGS-fMRI"
TTEST = "THINGS-fMRI, 100 test images x 12 presentations"
NSD = "NSD, 515 images seen 3 times"
S3 = "sub-01, sub-02, sub-03"
S8 = "subj01-subj08"
S7 = "subj02-subj08"

A = {4: "amendments/2026-10-03_step4-predictions.md", 5: "amendments/2026-10-08_step5-resolution-predictions.md",
     6: "amendments/2026-10-09_step6-predictions.md", 7: "amendments/2026-10-09_step7-offset-predictions.md",
     8: "amendments/2026-10-09_step8-generalization-predictions.md", 9: "amendments/2026-10-10_step9-separation-predictions.md",
     25: "amendments/2026-10-16_step25-predictions.md", 28: "amendments/2026-10-18_step28-predictions.md",
     40: "amendments/2026-11-16_step40-predictions.md", 41: "amendments/2026-11-16_step41-predictions.md",
     42: "amendments/2026-11-17_step42-predictions.md", "42b": "amendments/2026-11-17_step42b-predictions.md",
     43: "amendments/2026-11-17_step43-predictions.md", 44: "amendments/2026-11-19_step44-predictions.md",
     45: "amendments/2026-11-21_step45-predictions.md", 47: "amendments/2026-11-21_step47-predictions.md",
     "47b": "amendments/2026-11-21_step47b-predictions.md", "47c": "amendments/2026-11-22_step47c-predictions.md",
     48: "amendments/2026-11-22_step48-predictions.md", "48b": "amendments/2026-11-22_step48b-predictions.md",
     "prereg": "PREREGISTRATION.md"}

ROWS = [
    # paper v1: pre-specified primary analysis
    ("primary", "prereg", TALL + " 720 (ridge, reliable V1 voxels)", S3, "decision",
     "3b: 'The effect disappears' requires Delta_min to be smaller than the effect size the paper reports; the effect counts as surviving only if it holds at g >= 4 (one-sided exact sign-flip test over the 5 seeds).",
     prereg_verdict),
    # step 4
    ("S1", 4, T720 + " (RSA, 501 reliable V1 voxels)", "sub-01", "decision",
     "RSA with channel-wise standardized features; D_std = Random - BP, mean over 5 seeds. D_std <= 0.010: H_scale supported. D_std >= 0.020: H_scale not supported. Otherwise: ambiguous, reported as such.",
     s1_decision),
    ("S1-control", 4, T720 + " (RSA, 501 reliable V1 voxels)", "sub-01", "prediction",
     "Additional control criterion: |Random - BN-cal| under standardized RSA <= 0.010.", s1_control),
    ("S2", 4, TALL + " 720 (reliable V1 voxels, g = 1, random folds)", "sub-02, sub-03", "prediction",
     "For sub-02 and sub-03 the pattern holds if Random - BP under raw RSA > 0 AND Random - BP under ridge < 0. Reported per subject separately.",
     s2),
    # step 5
    ("R1", 5, T720, S3, "sanity", "Reproduction (gate). raw, V1, NATIVE 224: Random - BP = +0.030 +/- 0.001, with CI bounds within +/- 0.002 of [0.018, 0.042]; raw, V1, NATIVE 32: -0.001 +/- 0.001. Otherwise STOP.",
     kind_rule("step5/paper_comparisons_raw_vs_z.csv", "R1")),
    ("R2", 5, T720, S3, "decision", "R2 (primary). z, V1, NATIVE 224, Random - BP. Upper CI bound <= 0.010: H_mom explains the headline. Lower CI bound >= 0.015: H_mom does not explain it. Otherwise ambiguous.",
     r2_decision),
    ("R3", 5, T720, S3, "prediction", "Resolution slope S = (Random - BP at 224) - (Random - BP at 32), for raw and z. S_z <= 0.5 * S_raw: the resolution effect at V1 is predominantly moment-driven.",
     kind_rule("step5/paper_comparisons_raw_vs_z.csv", "R3")),
    ("R4", 5, T720, S3, "prediction", "LOC dissociation (prediction: survives). z, LOC, NATIVE 224, BP - Random: if its CI excludes 0 and the value is > 0, the LOC advantage of BP survives standardization.",
     kind_rule("step5/paper_comparisons_raw_vs_z.csv", "R4")),
    ("R5", 5, T720, S3, "prediction", "UPSAMPLED 224 (content limited to 32 px). z, V1: upper CI bound of Random - BP <= 0.010.",
     kind_rule("step5/paper_comparisons_raw_vs_z.csv", "R5")),
    ("M1 (step 5)", 5, T720, S3, "prediction", "Mechanism index m = ||mu||^2 / (||mu||^2 + sum_k sigma_k^2): the Spearman correlation between m and d = RSA_raw - RSA_z (V1, subject-averaged), over all cells (checkpoint x resolution) of the conditions Random and BP, is >= 0.6.",
     kind_rule("step5/paper_comparisons_raw_vs_z.csv", "M1")),
    # step 6
    ("P_ridge", 6, TALL + " 720 (ridge, reliable V1 voxels, g = 1)", "sub-01, sub-02 (sub-03 separately)", "prediction",
     "Random - BP (mean over sub-01/02, mean over seeds) <= 0 at all six NATIVE resolutions; and S_ridge = (224) - (32) <= 0.5 * S_raw, with S_raw = +0.0311 from step 5.",
     p_ridge),
    ("P_mech1", 6, T720 + " (features)", "-", "prediction", "For BP, m(UPSAMPLED 224) < m(NATIVE 224) < m(NATIVE 32) in at least 4 of 5 seeds (V1 layer). For Random, the range of m over these three conditions is smaller than half the BP range (mean over seeds).",
     kind_rule("step6/mechanism_bandpass.csv", "P_mech1")),
    ("P_mech2", 6, T720 + " (band-pass inputs)", S3, "prediction", "m_BP(notch) > m_BP(only) in 5 of 5 seeds. Over all cells (Random and BP x seeds x {NATIVE 32, NATIVE 224, UPSAMPLED 224, only_0-14, notch_0-14}), Spearman(m, d) >= 0.6, with d = RSA_raw - RSA_z at V1, subject-averaged.",
     kind_rule("step6/mechanism_bandpass.csv", "P_mech2")),
    # step 7
    ("O1", 7, T720, S3, "sanity", "Mathematical control. For D = euclidean, the RDMs under F = raw and F = center are identical (max |Delta| < 1e-9); correspondingly for B = paper vs center720 at D = euclidean. A violation of O1 stops the run.",
     kind_rule("step7/multiverse.csv", "O1")),
    ("O2", 7, T720, S3, "prediction", "A CI of S above 0 together with S >= 0.5 * S_paper occurs only in cells where D in {correlation, cosine} AND F in {raw, scale} AND B = paper; in all other cells the CI of S contains 0 or S < 0.5 * S_paper. Met if at most 2 of the 36 cells violate the prediction.",
     kind_rule("step7/multiverse.csv", "O2")),
    ("O3", 7, T720, S3, "prediction", "BP - Random at LOC, NATIVE 224, is > 0 (point estimate) in at least 30 of the 36 cells and has a CI above 0 in at least 18.",
     kind_rule("step7/multiverse.csv", "O3")),
    ("G1 (step 7)", 7, T720, S3, "prediction", "Spearman(c_gain, raw RSA) >= 0.6 over the cells condition {Random, BP} x seed x resolution (n = 70), raw RSA at F = raw, D = correlation, B = paper, V1.",
     kind_rule("step7/gain.csv", "G1")),
    ("G2 (step 7)", 7, T720, S3, "prediction", "Partial RSA (each RDM rank-regressed on its own gain RDM) at F = raw, D = correlation, B = paper: upper CI bound of the V1 gap at NATIVE 224 <= 0.010; S_partial <= 0.5 * S_paper.",
     kind_rule("step7/partial_rsa.csv", "G2")),
    # step 8
    ("GEN1", 8, T720 + " (ImageNet models)", S3, "prediction", "Spearman(c_gain, raw RSA in cell raw/correlation/paper) over all cells model x {trained, untrained s0-s2} x R at V1 >= 0.6.",
     kind_rule("step8/generalization.csv", "GEN1")),
    ("GEN2", 8, T720 + " (ImageNet models)", S3, "prediction", "In at least 4 of 6 models, at R = 224, two of the 4 bootstrap cells have gap CIs that exclude 0 in opposite directions.",
     kind_rule("step8/generalization.csv", "GEN2")),
    ("GEN3", 8, T720 + " (ImageNet models)", S3, "prediction", "trained - untrained at LOC, R = 224, is > 0 with a CI above 0 in all 4 bootstrap cells, for at least 5 of 6 models.",
     kind_rule("step8/generalization.csv", "GEN3")),
    ("GEN4", 8, T720 + " (ImageNet models)", S3, "prediction", "For each qualifying model (S_gen in raw/correlation/paper with a CI excluding 0): |S_gen| in z/correlation/paper <= 0.5 * |S_gen| in raw/correlation/paper. Met if this holds for all qualifying models.",
     kind_rule("step8/generalization.csv", "GEN4")),
    # step 9
    ("SEP1", 9, T720, S3, "sanity", "(gate) The identity holds numerically (max relative error < 1e-8) for all models, conditions and the fMRI side. Otherwise STOP.",
     kind_rule("step9/separation.csv", "SEP1")),
    ("SEP2", 9, T720, S3, "prediction", "V1, norm RDM, F = center: contrast > 0 with CI above 0 in the own CNN and in at least 4 of 6 ImageNet models.",
     kind_rule("step9/separation.csv", "SEP2")),
    ("SEP3", 9, T720, S3, "prediction", "V1, angle RDM, F = center: contrast < 0 with CI below 0 in the own CNN and in at least 4 of 6 ImageNet models.",
     kind_rule("step9/separation.csv", "SEP3")),
    ("SEP4", 9, T720, S3, "prediction", "LOC, angle RDM, F = center: trained - untrained (own CNN: BP - Random) > 0 with CI above 0 in the own CNN and in at least 5 of 6 ImageNet models.",
     kind_rule("step9/separation.csv", "SEP4")),
    ("SEP interpretation", 9, T720, S3, "interpretation", "If SEP2 and SEP3 are both met, the V1 advantage of untrained networks under euclidean distance sits in the gain magnitude, and in geometry trained networks are ahead. Otherwise the V1 ranking remains unidentifiable, with no attribution to gain or geometry.",
     step9_interp),
    # step 25
    ("M1 (step 25)", 25, T720, S3, "prediction", "own CNN, ortho, V1 gap at NATIVE 224: upper CI bound <= 0.010.", kind_rule("step25/mean_profile.csv", "M1")),
    ("M2 (step 25)", 25, T720, S3, "prediction", "own CNN, ortho, slope S_ortho <= 0.5 * S_paper.", kind_rule("step25/mean_profile.csv", "M2")),
    ("M3 (step 25)", 25, T720, S3, "prediction", "own CNN, par, V1 gap at NATIVE 224 > 0 with CI above 0, and the par slope has a CI above 0.",
     kind_rule("step25/mean_profile.csv", "M3")),
    ("M4 (step 25)", 25, T720 + " (ImageNet models)", S3, "prediction", "ImageNet, ortho, V1 gap at 224 with upper CI bound <= 0.010 in at least 4 of 6 models.",
     kind_rule("step25/mean_profile.csv", "M4")),
    ("M interpretation", 25, T720, S3, "interpretation", "M1, M2 and M3 met: in these data the raw-RSA advantage and its slope are carried by the component parallel to the mean profile and absent from the orthogonal remainder. M1 or M2 not met: the parallel component is not necessary. M3 not met: it is not sufficient. No biological claim follows.",
     step25_interp),
    ("F-R1", 25, TALL + " 720 (ridge, reliable V1 voxels)", S3, "prediction", "untrained minus Backprop in median MSE skill is <= 0 at every g with at most 1 of 5 seeds positive at each g (mean over subjects).",
     kind_rule("step25/ridge_skill.csv", "F-R1")),
    ("F-R2", 25, TALL + " 720 (ridge, reliable V1 voxels)", S3, "prediction", "the same for per-fold r.", kind_rule("step25/ridge_skill.csv", "F-R2")),
    ("F-R3", 25, TALL + " 720 (ridge, reliable V1 voxels)", "sub-01, sub-02", "prediction", "at g = 1 and g = 4 the real pooled median r of both conditions exceeds the 97.5th percentile of its own null in sub-01 and sub-02.",
     kind_rule("step25/ridge_permnull.csv", "F-R3")),
    ("F-R4", 25, TALL + " 720 (ridge, reliable V1 voxels)", S3, "prediction", "for Backprop, mean over subjects and seeds, joint median r minus bank-alone median r is <= 0.010 at g = 1 and g = 4 (the network adds little beyond the bank).",
     kind_rule("step25/ridge_joint.csv", "F-R4")),
    ("BN1", 25, T720, S3, "prediction", "for all three cells the classification of the 95% CI relative to 0 (above, below, includes) is unchanged compared with the fixed-normalisation intervals of step 7.",
     kind_rule("step25/bootnorm.csv", "BN1")),
    # step 28
    ("X1 (step 28)", 28, T720, S3, "prediction", "the paired contrast gap[full, full] minus gap[orthogonal, orthogonal] has a CI above 0.",
     kind_rule("step28/ablation_2x2.csv", "X1")),
    ("X interpretation", 28, T720, S3, "interpretation", "No attribution to a side is claimed unless exactly one mixed cell has a CI above 0 with a point estimate of at least half of gap[full, full] while the other has a CI that includes 0; if so, we say that the intervention on the other side removes the advantage, and nothing more.",
     step28_interp),
    # step 40
    ("B1", 40, TALL + " 720 (banded ridge, reliable V1 voxels)", S3, "sanity", "B1 (sanity): joint banded is not below bank alone by more than 0.002 for any condition, seed and subject. Violation means an implementation error, not a result.",
     kind_rule("step40/banded_summary.csv", "B1")),
    ("B2", 40, TALL + " 720 (banded ridge, reliable V1 voxels)", S3, "prediction", "B2: U_BP > 0 in at least 4 of 5 seeds and mean U_BP > 0.",
     not_interpreted("step40/banded_summary.csv", prefix="B2", why="B1 not met")),
    ("B3", 40, TALL + " 720 (banded ridge, reliable V1 voxels)", S3, "prediction", "B3: U_BP - U_Random > 0 in at least 4 of 5 seed pairs and mean > 0.",
     not_interpreted("step40/banded_summary.csv", prefix="B3", why="B1 not met")),
    ("C1", 40, TTEST, S3, "prediction", "C1: raw model units, correlation distance vs b1: Random - BP > 0 (point estimate).",
     kind_rule("step40/cvrsa_summary.csv", "C1", "roi == 'V1'")),
    ("C2", 40, TTEST, S3, "prediction", "C2: standardised model units, correlation distance vs b1: Random - BP <= 0 (point estimate).",
     kind_rule("step40/cvrsa_summary.csv", "C2", "roi == 'V1'")),
    ("C3", 40, TTEST, S3, "prediction", "C3: Euclidean model distance: the sign of Random - BP is the same against b2 and against b3 (raw units and standardised units each).",
     kind_rule("step40/cvrsa_summary.csv", "C3", "roi == 'V1'")),
    # step 41
    ("D-A1", 41, TALL + " 720 (banded ridge, g = 4)", S3, "sanity", "D-A1 (code check on real data): the joint banded path with alpha_net fixed at 1e12 reproduces bank-alone median r within 1e-6 in every fit. Failure means an implementation error.",
     kind_rule("step41/summary.csv", "D-A1")),
    ("D-A2", 41, TALL + " 720 (banded ridge, g = 4)", S3, "prediction", "D-A2 (selection variance): with a conditional one-dimensional search, no fit lies below bank alone by more than 0.002.",
     kind_rule("step41/summary.csv", "D-A2")),
    ("D-A interpretation", 41, TALL + " 720 (banded ridge, g = 4)", S3, "interpretation", "If D-A1 and D-A2 hold, the B1 violation of step 40 is attributed to selection variance of the two-dimensional search. The unique contribution U under the conditional search is then reported as an exploratory, post hoc estimator; B2 and B3 are applied to it. The two-dimensional estimator of step 40 remains not interpreted.",
     step41_interp),
    ("B2/B3 (conditional U)", 41, TALL + " 720 (banded ridge, g = 4)", S3, "prediction", "B2 and B3 applied to the unique contribution U under the conditional search (only if D-A1 and D-A2 hold).",
     literal("not interpreted", "step41/summary.csv", lambda: "D-A2 not met; U_conditional reported only")),
    ("N1", 41, TTEST, S3, "prediction", "N1: raw units, correlation distance, V1: Delta(1) > Delta(12) (point estimates).", kind_rule("step41/summary.csv", "N1")),
    ("N2", 41, TTEST, S3, "prediction", "N2: same cell: Spearman correlation between k and Delta(k) over the six values of k is <= -0.8.", kind_rule("step41/summary.csv", "N2")),
    ("N3", 41, TTEST, S3, "prediction", "N3: same cell: Delta(1) > 0 (point estimate).", kind_rule("step41/summary.csv", "N3")),
    ("N4", 41, TTEST, S3, "prediction", "N4: standardised units, correlation distance, V1: Delta(k) <= 0 for every k (point estimates).", kind_rule("step41/summary.csv", "N4")),
]
for rid, rule in (("P1", "P1 (paper cell: raw, correlation, V1, k = 3): mean Delta <= 0 (point estimate)."),
                  ("P2", "P2 (standardised, correlation, V1, k = 3): Delta < 0 in at least 6 of 8 subjects, sign-flip p <= 0.05, and the bootstrap interval of the mean excludes 0."),
                  ("P3", "P3 (standardised, correlation, LO-obj, k = 3): Delta < 0 in at least 6 of 8 subjects and sign-flip p <= 0.05."),
                  ("P4", "P4 (raw, correlation, V1): mean Delta(k = 3) <= mean Delta(k = 1).")):
    ROWS.append((rid, 42, NSD, S8, "prediction", rule, summary_rule("step42/summary.csv", rid)))
for rid, rule in (("Q1", "Q1 (V1, raw units, Euclidean distance): Delta > 0 in at least 5 of 7 subjects and sign-flip p <= 0.05."),
                  ("Q2", "Q2 (V1, N component): Delta > 0 in at least 5 of 7 subjects and sign-flip p <= 0.05."),
                  ("Q3", "Q3 (V1, A component): Delta < 0 in at least 5 of 7 subjects and sign-flip p <= 0.05."),
                  ("Q4", "Q4 (LO-obj, A component): Delta < 0 in at least 5 of 7 subjects and sign-flip p <= 0.05.")):
    ROWS.append((rid, "42b", NSD, S7, "prediction", rule, summary_rule("step42/summary.csv", rid)))
for rid, rule in (("M1 (step 43)", "M1: the interval of the 8-subject mean lies above 0 in at most 2 of the distinct analyses."),
                  ("M2 (step 43)", "M2: the interval lies below 0 in at least half of the distinct analyses."),
                  ("I1", "I1 (V1, angle component A): Delta < 0 for at least 5 of 6 architectures (8-subject mean)."),
                  ("I2", "I2 (V1, paper cell: raw, correlation): Delta <= 0 for at least 4 of 6 architectures (8-subject mean).")):
    ROWS.append((rid, 43, NSD, S8, "prediction", rule, summary_rule("step43/summary.csv", rid.split()[0])))
D44 = T720 + "; THINGS test images; NSD"
for rid, rule in (("W1", "W1 (sampling): over 1000 random subsets of 100 of the 720 images (seed 20261119), Delta exceeds -0.0002 (the THINGS test-image value at k = 1) in at least 95 % of subsets."),
                  ("W2", "W2 (luminance): in THINGS-720, partialling L-RDM and C-RDM reduces Delta to <= +0.010."),
                  ("W3", "W3 (model property): Spearman between model RDM and L-RDM is larger for Random than for BP in 5 of 5 seeds, in THINGS-720, THINGS test images and NSD."),
                  ("W4", "W4 (data property): the V1 correlation with L-RDM, divided by the dataset's LOSO lower bound, is larger in THINGS-720 than in THINGS test images (k = 1) and than in NSD (k = 1)."),
                  ("W5", "W5 (order): partialling T-RDM changes Delta in THINGS-720 by less than 0.005 in absolute value.")):
    ROWS.append((rid, 44, D44, S3 + "; " + S8, "prediction", rule, summary_rule("step44/summary.csv", rid)))
ROWS.append(("W interpretation (step 44)", 44, D44, S3 + "; " + S8, "interpretation",
             "W2, W3 and W4 met -> the advantage is attributed to luminance dominance of the THINGS-720 V1 RDM combined with the untrained network's greater luminance sensitivity. W5 not met -> presentation order contributes. W1 not met -> the test-image difference is compatible with sampling variation of 100 images. Any other pattern -> reported as not identified.",
             summary_interp("step44/summary.csv")))
for rid, rule in (("W6", "W6: in THINGS-720 at V1, partialling L-RDM, C-RDM and LP-RDM together reduces Delta to <= +0.010."),
                  ("W7", "W7: in THINGS-720 at V1, the correlation of the V1 RDM with LP-RDM, divided by the LOSO lower bound, is larger than the same ratio in THINGS test images (k = 1) and in NSD (k = 1).")):
    ROWS.append((rid, 45, D44, S3 + "; " + S8, "prediction", rule, summary_rule("step45/summary.csv", rid)))
ROWS.append(("W interpretation (step 45)", 45, D44, S3 + "; " + S8, "interpretation",
             "W6 and W7 met, together with W3 and W4 of step 44 -> the THINGS-720 advantage is attributed to coarse luminance structure dominating its V1 RDM, combined with the untrained network's greater sensitivity to it. Otherwise -> not identified; the explained fraction is reported.",
             summary_interp("step45/summary.csv")))
for rid, rule in (("RC1", "RC1: with run centring, V1 Delta <= +0.010."),
                  ("RC2", "RC2: with run centring, rho(V1, L-RDM) / LOSO lower bound (bound recomputed on run-centred data) < 0.5."),
                  ("RC3", "RC3: the across-run variance of the run-mean luminance of the 720 images is larger for the sub-01/02 assignment than for the sub-03 assignment."),
                  ("RC4", "RC4: with run centring, the THINGS-720 V1 multiverse (36 cells, 28 distinct analyses, rule of the paper) has at most 2 distinct analyses with the interval above 0."),
                  ("CO1", "CO1: without run centring, partialling the predecessor-luminance RDM |L_prev(i) - L_prev(j)| (pairs without a predecessor excluded) changes V1 Delta by at least 0.005 in absolute value.")):
    ROWS.append((rid, 47, T720, S3, "prediction", rule, summary_rule("step47/summary.csv", rid)))
ROWS.append(("RC interpretation", 47, T720, S3, "interpretation",
             "RC1, RC2 and RC3 met -> the THINGS-720 advantage is attributed to run-level structure shared by subjects with the same run assignment, combined with the untrained network's greater luminance sensitivity. RC1 not met -> run structure is not the explanation; the explained fraction is reported. CO1 met -> carry-over from the preceding image contributes.",
             summary_interp("step47/summary.csv", "amendment rule (part A)")))
for rid, rule in (("RD1", "RD1: at g = 4, Random - BP < 0 in at least 6 of 8 subjects (seed means) and one-sided sign-flip p <= 0.05."),
                  ("RD2", "RD2: the same holds at g = 1 and at g = 8.")):
    ROWS.append((rid, 47, NSD + " (ridge, reliable V1 voxels)", S8, "prediction", rule, summary_rule("step47/summary.csv", rid)))
for rid, rule in (("X1 (step 47b)", "X1: Delta A > 0 in sub-01 and in sub-02."),
                  ("X2 (step 47b)", "X2: the observed Delta A of sub-01 and of sub-02 lies within the central 95 % of its null distribution."),
                  ("X3 (step 47b)", "X3: the brain run RDM correlates positively with the run-index RDM in at least 2 of the 3 subjects.")):
    ROWS.append((rid, "47b", T720 + " session (run means)", S3, "prediction", rule, summary_rule("step47b/summary.csv", rid.split()[0])))
ROWS.append(("X interpretation (step 47b)", "47b", T720 + " session (run means)", S3, "interpretation",
             "X1, X2 and X3 met -> the run-level part of the THINGS-720 advantage is attributed to drift across runs together with a chance alignment of the untrained network's run-mean features in the run assignment shared by sub-01 and sub-02. X1 met and X2 not met (Delta A above the 97.5th percentile) -> attributed to stimulus-driven run-level structure. Otherwise -> not identified.",
             summary_interp("step47b/summary.csv", "amendment rule (part X)")))
for rid, data, subj, rule in (
        ("Y1", TALL + " 720 (ridge)", S3, "Y1 (THINGS ridge, primary analysis): with run centring, g = 4, Random - BP < 0 in 5 of 5 seeds (mean over subjects)."),
        ("Y2", NSD, S8, "Y2 (NSD RSA): with run centring, V1 paper cell and V1 standardised cell: Delta < 0 in at least 6 of 8 subjects each."),
        ("Y3", NSD + " (ridge)", S8, "Y3 (NSD ridge): with run centring, g = 4, V1: Random - BP < 0 in at least 6 of 8 subjects."),
        ("Y4", NSD + " (ridge, all V1 voxels)", S8, "Y4 (NSD ridge, all voxels): without run centring, g = 4, all V1 voxels (no reliability selection): Random - BP < 0 in at least 6 of 8 subjects.")):
    ROWS.append((rid, "47b", data, subj, "prediction", rule, summary_rule("step47b/summary.csv", rid)))
for rid, rule in (("Z1", "Z1: V2 <= +0.015 (run deviations alone carry most of the effect)."),
                  ("Z2", "Z2: V4 >= +0.020 and V5 <= +0.012 (the run deviations act along the mean-profile axis)."),
                  ("Z3", "Z3: the across-run SD of the scalar d_r . m_hat is smallest in sub-03.")):
    ROWS.append((rid, "47c", T720, S3, "prediction", rule, summary_rule("step47c/summary.csv", rid)))
ROWS.append(("Z interpretation", "47c", T720, S3, "interpretation",
             "Z1 and Z2 met -> the run-level part of the advantage is attributed to run-wise shifts of global activation along the voxel mean-profile axis. Z1 not met and V1 <= +0.015 -> the effect of run centring is attributed mainly to removing the session mean profile (equivalent to voxel centring), not to between-run structure; the description of step 47 as \"run structure\" is then corrected in the manuscript. Any other pattern -> not identified; all variants are reported.",
             summary_interp("step47c/summary.csv")))
D48 = T720 + "; THINGS test images; NSD"
for rid, data, subj, rule in (
        ("NB1", NSD, S8, "NB1: NSD, V1 paper cell, k = 1: Delta(N-raw) > Delta(N-session) in at least 6 of 8 subjects."),
        ("NB2", NSD, S8, "NB2: NSD, V1 paper cell, k = 1: Delta(N-raw) > 0 in at least 6 of 8 subjects and sign_flip_p(direction positive) <= 0.05."),
        ("NB3", T720, S3, "NB3: THINGS-720, V1 paper cell: Delta(T-session) <= +0.010."),
        ("G1 (step 48)", T720, S3, "G1: THINGS-720, T-global, V1: |Spearman(g, CE)| > |Spearman(g, L)| in at least 2 of 3 subjects."),
        ("G2 (step 48)", T720, S3, "G2: THINGS-720, T-global, V1: Spearman(g, g^M Random) > Spearman(g, g^M BP) in all 3 subjects."),
        ("G3 (step 48)", NSD, S8, "G3: NSD, N-raw, k = 1, V1: Spearman(g, g^M Random) > Spearman(g, g^M BP) in at least 6 of 8 subjects.")):
    ROWS.append((rid, 48, data, subj, "prediction", rule, summary_rule("step48/summary.csv", rid.split()[0])))
ROWS.append(("NB/G interpretation", 48, D48, S3 + "; " + S8, "interpretation",
             "NB1, NB2, G2 and G3 met -> the untrained advantage is attributed to a common voxel response profile left in by the normalisation, through per-image response amplitude that the untrained network tracks better; in both datasets. NB1 met and NB2 not met -> the normalisation convention moves NSD in the predicted direction but does not create an untrained advantage there. NB1 not met -> the mechanism is not supported in NSD.",
             summary_interp("step48/summary.csv")))
for rid, data, subj, rule in (
        ("D1", D48, S3 + "; " + S8, "D1 (model side): Spearman(model RDM, A^M) is larger for Random than for BP in 5 of 5 seeds, in each of THINGS-720, THINGS test images and NSD-515."),
        ("D2", T720 + "; NSD", S3 + "; " + S8, "D2 (brain side): Spearman(brain RDM, A) is larger under T-raw than under T-session in each THINGS-720 subject, and larger under N-raw than under N-session (k = 1) in at least 7 of 8 NSD subjects."),
        ("D3", NSD, S8, "D3 (bridge): in NSD under N-raw (k = 1), the two-sided partial reduces Delta from +0.1165 to <= +0.020, with Delta <= +0.020 in at least 6 of 8 subjects.")):
    ROWS.append((rid, "48b", data, subj, "prediction", rule, summary_rule("step48b/summary.csv", rid)))
ROWS.append(("D interpretation", "48b", D48, S3 + "; " + S8, "interpretation",
             "D1, D2 and D3 met -> the untrained advantage under conventions that leave a common profile in both patterns is attributed to amplitude structure dominating both RDMs, with the untrained network's RDM more amplitude-dominated; not to a better match of per-image amplitude. Any other pattern -> not identified; all values are reported.",
             summary_interp("step48b/summary.csv")))

NOTES = {
    "NB/G interpretation": "The rule has no case for 'NB1 and NB2 met, G2 or G3 not met'; this pattern occurred and is reported literally as 'no statement of the rule applies' (amendment unchanged).",
    "RC interpretation": "RC1 met, RC2 and RC3 not met: none of the stated cases applies.",
    "primary": "Pre-specified primary analysis (plan dated 2026-08-20, placed under version control on 2026-10-03; criteria fixed before the analysis).",
    "S1": "Decision rule with labelled outcomes.",
    "R2": "Decision rule with labelled outcomes.",
    "C1": "V1 (primary); LOC descriptive.",
    "C2": "V1 (primary); LOC descriptive.",
    "C3": "V1 (primary); LOC descriptive.",
    "Q1": "subj01 excluded (it motivated Q1-Q4).",
}


def main():
    rows = []
    for rid, step, data, subj, kind, rule, get in ROWS:
        rel = A[step]
        h, t = commit_of(rel)
        result, value, src = get()
        stage = "v1" if step in ("prereg", 4, 5, 6, 7, 8, 9, 25, 28) else "follow-up"
        if kind in ("prediction", "sanity") and result not in ("met", "not met", "not interpreted"):
            raise SystemExit(f"ledger: {rid} has result {result}")
        rows.append(dict(id=rid, amendment_file=rel, amendment_commit=h, commit_time=t, data=data, subjects=subj, rule=rule,
                         result=result, value=value, source_csv=f"results/{src}", stage=stage, kind=kind,
                         note=NOTES.get(rid, "")))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pred = df[df.kind.isin(["prediction", "sanity"])]
    for st in ("v1", "follow-up"):
        p = pred[pred.stage == st]
        print(f"{st}: {len(p)} predictions; met {int((p.result == 'met').sum())}, not met {int((p.result == 'not met').sum())}, "
              f"not interpreted {int((p.result == 'not interpreted').sum())}")
    print(f"interpretation / decision rows: {int((~df.kind.isin(['prediction', 'sanity'])).sum())}")
    print("not met:", ", ".join(pred[pred.result == "not met"].id))
    print(f"wrote {OUT.relative_to(REPO)} ({len(df)} rows)")


if __name__ == "__main__":
    main()

"""
prereg_primary.py
=================
The preregistered primary analysis (PREREGISTRATION.md §2, config hash 206fa875cca5) for
sub-01, sub-02 and sub-03, run through the existing driver `scripts/pilot_fit.py::main`
UNCHANGED. Fit logic, constants, seeds, grids and defaults are not touched; only module
globals are rebound, as a per-subject loop:

  pilot_fit.SUBJECT        -> each of sub-01, sub-02, sub-03 (data loading)
  pilot_fit.GRIDS          -> [1, 2, 4, 8]          (the preregistered ladder; config.yaml)
  pilot_fit.load_features  -> the original function, reading g = 2 from
                              data/prereg_features (extracted by scripts/prereg_extract.py
                              with the existing extraction code) and g = 1, 4, 8 from
                              RIDGE_DATA_DIR/pilot_features as before
  pilot_fit.REPO           -> a temporary directory (pilot_fit writes pilot_results.csv)
  pilot_fit.fit_condition  -> the real fit.fit_condition with assert_low_boundary=False,
                              i.e. the D1 plateau gate of
                              amendments/2026-10-03_d1-plateau-gate.md: low-edge selections
                              at reliable voxels are logged (not aborted) and classified
                              after the fit with scripts/lower_edge_gate.py; only
                              non-plateau ("invalid") cells are marked invalid.

Everything pilot_fit.main runs is run: the 30 model checkpoints, the control band
(pixels, luminance+contrast, Gabor, permuted assignment) and the per-image cell shuffle
(g >= 2, Random and Backprop seed 0). The r(df) path that fit_condition returns
(median r over reliable voxels per alpha, df per alpha) is recorded as an instrument.
Alpha grid, min-MSE selection (D2, tolerance-rule amendment), 8 x 5 CV with seed 20260818:
fit_condition defaults, unchanged.

Runtime estimate before start (fits x measured seconds x 1.5); above 8 h the primary fits
would run first (here everything runs in one pass per subject; the estimate is reported).

Output: results/prereg/primary.csv, instruments_control_band.csv,
        instruments_cell_shuffle.csv, instruments_rdf_path.csv, lower_edge.csv
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import subprocess                                       # noqa: E402
import sys                                              # noqa: E402
import tempfile                                         # noqa: E402
import time                                             # noqa: E402
from concurrent.futures import ProcessPoolExecutor      # noqa: E402
from pathlib import Path                                # noqa: E402

import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (REPO / "src" / "ridge", REPO / "src" / "vendor", HERE):
    sys.path.insert(0, str(p))

SUBJECTS = ["sub-01", "sub-02", "sub-03"]
GRIDS = [1, 2, 4, 8]
G2DIR = REPO / "data" / "prereg_features"
OUT = REPO / "results" / "prereg"


def run_subject(subject):
    import pilot_fit
    from config_hash import config_hash
    from fit import fit_condition as real_fit
    from lower_edge_gate import gate, hits_from_fit

    feat_orig = pilot_fit.FEAT
    load_orig = pilot_fit.load_features

    def load_features(name, g):
        pilot_fit.FEAT = G2DIR if g == 2 else feat_orig
        try:
            return load_orig(name, g)
        finally:
            pilot_fit.FEAT = feat_orig

    records, rdf, gates = [], [], []

    def recording_fit(X, Y, groups, *, alphas, reliable, context="", **kw):
        res = real_fit(X, Y, groups, alphas=alphas, reliable=reliable, assert_low_boundary=False,
                       context=context, **kw)
        label, g = context.rsplit(" g", 1)
        g = int(g)
        hits = hits_from_fit(res, reliable, condition=label, g=g, subject=subject)
        cls = "none"
        if hits:
            gd = gate(hits, load_X=lambda h: X, Y=Y)
            gd["context"] = context
            gates.append(gd)
            cls = "invalid" if (gd.classification == "invalid").any() else "plateau"
        r = res["r"]
        records.append(dict(subject=subject, label=label, grid=g, d=res["d"],
                            r_med_reliable=float(np.median(r[reliable])), r_med_all=float(np.median(r)),
                            n_reliable=int(reliable.sum()), n_low_edge_hits=len(hits), gate=cls,
                            valid=cls != "invalid",
                            **{k: float(np.mean(res[k])) for k in ("lo_rel", "hi_rel", "lo_unrel", "hi_unrel")}))
        med_path = np.median(res["r_path"][:, reliable], axis=1)
        for a, rr, df in zip(res["alphas"], med_path, res["df"]):
            rdf.append(dict(subject=subject, label=label, grid=g, alpha=float(a), df=float(df),
                            r_med_reliable=float(rr)))
        return res

    pilot_fit.SUBJECT = subject
    pilot_fit.GRIDS = list(GRIDS)
    pilot_fit.load_features = load_features
    pilot_fit.fit_condition = recording_fit
    with tempfile.TemporaryDirectory() as td:
        pilot_fit.REPO = Path(td)
        pilot_fit.main()
    g = pd.concat(gates, ignore_index=True) if gates else pd.DataFrame()
    return records, rdf, g, config_hash()


def main():
    missing = [f for f in [f"model_{k}_seed{s}_g2.npy" for k in
               ("random", "backprop", "feedback_alignment", "predictive_coding", "stdp", "random_bn_calibrated")
               for s in range(5)] + [f"control_{c}_g2.npy" for c in ("pixels", "lumcontrast", "gabor")]
               if not (G2DIR / f).exists()]
    if missing:
        raise SystemExit(f"STOP: g = 2 features missing: {missing[:3]} ...")
    n_fits = len(SUBJECTS) * (len(GRIDS) * (30 + 3 + 1) + 3 * 2)
    est_h = n_fits * 10.0 / 3 * 1.5 / 3600        # ~10 s per fit with 4 threads, 3 subjects in parallel
    print(f"runtime estimate: {n_fits} fits, {est_h:.2f} h", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        out = list(ex.map(run_subject, SUBJECTS))
    print(f"fits done ({time.time() - t0:.0f}s)", flush=True)

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    h = out[0][3]
    rec = pd.DataFrame([r for o in out for r in o[0]])
    rdf = pd.DataFrame([r for o in out for r in o[1]])
    gates = [o[2] for o in out if len(o[2])]
    gd = pd.concat(gates, ignore_index=True) if gates else pd.DataFrame(columns=["classification"])

    rec["kind"] = np.select([rec.label.str.startswith("control:"), rec.label.str.contains("CELL-SHUFFLED")],
                            ["control", "shuffle"], "model")
    model = rec[rec.kind == "model"].copy()
    model["rule"] = model.label.str.rsplit(" s", n=1).str[0]
    model["seed_idx"] = model.label.str.rsplit(" s", n=1).str[1].astype(int)
    OUT.mkdir(parents=True, exist_ok=True)
    for df, name in ((model, "primary"), (rec[rec.kind == "control"], "instruments_control_band"),
                     (rec[rec.kind == "shuffle"], "instruments_cell_shuffle"),
                     (rdf, "instruments_rdf_path"), (gd, "lower_edge")):
        df = df.copy()
        df["config_hash"], df["git_head"] = h, head
        df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"model fits {len(model)}, controls {int((rec.kind == 'control').sum())}, "
          f"shuffles {int((rec.kind == 'shuffle').sum())}; low-edge hits {len(gd)}; "
          f"classes {gd.classification.value_counts().to_dict() if len(gd) else {}}", flush=True)


if __name__ == "__main__":
    main()

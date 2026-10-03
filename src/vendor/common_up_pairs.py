# Vendored from evaluation-resolution-rsa: code/upsampling/common_up.py
# Source HEAD: 7a9b3742cf4d739400bd52d0d6eb8a53c1bd517c
# sha256 of the original: c20c9d46ee9ae778d566932ed377677d46bfe55b15f6affdec22c320f188419e
# Taken verbatim: constants SUBJECTS, N_STIM, TRI and the functions ranks, stim_order,
# run_labels, pair_set. Not taken: boot_idx and everything else.
# Local changes: FMRI_DIR and META_DIR resolve through src/ridge/paths.py at call time.
"""Cross-run pair set and the RSA statistic of arXiv:2608.12408 v3.

Pair set P (cross-run): stimulus pairs that lie in different runs for EVERY subject
(210,205 of 258,840 pairs for the identity sample).

RSA = Spearman on the strict upper triangle, as centred unit-norm average ranks, so
rho(a, b) == ranks(a) @ ranks(b).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ridge"))
import paths  # noqa: E402

SUBJECTS = ["sub-01", "sub-02", "sub-03"]
N_STIM = 720
TRI = np.triu_indices(N_STIM, 1)


def ranks(v):
    """Centred, unit-norm average ranks, so spearman(a, b) == ranks(a) @ ranks(b)."""
    r = rankdata(v)
    r -= r.mean()
    return r / np.linalg.norm(r)


def stim_order():
    with open(paths.outputs_720_dir() / "stim_order_sub-01.txt") as f:
        return [l.strip() for l in f if l.strip()]


def run_labels():
    """Run (within the session the 720 come from) of each of the 720 stimuli, per subject."""
    order = stim_order()
    out = {}
    for s in SUBJECTS:
        st = pd.read_csv(paths.things_fmri_dir() / f"{s}_task-things_stimulus-metadata.csv")
        m = st.set_index("stimulus").loc[order]
        assert m.session.nunique() == 1
        out[s] = m.run.values
    return out


def pair_set(stims=None, runs=None):
    """(rows, cols, keep) of the pairs in P for a stimulus sample; keep is a mask on TRI.
    Same logic as learning-rules-rsa common_cr.pair_set."""
    runs = runs or run_labels()
    s = np.arange(N_STIM) if stims is None else np.asarray(stims)
    a, b = s[TRI[0]], s[TRI[1]]
    keep = a != b
    for r in runs.values():
        keep &= r[a] != r[b]
    return a[keep], b[keep], keep

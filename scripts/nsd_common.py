"""
nsd_common.py
=============
Shared helpers for step 42 (amendments/2026-11-17_step42-predictions.md): NSD replication.

Every NSD file and everything derived from it at voxel or RDM level is stored under
paths.nsd_dir() (NSD_DIR, outside the repository). Nothing printed here contains betas,
patterns or RDM entries; only shapes, counts and summary statistics.

Every step 42 CSV carries config_hash (of config_step42.yaml), git_head and seed.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np          # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "ridge"))
sys.path.insert(0, str(REPO / "src" / "vendor"))
sys.path.insert(0, str(REPO / "scripts"))

import paths                                            # noqa: E402
from config_hash import config_hash, load_config        # noqa: E402

CONFIG = REPO / "config_step42.yaml"
OUT = REPO / "results" / "step42"
LAB = {"random": "Random", "backprop": "BP"}


def cfg():
    return load_config(CONFIG)


def cfg_hash():
    return config_hash(cfg())


def git_head():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def stamp(df):
    df["config_hash"], df["git_head"] = cfg_hash(), git_head()
    if "seed" not in df:
        df["seed"] = np.nan
    df["seed"] = df["seed"].astype("Int64")
    return df


def nsd_dir(*parts) -> Path:
    d = paths.nsd_dir().joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── bucket ──────────────────────────────────────────────────────────────────
def s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    b = cfg()["bucket"]
    return boto3.client("s3", region_name=b["region"], config=Config(signature_version=UNSIGNED))


def bucket():
    return cfg()["bucket"]["name"]


def object_size(key):
    return int(s3().head_object(Bucket=bucket(), Key=key)["ContentLength"])


def download(key, dst: Path):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    s3().download_file(bucket(), key, str(tmp))
    tmp.replace(dst)
    return dst


def free_gb(path=None) -> float:
    return shutil.disk_usage(str(path or paths.nsd_dir())).free / 1e9


def require_free(extra_gb=0.0):
    need = cfg()["disk"]["min_free_gb"] + extra_gb
    f = free_gb()
    if f < need:
        raise SystemExit(f"STOP: {f:.1f} GB free, need at least {need:.1f} GB")
    return f


# ── design ──────────────────────────────────────────────────────────────────
def design_file():
    p = nsd_dir("design") / "nsd_expdesign.mat"
    if not p.exists():
        download(cfg()["design"]["file"], p)
    return p


def design(subject, mat=None):
    """Trials of the shared images shown three times to every subject.

    Returns dict with
      images   (n,)   73k IDs (1-based), sorted
      shared   (n,)   position in shared1000 (1-based), for the image file name
      trial    (n, 3) chronological trial index (0-based) of presentations 1, 2, 3
      session  (n, 3) session (1-based)
      pos      (n, 3) index within the session's 750 trials (0-based)
      run      (n, 3) run within the session (0-based)
    """
    import scipy.io as sio
    c = cfg()["design"]
    d = sio.loadmat(mat or design_file())
    mo = d["masterordering"].ravel().astype(np.int64) - 1          # index into the 10k
    sim = d["subjectim"].astype(np.int64)                          # 73k IDs, 1-based
    nses = c["sessions_completed"]
    tps = c["trials_per_session"]
    ok = np.ones(1000, bool)
    for n in nses:
        seq = mo[: n * tps]
        ok &= np.bincount(seq[seq < 1000], minlength=1000) == 3
    shared10k = np.where(ok)[0]                                    # positions in the 10k (< 1000)
    if len(shared10k) != c["n_images"]:
        raise SystemExit(f"STOP: {len(shared10k)} images seen 3x by all subjects, expected {c['n_images']}")
    s = int(subject[-2:]) - 1
    images = sim[s, shared10k]
    if not np.all(np.diff(images) > 0):
        raise SystemExit("STOP: shared image IDs not sorted")
    seq = mo[: nses[s] * tps]
    trial = np.stack([np.where(seq == k)[0] for k in shared10k])   # chronological, (n, 3)
    session = trial // tps + 1
    pos = trial % tps
    sp = d["stimpattern"].astype(bool)                             # (40, 12, 75)
    run_of = np.concatenate([np.full(sp[0, r].sum(), r) for r in range(sp.shape[1])])
    if len(run_of) != tps or any(np.concatenate([np.full(sp[q, r].sum(), r) for r in range(12)]).tolist()
                                 != run_of.tolist() for q in range(sp.shape[0])):
        raise SystemExit("STOP: stimpattern differs between sessions")
    run = run_of[pos]
    lo, hi = c["sessions_loaded"]
    if session.min() < lo or session.max() > hi:
        raise SystemExit(f"STOP: trials outside sessions {lo}-{hi}")
    sharedix = d["sharedix"].ravel().astype(np.int64)              # 73k IDs of shared1000, sorted
    shared = np.searchsorted(sharedix, images) + 1
    assert np.array_equal(sharedix[shared - 1], images)
    return dict(images=images, shared=shared, trial=trial, session=session, pos=pos, run=run)


# ── ROIs ────────────────────────────────────────────────────────────────────
def roi_specs():
    return {k: v for k, v in cfg()["rois"].items() if isinstance(v, dict)}


def roi_dir(subject):
    return nsd_dir("roi", subject)


def roi_files(subject):
    """Download the ROI files of one subject (if missing); return their local paths."""
    c = roi_specs()
    names = {Path(v["file"]).name for v in c.values()} | {c["LO-obj"]["and_tval"]}
    out = {}
    for n in sorted(names):
        key = (f"nsddata/ppdata/{subject}/func1pt8mm/roi/{n}" if not n.startswith("floc_")
               else f"nsddata/ppdata/{subject}/func1pt8mm/{n}")
        p = roi_dir(subject) / n
        if not p.exists():
            download(key, p)
        out[n] = p
    return out


def roi_masks(subject, ref_shape=None, ref_affine=None):
    """Boolean volume masks {V1, LO-obj, lateral}; grid check against the betas if given."""
    import nibabel as nib
    c = roi_specs()
    f = roi_files(subject)
    vols = {}
    for n, p in f.items():
        img = nib.load(str(p))
        if ref_shape is not None:
            if tuple(img.shape[:3]) != tuple(ref_shape[:3]) or not np.allclose(img.affine, ref_affine, atol=1e-4):
                raise SystemExit(f"STOP: {n} of {subject} is not on the beta grid "
                                 f"(shape {img.shape[:3]} vs {tuple(ref_shape[:3])})")
        vols[n] = np.asarray(img.dataobj)
    out = {}
    for roi, spec in c.items():
        v = vols[Path(spec["file"]).name]
        m = np.isin(np.rint(v).astype(np.int64), spec["labels"])
        if "and_tval" in spec:
            t = np.nan_to_num(vols[spec["and_tval"]].astype(np.float64), nan=-np.inf)
            m &= t > spec["tval_threshold"]
        out[roi] = m
    return out

"""
nsd_runmeans.py
===============
Step 47b (amendments/2026-11-21_step47b-predictions.md, config_step47b.yaml): run means of all
NSD trials, sessions 1-30 of all 8 subjects, for run centring.

Per session, as nsd_fetch.py (imported: beta_key, process; nsd_common: download, require_free):
  1. download betas_sessionBB.nii.gz to NSD_DIR/tmp (next file downloaded while the current one
     is processed; at most two on disk);
  2. nsd_fetch.process with all 750 trial positions: int16 / 300, z per voxel over the 750
     trials (ddof 0), on the V1 | lateral voxels of NSD_DIR/roi/<subject>/union.npz;
  3. gate: the trials of the 515 images equal the stored step 42 NSD_DIR/betas/<subject>/
     sessionBB.npz exactly (Z and valid; STOP otherwise);
  4. voxel mean over all trials of each of the 12 runs (stimpattern; checked against
     nsd_common.design for the image trials), stored as float32 (12, n_voxels) under
     NSD_DIR/runmeans/<subject>/sessionBB.npz (run means only);
  5. the downloaded file is deleted.
Run centring (centred_patterns): every stored presentation minus the mean of its (session, run).

Printed and logged: only shapes, counts, sizes, durations. Log in the repository:
results/step47b/nsd_runmeans_log.csv.

Usage:
  python scripts/nsd_runmeans.py gate            subj01 session 1 only (the setup gate)
  python scripts/nsd_runmeans.py estimate
  python scripts/nsd_runmeans.py all             all subjects, all sessions (resumes)
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads)

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config_hash import config_hash, load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config_step47b.yaml"
OUT = REPO / "results" / "step47b"
LOG = OUT / "nsd_runmeans_log.csv"
N_RUNS = 12


def cfg():
    return load_config(CONFIG)


def stamp(df):
    df["config_hash"] = config_hash(cfg())
    df["git_head"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
    if "seed" not in df:
        df["seed"] = np.nan
    df["seed"] = df["seed"].astype("Int64")
    return df


def run_of_position(subject):
    """Run (0-based) of each of the 750 trial positions of a session, from the design stimpattern
    (the expression of nsd_common.design); checked against design()['run'] for the image trials."""
    import scipy.io as sio
    d = sio.loadmat(C.design_file())
    sp = d["stimpattern"].astype(bool)
    run_of = np.concatenate([np.full(sp[0, r].sum(), r) for r in range(sp.shape[1])])
    des = C.design(subject)
    if len(run_of) != C.cfg()["design"]["trials_per_session"] or not np.array_equal(run_of[des["pos"]], des["run"]):
        raise SystemExit(f"STOP: run of trial position disagrees with the design ({subject})")
    return run_of, des


def run_means(Z, run_of, n_runs=N_RUNS):
    """Z (750, v) trials; run_of (750,). Returns (n_runs, v) float32 means and trials per run."""
    Z = np.asarray(Z, np.float64)
    M = np.stack([Z[run_of == r].mean(axis=0) for r in range(n_runs)])
    return M.astype(np.float32), np.bincount(run_of, minlength=n_runs)


def session_file(subject, s):
    return C.nsd_dir("runmeans", subject) / f"session{s:02d}.npz"


def process_session(path, subject, s, union, run_of, des):
    """Steps 2-4 for one downloaded file; returns a log row (no data)."""
    import nsd_fetch as F
    Z, valid, zm, zs = F.process(path, union, np.arange(len(run_of)), C.cfg()["betas"]["scale_divisor"])
    ref = np.load(C.nsd_dir("betas", subject) / f"session{s:02d}.npz")
    pos = des["pos"][des["session"] == s]
    if not np.array_equal(ref["positions"], pos):
        raise SystemExit(f"STOP: {subject} session {s}: stored positions differ from the design")
    same = np.array_equal(Z[pos], ref["Z"]) and np.array_equal(valid, ref["valid"])
    dev = float(np.abs(Z[pos].astype(np.float64) - ref["Z"]).max())
    if not same:
        raise SystemExit(f"STOP: {subject} session {s}: re-z-scored trials differ from step 42 (max |d| {dev:.1e})")
    M, n = run_means(Z, run_of)
    np.savez(session_file(subject, s), M=M, n_trials=n, valid=valid)
    return dict(subject=subject, session=s, n_voxels=int(Z.shape[1]), n_runs=len(n), trials_min=int(n.min()),
                trials_max=int(n.max()), n_trials=int(n.sum()), gate_identical=bool(same), gate_max_abs_dev=dev,
                n_invalid_voxels=int((~valid).sum()), zmean_absmax=float(np.abs(zm).max()),
                zsd_absdev_max=float(np.abs(zs - 1).max()))


def fetch_subject(subject, sessions=None):
    import nsd_fetch as F
    run_of, des = run_of_position(subject)
    sessions = sessions or sorted(np.unique(des["session"]).tolist())
    todo = [s for s in sessions if not session_file(subject, s).exists()]
    union = np.load(C.nsd_dir("roi", subject) / "union.npz")["union"]
    tmp = C.nsd_dir("tmp")
    sizes = {s: C.object_size(F.beta_key(subject, s)) for s in todo}

    def fetch(s):
        C.require_free(sizes[s] / 1e9)          # min_free_gb 30 (config_step42 = config_step47b)
        t = time.time()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        p = C.download(F.beta_key(subject, s), tmp / f"{subject}_betas_session{s:02d}.nii.gz")
        return p, time.time() - t, started

    rows = []
    with ThreadPoolExecutor(1) as pool:
        fut = pool.submit(fetch, todo[0]) if todo else None
        for i, s in enumerate(todo):
            path, t_dl, started = fut.result()
            fut = pool.submit(fetch, todo[i + 1]) if i + 1 < len(todo) else None
            t = time.time()
            row = process_session(path, subject, s, union, run_of, des)
            path.unlink()
            row.update(download_started_utc=started, size_gb=sizes[s] / 1e9, download_s=t_dl,
                       process_s=time.time() - t, free_gb=C.free_gb())
            rows.append(row)
            print(f"{subject} session {s:02d}: {sizes[s] / 1e9:.2f} GB, download {t_dl:.0f} s, process "
                  f"{row['process_s']:.0f} s, runs {row['n_runs']} ({row['trials_min']}-{row['trials_max']} trials), "
                  f"gate identical {row['gate_identical']}, free {row['free_gb']:.1f} GB", flush=True)
            write_log(rows[-1:])
    return rows


def write_log(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if LOG.exists():
        prev = pd.read_csv(LOG).drop(columns=["config_hash", "git_head", "seed"], errors="ignore")
        keep = ~prev.set_index(["subject", "session"]).index.isin(df.set_index(["subject", "session"]).index)
        df = pd.concat([prev[keep], df], ignore_index=True)
    df = df.sort_values(["subject", "session"]).reset_index(drop=True)
    stamp(df).to_csv(LOG, index=False)


# ── run centring ────────────────────────────────────────────────────────────
def centre(P, session, run, RM):
    """P (n, 3, v) presentations; session (n, 3) 1-based; run (n, 3) 0-based; RM {session: (12, v)}.
    Returns P minus the run mean of each presentation's (session, run), float64."""
    P = np.asarray(P, np.float64)
    out = np.empty_like(P)
    for i in range(P.shape[0]):
        for r in range(P.shape[1]):
            out[i, r] = P[i, r] - RM[int(session[i, r])][int(run[i, r])]
    return out


def load_runmeans(subject, sessions):
    RM = {}
    for s in sessions:
        f = session_file(subject, s)
        if not f.exists():
            raise SystemExit(f"STOP: run means of {subject} session {s} missing")
        RM[s] = np.load(f)["M"].astype(np.float64)
    return RM


def centred_patterns(subject):
    """(P_rc (515, 3, v) float64, session, run, rois, valid) - as nsd_rsa.load_patterns, run-centred."""
    from nsd_rsa import load_patterns
    P, session, run, rois, valid = load_patterns(subject)
    RM = load_runmeans(subject, sorted(np.unique(session).tolist()))
    return centre(P, session, run, RM), session, run, rois, valid


def estimate():
    rows = []
    for s in C.cfg()["subjects"]:
        d = pd.read_csv(C.OUT / f"fetch_{s}.csv")
        x = d[d.kind == "session"]
        rows.append((x.download_s.sum(), x.process_s.sum(), x.size_gb.sum()))
    dl, pr, gb = map(sum, zip(*rows))
    h = max(dl, pr) / 3600 * 1.5
    print(f"reload: download {dl / 3600:.2f} h, processing {pr / 3600:.2f} h (overlapped), {gb:.0f} GB; "
          f"estimate x 1.5: {h:.2f} h; free {C.free_gb():.0f} GB")
    return h


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "gate":
        t0 = time.time()
        r = fetch_subject("subj01", [1])
        print(f"gate subj01 session 1 done ({time.time() - t0:.0f} s)")
    elif which == "estimate":
        h = estimate()
        if h > cfg()["runtime"]["stop_hours_per_part"]:
            raise SystemExit(f"STOP: estimate {h:.2f} h above {cfg()['runtime']['stop_hours_per_part']} h")
    elif which == "all":
        t0 = time.time()
        for s in C.cfg()["subjects"]:
            fetch_subject(s)
        print(f"all subjects done ({time.time() - t0:.0f} s); free {C.free_gb():.1f} GB")
    else:
        raise SystemExit("usage: nsd_runmeans.py {gate|estimate|all}")

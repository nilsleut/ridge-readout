"""
nsd_raw_fetch.py
================
Step 48 (amendments/2026-11-22_step48-predictions.md, config_step48.yaml): NSD raw single-trial
betas of the 515 images and per-session voxel statistics, sessions 1-30, all 8 subjects.

Per session (download as nsd_fetch.py / nsd_runmeans.py; imported: nsd_fetch.beta_key,
nsd_common.download, require_free, design):
  1. download betas_sessionBB.nii.gz to NSD_DIR/tmp (next file downloaded while the current one is
     processed; at most two on disk);
  2. V1 | lateral voxels (NSD_DIR/roi/<subject>/union.npz): Q int16 (n_voxels, 750);
     X = Q / 300 in float64; mu, sd per voxel over the 750 trials (ddof 0); valid = sd > 0;
  3. gate: z = (X - mu) / sd (0 where invalid) at the 515-image trial positions, cast to float32,
     equals the step 42 NSD_DIR/betas/<subject>/sessionBB.npz Z exactly, and valid equals (STOP
     otherwise); the float32 path (R = float32(X)) is reported as max |d|;
  4. store NSD_DIR/raw48/<subject>/sessionBB.npz: Q (n_trials, v) int16, R float32, mu, sd float64,
     valid, positions;
  5. delete the downloaded file.

assemble(subject) -> (P_raw (515, 3, v) float64 % signal change, session, run, rois, valid, M, S)
with M, S the per-voxel mean and SD over all trials of the loaded sessions (pooled).

Printed and logged: only shapes, counts, sizes, durations, gate results. Log in the repository:
results/step48/nsd_raw_log.csv.

Usage:
  python scripts/nsd_raw_fetch.py gate       subj01 session 1 only
  python scripts/nsd_raw_fetch.py all        all subjects, all sessions (resumes)
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
CONFIG = REPO / "config_step48.yaml"
OUT = REPO / "results" / "step48"
LOG = OUT / "nsd_raw_log.csv"


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


def session_file(subject, s):
    return C.nsd_dir("raw48", subject) / f"session{s:02d}.npz"


def session_stats(Q, divisor):
    """Q (v, 750) int16 -> (X float64 (v, 750), mu (v,), sd (v,), valid) - the nsd_fetch.process
    expressions."""
    X = Q.astype(np.float64) / divisor
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return X, mu[:, 0], sd[:, 0], sd[:, 0] > 0


def zscore(X, mu, sd, valid):
    """X (n, v) float64 trials; the nsd_fetch.process z-score (0 for invalid voxels)."""
    return np.where(valid[None, :], (X - mu[None, :]) / np.where(valid, sd, 1.0)[None, :], 0.0)


def process_session(path, subject, s, union, des):
    import nibabel as nib
    divisor = C.cfg()["betas"]["scale_divisor"]
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.shape[3] != C.cfg()["design"]["trials_per_session"]:
        raise SystemExit(f"STOP: {path.name} has {data.shape[3]} trials")
    Q = data[union]                                              # int16 (v, 750)
    del data
    if Q.dtype != np.int16:
        raise SystemExit(f"STOP: {path.name} is {Q.dtype}, not int16")
    X, mu, sd, valid = session_stats(Q, divisor)
    pos = des["pos"][des["session"] == s]
    ref = np.load(C.nsd_dir("betas", subject) / f"session{s:02d}.npz")
    if not np.array_equal(ref["positions"], pos):
        raise SystemExit(f"STOP: {subject} session {s}: stored positions differ from the design")
    Xp = X[:, pos].T
    Z = zscore(Xp, mu, sd, valid).astype(np.float32)
    same = bool(np.array_equal(Z, ref["Z"]) and np.array_equal(valid, ref["valid"]))
    dev = float(np.abs(Z.astype(np.float64) - ref["Z"]).max())
    if not same:
        raise SystemExit(f"STOP: {subject} session {s}: z from raw and session statistics differs from step 42 "
                         f"(max |d| {dev:.1e})")
    R = Xp.astype(np.float32)
    Z32 = zscore(R.astype(np.float64), mu, sd, valid).astype(np.float32)
    dev32 = float(np.abs(Z32.astype(np.float64) - ref["Z"]).max())
    np.savez(session_file(subject, s), Q=Q[:, pos].T, R=R, mu=mu, sd=sd, valid=valid, positions=pos)
    return dict(subject=subject, session=s, n_voxels=int(Q.shape[0]), n_trials_kept=int(len(pos)),
                gate_identical=same, gate_max_abs_dev=dev, float32_path_identical=bool(np.array_equal(Z32, ref["Z"])),
                float32_path_max_abs_dev=dev32, n_invalid_voxels=int((~valid).sum()),
                mu_mean=float(mu[valid].mean()), sd_mean=float(sd[valid].mean()))


def fetch_subject(subject, sessions=None):
    import nsd_fetch as F
    des = C.design(subject)
    sessions = sessions or sorted(np.unique(des["session"]).tolist())
    todo = [s for s in sessions if not session_file(subject, s).exists()]
    union = np.load(C.nsd_dir("roi", subject) / "union.npz")["union"]
    tmp = C.nsd_dir("tmp")
    sizes = {s: C.object_size(F.beta_key(subject, s)) for s in todo}

    def fetch(s):
        C.require_free(sizes[s] / 1e9)
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
            row = process_session(path, subject, s, union, des)
            path.unlink()
            row.update(download_started_utc=started, size_gb=sizes[s] / 1e9, download_s=t_dl,
                       process_s=time.time() - t, free_gb=C.free_gb())
            rows.append(row)
            print(f"{subject} session {s:02d}: {row['size_gb']:.2f} GB, download {t_dl:.0f} s, process "
                  f"{row['process_s']:.0f} s, kept {row['n_trials_kept']} trials, gate identical {row['gate_identical']}, "
                  f"float32 path max |d| {row['float32_path_max_abs_dev']:.1e}, free {row['free_gb']:.1f} GB", flush=True)
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


# ── assembly and conventions ────────────────────────────────────────────────
def pooled_stats(mus, sds):
    """Per-voxel mean and SD over all trials of sessions with equal trial counts, from the session
    means and SDs (ddof 0)."""
    mus, sds = np.stack(mus), np.stack(sds)
    M = mus.mean(axis=0)
    V = (sds ** 2 + mus ** 2).mean(axis=0) - M ** 2
    return M, np.sqrt(np.maximum(V, 0.0))


def assemble(subject):
    """(P_raw (515, 3, v) float64, session, run, rois, valid, M, S, mus, sds) for the stored sessions;
    presentation layout of nsd_rsa.load_patterns (step 42 patterns)."""
    from nsd_rsa import load_patterns
    P42, session, run, rois, valid = load_patterns(subject)
    d = C.design(subject)
    sessions = sorted(np.unique(d["session"]).tolist())
    P = np.full(P42.shape, np.nan)
    mus, sds = {}, {}
    for s in sessions:
        f = session_file(subject, s)
        if not f.exists():
            raise SystemExit(f"STOP: raw betas of {subject} session {s} missing")
        z = np.load(f)
        sel = np.argwhere(d["session"] == s)
        if not np.array_equal(d["pos"][sel[:, 0], sel[:, 1]], z["positions"]):
            raise SystemExit(f"STOP: trial order of {subject} session {s} differs from the design")
        P[sel[:, 0], sel[:, 1]] = z["Q"].astype(np.float64) / C.cfg()["betas"]["scale_divisor"]
        mus[s], sds[s] = z["mu"], z["sd"]
    if np.isnan(P).any():
        raise SystemExit(f"STOP: missing presentations for {subject}")
    M, S = pooled_stats([mus[s] for s in sessions], [sds[s] for s in sessions])
    return P, session, run, rois, valid, M, S, mus, sds


def conventions(subject):
    """{'N-raw', 'N-global', 'N-session'}: (515, 3, v) float64, plus session, run, rois, valid."""
    P, session, run, rois, valid, M, S, mus, sds = assemble(subject)
    Sg = np.where(S > 0, S, 1.0)
    out = {"N-raw": P, "N-global": np.where(valid, (P - M) / Sg, 0.0)}
    Zs = np.empty_like(P)
    for i in range(P.shape[0]):
        for r in range(P.shape[1]):
            s = int(session[i, r])
            Zs[i, r] = zscore(P[i, r][None, :], mus[s], sds[s], sds[s] > 0)[0].astype(np.float32)
    out["N-session"] = Zs
    return out, session, run, rois, valid


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "gate":
        t0 = time.time()
        fetch_subject("subj01", [1])
        print(f"gate subj01 session 1 done ({time.time() - t0:.0f} s)")
    elif which == "all":
        t0 = time.time()
        for s in C.cfg()["subjects"]:
            fetch_subject(s)
        print(f"all subjects done ({time.time() - t0:.0f} s); free {C.free_gb():.1f} GB")
    else:
        raise SystemExit("usage: nsd_raw_fetch.py {gate|all}")

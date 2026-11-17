"""
nsd_fetch.py
============
Step 42 (amendments/2026-11-17_step42-predictions.md): load the NSD single-trial betas of
one subject, session by session.

Per session (only sessions that contain trials of the 515 images, i.e. 1-30):
  1. check free disk space (at least disk.min_free_gb plus the file size, else STOP);
  2. download betas_sessionBB.nii.gz (b3, func1pt8mm) to NSD_DIR/tmp;
  3. grid check of the ROI files and floc_objectstval against the betas (first session; STOP);
  4. take the voxels of V1 and 'lateral' (LO-obj is a subset of 'lateral'), cast to float,
     divide by 300, z-score every voxel over all 750 trials of the session (ddof 0);
     voxels with zero SD are marked invalid for the session;
  5. keep the trials of the 515 images, store float32 under NSD_DIR/betas/<subject>/;
  6. delete the downloaded file.
The next file is downloaded while the current one is processed (at most two on disk).
At the end the three presentations are assembled into NSD_DIR/patterns/<subject>.npz:
  P (515, 3, n_voxels) float32, presentation order chronological; session, run (515, 3);
  valid (n_voxels) = valid in every loaded session; local ROI masks; z-score statistics.

Printed and logged: only shapes, counts, sizes, durations and summary statistics.
Output in the repository: results/step42/fetch_<subject>.csv (per session: size, seconds,
free space, trials kept, invalid voxels).

Usage: python scripts/nsd_fetch.py <subject>
"""
import nsd_common as C     # noqa: I001  (pins BLAS threads)

import sys
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


def beta_key(subject, session):
    return C.cfg()["betas"]["path"].format(subject=subject, session=session)


def union_masks(subject, img):
    """Grid check; union voxel mask (V1 | lateral) and ROI masks local to the union."""
    m = C.roi_masks(subject, ref_shape=img.shape, ref_affine=img.affine)
    if np.any(m["LO-obj"] & ~m["lateral"]):
        raise SystemExit("STOP: LO-obj is not a subset of lateral")
    union = m["V1"] | m["lateral"]
    local = {k: v[union] for k, v in m.items()}
    return union, local


def process(path, union, positions, divisor):
    import nibabel as nib
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)                              # int16 (X, Y, Z, 750)
    if data.shape[3] != C.cfg()["design"]["trials_per_session"]:
        raise SystemExit(f"STOP: {path.name} has {data.shape[3]} trials")
    X = data[union].astype(np.float64) / divisor                # (n_voxels, 750)
    del data
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    valid = sd[:, 0] > 0
    Z = np.where(valid[:, None], (X - mu) / np.where(valid[:, None], sd, 1.0), 0.0)
    zmean, zsd = Z.mean(axis=1), Z.std(axis=1)
    return Z[:, positions].T.astype(np.float32), valid, zmean[valid], zsd[valid]


def main(subject):
    c = C.cfg()
    t0 = time.time()
    d = C.design(subject)
    sessions = sorted(np.unique(d["session"]).tolist())
    out_dir = C.nsd_dir("betas", subject)
    tmp = C.nsd_dir("tmp")
    divisor = c["betas"]["scale_divisor"]
    print(f"{subject}: {len(d['images'])} images, {d['trial'].size} trials in {len(sessions)} sessions "
          f"({sessions[0]}-{sessions[-1]}); free {C.free_gb():.1f} GB", flush=True)

    todo = [s for s in sessions if not (out_dir / f"session{s:02d}.npz").exists()]
    sizes = {s: C.object_size(beta_key(subject, s)) for s in sessions}
    union = local = None
    um = C.nsd_dir("roi", subject) / "union.npz"
    if um.exists():
        z = np.load(um)
        union = z["union"]
        local = {k: z[f"local_{k}"] for k in ("V1", "LO-obj", "lateral")}

    def fetch(s):
        C.require_free(sizes[s] / 1e9)
        t = time.time()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        p = C.download(beta_key(subject, s), tmp / f"{subject}_betas_session{s:02d}.nii.gz")
        return p, time.time() - t, started

    rows = []
    with ThreadPoolExecutor(1) as pool:
        fut = pool.submit(fetch, todo[0]) if todo else None
        for i, s in enumerate(todo):
            path, t_dl, started = fut.result()
            fut = pool.submit(fetch, todo[i + 1]) if i + 1 < len(todo) else None
            t = time.time()
            if union is None:
                import nibabel as nib
                img = nib.load(str(path))
                union, local = union_masks(subject, img)
                np.savez(um, union=union, **{f"local_{k}": v for k, v in local.items()})
                print(f"grid check passed; voxels V1 {int(local['V1'].sum())}, LO-obj "
                      f"{int(local['LO-obj'].sum())}, lateral {int(local['lateral'].sum())}, "
                      f"union {int(union.sum())}", flush=True)
            sel = d["session"] == s
            positions = d["pos"][sel]
            Z, valid, zm, zs = process(path, union, positions, divisor)
            np.savez(out_dir / f"session{s:02d}.npz", Z=Z, positions=positions, valid=valid,
                     zmean_absmax=np.abs(zm).max(), zsd_absdev_max=np.abs(zs - 1).max(),
                     zmean_q=np.quantile(zm, [0, .5, 1]), zsd_q=np.quantile(zs, [0, .5, 1]))
            path.unlink()
            rows.append(dict(kind="session", subject=subject, session=s, download_started_utc=started,
                             size_gb=sizes[s] / 1e9,
                             download_s=t_dl, process_s=time.time() - t, free_gb=C.free_gb(),
                             trials_kept=int(sel.sum()), n_invalid_voxels=int((~valid).sum()),
                             zmean_absmax=float(np.abs(zm).max()), zsd_absdev_max=float(np.abs(zs - 1).max())))
            print(f"session {s:02d}: {sizes[s] / 1e9:.2f} GB, download {t_dl:.0f} s, process "
                  f"{time.time() - t:.0f} s, kept {int(sel.sum())} trials, invalid voxels "
                  f"{int((~valid).sum())}, free {C.free_gb():.1f} GB", flush=True)

    assemble(subject, d, sessions, local)
    df = pd.DataFrame(rows)
    df = pd.concat([df, pd.DataFrame([dict(kind="total", subject=subject, size_gb=sum(sizes.values()) / 1e9,
                                           download_s=df.download_s.sum() if len(df) else 0.0,
                                           process_s=time.time() - t0, free_gb=C.free_gb(),
                                           trials_kept=int(d["trial"].size))])], ignore_index=True)
    C.OUT.mkdir(parents=True, exist_ok=True)
    f = C.OUT / f"fetch_{subject}.csv"
    if f.exists() and len(rows) < len(sessions):          # resumed run: keep earlier session rows
        prev = pd.read_csv(f)
        prev = prev[(prev.kind == "session") & ~prev.session.isin(df.session)]
        df = pd.concat([prev.drop(columns=["config_hash", "git_head", "seed"], errors="ignore"), df],
                       ignore_index=True)
    C.stamp(df).to_csv(f, index=False)
    print(f"done in {time.time() - t0:.0f} s; free {C.free_gb():.1f} GB")


def assemble(subject, d, sessions, local):
    src = C.nsd_dir("betas", subject)
    n = len(d["images"])
    P = None
    valid = None
    for s in sessions:
        z = np.load(src / f"session{s:02d}.npz")
        Z, pos = z["Z"], z["positions"]
        if P is None:
            P = np.full((n, 3, Z.shape[1]), np.nan, np.float32)
            valid = np.ones(Z.shape[1], bool)
        valid &= z["valid"]
        sel = np.argwhere(d["session"] == s)
        if not np.array_equal(d["pos"][sel[:, 0], sel[:, 1]], pos):
            raise SystemExit(f"STOP: trial order of session {s} differs from the design")
        P[sel[:, 0], sel[:, 1]] = Z
    if np.isnan(P).any():
        raise SystemExit("STOP: missing presentations after assembly")
    np.savez(C.nsd_dir("patterns") / f"{subject}.npz", P=P, session=d["session"], run=d["run"],
             images=d["images"], valid=valid, **{f"local_{k}": v for k, v in local.items()})
    print(f"assembled P {P.shape}; valid voxels {int(valid.sum())} of {len(valid)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: nsd_fetch.py <subject>")
    main(sys.argv[1])

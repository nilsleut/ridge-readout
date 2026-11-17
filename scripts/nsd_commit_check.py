"""
nsd_commit_check.py
===================
Step 42, rule (c): before every commit, no file under NSD_DIR and no file with voxel or RDM
content may be tracked or staged. Exit 1 (STOP) otherwise.

Checks, on the staged files (git diff --cached) and on all tracked files:
  1. no path resolves to a location inside NSD_DIR;
  2. no staged file has a data extension (.npy .npz .nii .nii.gz .mgz .h5 .hdf5 .mat .pkl);
  3. no staged file is byte-identical to a file under NSD_DIR (copies of NSD data);
  4. staged CSVs under results/step42: no column named like voxel, beta, pattern or rdm
     (count columns starting with "n_" excepted), and at most 5 MB.
The check reads NSD_DIR through paths.py (environment, then .env.local).

Usage: python scripts/nsd_commit_check.py [--repo PATH] [--install-hook]
"""
import hashlib
import subprocess
import sys
from pathlib import Path

DATA_EXT = (".npy", ".npz", ".nii", ".nii.gz", ".mgz", ".h5", ".hdf5", ".mat", ".pkl")
BAD_COLS = ("voxel", "beta", "pattern", "rdm")
MAX_CSV = 5_000_000


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def nsd_dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ridge"))
    import paths
    return paths.nsd_dir().resolve()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(repo, nsd=None):
    repo = Path(repo).resolve()
    nsd = Path(nsd).resolve() if nsd else nsd_dir()
    staged = [s for s in git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines() if s]
    tracked = [s for s in git(repo, "ls-files").splitlines() if s]
    problems = []
    for f in sorted(set(staged + tracked)):
        p = (repo / f).resolve()
        if p == nsd or nsd in p.parents:
            problems.append(f"under NSD_DIR: {f}")
    for f in staged:
        if f.lower().endswith(DATA_EXT):
            problems.append(f"data extension: {f}")
    sizes = {}
    if nsd.exists():
        for q in nsd.rglob("*"):
            if q.is_file():
                sizes.setdefault(q.stat().st_size, []).append(q)
    for f in staged:
        p = repo / f
        if not p.is_file():
            continue
        cands = sizes.get(p.stat().st_size, [])
        if cands and any(sha(p) == sha(q) for q in cands):
            problems.append(f"identical to a file under NSD_DIR: {f}")
        if f.startswith("results/step42/") and f.endswith(".csv"):
            if p.stat().st_size > MAX_CSV:
                problems.append(f"CSV larger than {MAX_CSV} bytes: {f}")
            header = p.read_text(encoding="utf-8").splitlines()[0].lower().split(",")
            bad = [h for h in header if any(b in h for b in BAD_COLS) and not h.startswith("n_")]
            if bad:
                problems.append(f"voxel/RDM-like columns {bad}: {f}")
    return staged, problems


HOOK = """#!/bin/sh
# step 42 rule (c): refuse commits that would track NSD data
exec python "$(git rev-parse --show-toplevel)/scripts/nsd_commit_check.py"
"""


def main(argv):
    repo = Path(__file__).resolve().parents[1]
    if "--repo" in argv:
        repo = Path(argv[argv.index("--repo") + 1])
    if "--install-hook" in argv:
        h = Path(git(repo, "rev-parse", "--git-path", "hooks").strip())
        h = h if h.is_absolute() else repo / h
        (h / "pre-commit").write_text(HOOK, encoding="utf-8", newline="\n")
        print(f"installed {h / 'pre-commit'}")
        return 0
    staged, problems = check(repo)
    print(f"nsd commit check: {len(staged)} staged file(s), {len(problems)} problem(s)")
    for p in problems:
        print("  " + p)
    if problems:
        print("STOP: commit refused")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""
prereg_extract.py
=================
Extract the missing g = 2 Conv1 features (and the control band at g = 2) for the
preregistered primary analysis, with the EXISTING extraction code:
`scripts/pilot_features_modal.py::extract_pilot`, run locally on the CPU.

The function body is taken from the source file by AST and executed unchanged except
for these textual substitutions, each asserted to occur the stated number of times:
  1. decorator `@app.function(...)` dropped (the FunctionDef is taken without it)
  2. dev = torch.device("cuda")                  -> torch.device("cpu")          (1x)
  3. Path("/data/outputs_720")                   -> OUTPUTS_720_DIR              (1x)
  4. Path("/data/object_images")                 -> THINGS_IMAGES_DIR            (1x)
  5. Path("/results/outputs_allseeds/checkpoints") -> CKPT_V12_DIR               (1x)
  6. Path("/results/pilot_features")             -> data/prereg_features         (1x)
  7. num_workers=4                               -> num_workers=0  (Windows cannot
     pickle the nested Dataset class into worker processes)                     (2x)
  8. results_vol.commit()                        -> no-op stub                    (1x)
Constants RULES, SEEDS, PILOT_RES are read from the same file (ast.literal_eval).
Called with grids = [1, 2, 4, 8].

Reproduction gate (STOP on failure): every g = 1, 4, 8 array written here must equal
the corresponding file in RIDGE_DATA_DIR/pilot_features (extracted on a Modal T4) to
max |diff| <= 1e-6.

Output: data/prereg_features/*.npy (gitignored).
"""
import ast
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "ridge"))
sys.path.insert(0, str(REPO / "src" / "vendor"))

import paths  # noqa: E402

SRC = REPO / "scripts" / "pilot_features_modal.py"
OUT = REPO / "data" / "prereg_features"
TOL = 1e-6


def build():
    text = SRC.read_text(encoding="utf-8")
    tree = ast.parse(text)
    consts = {}
    fn = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("RULES", "SEEDS", "PILOT_RES"):
            consts[node.targets[0].id] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == "extract_pilot":
            fn = node
    assert fn is not None and set(consts) == {"RULES", "SEEDS", "PILOT_RES"}
    body = ast.get_source_segment(text, fn)
    subs = [
        ('dev = torch.device("cuda")', 'dev = torch.device("cpu")', 1),
        ('Path("/data/outputs_720")', f'Path(r"{paths.outputs_720_dir()}")', 1),
        ('Path("/data/object_images")', f'Path(r"{paths.things_images_dir()}")', 1),
        ('Path("/results/outputs_allseeds/checkpoints")', f'Path(r"{paths.ckpt_v12_dir()}")', 1),
        ('Path("/results/pilot_features")', f'Path(r"{OUT}")', 1),
        ("num_workers=4", "num_workers=0", 2),
        ("results_vol.commit()", "None", 1),
    ]
    for old, new, n in subs:
        c = body.count(old)
        if c != n:
            raise SystemExit(f"STOP: substitution '{old}' found {c} times, expected {n}")
        body = body.replace(old, new)
    ns = dict(consts, GRIDS=[1, 4])
    exec(compile(body, str(SRC), "exec"), ns)
    return ns["extract_pilot"]


def gate():
    ref = paths.ridge_data_dir() / "pilot_features"
    worst, n = 0.0, 0
    for f in sorted(OUT.glob("*_g[148].npy")):
        r = ref / f.name
        if not r.exists():
            raise SystemExit(f"STOP: no reference for {f.name}")
        d = float(np.abs(np.load(f).astype(np.float64) - np.load(r).astype(np.float64)).max())
        worst, n = max(worst, d), n + 1
        print(f"  {f.name}: max |diff| {d:.2e}", flush=True)
    print(f"gate: {n} arrays, max |diff| {worst:.2e} (tolerance {TOL:.0e})", flush=True)
    if n != len(list(ref.glob("*_g[148].npy"))) or worst > TOL:
        raise SystemExit("STOP: reproduction gate failed")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build()(grids=[1, 2, 4, 8])
    gate()

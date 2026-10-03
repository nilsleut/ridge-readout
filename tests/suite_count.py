"""
suite_count.py
==============
Generate the control-suite count AT REPORT TIME. Never write it in prose.

The count drifted 78 -> 83 -> 89 -> 97 across three reports because it was typed
by hand at the moment of writing. Each value was correct when written and wrong by
the time it was read. This is the same class of failure the config hash prevents in
data, so it gets the same treatment: generated, tied to the config hash, substituted.

    python code/ridge/suite_count.py            # human-readable line
    python code/ridge/suite_count.py --json     # machine-readable
    python code/ridge/suite_count.py --stamp FILE   # substitute into a markdown file

`--stamp` replaces the contents of every
    <!--SUITE-->...<!--/SUITE-->
marker in FILE with the freshly generated line.
"""
import io, json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "ridge"))
from config_hash import config_hash  # noqa: E402

SUITES = {
    "phase2 controls": "test_phase2_controls.py",
    "loader hardening": "test_loader_hardening.py",
}


def run_one(script: str):
    r = subprocess.run([sys.executable, str(HERE / script)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"all (\d+) checks passed", out)
    if m:
        return int(m.group(1)), 0
    m = re.search(r"(\d+) failure\(s\) of (\d+) checks", out)
    if m:
        return int(m.group(2)), int(m.group(1))
    return None, None


def collect():
    res, total, failed = {}, 0, 0
    for name, script in SUITES.items():
        n, f = run_one(script)
        res[name] = {"checks": n, "failures": f}
        if n:
            total += n
            failed += f or 0
    gate = subprocess.run([sys.executable, str(HERE.parent / "scripts" / "determinism_gate.py"), "--self-test"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    res["determinism gate self-test"] = {
        "checks": "pass" if gate.returncode == 0 else "FAIL", "failures": gate.returncode}
    return {"suites": res, "total_checks": total, "total_failures": failed,
            "config_hash": config_hash()}


def line(d):
    parts = ", ".join(f"{k} {v['checks']}" for k, v in d["suites"].items()
                      if isinstance(v["checks"], int))
    status = "all passing" if d["total_failures"] == 0 else f"{d['total_failures']} FAILING"
    return (f"**{d['total_checks']} checks, {status}** ({parts}; "
            f"determinism-gate self-test "
            f"{d['suites']['determinism gate self-test']['checks']}) "
            f"against config `{d['config_hash']}`")


if __name__ == "__main__":
    d = collect()
    if "--json" in sys.argv:
        print(json.dumps(d, indent=2))
    elif "--stamp" in sys.argv:
        f = Path(sys.argv[sys.argv.index("--stamp") + 1])
        s = io.open(f, encoding="utf-8").read()
        new, n = re.subn(r"<!--SUITE-->.*?<!--/SUITE-->",
                         f"<!--SUITE-->{line(d)}<!--/SUITE-->", s, flags=re.S)
        io.open(f, "w", encoding="utf-8").write(new)
        print(f"stamped {n} marker(s) in {f.name}: {line(d)}")
    else:
        print(line(d))

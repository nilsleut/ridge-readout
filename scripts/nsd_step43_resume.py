"""
nsd_step43_resume.py
====================
Resume of the step 43 run after the multiverse part had finished (results/step43/multiverse_nsd.csv
written) and the process was ended during the ImageNet bootstrap. Runs the remaining parts with the
unchanged functions of nsd_step43 (run_imagenet, run_resolution, run_ceiling) and writes summary.csv
as nsd_step43.run_all does; M1 and M2 are read from the summary rows of multiverse_nsd.csv.

Run: python scripts/nsd_step43_resume.py [workers]
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsd_common as C          # noqa: E402
import nsd_step43 as NS         # noqa: E402


def multiverse_rules():
    mv = pd.read_csv(NS.OUT / "multiverse_nsd.csv", low_memory=False)
    s = mv[mv.kind == "summary"].set_index("quantity").value.astype(float)
    n_cls = int(s["distinct analyses"])
    above = int(s["distinct analyses, interval above 0"])
    below = int(s["distinct analyses, interval below 0"])
    rules = [dict(rule="M1", quantity="distinct analyses with the interval above 0 <= 2", value=above,
                  passed=bool(above <= 2), note=f"{above} above, {below} below, {n_cls - above - below} include 0 of {n_cls}"),
             dict(rule="M2", quantity="distinct analyses with the interval below 0 >= half", value=below,
                  passed=bool(below >= n_cls / 2), note=f"{below} of {n_cls} (half = {n_cls / 2:g})")]
    return rules, float(s["runtime s"])


def main(workers=8):
    C.require_free(NS.cfg()["disk"]["min_free_gb"])
    rules, t_mv = multiverse_rules()
    times = {"multiverse": t_mv}
    t0 = time.time()
    for name, fn in (("imagenet", NS.run_imagenet), ("resolution", NS.run_resolution), ("ceiling", NS.run_ceiling)):
        t = time.time()
        r = fn(workers)
        if r:
            rules += r
        times[name] = time.time() - t
        print(f"== {name} done in {times[name]:.0f} s", flush=True)
    summ = pd.DataFrame(rules + [dict(rule="runtime", quantity=f"{k} wall seconds", value=v) for k, v in times.items()]
                        + [dict(rule="runtime", quantity="total wall seconds (multiverse + resumed parts)",
                                value=t_mv + time.time() - t0)])
    NS.stamp(summ).to_csv(NS.OUT / "summary.csv", index=False)
    print(summ[["rule", "quantity", "value", "passed", "note"]].to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)

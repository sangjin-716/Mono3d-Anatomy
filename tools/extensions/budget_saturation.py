"""Produces reports/extensions/budget_saturation.txt (a re-run writes
reports_rerun/extensions/budget_saturation.txt); supplementary Sec. L ("budgets can only be matched
downward"; N=20 is the largest common budget).

How far DOWN can we budget-match? For each detector, fraction of val images whose
COMPLETE pool has >= N candidates, for N in {5,10,20,30,50}. A common budget N is only
truly 'matched' where every detector can supply N. READ-ONLY.
MonoFlex*/MonoGround* read the original-environment dumps (stems monoflex_orig /
monoground_orig); the query-based and M3D-RPN pools read the auxiliary dumps
(see tools/extensions/_ext.py).
Run from the repository root: python tools/extensions/budget_saturation.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ext import dump_path, out_path, aux_dump_path  # noqa: E402
import numpy as np, pandas as pd, datetime  # noqa: E402

OUT = out_path("extensions/budget_saturation.txt")
out = []
def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

NS = [5, 10, 20, 30, 50]
w(f"# budget_saturation {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# frac of 3769 val images whose COMPLETE candidate pool holds >= N candidates")
w(f"{'detector':12s} {'pool_src':10s} " + " ".join(f"N>={n:<4d}" for n in NS) + "   max/img")

def report(name, src, counts):
    c = np.asarray(counts)
    w(f"{name:12s} {src:10s} " + " ".join(f"{(c>=n).mean():<7.3f}" for n in NS) +
      f"   {c.max():.0f}")

CNET = {"MonoDLE": dump_path("monodle"), "GUPNet": dump_path("gupnet"),
        "DEVIANT": dump_path("deviant"), "MonoCon": dump_path("monocon"),
        "MonoFlex*": dump_path("monoflex_orig"),
        "MonoGround*": dump_path("monoground_orig")}
for name, p in CNET.items():
    df = pd.read_csv(p, usecols=["sid"])
    report(name, "cnet-decode", df.groupby("sid").size().values)

QUERY = {"MonoDETR": "monodetr", "MonoDGP": "monodgp", "MonoCoP": "official_monocop",
         "MonoCLUE": "monoclue", "MonoIA": "monoia"}
for name, f in QUERY.items():
    p = aux_dump_path(f"{f}_val_preflatten.csv")
    head = pd.read_csv(p, nrows=2)
    if "class_id" in head.columns:
        df = pd.read_csv(p, usecols=["sid", "class_id", "car_channel"])
        car = df[df.class_id == df.car_channel]
    else:
        car = pd.read_csv(p, usecols=["sid"])
    report(name, "query-car", car.groupby("sid").size().values)

df = pd.read_csv(aux_dump_path("m3drpn_val_floor0.csv"), usecols=["sid"])
report("M3D-RPN", "floor0", df.groupby("sid").size().values)

open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

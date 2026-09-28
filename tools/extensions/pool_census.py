"""Ported from the camera-ready check pool_census.py (same file name) for the public release.
Computation unchanged. Produces reports/extensions/pool_census.txt (a re-run writes
reports_rerun/extensions/pool_census.txt); supplementary Sec. L (complete pre-selection pool
sizes, 23.4 to 2138.3 candidates per image).

Measure per-image CANDIDATE POOL sizes for all 12 panel detectors, using the same pool
definitions as tools/decomp/{pool_waterfall,e4_fp_tp_decomp}.py. READ-ONLY on all dumps.
The CenterNet-style rows read the released modern-environment dumps (stems monoflex /
monoground for MonoFlex and MonoGround); the query-based and M3D-RPN complete pools read the
auxiliary dumps (see tools/extensions/_ext.py).
Run from the repository root: python tools/extensions/pool_census.py
"""
import os, sys, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ext import paths, dump_path, out_path, aux_dump_path  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402

VAL = paths.VAL_LIST
OUT = out_path("extensions/pool_census.txt")
out = []
def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

nval = len([x for x in open(VAL).read().split()])
w(f"# pool_census run {datetime.datetime.now().isoformat(timespec='seconds')}  (val images={nval})")
w("# COMPLETE = pool used as stage-A in pool_waterfall.py ; FINAL = native_pool() in e4_fp_tp_decomp.py")
w("")
w(f"{'detector':11s} {'fam':7s} {'complete/img':>26s} {'final/img':>24s} {'score floor':>22s}")

QUERY = {"MonoDETR": "monodetr", "MonoDGP": "monodgp", "MonoCoP": "official_monocop",
         "MonoCLUE": "monoclue", "MonoIA": "monoia"}
CNET = {"MonoDLE": "monodle", "MonoFlex": "monoflex", "GUPNet": "gupnet",
        "DEVIANT": "deviant", "MonoGround": "monoground", "MonoCon": "monocon"}

def stats(counts):
    c = np.asarray(counts, float)
    return f"mean{c.mean():8.1f} med{np.median(c):7.1f} min{c.min():6.0f} max{c.max():7.0f}"

rows = []
for name, f in CNET.items():
    df = pd.read_csv(dump_path(f), usecols=["sid", "cls", "V"])
    thr = 0.4 if name == "MonoCon" else 0.2
    col = "cls" if name == "MonoDLE" else "V"
    comp = df.groupby("sid").size()
    fin = df[df[col] >= thr].groupby("sid").size().reindex(comp.index, fill_value=0)
    w(f"{name:11s} {'cnet':7s} {stats(comp.values):>26s} {stats(fin.values):>24s} "
      f"  cls_min={df.cls.min():.5f} V_min={df.V.min():.5f}  gate={col}>={thr}")
    rows.append((name, comp.values, fin.values))

for name, f in QUERY.items():
    p = aux_dump_path(f"{f}_val_preflatten.csv")
    head = pd.read_csv(p, nrows=2)
    if "class_id" in head.columns:
        df = pd.read_csv(p, usecols=["sid", "query_id", "class_id", "car_channel",
                                     "flat_rank", "cls", "V"])
        car = df[df.class_id == df.car_channel].copy()
        car["flatrank"] = car["flat_rank"]
        allrows = df.groupby("sid").size()
    else:
        df = pd.read_csv(p, usecols=["sid", "query_id", "layer_id", "cls_car",
                                     "V_car", "flat_rank_car"])
        car = df.copy()
        car["flatrank"] = car["flat_rank_car"]; car["V"] = car["V_car"]; car["cls"] = car["cls_car"]
        allrows = df.groupby("sid").size()
    comp = car.groupby("sid").size()
    fin = car[(car.flatrank < 50) & (car.cls >= 0.2)].groupby("sid").size().reindex(comp.index, fill_value=0)
    w(f"{name:11s} {'query':7s} {stats(comp.values):>26s} {stats(fin.values):>24s} "
      f"  cls_min={car.cls.min():.5f}  gate=flatrank<50 & cls>=0.2  | raw dump rows/img "
      f"mean={allrows.mean():.1f}")
    rows.append((name, comp.values, fin.values))

# anchor: floor-0 (2.87 GB) — read only sid + V + bbox for NMS-free size census
p = aux_dump_path("m3drpn_val_floor0.csv")
df = pd.read_csv(p, usecols=["sid", "V"])
comp = df.groupby("sid").size()
w(f"{'M3D-RPN':11s} {'anchor':7s} {stats(comp.values):>26s} "
  f"{'(final = NMS0.4->top40->V>=0.75, see below)':>24s}   V_min={df.V.min():.5f}")
w("")
w(f"M3D-RPN floor-0 total rows = {len(df)}; images at 300-cap = {(comp.values>=300).sum()}")
fin_native = pd.read_csv(dump_path("m3drpn"), usecols=["sid", "V"])
w(f"m3drpn_val.csv (non-floor0 native dump) rows = {len(fin_native)}, "
  f"per-img {stats(fin_native.groupby('sid').size().values)}, V_min={fin_native.V.min():.5f}")

open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

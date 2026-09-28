"""Produces reports/e12_replacement.txt.

E12: replacement-share re-print.
Kernels/stages are copied unchanged from gt_state_matrix.py (which copied pool_waterfall.py), so
the script is self-contained; gt_state_matrix.py executes at module level and must not be imported.
Per moderate GT with an accurate pool candidate (IoU3D>=0.7, |dz|<8m): did the BEST pool
candidate survive native eligibility? Outcomes among A-pass GTs:
  BEST_SURVIVED  best-IoU candidate row is among the eligibility survivors
  REPLACED       best dropped, but final output still has an accurate candidate (c>=0.7)
  LOST           best dropped and no accurate candidate in final (face-ii numerator)
GATE: counts must satisfy BEST_SURVIVED+REPLACED+LOST == A-pass, and LOST/A-pass must
equal the face-ii of reports/pool_waterfall.txt to 3 decimals; otherwise output is discarded.
Complete native pools need the pre-flatten query dumps and the floor-0 M3D-RPN dump in DUMP_DIR.
Run from the repository root: python tools/decomp/e12_replacement.py
"""
import os, sys, re, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import ROOT, paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
import ap_corrector_arc as arc

DIAG = cache_dir("decomp")   # work dirs + npz caches (dumps are read through dump_path)
OUT = out_path("e12_replacement.txt")
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def moderate_gts(sid):
    p = os.path.join(arc.LABEL_DIR, f"{sid:06d}.txt")
    res = []
    if not os.path.exists(p):
        return res
    for ln in open(p):
        t = ln.split()
        if t[0] != "Car":
            continue
        trunc, occ = float(t[1]), int(t[2])
        hpix = float(t[7]) - float(t[5])
        if occ <= 1 and trunc <= 0.3 and hpix > 25:
            res.append((float(t[8]), float(t[9]), float(t[10]), float(t[11]),
                        float(t[12]), float(t[13]), float(t[14])))
    return res


def best_iou_and_arg(g, gts):
    """per GT: (best IoU3D, dataframe index of argmax candidate or -1)."""
    if len(g) == 0:
        return [(0.0, -1)] * len(gts)
    px = g.x_3d.values; py = g.y_3d.values; pz = g.z_3d.values
    ph = g.h_3d.values; pw = g.w_3d.values; pl = g.l_3d.values; pr = g.ry.values
    idx = g.index.values
    res = []
    for t in gts:
        gp = arc.poly(t[3], t[5], t[1], t[2], t[6])
        cand = np.where(np.abs(pz - t[5]) < 8.0)[0]
        best, barg = 0.0, -1
        for k in cand:
            pp = arc.poly(px[k], pz[k], pw[k], pl[k], pr[k])
            v = arc.iou3d(pp, py[k], ph[k], pl[k], pw[k], t, gp)
            if v > best:
                best, barg = v, idx[k]
        res.append((best, barg))
    return res


def stages_for(name):
    if name in ("MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"):
        f = {"MonoDETR": "monodetr", "MonoDGP": "monodgp", "MonoCoP": "official_monocop",
             "MonoCLUE": "monoclue", "MonoIA": "monoia"}[name]
        df = pd.read_csv(os.path.join(paths.DUMP_DIR, f"{f}_val_preflatten.csv"))
        if "class_id" in df.columns:
            car = df[df.class_id == df.car_channel].copy()
            car["flatrank"] = df.loc[car.index, "flat_rank"]
        else:
            car = df.copy(); car["flatrank"] = car["flat_rank_car"]
            car["V"] = car["V_car"]; car["cls"] = car["cls_car"]
        pool = car
        elig = lambda d: d[(d.flatrank < 50) & (d.cls >= 0.2)]
        return pool, elig
    if name == "M3D-RPN":
        df = pd.read_csv(os.path.join(paths.DUMP_DIR, "m3drpn_val_floor0.csv"))
        pool = df

        def elig(d):
            if len(d) == 0:
                return d
            dd = d.sort_values("V", ascending=False)
            b = dd[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            keep = []
            order = np.arange(len(dd))
            while order.size > 0:
                i = order[0]; keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
                wq = np.maximum(0, xx2 - xx1 + 1); hq = np.maximum(0, yy2 - yy1 + 1)
                ovr = wq * hq / (areas[i] + areas[order[1:]] - wq * hq)
                order = order[np.where(ovr <= 0.4)[0] + 1]
            kk = dd.iloc[keep].head(40)
            return kk[kk["V"] >= 0.75]
        return pool, elig
    f = {"MonoDLE": "monodle", "MonoFlex": "monoflex", "GUPNet": "gupnet",
         "DEVIANT": "deviant", "MonoGround": "monoground", "MonoCon": "monocon"}[name]
    df = pd.read_csv(dump_path(f))
    thr_native = 0.4 if name == "MonoCon" else 0.2
    col = "cls" if name == "MonoDLE" else "V"
    pool = df
    elig = lambda d, c=col, t=thr_native: d[d[c] >= t]
    return pool, elig


MODELS = ["M3D-RPN", "MonoDLE", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround", "MonoCon",
          "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]
val = [int(x) for x in open(arc.VAL_LIST).read().split()]
ref = {}
for ln in open(os.path.join(ROOT, "reports", "pool_waterfall.txt")):
    m = re.match(r"\[done\] (\S+)\s+\(\w+\) GT=\d+ \| A@0.5=[\d.]+ A@0.7=([\d.]+) "
                 r"B@0.7=[\d.]+ C@0.7=[\d.]+ \| suppressed-acc\(face-ii share\)=([\d.]+)", ln)
    if m:
        ref[m.group(1)] = (float(m.group(1 + 1)), float(m.group(3)))

w(f"# e12_replacement run {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# among A-pass GTs (accurate pool candidate exists): did the BEST candidate survive?")
w("")
fail = False
for name in MODELS:
    pool, elig = stages_for(name)
    bysid = {s: g for s, g in pool.groupby("sid")}
    nA = nSurv = nRepl = nLost = 0
    for sid in val:
        gts = moderate_gts(sid)
        if not gts:
            continue
        g = bysid.get(sid, pool.iloc[0:0])
        ge = elig(g)
        ge_idx = set(ge.index.values)
        ab = best_iou_and_arg(g, gts)
        cb = best_iou_and_arg(ge, gts)
        for j in range(len(gts)):
            if ab[j][0] < 0.7:
                continue
            nA += 1
            if ab[j][1] in ge_idx:
                nSurv += 1
            elif cb[j][0] >= 0.7:
                nRepl += 1
            else:
                nLost += 1
    fii = nLost / max(nA, 1)
    ok = (nSurv + nRepl + nLost == nA) and abs(round(fii, 3) - ref[name][1]) < 1e-9
    w(f"[done] {name:10s} A-pass={nA:5d} | best-survived={nSurv/max(nA,1):.3f} "
      f"REPLACED={nRepl/max(nA,1):.3f} LOST={fii:.3f} (ref face-ii {ref[name][1]:.3f}) "
      f"-> {'GATE-PASS' if ok else 'GATE-FAIL'}")
    fail |= not ok
w("")
if fail:
    w("!! GATE FAILURE — discard this output (the frozen reports/pool_waterfall.txt stands) !!")
else:
    w("READ: REPLACED = suppression that is harmless at GT level (another accurate candidate")
    w("of the same GT reaches the final output). LOST equals the face-ii numerator by")
    w("construction; gate enforces 3-decimal equality with pool_waterfall.txt.")
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

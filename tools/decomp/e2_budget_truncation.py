"""Produces reports/final_run/e2_budget_truncation.txt.

E2: anchor candidate-budget truncation sensitivity (DIAGNOSTIC ONLY; it does not and
cannot remove the budget confound, and is reported as budget sensitivity).
Floor-0 M3D-RPN pool truncated per frame to native-score top-150 / top-50; recompute
official-moderate existence A@0.7/A@0.5 (waterfall kernels unchanged). Plus
hard-core-minus-anchor sensitivity from gt_state_matrix.csv (union over the 11 non-anchor
models; run gt_state_matrix.py first, it writes <CACHE_DIR>/decomp/gt_state_matrix.csv).
Needs the floor-0 M3D-RPN dump (m3drpn_val_floor0.csv) in DUMP_DIR.
Run from the repository root: python tools/decomp/e2_budget_truncation.py
"""
import os, sys, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
import ap_corrector_arc as arc

DIAG = cache_dir("decomp")   # work dirs + npz caches (dumps are read through dump_path)
OUT = out_path("final_run/e2_budget_truncation.txt")
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


def best_iou_per_gt(g, gts):
    if len(g) == 0:
        return [0.0] * len(gts)
    px = g.x_3d.values; py = g.y_3d.values; pz = g.z_3d.values
    ph = g.h_3d.values; pw = g.w_3d.values; pl = g.l_3d.values; pr = g.ry.values
    res = []
    for t in gts:
        gp = arc.poly(t[3], t[5], t[1], t[2], t[6])
        cand = np.where(np.abs(pz - t[5]) < 8.0)[0]
        best = 0.0
        for k in cand:
            pp = arc.poly(px[k], pz[k], pw[k], pl[k], pr[k])
            v = arc.iou3d(pp, py[k], ph[k], pl[k], pw[k], t, gp)
            if v > best:
                best = v
        res.append(best)
    return res


val = [int(x) for x in open(arc.VAL_LIST).read().split()]
w(f"# e2_budget_truncation run {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# DIAGNOSTIC: budget sensitivity only — does NOT remove the family-budget confound")
w("")
df = pd.read_csv(os.path.join(paths.DUMP_DIR, "m3drpn_val_floor0.csv"))
bysid = {s: g for s, g in df.groupby("sid")}
for K in (3000, 150, 50):
    nGT = nA7 = nA5 = 0
    far = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}  # bin -> [n, A7]
    for sid in val:
        gts = moderate_gts(sid)
        if not gts:
            continue
        g = bysid.get(sid, df.iloc[0:0])
        if K < 3000 and len(g) > K:
            g = g.nlargest(K, "V")
        a = best_iou_per_gt(g, gts)
        for j, t in enumerate(gts):
            nGT += 1
            zb = 0 if t[5] < 15 else 1 if t[5] < 30 else 2 if t[5] < 45 else 3
            far[zb][0] += 1
            if a[j] >= 0.5:
                nA5 += 1
            if a[j] >= 0.7:
                nA7 += 1; far[zb][1] += 1
    w(f"[K={K:4d}] A@0.7={nA7/nGT:.3f} A@0.5={nA5/nGT:.3f} | per-bin A7: " +
      " ".join(f"{far[i][1]/max(far[i][0],1):.3f}" for i in range(4)) +
      f"  (GT={nGT})")
w("")
w("reference: CenterNet native decoded pools A@0.7 = 0.271-0.316 (28-50 cands/img);")
w("query 150-hypothesis pools A@0.7 = 0.360-0.457.")
sm = pd.read_csv(f"{DIAG}/gt_state_matrix.csv")
non_anchor = ["MonoDLE", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround", "MonoCon",
              "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]
A11 = np.stack([(sm[f"{m}_a"] >= 0.7).values for m in non_anchor], 1)
hard11 = (~A11.any(1))
w(f"hard-core sensitivity: full-12 = 1176/7874 = 0.149; minus-anchor (11 models) = "
  f"{int(hard11.sum())}/7874 = {hard11.mean():.3f}")
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

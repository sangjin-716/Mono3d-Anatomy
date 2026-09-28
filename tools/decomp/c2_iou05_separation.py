"""Ported from tools/decomp/c2_iou05_separation.py for the public release. Computation unchanged.
Produces reports/final_run/c2_iou05_separation.txt.

C2 — does the E4 separation result (FP-demotion ~= full, TP-reorder ~0) survive at IoU0.5?
Copy-extend of e4_fp_tp_decomp.py (not modified): same native pools, same o_act cache,
but TP labels at IoU3D>=0.5 and AP evaluated at min_overlap=0.5. Tests whether the separation
is an IoU0.7-cliff artifact.
Native pools need, besides the released dumps, the pre-flatten query dumps
(<f>_val_preflatten.csv) and the floor-0 M3D-RPN dump (m3drpn_val_floor0.csv) in DUMP_DIR.
The o_act cache is the one written by e4_fp_tp_decomp.py (recomputed, not saved, if absent).
Run from the repository root: python tools/decomp/c2_iou05_separation.py
"""
import os, sys, shutil, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti
import evaluator.kitti_eval.kitti_common as kc
from exact_ap import ap_summaries
from depth_share_bridge import iou_act_and_zstar, DIAG

val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
WORK = f"{DIAG}/_c2work"
OUT = out_path("final_run/c2_iou05_separation.txt")
out = []
THR = 0.5  # the only change vs E4


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def annos_for(df, score):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    return kc.get_label_annos(d, val)


def gts_of(sid):
    p = os.path.join(arc.LABEL_DIR, f"{sid:06d}.txt"); res = []
    if not os.path.exists(p):
        return res
    for ln in open(p):
        t = ln.split()
        if t[0] == "Car":
            res.append((float(t[8]), float(t[9]), float(t[10]), float(t[11]),
                        float(t[12]), float(t[13]), float(t[14])))
    return res


def tp_labels(df, thr):
    lab = np.zeros(len(df), bool)
    px = df.x_3d.values; py = df.y_3d.values; pz = df.z_3d.values
    ph = df.h_3d.values; pw = df.w_3d.values; pl = df.l_3d.values; pr = df.ry.values
    for sid, g in df.groupby("sid"):
        gts = gts_of(int(sid))
        if not gts:
            continue
        order = g.index.values[np.argsort(-g.V.values, kind="stable")]
        used = [False] * len(gts)
        gps = [arc.poly(t[3], t[5], t[1], t[2], t[6]) for t in gts]
        for i in order:
            best, barg = 0.0, -1
            for j, t in enumerate(gts):
                if used[j] or abs(pz[i] - t[5]) >= 8.0:
                    continue
                pp = arc.poly(px[i], pz[i], pw[i], pl[i], pr[i])
                v = arc.iou3d(pp, py[i], ph[i], pl[i], pw[i], t, gps[j])
                if v > best:
                    best, barg = v, j
            if best >= thr:
                used[barg] = True; lab[df.index.get_loc(i)] = True
    return lab


def native_pool(name):
    if name in ("MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"):
        f = {"MonoDETR": "monodetr", "MonoDGP": "monodgp", "MonoCoP": "official_monocop",
             "MonoCLUE": "monoclue", "MonoIA": "monoia"}[name]
        df = pd.read_csv(os.path.join(paths.DUMP_DIR, f"{f}_val_preflatten.csv"))
        if "class_id" in df.columns:
            car = df[df.class_id == df.car_channel].copy(); car["flatrank"] = df.loc[car.index, "flat_rank"]
        else:
            car = df.copy(); car["flatrank"] = car["flat_rank_car"]; car["V"] = car["V_car"]; car["cls"] = car["cls_car"]
        return car[(car.flatrank < 50) & (car.cls >= 0.2)].reset_index(drop=True)
    if name == "M3D-RPN":
        df = pd.read_csv(os.path.join(paths.DUMP_DIR, "m3drpn_val_floor0.csv")); outp = []
        for sid, d in df.groupby("sid"):
            dd = d.sort_values("V", ascending=False)
            b = dd[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]; areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            keep = []; order = np.arange(len(dd))
            while order.size > 0:
                i = order[0]; keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
                wq = np.maximum(0, xx2 - xx1 + 1); hq = np.maximum(0, yy2 - yy1 + 1)
                ovr = wq * hq / (areas[i] + areas[order[1:]] - wq * hq)
                order = order[np.where(ovr <= 0.4)[0] + 1]
            kk = dd.iloc[keep].head(40); outp.append(kk[kk["V"] >= 0.75])
        return pd.concat(outp).reset_index(drop=True)
    f = {"MonoDLE": "monodle", "MonoFlex": "monoflex", "GUPNet": "gupnet",
         "DEVIANT": "deviant", "MonoGround": "monoground", "MonoCon": "monocon"}[name]
    df = pd.read_csv(dump_path(f)); thr = 0.4 if name == "MonoCon" else 0.2
    col = "cls" if name == "MonoDLE" else "V"
    return df[df[col] >= thr].reset_index(drop=True)


MODELS = ["M3D-RPN", "MonoDLE", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround", "MonoCon",
          "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]
FAM = {"M3D-RPN": "anchor", "MonoDETR": "query", "MonoDGP": "query", "MonoCoP": "query",
       "MonoCLUE": "query", "MonoIA": "query"}

w(f"# c2_iou05_separation run {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# E4 decomposition re-evaluated at IoU3D 0.5 (TP labels @0.5, AP min_overlap=0.5);")
w("# question: does FP-demotion still reproduce the full oracle and TP-reorder ~0 at the looser bar?")
w("")
res = []
for name in MODELS:
    pool = native_pool(name)
    cache = f"{DIAG}/_e4cache_{name.replace('-', '')}.npz"
    o_act = np.load(cache)["o_act"] if os.path.exists(cache) else iou_act_and_zstar(pool)[0]
    assert len(o_act) == len(pool)
    lab = tp_labels(pool, THR)
    V = pool["V"].values.astype(float); BIG = V.max() + 1.0
    s_fpd = np.where(lab, V + BIG, V)
    s_tpr = V.copy(); tp = np.where(lab)[0]
    if len(tp):
        slot = np.sort(V[tp])[::-1]; order = tp[np.lexsort((-V[tp], -o_act[tp]))]; s_tpr[order] = slot
    ap = {}
    for tag, sc in [("base", V), ("fpd", s_fpd), ("tpr", s_tpr), ("full", o_act)]:
        ap[tag] = ap_summaries(GT, annos_for(pool, sc), min_overlap=THR)["allpoint"]
    res.append((name, ap, lab.mean()))
    w(f"[done] {name:10s} base05={ap['base']:6.2f} FPdem=+{ap['fpd']-ap['base']:5.2f} "
      f"TPreord=+{ap['tpr']-ap['base']:5.2f} full=+{ap['full']-ap['base']:5.2f} | TPshare05={lab.mean():.3f}")
w("")
for fam in ("anchor", "cnet", "query"):
    s = [r for r in res if FAM.get(r[0], "cnet") == fam]
    if s:
        med = lambda k: float(np.median([r[1][k] - r[1]["base"] for r in s]))
        w(f"{fam:6s} n={len(s)} FPdem med +{med('fpd'):.2f}  TPreord med +{med('tpr'):.2f}  full med +{med('full'):.2f}")
w("")
w("READ: if FPdem still ~= full and TPreord ~0 at IoU0.5 -> separation finding is NOT an")
w("IoU0.7-cliff artifact (robust). If TPreord grows positive -> separation is partly")
w("threshold-specific; scope the claim to IoU0.7.")
if os.path.exists(WORK):
    shutil.rmtree(WORK)
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

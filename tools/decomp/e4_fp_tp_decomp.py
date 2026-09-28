"""Ported from tools/decomp/e4_fp_tp_decomp.py for the public release. Computation unchanged.
Produces reports/e4_fp_tp_decomp.txt (and the _e4cache_<model>.npz files that
exp1_true_ceiling.py, c2_iou05_separation.py and diffsweep_headroom_sep.py reuse).

E4 — FP-demotion vs TP-quality-reordering decomposition of the ordering gap.
Definitions frozen in reports/e4_prereg.md before this run. Native pools,
exact_ap evaluator, o_act via iou_act_and_zstar, TP/FP labels frozen from baseline order.
Native pools need, besides the released dumps, the pre-flatten query dumps
(<f>_val_preflatten.csv) and the floor-0 M3D-RPN dump (m3drpn_val_floor0.csv) in DUMP_DIR.
Run from the repository root: python tools/decomp/e4_fp_tp_decomp.py
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
WORK = f"{DIAG}/_e4work"
OUT = out_path("e4_fp_tp_decomp.txt")
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def annos_for(df, score):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    return kc.get_label_annos(d, val)


def gts_of(sid):
    p = os.path.join(arc.LABEL_DIR, f"{sid:06d}.txt")
    res = []
    if not os.path.exists(p):
        return res
    for ln in open(p):
        t = ln.split()
        if t[0] == "Car":   # any difficulty (prereg: labels vs ALL Car GTs)
            res.append((float(t[8]), float(t[9]), float(t[10]), float(t[11]),
                        float(t[12]), float(t[13]), float(t[14])))
    return res


def tp_labels(df):
    """frozen baseline labels: per frame, V-desc greedy 1-to-1 match at IoU3D>=0.7."""
    lab = np.zeros(len(df), bool)
    pos = {s: g for s, g in df.groupby("sid")}
    for sid, g in pos.items():
        gts = gts_of(int(sid))
        if not gts:
            continue
        order = g.index.values[np.argsort(-g.V.values, kind="stable")]
        used = [False] * len(gts)
        px = df.x_3d; py = df.y_3d; pz = df.z_3d
        ph = df.h_3d; pw = df.w_3d; pl = df.l_3d; pr = df.ry
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
            if best >= 0.7:
                used[barg] = True; lab[df.index.get_loc(i)] = True
    return lab


def native_pool(name):
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
        pred = (car.flatrank < 50) & (car.cls >= 0.2)
        if "in_final" in car.columns:
            assert (pred == (car.in_final == 1)).all(), f"in_final mismatch {name}"
        return car[pred].reset_index(drop=True)
    if name == "M3D-RPN":
        df = pd.read_csv(os.path.join(paths.DUMP_DIR, "m3drpn_val_floor0.csv"))
        outp = []
        for sid, d in df.groupby("sid"):
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
            outp.append(kk[kk["V"] >= 0.75])
        return pd.concat(outp).reset_index(drop=True)
    f = {"MonoDLE": "monodle", "MonoFlex": "monoflex", "GUPNet": "gupnet",
         "DEVIANT": "deviant", "MonoGround": "monoground", "MonoCon": "monocon"}[name]
    df = pd.read_csv(dump_path(f))
    thr = 0.4 if name == "MonoCon" else 0.2
    col = "cls" if name == "MonoDLE" else "V"
    return df[df[col] >= thr].reset_index(drop=True)


BASE_REF = {"MonoDETR": 21.18, "MonoDGP": 22.82, "MonoCoP": 24.34, "MonoCLUE": 24.55,
            "MonoIA": 25.18, "MonoDLE": 15.09, "MonoFlex": 16.27, "GUPNet": 17.11,
            "DEVIANT": 17.48, "MonoGround": 17.47, "MonoCon": 19.59, "M3D-RPN": 11.49}
GAP_REF = {"MonoDETR": (11.42, 0.10), "MonoDGP": (8.41, 0.10), "MonoCoP": (11.58, 0.10),
           "MonoCLUE": (9.73, 0.10), "MonoIA": (11.64, 0.10), "MonoDLE": (14.03, 0.10),
           "MonoFlex": (10.28, 0.10), "GUPNet": (12.04, 0.10), "DEVIANT": (11.83, 0.10),
           "MonoGround": (10.80, 0.10), "MonoCon": (11.36, 0.40), "M3D-RPN": (11.60, 0.40)}
MODELS = ["M3D-RPN", "MonoDLE", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround", "MonoCon",
          "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]
FAM = {"M3D-RPN": "anchor", "MonoDETR": "query", "MonoDGP": "query", "MonoCoP": "query",
       "MonoCLUE": "query", "MonoIA": "query"}

w(f"# e4_fp_tp_decomp run {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# prereg: reports/e4_prereg.md (definitions frozen before run); native pools;")
w("# labels frozen from baseline V order (arc matcher, any-difficulty Car GTs, IoU3D>=0.7)")
w("")
res = []
for name in MODELS:
    pool = native_pool(name)
    cache = f"{DIAG}/_e4cache_{name.replace('-','')}.npz"
    if os.path.exists(cache):
        z = np.load(cache); o_act = z["o_act"]; lab = z["lab"].astype(bool)
        assert len(o_act) == len(pool)
    else:
        o_act, _ = iou_act_and_zstar(pool)
        lab = tp_labels(pool)
        np.savez_compressed(cache, o_act=o_act, lab=lab)
    V = pool["V"].values.astype(float)
    BIG = V.max() + 1.0
    s_fpd = np.where(lab, V + BIG, V)
    s_tpr = V.copy()
    tp_idx = np.where(lab)[0]
    if len(tp_idx):
        slot = np.sort(V[tp_idx])[::-1]
        order = tp_idx[np.lexsort((-V[tp_idx], -o_act[tp_idx]))]
        s_tpr[order] = slot
    aps = {}
    for tag, sc in [("base", V), ("fpd", s_fpd), ("tpr", s_tpr), ("full", o_act)]:
        s = ap_summaries(GT, annos_for(pool, sc))
        aps[tag] = (s["allpoint"], s["r40_official"])
    gb_ok = abs(aps["base"][0] - BASE_REF[name]) <= 0.05
    gfull = aps["full"][0] - aps["base"][0]
    ref, tol = GAP_REF[name]
    gg_ok = abs(gfull - ref) <= tol
    res.append((name, aps, lab.mean()))
    w(f"[done] {name:10s} GA={'PASS' if gb_ok else 'FAIL'} GB={'PASS' if gg_ok else 'FAIL'} "
      f"| base={aps['base'][0]:6.2f}({aps['base'][1]:5.2f}) "
      f"FPdem=+{aps['fpd'][0]-aps['base'][0]:5.2f} TPreord=+{aps['tpr'][0]-aps['base'][0]:5.2f} "
      f"full=+{gfull:5.2f} (ref +{ref:.2f}) | TPshare={lab.mean():.3f}")

w("")
w("== family medians (all-point gaps) ==")
for fam in ("anchor", "cnet", "query"):
    sel = [r for r in res if FAM.get(r[0], "cnet") == fam]
    if not sel:
        continue
    med = lambda k: float(np.median([r[1][k][0] - r[1]["base"][0] for r in sel]))
    w(f"{fam:6s} n={len(sel)}  FPdem={med('fpd'):+.2f}  TPreord={med('tpr'):+.2f}  full={med('full'):+.2f}")
w("")
w("READ: no additivity assumed; interpretation rules fixed in e4_prereg.md. GA/GB gate")
w("failures void interpretation for that model (disclosed, not absorbed).")
if os.path.exists(WORK):
    shutil.rmtree(WORK)
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

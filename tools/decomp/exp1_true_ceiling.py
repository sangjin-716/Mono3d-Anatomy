"""Produces reports/exp1_true_ceiling.txt.

EXP-1: the TRUE fixed-pool ceiling AP* = M / n_gt.

For each detector, on the SAME native pool e4_fp_tp_decomp uses, compute the maximum
bipartite matching M between valid (Moderate) GTs and pool predictions at IoU3D>=0.7
(official overlaps + official _prepare_data valid-set), so ceiling recall = sum(M)/sum(valid_gt)
and ceiling all-point AP = 100*ceiling_recall (perfect-separation upper bound, exact max over
labelings). Compare to native, true-IoU re-sort (the paper's "ceiling"), FP-demotion.

The reported true-IoU "ceiling" is a LOWER bound; this prints how far below the true
ceiling it sits.

Run e4_fp_tp_decomp.py first: this script reads its _e4cache_<model>.npz (o_act + fixed TP
labels); without the cache the FPdem column is nan. Native pools need the pre-flatten query
dumps and the floor-0 M3D-RPN dump in DUMP_DIR. No training.
Run from the repository root: python tools/decomp/exp1_true_ceiling.py
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
from evaluator.kitti_eval.eval import calculate_iou_partly, _prepare_data
from depth_share_bridge import DIAG, iou_act_and_zstar

val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
WORK = f"{DIAG}/_exp1work"
OUT = out_path("exp1_true_ceiling.txt")
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def annos_for(df, score):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    return kc.get_label_annos(d, val)


def native_pool(name):   # verbatim from e4_fp_tp_decomp.py
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


def max_matching(rows, ng):
    """Kuhn's algorithm. rows[d] = list of gt-indices adjacent to det d. Returns max matching."""
    matchG = [-1] * ng

    def try_kuhn(d, seen):
        for g in rows[d]:
            if not seen[g]:
                seen[g] = True
                if matchG[g] == -1 or try_kuhn(matchG[g], seen):
                    matchG[g] = d
                    return True
        return False

    M = 0
    for d in range(len(rows)):
        if try_kuhn(d, [False] * ng):
            M += 1
    return M


def ceiling_recall(dt_annos, name):
    """exact max #valid-GT simultaneously matchable to pool preds at IoU3D>=0.7 / #valid-GT."""
    overlaps, _, _, _ = calculate_iou_partly(dt_annos, GT, 2)          # metric=2 -> 3D IoU
    (_, _, ig_gts, ig_dets, _, _, n_valid_gt) = _prepare_data(GT, dt_annos, 0, 1)  # Car, Moderate
    sumM = 0
    for i in range(len(GT)):
        ov = overlaps[i]
        ig = np.asarray(ig_gts[i]); idt = np.asarray(ig_dets[i])
        # orient overlaps to (n_dt, n_gt)
        if ov.shape == (len(idt), len(ig)):
            pass
        elif ov.shape == (len(ig), len(idt)):
            ov = ov.T
        else:
            raise AssertionError(f"{name} img{i}: overlaps {ov.shape} vs dt{len(idt)} gt{len(ig)}")
        vdt = np.where(idt == 0)[0]; vgt = np.where(ig == 0)[0]
        if len(vdt) == 0 or len(vgt) == 0:
            continue
        sub = ov[np.ix_(vdt, vgt)] >= 0.7
        rows = [np.where(sub[d])[0].tolist() for d in range(len(vdt))]
        sumM += max_matching(rows, len(vgt))
    return sumM, int(n_valid_gt)


MODELS = ["M3D-RPN", "MonoDLE", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround", "MonoCon",
          "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]

w(f"# exp1_true_ceiling {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# AP* = 100 * sum(M)/sum(valid_gt) ; M = max bipartite matching pool-pred x valid(Mod)-GT @ IoU3D>=0.7")
w("# AP* is the EXACT upper bound on evaluator-realizable AP (max over labelings); native/true-IoU/FP-dem are LOWER bounds.")
w("# allpoint AP, Car, Moderate, IoU0.7. pools verbatim from e4_fp_tp_decomp.native_pool.")
w("")
w(f"{'detector':10s} {'base':>6s} {'trueIoU':>7s} {'FPdem':>6s} {'CEIL*':>6s} {'recall*':>7s} "
  f"{'ceil-trueIoU':>12s} {'eff_base':>8s} {'eff_trueIoU':>11s}")
rows_out = []
for name in MODELS:
    pool = native_pool(name)
    cache = f"{DIAG}/_e4cache_{name.replace('-','')}.npz"
    if os.path.exists(cache):
        z = np.load(cache); o_act = z["o_act"]; lab = z["lab"].astype(bool)
        assert len(o_act) == len(pool), f"cache len {len(o_act)} vs pool {len(pool)} ({name})"
    else:
        o_act, _ = iou_act_and_zstar(pool); lab = None  # lab not needed if we skip FPdem
    V = pool["V"].values.astype(float)
    dt_base = annos_for(pool, V)
    base_ap = ap_summaries(GT, dt_base)["allpoint"]
    sumM, n_gt = ceiling_recall(dt_base, name)
    ceil_ap = 100.0 * sumM / n_gt
    full_ap = ap_summaries(GT, annos_for(pool, o_act))["allpoint"]          # true-IoU re-sort
    if lab is not None:
        s_fpd = np.where(lab, V + V.max() + 1.0, V)
        fpd_ap = ap_summaries(GT, annos_for(pool, s_fpd))["allpoint"]
    else:
        fpd_ap = float("nan")
    rows_out.append((name, base_ap, full_ap, fpd_ap, ceil_ap, sumM / n_gt))
    w(f"{name:10s} {base_ap:6.2f} {full_ap:7.2f} {fpd_ap:6.2f} {ceil_ap:6.2f} {sumM/n_gt:7.3f} "
      f"{ceil_ap-full_ap:+12.2f} {base_ap/ceil_ap:8.3f} {full_ap/ceil_ap:11.3f}")

w("")
arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows_out])
w("== panel medians (all-point AP) ==")
w(f"base={np.median(arr[:,0]):.2f}  trueIoU={np.median(arr[:,1]):.2f}  FPdem={np.nanmedian(arr[:,2]):.2f}  "
  f"CEIL*={np.median(arr[:,3]):.2f}")
w(f"median (CEIL* - trueIoU) understatement = {np.median(arr[:,3]-arr[:,1]):+.2f} AP")
w(f"median (CEIL* - FPdem)                  = {np.nanmedian(arr[:,3]-arr[:,2]):+.2f} AP")
w(f"median native efficiency  base/CEIL*    = {np.median(arr[:,0]/arr[:,3]):.3f}")
w(f"median reported efficiency trueIoU/CEIL*= {np.median(arr[:,1]/arr[:,3]):.3f}")
w("")
w("READ: trueIoU re-sort (paper's headline 'ceiling') is a LOWER bound; CEIL*=M/n_gt is the exact")
w("max-over-labelings upper bound. (CEIL* - trueIoU) is how much the reported number UNDERSTATES the")
w("true fixed-pool re-ranking ceiling. FP-dem<=CEIL* and trueIoU<=CEIL* must hold (sanity).")
if os.path.exists(WORK):
    shutil.rmtree(WORK)
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

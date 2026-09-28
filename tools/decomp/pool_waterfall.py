"""Produces reports/pool_waterfall.txt.

Pool waterfall, per family and per model:
candidate existence → eligibility → budget → final ranking, on COMPLETE native pools.

Data sources (all in DUMP_DIR):
  query family : validated v2 pre-flatten dumps <f>_val_preflatten.csv (Car hypotheses = every
                 query's Car channel)
  M3D-RPN      : floor-0 dump m3drpn_val_floor0.csv (complete native pre-NMS top-3000
                 Car-argmax pool; validated)
  CenterNet    : released full dumps (top-K heatmap pools incl sub-threshold; validated)
The pre-flatten and floor-0 dumps are not among the 14 released val dumps.

Per moderate-GT (GT-side, candidate gate = |z_cand − z_gt| < 8 m; SAFE for IoU≥0.5 existence
since IoU≥0.5 forces small Δz):
  stage A existence  : best IoU3D over the COMPLETE pool   (acc@0.5 / acc@0.7 flags)
  stage B eligibility: best IoU3D over eligibility survivors (native thr; M3D-RPN: writer 0.75
                       + its native NMS@0.4+top40; query: K_flat50∧cls≥0.2; CenterNet: thr0.2)
  stage C final      : found in final output at IoU≥0.7 under native ranking
Faces per GT (moderate): MISSING = no accurate candidate in pool (A fails);
SUPPRESSED-ACC = A passes (IoU≥0.7 candidate exists) but C fails; among suppressed,
REPLACED = another same-GT candidate survives to final with IoU≥0.7 (by construction
suppressed-acc & C-fail means NOT replaced; replacement is counted at the candidate level:
the best pool candidate was dropped but a different candidate of the same GT made C pass →
those GTs are NOT counted suppressed; so non-replacement is explicit).
Outputs per model: waterfall counts overall + far bins; face-(ii) share =
(A-pass ∧ C-fail GTs) / A-pass GTs.
Run from the repository root: python tools/decomp/pool_waterfall.py
"""
import os, sys, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import apply_nms

DIAG = cache_dir("decomp")   # work dirs + npz caches (dumps are read through dump_path)
BINS = [(0, 15), (15, 30), (30, 45), (45, 1e9)]
OUT = out_path("pool_waterfall.txt")
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def moderate_gts(sid):
    """moderate-filter Car GTs: occ<=1, trunc<=0.3, bbox h>25px (official moderate)."""
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
    """g: candidate frame subset (DataFrame with box cols); returns per-GT best IoU3D."""
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


def stages_for(name):
    """return (pool_df, eligibility_fn, final_fn) per model with native semantics."""
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
        final = elig                                   # no NMS/cap downstream
        return pool, elig, final
    if name == "M3D-RPN":
        df = pd.read_csv(os.path.join(paths.DUMP_DIR, "m3drpn_val_floor0.csv"))
        pool = df

        def elig(d):                                   # native: NMS@0.4 -> top40 -> 0.75
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
        return pool, elig, elig
    # CenterNet family: established full dumps; native = thr (no box NMS)
    f = {"MonoDLE": "monodle", "MonoFlex": "monoflex", "GUPNet": "gupnet",
         "DEVIANT": "deviant", "MonoGround": "monoground", "MonoCon": "monocon"}[name]
    df = pd.read_csv(dump_path(f))
    thr_native = 0.4 if name == "MonoCon" else 0.2
    col = "cls" if name == "MonoDLE" else "V"   # MonoDLE natively thresholds the heatmap score
    pool = df
    elig = lambda d, c=col, t=thr_native: d[d[c] >= t]
    return pool, elig, elig


MODELS = ["M3D-RPN", "MonoDLE", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround", "MonoCon",
          "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]
FAM = {"M3D-RPN": "anchor", "MonoDETR": "query", "MonoDGP": "query", "MonoCoP": "query",
       "MonoCLUE": "query", "MonoIA": "query"}
val = [int(x) for x in open(arc.VAL_LIST).read().split()]

w(f"# pool_waterfall run {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# GT-side per moderate Car GT; candidate gate |dz|<8m (safe for IoU>=0.5 existence);")
w("# A=existence in COMPLETE pool, B=eligibility survivors, C=final output; IoU3D thresholds 0.5/0.7")
w("")
for name in MODELS:
    fam = FAM.get(name, "cnet")
    pool, elig, final = stages_for(name)
    bysid = {s: g for s, g in pool.groupby("sid")}
    nA5 = nA7 = nB7 = nC7 = nGT = 0
    far = {i: [0, 0, 0, 0] for i in range(len(BINS))}   # per bin: nGT, A7, C7, A5
    supp7 = 0
    for sid in val:
        gts = moderate_gts(sid)
        if not gts:
            continue
        g = bysid.get(sid, pool.iloc[0:0])
        ge = elig(g); gf = final(g)
        a = best_iou_per_gt(g, gts)
        b = best_iou_per_gt(ge, gts)
        c = b if gf is ge or len(gf) == len(ge) else best_iou_per_gt(gf, gts)
        for j, t in enumerate(gts):
            nGT += 1
            zb = next(i for i, (lo, hi) in enumerate(BINS) if lo <= t[5] < hi)
            far[zb][0] += 1
            if a[j] >= 0.5:
                nA5 += 1; far[zb][3] += 1
            if a[j] >= 0.7:
                nA7 += 1; far[zb][1] += 1
                if c[j] < 0.7:
                    supp7 += 1
            if b[j] >= 0.7:
                nB7 += 1
            if c[j] >= 0.7:
                nC7 += 1; far[zb][2] += 1
    fii = supp7 / max(nA7, 1)
    w(f"[done] {name:10s} ({fam}) GT={nGT} | A@0.5={nA5/nGT:.3f} A@0.7={nA7/nGT:.3f} "
      f"B@0.7={nB7/nGT:.3f} C@0.7={nC7/nGT:.3f} | suppressed-acc(face-ii share)={fii:.3f}")
    fb = " ".join(f"[{lo}-{int(hi) if hi<1e9 else '+'}] n={far[i][0]} A7={far[i][1]/max(far[i][0],1):.2f} "
                  f"C7={far[i][2]/max(far[i][0],1):.2f}" for i, (lo, hi) in enumerate(BINS))
    w(f"        bins: {fb}")
    # far-field missing-share among 30m+ misses: no pool candidate IoU>=0.5
    miss30 = sum(far[i][0] - far[i][2] for i in (2, 3))
    noc30 = sum(far[i][0] - far[i][3] for i in (2, 3))
    w(f"        30m+ missed GTs={miss30}, of which NO pool candidate IoU>=0.5: "
      f"{noc30} ({noc30/max(miss30,1):.2f})")
    w("")

w("READ: rubric (claim_decision_tree #5): S needs 30m+ missing-share >= 0.70 (panel median),")
w("non-replacement counted via face-ii construction, face-ii share >= 0.10 for >= 8/12.")
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

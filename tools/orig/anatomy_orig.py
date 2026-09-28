"""Ported from mono3d_crossdataset/tools/anatomy_orig.py for the public release. Computation
unchanged. Produces reports_orig/oracle_anatomy_orig.txt.

ORACLE ANATOMY LADDER (GATE-1) for the original-environment MonoFlex*/MonoGround* dumps
(a copy of tools/decomp/oracle_anatomy.py with inputs switched) — what is the "+oracle-depth"
ceiling made of?

Per detector, S5 kept pool, native V scores kept FIXED; geometry oracles only:
  base       : boxes as-is
  pure_z     : z -> matched-GT z; x,y FROZEN (the 1-D range fix; NO lateral correction)
  ray        : ray-slide to GT z (x,y slide along camera ray) — the paper's existing lever
  centre     : (x,y,z) -> GT 3D centre (dims/yaw kept)
  y_only     : y -> GT y (elevation only)
  dims       : (h,w,l) -> GT dims (position/yaw kept)
  yaw        : ry -> GT ry
  allfac     : centre+dims+yaw (full GT replacement; IoU=1 for matched boxes)
Replacements apply to boxes with a matched GT (best IoU3D > 0 over ALL GTs, no z-gate);
unmatched boxes stay unchanged. AP = all-point interpolated (R40 secondary) via validated
exact_ap. Per-bin direction (pre-declared operationalization, set BEFORE results): per
GT-distance bin, recall@IoU0.7 gain of pure_z vs base must be > 0 (counts per detector).

Output: per-detector G(variant), ratio=G_z/G_centre, |ray - pure_z|, single-factor sizes,
sub-additivity |sum(singles) - allfac|, per-bin direction table (the GATE-1 rubric inputs).
Run from the repository root: python tools/orig/anatomy_orig.py
"""
import os, sys, shutil, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
from tools.orig._orig_common import ORIG
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti, pool_mask, apply_nms
from evaluator.kitti_utils import Calibration
import evaluator.kitti_eval.kitti_common as kc
from exact_ap import ap_summaries

DETS = [("MonoFlex*", "monoflex_orig"),
        ("MonoGround*", "monoground_orig")]
VARIANTS = ["pure_z", "ray", "centre", "y_only", "dims", "yaw", "allfac"]
BINS = [(0, 15), (15, 30), (30, 45), (45, 1e9)]
val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
W = f"{ORIG}/_anatomy_orig"
OUT = out_path("oracle_anatomy_orig.txt")
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def ap_of(df, cols, score):
    d2 = df.copy()
    for k, v in cols.items():
        d2[k] = v
    if os.path.exists(W):
        shutil.rmtree(W)
    dd = os.path.join(W, "data")
    write_kitti(d2.reset_index(drop=True), np.asarray(score, float), dd, val)
    s = ap_summaries(GT, kc.get_label_annos(dd, val))
    return s["allpoint"], s["r40_official"]


def match_and_variants(kept):
    """per box: best GT (IoU3D>0, all GTs) + variant geometries + variant IoU vs that GT."""
    n = len(kept)
    cols = {v: {"x_3d": kept.x_3d.values.copy(), "y_3d": kept.y_3d.values.copy(),
                "z_3d": kept.z_3d.values.copy(), "h_3d": kept.h_3d.values.copy(),
                "w_3d": kept.w_3d.values.copy(), "l_3d": kept.l_3d.values.copy(),
                "ry": kept.ry.values.copy()} for v in VARIANTS}
    iou_v = {v: np.zeros(n) for v in (["base"] + VARIANTS)}
    gt_key = np.array([""] * n, dtype=object)
    gt_z = np.full(n, np.nan)
    for sid, g in kept.groupby("sid"):
        gts = arc.read_gt(int(sid))
        if not gts:
            continue
        cal = Calibration(os.path.join(arc.CALIB_DIR, f"{int(sid):06d}.txt"))
        tx, ty = cal.tx, cal.ty
        gps = [arc.poly(t[3], t[5], t[1], t[2], t[6]) for t in gts]
        for k in g.index.values:
            x0, y0, z0 = kept.at[k, "x_3d"], kept.at[k, "y_3d"], kept.at[k, "z_3d"]
            h3, w3, l3, ry = kept.at[k, "h_3d"], kept.at[k, "w_3d"], kept.at[k, "l_3d"], kept.at[k, "ry"]
            if z0 <= 0.1:
                continue
            pp = arc.poly(x0, z0, w3, l3, ry)
            best, bj = 0.0, -1
            for j, t in enumerate(gts):
                v_ = arc.iou3d(pp, y0, h3, l3, w3, t, gps[j])
                if v_ > best:
                    best, bj = v_, j
            iou_v["base"][k] = best
            if bj < 0 or best <= 0.0:
                continue
            t = gts[bj]; gp = gps[bj]
            gx, gy, gz = t[4], None, t[5]
            # gts tuple layout from arc.read_gt: (h, w, l?, x, y?, z, ry)? -> use poly args
            # arc.poly(t[3], t[5], t[1], t[2], t[6]) = poly(x=t3, z=t5, w=t1, l=t2, ry=t6)
            gh, gw, gl = t[0], t[1], t[2]
            gx, gz, gry = t[3], t[5], t[6]
            gy = t[4]
            gt_key[k] = f"{int(sid)}:{bj}"; gt_z[k] = gz
            s = gz / z0
            geo = {
                "pure_z": (x0, y0, gz, h3, w3, l3, ry),
                "ray": (tx + (x0 - tx) * s, (h3 / 2 + ty) + (y0 - h3 / 2 - ty) * s, gz,
                        h3, w3, l3, ry),
                "centre": (gx, gy, gz, h3, w3, l3, ry),
                "y_only": (x0, gy, z0, h3, w3, l3, ry),
                "dims": (x0, y0, z0, gh, gw, gl, ry),
                "yaw": (x0, y0, z0, h3, w3, l3, gry),
                "allfac": (gx, gy, gz, gh, gw, gl, gry),
            }
            for vname, (vx, vy, vz, vh, vw, vl, vr) in geo.items():
                c = cols[vname]
                c["x_3d"][k] = vx; c["y_3d"][k] = vy; c["z_3d"][k] = vz
                c["h_3d"][k] = vh; c["w_3d"][k] = vw; c["l_3d"][k] = vl; c["ry"][k] = vr
                ppv = arc.poly(vx, vz, vw, vl, vr)
                iou_v[vname][k] = arc.iou3d(ppv, vy, vh, vl, vw, t, gp)
    return cols, iou_v, gt_key, gt_z


def perbin_recall07(kept, iou_arr, gt_key, gt_z):
    """per GT-distance bin: #GTs whose best assigned-box IoU under the variant >= 0.7."""
    df = pd.DataFrame({"k": gt_key, "z": gt_z, "i": iou_arr})
    df = df[df.k != ""]
    bestper = df.groupby("k").agg(z=("z", "first"), i=("i", "max"))
    res = []
    for lo, hi in BINS:
        m = (bestper.z >= lo) & (bestper.z < hi)
        res.append(int((bestper.i[m] >= 0.7).sum()))
    return res


w(f"# oracle_anatomy run {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# variants on matched boxes only (best-IoU GT, no z-gate); native V scores fixed")
w("# AP = all-point interpolated (R40 in parens); per-bin = recall@IoU0.7 by GT z-bin")
w("")
rows = []
for name, f in DETS:
    df = pd.read_csv(dump_path(f)).reset_index(drop=True)
    pre = df[pool_mask(df, "thr0.2")].copy()
    keep = apply_nms(pre, pre["V"].values, 0.5)
    kept = pre[keep].copy().reset_index(drop=True)
    V = kept["V"].values.astype(float)
    cols, iou_v, gt_key, gt_z = match_and_variants(kept)
    base_a, base_r = ap_of(kept, {}, V)
    res = {"base": (base_a, base_r)}
    for vname in VARIANTS:
        res[vname] = ap_of(kept, cols[vname], V)
    rb = {vn: perbin_recall07(kept, iou_v[vn], gt_key, gt_z) for vn in ["base", "pure_z", "centre"]}
    G = {vn: res[vn][0] - base_a for vn in VARIANTS}
    ratio = G["pure_z"] / G["centre"] if G["centre"] > 1e-9 else float("nan")
    subadd = abs((G["centre"] + G["dims"] + G["yaw"]) - G["allfac"])
    rows.append(dict(name=name, base=base_a, G=G, ratio=ratio, rb=rb, subadd=subadd))
    w(f"[done] {name:10s} base={base_a:6.2f} | Gz={G['pure_z']:+6.2f} ray={G['ray']:+6.2f} "
      f"centre={G['centre']:+6.2f} y={G['y_only']:+5.2f} dims={G['dims']:+5.2f} "
      f"yaw={G['yaw']:+5.2f} all={G['allfac']:+6.2f} | ratio={ratio:.3f} "
      f"|ray-z|={abs(G['ray']-G['pure_z']):.2f} subadd={subadd:.2f}")
    w(f"        perbin recall@0.7 base={rb['base']} pure_z={rb['pure_z']} centre={rb['centre']}")

w("")
w("=" * 100)
w("GATE-1 rubric inputs (claim_decision_tree.md):")
gz = np.array([r["G"]["pure_z"] for r in rows]); gc = np.array([r["G"]["centre"] for r in rows])
ra = np.array([r["ratio"] for r in rows])
w(f"  G_z panel median = {np.median(gz):+.2f}   (S needs >= 10.0; P needs >= 5.0)")
w(f"  ratio median = {np.median(ra):.3f}  ratio>=0.80: {(ra>=0.80).sum()}/12  >=0.50: {(ra>=0.50).sum()}/12")
w(f"  |ray - pure_z| median = {np.median([abs(r['G']['ray']-r['G']['pure_z']) for r in rows]):.2f} (S needs <= 2.0)")
w(f"  singles medians: y={np.median([r['G']['y_only'] for r in rows]):+.2f} "
  f"dims={np.median([r['G']['dims'] for r in rows]):+.2f} "
  f"yaw={np.median([r['G']['yaw'] for r in rows]):+.2f} (S needs each <= 2.0)")
nbin_ok = sum(1 for r in rows if all(pz > b for pz, b in zip(r["rb"]["pure_z"], r["rb"]["base"])))
w(f"  per-bin pure_z recall gain > 0 in ALL bins: {nbin_ok}/12 detectors (S needs >= 10)")
w(f"  sub-additivity |sum(singles)-allfac| median = {np.median([r['subadd'] for r in rows]):.2f} (disclosure)")
if os.path.exists(W):
    shutil.rmtree(W)
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

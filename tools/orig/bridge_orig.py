"""Ported from mono3d_crossdataset/tools/bridge_orig.py for the public release. Computation
unchanged. Produces reports_orig/depth_share_bridge_orig.txt.

DEPTH-SHARE BRIDGE for the original-environment MonoFlex*/MonoGround* dumps (a copy of
tools/decomp/depth_share_bridge.py with inputs switched) — is the un-closable ~13AP rank gap
depth-located? (AP currency)

Per detector on the S5 kept pool (post-NMS; gap anatomy showed this pool has ~no
duplicates, so its gap = TP-vs-FP ordering failure):
  base = AP(V)                      (detector as-is)
  A    = AP(rank by IoU_act)        (full order-ceiling; IoU vs ALL GTs, NO z-gate —
                                     the |dz|<8 gate would censor depth-displaced boxes)
  B    = AP(rank by IoU_z*)         (rank by "ray-conditional quality at oracle depth":
                                     each box slid along the camera ray to each GT's z —
                                     IDENTICAL transform to the +50 oracle-depth lever —
                                     and scored by the best resulting IoU3D)
  depth share of gap = (A - B) / (A - base)
     B ~= base  -> ordering the pool needs depth knowledge (gap = depth-blindness)
     B ~= A     -> gap closable without depth knowledge (depth-rank decoupled)
Plus FP separation: among boxes with IoU_act < 0.1, can V tell depth-displaced
would-be-TPs (IoU_z* >= 0.5) from true ghosts (IoU_z* < 0.1)? AUROC(V).
Scope note: this tests the detector's OWN score channel; cross-detector depth disagreement
(rho~0.40) already shows the depth-error information exists — claim is V doesn't carry it.

Reuses the <stem>_bridgecache.npz oracle-IoU cache written by gap_exact_orig.py (or builds it).
Run from the repository root: python tools/orig/bridge_orig.py
"""
import os, sys, shutil
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
from tools.orig._orig_common import ORIG
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti, pool_mask, apply_nms
from evaluator.kitti_utils import Calibration
import evaluator.kitti_eval.kitti_common as kc
from evaluator.kitti_eval.eval import do_eval

DETS = [("MonoFlex*", "monoflex_orig"), ("MonoGround*", "monoground_orig")]
val_list = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val_list)
ov07 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7]] * 3)
ov05 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
MO = np.stack([ov07, ov05], 0)[:, :, [0]]
WORK = f"{ORIG}/_dbridge_orig"
OUT = out_path("depth_share_bridge_orig.txt")
outlines = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); outlines.append(s)


def ev(df, score):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data"); write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val_list)
    return float(do_eval(GT, kc.get_label_annos(d, val_list), [0], MO, compute_aos=False, DIForDIS=True)[6][0, 1, 0])


def auroc(pos, neg):
    """rank-based AUROC of score separating pos from neg (no sklearn dependency)."""
    s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    # average ties
    sv = s[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rp = r[y == 1].sum()
    n1, n0 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def iou_act_and_zstar(kept):
    """per box: IoU_act = max IoU3D over ALL GTs (no z-gate);
    IoU_z* = max over GTs of IoU3D after ray-sliding the box to that GT's z
    (s=z_gt/z_pred; x,y slide along the ray through the box's 3D center; same
    transform as clean_transfer_strong/oracle-depth)."""
    o_act = np.zeros(len(kept)); o_z = np.zeros(len(kept))
    for sid, g in kept.groupby("sid"):
        gts = arc.read_gt(int(sid))
        if not gts:
            continue
        cal = Calibration(os.path.join(arc.CALIB_DIR, f"{int(sid):06d}.txt"))
        tx, ty = cal.tx, cal.ty
        gps = [arc.poly(t[3], t[5], t[1], t[2], t[6]) for t in gts]
        for k in g.index.values:
            x0 = kept.at[k, "x_3d"]; y0 = kept.at[k, "y_3d"]; z0 = kept.at[k, "z_3d"]
            h3 = kept.at[k, "h_3d"]; w3 = kept.at[k, "w_3d"]; l3 = kept.at[k, "l_3d"]
            ry = kept.at[k, "ry"]
            if z0 <= 0.1:
                continue
            pp = arc.poly(x0, z0, w3, l3, ry)
            ba = 0.0; bz = 0.0
            for j, t in enumerate(gts):
                v = arc.iou3d(pp, y0, h3, l3, w3, t, gps[j])
                if v > ba:
                    ba = v
                zg = t[5]
                s = zg / z0
                xs = tx + (x0 - tx) * s
                ys = (h3 / 2.0 + ty) + (y0 - h3 / 2.0 - ty) * s
                pps = arc.poly(xs, zg, w3, l3, ry)
                vz = arc.iou3d(pps, ys, h3, l3, w3, t, gps[j])
                if vz > bz:
                    bz = vz
            o_act[k] = ba; o_z[k] = bz
    return o_act, o_z


if __name__ == "__main__":
    rows = []
    for name, f in DETS:
        df = pd.read_csv(dump_path(f)).reset_index(drop=True)
        pre = df[pool_mask(df, "thr0.2")].copy()
        keep = apply_nms(pre, pre["V"].values, 0.5)
        kept = pre[keep].copy().reset_index(drop=True)
        cache = f"{ORIG}/{f}_bridgecache.npz"
        if os.path.exists(cache):
            z_ = np.load(cache); o_act, o_z = z_["o_act"], z_["o_z"]
            assert len(o_act) == len(kept), f"bridgecache misaligned {name}"
        else:
            o_act, o_z = iou_act_and_zstar(kept)
            np.savez_compressed(cache, o_act=o_act, o_z=o_z)
        V = kept["V"].values.astype(float)

        base = ev(kept, V)
        A = ev(kept, o_act)
        B = ev(kept, o_z)
        share = (A - B) / max(A - base, 1e-9)

        # FP separation: depth-displaced would-be-TPs vs true ghosts
        fp = o_act < 0.1
        posm = fp & (o_z >= 0.5); negm = fp & (o_z < 0.1)
        au = auroc(V[posm], V[negm]) if posm.sum() >= 30 and negm.sum() >= 30 else float("nan")

        sp_act = spearmanr(V, o_act).correlation
        sp_z = spearmanr(V, o_z).correlation
        rows.append(dict(name=name, base=base, A=A, B=B, share=share, au=au,
                         npos=int(posm.sum()), nneg=int(negm.sum()), sp_act=sp_act, sp_z=sp_z))
        w(f"[done] {name:10s} base={base:6.2f} A(IoU)={A:6.2f} B(IoU_z*)={B:6.2f} "
          f"depth-share={share:5.1%} AUROC={au:.3f} (n+={posm.sum()},n-={negm.sum()}) "
          f"sp(V,act)={sp_act:+.3f} sp(V,z*)={sp_z:+.3f}")

    w("")
    w("=" * 104)
    w("DEPTH-SHARE BRIDGE (S5 pool, official eval) — how much of the order-ceiling gap requires depth knowledge")
    w("  depth share = (A - B)/(A - base); B-base = gap closable WITHOUT depth knowledge")
    w("  AUROC: V separating depth-displaced would-be-TPs (IoU_z*>=0.5, IoU_act<0.1) from ghosts (IoU_z*<0.1)")
    w("=" * 104)
    hdr = (f"{'detector':10s} {'base':>6} {'A=IoU':>6} {'B=IoU_z*':>8} {'gap':>6} {'B-base':>7} "
           f"{'depth-share':>11} | {'AUROC(V)':>8} {'n+':>5} {'n-':>6} | {'sp(V,act)':>9} {'sp(V,z*)':>8}")
    w(hdr); w("-" * len(hdr))
    for r in rows:
        w(f"{r['name']:10s} {r['base']:>6.2f} {r['A']:>6.2f} {r['B']:>8.2f} {r['A']-r['base']:>+6.2f} "
          f"{r['B']-r['base']:>+7.2f} {r['share']:>11.1%} | {r['au']:>8.3f} {r['npos']:>5d} {r['nneg']:>6d} | "
          f"{r['sp_act']:>+9.3f} {r['sp_z']:>+8.3f}")

    w("")
    w("READ: depth-share ~ 1 & AUROC ~ 0.5 across families => the in-pool ordering failure is")
    w("depth-located AND invisible to the detector's own score channel (cross-")
    w("detector disagreement predicts |dz| at rho~0.40, but V does not carry it).")
    w("depth-share ~ 0 => rank and depth decoupled; the gap would be a pure scoring problem.")

    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    open(OUT, "w").write("\n".join(outlines) + "\n")
    w(f"[written] {OUT}")

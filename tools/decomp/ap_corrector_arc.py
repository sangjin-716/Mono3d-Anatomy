"""Shared matcher module (read_gt, poly, iou3d, match, DET feature list, KITTI paths) imported by
the other tools/decomp scripts. It writes no report of its own. Its CLI needs a detector's
train-split dump, which is not part of the released dumps.

Parametrized post-hoc depth-corrector AP gate for ANY detector with train+val
dumps (same 33-col schema as dgp/cop). Produces the 4-number table used for the
cross-architecture generality claim:
    baseline / clean train->val (legit) / OOF-on-val (in-dist upper bound) / oracle-z (ceiling)
All native V (no rescoring), ray-consistent z-only correction, identical emit/eval
path (official kitti_eval).

  python tools/decomp/ap_corrector_arc.py --name MonoCoP \
      --train_csv <train dump csv> --val_csv <val dump csv> [--base_s5 21.76 --base_s1 22.43]
"""
import os, sys, math, shutil, argparse
import numpy as np, pandas as pd
from shapely.geometry import Polygon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path  # noqa: E402

LABEL_DIR = paths.LABEL_DIR
CALIB_DIR = paths.CALIB_DIR
VAL_LIST = paths.VAL_LIST
DET = ["cls", "sigma", "log_sigma_raw", "V", "z_pred", "bbox_h_pix", "bbox_w_pix",
       "bbox_area_pix", "x_3d", "y_3d", "h_3d", "w_3d", "l_3d", "ry", "dup_rank"]

from dgp_cop_oracle_matrix import write_kitti, pool_mask, apply_nms  # noqa
from evaluator.kitti_utils import Calibration  # noqa
import evaluator.kitti_eval.kitti_common as kc  # noqa
from evaluator.kitti_eval.eval import do_eval  # noqa


def read_gt(sid):
    out = []; p = os.path.join(LABEL_DIR, f"{sid:06d}.txt")
    if not os.path.exists(p):
        return out
    for ln in open(p):
        t = ln.split()
        if t[0] == "Car":
            out.append((float(t[8]), float(t[9]), float(t[10]), float(t[11]),
                        float(t[12]), float(t[13]), float(t[14])))  # h w l x y z ry
    return out


def poly(x, z, w, l, ry):
    xc = np.array([l/2, l/2, -l/2, -l/2]); zc = np.array([w/2, -w/2, -w/2, w/2])
    c, s = math.cos(ry), math.sin(ry)
    return Polygon(np.c_[x + c*xc + s*zc, z - s*xc + c*zc])


def iou3d(pp, py, ph, pl, pw, g, gp):
    try:
        inter = pp.intersection(gp).area
    except Exception:
        return 0.0
    hov = max(0.0, min(py, g[4]) - max(py - ph, g[4] - g[0])); iv = inter*hov
    u = pl*pw*ph + g[2]*g[1]*g[0] - iv
    return iv/u if u > 0 else 0.0


def match(df):
    """Per pred -> best Car GT (iou>=.05). Returns gtz (nan if unmatched), dz=z3-gtz."""
    z3 = df.z_3d.values; px = df.x_3d.values; py = df.y_3d.values
    pw = df.w_3d.values; pl = df.l_3d.values; pry = df.ry.values; ph = df.h_3d.values
    gtz = np.full(len(df), np.nan); dz = np.full(len(df), np.nan)
    for sid, g in df.groupby("sid"):
        gts = read_gt(int(sid))
        if not gts:
            continue
        gps = [poly(t[3], t[5], t[1], t[2], t[6]) for t in gts]
        gz = np.array([t[5] for t in gts])
        for k in g.index.values:
            cand = np.where(np.abs(z3[k] - gz) < 8.0)[0]
            if cand.size == 0:
                continue
            pp = poly(px[k], z3[k], pw[k], pl[k], pry[k]); best = 0.0; bj = -1
            for j in cand:
                io = iou3d(pp, py[k], ph[k], pl[k], pw[k], gts[j], gps[j])
                if io > best:
                    best = io; bj = j
            if bj >= 0 and best >= 0.05:
                gtz[k] = gts[bj][5]; dz[k] = z3[k] - gts[bj][5]
    return gtz, dz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--base_s5", type=float, default=None)
    ap.add_argument("--base_s1", type=float, default=None)
    args = ap.parse_args()
    val_list = [int(x) for x in open(VAL_LIST).read().split()]
    gt_annos = kc.get_label_annos(LABEL_DIR, val_list)

    # --- train corrector ---
    tr = pd.read_csv(args.train_csv).reset_index(drop=True)
    _, dz_tr = match(tr); mtr = ~np.isnan(dz_tr)
    from sklearn.ensemble import HistGradientBoostingRegressor
    gbm_tv = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0)
    gbm_tv.fit(tr[DET].astype(float).values[mtr], dz_tr[mtr])
    print(f"[{args.name}] train matched={mtr.sum()}/{len(tr)}")

    # --- val: match (oracle z + OOF target) ---
    df = pd.read_csv(args.val_csv).reset_index(drop=True)
    gtz_v, dz_v = match(df); mv = ~np.isnan(dz_v)
    print(f"[{args.name}] val matched={mv.sum()}/{len(df)}")
    Xv = df[DET].astype(float).values
    pdz_tv = gbm_tv.predict(Xv)
    # OOF-on-val (in-dist upper bound), folds by sid%5, train on matched of other folds
    fold = df.sid.values % 5
    pdz_oof = np.zeros(len(df))
    for f in range(5):
        trm = mv & (fold != f)
        g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0)
        g.fit(Xv[trm], dz_v[trm]); pdz_oof[fold == f] = g.predict(Xv[fold == f])
    print(f"[{args.name}] pdz train->val mean={pdz_tv.mean():+.3f}  OOF mean={pdz_oof.mean():+.3f}")

    # --- ray-consistent reproject ---
    u = (df.bbox_x1.values + df.bbox_x2.values)/2.0
    v = (df.bbox_y1.values + df.bbox_y2.values)/2.0
    cu = np.zeros(len(df)); cv = np.zeros(len(df)); fu = np.zeros(len(df))
    fv = np.zeros(len(df)); tx = np.zeros(len(df)); ty = np.zeros(len(df))
    for sid, g in df.groupby("sid"):
        cal = Calibration(os.path.join(CALIB_DIR, f"{int(sid):06d}.txt")); i = g.index.values
        cu[i] = cal.cu; cv[i] = cal.cv; fu[i] = cal.fu; fv[i] = cal.fv; tx[i] = cal.tx; ty[i] = cal.ty
    h3 = df.h_3d.values

    # ray-consistent depth edit: scale the EXISTING 3D point along its camera ray
    # (detector-agnostic; identical to img_to_rect-reproj when u,v is the proj pixel,
    #  but correct for detectors that back-project from a non-bbox-center pixel, e.g. MonoFlex).
    x0 = df.x_3d.values; y0 = df.y_3d.values; z0 = df.z_3d.values

    def reproj(z):
        s = z / z0
        x = tx + (x0 - tx) * s
        y = (h3 / 2.0 + ty) + (y0 - h3 / 2.0 - ty) * s
        return x, y, z

    variants = {
        "baseline": reproj(df.z_3d.values),
        "clean train->val": reproj(df.z_3d.values - pdz_tv),
        "OOF-on-val (in-dist)": reproj(df.z_3d.values - pdz_oof),
        "oracle-z (ceiling)": reproj(np.where(mv, gtz_v, df.z_3d.values)),
    }

    ov07 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7]]*3)
    ov05 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                     [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    min_ov = np.stack([ov07, ov05], 0)[:, :, [0]]
    ALLCELLS = {"S1": ("all", None), "S2": ("thr0.2", None), "S5": ("thr0.2", "nms")}
    WORK = os.path.join(cache_dir("decomp"), f"_arc_{args.name}")
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)

    def ev(d, label, cellnames):
        res = {}
        for cl in cellnames:
            pool, nms = ALLCELLS[cl]
            base = d[pool_mask(d, pool)].copy(); sc = base["V"].values
            if nms == "nms":
                keep = apply_nms(base, base["V"].values, 0.5); base = base[keep].copy(); sc = base["V"].values
            ddir = os.path.join(WORK, f"{label}_{cl}".replace(" ", "_").replace(">", ""), "data")
            write_kitti(base, sc, ddir, val_list)
            dt = kc.get_label_annos(ddir, val_list)
            r = do_eval(gt_annos, dt, [0], min_ov, compute_aos=False, DIForDIS=True)
            res[cl] = dict(m7=float(r[6][0, 1, 0]), e7=float(r[6][0, 0, 0]), h7=float(r[6][0, 2, 0]),
                           m5=float(r[6][0, 1, 1]), bev7=float(r[5][0, 1, 0]))
            shutil.rmtree(os.path.dirname(ddir), ignore_errors=True)
        return res

    print(f"\n[{args.name}] train matched Δz: mean {np.nanmean(dz_tr):+.3f} std {np.nanstd(dz_tr):.3f} | "
          f"val matched Δz: mean {np.nanmean(dz_v):+.3f} std {np.nanstd(dz_v):.3f}", flush=True)
    cellsel = {"baseline": ["S1", "S2", "S5"], "clean train->val": ["S2", "S5"],
               "OOF-on-val (in-dist)": ["S2", "S5"], "oracle-z (ceiling)": ["S2", "S5"]}
    res = {}
    print(f"\n==== {args.name}: post-hoc depth corrector AP (native V) ====")
    print(f"  {'variant':<24} {'S2 m@.7':>9} {'S5 m@.7':>9} {'S5 m@.5':>9} {'S5 bevM':>9}")
    for vn, (vx, vy, vz) in variants.items():
        d = df.copy(); d["x_3d"] = vx; d["y_3d"] = vy; d["z_3d"] = vz
        res[vn] = ev(d, vn, cellsel[vn])
        s2 = res[vn].get("S2", {}).get("m7", float("nan"))
        extra = f"   [S1 m@.7={res[vn]['S1']['m7']:.2f}]" if vn == "baseline" else ""
        print(f"  {vn:<24} {s2:>9.2f} {res[vn]['S5']['m7']:>9.2f} {res[vn]['S5']['m5']:>9.2f} "
              f"{res[vn]['S5']['bev7']:>9.2f}{extra}", flush=True)

    print(f"\n[gate] baseline  S1={res['baseline']['S1']['m7']:.2f}  S2={res['baseline']['S2']['m7']:.2f}  "
          f"S5={res['baseline']['S5']['m7']:.2f}" + (f"  (expect native≈{args.base_s5})" if args.base_s5 else ""))
    print(f"\n==== {args.name} VERDICT (legit ≤0? in-dist +X? oracle huge?) ====")
    for cell in ["S2", "S5"]:
        b = res["baseline"][cell]["m7"]; tv = res["clean train->val"][cell]["m7"]
        oof = res["OOF-on-val (in-dist)"][cell]["m7"]; orc = res["oracle-z (ceiling)"][cell]["m7"]
        same = "REPLICATES" if (tv - b <= 0.20 and oof - b > 0.5) else "DIFFERS"
        print(f"  [{cell}] baseline={b:.2f}  clean train→val={tv:.2f} (Δ{tv-b:+.2f})  "
              f"OOF in-dist={oof:.2f} (Δ{oof-b:+.2f})  oracle-z={orc:.2f} (Δ{orc-b:+.2f})  → {same}")
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()

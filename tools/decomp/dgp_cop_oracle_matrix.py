"""Ported from tools/dgp_cop_oracle_matrix.py for the public release. Computation unchanged.
Produces no report cited by the paper; its helpers (write_kitti, pool_mask, apply_nms,
greedy_nms_keep) are the shared pool/NMS/KITTI-writer kernels imported by the tools/decomp scripts.
The CLI below writes <out_dir>/oraclemtx_<name>.csv (reports_rerun/ by default).

Controlled oracle-protocol matrix for one model's val records.

Reconciles the all-50/thr-0 oracle (small gap) against the post-top-50/thr-0.2
oracle (large gap). For each (pool, score) cell: AP3D/BEV Mod, n_boxes, preds/img,
score-distribution, and a confirmation that box GEOMETRY is never changed (only the
set of written boxes and the score column change).

Cells:
  S1  all50_thr0        pool = all 50 queries, no threshold        (unified protocol)
  S2  thr0.2            pool = cls >= 0.2                          (quality-line filter)
  S3  thr0.05           pool = cls >= 0.05                         (threshold sensitivity)
  S4  nms0.5            pool = all 50 + greedy 2D-NMS @ IoU2D 0.5  (dedup; per-score order)
  S5  thr0.2_nms0.5     pool = cls>=0.2 then NMS                   (thr + dedup)

For each cell we eval TWO scores on the SAME pool:
  - V          (baseline cls*sigma)
  - oracle     (max_iou_3d per prediction)
so the oracle GAP within a cell isolates reranking headroom for THAT pool.

NMS uses the active score's order (V for baseline col, max_iou_3d for oracle col),
matching how a real detector would dedup under each ranking.

GT = KITTI labels (paths.LABEL_DIR). Official KITTI evaluator: evaluator/kitti_eval (numba.cuda).
Run from the repository root:
  python tools/decomp/dgp_cop_oracle_matrix.py --name MonoDGP --val_csv <dump csv>
"""
from __future__ import annotations
import os, sys, shutil, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path  # noqa: E402


def iou2d_mat(b):
    x1 = np.maximum(b[:, 0][:, None], b[:, 0][None])
    y1 = np.maximum(b[:, 1][:, None], b[:, 1][None])
    x2 = np.minimum(b[:, 2][:, None], b[:, 2][None])
    y2 = np.minimum(b[:, 3][:, None], b[:, 3][None])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    u = a[:, None] + a[None] - inter
    return np.where(u > 0, inter / u, 0.0)


def greedy_nms_keep(df_img, score, thr=0.5):
    """Return boolean keep-mask for rows of one image, greedy NMS on 2D IoU by score."""
    b = df_img[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values.astype(float)
    s = np.asarray(score, float)
    order = np.argsort(-s)
    M = iou2d_mat(b)
    keep = np.ones(len(s), bool)
    suppressed = np.zeros(len(s), bool)
    for i in order:
        if suppressed[i]:
            continue
        sup = (M[i] >= thr) & (np.arange(len(s)) != i) & (~suppressed)
        suppressed |= sup
    keep = ~suppressed
    return keep


def write_kitti(df, score, ddir, val_sids):
    if os.path.exists(ddir):
        shutil.rmtree(ddir)
    os.makedirs(ddir)
    df = df.assign(scoreval=np.asarray(score, float))
    for sid, g in df.groupby("sid"):
        with open(os.path.join(ddir, f"{int(sid):06d}.txt"), "w") as f:
            for r in g.itertuples(index=False):
                f.write(f"Car 0.0 0 {r.alpha:.4f} {r.bbox_x1:.4f} {r.bbox_y1:.4f} "
                        f"{r.bbox_x2:.4f} {r.bbox_y2:.4f} {r.h_3d:.4f} {r.w_3d:.4f} "
                        f"{r.l_3d:.4f} {r.x_3d:.4f} {r.y_3d:.4f} {r.z_3d:.4f} "
                        f"{r.ry:.4f} {r.scoreval:.6f}\n")
    for sid in val_sids:
        p = os.path.join(ddir, f"{sid:06d}.txt")
        if not os.path.exists(p):
            open(p, "w").close()


def make_evaler(label_dir, val_list):
    from evaluator.kitti_eval.eval import get_official_eval_result
    import evaluator.kitti_eval.kitti_common as kc
    val = [int(l) for l in open(val_list) if l.strip().isdigit()]
    gt = kc.get_label_annos(label_dir, val)

    def _eval(ddir):
        dt = kc.get_label_annos(ddir, val)
        r = get_official_eval_result(gt, dt, current_classes=[0])
        rd = r[1] if isinstance(r, tuple) else r
        g = lambda k: float(rd.get(k, 0.0))
        return (g("Car_3d_easy_R40"), g("Car_3d_moderate_R40"), g("Car_3d_hard_R40"),
                g("Car_bev_moderate_R40"))
    return _eval, val


def pool_mask(df, kind):
    if kind == "all":
        return np.ones(len(df), bool)
    if kind == "thr0.2":
        return (df["cls"].values >= 0.2)
    if kind == "thr0.05":
        return (df["cls"].values >= 0.05)
    raise ValueError(kind)


def apply_nms(df, score_col_vals, thr=0.5):
    keep = np.zeros(len(df), bool)
    pos = {ri: i for i, ri in enumerate(df.index)}
    for sid, g in df.groupby("sid"):
        k = greedy_nms_keep(g, score_col_vals[[pos[ri] for ri in g.index]], thr)
        keep[[pos[ri] for ri in g.index]] = k
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--val_csv", required=True, help="dump csv, or a dump stem from tools/_release.py")
    ap.add_argument("--label_dir", default=paths.LABEL_DIR)
    ap.add_argument("--val_list", default=paths.VAL_LIST)
    args = ap.parse_args()
    val_csv = args.val_csv if os.path.exists(args.val_csv) else dump_path(args.val_csv)

    df = pd.read_csv(val_csv).reset_index(drop=True)
    n_img = df["sid"].nunique()
    evaler, val_sids = make_evaler(args.label_dir, args.val_list)
    workdir = os.path.join(cache_dir("decomp"), f"_oraclemtx_{args.name}")
    os.makedirs(workdir, exist_ok=True)

    cells = [
        ("S1 all50_thr0", "all", None),
        ("S2 thr0.2", "thr0.2", None),
        ("S3 thr0.05", "thr0.05", None),
        ("S4 all50_nms0.5", "all", "nms"),
        ("S5 thr0.2_nms0.5", "thr0.2", "nms"),
    ]
    geom_cols = ["x_3d", "y_3d", "z_3d", "h_3d", "w_3d", "l_3d", "ry"]
    geom_ref = df[geom_cols].copy()

    print(f"\n==== {args.name}  (n_img={n_img}, total rows={len(df)}) ====", flush=True)
    print(f"{'cell':<18}{'score':<8}{'nbox':>8}{'/img':>7}{'AP3D_M':>9}{'BEV_M':>8}"
          f"{'meanSc':>8}", flush=True)
    results = []
    for label, pool, nms in cells:
        m = pool_mask(df, pool)
        base = df[m].copy()
        if len(base) == 0:
            continue
        for score_name, score_col in [("V", "V"), ("oracle", "max_iou_3d")]:
            sub = base
            sc = base[score_col].values
            if nms == "nms":
                keep = apply_nms(base, base[score_col].values, 0.5)
                sub = base[keep].copy()
                sc = sub[score_col].values
            ddir = os.path.join(workdir, f"{label}_{score_name}".replace(" ", "_"), "data")
            write_kitti(sub, sc, ddir, val_sids)
            e, md, h, bevm = evaler(ddir)
            shutil.rmtree(os.path.dirname(ddir), ignore_errors=True)
            nbox = len(sub)
            results.append(dict(cell=label, score=score_name, nbox=nbox,
                                per_img=nbox / n_img, AP3D_E=e, AP3D_M=md, AP3D_H=h,
                                BEV_M=bevm, meanSc=float(np.mean(sc))))
            print(f"{label:<18}{score_name:<8}{nbox:>8}{nbox/n_img:>7.1f}{md:>9.2f}"
                  f"{bevm:>8.2f}{np.mean(sc):>8.3f}", flush=True)
    # geometry unchanged check
    geom_ok = df[geom_cols].equals(geom_ref)
    rdf = pd.DataFrame(results)
    out = out_path(f"oraclemtx_{args.name}.csv")
    rdf.to_csv(out, index=False)
    # gaps
    print("\n-- oracle gap per cell (oracle - V, AP3D Mod) --", flush=True)
    for label, _, _ in cells:
        v = rdf[(rdf.cell == label) & (rdf.score == "V")]
        o = rdf[(rdf.cell == label) & (rdf.score == "oracle")]
        if len(v) and len(o):
            print(f"  {label:<18} V={v.AP3D_M.iloc[0]:.2f}  oracle={o.AP3D_M.iloc[0]:.2f}"
                  f"  gap={o.AP3D_M.iloc[0]-v.AP3D_M.iloc[0]:+.2f}", flush=True)
    print(f"\ngeometry unchanged across cells: {geom_ok}", flush=True)
    print(f"wrote {out}", flush=True)
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

"""Generic per-prediction dump for MonoDGP and (clean) MonoCoP -> 33-col schema.

Ported from tools/dgp_cop_dump.py for the public release. Computation unchanged.
Produces data/dumps/dgp_val.csv (MonoDGP) and data/dumps/official_monocop_val.csv (MonoCoP),
the two 33-column released dumps (23 prediction columns + 10 GT-matching columns).
Changes: the GT label dir is --label_dir (default paths.LABEL_DIR; the original looked for
<repo>/data/KITTIDataset or <repo>/data/kitti, i.e. the same KITTI labels), and --root_dir
optionally overrides cfg["dataset"]["root_dir"] instead of placing KITTI at the repo-relative
path the config expects (the original runs used repo-relative data symlinks).

Inference only — no training, no weight/arch change. Emits ALL queries (no score
threshold), score V = cls.max * exp(-log_sigma), and matches each query to Car GT
by greedy IoU (V-descending) for oracle/analysis.

Works for any repo whose model forward is
  model(inputs, calibs, targets, img_sizes, dn_args=0)
and whose decode is  cls = sigmoid(logits).max ; sigma = exp(-pred_depth[...,1]) ;
z = pred_depth[...,0]  (MonoDGP clean main; MonoCoP clean 9b0867b).

Per-prediction columns:
  sid, pred_idx, cls, sigma, log_sigma_raw, V, z_pred,
  bbox_h_pix, bbox_w_pix, bbox_area_pix, x_3d, y_3d, z_3d, h_3d, w_3d, l_3d, ry, alpha,
  bbox_x1..y2, dup_rank, dist_bin_4,
  max_iou_3d, max_iou_bev, matched_gt_idx, tp_label_iou07,
  gt_x, gt_y, gt_z, depth_err, center_err_bev

The `lib.*` imports are the detector repo passed as --repo (run with cwd = that repo).
Run with the repo's matching conda env on a free GPU, e.g.
  python adapters/dgp_cop_dump.py --repo <MonoDGP> --cfg <MonoDGP>/configs/monodgp.yaml \
      --ckpt <monodgp ckpt> --split val --out <OUT_DIR>/dumps/dgp_val.csv --tag dgp_val
"""
from __future__ import annotations
import os, sys, time, shutil, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence


def iou2d_matrix(boxes):
    x1 = boxes[:, 0][:, None]; y1 = boxes[:, 1][:, None]
    x2 = boxes[:, 2][:, None]; y2 = boxes[:, 3][:, None]
    import numpy as np
    X1 = np.maximum(x1, boxes[:, 0][None]); Y1 = np.maximum(y1, boxes[:, 1][None])
    X2 = np.minimum(x2, boxes[:, 2][None]); Y2 = np.minimum(y2, boxes[:, 3][None])
    iw = np.clip(X2 - X1, 0, None); ih = np.clip(Y2 - Y1, 0, None)
    inter = iw * ih
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None] - inter
    return np.where(union > 0, inter / union, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--out", required=True, help="output csv path")
    ap.add_argument("--tag", required=True, help="short tag for kitti tmp dir")
    ap.add_argument("--label_dir", default=paths.LABEL_DIR, help="KITTI training/label_2")
    ap.add_argument("--root_dir", default=None, help="optional override of cfg['dataset']['root_dir']")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import numpy as np, pandas as pd, torch, yaml
    from lib.helpers.model_helper import build_model
    from lib.helpers.decode_helper import get_heading_angle
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    from lib.datasets.kitti.kitti_eval_python import kitti_common as kc
    from lib.datasets.kitti.kitti_eval_python.eval import calculate_iou_partly
    from torch.utils.data import DataLoader

    LABEL_DIR = args.label_dir
    assert os.path.isdir(LABEL_DIR), f"label dir not found: {LABEL_DIR}"

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    if args.root_dir:
        cfg["dataset"]["root_dir"] = args.root_dir

    device = torch.device("cuda")
    model, _ = build_model(cfg["model"])
    model = model.to(device)
    st = torch.load(args.ckpt, map_location=device)
    sd = st.get("model_state", st)
    miss = model.load_state_dict(sd, strict=False)
    # report load fidelity (must be clean for a faithful baseline)
    nmiss = len(getattr(miss, "missing_keys", []))
    nunexp = len(getattr(miss, "unexpected_keys", []))
    print(f"[load] missing={nmiss} unexpected={nunexp}", flush=True)
    if nmiss:
        print("  missing sample:", list(miss.missing_keys)[:8], flush=True)
    if nunexp:
        print("  unexpected sample:", list(miss.unexpected_keys)[:8], flush=True)
    model.eval()

    ds = KITTI_Dataset(split=args.split, cfg=cfg["dataset"])
    ds.data_augmentation = False
    if hasattr(ds, "istrain"):
        ds.istrain = False
    loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=False,
                        pin_memory=True, collate_fn=getattr(ds, "collate_fn", None))
    cms = ds.cls_mean_size
    sids = sorted(int(x) for x in ds.idx_list)
    gt_annos = kc.get_label_annos(LABEL_DIR, sids)
    idx_map = {s: i for i, s in enumerate(sids)}

    kdir = os.path.join(os.path.dirname(args.out), f"_kitti_{args.tag}", "data")
    if os.path.exists(kdir):
        shutil.rmtree(kdir)
    os.makedirs(kdir)

    # ---- forward, dump KITTI txt (all queries, V score), collect rows ----
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for bi, (inputs, calibs, targets, info) in enumerate(loader):
            inputs = inputs.to(device); calibs = calibs.to(device)
            img_sizes = info["img_size"].to(device)
            for k in targets:
                if torch.is_tensor(targets[k]):
                    targets[k] = targets[k].to(device)
            outputs = model(inputs, calibs, targets, img_sizes, dn_args=0)
            logits = outputs["pred_logits"]
            cls = logits.sigmoid().max(-1).values
            log_sigma = outputs["pred_depth"][..., 1]
            sigma = torch.exp(-log_sigma)
            pred_z = outputs["pred_depth"][..., 0]
            V = cls * sigma
            boxes = outputs["pred_boxes"]
            H = img_sizes[:, 1].clamp(min=1.0).unsqueeze(1)
            W = img_sizes[:, 0].clamp(min=1.0).unsqueeze(1)
            bbox_h = ((boxes[..., 4] + boxes[..., 5]) * H).clamp(min=1.0)
            bbox_w = ((boxes[..., 2] + boxes[..., 3]) * W).clamp(min=1.0)
            dim = outputs["pred_3d_dim"]
            h3d = dim[..., 0].cpu().numpy() + cms[0][0]
            w3d = dim[..., 1].cpu().numpy() + cms[0][1]
            l3d = dim[..., 2].cpu().numpy() + cms[0][2]
            heading = outputs["pred_angle"].cpu().numpy()
            xs = (boxes[..., 0] * W).cpu().numpy()
            ys = (boxes[..., 1] * H).cpu().numpy()
            cls_n = cls.cpu().numpy(); sigma_n = sigma.cpu().numpy()
            ls_n = log_sigma.cpu().numpy(); V_n = V.cpu().numpy()
            z_n = pred_z.cpu().numpy()
            bh_n = bbox_h.cpu().numpy(); bw_n = bbox_w.cpu().numpy()
            B, Q = cls.shape
            for b in range(B):
                sid = int(info["img_id"][b])
                calib = ds.get_calib(sid)
                lines = []
                rec_local = []
                boxes2d = np.zeros((Q, 4), np.float32)
                for q in range(Q):
                    x_img = float(xs[b, q]); y_img = float(ys[b, q])
                    w_img = float(bw_n[b, q]); h_img = float(bh_n[b, q])
                    bbox = [x_img - w_img / 2, y_img - h_img / 2,
                            x_img + w_img / 2, y_img + h_img / 2]
                    boxes2d[q] = bbox
                    depth = float(z_n[b, q])
                    loc = calib.img_to_rect(np.array([x_img]), np.array([y_img]),
                                            np.array([depth])).reshape(-1)
                    hh, ww, ll = float(h3d[b, q]), float(w3d[b, q]), float(l3d[b, q])
                    loc[1] += hh / 2
                    alpha = get_heading_angle(heading[b, q])
                    ry = calib.alpha2ry(alpha, x_img)
                    sc = float(V_n[b, q])
                    lines.append(f"Car 0.0 0 {alpha:.4f} "
                                 f"{bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f} "
                                 f"{hh:.4f} {ww:.4f} {ll:.4f} "
                                 f"{loc[0]:.4f} {loc[1]:.4f} {loc[2]:.4f} {ry:.4f} {sc:.6f}\n")
                    rec_local.append(dict(
                        sid=sid, pred_idx=q,
                        cls=float(cls_n[b, q]), sigma=float(sigma_n[b, q]),
                        log_sigma_raw=float(ls_n[b, q]), V=sc, z_pred=depth,
                        bbox_h_pix=h_img, bbox_w_pix=w_img, bbox_area_pix=h_img * w_img,
                        x_3d=float(loc[0]), y_3d=float(loc[1]), z_3d=float(loc[2]),
                        h_3d=hh, w_3d=ww, l_3d=ll, ry=ry, alpha=alpha,
                        bbox_x1=bbox[0], bbox_y1=bbox[1], bbox_x2=bbox[2], bbox_y2=bbox[3]))
                with open(os.path.join(kdir, f"{sid:06d}.txt"), "w") as f:
                    f.writelines(lines)
                # dup_rank: # other queries with IoU2D>=0.5 and strictly higher V
                iou = iou2d_matrix(boxes2d)
                vv = V_n[b]
                for q in range(Q):
                    nb = (iou[q] >= 0.5)
                    nb[q] = False
                    rec_local[q]["dup_rank"] = int(((vv > vv[q]) & nb).sum())
                rows.extend(rec_local)
            if bi % 50 == 0:
                print(f"  batch {bi}/{len(loader)} rows={len(rows)} {time.time()-t0:.0f}s",
                      flush=True)
    df = pd.DataFrame(rows)
    print(f"[forward] {len(df)} rows, {time.time()-t0:.0f}s", flush=True)

    # ---- IoU matching (Car GT only), greedy V-descending ----
    dt = kc.get_label_annos(kdir, sids)
    ov3d, _, _, _ = calculate_iou_partly(gt_annos, dt, metric=2, num_parts=50)
    ovbev, _, _, _ = calculate_iou_partly(gt_annos, dt, metric=1, num_parts=50)

    N = len(df)
    max3 = np.zeros(N, np.float32); maxb = np.zeros(N, np.float32)
    mgi = np.full(N, -1, np.int32); tp = np.zeros(N, np.int8)
    gtx = np.full(N, np.nan, np.float32); gty = np.full(N, np.nan, np.float32)
    gtz = np.full(N, np.nan, np.float32)
    # group row positions by sid, in pred_idx order (== txt line order == ov column order)
    by_sid = {s: [] for s in sids}
    for ri, s in enumerate(df["sid"].values):
        by_sid[int(s)].append(ri)
    for s in sids:
        ris = by_sid[s]  # already pred_idx order
        gi = idx_map[s]
        m3 = ov3d[gi]; mb = ovbev[gi]
        g = gt_annos[gi]
        is_car = np.array([n == "Car" for n in g["name"]], bool)
        loc = g["location"]
        if m3.size == 0 or not is_car.any():
            continue
        # per-query max iou over Car gts
        for col, ri in enumerate(ris):
            if col >= m3.shape[1]:
                break
            i3 = m3[:, col]; ib = mb[:, col]
            max3[ri] = float(i3[is_car].max())
            maxb[ri] = float(ib[is_car].max())
        # greedy TP assignment by V desc
        order = sorted(range(len(ris)), key=lambda c: -df["V"].values[ris[c]])
        matched = np.zeros(len(g["name"]), bool)
        for col in order:
            if col >= m3.shape[1]:
                continue
            ri = ris[col]
            i3 = m3[:, col]
            valid = is_car & (~matched)
            if not valid.any():
                continue
            vi = np.where(valid, i3, -1.0)
            best = int(vi.argmax()); biou = float(vi[best])
            if biou >= 0.7:
                tp[ri] = 1; mgi[ri] = best; matched[best] = True
                gtx[ri] = float(loc[best, 0]); gty[ri] = float(loc[best, 1])
                gtz[ri] = float(loc[best, 2])
    df["max_iou_3d"] = max3; df["max_iou_bev"] = maxb
    df["matched_gt_idx"] = mgi; df["tp_label_iou07"] = tp
    df["gt_x"] = gtx; df["gt_y"] = gty; df["gt_z"] = gtz
    df["depth_err"] = np.abs(df["z_pred"] - df["gt_z"])
    df["center_err_bev"] = np.hypot(df["x_3d"] - df["gt_x"], df["z_3d"] - df["gt_z"])
    df["dist_bin_4"] = pd.cut(df["z_3d"], bins=[-np.inf, 15, 30, 45, np.inf],
                              labels=["0-15", "15-30", "30-45", "45+"]).astype(str)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[dump:{args.split}] wrote {args.out} ({len(df)} rows, "
          f"TP={int(tp.sum())}, good>=0.7={int((max3>=0.7).sum())}) {time.time()-t0:.0f}s",
          flush=True)
    shutil.rmtree(os.path.dirname(kdir))


if __name__ == "__main__":
    main()

"""MonoFlex per-detection dump adapter (MODERN environment) -> 23-col schema.

Ported from tools/decomp/adapters/monoflex_dump.py for the public release. Computation
unchanged. Produces data/dumps/monoflex_modern_val.csv (the script writes monoflex_val.csv and
monoflex_train.csv into --outdir, default <OUT_DIR>/dumps/modern/; rename the val file to
monoflex_modern_val.csv). The panel entry MonoFlex* uses the ORIGINAL-environment dump
instead (adapters/monoflex_dump_orig.py); this modern dump drifts -1.97 AP (see README).

The 23 columns are the prediction part of the 33-col DETR-dump schema. Inference-only, read-only on the repo. Dumps BOTH val+train in
one GPU pass (model loaded once). DETECTIONS_THRESHOLD=0 -> full top-50 pool;
model.test=True so PostProcessor only needs calib/size/pad_size (no GT fields).

Column mapping (MonoFlex result tensor [N,14] = cls,alpha,box2d(4),h,w,l,x,y,z,ry,score
+ eval_utils{vis_scores=heatmap score, estimated_depth_error=sigma}):
  cls           <- vis_scores      (heatmap score; the THRESHOLDING score, like DGP cls)
  V             <- result[:,13]    (uncertainty-modulated score; the native RANKING score)
  z_pred,z_3d   <- result[:,11]    (ensemble depth = loc z)
  sigma         <- exp(-log(ede)),  log_sigma_raw <- log(ede)   (ede=estimated_depth_error)
  x_3d,y_3d     <- result[:,9],[:,10]  (cam frame, y already bottom-center)
  h,w,l_3d      <- result[:,6],[:,7],[:,8]   ry<-[:,12]  alpha<-[:,1]
  bbox          <- result[:,2:6]
  dup_rank      <- # Car dets with 2D-IoU>=0.5 and higher V
The `config`/`model`/`utils`/`data` imports are the upstream MonoFlex packages (sys.path gets
--repo). Run in the MonoFlex env: python adapters/monoflex_dump.py [--repo ...] [--ckpt ...] [--kitti_dir ...]
"""
import os, sys, time, argparse, csv
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths, cache_dir  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--repo", default=os.path.join(paths.UPSTREAM_ROOT, "MonoFlex"))
REPO = os.path.abspath(_pre.parse_known_args()[0].repo)
sys.path.insert(0, REPO)
import numpy as np, torch

FIELDS = ["sid", "pred_idx", "cls", "sigma", "log_sigma_raw", "V", "z_pred",
          "bbox_h_pix", "bbox_w_pix", "bbox_area_pix", "x_3d", "y_3d", "z_3d",
          "h_3d", "w_3d", "l_3d", "ry", "alpha",
          "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "dup_rank"]


def iou2d_mat(b):
    x1 = np.maximum(b[:, 0][:, None], b[:, 0][None]); y1 = np.maximum(b[:, 1][:, None], b[:, 1][None])
    x2 = np.minimum(b[:, 2][:, None], b[:, 2][None]); y2 = np.minimum(b[:, 3][:, None], b[:, 3][None])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None); inter = iw * ih
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); u = a[:, None] + a[None] - inter
    return np.where(u > 0, inter / u, 0.0)


def run_split(model, cfg, split, device, out_csv):
    from data import build_test_loader
    cfg.defrost(); cfg.DATASETS.TEST_SPLIT = split
    loader = build_test_loader(cfg, is_train=False)
    if isinstance(loader, (list, tuple)):
        loader = loader[0]
    rows = []; t0 = time.time()
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            images = batch["images"].to(device)
            targets = [t.to(device) for t in batch["targets"]]
            image_ids = batch["img_ids"]
            assert len(image_ids) == 1, f"expected batch 1, got {len(image_ids)}"
            result, eval_utils, _ = model(images, targets)
            result = result.cpu().numpy()
            ede = eval_utils.get('estimated_depth_error', None)
            vis = eval_utils.get('vis_scores', None)
            ede = ede.cpu().numpy().reshape(-1) if ede is not None else np.full(len(result), np.nan)
            vis = vis.cpu().numpy().reshape(-1) if vis is not None else result[:, 13]
            sid = int(image_ids[0])
            # col0 = class_id + 0.02*(within-class idx) quirk; official uses int(col0)
            car = np.where(np.floor(result[:, 0] + 1e-6).astype(int) == 0)[0]   # Car only
            if car.size == 0:
                continue
            R = result[car]; EDE = ede[car]; VIS = vis[car]
            box = R[:, 2:6].astype(float)
            iou = iou2d_mat(box); V = R[:, 13]
            for j in range(len(car)):
                nb = (iou[j] >= 0.5).copy(); nb[j] = False
                dup = int(((V > V[j]) & nb).sum())
                edej = max(float(EDE[j]), 1e-6)
                x1, y1, x2, y2 = box[j]
                rows.append(dict(
                    sid=sid, pred_idx=j,
                    cls=float(VIS[j]), sigma=float(np.exp(-np.log(edej))), log_sigma_raw=float(np.log(edej)),
                    V=float(R[j, 13]), z_pred=float(R[j, 11]),
                    bbox_h_pix=float(y2 - y1), bbox_w_pix=float(x2 - x1), bbox_area_pix=float((y2 - y1) * (x2 - x1)),
                    x_3d=float(R[j, 9]), y_3d=float(R[j, 10]), z_3d=float(R[j, 11]),
                    h_3d=float(R[j, 6]), w_3d=float(R[j, 7]), l_3d=float(R[j, 8]),
                    ry=float(R[j, 12]), alpha=float(R[j, 1]),
                    bbox_x1=float(x1), bbox_y1=float(y1), bbox_x2=float(x2), bbox_y2=float(y2),
                    dup_rank=dup))
            if bi % 500 == 0:
                print(f"  [{split}] {bi} imgs, rows={len(rows)}, {time.time()-t0:.0f}s", flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    nimg = len({r["sid"] for r in rows})
    print(f"[{split}] wrote {out_csv} ({len(rows)} rows, {nimg} imgs) {time.time()-t0:.0f}s", flush=True)


def main():
    from config import cfg
    from model.detector import KeypointDetector
    from utils.check_point import DetectronCheckpointer
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--ckpt", default=os.path.join(paths.UPSTREAM_ROOT, "ckpts", "monoflex", "monoflex_official.pth"))
    ap.add_argument("--cfg", default=os.path.join(REPO, "runs/monoflex.yaml"))
    ap.add_argument("--kitti_dir", default=os.path.join(paths.UPSTREAM_ROOT, "kitti_monoflex"),
                    help="KITTI in the MonoFlex layout: <dir>/training/{image_2,calib,label_2,ImageSets}")
    ap.add_argument("--outdir", default=os.path.join(paths.OUT_DIR, "dumps", "modern"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg.merge_from_file(args.cfg)
    cfg.SOLVER.IMS_PER_BATCH = 1; cfg.DATALOADER.NUM_WORKERS = 4
    cfg.TEST.DETECTIONS_THRESHOLD = 0.0          # full top-50 pool
    cfg.TEST.EVAL_DIS_IOUS = False; cfg.TEST.EVAL_DEPTH = False
    cfg.MODEL.USE_SYNC_BN = False
    # repoint dataset root (repo hardcodes /opt/data1/zyp/...) via a patched catalog
    # file (build.py import_file's cfg.PATHS_CATALOG fresh, so we override that path).
    os.environ["MONOFLEX_KITTI_DIR"] = os.path.abspath(args.kitti_dir)
    cfg.PATHS_CATALOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mf_paths_catalog.py")
    device = torch.device("cuda")
    model = KeypointDetector(cfg); model.to(device)
    model.test = True                             # PostProcessor test path: only calib/size/pad_size
    ckptr = DetectronCheckpointer(cfg, model, save_dir=cache_dir("mf_ckpt"))
    ckptr.load(args.ckpt, use_latest=False)
    print(f"[load] ckpt={args.ckpt}", flush=True)
    for split in ["val", "train"]:
        run_split(model, cfg, split, device, os.path.join(args.outdir, f"monoflex_{split}.csv"))


if __name__ == "__main__":
    main()

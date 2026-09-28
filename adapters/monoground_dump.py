"""MonoGround per-prediction val dump (MODERN environment), 23-col format.

Produces data/dumps/monoground_modern_val.csv (written to <OUT_DIR>/dumps/monoground_val.csv by
default; the released file name adds "_modern"). The panel entry MonoGround* uses the
ORIGINAL-environment dump instead (adapters/monoground_dump_orig.py); the modern build drifts
-1.90 AP (see adapters/README.md).

Nearly the same code as adapters/monoflex_dump.py (same MonoFlex-codebase
forward contract: result[N,14] = cls,alpha,box2d(4),h,w,l,x,y,z,ry,score + eval_utils
{vis_scores, estimated_depth_error}), pointed at the MonoGround repo/ckpt.
  cls <- vis_scores (heatmap score), V <- result[:,13] (native ranking score,
  thresholded at DETECTIONS_THRESHOLD 0.2 natively; we dump at 0.0 = full top-K pool)
Data: the repo's own config/paths_catalog.py is used; set its DatasetCatalog.DATA_DIR to a
KITTI copy in the MonoFlex layout (<DATA_DIR>/training/{image_2,calib,label_2,ImageSets}).
This is a one-line edit.
The `config`/`model`/`utils`/`data` imports are the upstream MonoGround packages (cwd = repo).
Run: python adapters/monoground_dump.py [--repo ...] [--ckpt ...] [--out ...]  (any cwd)
"""
import os, sys, time, csv, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths, cache_dir  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence

# NOTE: do NOT pin OMP threads here — MonoGround's postprocess (ground-depth soft
# ensemble) is CPU-heavy torch; single-threading it made the dump ~10x slower.
_ap = argparse.ArgumentParser()
_ap.add_argument("--repo", default=os.path.join(paths.UPSTREAM_ROOT, "MonoGround"))
_ap.add_argument("--ckpt", default=os.path.join(paths.UPSTREAM_ROOT, "ckpts", "monoground", "monoground.pth"))
_ap.add_argument("--out", default=os.path.join(paths.OUT_DIR, "dumps", "monoground_val.csv"))
_args = _ap.parse_args()
REPO = os.path.abspath(_args.repo)
CKPT = os.path.abspath(_args.ckpt)
OUT = os.path.abspath(_args.out)
sys.path.insert(0, REPO)
os.chdir(REPO)
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


def main():
    from config import cfg
    from model.detector import KeypointDetector
    from utils.check_point import DetectronCheckpointer
    from data import build_test_loader

    cfg.merge_from_file(os.path.join(REPO, "runs/monoground.yaml"))
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.TEST.DETECTIONS_THRESHOLD = 0.0          # full pool tap
    cfg.TEST.EVAL_DIS_IOUS = False
    cfg.TEST.EVAL_DEPTH = False
    cfg.MODEL.USE_SYNC_BN = False
    cfg.OUTPUT_DIR = "./tmp_dump"
    device = torch.device("cuda")
    model = KeypointDetector(cfg); model.to(device)
    model.test = True
    ckptr = DetectronCheckpointer(cfg, model, save_dir=cache_dir("mg_ckpt"))
    ckptr.load(CKPT, use_latest=False)
    print(f"[load] ckpt={CKPT}", flush=True)

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
            if len(result) == 0:
                continue
            ede = eval_utils.get('estimated_depth_error', None)
            vis = eval_utils.get('vis_scores', None)
            ede = ede.cpu().numpy().reshape(-1) if ede is not None else np.full(len(result), np.nan)
            vis = vis.cpu().numpy().reshape(-1) if vis is not None else result[:, 13]
            sid = int(image_ids[0])
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
                print(f"  {bi} imgs, rows={len(rows)}, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    nimg = len({r["sid"] for r in rows})
    print(f"[done] wrote {OUT} ({len(rows)} rows, {nimg} imgs) {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

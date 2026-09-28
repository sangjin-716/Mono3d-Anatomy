"""MonoDLE per-prediction val dump, 23-col CenterNet-family format
(same schema as gupnet_val.csv / monoflex_*_val.csv).

Produces data/dumps/monodle_val.csv (written to <OUT_DIR>/dumps/ by default, so a re-run never
overwrites the downloaded dump).

Read-only tap: reuses MonoDLE's dataloader / model / extract_dets_from_outputs and
replicates decode_detections math WITHOUT the score threshold, dumping every decoded
Car candidate (cls_id==1) of the K=50 heatmap top-k per image.

Column semantics (MonoDLE):
  cls = heatmap top-k score (the quantity the repo thresholds at 0.2)
  V   = final output score = heatmap * exp(-sigma_raw)  (the ranking score)
  sigma = exp(-sigma_raw) (= dets last col), log_sigma_raw = raw sigma head output
  z_pred = z_3d = decoded depth; dup_rank = #other Car preds IoU2D>=0.5 with higher V

The `lib.*` imports are the upstream MonoDLE package (the script runs with cwd = the MonoDLE
repo, whose dataset root_dir 'data/KITTI' is repo-relative). Config:
adapters/configs/monodle/kitti_accv_eval.yaml (upstream experiments/example/kitti_example.yaml
with a single GPU and an explicit root_dir).
Run (MonoDLE env, torch 1.10 in our runs):
  python adapters/monodle_dump.py [--repo <UPSTREAM_ROOT>/MonoDLE] [--ckpt ...] [--out ...]
"""
import os, sys, time, logging, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
_ap = argparse.ArgumentParser()
_ap.add_argument("--repo", default=os.path.join(paths.UPSTREAM_ROOT, "MonoDLE"))
_ap.add_argument("--ckpt", default=os.path.join(paths.UPSTREAM_ROOT, "ckpts", "monodle", "checkpoint_epoch_140.pth"))
_ap.add_argument("--cfg", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "configs", "monodle", "kitti_accv_eval.yaml"))
_ap.add_argument("--out", default=os.path.join(paths.OUT_DIR, "dumps", "monodle_val.csv"))
_args = _ap.parse_args()
REPO = os.path.abspath(_args.repo)
CKPT = os.path.abspath(_args.ckpt)
CFG = os.path.abspath(_args.cfg)
OUT = os.path.abspath(_args.out)
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np, pandas as pd, torch, yaml
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs, get_heading_angle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monodle_dump")


def iou2d_matrix(boxes):
    x1 = boxes[:, 0][:, None]; y1 = boxes[:, 1][:, None]
    x2 = boxes[:, 2][:, None]; y2 = boxes[:, 3][:, None]
    X1 = np.maximum(x1, boxes[:, 0][None]); Y1 = np.maximum(y1, boxes[:, 1][None])
    X2 = np.minimum(x2, boxes[:, 2][None]); Y2 = np.minimum(y2, boxes[:, 3][None])
    iw = np.clip(X2 - X1, 0, None); ih = np.clip(Y2 - Y1, 0, None)
    inter = iw * ih
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None] - inter
    return np.where(union > 0, inter / union, 0.0)


def main():
    cfg = yaml.load(open(CFG), Loader=yaml.Loader)
    train_loader, test_loader = build_dataloader(cfg["dataset"])
    model = build_model(cfg["model"])
    load_checkpoint(model=model, optimizer=None, filename=CKPT, logger=logger,
                    map_location="cuda:0")
    model.cuda().eval()
    ds = test_loader.dataset
    cms = ds.cls_mean_size
    CAR = 1  # ds.cls2id['Car']

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for bi, (inputs, _, info) in enumerate(test_loader):
            outputs = model(inputs.cuda())
            dets = extract_dets_from_outputs(outputs=outputs, K=50).detach().cpu().numpy()
            info_np = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
                       for k, v in info.items()}
            calibs = [ds.get_calib(int(idx)) for idx in info_np["img_id"]]
            B = dets.shape[0]
            for i in range(B):
                sid = int(info_np["img_id"][i])
                rx = float(info_np["bbox_downsample_ratio"][i][0])
                ry_ratio = float(info_np["bbox_downsample_ratio"][i][1])
                rec_local = []
                for j in range(dets.shape[1]):
                    if int(dets[i, j, 0]) != CAR:
                        continue
                    hm = float(dets[i, j, 1])           # heatmap score (threshold target)
                    sig_t = float(dets[i, j, -1])       # exp(-sigma_raw)
                    V = hm * sig_t                      # final output score
                    x = dets[i, j, 2] * rx; y = dets[i, j, 3] * ry_ratio
                    w = dets[i, j, 4] * rx; h = dets[i, j, 5] * ry_ratio
                    bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
                    depth = float(dets[i, j, 6])
                    dims = dets[i, j, 31:34] + cms[CAR]
                    x3d = dets[i, j, 34] * rx; y3d = dets[i, j, 35] * ry_ratio
                    loc = calibs[i].img_to_rect(np.array([x3d]), np.array([y3d]),
                                                np.array([depth])).reshape(-1)
                    loc[1] += dims[0] / 2
                    alpha = float(get_heading_angle(dets[i, j, 7:31]))
                    ry = float(calibs[i].alpha2ry(alpha, x3d))
                    rec_local.append(dict(
                        sid=sid, pred_idx=j, cls=hm, sigma=sig_t,
                        log_sigma_raw=float(-np.log(max(sig_t, 1e-12))), V=V, z_pred=depth,
                        bbox_h_pix=float(bbox[3] - bbox[1]),
                        bbox_w_pix=float(bbox[2] - bbox[0]),
                        bbox_area_pix=float((bbox[3] - bbox[1]) * (bbox[2] - bbox[0])),
                        x_3d=float(loc[0]), y_3d=float(loc[1]), z_3d=float(loc[2]),
                        h_3d=float(dims[0]), w_3d=float(dims[1]), l_3d=float(dims[2]),
                        ry=ry, alpha=alpha,
                        bbox_x1=float(bbox[0]), bbox_y1=float(bbox[1]),
                        bbox_x2=float(bbox[2]), bbox_y2=float(bbox[3]), dup_rank=0))
                if rec_local:
                    boxes = np.array([[r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]]
                                      for r in rec_local])
                    vv = np.array([r["V"] for r in rec_local])
                    iou = iou2d_matrix(boxes)
                    for q in range(len(rec_local)):
                        nb = iou[q] >= 0.5; nb[q] = False
                        rec_local[q]["dup_rank"] = int(((vv > vv[q]) & nb).sum())
                rows.extend(rec_local)
            if bi % 30 == 0:
                print(f"  batch {bi}/{len(test_loader)} rows={len(rows)} {time.time()-t0:.0f}s",
                      flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    g = df.groupby("sid").size()
    print(f"[done] {len(df)} rows, sids={df.sid.nunique()}, "
          f"rows/sid min/med/max={g.min()}/{int(g.median())}/{g.max()}, "
          f"{time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

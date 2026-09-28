"""MonoCon (2gunsu reimpl) per-prediction val dump — 23-col CenterNet-family format.

Ported from tools/monocon_dump.py for the public release. Computation unchanged.
Produces data/dumps/monocon_val.csv (written to <OUT_DIR>/dumps/ by default, so a re-run never
overwrites the downloaded dump).

Read-only tap: sets head.test_thres = -1e9 so all K=30 native top-k candidates pass,
then reuses the repo's OWN _get_eval_formats pipeline (origin shift, validity filter,
2D clip, KITTI anno dict) — guaranteeing identical geometry to the native writer.
log_sigma_raw / heatmap score / pred_idx are recomputed from pred_dict with the same
get_local_maximum+topk and attached to anno rows by exact combined-score matching.

Column semantics (MonoCon, like GUPNet/DEVIANT):
  cls = V = combined score (heatmap * exp(-d1)) — the quantity natively thresholded (0.4)
  log_sigma_raw = depth head channel 1 (d1); sigma = exp(-d1)
  K = 30 (native max_objs; smaller than other CenterNets' 50 — native pool definition)

Config: the config.yaml shipped next to the released 2gunsu checkpoint; the original run used a
copy (config_accv.yaml) whose only difference was DATA.ROOT, which this port sets from
--kitti_root (default paths.KITTI_ROOT). The upstream engine is used as is except for
adapters/patches/MonoCon_base_engine_map_location.patch (torch.load(..., map_location='cuda:0')).
Run (MonoCon env, torch 1.10 in our runs; any cwd):
  python adapters/monocon_dump.py [--repo <UPSTREAM_ROOT>/MonoCon] [--cfg ...] [--ckpt ...]
"""
import os, sys, time, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
_ap = argparse.ArgumentParser()
_ap.add_argument("--repo", default=os.path.join(paths.UPSTREAM_ROOT, "MonoCon"))
_ap.add_argument("--cfg", default=os.path.join(paths.UPSTREAM_ROOT, "ckpts", "monocon", "pretrained", "config.yaml"))
_ap.add_argument("--ckpt", default=os.path.join(paths.UPSTREAM_ROOT, "ckpts", "monocon", "pretrained", "best.pth"))
_ap.add_argument("--kitti_root", default=paths.KITTI_ROOT)
_ap.add_argument("--out", default=os.path.join(paths.OUT_DIR, "dumps", "monocon_val.csv"))
_args = _ap.parse_args()
REPO = os.path.abspath(_args.repo)
CFG = os.path.abspath(_args.cfg)
CKPT = os.path.abspath(_args.ckpt)
OUT = os.path.abspath(_args.out)
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np, pandas as pd, torch
from tqdm import tqdm
from engine.monocon_engine import MonoconEngine
from utils.engine_utils import load_cfg, move_data_device
from model.dense_heads.monocon_heads import get_local_maximum, get_topk_from_heatmap, \
    transpose_and_gather_feat


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
    cfg = load_cfg(CFG)
    cfg.DATA.ROOT = os.path.abspath(_args.kitti_root)   # the only change of config_accv.yaml
    cfg.GPU_ID = 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine = MonoconEngine(cfg, auto_resume=False, is_test=True)
    engine.load_checkpoint(CKPT, verbose=True)
    model = engine.model
    model.eval()
    model.head.test_thres = -1e9        # dump-all tap (native K=30 kept)
    K = model.head.topk
    dev = engine.current_device

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for test_data in tqdm(engine.test_loader, desc="dump"):
            test_data = move_data_device(test_data, dev)
            pred_dict = model(test_data, return_loss=False)
            ef = model.head._get_eval_formats(test_data, pred_dict)
            annos = ef["img_bbox"]      # list per image: kitti anno dicts

            # recompute the SAME topk internals for sigma / heatmap-score columns
            hm = get_local_maximum(pred_dict["center_heatmap_pred"],
                                   kernel=model.head.local_maximum_kernel)
            scores_hm, indices, _, _, _ = get_topk_from_heatmap(hm, k=K)     # (B,K)
            d = transpose_and_gather_feat(pred_dict["depth_pred"], indices)  # (B,K,2)
            d1 = d[:, :, 1]
            combined = (scores_hm * torch.exp(-d1)).cpu().numpy()            # (B,K)
            d1 = d1.cpu().numpy()

            for i, anno in enumerate(annos):
                sid = int(anno["sample_idx"][0]) if len(anno["name"]) else None
                if sid is None:
                    continue
                comb_i = combined[i]; d1_i = d1[i]
                rec_local = []
                for j in range(len(anno["name"])):
                    if anno["name"][j] != "Car":
                        continue
                    score = float(anno["score"][j])
                    # match this anno row back to its topk slot by exact score
                    k_idx = int(np.argmin(np.abs(comb_i - score)))
                    bbox = anno["bbox"][j]            # x1,y1,x2,y2 (clipped, orig px)
                    dim = anno["dimensions"][j]       # anno convention (l,h,w)
                    loc = anno["location"][j]         # bottom-center camera coords
                    lsr = float(d1_i[k_idx])
                    rec_local.append(dict(
                        sid=sid, pred_idx=k_idx, cls=score, sigma=float(np.exp(-lsr)),
                        log_sigma_raw=lsr, V=score, z_pred=float(loc[2]),
                        bbox_h_pix=float(bbox[3] - bbox[1]),
                        bbox_w_pix=float(bbox[2] - bbox[0]),
                        bbox_area_pix=float((bbox[3] - bbox[1]) * (bbox[2] - bbox[0])),
                        x_3d=float(loc[0]), y_3d=float(loc[1]), z_3d=float(loc[2]),
                        h_3d=float(dim[1]), w_3d=float(dim[2]), l_3d=float(dim[0]),
                        ry=float(anno["rotation_y"][j]), alpha=float(anno["alpha"][j]),
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

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    g = df.groupby("sid").size()
    print(f"[done] {len(df)} rows, sids={df.sid.nunique()}, "
          f"rows/sid min/med/max={g.min()}/{int(g.median())}/{g.max()}, "
          f"{time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

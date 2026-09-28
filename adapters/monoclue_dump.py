"""MonoCLUE val per-detection dump -> shared 23-col schema (matches monodetr_val.csv).

Rewritten from the one-off script that wrote the released monoclue_val.csv. Produces
data/dumps/monoclue_val.csv (written to <OUT_DIR>/dumps/monoclue_val.csv by default, so a re-run
never overwrites the downloaded dump).
Config: the repo's configs/monoclue.yaml with dataset.root_dir set to --kitti_root (applied in
memory). --ckpt is the released checkpoint (default <repo>/outputs/monoclue/checkpoint_best.pth).
Same NATIVE decode as MonoDETR/DGP (extract_dets_from_outputs, score=cls*sigma, NMS-free).
SAM=False (sam is a train-only target; inference doesn't need it).
The lib.* imports are the upstream MonoCLUE packages (--repo). Run in a MonoCLUE env (torch 1.9
+ MSDeformAttn in our runs):
  python adapters/monoclue_dump.py --repo <UPSTREAM_ROOT>/MonoCLUE --ckpt <MonoCLUE released .pth>
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, csv, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence
_ap = argparse.ArgumentParser()
_ap.add_argument("--repo", default=os.path.join(paths.UPSTREAM_ROOT, "MonoCLUE"))
_ap.add_argument("--cfg", default=None, help="default <repo>/configs/monoclue.yaml")
_ap.add_argument("--ckpt", default=None, help="default <repo>/outputs/monoclue/checkpoint_best.pth")
_ap.add_argument("--kitti_root", default=paths.KITTI_ROOT)
_ap.add_argument("--out", default=os.path.join(paths.OUT_DIR, "dumps", "monoclue_val.csv"))
ARGS = _ap.parse_args()
REPO = os.path.abspath(ARGS.repo)
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from lib.helpers.model_helper import build_model
from lib.helpers.decode_helper import extract_dets_from_outputs, get_heading_angle
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset

FIELDS = ["sid", "pred_idx", "cls", "sigma", "log_sigma_raw", "V", "z_pred",
          "bbox_h_pix", "bbox_w_pix", "bbox_area_pix", "x_3d", "y_3d", "z_3d",
          "h_3d", "w_3d", "l_3d", "ry", "alpha",
          "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "dup_rank"]
CAR = 1
OUT = ARGS.out


def iou2d_mat(b):
    x1 = np.maximum(b[:, 0][:, None], b[:, 0][None]); y1 = np.maximum(b[:, 1][:, None], b[:, 1][None])
    x2 = np.minimum(b[:, 2][:, None], b[:, 2][None]); y2 = np.minimum(b[:, 3][:, None], b[:, 3][None])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None); inter = iw * ih
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); u = a[:, None] + a[None] - inter
    return np.where(u > 0, inter / u, 0.0)


def main():
    cfg = yaml.load(open(ARGS.cfg or os.path.join(REPO, "configs", "monoclue.yaml")), Loader=yaml.Loader)
    cfg["dataset"]["root_dir"] = ARGS.kitti_root.rstrip("/") + "/"
    ds = KITTI_Dataset(split="val", cfg=cfg["dataset"], SAM=False)
    loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=False)
    cms = ds.cls_mean_size
    device = torch.device("cuda")
    model, _ = build_model(cfg["model"]); model = model.to(device).eval()
    st = torch.load(ARGS.ckpt or "%s/outputs/monoclue/checkpoint_best.pth" % REPO, map_location=device)
    miss = model.load_state_dict(st.get("model_state", st), strict=False)
    print(f"[load] missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)} epoch={st.get('epoch')}", flush=True)
    rows = []
    with torch.no_grad():
        for bi, (inputs, calibs, targets, info) in enumerate(loader):
            inputs = inputs.to(device); calibs = calibs.to(device)
            img_sizes = info["img_size"].to(device)
            outputs = model(inputs, calibs, img_sizes, dn_args=0)
            dets = extract_dets_from_outputs(outputs=outputs, K=50, topk=50).detach().cpu().numpy()
            imgsz = info["img_size"].numpy(); ids = info["img_id"].numpy()
            for i in range(dets.shape[0]):
                sid = int(ids[i]); W, H = float(imgsz[i][0]), float(imgsz[i][1])
                cal = ds.get_calib(sid); recs = []
                for j in range(dets.shape[1]):
                    if int(dets[i, j, 0]) != CAR:
                        continue
                    clsc = float(dets[i, j, 1]); sg = float(dets[i, j, 36]); V = clsc * sg
                    x = float(dets[i, j, 2] * W); y = float(dets[i, j, 3] * H)
                    w = float(dets[i, j, 4] * W); h2 = float(dets[i, j, 5] * H)
                    bbox = [x - w/2, y - h2/2, x + w/2, y + h2/2]
                    depth = float(dets[i, j, 6])
                    dims = dets[i, j, 31:34] + cms[CAR]
                    h3, w3, l3 = float(dims[0]), float(dims[1]), float(dims[2])
                    x3 = float(dets[i, j, 34] * W); y3 = float(dets[i, j, 35] * H)
                    loc = cal.img_to_rect(np.array([x3]), np.array([y3]), np.array([depth])).reshape(-1)
                    lx, ly, lz = float(loc[0]), float(loc[1]) + h3/2.0, float(loc[2])
                    alpha = float(get_heading_angle(dets[i, j, 7:31])); ry = float(cal.alpha2ry(alpha, x))
                    sg = max(sg, 1e-6)
                    recs.append((clsc, sg, V, depth, bbox, lx, ly, lz, h3, w3, l3, ry, alpha))
                if not recs:
                    continue
                boxes = np.array([r[4] for r in recs]); Vs = np.array([r[2] for r in recs])
                iou = iou2d_mat(boxes)
                for q, r in enumerate(recs):
                    nb = (iou[q] >= 0.5).copy(); nb[q] = False
                    dup = int(((Vs > Vs[q]) & nb).sum())
                    clsc, sg, V, depth, bbox, lx, ly, lz, h3, w3, l3, ry, alpha = r
                    rows.append(dict(sid=sid, pred_idx=q, cls=clsc, sigma=sg, log_sigma_raw=float(-np.log(sg)),
                        V=V, z_pred=depth, bbox_h_pix=bbox[3]-bbox[1], bbox_w_pix=bbox[2]-bbox[0],
                        bbox_area_pix=(bbox[3]-bbox[1])*(bbox[2]-bbox[0]),
                        x_3d=lx, y_3d=ly, z_3d=lz, h_3d=h3, w_3d=w3, l_3d=l3, ry=ry, alpha=alpha,
                        bbox_x1=bbox[0], bbox_y1=bbox[1], bbox_x2=bbox[2], bbox_y2=bbox[3], dup_rank=dup))
            if bi % 50 == 0:
                print(f"  batch {bi}/{len(loader)} rows={len(rows)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {OUT} ({len(rows)} rows, {len({r['sid'] for r in rows})} imgs)", flush=True)


if __name__ == "__main__":
    main()

"""MonoDETR per-detection dump adapter -> shared 23-col schema (val+train).

Produces data/dumps/monodetr_val.csv (the script writes monodetr_val.csv and monodetr_train.csv
into --outdir, default <OUT_DIR>/dumps/).
Config: the repo's configs/monodetr.yaml with dataset.root_dir set from --root_dir
(default paths.KITTI_ROOT). The `lib.*` imports are the upstream MonoDETR package (--repo).
Uses MonoDETR's NATIVE extract_dets_from_outputs (top-K over query×class) so the
dump is faithful (reproduces official mod 20.83); dgp_cop_dump's per-query decode
differs (gave 20.13). Inference-only, repo read-only.

MonoDETR detections tensor cols: [0]label [1]cls(=topk prob, the THRESHOLD score)
[2-3]xs2d/ys2d [4-5]size_2d(w,h) [6]depth [7-30]heading [31-33]size_3d-offset
[34-35]xs3d/ys3d [36]sigma(=exp(-logσ)). Written KITTI score = cls·sigma (so V=cls·σ);
threshold is on cls. loc = img_to_rect(xs3d*W, ys3d*H, depth); y+=h/2. dims=col31:34+mean.
Run in a MonoDETR env (torch 1.9 with MSDeformAttn built, in our runs).
"""
import os, sys, time, csv, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--repo", default=os.path.join(paths.UPSTREAM_ROOT, "MonoDETR"))
REPO = os.path.abspath(_pre.parse_known_args()[0].repo)
sys.path.insert(0, REPO)
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from lib.helpers.model_helper import build_model
from lib.helpers.decode_helper import extract_dets_from_outputs, get_heading_angle
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset

FIELDS = ["sid", "pred_idx", "cls", "sigma", "log_sigma_raw", "V", "z_pred",
          "bbox_h_pix", "bbox_w_pix", "bbox_area_pix", "x_3d", "y_3d", "z_3d",
          "h_3d", "w_3d", "l_3d", "ry", "alpha",
          "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "dup_rank"]
CAR = 1   # cls2id: Pedestrian0 Car1 Cyclist2


def iou2d_mat(b):
    x1 = np.maximum(b[:, 0][:, None], b[:, 0][None]); y1 = np.maximum(b[:, 1][:, None], b[:, 1][None])
    x2 = np.minimum(b[:, 2][:, None], b[:, 2][None]); y2 = np.minimum(b[:, 3][:, None], b[:, 3][None])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None); inter = iw * ih
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); u = a[:, None] + a[None] - inter
    return np.where(u > 0, inter / u, 0.0)


def run_split(model, ds_cfg, split, device, out_csv):
    ds = KITTI_Dataset(split=split, cfg=dict(ds_cfg))
    if hasattr(ds, "data_augmentation"): ds.data_augmentation = False
    loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=False,
                        collate_fn=getattr(ds, "collate_fn", None))
    cms = ds.cls_mean_size
    rows = []; t0 = time.time()
    model.eval()
    with torch.no_grad():
        for bi, (inputs, calibs, targets, info) in enumerate(loader):
            inputs = inputs.to(device); calibs = calibs.to(device)
            img_sizes = info["img_size"].to(device)
            for k in targets:
                if torch.is_tensor(targets[k]): targets[k] = targets[k].to(device)
            outputs = model(inputs, calibs, targets, img_sizes, dn_args=0)
            dets = extract_dets_from_outputs(outputs=outputs, K=50, topk=50).detach().cpu().numpy()
            B = dets.shape[0]
            imgsz = info["img_size"].numpy() if torch.is_tensor(info["img_size"]) else np.asarray(info["img_size"])
            ids = info["img_id"].numpy() if torch.is_tensor(info["img_id"]) else np.asarray(info["img_id"])
            for i in range(B):
                sid = int(ids[i]); W, H = float(imgsz[i][0]), float(imgsz[i][1])
                cal = loader.dataset.get_calib(sid)
                recs = []
                for j in range(dets.shape[1]):
                    if int(dets[i, j, 0]) != CAR:
                        continue
                    clsc = float(dets[i, j, 1]); sg = float(dets[i, j, 36])
                    V = clsc * sg
                    x = float(dets[i, j, 2] * W); y = float(dets[i, j, 3] * H)
                    w = float(dets[i, j, 4] * W); h2 = float(dets[i, j, 5] * H)
                    bbox = [x - w/2, y - h2/2, x + w/2, y + h2/2]
                    depth = float(dets[i, j, 6])
                    dims = dets[i, j, 31:34] + cms[CAR]
                    h3, w3, l3 = float(dims[0]), float(dims[1]), float(dims[2])
                    x3 = float(dets[i, j, 34] * W); y3 = float(dets[i, j, 35] * H)
                    loc = cal.img_to_rect(np.array([x3]), np.array([y3]), np.array([depth])).reshape(-1)
                    lx, ly, lz = float(loc[0]), float(loc[1]) + h3/2.0, float(loc[2])
                    alpha = float(get_heading_angle(dets[i, j, 7:31]))
                    ry = float(cal.alpha2ry(alpha, x))
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
                    rows.append(dict(
                        sid=sid, pred_idx=q, cls=clsc, sigma=sg, log_sigma_raw=float(-np.log(sg)),
                        V=V, z_pred=depth, bbox_h_pix=bbox[3]-bbox[1], bbox_w_pix=bbox[2]-bbox[0],
                        bbox_area_pix=(bbox[3]-bbox[1])*(bbox[2]-bbox[0]),
                        x_3d=lx, y_3d=ly, z_3d=lz, h_3d=h3, w_3d=w3, l_3d=l3, ry=ry, alpha=alpha,
                        bbox_x1=bbox[0], bbox_y1=bbox[1], bbox_x2=bbox[2], bbox_y2=bbox[3], dup_rank=dup))
            if bi % 50 == 0:
                print(f"  [{split}] batch {bi}/{len(loader)} rows={len(rows)} {time.time()-t0:.0f}s", flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[{split}] wrote {out_csv} ({len(rows)} rows, {len({r['sid'] for r in rows})} imgs) {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--ckpt", default=os.path.join(paths.UPSTREAM_ROOT, "ckpts", "monodetr", "monodetr_best.pth"))
    ap.add_argument("--cfg", default=os.path.join(REPO, "configs", "monodetr.yaml"))
    ap.add_argument("--root_dir", default=paths.KITTI_ROOT, help="dataset.root_dir override (as in the original run)")
    ap.add_argument("--outdir", default=os.path.join(paths.OUT_DIR, "dumps"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    cfg["dataset"]["root_dir"] = args.root_dir
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda")
    model, _ = build_model(cfg["model"]); model = model.to(device)
    st = torch.load(args.ckpt, map_location=device); sd = st.get("model_state", st)
    miss = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(getattr(miss,'missing_keys',[]))} unexpected={len(getattr(miss,'unexpected_keys',[]))}", flush=True)
    for split in ["val", "train"]:
        run_split(model, cfg["dataset"], split, device, os.path.join(args.outdir, f"monodetr_{split}.csv"))


if __name__ == "__main__":
    main()

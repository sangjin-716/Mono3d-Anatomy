"""GUPNet per-detection dump adapter -> shared 23-col schema.

Ported from tools/decomp/adapters/gupnet_dump.py for the public release. Computation unchanged.
Produces data/dumps/gupnet_val.csv (the script writes gupnet_val.csv and gupnet_train.csv into
--outdir, default <OUT_DIR>/dumps/).

Inference-only, read-only on the repo. Dumps BOTH val+train (model loaded once),
threshold 0 (full top-50 Car pool), augmentation OFF (clean train inference).

GUPNet decode (mirrors lib/helpers/decode_helper.decode_detections):
  detection tensor [B,50,36]: [0]cls(Car=1) [1]V=heatmap*exp(-sqrt(σ)) [2-3]2Dcenter
  [4-5]2Dsize [6]depth z [7-30]heading [31-33]size3d-offset [34-35]3Dproj-center.
  σ tapped from outputs['depth'].view(B,K,2)[:,:,1] (=logσ), aligned to dets rows.
  loc = img_to_rect(x3d,y3d,depth); y += h/2 (bottom-center). dims = dets[31:34]+mean (=h,w,l).
  V is BOTH the threshold (≥0.2) and ranking score → cls:=V.
The `lib.*` imports are the upstream GUPNet package (<repo>/code). --root is the GUPNet data
root: <root>/KITTI/{ImageSets,training}. Run in a GUPNet env (torch 1.9 in our runs), any cwd:
  python adapters/gupnet_dump.py [--code <UPSTREAM_ROOT>/GUPNet/code] [--ckpt ...] [--root ...]
"""
import os, sys, time, csv, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--code", default=os.path.join(paths.UPSTREAM_ROOT, "GUPNet", "code"))
GUP = os.path.abspath(_pre.parse_known_args()[0].code)
sys.path.insert(0, GUP)
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from lib.datasets.kitti import KITTI
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs, get_heading_angle

FIELDS = ["sid", "pred_idx", "cls", "sigma", "log_sigma_raw", "V", "z_pred",
          "bbox_h_pix", "bbox_w_pix", "bbox_area_pix", "x_3d", "y_3d", "z_3d",
          "h_3d", "w_3d", "l_3d", "ry", "alpha",
          "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "dup_rank"]


class _Log:
    def info(self, *a, **k): pass


def iou2d_mat(b):
    x1 = np.maximum(b[:, 0][:, None], b[:, 0][None]); y1 = np.maximum(b[:, 1][:, None], b[:, 1][None])
    x2 = np.minimum(b[:, 2][:, None], b[:, 2][None]); y2 = np.minimum(b[:, 3][:, None], b[:, 3][None])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None); inter = iw * ih
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); u = a[:, None] + a[None] - inter
    return np.where(u > 0, inter / u, 0.0)


def _np(x):
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def run_split(model, ds_cfg, root, split, device, cms, out_csv):
    ds = KITTI(root_dir=root, split=split, cfg=dict(ds_cfg)); ds.data_augmentation = False
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, drop_last=False)
    rows = []; t0 = time.time()
    model.eval()
    with torch.no_grad():
        for bi, (inputs, calibs, coord_ranges, _, info) in enumerate(loader):
            inputs = inputs.to(device); calibs_t = calibs.to(device); coord_ranges = coord_ranges.to(device)
            outputs = model(inputs, coord_ranges, calibs_t, K=50, mode='test')
            B = inputs.shape[0]
            dets = extract_dets_from_outputs(outputs=outputs, K=50).detach().cpu().numpy()
            sigma = np.exp(outputs['depth'].view(B, 50, -1)[:, :, 1].detach().cpu().numpy())   # σ aligned to dets rows
            ids = _np(info['img_id']); ratios = _np(info['bbox_downsample_ratio'])
            for i in range(B):
                sid = int(ids[i]); rx, ry = float(ratios[i][0]), float(ratios[i][1])
                cal = ds.get_calib(sid)
                recs = []
                for j in range(dets.shape[1]):
                    if int(dets[i, j, 0]) != 1:        # Car = class 1
                        continue
                    V = float(dets[i, j, 1])
                    x = float(dets[i, j, 2] * rx); y = float(dets[i, j, 3] * ry)
                    w = float(dets[i, j, 4] * rx); h2 = float(dets[i, j, 5] * ry)
                    bbox = [x - w / 2, y - h2 / 2, x + w / 2, y + h2 / 2]
                    depth = float(dets[i, j, 6])
                    alpha = float(get_heading_angle(dets[i, j, 7:31]))
                    ryaw = float(cal.alpha2ry(alpha, x))
                    dims = dets[i, j, 31:34] + cms[1]    # Car mean -> [h,w,l]
                    if (dims < 0).any():
                        continue
                    h3, w3, l3 = float(dims[0]), float(dims[1]), float(dims[2])
                    x3 = float(dets[i, j, 34] * rx); y3 = float(dets[i, j, 35] * ry)
                    lx = (x3 - cal.cu) * depth / cal.fu + cal.tx
                    ly = (y3 - cal.cv) * depth / cal.fv + cal.ty + h3 / 2.0
                    sg = max(float(sigma[i, j]), 1e-6)
                    recs.append((V, bbox, depth, alpha, ryaw, h3, w3, l3, float(lx), float(ly), depth, sg))
                if not recs:
                    continue
                boxes = np.array([r[1] for r in recs]); Vs = np.array([r[0] for r in recs])
                iou = iou2d_mat(boxes)
                for q, r in enumerate(recs):
                    nb = (iou[q] >= 0.5).copy(); nb[q] = False
                    dup = int(((Vs > Vs[q]) & nb).sum())
                    V, bbox, depth, alpha, ryaw, h3, w3, l3, lx, ly, lz, sg = r
                    rows.append(dict(
                        sid=sid, pred_idx=q, cls=V, sigma=float(np.exp(-np.log(sg))),
                        log_sigma_raw=float(np.log(sg)), V=V, z_pred=depth,
                        bbox_h_pix=bbox[3] - bbox[1], bbox_w_pix=bbox[2] - bbox[0],
                        bbox_area_pix=(bbox[3] - bbox[1]) * (bbox[2] - bbox[0]),
                        x_3d=lx, y_3d=ly, z_3d=lz, h_3d=h3, w_3d=w3, l_3d=l3, ry=ryaw, alpha=alpha,
                        bbox_x1=bbox[0], bbox_y1=bbox[1], bbox_x2=bbox[2], bbox_y2=bbox[3], dup_rank=dup))
            if bi % 100 == 0:
                print(f"  [{split}] batch {bi}/{len(loader)} rows={len(rows)} {time.time()-t0:.0f}s", flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[{split}] wrote {out_csv} ({len(rows)} rows, {len({r['sid'] for r in rows})} imgs) {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=GUP)
    ap.add_argument("--ckpt", default=os.path.join(paths.UPSTREAM_ROOT, "ckpts", "gupnet", "gupnet_val.pth"))
    ap.add_argument("--cfg", default=os.path.join(GUP, "experiments/config.yaml"))
    ap.add_argument("--root", default=os.path.join(os.path.dirname(GUP), "data"))
    ap.add_argument("--outdir", default=os.path.join(paths.OUT_DIR, "dumps"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cfg = yaml.safe_load(open(args.cfg))
    ds_cfg = cfg['dataset']; ds_cfg['root_dir'] = args.root
    device = torch.device("cuda")
    # need cls_mean_size for model build + decode
    probe = KITTI(root_dir=args.root, split='val', cfg=dict(ds_cfg))
    cms = probe.cls_mean_size
    model = build_model(cfg['model'], cms)
    load_checkpoint(model=model, optimizer=None, filename=args.ckpt, logger=_Log(), map_location=device)
    model.to(device)
    print(f"[load] ckpt={args.ckpt}", flush=True)
    for split in ["val", "train"]:
        run_split(model, ds_cfg, args.root, split, device, cms, os.path.join(args.outdir, f"gupnet_{split}.csv"))


if __name__ == "__main__":
    main()

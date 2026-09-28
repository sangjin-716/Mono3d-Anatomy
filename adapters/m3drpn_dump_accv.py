# M3D-RPN per-prediction val dump — 23-col format (anchor-family adapter).
#
# Ported from M3D-RPN/scripts/m3drpn_dump_accv.py (our script inside the upstream M3D-RPN clone)
# for the public release. Computation unchanged. Produces data/dumps/m3drpn_val.csv
# (written to <OUT_DIR>/dumps/m3drpn_val.csv by default, so a re-run never overwrites the
# downloaded dump).
#
# Tap = PRE-NMS candidate pool of im_detect_3d (top nms_topN_pre=3000 by score),
# Car rows with score >= 0.05 (cap 300/img), each decoded EXACTLY like the native
# writer in test_kitti_3d: convertAlpha2Rot -> hill_climb -> convertRot2Alpha,
# y3d += h3d/2. File ids are the renumbered split ids; sid column is mapped back
# to original KITTI ids. The mapping is new_id i -> i-th entry of ImageSets/val.txt (the order
# in which data/kitti_split1/setup_split.py renumbers the val images); the original run read the
# same mapping from a CSV (val_id_map.csv, identical content; pass it with --id_map if you have it).
#
# Column notes: cls = V = class softmax score (the only score M3D-RPN has; its
# writer thresholds the same quantity at 0.75). No uncertainty head exists ->
# sigma = 1.0, log_sigma_raw = 0.0 (constant; sigma-based probes see it as flat).
#
# Run from the M3D-RPN repo root (it imports the repo's lib.*), in a torch-1.x env with the
# py_cpu_nms patch applied (adapters/patches/M3D-RPN_rpn_util_py_cpu_nms.patch):
#   cd <UPSTREAM_ROOT>/M3D-RPN && python <release>/adapters/m3drpn_dump_accv.py
import os, sys, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence

from importlib import import_module
from easydict import EasyDict as edict
import time, math
import numpy as np

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from lib.imdb_util import *   # Preprocess, read_kitti_cal, list_files, file_parts, ...
from lib.rpn_util import bbox_transform_inv, hill_climb, convertAlpha2Rot, convertRot2Alpha
import torch
import pandas as pd
from time import time as _now   # star-import above shadows the time module

_ap = argparse.ArgumentParser()
_ap.add_argument("--release_dir", default=os.path.join(paths.UPSTREAM_ROOT, "ckpts", "m3drpn", "M3D-RPN-Release"),
                 help="unzipped M3D-RPN-Release.zip (official val1 model)")
_ap.add_argument("--id_map", default=None, help="optional CSV new_id,orig_sid (default: from paths.VAL_LIST)")
_ap.add_argument("--out", default=os.path.join(paths.OUT_DIR, "dumps", "m3drpn_val.csv"))
_args = _ap.parse_args()

conf_path = os.path.join(_args.release_dir, 'm3d_rpn_depth_aware_val1_config.pkl')
weights_path = os.path.join(_args.release_dir, 'm3d_rpn_depth_aware_val1')
OUT = _args.out
SCORE_MIN = 0.05
CAP_PER_IMG = 300

conf = edict(pickle_read(conf_path))
conf.pretrained = None
data_path = os.path.join(os.getcwd(), 'data')

init_torch(conf.rng_seed, conf.cuda_seed)
net = import_module('models.' + conf.model).build(conf)
load_weights(net, weights_path, remove_module=True)
net.eval()

if _args.id_map:
    id_map = {r.split(',')[0]: int(r.split(',')[1]) for r in
              open(_args.id_map).read().strip().split('\n')[1:]}
else:
    id_map = {"%06d" % i: int(s) for i, s in enumerate(open(paths.VAL_LIST).read().split())}


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


def detect_prenms(im, net, rpn_conf, preprocess, p2):
    """im_detect_3d body with the NMS step REMOVED — returns the sorted pre-NMS
    top-3000 pool: aboxes[N, 13] = x1,y1,x2,y2,score,cls_id, x3d,y3d,z3d,w3d,h3d,l3d,ry3d."""
    imH_orig, imW_orig = im.shape[0], im.shape[1]
    im = preprocess(im)
    im = torch.from_numpy(im[np.newaxis, :, :, :]).cuda()
    scale_factor = im.shape[2] / imH_orig

    cls, prob, bbox_2d, bbox_3d, feat_size, rois = net(im)

    bbox_x = bbox_2d[:, :, 0]; bbox_y = bbox_2d[:, :, 1]
    bbox_w = bbox_2d[:, :, 2]; bbox_h = bbox_2d[:, :, 3]

    bbox_x3d = bbox_3d[:, :, 0]; bbox_y3d = bbox_3d[:, :, 1]; bbox_z3d = bbox_3d[:, :, 2]
    bbox_w3d = bbox_3d[:, :, 3]; bbox_h3d = bbox_3d[:, :, 4]; bbox_l3d = bbox_3d[:, :, 5]
    bbox_ry3d = bbox_3d[:, :, 6]

    bbox_x3d = bbox_x3d * conf.bbox_stds[:, 4][0] + conf.bbox_means[:, 4][0]
    bbox_y3d = bbox_y3d * conf.bbox_stds[:, 5][0] + conf.bbox_means[:, 5][0]
    bbox_z3d = bbox_z3d * conf.bbox_stds[:, 6][0] + conf.bbox_means[:, 6][0]
    bbox_w3d = bbox_w3d * conf.bbox_stds[:, 7][0] + conf.bbox_means[:, 7][0]
    bbox_h3d = bbox_h3d * conf.bbox_stds[:, 8][0] + conf.bbox_means[:, 8][0]
    bbox_l3d = bbox_l3d * conf.bbox_stds[:, 9][0] + conf.bbox_means[:, 9][0]
    bbox_ry3d = bbox_ry3d * conf.bbox_stds[:, 10][0] + conf.bbox_means[:, 10][0]

    tracker = rois[:, 4].cpu().detach().numpy().astype(np.int64)
    src_3d = torch.from_numpy(conf.anchors[tracker, 4:]).cuda().type(torch.cuda.FloatTensor)

    widths = rois[:, 2] - rois[:, 0] + 1.0
    heights = rois[:, 3] - rois[:, 1] + 1.0
    ctr_x = rois[:, 0] + 0.5 * widths
    ctr_y = rois[:, 1] + 0.5 * heights

    bbox_x3d = bbox_x3d[0, :] * widths + ctr_x
    bbox_y3d = bbox_y3d[0, :] * heights + ctr_y
    bbox_z3d = src_3d[:, 0] + bbox_z3d[0, :]
    bbox_w3d = torch.exp(bbox_w3d[0, :]) * src_3d[:, 1]
    bbox_h3d = torch.exp(bbox_h3d[0, :]) * src_3d[:, 2]
    bbox_l3d = torch.exp(bbox_l3d[0, :]) * src_3d[:, 3]
    bbox_ry3d = src_3d[:, 4] + bbox_ry3d[0, :]

    coords_3d = torch.stack((bbox_x3d, bbox_y3d, bbox_z3d[:bbox_x3d.shape[0]],
                             bbox_w3d[:bbox_x3d.shape[0]], bbox_h3d[:bbox_x3d.shape[0]],
                             bbox_l3d[:bbox_x3d.shape[0]], bbox_ry3d[:bbox_x3d.shape[0]]), dim=1)

    deltas_2d = torch.cat((bbox_x[0, :, np.newaxis], bbox_y[0, :, np.newaxis],
                           bbox_w[0, :, np.newaxis], bbox_h[0, :, np.newaxis]), dim=1)
    coords_2d = bbox_transform_inv(rois, deltas_2d, means=conf.bbox_means[0, :],
                                   stds=conf.bbox_stds[0, :])

    coords_2d = coords_2d.cpu().detach().numpy()
    coords_3d = coords_3d.cpu().detach().numpy()
    prob = prob[0, :, :].cpu().detach().numpy()

    coords_2d[:, 0:4] /= scale_factor
    coords_3d[:, 0:2] /= scale_factor

    cls_pred = np.argmax(prob[:, 1:], axis=1) + 1
    scores = np.amax(prob[:, 1:], axis=1)

    aboxes = np.hstack((coords_2d, scores[:, np.newaxis]))
    sorted_inds = (-aboxes[:, 4]).argsort()
    aboxes = aboxes[sorted_inds, :]
    coords_3d = coords_3d[sorted_inds, :]
    cls_pred = cls_pred[sorted_inds]

    n = min(conf.nms_topN_pre, aboxes.shape[0])
    aboxes = aboxes[:n]; coords_3d = coords_3d[:n]; cls_pred = cls_pred[:n]

    if conf.clip_boxes:
        aboxes[:, 0] = np.clip(aboxes[:, 0], 0, imW_orig - 1)
        aboxes[:, 1] = np.clip(aboxes[:, 1], 0, imH_orig - 1)
        aboxes[:, 2] = np.clip(aboxes[:, 2], 0, imW_orig - 1)
        aboxes[:, 3] = np.clip(aboxes[:, 3], 0, imH_orig - 1)

    return np.hstack((aboxes, cls_pred[:, np.newaxis], coords_3d))


imlist = list_files(os.path.join(data_path, conf.dataset_test, 'validation', 'image_2', ''), '*.png')
preprocess = Preprocess([conf.test_scale], conf.image_means, conf.image_stds)

rows = []
t0 = _now()
ndumped_hist = []
for imind, impath in enumerate(imlist):
    im = cv2.imread(impath)
    base_path, name, ext = file_parts(impath)
    sid = id_map[name]
    p2 = read_kitti_cal(os.path.join(data_path, conf.dataset_test, 'validation', 'calib', name + '.txt'))
    p2_inv = np.linalg.inv(p2)

    with torch.no_grad():
        pool = detect_prenms(im, net, conf, preprocess, p2)

    rec_local = []
    for bi in range(pool.shape[0]):
        box = pool[bi]
        score = float(box[4])
        if score < SCORE_MIN:
            break                      # pool is sorted desc
        if conf.lbls[int(box[5] - 1)] != 'Car':
            continue
        if len(rec_local) >= CAP_PER_IMG:
            break
        x1, y1, x2, y2 = box[0:4]
        width = (x2 - x1 + 1); height = (y2 - y1 + 1)
        x3d, y3d, z3d, w3d, h3d, l3d, ry3d = box[6:13]

        coord3d = p2_inv.dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
        ry3d = convertAlpha2Rot(ry3d, coord3d[2], coord3d[0])
        box_2d = np.array([x1, y1, width, height])
        z3d, ry3d, _ = hill_climb(p2, p2_inv, box_2d, x3d, y3d, z3d, w3d, h3d, l3d, ry3d,
                                  step_r_init=0.3 * math.pi, r_lim=0.01)
        coord3d = p2_inv.dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
        alpha = convertRot2Alpha(ry3d, coord3d[2], coord3d[0])
        X3, Y3, Z3 = coord3d[0], coord3d[1], coord3d[2]
        Y3 += h3d / 2

        rec_local.append(dict(
            sid=sid, pred_idx=bi, cls=score, sigma=1.0, log_sigma_raw=0.0, V=score,
            z_pred=float(Z3),
            bbox_h_pix=float(y2 - y1), bbox_w_pix=float(x2 - x1),
            bbox_area_pix=float((y2 - y1) * (x2 - x1)),
            x_3d=float(X3), y_3d=float(Y3), z_3d=float(Z3),
            h_3d=float(h3d), w_3d=float(w3d), l_3d=float(l3d),
            ry=float(ry3d), alpha=float(alpha),
            bbox_x1=float(x1), bbox_y1=float(y1), bbox_x2=float(x2), bbox_y2=float(y2),
            dup_rank=0))

    if rec_local:
        boxes = np.array([[r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]] for r in rec_local])
        vv = np.array([r["V"] for r in rec_local])
        iou = iou2d_matrix(boxes)
        for q in range(len(rec_local)):
            nb = iou[q] >= 0.5; nb[q] = False
            rec_local[q]["dup_rank"] = int(((vv > vv[q]) & nb).sum())
    ndumped_hist.append(len(rec_local))
    rows.extend(rec_local)

    if (imind + 1) % 250 == 0:
        print(f"  {imind+1}/{len(imlist)} rows={len(rows)} "
              f"avg/img={np.mean(ndumped_hist):.1f} {_now()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
df.to_csv(OUT, index=False)
g = df.groupby("sid").size()
print(f"[done] {len(df)} rows, sids={df.sid.nunique()}, "
      f"rows/sid min/med/max={g.min()}/{int(g.median())}/{g.max()}, "
      f"capped@{CAP_PER_IMG}: {sum(1 for n in ndumped_hist if n >= CAP_PER_IMG)} imgs, "
      f"{_now()-t0:.0f}s -> {OUT}", flush=True)

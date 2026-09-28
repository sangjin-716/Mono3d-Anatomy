#!/usr/bin/env python
"""Ported from camera-ready checks/xds_nusc_json_to_oracle_pkl_FIXED.py for the public
release. Computation unchanged (two comments translated to English). Produces the per-camera
oracle pkl for the EPro-PnP-Det, FCOS3D and PGD nuScenes cells, consumed by nusc_oracle.py.

Height-convention-corrected copy of ct_to_oracle_pkl.py: the only code difference is that the
bottom-referenced camera tensor is wrapped with origin=(0.5, 1.0, 0.5) (y unchanged) instead of
origin=(0.5, 0.5, 0.5) (y += h/2). Use it for any submission json whose 'translation' is the box
centre, as the nuScenes format specifies.
nusc submission json (GLOBAL frame) -> per-camera oracle pkl in the exact shape nusc_oracle.py
consumes (pred_instances_3d + eval_ann_info), using the FCOS3D full dump as the per-image
GT/calib template (same val split, same 36114 images, same CameraInstance3DBoxes convention).
Usage: python xds_nusc_json_to_oracle_pkl_FIXED.py <results_nusc.json> <out.pkl> [--gate-only]
Convention gate (mandatory, runs first): push the devkit GT of N_GATE samples through the
SAME global->camera transform and require mean center distance < 0.1 m and dim mismatch
< 0.05 m vs the template's eval_ann_info gt_bboxes_3d. Aborts on failure.

Per-camera assignment: center-in-FOV rule: a global box goes to every camera whose image plane
contains its projected center with z>0 (in practice ~1 cam).
Inputs: <NUSC_ROOT> (nuScenes v1.0-trainval) and the FCOS3D DumpResults pkl
<cache>/crossbench/nusc/preds_fcos3d_full.pkl used as the per-image GT/calib template.
Env: mmdet3d 1.4.0 + nuscenes-devkit (see setup_nuscenes_stage1.sh).
"""
import os, sys, json, pickle, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, cache_dir

sys.path.insert(0, os.path.join(paths.UPSTREAM_ROOT, 'mmdetection3d'))   # optional source checkout
import torch
from mmdet3d.structures import CameraInstance3DBoxes
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

NUSC_ROOT = paths.NUSC_ROOT
TEMPLATE = os.path.join(cache_dir('crossbench', 'nusc'), 'preds_fcos3d_full.pkl')
CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
           'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
N_GATE = 40


def global_to_cam(translation, size, rotation, ego_pose, cs):
    """nuScenes global box -> camera-frame (mmdet3d Camera convention: x right, y down,
    z forward; yaw around -y measured from x-axis; dims (l, h, w) order per nuScenes mono3d)."""
    box = Box(translation, size, Quaternion(rotation))
    box.translate(-np.array(ego_pose['translation']))
    box.rotate(Quaternion(ego_pose['rotation']).inverse)
    box.translate(-np.array(cs['translation']))
    box.rotate(Quaternion(cs['rotation']).inverse)
    # now in camera frame. mmdet3d camera box: [x, y, z, l, h, w? -> see gate] + yaw
    c = box.center  # (x, y, z) camera frame
    w, l, h = box.wlh
    # yaw: rotation of box x-axis in camera x-z plane, mmdet3d convention yaw = -atan2(z, x)
    v = box.rotation_matrix @ np.array([1.0, 0.0, 0.0])
    yaw = -np.arctan2(v[2], v[0])
    return np.array([c[0], c[1] + h / 2.0, c[2], l, h, w, yaw], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results'); ap.add_argument('out', nargs='?', default='')
    ap.add_argument('--gate-only', action='store_true')
    args = ap.parse_args()

    print('[load] template pkl ...', flush=True)
    data = pickle.load(open(TEMPLATE, 'rb'))
    print(f'[load] {len(data)} images; devkit ...', flush=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=False)

    # index template images by sample_data token (from img_path) -> need token per image.
    # template entries carry 'sample_idx'/'img_path'; map via filename -> sample_data
    fname2sd = {}
    for sd in nusc.sample_data:
        if sd['sensor_modality'] == 'camera' and sd['is_key_frame']:
            fname2sd[sd['filename'].split('/')[-1]] = sd
    tmpl_sd = []
    for s in data:
        img = s.get('img_path') or s.get('lidar_path') or ''
        fn = str(img).split('/')[-1]
        sd = fname2sd.get(fn)
        assert sd is not None, f'sample_data not found for {fn[:60]}'
        tmpl_sd.append(sd)

    # ---- convention gate on N_GATE images with >=1 GT ----
    print('[gate] convention check ...', flush=True)
    checked = 0; cd = []; dd = []
    for s, sd in zip(data, tmpl_sd):
        gt = s['eval_ann_info']
        n = len(gt['gt_labels_3d'])
        if n == 0:
            continue
        sample = nusc.get('sample', sd['sample_token'])
        ego = nusc.get('ego_pose', sd['ego_pose_token'])
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        # devkit GT boxes visible in this camera
        _, boxes, _ = nusc.get_sample_data(sd['token'])
        got = []
        for b in boxes:
            # b is already in camera frame from get_sample_data; rebuild from global instead
            ann = nusc.get('sample_annotation', b.token)
            got.append(global_to_cam(ann['translation'], ann['size'], ann['rotation'], ego, cs))
        if not got:
            continue
        got = np.stack(got)
        ref = gt['gt_bboxes_3d'].tensor.cpu().numpy() if hasattr(gt['gt_bboxes_3d'], 'tensor') \
            else np.asarray(gt['gt_bboxes_3d'])
        # match by nearest center (sets may differ by visibility filter) — compare matched only
        for r in ref:
            d = np.linalg.norm(got[:, :3] - r[:3], axis=1)
            j = int(np.argmin(d))
            if d[j] < 0.5:
                cd.append(d[j]); dd.append(np.abs(got[j, 3:6] - r[3:6]).max())
        checked += 1
        if checked >= N_GATE:
            break
    cd, dd = np.array(cd), np.array(dd)
    print(f'[gate] matched {len(cd)} GTs over {checked} imgs | center dist mean={cd.mean():.4f} '
          f'p95={np.percentile(cd,95):.4f} | dim maxerr mean={dd.mean():.4f}', flush=True)
    assert len(cd) >= 50 and cd.mean() < 0.1 and dd.mean() < 0.05, 'CONVENTION GATE FAIL'
    print('[gate] PASS', flush=True)
    if args.gate_only:
        return
    # memory: the template prediction tensors are not needed -- keep only the GT and free the rest
    import gc
    for i in range(len(data)):
        data[i] = {'eval_ann_info': data[i]['eval_ann_info']}
    gc.collect()

    # ---- convert CT predictions ----
    print('[load] CT results ...', flush=True)
    res = json.load(open(args.results))['results']
    cls2id = {c: i for i, c in enumerate(CLASSES)}
    # remaining uses per sample token (freed from res once all 6 cameras have consumed it)
    remaining = {}
    for sd in tmpl_sd:
        remaining[sd['sample_token']] = remaining.get(sd['sample_token'], 0) + 1

    n_boxes = 0
    out = []
    for i, (s, sd) in enumerate(zip(data, tmpl_sd)):
        ego = nusc.get('ego_pose', sd['ego_pose_token'])
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        K = np.array(cs['camera_intrinsic'])
        W, H = sd['width'], sd['height']
        arr = []; scores = []; labels = []
        for det in res.get(sd['sample_token'], []):
            if det['detection_name'] not in cls2id:
                continue
            b = global_to_cam(det['translation'], det['size'], det['rotation'], ego, cs)
            # center-in-FOV test (project pre-bottom-shift center)
            c = np.array([b[0], b[1] - b[4] / 2.0, b[2]])
            if c[2] <= 0.1:
                continue
            uv = K @ c
            u, v = uv[0] / uv[2], uv[1] / uv[2]
            if not (0 <= u < W and 0 <= v < H):
                continue
            arr.append(b); scores.append(det['detection_score']); labels.append(cls2id[det['detection_name']])
        n_boxes += len(arr)
        remaining[sd['sample_token']] -= 1
        if remaining[sd['sample_token']] == 0:
            res.pop(sd['sample_token'], None)
        pred = dict(
            bboxes_3d=CameraInstance3DBoxes(
                torch.tensor(np.array(arr, dtype=np.float32).reshape(-1, 7)), box_dim=7,
                origin=(0.5, 1.0, 0.5)),
            scores_3d=torch.tensor(np.array(scores, dtype=np.float32)),
            labels_3d=torch.tensor(np.array(labels, dtype=np.int64)))
        out.append(dict(pred_instances_3d=pred, eval_ann_info=s['eval_ann_info']))
        if i % 5000 == 0:
            print(f'  {i}/{len(data)} ({n_boxes} boxes)', flush=True)
    print(f'[done] {n_boxes} boxes over {len(out)} images', flush=True)
    pickle.dump(out, open(args.out, 'wb'))
    print('saved:', args.out, flush=True)


if __name__ == '__main__':
    main()

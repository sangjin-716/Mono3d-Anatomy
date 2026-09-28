#!/usr/bin/env python
"""Writes a nuScenes submission json from a per-camera DumpResults pkl.

NOT used for any number in the paper. The FCOS3D and PGD reproduction-gate values and oracle
cells come from the official mmdet3d formatter (nusc_official_reformat.py, reformat_official.py).
This hand-written converter concatenates the six cameras of a sample without cross-view 3D NMS
or the per-class range filter, forces velocity to [0, 0] and uses a fixed default attribute, so
it under-reproduces the published FCOS3D/PGD mAP/NDS. It is kept because pgd_bisect_eval.py
uses its output as the first rung (V0) of the formatter ladder.

campkl_to_nusc_json.py -- per-camera oracle pkl (mmdet3d camera-frame preds) -> official
nuScenes detection submission json (global frame).

Roundtrip gate (mandatory): template GT tensors -> global via THIS transform must match the
devkit sample_annotation translation/size to numerical zero (inverse of the validated
global->cam gate). Aborts on failure.

Per-sample merge: 6 cameras' boxes are concatenated per sample_token; top-500 by score
(official submission cap). Class names mapped back from label ids.
Usage: python campkl_to_nusc_json.py <preds.pkl> <out.json> [--gate-only]
Env: mmdet3d 1.4.0 + nuscenes-devkit."""
import os, sys, json, pickle, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths

sys.path.insert(0, os.path.join(paths.UPSTREAM_ROOT, 'mmdetection3d'))   # optional source checkout
from nuscenes import NuScenes
from pyquaternion import Quaternion

NUSC_ROOT = paths.NUSC_ROOT
CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
           'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
DEFAULT_ATTR = {'car': 'vehicle.parked', 'truck': 'vehicle.parked', 'trailer': 'vehicle.parked',
                'bus': 'vehicle.parked', 'construction_vehicle': 'vehicle.parked',
                'bicycle': 'cycle.without_rider', 'motorcycle': 'cycle.without_rider',
                'pedestrian': 'pedestrian.standing', 'traffic_cone': '', 'barrier': ''}
N_GATE = 40


def cam_to_global(row, ego, cs):
    """mmdet3d camera tensor row [x, y_bottom, z, l, h, w, yaw] -> global (translation,
    size wlh, rotation quat). Exact inverse of ct_to_oracle_pkl.global_to_cam."""
    x, yb, z, l, h, w, yaw = [float(v) for v in row]
    c_cam = np.array([x, yb - h / 2.0, z])
    # camera-frame orientation: x-axis direction (cos(-yaw), 0, sin(-yaw)) since yaw = -atan2(v_z, v_x)
    R_cs = Quaternion(cs['rotation']).rotation_matrix
    R_ego = Quaternion(ego['rotation']).rotation_matrix
    c_g = R_ego @ (R_cs @ c_cam + np.array(cs['translation'])) + np.array(ego['translation'])
    vx_cam = np.array([np.cos(-yaw), 0.0, np.sin(-yaw)])
    vx_g = R_ego @ (R_cs @ vx_cam)
    yaw_g = np.arctan2(vx_g[1], vx_g[0])
    q = Quaternion(axis=[0, 0, 1], radians=yaw_g)
    return c_g.tolist(), [w, l, h], [q.w, q.x, q.y, q.z]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl'); ap.add_argument('out', nargs='?', default='')
    ap.add_argument('--gate-only', action='store_true')
    args = ap.parse_args()

    print('[load] pkl ...', flush=True)
    data = pickle.load(open(args.pkl, 'rb'))
    print(f'[load] {len(data)} images; devkit ...', flush=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=False)
    fname2sd = {}
    for sd in nusc.sample_data:
        if sd['sensor_modality'] == 'camera' and sd['is_key_frame']:
            fname2sd[sd['filename'].split('/')[-1]] = sd
    tmpl_sd = []
    for s in data:
        fn = str(s.get('img_path', '')).split('/')[-1]
        sd = fname2sd.get(fn)
        assert sd is not None, f'sample_data missing for {fn[:60]}'
        tmpl_sd.append(sd)

    # ---- roundtrip gate: GT cam tensor -> global vs devkit annotation ----
    print('[gate] roundtrip check ...', flush=True)
    checked = 0; errs = []; serrs = []
    for s, sd in zip(data, tmpl_sd):
        gt = s['eval_ann_info']
        if len(gt['gt_labels_3d']) == 0:
            continue
        ego = nusc.get('ego_pose', sd['ego_pose_token'])
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ref = gt['gt_bboxes_3d'].tensor.cpu().numpy() if hasattr(gt['gt_bboxes_3d'], 'tensor') \
            else np.asarray(gt['gt_bboxes_3d'])
        anns = [nusc.get('sample_annotation', b.token)
                for b in nusc.get_sample_data(sd['token'])[1]]
        if not anns:
            continue
        gtr = np.array([a['translation'] for a in anns])
        gsz = np.array([a['size'] for a in anns])
        for r in ref:
            tr, sz, _ = cam_to_global(r[:7], ego, cs)
            d = np.linalg.norm(gtr - np.array(tr), axis=1)
            j = int(np.argmin(d))
            if d[j] < 0.5:
                errs.append(d[j]); serrs.append(np.abs(gsz[j] - np.array(sz)).max())
        checked += 1
        if checked >= N_GATE:
            break
    errs, serrs = np.array(errs), np.array(serrs)
    print(f'[gate] matched {len(errs)} | center mean={errs.mean():.4f} size maxerr mean={serrs.mean():.4f}', flush=True)
    assert len(errs) >= 50 and errs.mean() < 0.05 and serrs.mean() < 0.05, 'ROUNDTRIP GATE FAIL'
    print('[gate] PASS', flush=True)
    if args.gate_only:
        return

    results = {}
    for i, (s, sd) in enumerate(zip(data, tmpl_sd)):
        ego = nusc.get('ego_pose', sd['ego_pose_token'])
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        pred = s['pred_instances_3d']
        boxes = pred['bboxes_3d'].tensor.cpu().numpy() if hasattr(pred['bboxes_3d'], 'tensor') \
            else np.asarray(pred['bboxes_3d'])
        scores = np.asarray(pred['scores_3d'].cpu() if hasattr(pred['scores_3d'], 'cpu') else pred['scores_3d'])
        labels = np.asarray(pred['labels_3d'].cpu() if hasattr(pred['labels_3d'], 'cpu') else pred['labels_3d'])
        lst = results.setdefault(sd['sample_token'], [])
        for b, sc, lb in zip(boxes, scores, labels):
            tr, sz, q = cam_to_global(b[:7], ego, cs)
            name = CLASSES[int(lb)]
            lst.append(dict(sample_token=sd['sample_token'], translation=tr, size=sz,
                            rotation=q, velocity=[0.0, 0.0], detection_name=name,
                            detection_score=float(sc), attribute_name=DEFAULT_ATTR[name]))
        if i % 5000 == 0:
            print(f'  {i}/{len(data)}', flush=True)
    # official cap: 500 boxes per sample
    for k in results:
        results[k] = sorted(results[k], key=lambda d: -d['detection_score'])[:500]
    # ensure every val sample key exists
    from nuscenes.utils import splits
    val_scenes = set(splits.val)
    for sample in nusc.sample:
        if nusc.get('scene', sample['scene_token'])['name'] in val_scenes:
            results.setdefault(sample['token'], [])
    sub = dict(meta=dict(use_camera=True, use_lidar=False, use_radar=False,
                         use_map=False, use_external=False), results=results)
    json.dump(sub, open(args.out, 'w'))
    print('saved:', args.out, f'({len(results)} samples)', flush=True)


if __name__ == '__main__':
    main()

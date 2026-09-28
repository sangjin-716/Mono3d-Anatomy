#!/usr/bin/env python
"""Produces the FCOS3D nuScenes reproduction-gate values (mAP / NDS and TP errors, printed as
[RESULT] lines) and the submission json
<out_root>/official_reformat_<tag>/pred_instances_3d/results_nusc.json used by the oracle chain.

Drives the OFFICIAL mmdet3d formatting path
(mmdet3d.evaluation.metrics.nuscenes_metric.NuScenesMetric._format_camera_bbox)
directly on the per-camera prediction pkl that DumpResults already wrote (no re-inference),
instead of the hand-written campkl_to_nusc_json.py.

What the official path adds that campkl_to_nusc_json.py drops:
  * predicted velocity   (bboxes_3d.tensor[:, 7:9]  -> box.velocity)
  * predicted attribute  (pred_instances_3d['attr_labels'] -> get_attr_name)
  * exact quaternion     (q2*q1 in cam frame, then rotated by cam2ego/ego2global)
  * per-class range filter in cam_nusc_box_to_global / global_nusc_box_to_cam
  * CROSS-VIEW rotated 3D NMS over the 6 cameras of a frame (box3d_multiclass_nms,
    nms_thr=0.05, score_thr=0.01, max_per_frame=500)

Usage: python nusc_official_reformat.py <preds.pkl> <tag> <out_root>
Env:   mmdet3d 1.4.0, mmdet 3.2.0, torch 2.1.2 (see setup_nuscenes_stage1.sh)
"""
import os
import os.path as osp
import pickle
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, cache_dir

DATA_ROOT = paths.NUSC_ROOT
ANN_FILE = osp.join(DATA_ROOT, 'nuscenes_infos_val.pkl')
CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]


def main():
    pkl_path, tag, out_root = sys.argv[1], sys.argv[2], sys.argv[3]
    out_dir = osp.join(out_root, f'official_reformat_{tag}')
    os.makedirs(out_dir, exist_ok=True)

    from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric
    import mmdet3d
    print(f'[env] mmdet3d {mmdet3d.__version__} from {mmdet3d.__file__}', flush=True)
    print(f'[env] torch {torch.__version__} cuda={torch.cuda.is_available()}', flush=True)

    t0 = time.time()
    print(f'[load] {pkl_path}', flush=True)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f'[load] {len(data)} images in {time.time() - t0:.0f}s', flush=True)

    # Build exactly what NuScenesMetric.process() would have put in self.results,
    # dropping everything else so the peak RSS stays bounded.
    results = []
    for i in range(len(data)):
        s = data[i]
        pred_3d = {k: v.to('cpu') for k, v in s['pred_instances_3d'].items()}
        results.append(dict(pred_instances_3d=pred_3d, sample_idx=s['sample_idx']))
        data[i] = None
    del data
    torch.cuda.empty_cache()
    print(f'[prep] {len(results)} results, keys={sorted(results[0]["pred_instances_3d"].keys())}',
          flush=True)
    print(f'[prep] box_dim={results[0]["pred_instances_3d"]["bboxes_3d"].box_dim}', flush=True)

    metric = NuScenesMetric(
        data_root=DATA_ROOT,
        ann_file=ANN_FILE,
        metric='bbox',
        modality=dict(use_camera=True, use_lidar=False),
        jsonfile_prefix=out_dir,
    )
    metric.dataset_meta = dict(classes=CLASSES, version='v1.0-trainval')

    t1 = time.time()
    md = metric.compute_metrics(results)
    print(f'[eval] done in {time.time() - t1:.0f}s', flush=True)

    pref = 'pred_instances_3d_NuScenes'
    print('[RESULT] %s  mAP=%.4f  NDS=%.4f' % (tag, md[f'{pref}/mAP'], md[f'{pref}/NDS']),
          flush=True)
    for k in ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
        kk = f'{pref}/{k}'
        if kk in md:
            print('[RESULT] %s %s=%.4f' % (tag, k, md[kk]), flush=True)
    for c in CLASSES:
        keys = [f'{pref}/{c}_AP_dist_{d}' for d in ['0.5', '1.0', '2.0', '4.0']]
        vals = [md[k] for k in keys if k in md]
        if vals:
            print('[AP] %-22s mean=%.4f  %s' % (c, sum(vals) / len(vals),
                                                ' '.join('%.4f' % v for v in vals)), flush=True)
    print('DONE_' + tag, flush=True)


if __name__ == '__main__':
    main()

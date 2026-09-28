#!/usr/bin/env python
"""Produces the PGD submission json
<out_dir>/pred_instances_3d/results_nusc.json (official mmdet3d formatter), which
eval_official.py scores for the PGD reproduction gate and the oracle chain converts.

Uses the OFFICIAL mmdet3d formatting path instead of the hand-written campkl_to_nusc_json.py:
mmdet3d.evaluation.metrics.nuscenes_metric.NuScenesMetric.format_results
-> _format_camera_bbox (output_to_nusc_box / cam_nusc_box_to_global /
   global_nusc_box_to_cam / box3d_multiclass_nms across the 6 views / get_attr_name).

REFORMAT ONLY -- no re-inference.  Input is the per-image DumpResults pkl
(<cache>/crossbench/nusc/preds_<det>_full.pkl, see configs/); nothing is overwritten.

Usage:
  python reformat_official.py {pgd|fcos3d} <out_dir> [--limit N_IMAGES]
Env: mmdet3d 1.4.0 (see setup_nuscenes_stage1.sh)
"""
import argparse
import os
import os.path as osp
import pickle
import sys
import time

import torch
from mmengine import load

from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, cache_dir

NUSC_ROOT = paths.NUSC_ROOT
ANN = osp.join(NUSC_ROOT, 'nuscenes_infos_val.pkl')
_W = cache_dir('crossbench', 'nusc')
PKL = {
    'pgd': osp.join(_W, 'preds_pgd_full.pkl'),
    'fcos3d': osp.join(_W, 'preds_fcos3d_full.pkl'),
    'smoke': osp.join(_W, 'smoke_fcos3d.pkl'),
}
CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
           'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
CAMERA_TYPES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('det', choices=list(PKL))
    ap.add_argument('out_dir')
    ap.add_argument('--limit', type=int, default=0,
                    help='only format the first N images (must be a multiple of 6)')
    args = ap.parse_args()

    t0 = time.time()
    print(f'[load] {PKL[args.det]}', flush=True)
    with open(PKL[args.det], 'rb') as f:
        data = pickle.load(f)
    print(f'[load] {len(data)} images in {time.time()-t0:.0f}s', flush=True)

    # ---- ordering gate: sample_idx must be 0..N-1 and camera order must be the
    # one _format_camera_bbox assumes (sample_idx % 6 -> CAMERA_TYPES[i]).
    nbad = 0
    for i, s in enumerate(data):
        if s['sample_idx'] != i:
            print(f'[gate] FAIL sample_idx {s["sample_idx"]} != {i}')
            nbad += 1
        if CAMERA_TYPES[i % 6] not in s['img_path']:
            print(f'[gate] FAIL cam {s["img_path"]} at idx {i} '
                  f'(expect {CAMERA_TYPES[i % 6]})')
            nbad += 1
        if nbad > 5:
            sys.exit('ORDERING GATE FAIL')
    assert nbad == 0, 'ORDERING GATE FAIL'
    assert len(data) % 6 == 0, f'{len(data)} images not a multiple of 6'
    print(f'[gate] ordering PASS ({len(data)} imgs = {len(data)//6} frames)', flush=True)

    if args.limit:
        assert args.limit % 6 == 0
        data = data[:args.limit]

    # ---- field census
    p0 = data[0]['pred_instances_3d']
    print('[fields] pred_instances_3d keys:', sorted(p0.keys()), flush=True)
    print('[fields] bboxes_3d box_dim:', p0['bboxes_3d'].box_dim,
          '(dims 7,8 = camera-frame velocity vx,vz)', flush=True)
    nbox = sum(len(s['pred_instances_3d']['scores_3d']) for s in data)
    print(f'[fields] total raw boxes = {nbox} '
          f'({nbox/len(data):.1f} per image, {6*nbox/len(data):.1f} per frame)', flush=True)

    results = []
    for s in data:
        p = s['pred_instances_3d']
        results.append(dict(
            sample_idx=s['sample_idx'],
            pred_instances_3d=dict(
                bboxes_3d=p['bboxes_3d'].to('cpu'),
                scores_3d=p['scores_3d'].cpu(),
                labels_3d=p['labels_3d'].cpu(),
                attr_labels=p['attr_labels'].cpu())))
    del data
    torch.cuda.empty_cache()

    os.makedirs(args.out_dir, exist_ok=True)
    metric = NuScenesMetric(
        data_root=NUSC_ROOT,
        ann_file=ANN,
        metric='bbox',
        modality=dict(use_camera=True, use_lidar=False, use_radar=False,
                      use_map=False, use_external=False))
    metric.dataset_meta = dict(classes=CLASSES, version='v1.0-trainval')
    metric.data_infos = load(ANN)['data_list']
    print(f'[infos] {len(metric.data_infos)} frames in ann_file', flush=True)

    t1 = time.time()
    result_dict, _ = metric.format_results(results, CLASSES, args.out_dir)
    print(f'[format] done in {time.time()-t1:.0f}s -> {result_dict}', flush=True)
    print('[format] json:', result_dict['pred_instances_3d'], flush=True)


if __name__ == '__main__':
    main()

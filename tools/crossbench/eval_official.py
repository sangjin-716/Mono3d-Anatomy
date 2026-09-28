#!/usr/bin/env python
"""Ported from camera-ready checks/xds_repair/eval_official.py for the public release.
Computation unchanged. Produces <out_dir>/metrics_summary.json and [RESULT] lines (mAP, NDS,
TP errors, per-class mean distance AP) for the PGD reproduction gate.

eval_official.py -- official nuScenes detection_cvpr_2019 eval on a submission json.
Usage: python eval_official.py <results.json> <out_dir>
"""
import json
import os
import sys

from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.config import config_factory

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, cache_dir

NUSC_ROOT = paths.NUSC_ROOT
res_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=False)
cfg = config_factory('detection_cvpr_2019')
ev = DetectionEval(nusc, config=cfg, result_path=res_path, eval_set='val',
                   output_dir=out_dir, verbose=True)
m = ev.main(plot_examples=0, render_curves=False)
print('[RESULT] %s  mAP=%.4f NDS=%.4f' % (res_path, m['mean_ap'], m['nd_score']), flush=True)
print('[RESULT] tp_errors', json.dumps(m['tp_errors']), flush=True)
print('[RESULT] mean_dist_aps', json.dumps(m['mean_dist_aps']), flush=True)

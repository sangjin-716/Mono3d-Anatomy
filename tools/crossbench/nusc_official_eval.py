#!/usr/bin/env python
"""Writes the official nuScenes devkit metrics (metrics_summary.json, mAP / NDS) under
crossbench/nusc/official_eval_<tag>/ in the re-run output directory.

nusc_official_eval.py -- reproduction gate: official nuScenes detection eval
(detection_cvpr_2019, val split) on a results json (nuScenes submission format); the result is
compared with the published number.
Used for the CenterNet (CenterTrack e140) gate: the input is CenterTrack's own
results_nuscenes_det.json.
Usage: python nusc_official_eval.py <results.json> <tag>"""
import sys, json, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, out_path

from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.config import config_factory

NUSC_ROOT = paths.NUSC_ROOT

res_path, tag = sys.argv[1], sys.argv[2]
out_dir = os.path.dirname(out_path(os.path.join('crossbench', 'nusc', f'official_eval_{tag}', 'metrics_summary.json')))
os.makedirs(out_dir, exist_ok=True)

nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=False)
cfg = config_factory('detection_cvpr_2019')
ev = DetectionEval(nusc, config=cfg, result_path=res_path, eval_set='val',
                   output_dir=out_dir, verbose=True)
metrics = ev.main(plot_examples=0, render_curves=False)
print('[gate]', tag, 'mAP=%.4f NDS=%.4f' % (metrics['mean_ap'], metrics['nd_score']), flush=True)

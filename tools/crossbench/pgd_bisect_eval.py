#!/usr/bin/env python
"""Writes crossbench/nusc/xdsfix_pgd/bisect_rows.json and eval_<variant>/ devkit outputs in the
re-run output directory, read by nds_residual.py.

pgd_bisect_eval.py -- bisection of the PGD nuScenes gap between the hand-written converter
(campkl_to_nusc_json.py) and the official mmdet3d formatter.

Runs the OFFICIAL nuScenes devkit eval (detection_cvpr_2019, val) on a ladder of
submission jsons that differ by exactly one information channel, so the gap of the
hand-written converter is DECOMPOSED, not just reported.

Ladder (all evaluated with one devkit instance, identical code path):
  V0  OLD  : hand-written converter json  (no cross-view NMS, no class-range filter,
             top-500 on the raw pool, velocity=[0,0], fixed DEFAULT_ATTR)
  V1  GEO  : official-formatter json, but velocity zeroed AND attributes overwritten with
             the converter's DEFAULT_ATTR  -> official geometry-selection ONLY
  V2  +VEL : official-formatter json, attributes overwritten with DEFAULT_ATTR,
             official velocity kept
  V3  +ATT : official-formatter json, velocity zeroed, official attributes kept
  V4  FULL : official-formatter json as produced  (the PGD reproduction-gate cell)

Reads:  V0 json (<cache>/crossbench/nusc/pgd_global.json, from run_ib_gates.sh) and V4 json
        (<cache>/crossbench/nusc/xds_repair/official_pgd/pred_instances_3d/results_nusc.json,
        from run_repair.sh); both are left untouched.
Writes: variant jsons to a scratch dir, deleted immediately after each eval.
Env: mmdet3d 1.4.0 + nuscenes-devkit.
"""
import json
import os
import os.path as osp
import shutil
import sys
import time

from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, cache_dir, out_path

NUSC_ROOT = paths.NUSC_ROOT
_W = cache_dir('crossbench', 'nusc')
OLD_JSON = osp.join(_W, 'pgd_global.json')
NEW_JSON = osp.join(_W, 'xds_repair', 'official_pgd', 'pred_instances_3d', 'results_nusc.json')
OUT = osp.dirname(out_path(osp.join('crossbench', 'nusc', 'xdsfix_pgd', 'bisect_rows.json')))
SCRATCH = osp.join(OUT, 'scratch')

# verbatim from campkl_to_nusc_json.py
DEFAULT_ATTR = {'car': 'vehicle.parked', 'truck': 'vehicle.parked',
                'trailer': 'vehicle.parked', 'bus': 'vehicle.parked',
                'construction_vehicle': 'vehicle.parked',
                'bicycle': 'cycle.without_rider', 'motorcycle': 'cycle.without_rider',
                'pedestrian': 'pedestrian.standing', 'traffic_cone': '', 'barrier': ''}

os.makedirs(SCRATCH, exist_ok=True)
print('[devkit] loading v1.0-trainval ...', flush=True)
t0 = time.time()
nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=False)
cfg = config_factory('detection_cvpr_2019')
print(f'[devkit] loaded in {time.time()-t0:.0f}s', flush=True)


def run_eval(name, path):
    od = osp.join(OUT, 'eval_' + name)
    os.makedirs(od, exist_ok=True)
    print(f'\n{"="*70}\n[EVAL] {name}  <- {path}\n{"="*70}', flush=True)
    t = time.time()
    ev = DetectionEval(nusc, config=cfg, result_path=path, eval_set='val',
                       output_dir=od, verbose=True)
    m = ev.main(plot_examples=0, render_curves=False)
    print(f'[RESULT] {name}  mAP={m["mean_ap"]:.4f}  NDS={m["nd_score"]:.4f}  '
          f'({time.time()-t:.0f}s)', flush=True)
    print(f'[RESULT] {name}  tp_errors {json.dumps(m["tp_errors"])}', flush=True)
    return dict(name=name, mAP=m['mean_ap'], NDS=m['nd_score'], tp=m['tp_errors'])


def make_variant(base, tag, zero_vel, default_attr):
    """Derive a variant json from `base` by switching off one/both channels."""
    print(f'[variant] building {tag} (zero_vel={zero_vel}, default_attr={default_attr})',
          flush=True)
    d = json.load(open(base))
    n = 0
    for boxes in d['results'].values():
        for b in boxes:
            if zero_vel:
                b['velocity'] = [0.0, 0.0]
            if default_attr:
                b['attribute_name'] = DEFAULT_ATTR[b['detection_name']]
            n += 1
    p = osp.join(SCRATCH, f'{tag}.json')
    json.dump(d, open(p, 'w'))
    del d
    print(f'[variant] {tag}: {n} boxes -> {p} '
          f'({osp.getsize(p)/1e6:.0f} MB)', flush=True)
    return p


rows = []
# --- V0: our old converter json, as-is
rows.append(run_eval('V0_OLD_ours', OLD_JSON))

# --- V1/V2/V3 derived from the official-formatter json
for tag, zv, da in [('V1_GEO_only', True, True),
                    ('V2_GEO_VEL', False, True),
                    ('V3_GEO_ATT', True, False)]:
    p = make_variant(NEW_JSON, tag, zv, da)
    rows.append(run_eval(tag, p))
    os.remove(p)
    print(f'[cleanup] removed {p}', flush=True)

# --- V4: the repaired json as produced
rows.append(run_eval('V4_FULL_official', NEW_JSON))

shutil.rmtree(SCRATCH, ignore_errors=True)

print('\n\n' + '=' * 78)
print('PGD nuScenes FORMATTER BISECTION LADDER  (official devkit, val, 6019 samples)')
print('=' * 78)
print(f'{"variant":<20}{"mAP":>8}{"NDS":>8}{"mATE":>8}{"mASE":>8}{"mAOE":>8}'
      f'{"mAVE":>8}{"mAAE":>8}')
for r in rows:
    t = r['tp']
    print(f'{r["name"]:<20}{r["mAP"]*100:8.2f}{r["NDS"]*100:8.2f}'
          f'{t["trans_err"]:8.3f}{t["scale_err"]:8.3f}{t["orient_err"]:8.3f}'
          f'{t["vel_err"]:8.3f}{t["attr_err"]:8.3f}')
print('-' * 78)
by = {r['name']: r for r in rows}
def d(a, b, k):
    return (by[a][k] - by[b][k]) * 100
print('DECOMPOSITION of the repair (percentage points):')
print(f'  step 1  geometry selection (cross-view NMS + class-range filter + cap order)')
print(f'          V0_OLD -> V1_GEO_only        dmAP {d("V1_GEO_only","V0_OLD_ours","mAP"):+6.2f}   '
      f'dNDS {d("V1_GEO_only","V0_OLD_ours","NDS"):+6.2f}')
print(f'  step 2a velocity        V1 -> V2     dmAP {d("V2_GEO_VEL","V1_GEO_only","mAP"):+6.2f}   '
      f'dNDS {d("V2_GEO_VEL","V1_GEO_only","NDS"):+6.2f}')
print(f'  step 2b attribute       V1 -> V3     dmAP {d("V3_GEO_ATT","V1_GEO_only","mAP"):+6.2f}   '
      f'dNDS {d("V3_GEO_ATT","V1_GEO_only","NDS"):+6.2f}')
print(f'  step 3  both            V1 -> V4     dmAP {d("V4_FULL_official","V1_GEO_only","mAP"):+6.2f}   '
      f'dNDS {d("V4_FULL_official","V1_GEO_only","NDS"):+6.2f}')
print(f'  TOTAL                   V0 -> V4     dmAP {d("V4_FULL_official","V0_OLD_ours","mAP"):+6.2f}   '
      f'dNDS {d("V4_FULL_official","V0_OLD_ours","NDS"):+6.2f}')
print('-' * 78)
print('PUBLISHED (mmdet3d configs/pgd/metafile.yml, row')
print('  pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d_finetune):  mAP 35.8  NDS 42.5')
print(f'  repaired V4 delta vs published:  dmAP '
      f'{by["V4_FULL_official"]["mAP"]*100-35.8:+.2f}   '
      f'dNDS {by["V4_FULL_official"]["NDS"]*100-42.5:+.2f}')
print('=' * 78)
json.dump(rows, open(osp.join(OUT, 'bisect_rows.json'), 'w'), indent=2)

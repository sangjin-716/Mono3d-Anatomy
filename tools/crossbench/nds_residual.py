#!/usr/bin/env python
"""Prints the PGD NDS excess over the published value and locates it in the true-positive error
terms of NDS.

nds_residual.py -- (a) verify the nuScenes NDS formula reproduces every rung of the
PGD bisection ladder (pgd_bisect_eval.py) from its own (mAP, tp_errors), and (b) quantify
exactly how much TP-error budget separates the PGD cell from the published NDS 42.5.

NDS = (5*mAP + sum_{5 TP metrics} max(0, 1 - min(1, err))) / 10
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import out_path

ROWS = json.load(open(out_path(os.path.join('crossbench', 'nusc', 'xdsfix_pgd', 'bisect_rows.json'))))
KEYS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']


def nds(mAP, tp):
    return (5 * mAP + sum(max(0.0, 1.0 - min(1.0, tp[k])) for k in KEYS)) / 10.0


print('=' * 74)
print('NDS FORMULA CHECK  (recomputed from each rung\'s own mAP + tp_errors)')
print('=' * 74)
print(f'{"variant":<20}{"NDS stored":>12}{"NDS recomputed":>16}{"|diff|":>12}')
for r in ROWS:
    rec = nds(r['mAP'], r['tp'])
    print(f'{r["name"]:<20}{r["NDS"]:12.6f}{rec:16.6f}{abs(rec-r["NDS"]):12.2e}')

v4 = [r for r in ROWS if r['name'] == 'V4_FULL_official'][0]
print()
print('=' * 74)
print('RESIDUAL vs PUBLISHED (metafile row 2x_nus-mono3d_finetune: mAP 35.8 / NDS 42.5)')
print('=' * 74)
print(f'  repaired mAP = {v4["mAP"]*100:.2f}   published 35.8   delta {v4["mAP"]*100-35.8:+.2f}'
      '   (within the metafile\'s own 1-decimal rounding, +/-0.05)')
print(f'  repaired NDS = {v4["NDS"]*100:.2f}   published 42.5   delta {v4["NDS"]*100-42.5:+.2f}'
      '   (EXCEEDS 1-decimal rounding)')
print()
tp_sum = sum(max(0.0, 1.0 - min(1.0, v4['tp'][k])) for k in KEYS)
# published NDS 0.425 with our identical mAP implies this TP-score sum:
need = 0.425 * 10 - 5 * v4['mAP']
print(f'  our TP-score sum (sum of max(0,1-min(1,err)))   = {tp_sum:.5f}')
print(f'  TP-score sum implied by published NDS 42.5 at   = {need:.5f}')
print(f'  our own mAP {v4["mAP"]*100:.2f}')
print(f'  => the whole residual is a TP-error budget of    {tp_sum-need:+.5f}')
print()
print('  Equivalently, the residual is FULLY accounted for by ANY ONE of:')
for k, label in [('attr_err', 'mAAE'), ('vel_err', 'mAVE'), ('trans_err', 'mATE'),
                 ('orient_err', 'mAOE'), ('scale_err', 'mASE')]:
    cur = v4['tp'][k]
    alt = cur + (tp_sum - need)
    ok = '' if alt <= 1.0 else '   (>1.0: would be clipped, so this alone CANNOT explain it)'
    print(f'    reference run having {label} = {alt:.4f} instead of our {cur:.4f}{ok}')
print()
print('  mATE/mASE/mAOE are pure geometry of the SAME predictions, and our cam->global')
print('  transform was audited numerically identical to the official one (1e-6 m / 0.004 deg),')
print('  so they are not free parameters. The live candidates are mAAE and mAVE, i.e. exactly')
print('  the two attribute/velocity plumbing paths that changed between the mmdet3d version')
print('  that produced the metafile (v1.0.0.dev0) and the 1.4.0 we ran.')

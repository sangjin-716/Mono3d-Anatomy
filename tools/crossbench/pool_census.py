"""Ported from camera-ready checks/xdsfix_fcos3d_oracle/pool_census.py for the public release.
Computation unchanged. Produces the car-prediction census of a converted oracle pkl (console); in the paper it gives
the FCOS3D boxes per image (pred_car per image).

pool_census.py — characterise the fixed detection pool an oracle cell was computed on.
Reports, for the Car class (label 0): total pred boxes, car pred boxes, per-image rate,
score distribution, and camera-frame depth (z) histogram. GT side is identical across pkls
(same eval_ann_info template) and is printed once as a reference.
Usage: python pool_census.py <oracle.pkl> <tag>
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths
sys.path.insert(0, os.path.join(paths.UPSTREAM_ROOT, 'mmdetection3d'))   # optional source checkout

path, tag = sys.argv[1], sys.argv[2]
d = pickle.load(open(path, 'rb'))
n_all = 0
zs, sc_all = [], []
n_gt_car = 0
gz = []
for s in d:
    p = s['pred_instances_3d']; g = s['eval_ann_info']
    pl = np.asarray(p['labels_3d']); sc = np.asarray(p['scores_3d'])
    n_all += len(sc)
    pt = p['bboxes_3d'].tensor.cpu().numpy() if hasattr(p['bboxes_3d'], 'tensor') else np.asarray(p['bboxes_3d'])
    m = pl == 0
    if m.sum():
        zs.append(pt[m, 2]); sc_all.append(sc[m])
    gl = np.asarray(g['gt_labels_3d'])
    gm = gl == 0
    n_gt_car += int(gm.sum())
    if gm.sum():
        gtt = g['gt_bboxes_3d'].tensor.cpu().numpy() if hasattr(g['gt_bboxes_3d'], 'tensor') else np.asarray(g['gt_bboxes_3d'])
        gz.append(gtt[gm, 2])
Z = np.concatenate(zs) if zs else np.array([])
S = np.concatenate(sc_all) if sc_all else np.array([])
GZ = np.concatenate(gz) if gz else np.array([])
print(f'[census:{tag}] imgs={len(d)} pred_all={n_all} ({n_all/len(d):.1f}/img) '
      f'pred_car={len(Z)} ({len(Z)/len(d):.1f}/img)', flush=True)
if len(Z):
    for lo, hi in [(0, 30), (30, 50), (50, 1e9)]:
        m = (Z >= lo) & (Z < hi)
        print(f'    car preds z[{lo},{hi if hi<1e9 else "inf"}) : {int(m.sum())} ({100*m.mean():.1f}%)')
    print(f'    car pred score: min={S.min():.4f} p1={np.percentile(S,1):.4f} '
          f'median={np.median(S):.4f} max={S.max():.4f}')
    print(f'    car pred z: median={np.median(Z):.2f} p95={np.percentile(Z,95):.2f} max={Z.max():.2f}')
print(f'    GT car (reference, identical across pkls) n={n_gt_car}')
if len(GZ):
    for lo, hi in [(0, 30), (30, 50), (50, 1e9)]:
        m = (GZ >= lo) & (GZ < hi)
        print(f'    car GT   z[{lo},{hi if hi<1e9 else "inf"}) : {int(m.sum())} ({100*m.mean():.1f}%)')

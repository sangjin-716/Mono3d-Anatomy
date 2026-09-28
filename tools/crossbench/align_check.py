"""Height-convention sanity check on a converted oracle pkl (prints to the console).

Same logic as xds_epro_fixed_alignment_check.py, parameterised over the pkl path.
Target median_dy ~ 0; the +h/2 converter bug gives ~+0.86.
"""
import os, pickle, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths
sys.path.insert(0, os.path.join(paths.UPSTREAM_ROOT, 'mmdetection3d'))   # optional source checkout
path, tag = sys.argv[1], sys.argv[2]
d = pickle.load(open(path, 'rb'))
DY = []
for s in d[:4000]:
    p = s['pred_instances_3d']; g = s['eval_ann_info']
    pl = np.asarray(p['labels_3d']); gl = np.asarray(g['gt_labels_3d']); sc = np.asarray(p['scores_3d'])
    pt = p['bboxes_3d'].tensor.cpu().numpy()
    gt = g['gt_bboxes_3d'].tensor.cpu().numpy() if hasattr(g['gt_bboxes_3d'], 'tensor') else np.asarray(g['gt_bboxes_3d'])
    m = (pl == 0) & (sc > 0.3); gm = gl == 0
    if m.sum() == 0 or gm.sum() == 0:
        continue
    P = pt[m]; G = gt[gm]
    for gi in range(len(G)):
        dxz = np.hypot(P[:, 0] - G[gi, 0], P[:, 2] - G[gi, 2]); j = int(np.argmin(dxz))
        if dxz[j] <= 1.0:
            DY.append(P[j, 1] - G[gi, 1])
DY = np.array(DY)
nb = sum(len(s['pred_instances_3d']['scores_3d']) for s in d)
print('[align:%s] n_pairs=%d median_dy=%+.3f mean_dy=%+.3f | total_pred_boxes=%d over %d imgs'
      % (tag, len(DY), np.median(DY) if len(DY) else float('nan'),
         DY.mean() if len(DY) else float('nan'), nb, len(d)), flush=True)

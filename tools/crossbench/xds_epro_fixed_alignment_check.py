"""Ported from camera-ready checks/xds_epro_fixed_alignment_check.py for the public
release. Computation unchanged. Produces a console check (no file): median vertical offset
between car predictions (score > 0.3) and the nearest car GT (within 1 m in x-z) in the
EPro-PnP-Det pkl written by xds_nusc_json_to_oracle_pkl_FIXED.py (target ~0).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, cache_dir
NUSC_WORK = cache_dir('crossbench', 'nusc')
import pickle, sys, numpy as np
sys.path.insert(0, os.path.join(paths.UPSTREAM_ROOT, 'mmdetection3d'))
d=pickle.load(open(os.path.join(NUSC_WORK, 'preds_epropnp_full_FIXED.pkl'),'rb'))
DY=[]
for s in d[:4000]:
    p=s['pred_instances_3d']; g=s['eval_ann_info']
    pl=np.asarray(p['labels_3d']); gl=np.asarray(g['gt_labels_3d']); sc=np.asarray(p['scores_3d'])
    pt=p['bboxes_3d'].tensor.cpu().numpy()
    gt=g['gt_bboxes_3d'].tensor.cpu().numpy() if hasattr(g['gt_bboxes_3d'],'tensor') else np.asarray(g['gt_bboxes_3d'])
    m=(pl==0)&(sc>0.3); gm=gl==0
    if m.sum()==0 or gm.sum()==0: continue
    P=pt[m]; G=gt[gm]
    for gi in range(len(G)):
        dxz=np.hypot(P[:,0]-G[gi,0],P[:,2]-G[gi,2]); j=int(np.argmin(dxz))
        if dxz[j]<=1.0: DY.append(P[j,1]-G[gi,1])
DY=np.array(DY)
print('EPro-PnP FIXED pkl: n_pairs=%d median_dy=%+.3f mean_dy=%+.3f  (target ~0; buggy converter gave +0.86)'%(len(DY),np.median(DY),DY.mean()))

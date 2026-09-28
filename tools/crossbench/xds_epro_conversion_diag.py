"""Console check (writes no file): median position / size / yaw of
car predictions (score > 0.3) and car GT in two converted oracle pkls, EPro-PnP-Det converted
with ct_to_oracle_pkl.py (preds_epropnp_full.pkl) and CenterNet (preds_centertrack_full.pkl).
It exposes the +h/2 vertical offset that led to xds_nusc_json_to_oracle_pkl_FIXED.py.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, cache_dir
NUSC_WORK = cache_dir('crossbench', 'nusc')
import pickle, sys, numpy as np
sys.path.insert(0, os.path.join(paths.UPSTREAM_ROOT, 'mmdetection3d'))
def dump(path,tag,N=4000):
    d=pickle.load(open(path,'rb'))
    P=[];G=[]
    for s in d[:N]:
        p=s['pred_instances_3d']; g=s['eval_ann_info']
        pl=np.asarray(p['labels_3d']); gl=np.asarray(g['gt_labels_3d']); sc=np.asarray(p['scores_3d'])
        pt=p['bboxes_3d'].tensor.cpu().numpy()
        gt=g['gt_bboxes_3d'].tensor.cpu().numpy() if hasattr(g['gt_bboxes_3d'],'tensor') else np.asarray(g['gt_bboxes_3d'])
        m=(pl==0)&(sc>0.3)
        if m.any(): P.append(pt[m][:,:7])
        if (gl==0).any(): G.append(gt[gl==0][:,:7])
    P=np.concatenate(P); G=np.concatenate(G)
    for nm,A in [('PRED',P),('GT',G)]:
        print('%-6s %-30s n=%d  x %.2f  y %.2f  z %.2f | dims(l,h,w) %.2f %.2f %.2f | yaw med %.2f p5 %.2f p95 %.2f'%(
            nm,tag,len(A),np.median(A[:,0]),np.median(A[:,1]),np.median(A[:,2]),
            np.median(A[:,3]),np.median(A[:,4]),np.median(A[:,5]),
            np.median(A[:,6]),np.percentile(A[:,6],5),np.percentile(A[:,6],95)))
    del d
dump(os.path.join(NUSC_WORK, 'preds_epropnp_full.pkl'),'EPro-PnP (converted)')
dump(os.path.join(NUSC_WORK, 'preds_centertrack_full.pkl'),'CenterTrack (converted, works)')

"""Console check (writes no file) of which height reference
CenterTrack writes as 'translation' in its nuScenes results json (box centre, bottom or top),
by matching car detections (score > 0.5, first 40 MB of the json) to devkit car annotations
within 1.5 m in BEV. Supports the height convention used in ct_to_oracle_pkl.py.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, cache_dir
NUSC_WORK = cache_dir('crossbench', 'nusc')
CT_JSON = os.path.join(paths.UPSTREAM_ROOT, 'CenterTrack', 'exp', 'ddd', 'nusc_e140_full', 'results_nuscenes_det.json')
import re, json, numpy as np, sys
from nuscenes import NuScenes
raw = open(CT_JSON,'rb').read(40_000_000).decode('utf8','ignore')
objs = re.findall(r'\{"sample_token": "[0-9a-f]{32}".*?\}', raw)
dets=[]
for o in objs:
    try: d=json.loads(o)
    except Exception: continue
    if d.get('detection_name')=='car' and d.get('detection_score',0)>0.5: dets.append(d)
print('parsed car dets (score>0.5) from first 40MB:', len(dets))
nusc = NuScenes(version='v1.0-trainval', dataroot=paths.NUSC_ROOT, verbose=False)
dz_center=[]; dz_bottom=[]; dz_top=[]
for d in dets[:400]:
    s = nusc.get('sample', d['sample_token'])
    best=None; bd=1e9
    for tk in s['anns']:
        a = nusc.get('sample_annotation', tk)
        if 'vehicle.car' not in a['category_name']: continue
        dd = np.hypot(a['translation'][0]-d['translation'][0], a['translation'][1]-d['translation'][1])
        if dd<bd: bd=dd; best=a
    if best is None or bd>1.5: continue
    hg = best['size'][2]; zg = best['translation'][2]; zd = d['translation'][2]
    dz_center.append(zd-zg); dz_bottom.append(zd-(zg-hg/2)); dz_top.append(zd-(zg+hg/2))
print('matched pairs (BEV dist<1.5m):', len(dz_center))
for nm,v in [('det_z - GT_center',dz_center),('det_z - GT_bottom',dz_bottom),('det_z - GT_top',dz_top)]:
    v=np.array(v); print('  %-20s median %+.3f  mean %+.3f'%(nm,np.median(v),v.mean()))

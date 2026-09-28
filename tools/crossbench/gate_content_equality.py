"""Compares two converted oracle pkls element by element and prints the max abs difference and
md5 of boxes, scores and labels.

This is a stronger form of the reproduction check: a freshly converted EPro-PnP-Det oracle pkl
and the earlier one that produced the table row must be identical in content (per-image box
tensors, scores, labels), not merely identical in the 2-decimal oracle printout.
Usage: python gate_content_equality.py <mine.pkl> <reference.pkl>
"""
import os, pickle, sys, hashlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths
sys.path.insert(0, os.path.join(paths.UPSTREAM_ROOT, 'mmdetection3d'))   # optional source checkout

A = pickle.load(open(sys.argv[1], 'rb'))
B = pickle.load(open(sys.argv[2], 'rb'))
print('len', len(A), len(B))
assert len(A) == len(B)
mism = 0
maxd = 0.0
hb, hs, hl = hashlib.md5(), hashlib.md5(), hashlib.md5()
nb = 0
for a, b in zip(A, B):
    pa, pb = a['pred_instances_3d'], b['pred_instances_3d']
    ta = pa['bboxes_3d'].tensor.cpu().numpy()
    tb = pb['bboxes_3d'].tensor.cpu().numpy()
    sa = np.asarray(pa['scores_3d']); sb = np.asarray(pb['scores_3d'])
    la = np.asarray(pa['labels_3d']); lb = np.asarray(pb['labels_3d'])
    nb += len(sa)
    if ta.shape != tb.shape or sa.shape != sb.shape or la.shape != lb.shape:
        mism += 1
        continue
    if ta.size:
        maxd = max(maxd, float(np.abs(ta - tb).max()))
    if sa.size:
        maxd = max(maxd, float(np.abs(sa - sb).max()))
    hb.update(np.ascontiguousarray(ta, dtype=np.float32).tobytes())
    hs.update(np.ascontiguousarray(sa, dtype=np.float32).tobytes())
    hl.update(np.ascontiguousarray(la, dtype=np.int64).tobytes())
print('shape_mismatch_images', mism, 'total_pred_boxes', nb)
print('max abs elementwise diff (boxes+scores):', maxd)
print('content md5 boxes  :', hb.hexdigest())
print('content md5 scores :', hs.hexdigest())
print('content md5 labels :', hl.hexdigest())

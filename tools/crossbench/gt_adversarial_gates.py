#!/usr/bin/env python
"""Adversarial gates A (alignment vs predictions) + B (projection consistency) for converted Waymo GT.

Ported from mono3d_crossdataset/tools/gt_adversarial_gates.py for the public release. Computation unchanged
(the console labels were translated to English). Produces a console summary and
crossbench/waymo/gates_AB.json under the re-run output directory.
Gate A: fraction of frames whose confident GUPNet predictions (score > 0.3) match some
converted Car GT in 2D (IoU > 0.5). Gate B: projected 3D centre of every Car GT inside its
2D box (+20 px margin), z > 0, and 2D box inside the 1920x1280 image.
"""
import os, sys, numpy as np, json, collections

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, out_path

GT   = os.path.join(paths.WAYMO_ROOT, 'waymo_kitti', 'validation')
PRED = os.path.join(paths.WAYMO_ROOT, 'predictions')
DEV  = os.path.join(paths.UPSTREAM_ROOT, 'DEVIANT', 'data', 'waymo', 'ImageSets')

def load_txt(path, want_cls=None):
    if not os.path.exists(path) or os.path.getsize(path) == 0: return np.zeros((0,16)), []
    rows, cls = [], []
    for ln in open(path):
        p = ln.split()
        if not p: continue
        if want_cls and p[0] not in want_cls: continue
        rows.append([float(v) for v in p[4:15]] + [float(p[15]) if len(p) > 15 else 0.0])
        cls.append(p[0])
    return (np.array(rows) if rows else np.zeros((0,12))), cls
# cols: 0-3 bbox2d, 4 h,5 w,6 l, 7 x,8 y,9 z, 10 ry, 11 score/pts

def iou2d(a, b):  # a:(N,4) b:(M,4)
    if len(a)==0 or len(b)==0: return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:,None,0], b[None,:,0]); y1 = np.maximum(a[:,None,1], b[None,:,1])
    x2 = np.minimum(a[:,None,2], b[None,:,2]); y2 = np.minimum(a[:,None,3], b[None,:,3])
    inter = np.clip(x2-x1,0,None) * np.clip(y2-y1,0,None)
    aa = (a[:,2]-a[:,0])*(a[:,3]-a[:,1]); bb = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter / (aa[:,None] + bb[None,:] - inter + 1e-9)

def parse_calib(path):
    for ln in open(path):
        if ln.startswith('P2:'):
            v = [float(x) for x in ln.split()[1:]]
            return np.array(v).reshape(3,4)
    return None

# seq -> segment (line order of val_org)
lines = [l.strip().split() for l in open(f'{DEV}/val_org.txt') if l.strip()]
seg_of = {i: s for i, (s, _) in enumerate(lines)}
GAP_SEGS = {'1071392229495085036_1844_790_1864_790','14811410906788672189_373_113_393_113',
            '17135518413411879545_1480_000_1500_000','18305329035161925340_4466_730_4486_730',
            '18333922070582247333_320_280_340_280','8888517708810165484_1549_770_1569_770',
            '9443948810903981522_6538_870_6558_870','967082162553397800_5102_900_5122_900'}
def is_gap(seq):
    s = seg_of[seq]
    s = s[len('segment-'):] if s.startswith('segment-') else s
    s = s[:-len('_with_camera_labels')] if s.endswith('_with_camera_labels') else s
    return s in GAP_SEGS

A = dict(frames_scored=0, matched=0, gap_frames_scored=0, gap_matched=0)
segstat = collections.defaultdict(lambda: [0,0])   # seg -> [scored, matched]
B = dict(gt_boxes=0, center_in_bbox=0, z_nonpos=0, bbox_oob=0)

for seq in range(39848):
    sid = f'{seq:06d}'
    gt, gcls = load_txt(f'{GT}/label/{sid}.txt', want_cls={'Car'})
    # ---- Gate B: projection consistency (all Car GT) ----
    if len(gt):
        P2 = parse_calib(f'{GT}/calib/{sid}.txt')
        x, y, z, h = gt[:,7], gt[:,8], gt[:,9], gt[:,4]
        yc = y - h/2                          # box centre (y in label = bottom)
        B['gt_boxes'] += len(gt)
        B['z_nonpos'] += int((z <= 0).sum())
        ok_z = z > 0
        u = P2[0,0]*x[ok_z]/z[ok_z] + P2[0,2]; v = P2[1,1]*yc[ok_z]/z[ok_z] + P2[1,2]
        bb = gt[ok_z,0:4]
        # margin 20px (projected_lidar_box is clip/amodal-differenced)
        inb = (u >= bb[:,0]-20) & (u <= bb[:,2]+20) & (v >= bb[:,1]-20) & (v <= bb[:,3]+20)
        B['center_in_bbox'] += int(inb.sum())
        B['bbox_oob'] += int(((gt[:,2] < 0) | (gt[:,0] > 1920) | (gt[:,3] < 0) | (gt[:,1] > 1280)).sum())
    # ---- Gate A: prediction alignment (GUPNet, score>0.3) ----
    pr, _ = load_txt(f'{PRED}/gupnet/{sid}.txt', want_cls={'Car'})
    if len(pr): pr = pr[pr[:,11] > 0.3]
    if len(pr) and len(gt):
        m = iou2d(pr[:,0:4], gt[:,0:4])
        hit = (m.max(axis=1) > 0.5).mean()    # fraction of confident preds matching some GT 2D
        A['frames_scored'] += 1; A['matched'] += int(hit > 0.5)
        st = segstat[seg_of[seq]]; st[0] += 1; st[1] += int(hit > 0.5)
        if is_gap(seq):
            A['gap_frames_scored'] += 1; A['gap_matched'] += int(hit > 0.5)

print("=== Gate A: prediction-GT alignment (GUPNet conf>0.3, 2D IoU>0.5) ===")
print(f"  all:       {A['matched']}/{A['frames_scored']} frames aligned ({100*A['matched']/max(A['frames_scored'],1):.2f}%)")
print(f"  gap segs:  {A['gap_matched']}/{A['gap_frames_scored']} ({100*A['gap_matched']/max(A['gap_frames_scored'],1):.2f}%)")
bad = sorted(((s, sc, mt) for s,(sc,mt) in segstat.items() if sc>10 and mt/sc<0.8), key=lambda t:t[2]/t[1])[:10]
print(f"  segments with alignment < 80%: {len(bad)}")
for s, sc, mt in bad: print(f"    {s[:50]}: {mt}/{sc}")
print("=== Gate B: projection consistency (Car GT) ===")
print(f"  GT boxes: {B['gt_boxes']}, centre-in-2Dbox(+20px): {100*B['center_in_bbox']/max(B['gt_boxes'],1):.2f}%")
print(f"  z<=0: {B['z_nonpos']} ({100*B['z_nonpos']/max(B['gt_boxes'],1):.3f}%), bbox fully-outside-image: {B['bbox_oob']}")
json.dump(dict(A=A, B=B), open(out_path('crossbench/waymo/gates_AB.json'),'w'), indent=1)

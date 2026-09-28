#!/usr/bin/env python
"""
Ported from mono3d_crossdataset/tools/nusc_oracle.py for the public release. Computation
unchanged. Produces the nuScenes base / true-IoU re-sort cells of the preliminary
cross-benchmark audit (printed to stdout; the run scripts append it to a log).

nusc_oracle.py -- same fixed-pool oracle diagnostics as Waymo/KITTI, on a nuScenes per-camera
prediction pkl (mmdet3d DumpResults layout, or the output of ct_to_oracle_pkl.py /
xds_nusc_json_to_oracle_pkl_FIXED.py).
Uses mmdet3d native 3D IoU (camera frame, coordinate-safe) between pred + GT CameraInstance3DBoxes.
Car class only (label 0). Reports base all-point AP / true-IoU re-sort / AP* / coverage
at IoU3D 0.7 and 0.5 (the eta/rho ratios printed alongside are not used in the paper).
Per-image = per-camera (mv_image_based), matching KITTI's per-image protocol.
Usage: python nusc_oracle.py <dump.pkl> [--tag smoke]
"""
import sys, os, argparse, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths

_MMDET3D = os.path.join(paths.UPSTREAM_ROOT, 'mmdetection3d')   # optional source checkout
sys.path.insert(0, _MMDET3D)
os.environ.setdefault('PYTHONPATH', _MMDET3D)
from mmdet3d.structures import CameraInstance3DBoxes

CAR = 0  # nuScenes class order: car=0

def _cpu(b):
    """move CameraInstance3DBoxes to CPU (the dump may hold cuda tensors)"""
    return CameraInstance3DBoxes(b.tensor.detach().cpu(), box_dim=b.tensor.shape[-1],
                                 origin=(0.5, 0.5, 0.5)) if hasattr(b,'tensor') else b

def all_point_ap(scores, is_tp, n_gt):
    """KITTI-style all-point interpolated AP from per-detection (score, tp?) + n_gt."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind='stable')
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(1 - is_tp[order])
    recall = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1)
    # monotone precision envelope (all-point / VOC2010)
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return 100.0 * ap

def greedy_match(iou, scores, thr):
    """KITTI-style: preds in score-desc order, each matches highest-IoU unmatched GT >= thr.
       Returns is_tp array aligned to pred order (original), and matched count."""
    n_pred, n_gt = iou.shape
    is_tp = np.zeros(n_pred, dtype=np.float64)
    gt_taken = np.zeros(n_gt, dtype=bool)
    for pi in np.argsort(-scores, kind='stable'):
        if n_gt == 0:
            break
        cand = iou[pi].copy()
        cand[gt_taken] = -1
        gj = int(np.argmax(cand))
        if cand[gj] >= thr:
            is_tp[pi] = 1.0
            gt_taken[gj] = True
    return is_tp

def bipartite_ceiling(iou, thr):
    """AP* numerator: max cardinality matching at IoU>=thr (Hungarian on binary)."""
    from scipy.optimize import linear_sum_assignment
    if iou.size == 0:
        return 0
    edge = (iou >= thr).astype(np.float64)
    r, c = linear_sum_assignment(edge, maximize=True)
    return int(edge[r, c].sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump'); ap.add_argument('--tag', default='full')
    args = ap.parse_args()
    data = pickle.load(open(args.dump, 'rb'))
    print(f'[load] {len(data)} images', flush=True)

    for thr, key in [(0.7, 'iou07'), (0.5, 'iou05')]:
        all_scores, all_tp = [], []
        n_gt_total = 0
        M_total = 0
        for s in data:
            pred = s['pred_instances_3d']; gt = s['eval_ann_info']
            pl = pred['labels_3d'].numpy() if hasattr(pred['labels_3d'], 'numpy') else np.asarray(pred['labels_3d'])
            gl = np.asarray(gt['gt_labels_3d'])
            pm = pl == CAR; gm = gl == CAR
            ng = int(gm.sum()); n_gt_total += ng
            if ng == 0:
                # all car preds are FP
                sc = pred['scores_3d'].numpy()[pm] if pm.any() else np.array([])
                all_scores.append(sc); all_tp.append(np.zeros(len(sc)))
                continue
            pb = pred['bboxes_3d'][pm]; gb = gt['gt_bboxes_3d'][gm]
            sc = pred['scores_3d'].numpy()[pm]
            if len(pb) == 0:
                continue
            iou = CameraInstance3DBoxes.overlaps(_cpu(pb), _cpu(gb), mode='iou').numpy()
            is_tp = greedy_match(iou, sc, thr)
            all_scores.append(sc); all_tp.append(is_tp)
            M_total += bipartite_ceiling(iou, thr)
        S = np.concatenate(all_scores) if all_scores else np.array([])
        T = np.concatenate(all_tp) if all_tp else np.array([])
        base = all_point_ap(S, T, n_gt_total)
        # re-sort: score := per-pred max IoU to any car GT (recompute tp order is same set, new scores)
        # rebuild resort scores per image
        rs_scores, rs_tp = [], []
        for s in data:
            pred = s['pred_instances_3d']; gt = s['eval_ann_info']
            pl = pred['labels_3d'].numpy(); gl = np.asarray(gt['gt_labels_3d'])
            pm = pl == CAR; gm = gl == CAR
            if pm.sum() == 0: continue
            if gm.sum() == 0:
                rs_scores.append(np.zeros(int(pm.sum()))); rs_tp.append(np.zeros(int(pm.sum()))); continue
            pb = pred['bboxes_3d'][pm]; gb = gt['gt_bboxes_3d'][gm]
            iou = CameraInstance3DBoxes.overlaps(_cpu(pb), _cpu(gb), mode='iou').numpy()
            q = iou.max(axis=1)  # oracle score = best true IoU
            is_tp = greedy_match(iou, q, thr)
            rs_scores.append(q); rs_tp.append(is_tp)
        RS = np.concatenate(rs_scores); RT = np.concatenate(rs_tp)
        resort = all_point_ap(RS, RT, n_gt_total)
        ap_star = 100.0 * M_total / max(n_gt_total, 1)
        eta = base / ap_star if ap_star > 1e-6 else 0
        rho = (resort - base) / (ap_star - base) if (ap_star - base) > 1e-6 else 0
        cov = 100.0 * M_total / max(n_gt_total, 1)
        print(f'[{key}] base={base:.2f} resort={resort:.2f} (gain +{resort-base:.2f}) '
              f'AP*={ap_star:.2f} | eta={eta:.2f} rho={rho:.2f} | coverage={M_total}/{n_gt_total}={cov:.1f}% '
              f'(uncov {100-cov:.1f}%)', flush=True)

if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Ported from mono3d_crossdataset/tools/kitti_pred_oracle.py for the public release. Computation
unchanged. Produces kitti_orig/oracle_<tag>.json in the re-run output directory (stdout carries
the same cells). Not used by the Waymo/nuScenes audit; it applies the same fixed-pool
constructs to any directory of KITTI-format predictions on KITTI val.

kitti_pred_oracle.py -- fixed-pool oracle diagnostics for KITTI-format predictions vs KITTI val GT.
Same constructs as the paper/Waymo: base all-point AP (official kernels via exact_ap),
true-IoU re-sort, AP* (bipartite ceiling), coverage. Car, moderate difficulty, IoU 0.7/0.5.
Usage (from the repository root):
  python tools/crossbench/kitti_pred_oracle.py --pred <dir with %06d.txt> [--tag name]
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, out_path

from evaluator import exact_ap
import evaluator.kitti_eval.eval as E
from evaluator.kitti_eval.kitti_common import get_label_annos
from scipy.optimize import linear_sum_assignment

GT_DIR = paths.LABEL_DIR
VAL_TXT = paths.VAL_LIST

def cmask(anno):
    n = anno['name']
    return np.array([x == 'Car' for x in n], dtype=bool) if len(n) else np.zeros(0, dtype=bool)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--tag', default='run')
    args = ap.parse_args()
    ids = [int(l.strip()) for l in open(VAL_TXT) if l.strip()]
    print(f'[load] GT {len(ids)} frames + preds from {args.pred}', flush=True)
    gt = get_label_annos(GT_DIR, ids)
    dt = get_label_annos(args.pred, ids)

    overlaps, parted, tdn, tgn = E.calculate_iou_partly(dt, gt, metric=2, num_parts=50)
    for i in range(len(ids)):
        nd, ng = len(dt[i]['name']), len(gt[i]['name'])
        assert overlaps[i].shape == (nd, ng)
        break

    out = {}
    for min_ov, key in [(0.7, 'iou07'), (0.5, 'iou05')]:
        base = exact_ap.ap_summaries(gt, dt, current_class=0, difficulty=1, metric=2, min_overlap=min_ov)
        # re-sort: score := max true IoU vs Car GT
        dt_rs = []
        for i in range(len(ids)):
            a = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in dt[i].items()}
            if len(a['name']):
                cg = cmask(gt[i])
                if cg.sum() and overlaps[i].size:
                    a['score'] = overlaps[i][:, cg].max(axis=1).astype(np.float64)
                else:
                    a['score'] = np.zeros(len(a['name']))
            dt_rs.append(a)
        rs = exact_ap.ap_summaries(gt, dt_rs, current_class=0, difficulty=1, metric=2, min_overlap=min_ov)
        # AP* vs valid (KITTI moderate rule) Car GT
        M, n_gt = 0, 0
        for i in range(len(ids)):
            cg = cmask(gt[i])
            if not cg.sum(): continue
            occ = gt[i]['occluded'][cg]; trunc = gt[i]['truncated'][cg]
            h2d = gt[i]['bbox'][cg][:, 3] - gt[i]['bbox'][cg][:, 1]
            valid = (h2d >= 25.0) & (occ <= 1) & (trunc <= 0.30)
            n_gt += int(valid.sum())
            if valid.sum() and len(dt[i]['name']) and overlaps[i].size:
                cd = cmask(dt[i])
                ov = overlaps[i][np.ix_(cd, cg)][:, valid]
                if ov.size:
                    edge = (ov >= min_ov).astype(np.float64)
                    r, c = linear_sum_assignment(edge, maximize=True)
                    M += int(edge[r, c].sum())
        ap_star = 100.0 * M / max(n_gt, 1)
        b = base['allpoint']; r_ = rs['allpoint']
        eta = b / ap_star if ap_star > 1e-6 else 0
        rho = (r_ - b) / (ap_star - b) if ap_star - b > 1e-6 else 0
        out[key] = dict(base=b, base_r40=base['r40_official'], resort=r_, ap_star=ap_star,
                        eta=eta, rho=rho, M=M, n_gt=n_gt)
        print(f'[{args.tag}][{key}] base={b:.2f} (R40 {base["r40_official"]:.2f}) '
              f'resort={r_:.2f} (+{r_-b:.2f}) AP*={ap_star:.2f} | eta={eta:.2f} rho={rho:.2f} '
              f'| coverage={M}/{n_gt}={100*M/max(n_gt,1):.1f}%', flush=True)
    json.dump(out, open(out_path(os.path.join('kitti_orig', f'oracle_{args.tag}.json')), 'w'), indent=1)

if __name__ == '__main__':
    main()

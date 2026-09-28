#!/usr/bin/env python
"""
Ported from mono3d_crossdataset/tools/waymo_core_analysis.py for the public release.
Computation unchanged. Produces crossbench/waymo/core_table_full.json (re-run output directory):
the Waymo base / true-IoU re-sort cells of the preliminary cross-benchmark audit.

waymo_core_analysis.py -- the paper's core fixed-pool diagnostics on Waymo val (3 detectors),
using the paper's OWN evaluator (evaluator/exact_ap.py + official KITTI matching kernels) verbatim.

Outputs per detector, Car, difficulty=moderate-rule (reduces to 2D-height>=25px since occ=trunc=0),
metric=3D, IoU 0.7 and 0.5:
  base allpoint AP | true-IoU re-sort allpoint AP (gap) | AP* = M/n_gt ceiling | coverage
Inputs: <WAYMO_ROOT>/waymo_kitti/validation/label (from waymo_v2_to_kitti_gt.py) and the released
KITTI-format Waymo val predictions under <WAYMO_ROOT>/predictions/{gupnet,deviant,monorcnnpp}.

Usage: python tools/crossbench/waymo_core_analysis.py [--smoke N]
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, out_path

from evaluator import exact_ap
import evaluator.kitti_eval.eval as E
from evaluator.kitti_eval.kitti_common import get_label_annos
from scipy.optimize import linear_sum_assignment

GT_DIR   = os.path.join(paths.WAYMO_ROOT, 'waymo_kitti', 'validation', 'label')
PRED     = os.path.join(paths.WAYMO_ROOT, 'predictions')
OUT_DIR  = os.path.dirname(out_path(os.path.join('crossbench', 'waymo', 'core_table_full.json')))
N_ALL    = 39848
MRC_OFF  = 52386

def prep_monorcnn():
    """one-time: VEHICLE->Car + id-offset remap into monorcnnpp_kitti/"""
    dst = f'{PRED}/monorcnnpp_kitti'
    if os.path.isdir(dst) and len(os.listdir(dst)) == N_ALL: return dst
    os.makedirs(dst, exist_ok=True)
    for i in range(N_ALL):
        src = f'{PRED}/monorcnnpp/{i+MRC_OFF:06d}.txt'
        txt = ''
        if os.path.exists(src):
            txt = open(src).read().replace('VEHICLE', 'Car')
        with open(f'{dst}/{i:06d}.txt', 'w') as f: f.write(txt)
    return dst

def load_annos(folder, ids):
    return get_label_annos(folder, ids)

def cmask(anno):
    """robust Car mask (empty annos give float empty arrays where ==' Car' is scalar False)"""
    n = anno['name']
    return np.array([x == 'Car' for x in n], dtype=bool) if len(n) else np.zeros(0, dtype=bool)

def car_gt_boxes3d(anno):
    m = anno['name'] == 'Car'
    if m.sum() == 0: return np.zeros((0,7)), np.zeros(0)
    loc, dim, ry = anno['location'][m], anno['dimensions'][m], anno['rotation_y'][m]
    h2d = anno['bbox'][m][:,3] - anno['bbox'][m][:,1]
    return np.concatenate([loc, dim, ry[:,None]], 1), h2d

def det_boxes3d(anno):
    m = anno['name'] == 'Car'
    if m.sum() == 0: return np.zeros((0,7)), np.zeros(0), m
    loc, dim, ry = anno['location'][m], anno['dimensions'][m], anno['rotation_y'][m]
    return np.concatenate([loc, dim, ry[:,None]], 1), anno['score'][m], m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0)
    args = ap.parse_args()
    ids = list(range(args.smoke if args.smoke else N_ALL))
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f'smoke{args.smoke}' if args.smoke else 'full'

    print(f'[load] GT annos ({len(ids)}) ...', flush=True)
    gt = load_annos(GT_DIR, ids)
    mrc_dir = prep_monorcnn()
    DET = {'gupnet': f'{PRED}/gupnet', 'deviant': f'{PRED}/deviant', 'monorcnnpp': mrc_dir}

    results = {}
    for name, pdir in DET.items():
        print(f'[{name}] load preds ...', flush=True)
        dt = load_annos(pdir, ids)

        # ---- per-box true 3D IoU (official rotate-iou kernels via calculate_iou_partly) ----
        print(f'[{name}] IoU (official kernels) ...', flush=True)
        overlaps, parted, total_dt_num, total_gt_num = E.calculate_iou_partly(dt, gt, metric=2, num_parts=200)
        # overlaps[i]: rows=dt_i, cols=gt_i  (assert once)
        for i in range(len(ids)):
            nd, ng = len(dt[i]['name']), len(gt[i]['name'])
            assert overlaps[i].shape == (nd, ng), (i, overlaps[i].shape, nd, ng)
            break

        res = {}
        for min_ov, key in [(0.7, 'iou07'), (0.5, 'iou05')]:
            # ---- base + re-sort via the paper's evaluator ----
            base = exact_ap.ap_summaries(gt, dt, current_class=0, difficulty=1,
                                         metric=2, min_overlap=min_ov)
            # re-sort: score := max true-IoU vs Car GT (any validity), no box changed
            dt_rs = []
            for i in range(len(ids)):
                a = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in dt[i].items()}
                if len(a['name']):
                    car_gt = cmask(gt[i])
                    if car_gt.sum() and overlaps[i].size:
                        a['score'] = overlaps[i][:, car_gt].max(axis=1).astype(np.float64)
                    else:
                        a['score'] = np.zeros(len(a['name']))
                dt_rs.append(a)
            rs = exact_ap.ap_summaries(gt, dt_rs, current_class=0, difficulty=1,
                                       metric=2, min_overlap=min_ov)

            # ---- AP* = M/n_gt (max bipartite matching vs VALID Car GT: h2d>=25) ----
            M, n_gt = 0, 0
            for i in range(len(ids)):
                car_gt = cmask(gt[i])
                if car_gt.sum():
                    h2d = gt[i]['bbox'][car_gt][:,3] - gt[i]['bbox'][car_gt][:,1]
                    valid = h2d >= 25.0
                    n_gt += int(valid.sum())
                    if valid.sum() and len(dt[i]['name']) and overlaps[i].size:
                        car_dt = cmask(dt[i])
                        ov = overlaps[i][np.ix_(car_dt, car_gt)][:, valid]
                        if ov.size:
                            edge = (ov >= min_ov).astype(np.float64)
                            r, c = linear_sum_assignment(edge, maximize=True)
                            M += int(edge[r, c].sum())
            ap_star = 100.0 * M / max(n_gt, 1)

            res[key] = dict(
                base_allpoint = base['allpoint'], base_r40 = base['r40_official'],
                resort_allpoint = rs['allpoint'],
                resort_gain = rs['allpoint'] - base['allpoint'],
                ap_star = ap_star, ap_star_gap = ap_star - base['allpoint'],
                coverage_M = M, n_gt = n_gt, coverage = M / max(n_gt, 1))
            print(f'[{name}][{key}] base={base["allpoint"]:.2f} (R40 {base["r40_official"]:.2f}) '
                  f'resort={rs["allpoint"]:.2f} (gain +{rs["allpoint"]-base["allpoint"]:.2f}) '
                  f'AP*={ap_star:.2f} (gap +{ap_star-base["allpoint"]:.2f}) '
                  f'coverage={M}/{n_gt}={100*M/max(n_gt,1):.1f}%', flush=True)
        results[name] = res

    with open(f'{OUT_DIR}/core_table_{tag}.json', 'w') as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))

if __name__ == '__main__':
    main()

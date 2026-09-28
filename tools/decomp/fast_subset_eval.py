"""Produces no report (library).

Cached subset evaluator for KITTI AP3D R40 (Car, moderate, IoU0.7), for the drive-cluster bootstrap.
NOT a reimplementation: it precomputes the per-image 3D-IoU overlaps (the only GPU step, via the OFFICIAL
calculate_iou_partly) + the official _prepare_data ONCE per variant, then evaluates any image SUBSET by
re-assembling the official `parted_overlaps` (block-diagonal) and running the OFFICIAL jit aggregation
(compute_statistics_jit / fused_compute_statistics / get_thresholds / get_mAP_R40). Subset eval is CPU-only
(no GPU contention). Parity vs do_eval is exact by construction.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np
import evaluator.kitti_eval.kitti_common as kc
from evaluator.kitti_eval import eval as E

# Car, moderate, IoU0.7 -> min_overlaps cell [overlap_set=0, metric=2(3d), class=0]
_OV = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7]] * 3)  # [3 difficulty, 6 class] for 3d row
MIN_OVERLAP = 0.7
METRIC = 2; CLASS = 0; DIFF = 1  # moderate


class CachedEval:
    def __init__(self, gt_annos, dt_annos):
        # NOTE eval_class calls calculate_iou_partly(dt, gt): first arg dt, second gt.
        ov, parted, total_dt, total_gt = E.calculate_iou_partly(dt_annos, gt_annos, METRIC, 50)
        self.overlaps = ov  # per-image (det x gt) matrices
        # official per-image prepared data for (Car, moderate)
        rets = E._prepare_data(gt_annos, dt_annos, CLASS, DIFF, DIForDIS=True)
        (self.gt_datas, self.dt_datas, self.ignored_gts, self.ignored_dets,
         self.dontcares, self.total_dc_num, _) = rets
        self.n = len(gt_annos)
        self.num_valid_gt = []  # per-image valid GT count (for recall denom on subsets)
        self.tp_scores = []     # per-image TP scores at IoU0.7 (compute_fp=False) — precomputed for fast bootstrap
        for i in range(self.n):
            self.num_valid_gt.append(int((self.ignored_gts[i] == 0).sum()))
            r = E.compute_statistics_jit(self.overlaps[i], self.gt_datas[i], self.dt_datas[i],
                                         self.ignored_gts[i], self.ignored_dets[i], self.dontcares[i],
                                         METRIC, min_overlap=MIN_OVERLAP, thresh=0.0, compute_fp=False)
            self.tp_scores.append(r[4])
        self.num_valid_gt = np.array(self.num_valid_gt)

    def eval_subset(self, positions):
        """positions: list/array of image indices (0..n-1), repeats allowed. Returns AP3D R40 mod (%)."""
        pos = list(positions)
        ov = [self.overlaps[p] for p in pos]
        gd = [self.gt_datas[p] for p in pos]; dd = [self.dt_datas[p] for p in pos]
        ig = [self.ignored_gts[p] for p in pos]; idd = [self.ignored_dets[p] for p in pos]
        dc = [self.dontcares[p] for p in pos]
        nvg = int(self.num_valid_gt[pos].sum())
        # pass 1: thresholds from PRECOMPUTED per-image TP scores (identical to per-image jit gather, fast)
        thr_all = np.concatenate([self.tp_scores[p] for p in pos]) if pos else np.zeros(0)
        thresholds = np.array(E.get_thresholds(thr_all, nvg))
        if len(thresholds) == 0:
            return 0.0
        # pass 2: PR over thresholds using official fused_compute_statistics on re-parted block overlaps
        split_parts = E.get_split_parts(len(pos), 50)
        gt_nums = np.array([gd[i].shape[0] for i in range(len(pos))])
        dt_nums = np.array([dd[i].shape[0] for i in range(len(pos))])
        dc_nums = np.array([dc[i].shape[0] for i in range(len(pos))])
        pr = np.zeros([len(thresholds), 4])
        idx = 0
        for num_part in split_parts:
            sl = slice(idx, idx + num_part)
            # block-diagonal parted overlap (det x gt); off-blocks unread by fused
            tot_dt = int(dt_nums[sl].sum()); tot_gt = int(gt_nums[sl].sum())
            parted = np.zeros((tot_dt, tot_gt), np.float64)
            do = go = 0
            for i in range(idx, idx + num_part):
                d, g = dd[i].shape[0], gd[i].shape[0]
                parted[do:do + d, go:go + g] = ov[i]
                do += d; go += g
            gd_p = np.concatenate(gd[sl], 0); dd_p = np.concatenate(dd[sl], 0)
            dc_p = np.concatenate(dc[sl], 0)
            ig_p = np.concatenate(ig[sl], 0); idd_p = np.concatenate(idd[sl], 0)
            E.fused_compute_statistics(parted, pr, gt_nums[sl], dt_nums[sl], dc_nums[sl],
                                       gd_p, dd_p, dc_p, ig_p, idd_p, METRIC,
                                       min_overlap=MIN_OVERLAP, thresholds=thresholds, compute_aos=False)
            idx += num_part
        prec = np.zeros(41)
        for i in range(len(thresholds)):
            prec[i] = pr[i, 0] / (pr[i, 0] + pr[i, 1]) if (pr[i, 0] + pr[i, 1]) > 0 else 0.0
        for i in range(len(thresholds)):
            prec[i] = prec[i:].max()
        return float(E.get_mAP_R40(prec))


def load_dt(kitti_dir, sids):
    return kc.get_label_annos(kitti_dir, sids)

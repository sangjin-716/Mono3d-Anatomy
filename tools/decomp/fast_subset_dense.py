"""Produces no report (library + validation gate printed to stdout).

Dense-sampled (401-pt) subset evaluator, a subclass of the validated CachedEval
(fast_subset_eval.py is not modified). Bootstrap primary metric decision (fixed before running):
true all-point AP needs ~10k thresholds per replicate (infeasible at B=2000), so the paired
bootstrap uses the 401-pt dense interpolated AP, whose convergence to all-point was measured
at ≈0.07–0.09 absolute (exact_ap ladder: DGP 22.834→22.905, GUPNet 17.030→17.122), a bias
that cancels in PAIRED ΔAP. Official R40 is computed alongside as secondary.
Validation gate: full-set eval == exact_ap param401 and == official R40 (<0.02).
Run from the repository root: python tools/decomp/fast_subset_dense.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np
from evaluator.kitti_eval import eval as E
from fast_subset_eval import CachedEval, METRIC, MIN_OVERLAP

N_DENSE = 401


class CachedEvalDense(CachedEval):
    def _pr_at(self, pos, thresholds):
        ov = [self.overlaps[p] for p in pos]
        gd = [self.gt_datas[p] for p in pos]; dd = [self.dt_datas[p] for p in pos]
        ig = [self.ignored_gts[p] for p in pos]; idd = [self.ignored_dets[p] for p in pos]
        dc = [self.dontcares[p] for p in pos]
        split_parts = E.get_split_parts(len(pos), 50)
        gt_nums = np.array([gd[i].shape[0] for i in range(len(pos))])
        dt_nums = np.array([dd[i].shape[0] for i in range(len(pos))])
        dc_nums = np.array([dc[i].shape[0] for i in range(len(pos))])
        pr = np.zeros([len(thresholds), 4])
        idx = 0
        for num_part in split_parts:
            sl = slice(idx, idx + num_part)
            tot_dt = int(dt_nums[sl].sum()); tot_gt = int(gt_nums[sl].sum())
            parted = np.zeros((tot_dt, tot_gt), np.float64)
            do = go = 0
            for i in range(idx, idx + num_part):
                d, g = dd[i].shape[0], gd[i].shape[0]
                parted[do:do + d, go:go + g] = ov[i]
                do += d; go += g
            E.fused_compute_statistics(parted, pr, gt_nums[sl], dt_nums[sl], dc_nums[sl],
                                       np.concatenate(gd[sl], 0), np.concatenate(dd[sl], 0),
                                       np.concatenate(dc[sl], 0), np.concatenate(ig[sl], 0),
                                       np.concatenate(idd[sl], 0), METRIC,
                                       min_overlap=MIN_OVERLAP, thresholds=thresholds,
                                       compute_aos=False)
            idx += num_part
        return pr

    def eval_subset_pair(self, positions):
        """(dense401 AP, official R40) for an image multiset — one stats pass per grid."""
        pos = list(positions)
        nvg = int(self.num_valid_gt[pos].sum())
        thr_all = np.concatenate([self.tp_scores[p] for p in pos]) if pos else np.zeros(0)
        if nvg == 0 or len(thr_all) == 0:
            return 0.0, 0.0
        res = []
        for n in (N_DENSE, 41):
            ths = np.array(E.get_thresholds(thr_all.copy(), nvg, num_sample_pts=n))
            pr = self._pr_at(pos, ths)
            prec = np.zeros(n)
            for i in range(len(ths)):
                prec[i] = pr[i, 0] / (pr[i, 0] + pr[i, 1]) if (pr[i, 0] + pr[i, 1]) > 0 else 0.0
            for i in range(len(ths)):
                prec[i] = prec[i:].max()
            res.append(float(prec[1:].sum() / (n - 1) * 100))
        return res[0], res[1]


if __name__ == "__main__":
    # validation gate on 2 detectors (full val set)
    import os, shutil, pandas as pd
    import ap_corrector_arc as arc
    from dgp_cop_oracle_matrix import write_kitti, pool_mask, apply_nms
    import evaluator.kitti_eval.kitti_common as kc
    from exact_ap import ap_summaries
    val = [int(x) for x in open(arc.VAL_LIST).read().split()]
    GT = kc.get_label_annos(arc.LABEL_DIR, val)
    W = os.path.join(cache_dir("decomp"), "_fsd")
    for f in ("dgp", "gupnet"):
        df = pd.read_csv(dump_path(f))
        pre = df[pool_mask(df, "thr0.2")].copy()
        keep = apply_nms(pre, pre["V"].values, 0.5); kept = pre[keep].copy()
        if os.path.exists(W):
            shutil.rmtree(W)
        write_kitti(kept.reset_index(drop=True), kept["V"].values.astype(float),
                    os.path.join(W, "data"), val)
        DT = kc.get_label_annos(os.path.join(W, "data"), val)
        ce = CachedEvalDense(GT, DT)
        d401, r40 = ce.eval_subset_pair(range(len(val)))
        s = ap_summaries(GT, DT, ladder=(401,))
        ok = abs(d401 - s["param401"]) < 0.02 and abs(r40 - s["r40_official"]) < 0.02
        print(f"{f}: dense401 {d401:.3f} vs {s['param401']:.3f}; R40 {r40:.3f} vs "
              f"{s['r40_official']:.3f} [{'PASS' if ok else 'FAIL'}]", flush=True)
    if os.path.exists(W):
        shutil.rmtree(W)

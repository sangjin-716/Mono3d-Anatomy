"""All-point interpolated AP ("all-threshold PR area") for KITTI, a de-quantized AP summary.

The official KITTI kernels are imported from the vendored evaluator/kitti_eval, so no detector
repo is needed. Writes no report; the __main__ validation gates print to stdout.

NOT a "continuous/true AUC": recall is still discrete (finite GT), precision uses the official
monotone interpolation envelope, integration is stepwise, and everything is downstream of the
official KITTI matching rules. We call it "all-point interpolated AP".

Reuses the official matching code unchanged (calculate_iou_partly / _prepare_data /
compute_statistics_jit / fused_compute_statistics / get_thresholds); the matching is not
re-implemented. Differences are ONLY (a) which thresholds are evaluated and (b) how the
envelope is summarized:
  R40_official  = mean of 41-pt envelope at indices 1..40            (== get_mAP_R40)
  R11_official  = mean of 41-pt envelope at indices 0,4,...,40       (== get_mAP, true R11)
  paramN        = mean of N-pt envelope at indices 1..N-1            (sampling-ladder diagnostic;
                                                                      param41 == R40_official)
  allpoint      = step-integral of the envelope over RAW recall using ALL unique detection
                  thresholds; recall ties collapsed to max envelope precision.
Tie policy: thresholds are score values, so all detections with equal score enter together
(official kernel semantics, >= threshold); the all-threshold list is np.unique(scores).

Validation gates (run __main__):
  (1) toy examples with hand-computable AP (perfect ranking / tied-score with FP);
  (2) param41 == official do_eval R40 (<0.01) and R11_official == do_eval R11 on real dumps
      (2 detectors quick, all 12 with --full).
Run from the repository root: python evaluator/exact_ap.py [--full]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import evaluator.kitti_eval.eval as E
from evaluator.kitti_eval.eval import (
    calculate_iou_partly, _prepare_data, compute_statistics_jit,
    fused_compute_statistics, get_split_parts, get_thresholds)


def _pr_env_at_thresholds(gt_annos, dt_annos, current_class, difficulty, metric,
                          min_overlap, thresholds, parted_overlaps,
                          total_dt_num, total_gt_num, split_parts, DIForDIS=True):
    """raw recall + monotone-enveloped precision at the given thresholds (official kernels)."""
    rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty, DIForDIS=DIForDIS)
    (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
     dontcares, total_dc_num, total_num_valid_gt) = rets
    thresholds = np.array(thresholds)
    pr = np.zeros([len(thresholds), 4])
    idx = 0
    for j, num_part in enumerate(split_parts):
        gp = np.concatenate(gt_datas_list[idx:idx + num_part], 0)
        dp = np.concatenate(dt_datas_list[idx:idx + num_part], 0)
        dcp = np.concatenate(dontcares[idx:idx + num_part], 0)
        idp = np.concatenate(ignored_dets[idx:idx + num_part], 0)
        igp = np.concatenate(ignored_gts[idx:idx + num_part], 0)
        fused_compute_statistics(
            parted_overlaps[j], pr, total_gt_num[idx:idx + num_part],
            total_dt_num[idx:idx + num_part], total_dc_num[idx:idx + num_part],
            gp, dp, dcp, igp, idp, metric, min_overlap=min_overlap,
            thresholds=thresholds, compute_aos=False)
        idx += num_part
    n = len(thresholds)
    prec = np.zeros(n); rec = np.zeros(n)
    for i in range(n):
        rec[i] = pr[i, 0] / (pr[i, 0] + pr[i, 2]) if (pr[i, 0] + pr[i, 2]) > 0 else 0.0
        prec[i] = pr[i, 0] / (pr[i, 0] + pr[i, 1]) if (pr[i, 0] + pr[i, 1]) > 0 else 0.0
    prec_env = prec.copy()
    for i in range(n):                      # official monotone precision envelope
        prec_env[i] = np.max(prec[i:])
    # recall stays RAW (official code never uses the recall axis; the integral needs it raw)
    return rec, prec_env, total_num_valid_gt


def _all_scores(gt_annos, dt_annos, current_class, difficulty, metric, min_overlap,
                overlaps, DIForDIS=True):
    rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty, DIForDIS=DIForDIS)
    (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
     dontcares, total_dc_num, total_num_valid_gt) = rets
    th = []
    for i in range(len(gt_annos)):
        r = compute_statistics_jit(
            overlaps[i], gt_datas_list[i], dt_datas_list[i], ignored_gts[i],
            ignored_dets[i], dontcares[i], metric, min_overlap=min_overlap,
            thresh=0.0, compute_fp=False)
        th += r[4].tolist()
    return np.array(th), total_num_valid_gt


def _pad_env(prec_env, n):
    """zero-pad envelope to n entries (official arrays are zero-initialized to N pts)."""
    if len(prec_env) >= n:
        return prec_env[:n]
    return np.concatenate([prec_env, np.zeros(n - len(prec_env))])


def ap_summaries(gt_annos, dt_annos, current_class=0, difficulty=1, metric=2,
                 min_overlap=0.7, num_parts=50, DIForDIS=True, ladder=()):
    """One IoU pass -> dict with r11_official, r40_official, allpoint (+ paramN for N in ladder)."""
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    split_parts = get_split_parts(len(gt_annos), num_parts)
    scores, nvg = _all_scores(gt_annos, dt_annos, current_class, difficulty, metric,
                              min_overlap, overlaps, DIForDIS)
    out = {}
    if nvg == 0 or len(scores) == 0:
        base = {"r11_official": 0.0, "r40_official": 0.0, "allpoint": 0.0}
        base.update({f"param{n}": 0.0 for n in ladder})
        return base

    # official 41-pt envelope -> both official summaries from the SAME array
    th41 = np.array(get_thresholds(scores.copy(), nvg, num_sample_pts=41))
    _, env41, _ = _pr_env_at_thresholds(gt_annos, dt_annos, current_class, difficulty, metric,
                                        min_overlap, th41, parted_overlaps,
                                        total_dt_num, total_gt_num, split_parts, DIForDIS)
    env41 = _pad_env(env41, 41)
    out["r40_official"] = float(env41[1:].sum() / 40 * 100)          # == get_mAP_R40
    out["r11_official"] = float(env41[0::4].sum() / 11 * 100)        # == get_mAP (true R11)

    for n in ladder:                                                  # sampling-ladder diagnostic
        thn = np.array(get_thresholds(scores.copy(), nvg, num_sample_pts=n))
        _, envn, _ = _pr_env_at_thresholds(gt_annos, dt_annos, current_class, difficulty, metric,
                                           min_overlap, thn, parted_overlaps,
                                           total_dt_num, total_gt_num, split_parts, DIForDIS)
        envn = _pad_env(envn, n)
        out[f"param{n}"] = float(envn[1:].sum() / (n - 1) * 100)

    # all-point interpolated AP over ALL unique detection thresholds
    ths = np.unique(scores)[::-1]
    rec, env, _ = _pr_env_at_thresholds(gt_annos, dt_annos, current_class, difficulty, metric,
                                        min_overlap, ths, parted_overlaps,
                                        total_dt_num, total_gt_num, split_parts, DIForDIS)
    # collapse recall ties: keep max envelope precision per unique recall value
    ur = np.unique(rec)
    up = np.array([env[rec == v].max() for v in ur])
    r = np.concatenate([[0.0], ur]); p = np.concatenate([[up[0] if len(up) else 0.0], up])
    out["allpoint"] = float(np.sum(np.diff(r) * p[1:]) * 100)
    return out


# ----------------------------- validation gates -----------------------------

def _toy_annos(dt_spec):
    """1-image toy. GT: two easy Cars (identical boxes reused by DTs => IoU=1).
    dt_spec: list of (which_gt_or_None, score). None => far-displaced FP."""
    def boxes(n):
        return dict(
            name=np.array(["Car"] * n), truncated=np.zeros(n), occluded=np.zeros(n, np.int64),
            alpha=np.zeros(n), bbox=np.tile([100.0, 100.0, 200.0, 200.0], (n, 1)),
            dimensions=np.tile([4.0, 1.5, 1.6], (n, 1)),     # (l,h,w) anno convention
            location=np.zeros((n, 3)), rotation_y=np.zeros(n))
    gt = boxes(2)
    gt["location"] = np.array([[0.0, 1.5, 10.0], [3.0, 1.5, 20.0]])
    n = len(dt_spec)
    dt = boxes(n)
    dt["score"] = np.array([s for _, s in dt_spec], float)
    loc = []
    for which, _ in dt_spec:
        loc.append(gt["location"][which] if which is not None else np.array([20.0, 1.5, 50.0]))
    dt["location"] = np.array(loc)
    gt["score"] = np.zeros(2)
    return [gt], [dt]


def run_toys():
    """allpoint vs HAND-MATH; r40/r11 vs OFFICIAL do_eval on the same toy annos.
    (With only 2 GTs the official R40 fills 2 of 41 recall slots -> R40=2.5 even for a
    perfect detector — the quantization pathology under study; so R40 is gated against the
    official code, not against intuition.)"""
    from evaluator.kitti_eval.eval import do_eval as _de
    ov07 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7]] * 3)
    ov05 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    MO_ = np.stack([ov07, ov05], 0)[:, :, [0]]

    def gate(tag, spec, hand_allpoint):
        g, d = _toy_annos(spec)
        s = ap_summaries(g, d)
        r = _de(g, d, [0], MO_, compute_aos=False, DIForDIS=True)
        off40, off11 = float(r[6][0, 1, 0]), float(r[2][0, 1, 0])
        ok = (abs(s["allpoint"] - hand_allpoint) < 0.5
              and abs(s["r40_official"] - off40) < 1e-6
              and abs(s["r11_official"] - off11) < 1e-6)
        print(f"{tag}: allpoint={s['allpoint']:.2f} (hand {hand_allpoint:.2f})  "
              f"R40 mine/off={s['r40_official']:.2f}/{off40:.2f}  "
              f"R11 mine/off={s['r11_official']:.2f}/{off11:.2f}  [{'PASS' if ok else 'FAIL'}]",
              flush=True)
        return ok

    okA = gate("toyA perfect-rank", [(0, 0.9), (1, 0.8), (None, 0.7)], 100.0)
    okB = gate("toyB tied-scores ", [(0, 0.8), (1, 0.8), (None, 0.8)], 200 / 3)
    return okA and okB


if __name__ == "__main__":
    import shutil, pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "decomp"))
    from tools._release import dump_path, cache_dir
    import ap_corrector_arc as arc
    from dgp_cop_oracle_matrix import write_kitti, pool_mask, apply_nms
    import evaluator.kitti_eval.kitti_common as kc
    from evaluator.kitti_eval.eval import do_eval

    if not run_toys():
        sys.exit("TOY GATE FAILED")

    dets = [("dgp",), ("gupnet",)]
    if "--full" in sys.argv:
        from depth_share_bridge import DETS as D12
        dets = [(f,) for _, f in D12]
    val = [int(x) for x in open(arc.VAL_LIST).read().split()]
    GT = kc.get_label_annos(arc.LABEL_DIR, val)
    ov07 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7]] * 3)
    ov05 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    MO = np.stack([ov07, ov05], 0)[:, :, [0]]
    W = os.path.join(cache_dir("decomp"), "_exactval")
    allok = True
    for (f,) in dets:
        df = pd.read_csv(dump_path(f)).reset_index(drop=True)
        pre = df[pool_mask(df, "thr0.2")].copy()
        keep = apply_nms(pre, pre["V"].values, 0.5); kept = pre[keep].copy().reset_index(drop=True)
        if os.path.exists(W):
            shutil.rmtree(W)
        write_kitti(kept.reset_index(drop=True), kept["V"].values.astype(float), os.path.join(W, "data"), val)
        DT = kc.get_label_annos(os.path.join(W, "data"), val)
        r = do_eval(GT, DT, [0], MO, compute_aos=False, DIForDIS=True)
        off40, off11 = float(r[6][0, 1, 0]), float(r[2][0, 1, 0])
        s = ap_summaries(GT, DT, ladder=(101, 401))
        ok = abs(off40 - s["r40_official"]) < 0.01 and abs(off11 - s["r11_official"]) < 0.01
        allok &= ok
        print(f"{f:16s} R40 off/mine={off40:7.3f}/{s['r40_official']:7.3f}  "
              f"R11 off/mine={off11:7.3f}/{s['r11_official']:7.3f}  "
              f"p101={s['param101']:.3f} p401={s['param401']:.3f} allpoint={s['allpoint']:.3f} "
              f"[{'PASS' if ok else 'FAIL'}]", flush=True)
    if os.path.exists(W):
        shutil.rmtree(W)
    print("ALL GATES:", "PASS" if allok else "FAIL")

"""Produces reports/gap_exact.txt (and the _bridgecache_<f>.npz o_act/o_z caches reused by
gap_metric_robustness.py and e3_endpoint_gap_boot.py; built on first run).

Order-ceiling gap under official R11 / official R40 / all-point interpolated AP,
12 detectors. Embedded validation: for every detector the base pool's ap_summaries must match
official do_eval R40 AND R11 (<0.01) or the run aborts (per-detector evaluator gate).
base = AP(V); ceiling = AP(rank by oracle IoU_act, from _bridgecache_<f>.npz).
Run from the repository root: python tools/decomp/gap_exact.py
"""
import os, sys, shutil, datetime, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti, pool_mask, apply_nms
import evaluator.kitti_eval.kitti_common as kc
from evaluator.kitti_eval.eval import do_eval
from exact_ap import ap_summaries
from depth_share_bridge import iou_act_and_zstar, DETS, DIAG

val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
ov07 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7]] * 3)
ov05 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
MO = np.stack([ov07, ov05], 0)[:, :, [0]]
WORK = f"{DIAG}/_gapexact"
OUT = out_path("gap_exact.txt")
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def annos_for(df, score):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    return kc.get_label_annos(d, val)


w(f"# gap_exact run {datetime.datetime.now().isoformat(timespec='seconds')}")
w(f"# script tools/decomp/gap_exact.py + exact_ap.py (toy+official gates PASS this version)")
w(f"# pools: S5 (cls>=0.2 + 2D-NMS@0.5 by V); ceiling scores: _bridgecache_<f>.npz o_act")
w(f"# metric: Car moderate IoU0.7; 'allpoint' = all-point interpolated AP (NOT continuous AUC)")
w("")

rows = []
for name, f in DETS:
    df = pd.read_csv(dump_path(f)).reset_index(drop=True)
    md5 = hashlib.md5(open(dump_path(f), "rb").read(1 << 20)).hexdigest()[:8]
    pre = df[pool_mask(df, "thr0.2")].copy()
    keep = apply_nms(pre, pre["V"].values, 0.5)
    kept = pre[keep].copy().reset_index(drop=True)
    cache = f"{DIAG}/_bridgecache_{f}.npz"
    if os.path.exists(cache):
        z = np.load(cache); o_act = z["o_act"]
        assert len(o_act) == len(kept), f"cache misaligned for {f}"
    else:
        o_act, o_z = iou_act_and_zstar(kept)
        np.savez_compressed(cache, o_act=o_act, o_z=o_z)

    DTb = annos_for(kept, kept["V"].values.astype(float))
    r = do_eval(GT, DTb, [0], MO, compute_aos=False, DIForDIS=True)
    off40, off11 = float(r[6][0, 1, 0]), float(r[2][0, 1, 0])
    sb = ap_summaries(GT, DTb)
    assert abs(sb["r40_official"] - off40) < 0.01 and abs(sb["r11_official"] - off11) < 0.01, \
        f"EVALUATOR GATE FAIL {name}: {sb} vs official {off40}/{off11}"

    DTc = annos_for(kept, o_act)
    sc = ap_summaries(GT, DTc)

    rows.append(dict(name=name, b=sb, c=sc))
    w(f"[done] {name:10s} dump@{md5} GATE-OK | base R11/R40/allpt="
      f"{sb['r11_official']:6.2f}/{sb['r40_official']:6.2f}/{sb['allpoint']:6.2f}  "
      f"ceil={sc['r11_official']:6.2f}/{sc['r40_official']:6.2f}/{sc['allpoint']:6.2f}  "
      f"gap_allpt={sc['allpoint']-sb['allpoint']:+6.2f}")

w("")
w("=" * 100)
w("ORDER-CEILING GAP — official R11 / official R40 / all-point interpolated AP (same pools+scores)")
w("=" * 100)
hdr = (f"{'detector':10s} | {'baseAP*':>7} {'ceilAP*':>7} {'gapAP*':>7} | "
       f"{'gapR40':>6} {'gapR11':>6}   (AP* = all-point)")
w(hdr); w("-" * len(hdr))
gex = []
for r_ in rows:
    ge = r_["c"]["allpoint"] - r_["b"]["allpoint"]; gex.append(ge)
    w(f"{r_['name']:10s} | {r_['b']['allpoint']:>7.2f} {r_['c']['allpoint']:>7.2f} {ge:>+7.2f} | "
      f"{r_['c']['r40_official']-r_['b']['r40_official']:>+6.2f} "
      f"{r_['c']['r11_official']-r_['b']['r11_official']:>+6.2f}")
gex = np.array(gex); bex = np.array([r_["b"]["allpoint"] for r_ in rows])
w("")
w(f"all-point gap: range {gex.min():+.2f}..{gex.max():+.2f}  median {np.median(gex):+.2f}  sd {gex.std():.2f}")
w(f"spearman(gap_allpt, base_allpt) over 12 = {spearmanr(bex, gex).correlation:+.3f}  "
  f"(descriptive; 12 non-independent detectors)")
w(f"gap_allpt > 0: {(gex > 0).sum()}/12;  >= +6: {(gex >= 6).sum()}/12")
w("")
w("READ (bounded): all-point interpolated AP is the de-quantized gap-size judge for GATE-2.")
w("Interpretation follows the prereg_2b_sweep.md criteria; the spearman line is read only")
w("as 'no clear monotone decrease observed', never as a law. Depth-relatedness and the missing/suppressed split")
w("remain gated on #1 and #5 respectively.")
if os.path.exists(WORK):
    shutil.rmtree(WORK)
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

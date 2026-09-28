"""Produces reports/gap_metric_robustness.txt.

Gap metric robustness: does the stable +12-15 rank gap survive the metric?

Known issue (seen on 3 detectors): all 12 R40 order-ceilings sit exactly on the
2.5-AP grid (recall-quantization), and gap_R11 < gap_R40 by ~3-4 AP. This script computes
base and order-ceiling under BOTH R11 (r[2]) and R40 (r[6]) from the same eval calls for
all 12 detectors, so the claim can be stated in the form that survives both summaries.

Also writes the oracle-IoU cache (_bridgecache_<f>.npz: o_act, o_z aligned to the S5 kept
pool rebuilt deterministically) for reuse by the anatomy ladder and the sweep.
Run from the repository root: python tools/decomp/gap_metric_robustness.py
"""
import os, sys, shutil
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti, pool_mask, apply_nms
import evaluator.kitti_eval.kitti_common as kc
from evaluator.kitti_eval.eval import do_eval
from depth_share_bridge import iou_act_and_zstar, DETS, DIAG

val_list = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val_list)
ov07 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7]] * 3)
ov05 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
MO = np.stack([ov07, ov05], 0)[:, :, [0]]
WORK = f"{DIAG}/_gmr"
OUT = out_path("gap_metric_robustness.txt")
outlines = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); outlines.append(s)


def ev2(df, score):
    """(R11 mod, R40 mod) from one eval; also (R40 mod @IoU0.5) for the IoU0.5 column."""
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data"); write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val_list)
    r = do_eval(GT, kc.get_label_annos(d, val_list), [0], MO, compute_aos=False, DIForDIS=True)
    return float(r[2][0, 1, 0]), float(r[6][0, 1, 0]), float(r[6][0, 1, 1])


rows = []
for name, f in DETS:
    df = pd.read_csv(dump_path(f)).reset_index(drop=True)
    pre = df[pool_mask(df, "thr0.2")].copy()
    keep = apply_nms(pre, pre["V"].values, 0.5)
    kept = pre[keep].copy().reset_index(drop=True)

    cache = f"{DIAG}/_bridgecache_{f}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        o_act, o_z = z["o_act"], z["o_z"]
        assert len(o_act) == len(kept), f"cache misaligned for {f}"
    else:
        o_act, o_z = iou_act_and_zstar(kept)
        np.savez_compressed(cache, o_act=o_act, o_z=o_z)
    V = kept["V"].values.astype(float)

    b11, b40, b40lo = ev2(kept, V)
    c11, c40, c40lo = ev2(kept, o_act)
    rows.append(dict(name=name, b11=b11, b40=b40, c11=c11, c40=c40,
                     g11=c11 - b11, g40=c40 - b40, b40lo=b40lo, c40lo=c40lo))
    w(f"[done] {name:10s} base R11/R40={b11:6.2f}/{b40:6.2f}  ceil R11/R40={c11:6.2f}/{c40:6.2f}  "
      f"gap R11={c11-b11:+6.2f} R40={c40-b40:+6.2f}  |  IoU0.5 R40 base/ceil={b40lo:.2f}/{c40lo:.2f}")

w("")
w("=" * 100)
w("GAP METRIC ROBUSTNESS — order-ceiling gap under R11 vs R40 (same pools, same oracle scores)")
w("=" * 100)
hdr = (f"{'detector':10s} {'baseR11':>8} {'baseR40':>8} {'ceilR11':>8} {'ceilR40':>8} "
       f"{'gapR11':>7} {'gapR40':>7} {'shrink':>7} | {'gap@IoU0.5(R40)':>15}")
w(hdr); w("-" * len(hdr))
for r in rows:
    w(f"{r['name']:10s} {r['b11']:>8.2f} {r['b40']:>8.2f} {r['c11']:>8.2f} {r['c40']:>8.2f} "
      f"{r['g11']:>+7.2f} {r['g40']:>+7.2f} {r['g11']-r['g40']:>+7.2f} | {r['c40lo']-r['b40lo']:>+15.2f}")

g11 = np.array([r["g11"] for r in rows]); g40 = np.array([r["g40"] for r in rows])
w("")
w(f"R40 gap: range {g40.min():+.2f}..{g40.max():+.2f}  sd={g40.std():.2f}")
w(f"R11 gap: range {g11.min():+.2f}..{g11.max():+.2f}  sd={g11.std():.2f}")
from scipy.stats import spearmanr
w(f"spearman(gapR11, gapR40) across 12 = {spearmanr(g11, g40).correlation:+.3f}")
w(f"spearman(gapR11 vs baseR11)        = {spearmanr([r['b11'] for r in rows], g11).correlation:+.3f}")
w("")
w("READ: the C1 claim must be the version that survives BOTH columns. If R11 gaps stay")
w("large and ~stable (even if smaller than R40), the qualitative 'large, non-closing")
w("ordering headroom' stands and the R40 number is reported as an R40 diagnostic with the")
w("quantization caveat. Gap@IoU0.5 column addresses the metric-tautology concern.")

if os.path.exists(WORK):
    shutil.rmtree(WORK)
open(OUT, "w").write("\n".join(outlines) + "\n")
w(f"[written] {OUT}")

#!/usr/bin/env python
"""Ported from mono3d_crossdataset/tools/gap_exact_orig.py for the public release. Computation
unchanged. Produces reports_orig/gap_exact_orig.txt.

The paper's gap_exact machinery (tools/decomp/gap_exact.py), verbatim, applied to the
ORIGINAL-ENV MonoFlex/MonoGround dumps (dump stems monoflex_orig / monoground_orig).
Also writes the S5-pool oracle-IoU cache <stem>_bridgecache.npz reused by bridge_orig.py and
gap_metric_robustness_orig.py.
Run from the repository root: python tools/orig/gap_exact_orig.py
"""
import os, sys, shutil, datetime, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
from tools.orig._orig_common import ORIG
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti, pool_mask, apply_nms
import evaluator.kitti_eval.kitti_common as kc
from evaluator.kitti_eval.eval import do_eval
from exact_ap import ap_summaries
from depth_share_bridge import iou_act_and_zstar

DETS_ORIG = [("MonoFlex*", dump_path("monoflex_orig")),
             ("MonoGround*", dump_path("monoground_orig"))]
val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
ov07 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7]] * 3)
ov05 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
MO = np.stack([ov07, ov05], 0)[:, :, [0]]
WORK = f"{ORIG}/_gapexact_orig"
OUT = out_path("gap_exact_orig.txt")
out = []

def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

def annos_for(df, score):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    return kc.get_label_annos(d, val)

w(f"# gap_exact_orig run {datetime.datetime.now().isoformat(timespec='seconds')}")
w(f"# same machinery as tools/decomp/gap_exact.py; pools S5 (cls>=0.2 + 2D-NMS@0.5 by V)")
w(f"# inputs: ORIGINAL-ENV dumps (torch1.4 rebuild, published numbers reproduced)")
w("")

for name, path in DETS_ORIG:
    df = pd.read_csv(path).reset_index(drop=True)
    md5 = hashlib.md5(open(path, "rb").read(1 << 20)).hexdigest()[:8]
    pre = df[pool_mask(df, "thr0.2")].copy()
    keep = apply_nms(pre, pre["V"].values, 0.5)
    kept = pre[keep].copy().reset_index(drop=True)
    cache = os.path.join(ORIG, os.path.basename(path).replace("_val.csv", "_bridgecache.npz"))
    if os.path.exists(cache):
        z = np.load(cache); o_act = z["o_act"]
        assert len(o_act) == len(kept), f"cache misaligned for {name}"
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
    w(f"[done] {name:11s} dump@{md5} GATE-OK | base R11/R40/allpt="
      f"{sb['r11_official']:6.2f}/{sb['r40_official']:6.2f}/{sb['allpoint']:6.2f}  "
      f"ceil={sc['r11_official']:6.2f}/{sc['r40_official']:6.2f}/{sc['allpoint']:6.2f}  "
      f"gap_allpt={sc['allpoint']-sb['allpoint']:+6.2f}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(out) + "\n")
print("saved:", OUT)

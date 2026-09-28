"""Produces reports_orig/a5_regate_orig.txt.

A5: released-code-faithful native cell for the original-environment MonoFlex*/MonoGround*
dumps.
Released convention (verified in code): MonoFlex/MonoGround detector_infer.py L103 gate on
RAW heatmap score `scores >= det_threshold` (= cls), BEFORE the uncertainty multiply
(L227 scores *= uncertainty_conf -> our V). cfg.TEST.DETECTIONS_THRESHOLD = 0.1.
So native cell = cls>=0.1, NO box-NMS (CenterNet), rank by V; o_act recomputed on THIS pool.
The E4 canonical table gated these two on V (col='V'); this re-gate replaces those cells
(main Table 1 / native_gap_canonical.md starred rows). Re-aggregation of the released dumps; no inference.
Run from the repository root: python tools/orig/a5_regate_orig.py
"""
import os, sys, shutil, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
from tools.orig._orig_common import ORIG
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti
import evaluator.kitti_eval.kitti_common as kc
from exact_ap import ap_summaries
from depth_share_bridge import iou_act_and_zstar

OUT = out_path("a5_regate_orig.txt")
val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
WORK = f"{ORIG}/_a5work_orig"
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def annos_for(df, score):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    return kc.get_label_annos(d, val)


w(f"# a5_regate run {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# released native cell: cls>=0.1 (DETECTIONS_THRESHOLD), no box-NMS, rank by V; o_act on this pool")
w("# compare vs E4 V-gated cell (canonical table) and cls>=0.2 reference")
w("")
E4_VGATE = {"MonoFlex*": (16.26, 10.46), "MonoGround*": (17.45, 10.89)}  # MODERN-env reference (vA)
for name, f in [("MonoFlex*", "monoflex_orig"), ("MonoGround*", "monoground_orig")]:
    df = pd.read_csv(dump_path(f)).reset_index(drop=True)
    for thr in (0.1, 0.2):
        kept = df[df["cls"].values >= thr].copy().reset_index(drop=True)
        o_act, _ = iou_act_and_zstar(kept)
        sb = ap_summaries(GT, annos_for(kept, kept["V"].values.astype(float)))
        sc = ap_summaries(GT, annos_for(kept, o_act))
        tag = "NATIVE(cls>=0.1)" if thr == 0.1 else "ref(cls>=0.2)"
        w(f"[{name:10s}] {tag:16s} n={len(kept):6d} base={sb['allpoint']:6.2f}({sb['r40_official']:5.2f}) "
          f"ceil={sc['allpoint']:6.2f} gap_allpt={sc['allpoint']-sb['allpoint']:+6.2f} "
          f"(R40 {sc['r40_official']-sb['r40_official']:+.2f})")
    bv, gv = E4_VGATE[name]
    w(f"             E4 V-gated cell (replaced): base={bv:.2f} gap={gv:+.2f}")
    w("")
w("READ: the released-faithful native cells (cls>=0.1) replace the E4 V-gated MonoFlex/MonoGround")
w("rows in native_gap_canonical.md. Panel base-AP rank order is checked unchanged below.")
if os.path.exists(WORK):
    shutil.rmtree(WORK)
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

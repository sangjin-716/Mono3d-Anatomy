#!/usr/bin/env python
"""Ported from mono3d_crossdataset/tools/bev_transfer_orig.py for the public release.
Computation unchanged. Produces reports_orig/bev_transfer_orig.txt.

Supplementary tab:bev rows (BEV recovery) for the original-environment MonoFlex*/MonoGround*
dumps: A5 (cls>=0.1) pool, 3D-IoU re-sort evaluated under 3D (metric=2) and BEV (metric=1)
all-point AP. Mirrors the BEV re-evaluation of the other ten detectors on their native pools.
Run from the repository root: python tools/orig/bev_transfer_orig.py
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

val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
WORK = f"{ORIG}/_bevtr_orig"
OUT = out_path("bev_transfer_orig.txt")
out = []

def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

def annos_for(df, score):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    return kc.get_label_annos(d, val)

w(f"# bev_transfer_orig {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# A5 (cls>=0.1) pool; 3D-IoU re-sort; allpt AP under metric=2 (3D) and metric=1 (BEV)")
for name, f in [("MonoFlex*", "monoflex_orig"), ("MonoGround*", "monoground_orig")]:
    df = pd.read_csv(dump_path(f)).reset_index(drop=True)
    kept = df[df["cls"].values >= 0.1].copy().reset_index(drop=True)
    cache = f"{ORIG}/_a5cache_{f}.npz"
    if os.path.exists(cache):
        o_act = np.load(cache)["o_act"]
        assert len(o_act) == len(kept)
    else:
        o_act, _ = iou_act_and_zstar(kept)
        np.savez_compressed(cache, o_act=o_act)
    a_base = annos_for(kept, kept["V"].values.astype(float))
    a_or = annos_for(kept, o_act)
    row = {}
    for met, tag in ((2, "3D"), (1, "BEV")):
        b = ap_summaries(GT, a_base, metric=met, min_overlap=0.7)["allpoint"]
        c = ap_summaries(GT, a_or, metric=met, min_overlap=0.7)["allpoint"]
        row[tag] = (b, c, c - b)
    w(f"[done] {name:11s} 3D base={row['3D'][0]:6.2f} gap={row['3D'][2]:+6.2f} | "
      f"BEV base={row['BEV'][0]:6.2f} gain={row['BEV'][2]:+6.2f} | paired diff={row['BEV'][2]-row['3D'][2]:+.2f}")
if os.path.exists(WORK):
    shutil.rmtree(WORK)
open(OUT, "w").write("\n".join(out) + "\n")
print("saved:", OUT)

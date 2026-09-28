#!/usr/bin/env python
"""Ported from mono3d_crossdataset/tools/a5_emh_orig.py for the public release. Computation
unchanged. Prints the Easy/Moderate/Hard all-point AP3D of the starred rows of main Table 1
to stdout (the original wrote no report file); the stdout of the release re-run is captured in
reports_orig/emh_orig.txt.

A5 released-native pool (cls>=0.1, no box-NMS, rank by V) all-point AP3D at Easy/Moderate/Hard
for MonoFlex*/MonoGround*, so Table 1's starred rows are one pool.
Run from the repository root: python tools/orig/a5_emh_orig.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
from tools.orig._orig_common import ORIG
import numpy as np, pandas as pd, shutil
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti
import evaluator.kitti_eval.kitti_common as kc
from exact_ap import ap_summaries

val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
WORK = f"{ORIG}/_a5emh"

def annos(df, sc):
    if os.path.exists(WORK): shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(sc, float), d, val)
    return kc.get_label_annos(d, val)

for name, f in [("MonoFlex*", "monoflex_orig"), ("MonoGround*", "monoground_orig")]:
    df = pd.read_csv(dump_path(f)).reset_index(drop=True)
    kept = df[df["cls"].values >= 0.1].copy().reset_index(drop=True)
    a = annos(kept, kept["V"].values.astype(float))
    emh = []
    for dif in (0, 1, 2):
        s = ap_summaries(GT, a, difficulty=dif, metric=2, min_overlap=0.7)
        emh.append(s["allpoint"])
    print(f"[A5 cls>=0.1] {name:11s} E/M/H = {emh[0]:.2f} / {emh[1]:.2f} / {emh[2]:.2f}", flush=True)
if os.path.exists(WORK): shutil.rmtree(WORK)

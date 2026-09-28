#!/usr/bin/env python
"""Ported from mono3d_crossdataset/tools/make_ladder_preds_orig.py for the public release.
Computation unchanged. Produces the ladder prediction dirs consumed by bootstrap_orig.py --ladder
(no report of its own).

S5 (cls>=0.2 + 2D-NMS@0.5 by V) prediction dirs for the ORIGINAL-ENV MonoFlex/MonoGround dumps,
in the exact format of the main-panel ladder dirs <CACHE_DIR>/decomp/_ladder_preds/<f>/data
(full val, native V scores), written to <CACHE_DIR>/orig/_ladder_preds_orig/<f>/data.
--with-main (release addition) also writes the four main-panel neighbours that the starred
ladder steps compare against (monodle, gupnet, deviant, monocon), with the same code; the paper
built those main-panel dirs with the identical pool/NMS/write_kitti recipe.
Run from the repository root: python tools/orig/make_ladder_preds_orig.py [--with-main]
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
from tools.orig._orig_common import ORIG
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti, pool_mask, apply_nms

val = [int(x) for x in open(arc.VAL_LIST).read().split()]
STEMS = ("monoflex_orig", "monoground_orig")
if "--with-main" in sys.argv[1:]:
    STEMS += ("monodle", "gupnet", "deviant", "monocon")
for f in STEMS:
    df = pd.read_csv(dump_path(f)).reset_index(drop=True)
    pre = df[pool_mask(df, "thr0.2")].copy()
    keep = apply_nms(pre, pre["V"].values, 0.5)
    kept = pre[keep].copy().reset_index(drop=True)
    dd = f"{ORIG}/_ladder_preds_orig/{f}/data"
    write_kitti(kept, kept["V"].values.astype(float), dd, val)
    n = len(os.listdir(dd))
    print(f"[done] {f}: {len(kept)} boxes -> {dd} ({n} files)", flush=True)
    assert n == len(val), f"file count {n} != {len(val)}"

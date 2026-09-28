#!/bin/bash
# Ported from camera-ready checks/waymo_gate/run_gate.sh for the public release.
# run_gate.sh <detname> <preddir> <pdset> <script> <outtag>
# Runs the OFFICIAL Waymo detection evaluator (waymo_open_dataset metrics ops, TF 2.11 / wod 1.6.1)
# via DEVIANT's own data/waymo/waymo_eval{,_0_5}.py, path plumbing patched only
# (waymo_eval_patched.py for IoU 0.7, waymo_eval_0_5_patched.py for IoU 0.5).
# Writes crossbench/waymo_gate/<outtag>.txt in the re-run output directory; gate_deltas_v2.py
# reads gate_<det>_fullval_iou07.txt / _iou05.txt from there. Example (IoU 0.7, GUPNet):
#   PYTHON=<waymo env python> bash tools/crossbench/run_gate.sh gupnet <WAYMO_ROOT>/predictions/gupnet \
#     <UPSTREAM_ROOT>/DEVIANT/data/waymo/ImageSets/val.txt tools/crossbench/waymo_eval_patched.py \
#     gate_gupnet_fullval_iou07
# MonoRCNN++ uses <WAYMO_ROOT>/predictions/monorcnnpp_kitti (written by waymo_core_analysis.py).
# GT: <WAYMO_ROOT>/waymo_gtorg holds validation_org/<segment>/label_0/<id>.txt, a byte-identical
# re-layout of waymo_v2_to_kitti_gt.py's label/%06d.txt (line i of val_org.txt = "<segment> <id>"
# for file %06d.txt of index i).
set -u
source "$(dirname "$0")/_paths.sh"
DET=$1; PDDIR=$2; PDSET=$3; SCRIPT=$4; TAG=$5
OUT=$XB_OUT/waymo_gate
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=""
export WEVAL_PD_SET=$PDSET
export WEVAL_GT_SET=$UPSTREAM_ROOT/DEVIANT/data/waymo/ImageSets/val_org.txt
export WEVAL_PD_DIR=$PDDIR
export WEVAL_GT_DIR=$WAYMO_ROOT/waymo_gtorg
echo "=== $TAG ==="
echo "det=$DET preddir=$PDDIR pdset=$PDSET script=$SCRIPT"
date -Is
$PY -u "$SCRIPT" 2>&1 | tee "$OUT/${TAG}.txt"
date -Is
echo "EXIT=$?"

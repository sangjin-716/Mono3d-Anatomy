#!/bin/bash
# Produces the CenterNet (CenterTrack e140) nuScenes oracle cell (logs_ct_oracle2.log, the
# [iou07] line is the paper's base / re-sort cell): ct_to_oracle_pkl.py on CenterTrack's
# results_nuscenes_det.json, then nusc_oracle.py.
# Set PYTHON to the interpreter of the mmdet3d 1.4.0 environment (setup_nuscenes_stage1.sh).
source "$(dirname "$0")/_paths.sh"
LOG=$XB_OUT/logs_ct_oracle2.log
echo "=== chain2 start $(date) ===" > "$LOG"
# the EPro-PnP-Det annotation converter is memory hungry: wait until it has finished
n=0
while pgrep -f "tools/data_converter/nuscenes_converter" > /dev/null; do
  sleep 120; n=$((n+1)); [ $n -gt 60 ] && break
done
echo "--- converter start $(date) ---" >> "$LOG"
$PY "$XB_HERE/ct_to_oracle_pkl.py" "$UPSTREAM_ROOT/CenterTrack/exp/ddd/nusc_e140_full/results_nuscenes_det.json" "$NUSC_WORK/preds_centertrack_full.pkl" >> "$LOG" 2>&1
if [ ! -f "$NUSC_WORK/preds_centertrack_full.pkl" ]; then echo "CT_CONVERT_FAIL" >> "$LOG"; exit 1; fi
echo "--- oracle start $(date) ---" >> "$LOG"
$PY "$XB_HERE/nusc_oracle.py" "$NUSC_WORK/preds_centertrack_full.pkl" --tag centertrack_full >> "$LOG" 2>&1
echo CT_ORACLE2_DONE >> "$LOG"

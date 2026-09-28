#!/bin/bash
# Ported from mono3d_crossdataset/tools/night_queue.sh for the public release.
# Serialises the memory-heavy jobs: waits for run_ib_gates.sh and run_ct_oracle_chain2.sh, then
# (1) runs waymo_core_analysis.py on the full Waymo val set -> the Waymo base / re-sort cells
#     (crossbench/waymo/core_table_full.json), and
# (2) runs the official nuScenes eval on CenterTrack's results json -> the CenterNet mAP gate
#     ("[gate] centertrack mAP=... NDS=..." line).
# PYTHON_KITTI: interpreter with this repository's evaluator requirements (numpy, numba, scipy);
# PYTHON: interpreter of the mmdet3d 1.4.0 environment (nuscenes-devkit).
source "$(dirname "$0")/_paths.sh"
LOG=$XB_OUT/logs_night_queue.log
echo "=== queue start $(date) ===" > "$LOG"
n=0
until grep -q "IB_GATES_DONE" "$XB_OUT/logs_ib_gates.log" 2>/dev/null && \
      grep -qE "CT_ORACLE2_DONE|CT_CONVERT_FAIL" "$XB_OUT/logs_ct_oracle2.log" 2>/dev/null; do
  sleep 180; n=$((n+1)); [ $n -gt 100 ] && { echo QUEUE_TIMEOUT >> "$LOG"; exit 1; }
done
echo "--- waymo full3 $(date) ---" >> "$LOG"
cd "$REPO" && ${PYTHON_KITTI:-python} tools/crossbench/waymo_core_analysis.py >> "$LOG" 2>&1
echo "--- CT official eval retry $(date) ---" >> "$LOG"
$PY "$XB_HERE/nusc_official_eval.py" \
  "$UPSTREAM_ROOT/CenterTrack/exp/ddd/nusc_e140_full/results_nuscenes_det.json" centertrack >> "$LOG" 2>&1
echo NIGHT_QUEUE_DONE >> "$LOG"

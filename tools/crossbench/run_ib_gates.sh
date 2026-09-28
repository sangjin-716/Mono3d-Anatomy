#!/bin/bash
# Ported from mono3d_crossdataset/tools/run_ib_gates.sh for the public release.
# FCOS3D/PGD: per-camera pkl -> global json (campkl_to_nusc_json.py) -> official devkit eval,
# run one after the other to bound memory.
# NOT the source of the paper's FCOS3D/PGD gate values: this hand-written conversion
# under-reproduces the published mAP/NDS. The gate values come from run_repair.sh (official
# mmdet3d formatter). The pgd_global.json written here is only the V0 rung of pgd_bisect_eval.py.
# Set PYTHON to the interpreter of the mmdet3d 1.4.0 environment.
source "$(dirname "$0")/_paths.sh"
LOG=$XB_OUT/logs_ib_gates.log
echo "=== IB gates start $(date) ===" > "$LOG"
for M in fcos3d pgd; do
  echo "--- $M convert $(date) ---" >> "$LOG"
  $PY "$XB_HERE/campkl_to_nusc_json.py" "$NUSC_WORK/preds_${M}_full.pkl" "$NUSC_WORK/${M}_global.json" >> "$LOG" 2>&1
  [ -f "$NUSC_WORK/${M}_global.json" ] || { echo "${M}_CONVERT_FAIL" >> "$LOG"; continue; }
  echo "--- $M official eval $(date) ---" >> "$LOG"
  $PY "$XB_HERE/nusc_official_eval.py" "$NUSC_WORK/${M}_global.json" $M >> "$LOG" 2>&1
done
echo IB_GATES_DONE >> "$LOG"

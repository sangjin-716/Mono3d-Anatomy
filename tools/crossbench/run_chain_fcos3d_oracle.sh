#!/bin/bash
# Produces the EPro-PnP-Det and FCOS3D nuScenes oracle cells (chain.log; the [iou07] lines are
# the paper's base / re-sort cells):
#   nusc submission json (global) --[xds_nusc_json_to_oracle_pkl_FIXED.py]--> per-camera oracle pkl
#                                 --[nusc_oracle.py]--> base / resort / AP* / coverage
# Two control runs of the original chain on the superseded hand-written conversion
# (campkl_to_nusc_json.py output and the raw DumpResults pkl) are omitted here; they do not
# enter the paper. A census step (pool_census.py, run by hand originally) is appended.
# Set PYTHON to the interpreter of the mmdet3d 1.4.0 environment.
set -u
source "$(dirname "$0")/_paths.sh"
W=$NUSC_WORK/xdsfix_fcos3d_oracle
L=$XB_OUT/xdsfix_fcos3d_oracle
mkdir -p "$W" "$L"
ORACLE=$XB_HERE/nusc_oracle.py
CONV=$XB_HERE/xds_nusc_json_to_oracle_pkl_FIXED.py
LOG=$L/chain.log
echo "=== chain start $(date) ===" > "$LOG"

run_one () {  # $1=tag  $2=input json  $3=out pkl
  echo "--- [$1] convert start $(date) ---" >> "$LOG"
  $PY "$CONV" "$2" "$3" >> "$LOG" 2>&1
  if [ ! -f "$3" ]; then echo "[$1] CONVERT_FAIL" >> "$LOG"; return 1; fi
  echo "--- [$1] convert done $(date) ($(du -h "$3" | cut -f1)) ---" >> "$LOG"
  echo "--- [$1] alignment check ---" >> "$LOG"
  $PY "$XB_HERE/align_check.py" "$3" "$1" >> "$LOG" 2>&1
  echo "--- [$1] oracle start $(date) ---" >> "$LOG"
  $PY "$ORACLE" "$3" --tag "$1" >> "$LOG" 2>&1
  echo "--- [$1] oracle done $(date) ---" >> "$LOG"
  echo "[$1] DONE" >> "$LOG"
}

# ---- STEP 1: EPro-PnP-Det (its own formatter json; official NDS within 0.07 of published) ----
# expected: iou07 base=8.06 resort=14.18 (gain +6.13)
run_one GATE_epropnp "$NUSC_WORK/epropnp_basic/results_nusc.json" "$W/preds_epropnp_GATE.pkl"

# ---- STEP 2: FCOS3D (official mmdet3d formatter json, mAP 32.13 / NDS 39.48) ----
# expected: iou07 base=4.23 resort=12.01 (gain +7.78)
run_one FCOS3D_REPAIRED "$NUSC_WORK/xds_repair/official_reformat_fcos3d/pred_instances_3d/results_nusc.json" "$W/preds_fcos3d_repaired.pkl"

# ---- census of the FCOS3D pool (boxes per image) ----
$PY "$XB_HERE/pool_census.py" "$W/preds_fcos3d_repaired.pkl" FCOS3D_REPAIRED > "$L/census_repaired.log" 2>&1

echo "=== chain end $(date) ===" >> "$LOG"
echo CHAIN_ALL_DONE >> "$LOG"

#!/bin/bash
# Produces the PGD nuScenes oracle cell and re-derives the EPro-PnP-Det cell (chain.log; the
# [iou07] lines are the paper's base / re-sort cells):
#   nusc json (global) --[xds_nusc_json_to_oracle_pkl_FIXED.py]--> per-camera oracle pkl
#                      --[nusc_oracle.py]--> base / resort / AP* / coverage
# One control run of the original chain on the superseded hand-written conversion
# (campkl_to_nusc_json.py output) is omitted here; it does not enter the paper.
# Run run_census.sh afterwards for the boxes-per-image census.
# Set PYTHON to the interpreter of the mmdet3d 1.4.0 environment.
set -u
source "$(dirname "$0")/_paths.sh"
W=$NUSC_WORK/xdsfix_pgd_oracle
L=$XB_OUT/xdsfix_pgd_oracle
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

# ---- STEP 2: PGD (official mmdet3d formatter json, mAP 35.84 / NDS 42.89) ----
# expected: iou07 base=6.33 resort=13.07 (gain +6.75)
run_one PGD_REPAIRED "$NUSC_WORK/xds_repair/official_pgd/pred_instances_3d/results_nusc.json" "$W/preds_pgd_repaired.pkl"

echo "=== chain end $(date) ===" >> "$LOG"
echo CHAIN_ALL_DONE >> "$LOG"

#!/bin/bash
# Ported from camera-ready checks/xdsfix_pgd_oracle/run_census.sh for the public release.
# Produces census.log: height-alignment checks and the car-prediction census (boxes per image,
# per-depth-bin coverage) of the PGD pool, the raw PGD DumpResults pool, and the EPro-PnP-Det
# pool. Run after run_chain_pgd_oracle.sh. Set PYTHON to the mmdet3d 1.4.0 interpreter.
set -u
source "$(dirname "$0")/_paths.sh"
W=$NUSC_WORK/xdsfix_pgd_oracle
L=$XB_OUT/xdsfix_pgd_oracle
mkdir -p "$L"
LOG=$L/census.log
echo "=== census start $(date) ===" > "$LOG"
echo "--- align (score>0.05, more pairs) repaired ---" >> "$LOG"
$PY "$XB_HERE/align_check2.py" "$W/preds_pgd_repaired.pkl" PGD_REPAIRED >> "$LOG" 2>&1
echo "--- align (score>0.05) OLD pkl (direct DumpResults) ---" >> "$LOG"
$PY "$XB_HERE/align_check2.py" "$NUSC_WORK/preds_pgd_full.pkl" PGD_OLD_PKL >> "$LOG" 2>&1
echo "--- census REPAIRED $(date) ---" >> "$LOG"
$PY "$XB_HERE/pred_census.py" "$W/preds_pgd_repaired.pkl" PGD_REPAIRED >> "$LOG" 2>&1
echo "--- census OLD (raw DumpResults pkl) $(date) ---" >> "$LOG"
$PY "$XB_HERE/pred_census.py" "$NUSC_WORK/preds_pgd_full.pkl" PGD_OLD >> "$LOG" 2>&1
echo "--- census EPro-PnP $(date) ---" >> "$LOG"
$PY "$XB_HERE/pred_census.py" "$W/preds_epropnp_GATE.pkl" EPROPNP >> "$LOG" 2>&1
echo "=== census end $(date) ===" >> "$LOG"
echo CENSUS_DONE >> "$LOG"

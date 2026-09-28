#!/bin/bash
# Ported from mono3d_crossdataset/tools/run_ct_oracle_chain.sh for the public release.
# CenterNet (CenterTrack e140) oracle chain, first version: wait for the CenterTrack inference
# output, convert it to a per-camera oracle pkl (ct_to_oracle_pkl.py), run nusc_oracle.py.
# The paper's CenterNet cell comes from run_ct_oracle_chain2.sh (same two steps).
# Set PYTHON to the interpreter of the mmdet3d 1.4.0 environment (setup_nuscenes_stage1.sh).
source "$(dirname "$0")/_paths.sh"
RES=$UPSTREAM_ROOT/CenterTrack/exp/ddd/nusc_e140_full/results_nuscenes_det.json
LOG=$XB_OUT/logs_ct_convert_oracle.log
n=0
while [ ! -f "$RES" ]; do sleep 300; n=$((n+1)); [ $n -gt 144 ] && exit 1; done
sleep 60  # let the writer finish the file
$PY "$XB_HERE/ct_to_oracle_pkl.py" "$RES" "$NUSC_WORK/preds_centertrack_full.pkl" > "$LOG" 2>&1
$PY "$XB_HERE/nusc_oracle.py" "$NUSC_WORK/preds_centertrack_full.pkl" --tag centertrack_full >> "$LOG" 2>&1
echo CT_ORACLE_CHAIN_DONE >> "$LOG"

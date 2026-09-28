#!/bin/bash
# Ported from mono3d_crossdataset/tools/run_epropnp_infer.sh for the public release.
# EPro-PnP-Det val inference (format-only) -> submission json <cache>/crossbench/nusc/
# epropnp_basic/results_nusc.json. That json is scored by xds_nusc_official_eval.py (NDS gate)
# and converted by xds_nusc_json_to_oracle_pkl_FIXED.py for the oracle cell.
# CUDA_HOME: a CUDA 11 toolkit matching the epropnp env (the original run used CUDA 11.0).
# NOTE: the last two lines are the original oracle step with ct_to_oracle_pkl.py; see the note in
#   run_epropnp_chain.sh -- the paper's cell uses xds_nusc_json_to_oracle_pkl_FIXED.py instead.
source "$(dirname "$0")/_paths.sh"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.0}
export PATH=$CUDA_HOME/bin:$PATH
R=${EPRO_REPO:-$UPSTREAM_ROOT/EPro-PnP/EPro-PnP-Det}
LOG=$XB_OUT/logs_epropnp_infer.log
PYE=${PYTHON_EPRO:-python}
echo "=== EPro infer $(date) ===" > "$LOG"
cd "$R"
PYTHONPATH=$R $PYE test.py configs/epropnp_det_basic.py \
  "$CKPT_DIR/nuscenes/epropnp/EPro-PnP-Det/epropnp_det_basic.pth" \
  --val-set --format-only --eval-options jsonfile_prefix=$NUSC_WORK/epropnp_basic >> "$LOG" 2>&1
RJ=$(ls $NUSC_WORK/epropnp_basic*.json 2>/dev/null | head -1)
[ -z "$RJ" ] && { echo "INFER OUTPUT MISSING" >> "$LOG"; exit 1; }
echo "--- oracle $(date) $RJ ---" >> "$LOG"
$PY "$XB_HERE/ct_to_oracle_pkl.py" "$RJ" "$NUSC_WORK/preds_epropnp_full.pkl" >> "$LOG" 2>&1
$PY "$XB_HERE/nusc_oracle.py" "$NUSC_WORK/preds_epropnp_full.pkl" --tag epropnp_full >> "$LOG" 2>&1
echo EPROPNP_INFER_DONE >> "$LOG"

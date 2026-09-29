#!/bin/bash
# EPro-PnP-Det: wait for the environment (setup_epropnp.sh), convert the nuScenes annotations
# with the release's converter, run val inference in format-only mode (submission json), then
# convert and score with the oracle.
# EPRO_REPO: a copy of EPro-PnP-Det from https://github.com/tjiiv-cprg/EPro-PnP, with data/nuscenes
#   linked to NUSC_ROOT. Its tools/data_converter/nuscenes_converter.py was patched to tolerate a
#   camera-only nuScenes copy: a missing LIDAR_TOP file is skipped (empty point set), which only
#   affects training-time object-coordinate maps, not val inference.
# PYTHON_EPRO: interpreter of the "epropnp" env; PYTHON: interpreter of the mmdet3d 1.4.0 env.
# NOTE: the oracle step below is the original one and uses ct_to_oracle_pkl.py. That converter's
#   height convention only suits CenterTrack's json, so for EPro-PnP-Det it yields a misplaced
#   pool. The EPro-PnP-Det cell in the paper was produced from the same submission json with
#   xds_nusc_json_to_oracle_pkl_FIXED.py (STEP 1 of run_chain_fcos3d_oracle.sh and
#   run_chain_pgd_oracle.sh).
source "$(dirname "$0")/_paths.sh"
R=${EPRO_REPO:-$UPSTREAM_ROOT/EPro-PnP/EPro-PnP-Det}
LOG=$XB_OUT/logs_epropnp_chain.log
echo "=== chain start $(date) ===" > "$LOG"

n=0
while ! grep -q "ENV_OK" "$XB_OUT/logs_epropnp_setup.log" 2>/dev/null; do
  sleep 60; n=$((n+1)); [ $n -gt 90 ] && { echo "ENV TIMEOUT" >> "$LOG"; exit 1; }
done
PYE=${PYTHON_EPRO:-python}
echo "--- env ready $(date) ---" >> "$LOG"

cd "$R"
if [ ! -f data/nuscenes/nuscenes_annotations_val.pkl ]; then
  echo "--- converter $(date) ---" >> "$LOG"
  PYTHONPATH=$R $PYE tools/data_converter/nuscenes_converter.py data/nuscenes --version v1.0-trainval >> "$LOG" 2>&1
fi
[ -f data/nuscenes/nuscenes_annotations_val.pkl ] || { echo "CONVERTER FAIL" >> "$LOG"; exit 1; }

echo "--- inference $(date) ---" >> "$LOG"
PYTHONPATH=$R $PYE test.py configs/epropnp_det_basic.py \
  "$CKPT_DIR/nuscenes/epropnp/EPro-PnP-Det/epropnp_det_basic.pth" \
  --val-set --format-only --eval-options jsonfile_prefix=$NUSC_WORK/epropnp_basic >> "$LOG" 2>&1

RJ=$(ls $NUSC_WORK/epropnp_basic*.json 2>/dev/null | head -1)
[ -z "$RJ" ] && { echo "INFERENCE OUTPUT MISSING" >> "$LOG"; exit 1; }
echo "--- oracle $(date) results=$RJ ---" >> "$LOG"
$PY "$XB_HERE/ct_to_oracle_pkl.py" "$RJ" "$NUSC_WORK/preds_epropnp_full.pkl" >> "$LOG" 2>&1
$PY "$XB_HERE/nusc_oracle.py" "$NUSC_WORK/preds_epropnp_full.pkl" --tag epropnp_full >> "$LOG" 2>&1
echo EPROPNP_CHAIN_DONE >> "$LOG"

#!/usr/bin/env bash
# PGD reproduction gate, reformat only: re-formats the existing per-image DumpResults pkl through
# the official mmdet3d NuScenesMetric.format_results path (reformat_official.py), then runs the
# official devkit eval (eval_official.py). No re-inference.
# Outputs under <cache>/crossbench/nusc/xds_repair/: official_pgd/pred_instances_3d/results_nusc.json
# and eval_pgd/metrics_summary.json.
# The FCOS3D gate was run the same way but through nusc_official_reformat.py, which formats and
# evaluates in one step:
#   $PYTHON tools/crossbench/nusc_official_reformat.py \
#       <cache>/crossbench/nusc/preds_fcos3d_full.pkl fcos3d <cache>/crossbench/nusc/xds_repair
# which writes xds_repair/official_reformat_fcos3d/pred_instances_3d/{results_nusc.json,metrics_summary.json}.
# Set PYTHON to the interpreter of the mmdet3d 1.4.0 environment.
set -u
source "$(dirname "$0")/_paths.sh"
W=$NUSC_WORK/xds_repair
mkdir -p "$W"
cd "$UPSTREAM_ROOT/mmdetection3d" || exit 1

for DET in pgd; do
  echo "################ $DET reformat $(date) ################"
  $PY "$XB_HERE/reformat_official.py" $DET "$W/official_$DET" 2>&1 | grep -v 'task/s'
  J=$W/official_$DET/pred_instances_3d/results_nusc.json
  ls -la "$J" || continue
  echo "################ $DET eval $(date) ################"
  $PY "$XB_HERE/eval_official.py" "$J" "$W/eval_$DET" 2>&1 | tail -60
done
echo "################ DONE $(date) ################"

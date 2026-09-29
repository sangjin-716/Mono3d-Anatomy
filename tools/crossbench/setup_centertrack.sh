#!/bin/bash
# CenterTrack (CenterNet e140 nuScenes checkpoint) setup, attempt 1: clone, DCNv2 build,
# nuScenes -> COCO conversion with CenterTrack's own converter. The DCNv2 build needs the
# torch-1.4 / CUDA-10 environment used for the original-environment MonoFlex runs
# (CT_ENV, default "monoflex_orig"). setup_centertrack2.sh and setup_centertrack3.sh are the
# follow-up attempts (submodule init, then a direct DCNv2 clone); the third one completed.
set -e
eval "$(conda shell.bash hook)"
conda activate "${CT_ENV:-monoflex_orig}"
export PATH=$CONDA_PREFIX/bin:$PATH
PYTHON=python
source "$(dirname "$0")/_paths.sh"
CT="$UPSTREAM_ROOT/CenterTrack"

echo "=== [1/3] clone CenterTrack ==="
mkdir -p "$UPSTREAM_ROOT"
[ -d "$CT" ] || git clone -q --depth 1 https://github.com/xingyizhou/CenterTrack "$CT"
cd "$CT"
pip install -q cython pycocotools scikit-learn motmetrics 2>&1 | grep -iE '^ERROR' | head -2 || echo "deps OK"

echo "=== [2/3] build DCNv2 (same recipe as the MonoFlex environment) ==="
DCN=$(find src -maxdepth 4 -type d -iname 'DCNv2' | head -1)
echo "DCNv2: $DCN"
cd "$DCN"
rm -rf build *.so 2>/dev/null || true
CC=gcc CXX=g++ python setup.py build develop 2>&1 | tail -2
python -c "from dcn_v2 import DCN; print('CenterTrack DCNv2 OK')"

echo "=== [3/3] nuScenes -> COCO conversion (CenterTrack converter) ==="
cd "$CT"
mkdir -p data
ln -sfn "$NUSC_ROOT" data/nuscenes
pip install -q nuscenes-devkit==1.0.5 2>&1 | tail -1 || pip install -q nuscenes-devkit 2>&1 | tail -1
python tools/convert_nuScenes.py 2>&1 | tail -5 || echo "conversion failed -- check paths / devkit version"
echo "=== done ==="

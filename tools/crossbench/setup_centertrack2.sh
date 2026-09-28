#!/bin/bash
# Ported from mono3d_crossdataset/tools/setup_centertrack2.sh for the public release.
# CenterTrack setup, attempt 2: initialise the DCNv2 submodule, build it, convert nuScenes.
set -e
eval "$(conda shell.bash hook)"
conda activate "${CT_ENV:-monoflex_orig}"
export PATH=$CONDA_PREFIX/bin:$PATH
PYTHON=python
source "$(dirname "$0")/_paths.sh"
CT="$UPSTREAM_ROOT/CenterTrack"
cd "$CT"

echo "=== [1/2] DCNv2 submodule + build ==="
git submodule update --init src/lib/model/networks/DCNv2 2>&1 | tail -1
cd src/lib/model/networks/DCNv2
rm -rf build *.so 2>/dev/null || true
CC=gcc CXX=g++ python setup.py build develop 2>&1 | tail -2
python -c "import sys; sys.path.insert(0,'.'); from dcn_v2 import DCN; print('DCNv2 build OK')"

echo "=== [2/2] nuScenes -> COCO conversion ==="
cd "$CT/tools"
python convert_nuScenes.py 2>&1 | tail -6
echo "=== outputs ==="
ls "$NUSC_ROOT"/anns* "$NUSC_ROOT"/annotations* 2>/dev/null | head -5
find "$CT/data/nuscenes" -maxdepth 2 -name '*.json' -newer "$CT/.git" 2>/dev/null | head -5

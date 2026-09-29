#!/bin/bash
# CenterTrack setup, attempt 3 (the one that completed): clone DCNv2 directly, build and import
# it, then convert nuScenes to CenterTrack's COCO format (annotations/val.json).
set -e
eval "$(conda shell.bash hook)"
conda activate "${CT_ENV:-monoflex_orig}"
export PATH=$CONDA_PREFIX/bin:$PATH
PYTHON=python
source "$(dirname "$0")/_paths.sh"
CT="$UPSTREAM_ROOT/CenterTrack"
cd "$CT"

echo "=== clone + build DCNv2 directly ==="
[ -d src/lib/model/networks/DCNv2/.git ] || git clone -q --depth 1 https://github.com/CharlesShang/DCNv2 src/lib/model/networks/DCNv2
cd src/lib/model/networks/DCNv2
rm -rf build *.so 2>/dev/null || true
CC=gcc CXX=g++ python setup.py build develop 2>&1 | tail -2
cd "$CT"
python -c "
import sys; sys.path.insert(0,'src/lib/model/networks/DCNv2')
from dcn_v2 import DCN; print('DCNv2 build+import OK')"

echo "=== nuScenes -> COCO conversion ==="
cd tools
python convert_nuScenes.py 2>&1 | tail -6
echo "=== check outputs ==="
find ../data/nuscenes/ -maxdepth 2 -name '*.json' 2>/dev/null | head -5

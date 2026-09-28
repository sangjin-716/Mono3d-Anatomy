#!/bin/bash

# Builds the ORIGINAL MonoFlex/MonoGround environment (python 3.7, torch 1.4.0 + CUDA 10.1,
# DCNv2 compiled against it) used for the panel entries MonoFlex* and MonoGround*, then runs each
# repo's own evaluation of its released checkpoint (reproduction gate G1).
#
# Usage: UPSTREAM_ROOT=<dir with MonoFlex/ and MonoGround/ clones> \
#        MONOFLEX_CKPT=<monoflex released .pth> MONOGROUND_CKPT=<monoground released .pth> \
#        KITTI_LAYOUT=<KITTI in MonoFlex layout: training/{image_2,calib,label_2,ImageSets}> \
#        OUT=<output dir> bash adapters/build_monoflex_orig_env.sh
# Requires conda on PATH. The original-environment copies are made next to the clones
# (MonoFlex_orig/, MonoGround_orig/) so the modern-environment clones stay untouched.
set -e
: "${UPSTREAM_ROOT:?}" "${MONOFLEX_CKPT:?}" "${MONOGROUND_CKPT:?}" "${KITTI_LAYOUT:?}" "${OUT:?}"
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== [1/4] original env: python3.7 + torch1.4+cu101 ==="
conda create -n monoflex_orig python=3.7 -y -q
conda activate monoflex_orig
pip install -q torch==1.4.0 torchvision==0.5.0 2>&1 | tail -1
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())"

echo "=== [2/4] CUDA 10.1 toolkit (nvcc for DCNv2) ==="
conda install -y -q -c conda-forge cudatoolkit-dev=10.1 2>&1 | tail -1 || echo "cudatoolkit-dev failed - install a CUDA 10.1 nvcc by other means"
which nvcc && nvcc --version | tail -1 || echo "no nvcc"
export PATH=$CONDA_PREFIX/bin:$PATH

echo "=== [3/4] repo copies + DCNv2 builds ==="
for R in MonoFlex MonoGround; do
  [ -d "$UPSTREAM_ROOT/${R}_orig" ] || cp -r "$UPSTREAM_ROOT/$R" "$UPSTREAM_ROOT/${R}_orig"
  cd "$UPSTREAM_ROOT/${R}_orig"
  pip install -q -r requirements.txt 2>&1 | tail -1 || echo "some requirements failed (continuing)"
  sed -i "s|DATA_DIR = .*|DATA_DIR = \"$KITTI_LAYOUT/\"|" config/paths_catalog.py
  DCN_DIR=$(find . -maxdepth 4 -type d -iname '*dcn*' | head -1)
  echo "DCNv2 at: $DCN_DIR"
  (cd "$DCN_DIR" && rm -rf build *.so 2>/dev/null; CC=gcc CXX=g++ python setup.py build develop 2>&1 | tail -5 \
     && python -c "from dcn_v2 import DCN; print('DCNv2 import OK')")
done

echo "=== [4/4] native evaluation of the released checkpoints (gate G1) ==="
cd "$UPSTREAM_ROOT/MonoFlex_orig"; export PYTHONPATH=$PWD
python tools/plain_train_net.py --config runs/monoflex.yaml --ckpt "$MONOFLEX_CKPT" --eval --output "$OUT/monoflex_orig_eval"
cd "$UPSTREAM_ROOT/MonoGround_orig"; export PYTHONPATH=$PWD
python tools/plain_train_net.py --config runs/monoground.yaml --ckpt "$MONOGROUND_CKPT" --eval --output "$OUT/monoground_orig_eval"

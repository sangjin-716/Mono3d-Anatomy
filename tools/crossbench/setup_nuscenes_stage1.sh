#!/bin/bash
# Ported from mono3d_crossdataset/tools/setup_nuscenes_stage1.sh for the public release.
# nuScenes stage 1: environment + mmdet3d + mono3d info generation (CPU/network only).
# Produces the conda env "nusc_mono" and nuscenes_infos_{train,val}.pkl; the Python scripts in
# this folder expect nuscenes_infos_val.pkl inside NUSC_ROOT, so the info files are written there
# (NUSC_ROOT is the nuScenes v1.0-trainval root, or a writable folder that links to it).
set -e
eval "$(conda shell.bash hook)"

echo "=== [1/5] conda env nusc_mono (py3.10 + torch cu118) ==="
conda create -n nusc_mono python=3.10 -y -q
conda activate nusc_mono
PYTHON=python
source "$(dirname "$0")/_paths.sh"
pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -q -U openmim

echo "=== [2/5] mmdet3d stack (mim) ==="
mim install -q mmengine "mmcv==2.1.0" "mmdet==3.2.0" "mmdet3d==1.4.0"
python -c "import mmdet3d, torch; print('mmdet3d', mmdet3d.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "=== [3/5] mmdet3d repo (for create_data, shallow) ==="
mkdir -p "$UPSTREAM_ROOT"
cd "$UPSTREAM_ROOT"
[ -d mmdetection3d ] || git clone -q --depth 1 https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
mkdir -p data
# link the raw nuScenes data read-only; nothing is written into the raw data folder
[ -L data/nuscenes ] || ln -s "$NUSC_ROOT" data/nuscenes

echo "=== [4/5] checkpoint load smoke test ==="
python - <<PYEOF
import torch
ck = torch.load('$CKPT_DIR/nuscenes/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth', map_location='cpu')
print('FCOS3D ckpt OK, keys:', len(ck['state_dict']))
PYEOF

echo "=== [5/5] mono3d info generation ==="
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir "$NUSC_ROOT" \
  --extra-tag nuscenes --version v1.0-trainval 2>&1 | tail -20
echo "=== done ==="
ls -la "$NUSC_ROOT"/nuscenes_infos_*.pkl

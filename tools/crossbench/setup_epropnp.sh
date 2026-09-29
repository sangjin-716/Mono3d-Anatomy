#!/bin/bash
# Environment for the EPro-PnP-Det release (torch 1.10 + cu113, mmcv-full 1.4.1, mmdet 2.19.1,
# pytorch3d 0.6.1). Produces the conda env "epropnp"; prints ENV_OK when the imports succeed.
set -x
eval "$(conda shell.bash hook)"
conda create -y -n epropnp python=3.7
conda activate epropnp
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.19.1 pyro-ppl==1.6.0 "numba>=0.48.0,<0.55.0" "numpy<1.20.0" nuscenes-devkit tensorboard scipy pycocotools
conda install -y pytorch3d==0.6.1 -c pytorch3d -c fvcore -c iopath -c conda-forge -c bottler 2>&1 | tail -5
python -c "import torch, mmcv, mmdet, pyro, pytorch3d; print('ENV_OK', torch.__version__, torch.cuda.is_available())"
echo SETUP_EPROPNP_DONE

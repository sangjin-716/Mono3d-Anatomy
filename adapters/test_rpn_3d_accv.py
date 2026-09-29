# Copy of the upstream M3D-RPN scripts/test_rpn_3d.py (MIT License, Copyright (c) 2020 Garrick
# Brazil, see adapters/LICENSE-M3D-RPN) with only the checkpoint paths and a devkit-eval guard
# changed (native-run half of the reproduction gate G1 for M3D-RPN).
#
# Produces the repo's native KITTI-format detections in output/tmp_results/data (renumbered split
# ids). Graded with the official evaluator in evaluator/kitti_eval, they give 14.531/11.073/8.646
# R40 E/M/H.
#
# Run from the M3D-RPN repo root, with adapters/patches/M3D-RPN_rpn_util_py_cpu_nms.patch applied:
#   cd <UPSTREAM_ROOT>/M3D-RPN && python <Mono3d-Anatomy>/adapters/test_rpn_3d_accv.py
import os, sys, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence

from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import numpy as np

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from lib.imdb_util import *

_ap = argparse.ArgumentParser()
_ap.add_argument("--release_dir", default=os.path.join(paths.UPSTREAM_ROOT, "ckpts", "m3drpn", "M3D-RPN-Release"),
                 help="unzipped M3D-RPN-Release.zip (official val1 model)")
_args = _ap.parse_args()

conf_path = os.path.join(_args.release_dir, 'm3d_rpn_depth_aware_val1_config.pkl')
weights_path = os.path.join(_args.release_dir, 'm3d_rpn_depth_aware_val1')

# load config
conf = edict(pickle_read(conf_path))
conf.pretrained = None

data_path = os.path.join(os.getcwd(), 'data')
results_path = os.path.join('output', 'tmp_results', 'data')

mkdir_if_missing(results_path, delete_if_exist=True)

init_torch(conf.rng_seed, conf.cuda_seed)

net = import_module('models.' + conf.model).build(conf)

load_weights(net, weights_path, remove_module=True)

net.eval()

print(pretty_print('conf', conf))

# devkit C++ evaluator is not compiled here; the results are graded with the official
# python evaluator afterwards — guard the repo's eval call.
try:
    test_kitti_3d(conf.dataset_test, net, conf, results_path, data_path, use_log=False)
except (FileNotFoundError, OSError) as e:
    print('detection files written; repo devkit eval skipped:', e)

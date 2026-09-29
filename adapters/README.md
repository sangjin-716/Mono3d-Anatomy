# Adapters

These scripts produce the per-prediction dumps that every analysis in this repository reads. Each
one loads a detector's released checkpoint in the repository that released it, runs it on KITTI val
(Chen split, 3769 images) and writes one row per native prediction.

You do not need them to reproduce the paper. The reports are computed from the released dumps
(see [`data/DUMPS.md`](../data/DUMPS.md)). The adapters are here so you can see how the dumps were
made, or regenerate them.

## Running an adapter

An adapter imports the upstream code (`lib.*`, `config`, `model`, ...), so it has to run in that
detector's own Python/PyTorch environment with its CUDA extensions built (DCNv2, MSDeformAttn,
...). Most adapters take `--repo` and `--ckpt`, with defaults under `<paths.UPSTREAM_ROOT>`. The
header of each script lists its arguments and the exceptions, and says which detector output goes
into `cls`, `V` and `sigma`.

We used torch 1.10 for M3D-RPN, MonoDLE, DEVIANT, MonoCon and the modern MonoFlex and MonoGround
builds, torch 1.9 for GUPNet and the MonoDETR family, and torch 1.4 for MonoFlex\* and
MonoGround\*. Results can shift with the environment, as MonoDLE and the modern MonoFlex and
MonoGround builds show below.

Most adapters can be started from any directory:

```bash
python adapters/deviant_dump.py --repo <UPSTREAM_ROOT>/DEVIANT --ckpt <checkpoint>
```

These have to be run from the root of the upstream repo: `m3drpn_dump_accv.py`,
`m3drpn_dump_floor0.py`, `test_rpn_3d_accv.py`, `dgp_cop_dump.py`, `detr_preflatten_dump.py` and
`detr_preflatten_native.py`. The last three also take `--root_dir`, which overrides the dataset
root in the upstream config.

```bash
cd <UPSTREAM_ROOT>/M3D-RPN && python <this repo>/adapters/m3drpn_dump_accv.py

cd <UPSTREAM_ROOT>/MonoDGP && python <this repo>/adapters/dgp_cop_dump.py \
    --repo <UPSTREAM_ROOT>/MonoDGP --cfg <UPSTREAM_ROOT>/MonoDGP/configs/monodgp.yaml \
    --ckpt <checkpoint> --split val --out <OUT_DIR>/dumps/dgp_val.csv --tag dgp_val
```

Output goes to `<paths.OUT_DIR>/dumps/` by default, so a re-run never overwrites a downloaded
dump. A few scripts write names that differ from the released files:

- `monoflex_dump_orig.py` writes `monoflex_val.csv` to `<OUT_DIR>/dumps/orig/`. The released
  file is `monoflex_orig_val.csv`.
- `monoflex_dump.py` writes `monoflex_val.csv` to `<OUT_DIR>/dumps/modern/`. The released file
  is `monoflex_modern_val.csv`.
- `monoground_dump.py` writes `monoground_val.csv`. The released file is
  `monoground_modern_val.csv`.
- `gupnet_dump.py`, `monodetr_dump.py` and the two MonoFlex scripts also write a `*_train.csv`.

These are cleaned-up versions of the scripts we ran, with paths turned into arguments. We have not
re-run them since, so please open an issue if one fails. `monoclue_dump.py` and `monoia_dump.py`
were rebuilt from the saved text of one-off scripts. Where we had used an edited copy of an
upstream config (usually only an absolute dataset root), the adapter makes the same edit in memory
and says so in its header. `monodle_dump.py` reads its eval config from
[`adapters/configs/monodle/kitti_accv_eval.yaml`](configs/monodle/kitti_accv_eval.yaml).

## Output format

One row per Car prediction, 23 columns:

```
sid, pred_idx, cls, sigma, log_sigma_raw, V, z_pred, bbox_h_pix, bbox_w_pix, bbox_area_pix,
x_3d, y_3d, z_3d, h_3d, w_3d, l_3d, ry, alpha, bbox_x1, bbox_y1, bbox_x2, bbox_y2, dup_rank
```

`cls` is the score the detector thresholds and `V` is the score it ranks by. `sigma` and
`log_sigma_raw` are its depth-uncertainty output, constant 1 and 0 for detectors that have none.
Boxes are in camera coordinates with `y_3d` at the bottom centre. `dup_rank` is the number of Car
predictions in the same image with 2D IoU >= 0.5 and a higher `V`.

`dgp_val.csv` and `official_monocop_val.csv` (both from `dgp_cop_dump.py`) have 10 more columns
from a matching against the KITTI Car labels. See [`data/DUMPS.md`](../data/DUMPS.md).

## Checks

Each of the twelve panel dumps passed two checks, called G1 and G2 in the supplementary material
(checks 1 and 2 in [`reports/detector_adapters_gates.md`](../reports/detector_adapters_gates.md)):

1. Reproduction: the released checkpoint, run through its own repository, gives the published Car
   Moderate AP_R40 on val.
2. Native match: the dump reconstructs the detector's own output box for box (per-image counts,
   scores and geometry).

## Per detector

The published/ours AP_R40 values are the ones in the supplementary reproduction table. The
native-match numbers are from
[`reports/detector_adapters_gates.md`](../reports/detector_adapters_gates.md). Checkpoints are the
ones linked from each repo's README unless noted otherwise.

| Detector | Commit | Adapter | Released dump | Mod AP_R40, published / ours | Native match |
|---|---|---|---|---|---|
| [M3D-RPN](https://github.com/garrickbrazil/M3D-RPN) | `bf204e3f95f6` | `m3drpn_dump_accv.py` | `m3drpn_val.csv` | 11.07 / 11.07 | NMS rebuilt from the dump gives 14.538/11.099/8.670 (E/M/H), native 14.531/11.073/8.646 |
| [MonoDLE](https://github.com/xinzhuma/monodle) | `e426aa65fdc7` | `monodle_dump.py` | `monodle_val.csv` | 13.72 / 14.57 | box for box (native writer rounds scores to 2 decimals) |
| [MonoFlex](https://github.com/zhangyp15/MonoFlex)\* | `ec6da017c325` | `monoflex_dump_orig.py` | `monoflex_orig_val.csv` | 17.51 / 17.34 | by construction, see below |
| [GUPNet](https://github.com/SuperMHP/GUPNet) | `d0e02cad228f` | `gupnet_dump.py` | `gupnet_val.csv` | 16.46 / 16.48 | native tester output graded by the same evaluator (`gupnet_native_eval.py`) |
| [DEVIANT](https://github.com/abhi1kumar/DEVIANT) | `2e6eca6e27d7` | `deviant_dump.py` | `deviant_val.csv` | 16.54 / 16.49 | boxes and scores exact (native writer rounds to 2 decimals) |
| [MonoGround](https://github.com/cfzd/MonoGround)\* | `05b3baf73228` | `monoground_dump_orig.py` | `monoground_orig_val.csv` | 18.69 / 18.69 | by construction, see below |
| [MonoCon](https://github.com/2gunsu/monocon-pytorch) | `908807bdd8d4` | `monocon_dump.py` | `monocon_val.csv` | 19.02 / 19.02 | 26.031/19.015/15.912 (E/M/H) vs native, difference <= 0.007 |
| [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR) | `6994b9f51240` | `monodetr_dump.py` | `monodetr_val.csv` | 20.83 / 20.83 | all 50x3 query-class hypotheses reproduce the native top-50 |
| [MonoDGP](https://github.com/PuFanqi23/MonoDGP) | `aa059a18214a` | `dgp_cop_dump.py` | `dgp_val.csv` | 22.34 / 22.29 | as MonoDETR |
| [MonoCoP](https://alanzhangcs.github.io/monocop-page) | not recorded | `dgp_cop_dump.py` | `official_monocop_val.csv` | 23.89 / 23.84 | as MonoDETR |
| [MonoCLUE](https://github.com/SungHunYang/MonoCLUE) | `016d3e8d3c99` | `monoclue_dump.py` | `monoclue_val.csv` | 24.10 / 24.20 | as MonoDETR, with the repo's own decode |
| [MonoIA](https://github.com/alanzhangcs/MonoIA) | `69d6ee30ca5e` | `monoia_dump.py` | `monoia_val.csv` | 24.40 / 24.48 | as MonoDETR, with the repo's own decode |

### Notes

M3D-RPN reports AP_R11 in its paper. The 11.07 is our AP_R40 evaluation of the released val1
model, and `test_rpn_3d_accv.py` is the native run for check 1. Point `--release_dir` at the
unzipped `M3D-RPN-Release.zip`, and apply
[`adapters/patches/M3D-RPN_rpn_util_py_cpu_nms.patch`](patches/M3D-RPN_rpn_util_py_cpu_nms.patch)
first. It makes the NMS run under torch 1.x and keeps the original +1-offset IoU. The dump keeps Car
boxes with score >= 0.05, at most 300 per image. `m3drpn_dump_floor0.py` writes the complete
pre-NMS pool `m3drpn_val_floor0.csv` (2.9 GB, release asset
`mono3d_anatomy_m3drpn_complete_pool_v1.zip`).

MonoDLE comes out 0.85 higher than published with the same code, data and evaluator. The
difference comes from the inference environment.

MonoFlex\* and MonoGround\* were dumped in the authors' original environment: Python 3.7, torch
1.4.0 and CUDA 10.1, with DCNv2 compiled against it. `build_monoflex_orig_env.sh` builds this
environment, and its step 4 runs each repo's own evaluation of the released checkpoint, which is
check 1 for these two. In a modern environment both lose about 1.9 AP (MonoFlex 15.54, MonoGround
16.79, see note 2 in [`reports/detector_adapters_gates.md`](../reports/detector_adapters_gates.md)).
`monoflex_dump.py` and `monoground_dump.py` make those modern dumps, and
[`data/DUMPS.md`](../data/DUMPS.md) says which analyses use which version. Check 2 holds by construction, since the adapters call the
repo's own PostProcessor with threshold 0 (top-50). There is no separate numeric check.

Both codebases read KITTI in the MonoFlex layout (`training/{image_2,calib,label_2,ImageSets}`).
The MonoFlex adapters take it with `--kitti_dir` through `mf_paths_catalog.py`. For MonoGround, set
`DATA_DIR` in the repo's `config/paths_catalog.py`. `build_monoflex_orig_env.sh` does this for the
original-environment copies.

For MonoCon we use the [2gunsu/monocon-pytorch](https://github.com/2gunsu/monocon-pytorch)
re-implementation and its released checkpoint. Its published value in the table above is the one
that repository's README reports for this checkpoint, not a number from the MonoCon paper. Apply
[`adapters/patches/MonoCon_base_engine_map_location.patch`](patches/MonoCon_base_engine_map_location.patch),
which adds `map_location` to the checkpoint load. Its native top-k is 30 and its threshold 0.4.

For DEVIANT we used the released `run_221` checkpoint.

The MonoCoP checkpoint is from [huggingface.co/zhihao406/MonoCoP](https://huggingface.co/zhihao406/MonoCoP).
Our clone had no git metadata, so we cannot give a commit.

The MonoIA checkpoint is `MonoIA_KITTI_Val.pth` from
[huggingface.co/zhihao406/MonoIA](https://huggingface.co/zhihao406/MonoIA).

For the five query-based detectors we also dumped every query x class hypothesis before the top-50
cut: `monodetr_preflatten_dump.py` for MonoDETR, `detr_preflatten_dump.py` for MonoDGP and
MonoCoP, and `detr_preflatten_native.py` for MonoCLUE and MonoIA, which need their repo's own
decode. They write `monodetr_val_preflatten.csv`, `monodgp_val_preflatten.csv`,
`official_monocop_val_preflatten.csv`, `monoclue_val_preflatten.csv` and
`monoia_val_preflatten.csv`, released as `mono3d_anatomy_query_complete_pools_v1.zip`.

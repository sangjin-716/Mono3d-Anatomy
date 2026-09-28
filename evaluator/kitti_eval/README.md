# kitti_eval — official KITTI object evaluation (Python port)

Vendored from MonoDGP (`lib/datasets/kitti/kitti_eval_python/`, https://github.com/PuFanqi23/MonoDGP),
which borrows it from [traveller59/kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python)
(MIT License, see `LICENSE`). `rotate_iou.py` is based on RRPN-revise (MIT License, header in the file).

Files are verbatim except one change in `kitti_common.py`: the `skimage` import is made lazy
(it is only needed to read image shapes, which this repository never does).
The matching kernels (`calculate_iou_partly`, `_prepare_data`, `compute_statistics_jit`,
`fused_compute_statistics`, `get_thresholds`, `do_eval`) are unchanged, so every AP in this
repository uses the official KITTI matching rules. The 3D/BEV rotated IoU runs on an NVIDIA GPU
through `numba.cuda`, as in the upstream evaluator.

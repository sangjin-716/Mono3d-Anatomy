# Where Did Monocular 3D Detection Improve? A Twelve-Detector Anatomy of Progress

**ACCV 2026** · Paper: coming soon

Code, detector outputs and reports for our analysis of twelve released monocular 3D detectors
(2019-2026) on KITTI val. The repository includes an all-point AP evaluator built on the official
KITTI evaluation, the per-prediction outputs of every detector, and the scripts and reports behind
the tables and figures of the paper.

## Setup

```bash
git clone https://github.com/sangjin-716/Mono3d-Anatomy.git
cd Mono3d-Anatomy
pip install -r requirements.txt
cp paths.example.py paths.py    # set KITTI_ROOT
```

The KITTI rotated-box IoU runs on an NVIDIA GPU through `numba.cuda`, which also needs NVVM and
libdevice from a CUDA toolkit (e.g. `conda install -c nvidia/label/cuda-12.1.0 cuda-nvcc`). Keep
numpy at 2.0.x to reproduce our M3D-RPN numbers exactly. `paths.example.py` lists the KITTI files
the scripts expect.

## Data

The detector outputs are attached to the
[v1.0 release](https://github.com/sangjin-716/Mono3d-Anatomy/releases/tag/v1.0).

```bash
mkdir -p data/dumps && cd data/dumps
wget https://github.com/sangjin-716/Mono3d-Anatomy/releases/download/v1.0/mono3d_anatomy_dumps_v1.zip
unzip mono3d_anatomy_dumps_v1.zip -x 'MD5SUMS*.txt' && cd ../..
```

This file is enough for most results. [`data/DUMPS.md`](data/DUMPS.md) has the three optional
files, the checksums and the column format.

## Usage

Check the evaluator against the official AP_R40 and AP_R11. It should end with `ALL GATES: PASS`.

```bash
python evaluator/exact_ap.py          # add --full for all twelve detectors
```

Reproduce a report. Scripts write to `reports_rerun/` and never overwrite `reports/`.

```bash
python tools/decomp/gap_exact.py
diff reports/gap_exact.txt reports_rerun/gap_exact.txt
```

[`docs/REPRODUCE.md`](docs/REPRODUCE.md) maps every table and figure to its script and report and
gives the run order and runtimes.

## Known issues

[`docs/ERRATA.md`](docs/ERRATA.md) lists three deviations found after acceptance. The yaw of the
MonoDGP and MonoCoP dumps differs from the repositories' output, MonoCLUE's dump is a separate
stochastic run, and six entries of the reproduction column are computed on the dumps.

## Citation

```bibtex
@inproceedings{jung2026where,
  title     = {Where Did Monocular 3D Detection Improve? A Twelve-Detector Anatomy of Progress},
  author    = {Jung, SangJin and Lee, YongHwan},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## License

Code: Apache-2.0. Reports and figure data: CC BY 4.0. Detector outputs (model predictions on KITTI
images): CC BY-NC-SA 4.0. Third-party files keep their MIT licenses (`evaluator/kitti_eval/LICENSE`,
`evaluator/LICENSE-MonoDGP`, `adapters/LICENSE-M3D-RPN`, `tools/crossbench/LICENSE-DEVIANT`).

The Waymo check in `tools/crossbench/` was made using the Waymo Open Dataset, provided by Waymo LLC
under the [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms).
No Waymo or nuScenes data or predictions are redistributed here.

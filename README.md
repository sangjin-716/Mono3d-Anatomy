# Where Did Monocular 3D Detection Improve? A Twelve-Detector Anatomy of Progress

SangJin Jung (Dankook University) and YongHwan Lee (Wonkwang University, corresponding author)

ACCV 2026. The paper link will be added once the proceedings are online.

We re-evaluate twelve released monocular 3D detectors (2019-2026) on KITTI val under a single
protocol and ask where they actually differ: how many objects they cover, how well they localize,
and how well their scores rank their own boxes. We extend TIDE3D's ranking-error diagnosis to these
released checkpoints. This repository has the evaluator, the analysis scripts, the per-prediction
outputs of all twelve detectors, and the reports behind the numbers in the paper.

Main results (KITTI val, Car, Moderate, IoU 0.7):

- The official AP_R40 snaps oracle AP ceilings onto a 2.5 AP grid. The same MonoIA pool gives an
  oracle gain of +8.9 under AP_R11 and +13.0 under AP_R40. We use all-point AP, which does not
  have this problem.
- The post-hoc rescoring we tested looks helpful with image-level folds, but on unseen drives it
  loses 2.6 to 13.6 AP.
- Higher-AP checkpoints localize better in the near field. Beyond 45 m there is no reliable trend.
  Counting within each detector's fixed set of boxes, most of the missing AP comes from objects
  that no output box covers at IoU 0.7.
- Ranking each detector's own boxes perfectly (matched boxes first) would add +11.4 to +14.0 AP
  (median +11.7). This is a bound on a fixed set of boxes, not a realizable gain. Whether it
  shrinks for better detectors cannot be decided from twelve checkpoints.

## Contents

```
evaluator/          all-point AP (exact_ap.py) on top of the official KITTI evaluation
adapters/           how each detector's outputs were dumped from its released checkpoint
tools/decomp/       analysis scripts for the twelve-detector panel
tools/orig/         the same analyses for MonoFlex* and MonoGround* (see below)
tools/extensions/   checks added for the camera-ready (candidate budgets, near misses, power, classes)
tools/crossbench/   preliminary Waymo and nuScenes check
reports/            outputs of the scripts, as used in the paper
reports_orig/       the same for MonoFlex* and MonoGround*
figures/            figure scripts and their input data
data/DUMPS.md       what is in the release files and which scripts need them
docs/REPRODUCE.md   which script and report each table and figure comes from
```

## Setup

You need Linux and an NVIDIA GPU, because the rotated-box IoU of the official KITTI evaluation
runs through `numba.cuda`.

```bash
git clone https://github.com/sangjin-716/Mono3d-Anatomy.git
cd Mono3d-Anatomy
pip install -r requirements.txt
cp paths.example.py paths.py    # then set KITTI_ROOT
```

`numba.cuda` also needs NVVM and libdevice from a CUDA toolkit, which pip does not install. With
conda, `conda install -c nvidia/label/cuda-12.1.0 cuda-nvcc` works. Check it with
`python -c "from numba import cuda; print(cuda.is_available())"`.

We tested with Python 3.9, numpy 2.0.2, pandas 2.3.3 and numba 0.60.0 (see
`requirements.txt`). Keep numpy at 2.0.x if you want our exact numbers: M3D-RPN has tied scores,
numpy 1.26 orders ties differently inside its NMS, and its results then move slightly.

Under `KITTI_ROOT` the scripts expect `training/label_2`, `training/calib`, the Chen split
(`ImageSets/train.txt`, `ImageSets/val.txt`), and, for the drive-disjoint split only, the devkit
files `mapping/train_mapping.txt` and `mapping/train_rand.txt`. All other paths are set in
`paths.py` or through environment variables.

## Data

The detector outputs are too large for git and are attached to the
[v1.0 release](https://github.com/sangjin-716/Mono3d-Anatomy/releases/tag/v1.0). The first file is
enough for most results. [`data/DUMPS.md`](data/DUMPS.md) lists what the other three are for.

```bash
mkdir -p data/dumps && cd data/dumps
BASE=https://github.com/sangjin-716/Mono3d-Anatomy/releases/download/v1.0
wget $BASE/mono3d_anatomy_dumps_v1.zip                 # 443 MB, needed
wget $BASE/mono3d_anatomy_query_complete_pools_v1.zip  # 276 MB, optional
wget $BASE/mono3d_anatomy_m3drpn_complete_pool_v1.zip  # 1.2 GB, optional
wget $BASE/mono3d_anatomy_matched_tables_v1.zip        # 27 MB, optional
sha256sum -c --ignore-missing <<'EOF'
3eb2d46701f481851b89f6f52c2472805d0467c5b8963fdd8b7d861ab4262edd  mono3d_anatomy_dumps_v1.zip
0a4696a30a166d4b3a0cbfeab92cbaa09ac91ab6769a15b90e45e90ea462362f  mono3d_anatomy_query_complete_pools_v1.zip
5e0bbce9d2bd06eaea8a5954ad4202aa7541a862e995f7137986b4625aa77497  mono3d_anatomy_m3drpn_complete_pool_v1.zip
47d71716b77a7e018bc9ced001c313448ffe3935a1e92c37ea4e7a2609ba8e83  mono3d_anatomy_matched_tables_v1.zip
EOF
for z in mono3d_anatomy_*_v1.zip; do unzip -n "$z" -x 'MD5SUMS*.txt'; done
md5sum -c --ignore-missing ../MD5SUMS.txt
cd ../..
```

## Check the evaluator

```bash
python evaluator/exact_ap.py          # a few minutes
python evaluator/exact_ap.py --full   # all twelve detectors
```

The script first checks two toy cases whose AP can be worked out by hand, then checks that its
AP_R40 and AP_R11 match the official evaluation on real dumps. Numba prints a lot of
`NumbaPerformanceWarning` messages along the way, and newer numba versions also warn about
`parallel=True`. You can ignore both. The quick run should end with:

```
toyA perfect-rank: allpoint=100.00 (hand 100.00)  R40 mine/off=2.50/2.50  R11 mine/off=9.09/9.09  [PASS]
toyB tied-scores : allpoint=66.67 (hand 66.67)  R40 mine/off=1.67/1.67  R11 mine/off=6.06/6.06  [PASS]
dgp              R40 off/mine= 22.290/ 22.290  R11 off/mine= 26.215/ 26.215  p101=22.763 p401=22.834 allpoint=22.905 [PASS]
gupnet           R40 off/mine= 16.481/ 16.481  R11 off/mine= 21.865/ 21.865  p101=16.933 p401=17.030 allpoint=17.122 [PASS]
ALL GATES: PASS
```

All-point AP uses every distinct score as a threshold and integrates the official interpolated
precision over recall, instead of sampling 40 or 11 recall points. The matching is the official
one.

## Reproducing the paper

Run the scripts from the repository root. They write to `reports_rerun/` and never touch
`reports/`, so you can diff the two:

```bash
python tools/decomp/gap_exact.py
diff reports/gap_exact.txt reports_rerun/gap_exact.txt
```

A re-run of a script should match its report in this repository apart from the header, timestamp,
path and package-version lines. [`docs/REPRODUCE.md`](docs/REPRODUCE.md) lists the few known
differences. It also maps every table and figure to its script and report, gives the order for the
scripts that share caches, and lists runtimes. Most scripts finish within an hour on one GPU and
the longest takes about 3.5 hours.

A few things to know:

- MonoFlex and MonoGround only reproduce their published numbers in the authors' original
  torch-1.4 environment, so the paper uses those outputs (marked with * in the paper).
  `tools/orig/` and `reports_orig/` hold the analyses on them. `reports/` was computed on a modern
  rebuild of the same two checkpoints, and both versions are in the release.
- Some reports are edited public copies. Those files start with a `# public copy of ...` line
  that gives the SHA-256 of the original and says what changed. Every number that is kept is
  unchanged. A few copies in `reports/extensions/` also leave out some lines, which their second
  header line names.
- The Waymo, nuScenes, Pedestrian and Cyclist results are preliminary.

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

Code is under Apache-2.0 (`LICENSE`). Reports and figure data are under CC BY 4.0
(`reports/LICENSE`, `figures/LICENSE`). The detector outputs in the release are model predictions
on KITTI images and are shared under CC BY-NC-SA 4.0. The vendored KITTI evaluation code and the
MonoDGP calibration reader keep their MIT licenses (`evaluator/kitti_eval/LICENSE`,
`evaluator/LICENSE-MonoDGP`).

We thank the authors of the twelve detectors for releasing their code and checkpoints, and the
KITTI, Waymo Open Dataset and nuScenes teams.

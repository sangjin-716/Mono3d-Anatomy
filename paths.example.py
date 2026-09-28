"""Local path configuration. Copy to paths.py and edit it (paths.py is gitignored).

Every script in this repository reads its paths from here, so no home directory is
hardcoded anywhere. Each value can also be set through the environment variable shown.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# KITTI 3D object detection, standard layout (training/label_2, training/calib, ImageSets/*.txt).
# The Chen split (ImageSets/val.txt, 3769 images) is the evaluation split of the paper.
# The drive-disjoint split also needs the two mapping files of the KITTI object devkit
# (devkit_object.zip, folder "mapping"), copied to
#     <KITTI_ROOT>/mapping/train_mapping.txt
#     <KITTI_ROOT>/mapping/train_rand.txt
# tools/decomp/frame_sequence.py builds the frame -> drive table from them. It is read by the
# drive-grouped scripts (clean_transfer_strong, phase1_drive_oof, bootstrap_floor,
# e3_endpoint_gap_boot, and tools/orig transfer_orig, transfer_save_orig, bootstrap_orig).
KITTI_ROOT = os.environ.get("KITTI_ROOT", os.path.join(_HERE, "data", "KITTIDataset"))
LABEL_DIR = os.environ.get("KITTI_LABEL_DIR", os.path.join(KITTI_ROOT, "training", "label_2"))
CALIB_DIR = os.environ.get("KITTI_CALIB_DIR", os.path.join(KITTI_ROOT, "training", "calib"))
VAL_LIST = os.environ.get("KITTI_VAL_LIST", os.path.join(KITTI_ROOT, "ImageSets", "val.txt"))
TRAIN_LIST = os.environ.get("KITTI_TRAIN_LIST", os.path.join(KITTI_ROOT, "ImageSets", "train.txt"))

# The files released with the paper (see data/DUMPS.md and data/MD5SUMS.txt). Unpack every
# release asset into this one folder: the per-prediction dumps, the complete candidate pools and
# the matched tables. Several scripts read the complete pools and the matched tables from
# DUMP_DIR directly, so keep them all here.
DUMP_DIR = os.environ.get("MONO3D_DUMP_DIR", os.path.join(_HERE, "data", "dumps"))

# Scratch space for temporary KITTI-format files and cached intermediate arrays.
CACHE_DIR = os.environ.get("MONO3D_CACHE_DIR", os.path.join(_HERE, "cache"))

# Where re-runs write their reports. Kept separate so a re-run never overwrites the
# reports in reports/ and reports_orig/ that the paper cites.
OUT_DIR = os.environ.get("MONO3D_OUT_DIR", os.path.join(_HERE, "reports_rerun"))

# Optional: only needed for tools/crossbench (Waymo / nuScenes preliminary audit).
WAYMO_ROOT = os.environ.get("WAYMO_ROOT", "")
NUSC_ROOT = os.environ.get("NUSC_ROOT", "")

# Optional: clones of the upstream detector repos (and, for tools/crossbench, their checkpoints and
# native outputs). Not needed for the reports in reports/ and reports_orig/. It is read by
#   adapters/                                    (regenerate the dumps),
#   tools/extensions/class_capability.py         (Parts C to E: native prediction folders and
#                                                 upstream READMEs; parts that find nothing are
#                                                 skipped and the report says so),
#   tools/extensions/difficulty_class_extension.py (Part B: native multi-class prediction folders),
#   tools/crossbench/                            (Waymo / nuScenes audit, through _paths.sh).
UPSTREAM_ROOT = os.environ.get("MONO3D_UPSTREAM_ROOT", os.path.join(_HERE, "upstream"))

"""Ported from tools/decomp/task0_sequence_mapping.py (the frame -> drive mapping part only) for
the public release. Computation unchanged. Produces <CACHE_DIR>/decomp/frame_sequence.csv, the
KITTI object-frame -> raw-drive table that the drive-grouped folds and the drive-cluster
bootstrap read (clean_transfer_strong.py, phase1_drive_oof.py, bootstrap_floor.py,
e3_endpoint_gap_boot.py). The original scripts read the same table from a pre-built CSV; it is
rebuilt here from the KITTI object devkit so no extra file has to be shipped.

Mapping convention (KITTI object devkit): object index i (0-based, %06d filename) ->
train_rand[i] (1-based line number) -> train_mapping[line-1] = (date, drive, frame_in_drive).
sequence label = drive (e.g. 2011_09_26_drive_0005_sync).
Needs <KITTI_ROOT>/mapping/train_mapping.txt and <KITTI_ROOT>/mapping/train_rand.txt
(devkit_object "mapping" folder).

Run from the repository root to (re)build the table: python tools/decomp/frame_sequence.py
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, cache_dir  # noqa: E402

MAP = os.path.join(paths.KITTI_ROOT, "mapping", "train_mapping.txt")
RAND = os.path.join(paths.KITTI_ROOT, "mapping", "train_rand.txt")


def frame_sequence_csv():
    """Path of frame_sequence.csv (built from the devkit mapping on first use)."""
    out_csv = os.path.join(cache_dir("decomp"), "frame_sequence.csv")
    if os.path.exists(out_csv):
        return out_csv
    mapping = [ln.split() for ln in open(MAP).read().splitlines() if ln.strip()]
    assert len(mapping) == 7481, f"mapping lines={len(mapping)}"
    rand = [int(x) for x in open(RAND).read().strip().split(",") if x.strip()]
    assert len(rand) == 7481, f"rand n={len(rand)}"
    assert min(rand) == 1 and max(rand) == 7481

    # frame i -> (date, drive, raw_frame); drive = sequence
    rows = []
    for i in range(7481):
        date, drive, raw = mapping[rand[i] - 1]
        rows.append((f"{i:06d}", i, date, drive, raw))
    fs = pd.DataFrame(rows, columns=["frame_id", "frame_idx", "date", "drive", "raw_frame"])
    fs.to_csv(out_csv, index=False)
    return out_csv


def load_frame_sequence():
    """Same DataFrame the original scripts obtained with pd.read_csv(<frame_sequence.csv>)."""
    return pd.read_csv(frame_sequence_csv())


if __name__ == "__main__":
    p = frame_sequence_csv()
    fs = load_frame_sequence()
    print(f"{p}: {len(fs)} frames, {fs.drive.nunique()} drives")

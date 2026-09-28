"""Ported from tools/decomp/paired_perbin_dz.py for the public release. Computation unchanged.
Produces reports/paired_perbin_dz.txt.

Paired per-distance-bin depth table (MonoCoP-Table-5 style, but confound-free):
per-DISTANCE-BIN median |dz| on the SAME common objects across detectors.
= "same matched objects, distance-binned depth error"; removes the matched-set confound of the
per-detector progression table.

Reuses paired_common_object.build_detection_matrix verbatim (S5 pool, arc IoU3D,
moderate GT, found = best-IoU>=0.3). Common set = GTs found by ALL 4 core detectors.
Run from the repository root: python tools/decomp/paired_perbin_dz.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np
from paired_common_object import build_detection_matrix, read_gt_mod_full  # reuse exact machinery

BINS = ["0-15", "15-30", "30-45", "45+"]
CORE = [("GUPNet", "gupnet", 16.5), ("MonoDETR", "monodetr", 21.0),
        ("MonoDGP", "dgp", 22.3), ("MonoCoP", "official_monocop", 23.9)]
VAL = paths.VAL_LIST
val_sids = [int(x) for x in open(VAL).read().split()]

mats = {}
for nm, f, ap in CORE:
    mats[nm] = build_detection_matrix(nm, f, val_sids)[0]   # (mat, n_kept) tuple
    print(f"  built {nm}", flush=True)

keys = set(mats[CORE[0][0]].keys())
common = [k for k in keys if all(mats[nm][k]["found"] for nm, _, _ in CORE)]
print(f"\ncommon objects (found by ALL 4 at IoU>=0.3): {len(common)}")

out = []
out.append("=" * 88)
out.append("PAIRED PER-BIN |dz| (m): SAME common objects, per distance bin  [STRONG evidence]")
out.append(f"common set = {len(common)} moderate Car GTs found by all 4 (best 3D-IoU>=0.3), S5 pool")
out.append("=" * 88)
hdr = f"{'detector':10s} {'AP':>6} | " + " ".join(f"{b:>8}" for b in BINS) + f" {'ALL':>8}"
out.append(hdr); out.append("-" * len(hdr))
nb = {b: 0 for b in BINS}
for k in common:
    nb[mats[CORE[0][0]][k]["bin"]] += 1
for nm, f, ap in CORE:
    m = mats[nm]
    row = []
    for b in BINS:
        v = np.array([m[k]["dz"] for k in common if m[k]["bin"] == b and not np.isnan(m[k]["dz"])])
        row.append(np.median(v) if v.size else float("nan"))
    allv = np.array([m[k]["dz"] for k in common if not np.isnan(m[k]["dz"])])
    out.append(f"{nm:10s} {ap:>6.1f} | " + " ".join(f"{x:8.3f}" for x in row) + f" {np.median(allv):8.3f}")
out.append("(n per bin: " + " ".join(f"{b}={nb[b]}" for b in BINS) + ")")
out.append("")
out.append("READ: same objects for every detector -> any |dz| decrease is REAL depth tightening,")
out.append("not easier-matched-set. This is the strong-evidence (matched-objects, distance-binned)")
out.append("category most method papers do not report.")
txt = "\n".join(out)
print("\n" + txt)
OUT = out_path("paired_perbin_dz.txt")
open(OUT, "w").write(txt + "\n")
print(f"\n[written] {OUT}")

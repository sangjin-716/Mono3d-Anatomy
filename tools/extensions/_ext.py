"""Shared setup for the scripts in tools/extensions/ (checks for supplementary Secs. L-O).

Usage at the top of a script in this directory:
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _ext import paths, dump_path, cache_dir, out_path, aux_dump_path, ROOT

It puts the repository root and tools/decomp/ on sys.path, so the shared kernels keep their
original module names (ap_corrector_arc, dgp_cop_oracle_matrix, exact_ap, depth_share_bridge).

Auxiliary dumps. Some checks read the COMPLETE pre-selection pools, which are larger than the
per-prediction dumps and are released as two separate assets (data/DUMPS.md; their MD5s are in
data/MD5SUMS.txt):
    monodetr_val_preflatten.csv, monodgp_val_preflatten.csv, official_monocop_val_preflatten.csv,
    monoclue_val_preflatten.csv, monoia_val_preflatten.csv   (all query x class hypotheses of the
                                                               five query-based detectors)
    m3drpn_val_floor0.csv                                    (M3D-RPN, up to 3000 boxes per image
                                                               before its NMS)
They are read from the first of: the environment variable MONO3D_AUX_DUMP_DIR,
paths.AUX_DUMP_DIR (if defined), paths.DUMP_DIR.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(ROOT, "tools", "decomp"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools._release import paths, dump_path, cache_dir, out_path  # noqa: E402,F401

AUX_DUMP_DIR = os.environ.get("MONO3D_AUX_DUMP_DIR",
                              getattr(paths, "AUX_DUMP_DIR", paths.DUMP_DIR))


def aux_dump_path(fname):
    """Path of an auxiliary dump file (complete-pool release assets, see data/DUMPS.md)."""
    return os.path.join(AUX_DUMP_DIR, fname)


def repo_rel(p):
    """Repository-relative form of a path, for printing in reports."""
    try:
        r = os.path.relpath(os.path.abspath(p), ROOT)
    except ValueError:
        return p
    return p if r.startswith("..") else r

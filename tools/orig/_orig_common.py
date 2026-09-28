"""Release helper shared by the scripts in tools/orig (not a port of an original script).

The scripts in tools/orig re-run the fixed-pool diagnostics of tools/decomp on the
original-environment (torch-1.4) dumps of MonoFlex* and MonoGround* (released dump stems
monoflex_orig / monoground_orig). Like the originals, they import the main-panel modules of
tools/decomp (ap_corrector_arc, dgp_cop_oracle_matrix, exact_ap, depth_share_bridge,
fast_subset_dense, frame_sequence) by bare module name, so importing this module puts
tools/decomp on sys.path. It also names the directories the scripts share.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "tools", "decomp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools._release import paths, dump_path, cache_dir, out_path  # noqa: E402,F401

# Work directories and npz caches of the tools/orig scripts. Caches are shared between scripts
# exactly as in the original workspace (e.g. <stem>_bridgecache.npz, _e4cache_<name>.npz).
ORIG = cache_dir("orig")
# Cache/work directory of the main-panel scripts in tools/decomp (the same cache_dir("decomp")
# they use). bootstrap_orig.py reads main-panel intermediates from here.
DECOMP = cache_dir("decomp")

REPORTS = os.path.join(ROOT, "reports")
REPORTS_ORIG = os.path.join(ROOT, "reports_orig")


def frozen_report(name):
    """A frozen main-panel report shipped in reports/ (read-only input)."""
    return os.path.join(REPORTS, name)


def orig_report(name):
    """A tools/orig report: the frozen copy in reports_orig/ when it is shipped there,
    otherwise the re-run written by the corresponding tools/orig script (OUT_DIR)."""
    p = os.path.join(REPORTS_ORIG, name)
    return p if os.path.exists(p) else out_path(name)


def extra_dump(fname):
    """Complete-pool dumps that a few 12-detector scripts read for the query family and M3D-RPN,
    released as two separate assets of release v1.0 (data/DUMPS.md, not in the per-prediction
    dump zip): <detector>_val_preflatten.csv (every query's Car channel before the top-50 flatten)
    and m3drpn_val_floor0.csv (the complete pre-NMS anchor pool, 2.9 GB). Looked up in
    $MONO3D_EXTRA_DUMP_DIR, else paths.EXTRA_DUMP_DIR, else paths.DUMP_DIR."""
    d = os.environ.get("MONO3D_EXTRA_DUMP_DIR", getattr(paths, "EXTRA_DUMP_DIR", paths.DUMP_DIR))
    return os.path.join(d, fname)

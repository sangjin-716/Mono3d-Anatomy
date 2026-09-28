"""Shared setup for every script in tools/: repository root on sys.path, local paths, dump files.

Usage at the top of a script:
    import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from tools._release import paths, dump_path, cache_dir, out_path

Run directly (python tools/_release.py) to list which released files are present in DUMP_DIR.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import paths  # noqa: E402  (local copy of paths.example.py)
except ImportError:  # pragma: no cover
    raise SystemExit("paths.py not found: copy paths.example.py to paths.py and edit it")

# File name of each released file (data/DUMPS.md, data/MD5SUMS.txt). All of them are unpacked
# into the one folder paths.DUMP_DIR.
DUMP_FILES = {
    # --- mono3d_anatomy_dumps_v1.zip: per-prediction dumps of the twelve detectors ---
    # The panel of the paper uses the original-environment dumps for MonoFlex* and MonoGround*
    # ("*_orig"). The "_modern" dumps are the same checkpoints rebuilt in a modern environment;
    # the frozen reports in reports/ (not reports_orig/) and the auxiliary class and near-miss
    # checks were computed on them.
    "m3drpn": "m3drpn_val.csv",
    "monodle": "monodle_val.csv",
    "monoflex": "monoflex_modern_val.csv",
    "gupnet": "gupnet_val.csv",
    "deviant": "deviant_val.csv",
    "monoground": "monoground_modern_val.csv",
    "monocon": "monocon_val.csv",
    "monodetr": "monodetr_val.csv",
    "dgp": "dgp_val.csv",
    "official_monocop": "official_monocop_val.csv",
    "monoclue": "monoclue_val.csv",
    "monoia": "monoia_val.csv",
    "monoflex_orig": "monoflex_orig_val.csv",
    "monoground_orig": "monoground_orig_val.csv",
    # --- mono3d_anatomy_query_complete_pools_v1.zip: every query x class hypothesis of the five
    # query-based detectors before their top-50 flatten (complete native pools) ---
    "monodetr_preflatten": "monodetr_val_preflatten.csv",
    "monodgp_preflatten": "monodgp_val_preflatten.csv",
    "official_monocop_preflatten": "official_monocop_val_preflatten.csv",
    "monoclue_preflatten": "monoclue_val_preflatten.csv",
    "monoia_preflatten": "monoia_val_preflatten.csv",
    # --- mono3d_anatomy_m3drpn_complete_pool_v1.zip: M3D-RPN, up to 3000 boxes per image before
    # its NMS (score floor 0) ---
    "m3drpn_floor0": "m3drpn_val_floor0.csv",
    # --- mono3d_anatomy_matched_tables_v1.zip: prediction-to-GT matched tables read only by
    # tools/decomp/c3_conditioned.py ---
    "matched_monodetr": "matched_monodetr_val.csv",
    "matched_dgp": "matched_dgp_val.csv",
    "matched_cop": "matched_cop_val.csv",
    "matched_monoia": "matched_monoia_val.csv",
    "matched_monoflex": "matched_monoflex_val.csv",
    "matched_gupnet": "matched_gupnet_val.csv",
}

# Release asset (GitHub release v1.0) -> the stems it contains.
ASSETS = {
    "mono3d_anatomy_dumps_v1.zip": [
        "m3drpn", "monodle", "monoflex", "gupnet", "deviant", "monoground", "monocon", "monodetr",
        "dgp", "official_monocop", "monoclue", "monoia", "monoflex_orig", "monoground_orig"],
    "mono3d_anatomy_query_complete_pools_v1.zip": [
        "monodetr_preflatten", "monodgp_preflatten", "official_monocop_preflatten",
        "monoclue_preflatten", "monoia_preflatten"],
    "mono3d_anatomy_m3drpn_complete_pool_v1.zip": ["m3drpn_floor0"],
    "mono3d_anatomy_matched_tables_v1.zip": [
        "matched_monodetr", "matched_dgp", "matched_cop", "matched_monoia", "matched_monoflex",
        "matched_gupnet"],
}


def dump_path(stem):
    """Path of a released file for a stem such as 'dgp', 'monoflex_orig' or 'm3drpn_floor0'."""
    return os.path.join(paths.DUMP_DIR, DUMP_FILES[stem])


def cache_dir(*parts):
    d = os.path.join(paths.CACHE_DIR, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def out_path(name):
    """Report path for a re-run (never inside the frozen reports/ tree)."""
    p = os.path.join(paths.OUT_DIR, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


if __name__ == "__main__":
    # Presence check only (no hashing; use md5sum -c data/MD5SUMS.txt for integrity).
    print(f"DUMP_DIR = {paths.DUMP_DIR}")
    for asset, stems in ASSETS.items():
        missing = [DUMP_FILES[s] for s in stems if not os.path.exists(dump_path(s))]
        state = "complete" if not missing else f"missing {len(missing)}/{len(stems)}: {', '.join(missing)}"
        print(f"  {asset:45s} {state}")

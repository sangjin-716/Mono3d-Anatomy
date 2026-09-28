"""Shared geometry/IO for the cross-architecture mono3D failure decomposition.

Single source of truth for the validated 3D-IoU (report §10: shapely BEV-polygon
intersection x height-overlap / union, loc y = bottom-center, ry = KITTI col-14).
The BEV polygon convention here matches MonoDGP's own Object3d.generate_corners3d()
exactly (x_corners=[l/2,...], z_corners=[w/2,...], R=[[c,0,s],[0,1,0],[-s,0,c]]),
so iou3d() is faithful to the detector's box parametrisation.

Reused verbatim from the validated probe scripts (recall_decomp3.py / axis_split).
Do NOT rewrite the IoU here per-detector — import from this module.
"""
from __future__ import annotations
import math
import numpy as np
from shapely.geometry import Polygon

DIST_BINS = [0, 15, 30, 45, 1e9]
DIST_LABELS = ["0-15", "15-30", "30-45", "45+"]


def dist_bin(z: float) -> str:
    for i in range(4):
        if DIST_BINS[i] <= z < DIST_BINS[i + 1]:
            return DIST_LABELS[i]
    return DIST_LABELS[-1]


def bev_polygon(x, z, w, l, ry):
    """BEV rectangle for a box centred at (x,z), size (w,l), heading ry.

    Matches Object3d.generate_corners3d: local x along l, local z along w,
    R = [[cos,0,sin],[0,1,0],[-sin,0,cos]].
    """
    xc = np.array([l / 2, l / 2, -l / 2, -l / 2])
    zc = np.array([w / 2, -w / 2, -w / 2, w / 2])
    c, s = math.cos(ry), math.sin(ry)
    return Polygon(np.c_[x + c * xc + s * zc, z - s * xc + c * zc])


def iou3d(px, py, pz, ph, pw, pl, pry, gx, gy, gz, gh, gw, gl, gry):
    """Validated 3D IoU. y = bottom-center of the box (KITTI location convention)."""
    try:
        inter = bev_polygon(px, pz, pw, pl, pry).intersection(
            bev_polygon(gx, gz, gw, gl, gry)).area
    except Exception:
        return 0.0
    hov = max(0.0, min(py, gy) - max(py - ph, gy - gh))
    iv = inter * hov
    u = pl * pw * ph + gl * gw * gh - iv
    return iv / u if u > 0 else 0.0

#!/usr/bin/env python
"""
Ported from mono3d_crossdataset/tools/waymo_v2_to_kitti_gt.py for the public release.
Computation unchanged. Produces the KITTI-format Waymo val ground truth (labels + calib)
under <WAYMO_ROOT>/waymo_kitti/validation/{label,calib}/ and conversion_report.json.

waymo_v2_to_kitti_gt.py -- Waymo v2 parquet -> KITTI-format GT (labels + calib), reproducing
DEVIANT/data/waymo/converter.py semantics EXACTLY, labels-only (no images, no lidar).

Faithful reproduction of DEVIANT data/waymo/converter.py:
  - objects: laser_labels (v2: lidar_box) that have a FRONT-camera projected box
             (v1: projected_lidar_labels name==1; v2: projected_lidar_box camera_name==1)
  - 2D bbox: that projected box, center +- size/2; skip degenerate
  - classes: VEHICLE->Car, PEDESTRIAN->Pedestrian, CYCLIST->Cyclist, SIGN->Sign (all kept)
  - filter_empty_3dboxes: skip num_lidar_points_in_box < 1
  - 3D: (x,y,z)=center, z -= h/2 (bottom center, vehicle frame),
        then pt_ref = homo(T_front_cam_to_ref) @ inv(T_front_cam_to_vehicle) @ [x,y,z,1]
        T_front_cam_to_ref = [[0,-1,0],[0,0,-1],[1,0,0]]
  - rotation_y = -heading - pi/2 ; truncated=0 occluded=0 alpha=-10 ; 16th col = num points
  - calib: P2 = intrinsic-only 3x4 (f_u f_v c_u c_v), P0/P1/P3 = eye(3x4), R0_rect = I,
           Tr_velo_to_cam = homo(T_front_cam_to_ref) @ inv(extrinsic)
Frame-ID mapping: ImageSets/val_org.txt line i -> sequential id i (000000..039847);
org6 = f"{file_idx:03d}{frame_idx:03d}" with file_idx = segment order (val_tfrecord sorted),
frame_idx = timestamp-ascending order within segment (validated per segment below).
Frame enumeration source = vehicle_pose (exactly one row per frame).

Inputs:  <WAYMO_ROOT>/waymo_v2/validation/{lidar_box,projected_lidar_box,camera_calibration,
         vehicle_pose}/<segment>.parquet  (official Waymo Open Dataset v2 validation release)
         <UPSTREAM_ROOT>/DEVIANT/data/waymo/ImageSets/val_org.txt
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths

V2   = os.path.join(paths.WAYMO_ROOT, 'waymo_v2', 'validation')
DEV  = os.path.join(paths.UPSTREAM_ROOT, 'DEVIANT', 'data', 'waymo', 'ImageSets')
OUT  = os.path.join(paths.WAYMO_ROOT, 'waymo_kitti', 'validation')
CLASS_MAP = {1: 'Car', 2: 'Pedestrian', 3: 'Sign', 4: 'Cyclist'}   # waymo type enum -> kitti
T_FC2REF = np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])

def homo(m3):
    m = np.eye(4); m[:3,:3] = m3; return m

def main():
    os.makedirs(f'{OUT}/label', exist_ok=True)
    os.makedirs(f'{OUT}/calib', exist_ok=True)

    # ---- mapping: val_org.txt line order = sequential id ----
    lines = [l.strip().split() for l in open(f'{DEV}/val_org.txt') if l.strip()]
    assert len(lines) == 39848, len(lines)
    # seg full name -> v2 context name;   collect (seg, org6) -> seq
    def v2name(seg):
        s = seg[len('segment-'):] if seg.startswith('segment-') else seg
        return s[:-len('_with_camera_labels')] if s.endswith('_with_camera_labels') else s
    seq_of = {}                       # (v2seg, org6) -> seq id
    segs_in_order = []                # v2 seg names, file_idx order
    for i, (seg, org6) in enumerate(lines):
        v2s = v2name(seg)
        seq_of[(v2s, org6)] = i
        fidx = int(org6[:3])
        while len(segs_in_order) <= fidx: segs_in_order.append(None)
        if segs_in_order[fidx] is None: segs_in_order[fidx] = v2s
        else: assert segs_in_order[fidx] == v2s, (fidx, v2s)
    assert len(segs_in_order) == 202 and all(s for s in segs_in_order)

    report = dict(frames=0, objects=dict(), empty_frames=0, skipped_no_front2d=0,
                  skipped_degenerate=0, skipped_no_points=0, mapping_errors=0)

    for fidx, seg in enumerate(segs_in_order):
        lb  = pd.read_parquet(f'{V2}/lidar_box/{seg}.parquet')
        plb = pd.read_parquet(f'{V2}/projected_lidar_box/{seg}.parquet')
        cal = pd.read_parquet(f'{V2}/camera_calibration/{seg}.parquet')
        vp  = pd.read_parquet(f'{V2}/vehicle_pose/{seg}.parquet',
                              columns=['key.frame_timestamp_micros'])

        # frame enumeration = vehicle_pose timestamps, ascending
        ts_sorted = sorted(vp['key.frame_timestamp_micros'].unique().tolist())

        # front-cam calib (camera_name==1)
        c = cal[cal['key.camera_name'] == 1].iloc[0]
        fu, fv, cu, cv = (c['[CameraCalibrationComponent].intrinsic.f_u'],
                          c['[CameraCalibrationComponent].intrinsic.f_v'],
                          c['[CameraCalibrationComponent].intrinsic.c_u'],
                          c['[CameraCalibrationComponent].intrinsic.c_v'])
        T_extr = np.array(c['[CameraCalibrationComponent].extrinsic.transform']).reshape(4,4)
        T_v2c  = np.linalg.inv(T_extr)             # vehicle -> front cam
        T_full = homo(T_FC2REF) @ T_v2c            # vehicle -> kitti ref cam
        P2 = np.zeros((3,4)); P2[0,0]=fu; P2[1,1]=fv; P2[0,2]=cu; P2[1,2]=cv; P2[2,2]=1
        eye34 = np.eye(4)[:3,:].reshape(12)
        Tr = (homo(T_FC2REF) @ np.linalg.inv(T_extr))[:3,:].reshape(12)
        calib_txt = ''
        for i in range(4):
            vals = P2.reshape(12) if i == 2 else eye34
            calib_txt += f'P{i}: ' + ' '.join(str(v) for v in vals) + '\n'
        calib_txt += 'R0_rect: ' + ' '.join(str(v) for v in np.eye(3).astype(np.float32).flatten()) + '\n'
        calib_txt += 'Tr_velo_to_cam: ' + ' '.join(str(v) for v in Tr) + '\n'

        # front-cam projected boxes, grouped by timestamp
        plb1 = plb[plb['key.camera_name'] == 1]
        pgrp = {ts: g for ts, g in plb1.groupby('key.frame_timestamp_micros')}
        lgrp = {ts: g for ts, g in lb.groupby('key.frame_timestamp_micros')}

        for frame_idx, ts in enumerate(ts_sorted):
            org6 = f'{fidx:03d}{frame_idx:03d}'
            key = (seg, org6)
            if key not in seq_of:
                report['mapping_errors'] += 1; continue
            seq = seq_of[key]; sid = f'{seq:06d}'
            with open(f'{OUT}/calib/{sid}.txt', 'w') as f: f.write(calib_txt)

            out_lines = []
            g2d = pgrp.get(ts)
            bbox2d = {}
            if g2d is not None:
                for _, r in g2d.iterrows():
                    cx, cy = r['[ProjectedLiDARBoxComponent].box.center.x'], r['[ProjectedLiDARBoxComponent].box.center.y']
                    sx, sy = r['[ProjectedLiDARBoxComponent].box.size.x'],  r['[ProjectedLiDARBoxComponent].box.size.y']
                    bbox2d[r['key.laser_object_id']] = (cx-sx/2, cy-sy/2, cx+sx/2, cy+sy/2)
            g3d = lgrp.get(ts)
            if g3d is not None:
                for _, r in g3d.iterrows():
                    oid = r['key.laser_object_id']
                    bb = bbox2d.get(oid)
                    if bb is None:
                        report['skipped_no_front2d'] += 1; continue
                    if bb[2]-bb[0] <= 0 or bb[3]-bb[1] <= 0:
                        report['skipped_degenerate'] += 1; continue
                    typ = int(r['[LiDARBoxComponent].type'])
                    if typ not in CLASS_MAP: continue
                    npts = r['[LiDARBoxComponent].num_lidar_points_in_box']
                    npts = 0 if pd.isna(npts) else int(npts)
                    if npts < 1:
                        report['skipped_no_points'] += 1; continue
                    l = r['[LiDARBoxComponent].box.size.x']; w = r['[LiDARBoxComponent].box.size.y']; h = r['[LiDARBoxComponent].box.size.z']
                    x = r['[LiDARBoxComponent].box.center.x']; y = r['[LiDARBoxComponent].box.center.y']
                    z = r['[LiDARBoxComponent].box.center.z'] - h/2
                    xr, yr, zr, _ = (T_full @ np.array([x,y,z,1.0])).tolist()
                    ry = -float(r['[LiDARBoxComponent].box.heading']) - math.pi/2
                    cls = CLASS_MAP[typ]
                    out_lines.append(f'{cls} 0 0 -10 {bb[0]:.2f} {bb[1]:.2f} {bb[2]:.2f} {bb[3]:.2f} '
                                     f'{h:.2f} {w:.2f} {l:.2f} {xr:.2f} {yr:.2f} {zr:.2f} {ry:.2f} {npts}')
                    report['objects'][cls] = report['objects'].get(cls, 0) + 1
            with open(f'{OUT}/label/{sid}.txt', 'w') as f:
                f.write('\n'.join(out_lines) + ('\n' if out_lines else ''))
            if not out_lines: report['empty_frames'] += 1
            report['frames'] += 1
        print(f'[{fidx+1:3d}/202] {seg[:40]:40} frames={len(ts_sorted)}', flush=True)

    with open(f'{OUT}/conversion_report.json', 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    assert report['frames'] == 39848, report['frames']
    assert report['mapping_errors'] == 0, report['mapping_errors']
    print('OK: 39,848 frames, mapping exact.')

if __name__ == '__main__':
    main()

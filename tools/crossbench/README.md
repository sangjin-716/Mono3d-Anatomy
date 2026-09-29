# Waymo and nuScenes check

These scripts produce supplementary Sec. P and the "Other Benchmarks" paragraph in Sec. 5.1. The
summary with sources is in `reports/extensions/crossbench_audit.md`.

This is a preliminary check, not a replication. We use one released checkpoint per detector (on
Waymo, its released predictions) and report no confidence intervals. Only step 2 has published
numbers to compare against, so we do not compare the step-4 magnitudes with KITTI or across
detectors.

The scripts need the external data and environments listed at the end. We have not re-run them
since moving them into this repository, so please open an issue if one fails.

## Pipeline

### 1. Predictions

Waymo: we use the released Waymo val predictions of GUPNet and DEVIANT (from the DEVIANT release,
runs 1050 and 1051) and of MonoRCNN++. We did not re-run these checkpoints, so the Waymo gate in
step 2 re-evaluates their released outputs and checks our ground-truth conversion and evaluator.

nuScenes:

- FCOS3D and PGD: mmdetection3d `tools/test.py` with `configs/fcos3d_dump_full.py` and
  `configs/pgd_dump_full.py`. These are the official configs with the evaluator replaced by
  `DumpResults`.
- CenterNet: CenterTrack's `nuScenes_3Ddetection_e140` checkpoint, run with `test.py ddd`.
- EPro-PnP-Det (basic): `run_epropnp_infer.sh`.
- Environments: `setup_nuscenes_stage1.sh`, `setup_centertrack3.sh`, `setup_epropnp.sh`.

### 2. Reproduce the published numbers

Each release is first run through the unmodified official evaluator.

Waymo:

- `waymo_v2_to_kitti_gt.py` builds the GT from the v2 parquet release. `gt_adversarial_gates.py`
  checks it.
- `run_gate.sh` runs the evaluator with `waymo_eval_patched.py` (IoU 0.7) or
  `waymo_eval_0_5_patched.py` (IoU 0.5).
- `gate_deltas_v2.py` prints the differences to the published numbers.

nuScenes:

- PGD goes through `run_repair.sh` and FCOS3D through `nusc_official_reformat.py`, both using the
  official mmdet3d formatter.
- CenterNet is evaluated with `nusc_official_eval.py` (run from `night_queue.sh`) and EPro-PnP-Det
  with `xds_nusc_official_eval.py`, each on the json its release writes.
- `pgd_bisect_eval.py` and `nds_residual.py` locate PGD's extra NDS in the TP-error terms.

`campkl_to_nusc_json.py` and `run_ib_gates.sh` are an older hand-written path that we replaced. No
number in the paper uses them.

### 3. Per-camera box pools

- `ct_to_oracle_pkl.py` converts the CenterTrack json. CenterTrack stores the box top as
  `translation`, and the script accounts for that.
- `xds_nusc_json_to_oracle_pkl_FIXED.py` converts jsons that follow the nuScenes format
  (EPro-PnP-Det, FCOS3D, PGD).
- Checks: `align_check.py`, `align_check2.py`, `xds_*_check.py`, `gate_content_equality.py`.
- Boxes per image: `pool_census.py`, and `pred_census.py` through `run_census.sh`.

### 4. True-IoU re-sort

Each Car pool is kept as is and re-sorted by true IoU, at IoU3D 0.7 with all-point AP.

- Waymo: `waymo_core_analysis.py`, with this repository's evaluator.
- nuScenes: `nusc_oracle.py`, with the mmdet3d 3D IoU. It is run by `run_chain_fcos3d_oracle.sh`,
  `run_chain_pgd_oracle.sh` and `run_ct_oracle_chain2.sh`.

## External data and code

None of this is included in the repository.

- Waymo Open Dataset v2 validation parquet (`lidar_box`, `projected_lidar_box`,
  `camera_calibration`, `vehicle_pose`) under `paths.WAYMO_ROOT/waymo_v2/validation`. Also under
  `paths.WAYMO_ROOT`: the converted GT in `waymo_kitti/validation/`, a copy of it in the layout the
  evaluator reads in `waymo_gtorg/validation_org/<segment>/label_0/`, and the predictions in
  `predictions/`.
- nuScenes v1.0-trainval (val split, 6,019 samples) at `paths.NUSC_ROOT`, with mmdet3d's
  `nuscenes_infos_val.pkl`.
- Under `paths.UPSTREAM_ROOT`: DEVIANT (Waymo `ImageSets` and the released Waymo runs),
  mmdetection3d 1.4.0 (FCOS3D, PGD), EPro-PnP (EPro-PnP-Det), CenterTrack, and `checkpoints/`.

The shell scripts read these paths through `_paths.sh`. Set `PYTHON` to the interpreter of the
environment the script needs.

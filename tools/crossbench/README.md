# Waymo and nuScenes check

Preliminary check behind supplementary Sec. P and the "Other Benchmarks" paragraph in Sec. 5.1.
Results and scope: [`reports/extensions/crossbench_audit.md`](../../reports/extensions/crossbench_audit.md).
Not re-run since moving into this repository; please open an issue if a script fails.

## Steps

1. Predictions. Waymo: released val predictions of GUPNet and DEVIANT (DEVIANT runs 1050, 1051)
   and MonoRCNN++. nuScenes: FCOS3D and PGD via mmdet3d `tools/test.py` with
   `configs/{fcos3d,pgd}_dump_full.py`, CenterNet via CenterTrack's `nuScenes_3Ddetection_e140` checkpoint and `test.py ddd`, EPro-PnP-Det via
   `run_epropnp_infer.sh`.
2. Published numbers. Waymo: `waymo_v2_to_kitti_gt.py`, `gt_adversarial_gates.py`, `run_gate.sh`
   (`waymo_eval_patched.py` IoU 0.7, `waymo_eval_0_5_patched.py` IoU 0.5), `gate_deltas_v2.py`.
   nuScenes: `run_repair.sh` (PGD), `nusc_official_reformat.py` (FCOS3D), `nusc_official_eval.py`
   via `night_queue.sh` (CenterNet), `xds_nusc_official_eval.py` (EPro-PnP-Det), then
   `pgd_bisect_eval.py` and `nds_residual.py`.
3. Box pools. `ct_to_oracle_pkl.py` (CenterTrack), `xds_nusc_json_to_oracle_pkl_FIXED.py` (others).
   Checks: `align_check*.py`, `xds_*_check.py`, `gate_content_equality.py`. Census: `pool_census.py`,
   `pred_census.py` via `run_census.sh`.
4. True-IoU re-sort (Car, IoU3D 0.7, all-point AP). Waymo: `waymo_core_analysis.py`. nuScenes:
   `nusc_oracle.py` via `run_chain_fcos3d_oracle.sh`, `run_chain_pgd_oracle.sh`,
   `run_ct_oracle_chain2.sh`.

`campkl_to_nusc_json.py` and `run_ib_gates.sh` are an older path. No paper number uses them.

## External data (not included)

- In `paths.WAYMO_ROOT`: Waymo Open Dataset v2 val parquet (`waymo_v2/validation`), converted GT
  (`waymo_kitti/validation/`, `waymo_gtorg/validation_org/`) and predictions (`predictions/`).
- nuScenes v1.0-trainval val at `paths.NUSC_ROOT`, with mmdet3d `nuscenes_infos_val.pkl`.
- In `paths.UPSTREAM_ROOT`: DEVIANT, mmdetection3d 1.4.0, EPro-PnP, CenterTrack, `checkpoints/`.

Environments: `setup_nuscenes_stage1.sh`, `setup_centertrack3.sh`, `setup_epropnp.sh`. Shell scripts
read paths through `_paths.sh`. Set `PYTHON` to the interpreter of the environment a script needs.

The two `waymo_eval_*patched.py` files are adapted from DEVIANT (MIT, `LICENSE-DEVIANT`).
The Waymo check uses the Waymo Open Dataset, provided by Waymo LLC under the [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms).

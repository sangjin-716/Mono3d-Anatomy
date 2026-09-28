# adapters — per-detector dump generators

Each adapter loads a detector's **released checkpoint** in its **own upstream repository** and
writes one row per native prediction of the KITTI val split (Chen split, 3769 images). These
dumps are the inputs of every analysis in this repository (see `data/DUMPS.md`; the released
files are listed in `tools/_release.py`). Nothing here needs to be re-run to reproduce the
paper: the reports are recomputed from the released dumps. The adapters are provided so that
the dumps themselves can be audited or regenerated.

**Environment.** The adapters import the upstream packages (`lib.*`, `config`, `model`, ...), so
each one runs in that detector's own Python/PyTorch environment with its CUDA extensions built
(DCNv2, MSDeformAttn, ...), with the clone given by `--repo` (default
`<paths.UPSTREAM_ROOT>/<Repo>`). They were **not re-run for the release** (they need the upstream
environments and a GPU); every `.py` here is byte-compiled only. Outputs default to
`<paths.OUT_DIR>/dumps/`, so a re-run never overwrites a downloaded dump.

**Schema.** 23 columns: `sid, pred_idx, cls, sigma, log_sigma_raw, V, z_pred, bbox_h_pix,
bbox_w_pix, bbox_area_pix, x_3d, y_3d, z_3d, h_3d, w_3d, l_3d, ry, alpha, bbox_x1, bbox_y1,
bbox_x2, bbox_y2, dup_rank`. `cls` is the detector's thresholding score, `V` its native ranking
score, `sigma`/`log_sigma_raw` its depth-uncertainty output (constant 1/0 where there is none),
boxes are in camera coordinates with `y_3d` at the bottom centre, and `dup_rank` counts
same-image Car predictions with 2D IoU >= 0.5 and a higher `V`. `dgp_val.csv` and
`official_monocop_val.csv` (written by `dgp_cop_dump.py`) carry 10 further columns from a matching
against the KITTI Car labels (`max_iou_3d, max_iou_bev, matched_gt_idx, tp_label_iou07, gt_x,
gt_y, gt_z, depth_err, center_err_bev, dist_bin_4`; defined in `data/DUMPS.md`). No script that
writes a frozen report reads them.

**Gates** (supplementary, "Validation Gates"; `reports/detector_adapters_gates.md`).
G1 *reproduction*: the released checkpoint, run through its own repository, reproduces the
published Car Moderate AP_R40 (val). G2 *tap equivalence*: the dump reconstructs the native
output box for box (per-image counts, scores, geometry).

## Per detector

| Detector | Upstream repo @ commit | Checkpoint | Adapter (dump) | G1 official / ours (Mod AP_R40) | G2 tap equivalence |
|---|---|---|---|---|---|
| M3D-RPN | github.com/garrickbrazil/M3D-RPN @ `bf204e3f95f6` | released val1 model (repo README) | `m3drpn_dump_accv.py` -> `m3drpn_val.csv`; `m3drpn_dump_floor0.py` -> complete pre-NMS pool `m3drpn_val_floor0.csv` (2.9 GB, release asset `mono3d_anatomy_m3drpn_complete_pool_v1.zip`); `test_rpn_3d_accv.py` = native run for G1; patch `patches/M3D-RPN_rpn_util_py_cpu_nms.patch` (torch-1.x port, same +1-offset NMS) | 11.07 (R40 re-eval; paper reports R11) / 11.07 | native-NMS reconstruction 14.538/11.099/8.670 vs native 14.531/11.073/8.646 (dump floor 0.05, cap 300/img) |
| MonoDLE | github.com/xinzhuma/monodle @ `e426aa65fdc7` | released checkpoint (repo README) | `monodle_dump.py` + `configs/monodle/kitti_accv_eval.yaml` | 13.72 / 14.57 (+0.85, inference-environment drift; code, data, evaluator verified identical) | box-level exact (native writer rounds scores to 2 dp) |
| MonoFlex\* | github.com/zhangyp15/MonoFlex @ `ec6da017c325` | released checkpoint (repo README) | `monoflex_dump_orig.py` (+ `mf_paths_catalog.py`) in the original environment built by `build_monoflex_orig_env.sh` -> `monoflex_orig_val.csv`. Modern-environment dump: `monoflex_dump.py` -> `monoflex_modern_val.csv` | 17.51 / 17.34 (original torch-1.4 env; the modern rebuild gives 15.54) | same decode as the native PostProcessor (top-50, threshold 0), no separate numeric record |
| GUPNet | github.com/SuperMHP/GUPNet @ `d0e02cad228f` | released checkpoint (repo README) | `gupnet_dump.py`; `gupnet_native_eval.py` = native tester run graded by our evaluator | 16.46 / 16.48 | native tester output vs dump, graded by the same evaluator (`gupnet_native_eval.py`) |
| DEVIANT | github.com/abhi1kumar/DEVIANT @ `2e6eca6e27d7` | released `run_221` checkpoint (repo README) | `deviant_dump.py` | 16.54 / 16.49 | boxes + scores exact (native writer rounds to 2 dp) |
| MonoGround\* | github.com/cfzd/MonoGround @ `05b3baf73228` | released checkpoint (repo README) | `monoground_dump_orig.py` in the original environment (`build_monoflex_orig_env.sh`) -> `monoground_orig_val.csv`. Modern-environment dump: `monoground_dump.py` -> `monoground_modern_val.csv` | 18.69 / 18.69 (original torch-1.4 env; the modern rebuild gives 16.79) | same decode as the native PostProcessor, no separate numeric record |
| MonoCon | github.com/2gunsu/monocon-pytorch @ `908807bdd8d4` | released checkpoint of this re-implementation (repo README) | `monocon_dump.py`; patch `patches/MonoCon_base_engine_map_location.patch` (checkpoint load `map_location`) | 19.02 / 19.02 | 26.031/19.015/15.912 vs native (d <= 0.007); native top-k 30, threshold 0.4 |
| MonoDETR | github.com/ZrrSkywalker/MonoDETR @ `6994b9f51240` | released checkpoint (repo README) | `monodetr_dump.py`; per-hypothesis pool `monodetr_preflatten_dump.py` -> `monodetr_val_preflatten.csv` | 20.83 / 20.83 | all 50x3 query-class hypotheses reproduce the native top-50 |
| MonoDGP | github.com/PuFanqi23/MonoDGP @ `aa059a18214a` | released checkpoint (repo README) | `dgp_cop_dump.py` -> `dgp_val.csv` (33 col); per-query pool `detr_preflatten_dump.py` -> `monodgp_val_preflatten.csv` | 22.34 / 22.29 | as MonoDETR |
| MonoCoP | MonoCoP authors' release (project page alanzhangcs.github.io/monocop-page); our clone has no git metadata, commit not recorded | huggingface.co/zhihao406/MonoCoP | `dgp_cop_dump.py` -> `official_monocop_val.csv` (33 col); per-query pool `detr_preflatten_dump.py` -> `official_monocop_val_preflatten.csv` | 23.89 / 23.84 | as MonoDETR |
| MonoCLUE | github.com/SungHunYang/MonoCLUE @ `016d3e8d3c99` | released checkpoint (repo README) | `monoclue_dump.py`; per-hypothesis pool `detr_preflatten_native.py` -> `monoclue_val_preflatten.csv` | 24.10 / 24.20 | as MonoDETR (native decode) |
| MonoIA | github.com/alanzhangcs/MonoIA @ `69d6ee30ca5e` | huggingface.co/zhihao406/MonoIA (`MonoIA_KITTI_Val.pth`) | `monoia_dump.py`; per-hypothesis pool `detr_preflatten_native.py` -> `monoia_val_preflatten.csv` | 24.40 / 24.48 | as MonoDETR (native decode) |

The five `*_val_preflatten.csv` pools are the release asset `mono3d_anatomy_query_complete_pools_v1.zip`
(`data/DUMPS.md`). G1 values are those of the supplementary reproduction table. Rows marked \* are the panel
entries built in the authors' original environment (Python 3.7, torch 1.4.0, CUDA 10.1, DCNv2
compiled against it; `build_monoflex_orig_env.sh`). The modern-environment rebuild of the two
MonoFlex-codebase repos drifts by about -1.9 AP (`reports/detector_adapters_gates.md`, note 2);
the `*_modern` dumps are released for the auxiliary checks that used them.

## Honest caveats

- **Needs the upstream environment.** Every adapter needs its repository,
  environment, compiled CUDA ops and checkpoint. The original runs used torch 1.10 (DEVIANT,
  MonoDLE, MonoCon, M3D-RPN, modern MonoFlex/MonoGround), torch 1.9 (GUPNet, MonoDETR family) and
  torch 1.4 (\* rows). Results can drift across environments (see MonoDLE and the modern
  MonoFlex/MonoGround rebuilds).
- **MonoCLUE and MonoIA 23-column dumps** were written by one-off scripts that were not kept as
  files; `monoclue_dump.py` and `monoia_dump.py` are recovered from the recorded text of those
  scripts (including the one recorded edit to the MonoIA script) with only the paths turned
  into arguments. They have not been re-run since recovery.
- **Configs.** Where the original runs used a copy of an upstream config whose only change was
  an absolute dataset root (or, for MonoIA, the focal-list entries), the adapter applies that
  change in memory or documents it in its header; the MonoDLE eval config is shipped in
  `configs/monodle/`.
- **MonoCoP commit.** The MonoCoP clone was used without git metadata, so no commit hash can be
  given.
- **Tap equivalence for MonoFlex\*/MonoGround\*** is by construction (the adapter calls the
  repo's own PostProcessor with threshold 0); G1 for these two was established on the repo's own
  evaluation in the original environment (`build_monoflex_orig_env.sh`, step 4).

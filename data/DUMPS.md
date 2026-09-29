# Data

Per-prediction outputs ("dumps") of twelve released detectors on KITTI val (Chen split, 3,769 images),
made from each released checkpoint with its own inference code (see [`adapters/README.md`](../adapters/README.md)).

| Asset (release v1.0) | Content | Zip | Unzipped |
|---|---|---:|---:|
| `mono3d_anatomy_dumps_v1.zip` | 14 per-prediction dumps (needed) | 443 MB | 1.01 GB |
| `mono3d_anatomy_query_complete_pools_v1.zip` | full pools of the 5 query-based detectors, before top-50 | 276 MB | 977 MB |
| `mono3d_anatomy_m3drpn_complete_pool_v1.zip` | full M3D-RPN pool, before NMS | 1.20 GB | 2.87 GB |
| `mono3d_anatomy_matched_tables_v1.zip` | prediction-to-GT matched tables of 6 detectors | 26.6 MB | 58.3 MB |

## Download

```bash
mkdir -p data/dumps && cd data/dumps    # DUMP_DIR in paths.py
BASE=https://github.com/sangjin-716/Mono3d-Anatomy/releases/download/v1.0
wget $BASE/mono3d_anatomy_dumps_v1.zip                 # needed
wget $BASE/mono3d_anatomy_query_complete_pools_v1.zip  # optional
wget $BASE/mono3d_anatomy_m3drpn_complete_pool_v1.zip  # optional
wget $BASE/mono3d_anatomy_matched_tables_v1.zip        # optional
sha256sum -c --ignore-missing <<'SUMS'
3eb2d46701f481851b89f6f52c2472805d0467c5b8963fdd8b7d861ab4262edd  mono3d_anatomy_dumps_v1.zip
0a4696a30a166d4b3a0cbfeab92cbaa09ac91ab6769a15b90e45e90ea462362f  mono3d_anatomy_query_complete_pools_v1.zip
5e0bbce9d2bd06eaea8a5954ad4202aa7541a862e995f7137986b4625aa77497  mono3d_anatomy_m3drpn_complete_pool_v1.zip
47d71716b77a7e018bc9ced001c313448ffe3935a1e92c37ea4e7a2609ba8e83  mono3d_anatomy_matched_tables_v1.zip
SUMS
for z in mono3d_anatomy_*_v1.zip; do unzip -n "$z" -x 'MD5SUMS*.txt'; done
md5sum -c --ignore-missing ../MD5SUMS.txt
cd ../..
python tools/_release.py    # which assets are complete in DUMP_DIR
```

`data/MD5SUMS.txt` has the MD5 of all 26 files. Reports tag each dump as `dump@<8 hex>`, the MD5 of its first 1 MiB.

## Optional assets

| Asset | Needed by |
|---|---|
| both pools | `tools/decomp/`: `e4_fp_tp_decomp.py`, `exp1_true_ceiling.py`, `c2_iou05_separation.py`, `diffsweep_headroom_sep.py`, `pool_waterfall.py`, `e12_replacement.py`, `gt_state_matrix.py`<br>`tools/orig/gt_state_matrix_vB.py`<br>`tools/extensions/`: `budget_matched.py`, `budget_saturation.py`, `pool_census.py`, `nearmiss_panelwide.py`, `difficulty_class_extension.py`, `gt_state_recount.py` (run after the two `gt_state_matrix*.py`) |
| query pools | `tools/decomp/detr_native_sweep.py`, `tools/extensions/class_capability.py` (Part B) |
| M3D-RPN pool | `tools/decomp/e2_budget_truncation.py` |
| matched tables | `tools/decomp/c3_conditioned.py` (skips a missing table silently; its report should list 6 detectors and 15 pairs) |

## Dumps

| File | Detector | Rows | File | Detector | Rows |
|---|---|---:|---|---|---:|
| `m3drpn_val.csv` | M3D-RPN | 794,481 | `monodle_val.csv` | MonoDLE | 188,450 |
| `monoflex_orig_val.csv` | MonoFlex\* | 105,945 | `gupnet_val.csv` | GUPNet | 114,224 |
| `deviant_val.csv` | DEVIANT | 128,576 | `monoground_orig_val.csv` | MonoGround\* | 104,552 |
| `monocon_val.csv` | MonoCon | 88,089 | `monodetr_val.csv` | MonoDETR | 185,676 |
| `dgp_val.csv` | MonoDGP | 188,450 | `official_monocop_val.csv` | MonoCoP | 188,450 |
| `monoclue_val.csv` | MonoCLUE | 188,450 | `monoia_val.csv` | MonoIA | 188,450 |
| `monoflex_modern_val.csv` | MonoFlex (modern rebuild) | 105,944 | `monoground_modern_val.csv` | MonoGround (modern rebuild) | 104,552 |

- MonoCon uses the 2gunsu/monocon-pytorch re-implementation and its checkpoint.
- MonoFlex\* and MonoGround\* are the panel versions, run in the authors' torch-1.4 environment (`tools/orig/`, `reports_orig/`). The modern rebuilds feed `reports/` and the class, near-miss and disagreement checks (stems `monoflex`, `monoground` in `tools/_release.py`).

## Columns

One row per candidate Car box. The first five rows are the 23 columns of every dump. The last four
are 10 extra columns only in `dgp_val.csv` and `official_monocop_val.csv` (9 from a GT matching, `dist_bin_4` from the predicted depth) (written by
`adapters/dgp_cop_dump.py`; the report scripts redo the matching from the KITTI labels instead).

| Columns | Meaning |
|---|---|
| `sid`, `pred_idx`, `cls`, `V` | KITTI sample id (index in `ImageSets/val.txt`), candidate index in the image, Car score, native ranking score |
| `sigma`, `log_sigma_raw` | depth-uncertainty term in `V` and its raw head output (1.0 and 0.0 without one; see `adapters/*_dump*.py`) |
| `z_pred`, `x_3d`, `y_3d`, `z_3d`, `h_3d`, `w_3d`, `l_3d`, `ry`, `alpha` | depth, 3D location (KITTI convention, `y_3d` at the bottom centre) and size (camera coordinates, m), rotation around y and observation angle (rad) |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`, `bbox_h_pix`, `bbox_w_pix`, `bbox_area_pix` | 2D box (pixels) |
| `dup_rank` | Car candidates in the same image with 2D IoU >= 0.5 and higher `V` |
| `max_iou_3d`, `max_iou_bev` | best IoU with any Car GT in the image (0 if none) |
| `matched_gt_idx`, `tp_label_iou07` | index into the image's KITTI label file and TP label from greedy matching in descending `V` at 3D IoU >= 0.7 (-1 and 0 if unmatched) |
| `gt_x`, `gt_y`, `gt_z`, `depth_err`, `center_err_bev` | matched GT location, abs(`z_pred` - `gt_z`), BEV centre error (m; empty if unmatched) |
| `dist_bin_4` | `0-15`, `15-30`, `30-45`, `45+` m on predicted `z_3d` |

- Query pools (31 or 33 columns, from `adapters/monodetr_preflatten_dump.py`, `adapters/detr_preflatten_dump.py`, `adapters/detr_preflatten_native.py`): all 50 queries per image with decoded box, class scores and stage flags `flat_rank`, `flat_rank_car`, `native_top50`, `thr_pass`, `in_final`. `monodetr_val_preflatten.csv`, `monoclue_val_preflatten.csv` and `monoia_val_preflatten.csv` have one row per query x class (565,350 rows); `monodgp_val_preflatten.csv` and `official_monocop_val_preflatten.csv` one per query, Car in `cls_car`, `V_car` (188,450 rows).
- `m3drpn_val_floor0.csv` (8,012,124 rows, from `adapters/m3drpn_dump_floor0.py`): the 23 dump columns for up to 3,000 pre-NMS boxes per image.
- Matched tables, each prediction matched to its best Car GT at 3D IoU >= 0.05: `matched_monodetr_val.csv` (26,136 rows), `matched_dgp_val.csv` (38,362), `matched_cop_val.csv` (36,572), `matched_monoia_val.csv` (27,899), `matched_monoflex_val.csv` (17,521, modern rebuild), `matched_gupnet_val.csv` (16,541). `c3_conditioned.py` reads `sid`, `z_pred`, `gt_z`, `iou3d`, `occ`, `trunc`, `dist_bin` (binned by `gt_z`). No MonoCLUE table; the build script is not released.

## License

Model outputs on KITTI images (KITTI is CC BY-NC-SA 3.0), released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). KITTI label files are not included, and no Waymo or nuScenes data or predictions are redistributed.
Please cite the paper, KITTI (Geiger et al., CVPR 2012) and the original detector papers.

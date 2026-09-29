# Data

The analyses run on per-prediction outputs ("dumps") of twelve released monocular 3D detectors on
KITTI val (Chen split, 3,769 images). Each dump was made from a released checkpoint with the
inference code of the repository that released it. For MonoCon this is the 2gunsu/monocon-pytorch
re-implementation and its checkpoint (see `adapters/README.md`). The twelve panel dumps pass the
reproduction and native-match checks listed in `adapters/README.md`, which also describes how each
one was made.

The files are too large for git, so they are attached to the GitHub release v1.0 as four zips:

| Asset | Content | Zip | Unzipped |
|---|---|---:|---:|
| `mono3d_anatomy_dumps_v1.zip` | the 14 per-prediction dumps (needed) | 443 MB | 1.01 GB |
| `mono3d_anatomy_query_complete_pools_v1.zip` | full candidate pools of the five query-based detectors, before the top-50 flatten | 276 MB | 977 MB |
| `mono3d_anatomy_m3drpn_complete_pool_v1.zip` | full candidate pool of M3D-RPN, before its NMS | 1.20 GB | 2.87 GB |
| `mono3d_anatomy_matched_tables_v1.zip` | prediction-to-ground-truth matched tables of six detectors | 26.6 MB | 58.3 MB |

## Download

Unpack everything into one folder, `DUMP_DIR` in `paths.py` (default `data/dumps/`). Several
scripts read the pools and the matched tables from there directly.

```bash
mkdir -p data/dumps && cd data/dumps
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
python tools/_release.py
```

`tools/_release.py` tells you, per asset, whether all of its files are in `DUMP_DIR`. Each zip also
has its own MD5 list, which the unzip line skips. `data/MD5SUMS.txt` has the MD5 of all 26 files,
so `md5sum` prints one OK per file (26 with all four assets).

`reports/gap_exact.txt` and `reports_orig/gap_exact_orig.txt` tag each dump they read as
`dump@<8 hex>`. The tag is the MD5 of the first 1 MiB of the file,
`hashlib.md5(open(f, "rb").read(1 << 20)).hexdigest()[:8]`. All 14 tags match the released files.

## Which scripts need which asset

The per-prediction dumps are read by the analysis scripts and by the real-dump checks in
`evaluator/exact_ap.py`. The other three assets are only needed for these scripts:

| Asset | Scripts |
|---|---|
| query pools and M3D-RPN pool | `tools/decomp/`: `e4_fp_tp_decomp.py`, `exp1_true_ceiling.py`, `c2_iou05_separation.py`, `diffsweep_headroom_sep.py`, `pool_waterfall.py`, `e12_replacement.py`, `gt_state_matrix.py`<br>`tools/orig/gt_state_matrix_vB.py`<br>`tools/extensions/`: `budget_matched.py`, `budget_saturation.py`, `pool_census.py`, `nearmiss_panelwide.py`, `difficulty_class_extension.py`, `gt_state_recount.py` |
| query pools only | `tools/decomp/detr_native_sweep.py`, `tools/extensions/class_capability.py` (Part B) |
| M3D-RPN pool only | `tools/decomp/e2_budget_truncation.py` |
| matched tables | `tools/decomp/c3_conditioned.py` |

`gt_state_recount.py` reads the per-GT CSVs written by `gt_state_matrix.py` and
`gt_state_matrix_vB.py`, so run those first. The rest of `tools/orig/` only looks at MonoFlex\*
and MonoGround\* and needs only the per-prediction dumps. `c3_conditioned.py` skips a missing
matched table without an error, so check that its report lists all six detectors and 15 pairs.

## Per-prediction dumps

These are in `mono3d_anatomy_dumps_v1.zip`.

| Detector | File | Rows | MD5 |
|---|---|---:|---|
| M3D-RPN | `m3drpn_val.csv` | 794,481 | `6565108cf91398cd5e3c597d3bb0526a` |
| MonoDLE | `monodle_val.csv` | 188,450 | `dc7f30f52dac642dc2edc05b4d930531` |
| MonoFlex\* | `monoflex_orig_val.csv` | 105,945 | `a7471ba61d603bc872f2f19df89ff4b6` |
| GUPNet | `gupnet_val.csv` | 114,224 | `1e45419a5e0a4f07e94f6abd40d62503` |
| DEVIANT | `deviant_val.csv` | 128,576 | `9abd7c859a05924761da25d45017bf28` |
| MonoGround\* | `monoground_orig_val.csv` | 104,552 | `8b24830fb47127d45e9d33f6d2b1a395` |
| MonoCon | `monocon_val.csv` | 88,089 | `2f78fc5aaaf6cdddeeebb80f13a088f0` |
| MonoDETR | `monodetr_val.csv` | 185,676 | `ae6c33c3beb3c3d6959be3b938d0bc0f` |
| MonoDGP | `dgp_val.csv` | 188,450 | `5f2d7300f6309b06c014d418045e1fa8` |
| MonoCoP | `official_monocop_val.csv` | 188,450 | `613c40ea9f9dbfc2f20cfe0aacff54ba` |
| MonoCLUE | `monoclue_val.csv` | 188,450 | `8b8162612e1ad829fa1e6cc79feafefd` |
| MonoIA | `monoia_val.csv` | 188,450 | `ca29b8d99d5aa7f0aba382def426bc54` |
| MonoFlex (modern rebuild) | `monoflex_modern_val.csv` | 105,944 | `306eaa334074383459c16a2364e9b994` |
| MonoGround (modern rebuild) | `monoground_modern_val.csv` | 104,552 | `878ee0a93903d4c04b9e5a3b08780ec6` |

The first twelve are the paper's panel. MonoFlex and MonoGround reproduce their published numbers
only in the authors' original torch-1.4 environment, so the panel uses those outputs (marked \*),
and `tools/orig/` and `reports_orig/` hold the analyses on them. The last two files are the same
checkpoints run in a modern environment. `reports/` was computed on them, and so were a few
auxiliary checks: the class and near-miss checks and the cross-detector disagreement analysis. In
`tools/_release.py` the stems `monoflex` and `monoground` point to the modern files, and
`monoflex_orig` and `monoground_orig` to the original-environment ones.

### Columns

Each row is one candidate Car box. Every dump has these 23 columns:

| Column | Meaning |
|---|---|
| `sid` | KITTI sample id (the image index in `ImageSets/val.txt`) |
| `pred_idx` | index of the candidate within its image |
| `cls` | the detector's Car classification score |
| `sigma`, `log_sigma_raw` | the depth-uncertainty term used in the native score, and the raw head output it comes from. Constant 1.0 and 0.0 for detectors without an uncertainty head. Per-detector definitions are in the header of each adapter script (`adapters/*_dump*.py`) |
| `V` | the detector's native ranking score, the one its own evaluation sorts by |
| `z_pred` | predicted depth of the 3D centre (m) |
| `bbox_h_pix`, `bbox_w_pix`, `bbox_area_pix` | 2D box height, width and area (pixels) |
| `x_3d`, `y_3d`, `z_3d` | 3D box location in camera coordinates (KITTI convention, m) |
| `h_3d`, `w_3d`, `l_3d` | 3D box dimensions (m) |
| `ry`, `alpha` | rotation around the camera y axis and observation angle (rad) |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | 2D box corners (pixels) |
| `dup_rank` | number of Car candidates in the same image with 2D IoU >= 0.5 and a higher `V` |

`dgp_val.csv` (MonoDGP) and `official_monocop_val.csv` (MonoCoP) have 10 more columns, 33 in total.
Their adapter, `adapters/dgp_cop_dump.py`, wrote them from a matching against the KITTI Car
labels. Nine come from the ground truth. `dist_bin_4` only bins the predicted depth.

| Column | Meaning |
|---|---|
| `max_iou_3d`, `max_iou_bev` | largest 3D and BEV IoU of the box with any Car ground truth in its image, using the official KITTI IoU kernels. 0 in images without a Car |
| `matched_gt_idx` | index of the ground-truth object in the image's KITTI label file, assigned by greedy matching in descending `V` at 3D IoU >= 0.7 with one box per object. -1 if none |
| `tp_label_iou07` | 1 if the box got a ground truth in that matching, else 0 |
| `gt_x`, `gt_y`, `gt_z` | 3D location of the assigned ground truth (camera coordinates, m). Empty if none |
| `depth_err` | abs(`z_pred` - `gt_z`) (m). Empty if none |
| `center_err_bev` | distance between (`x_3d`, `z_3d`) and (`gt_x`, `gt_z`) (m). Empty if none |
| `dist_bin_4` | `0-15`, `15-30`, `30-45` or `45+` m, binned on the predicted `z_3d` |

The report scripts do not use these columns. They redo the matching from the KITTI labels.

## Complete candidate pools

The per-prediction dumps hold what each detector outputs after its own candidate budget. For the
query-based detectors and M3D-RPN, the generation-versus-selection analyses need the full pool
before that budget. The CenterNet-style dumps already contain their full top-K pools.

| Asset | File | Rows | MD5 |
|---|---|---:|---|
| query pools | `monodetr_val_preflatten.csv` | 565,350 | `97792c45db4af69effe8c2a3e201c21c` |
| query pools | `monodgp_val_preflatten.csv` | 188,450 | `c228a44fbcc722157d9d1aefcb690b3d` |
| query pools | `official_monocop_val_preflatten.csv` | 188,450 | `9564d49b68d958dd34c60e81ecef00b9` |
| query pools | `monoclue_val_preflatten.csv` | 565,350 | `1eb46213f3ff53550e39b9c8af59d5de` |
| query pools | `monoia_val_preflatten.csv` | 565,350 | `ec629f9ec26163786cd47556e11c13fb` |
| M3D-RPN pool | `m3drpn_val_floor0.csv` | 8,012,124 | `c2ca44606c8b582db9d38dc6b7c40990` |

The `*_val_preflatten.csv` files (31 or 33 columns) have all 50 inference queries of every image,
with the decoded box, all class scores and the native stage flags: `flat_rank` and
`flat_rank_car` (rank among the 150 query x class hypotheses by raw class score), `native_top50`,
`thr_pass` and `in_final`. MonoDETR, MonoCLUE and MonoIA have one row per query x class hypothesis,
150 per image. MonoDGP and MonoCoP have one row per query, 50 per image, with the Car hypothesis
in `cls_car` and `V_car`. The columns are defined in the adapters that wrote the files:
`adapters/monodetr_preflatten_dump.py`, `adapters/detr_preflatten_dump.py` and
`adapters/detr_preflatten_native.py`.

`m3drpn_val_floor0.csv` has the same 23 columns as the per-prediction dumps, for the whole
pre-NMS pool: up to 3,000 boxes per image with no score floor. It was written by
`adapters/m3drpn_dump_floor0.py`. Running M3D-RPN's own NMS on it gives back its evaluated output.

## Matched tables

These are in `mono3d_anatomy_matched_tables_v1.zip`.

| File | Detector | From dump | Rows | MD5 |
|---|---|---|---:|---|
| `matched_monodetr_val.csv` | MonoDETR | `monodetr_val.csv` | 26,136 | `d573c708005f3bdeb71da9badcf966a4` |
| `matched_dgp_val.csv` | MonoDGP | `dgp_val.csv` | 38,362 | `52001dc31640ef8ccaf3ada68ba67011` |
| `matched_cop_val.csv` | MonoCoP | `official_monocop_val.csv` | 36,572 | `529839aa01fa0ff178277264e75d9f66` |
| `matched_monoia_val.csv` | MonoIA | `monoia_val.csv` | 27,899 | `05c78d683b0b889c809d1acb458bf713` |
| `matched_monoflex_val.csv` | MonoFlex (modern rebuild) | `monoflex_modern_val.csv` | 17,521 | `15600437fa94cbf143d929b86ee301f3` |
| `matched_gupnet_val.csv` | GUPNet | `gupnet_val.csv` | 16,541 | `5b037376164bdc91890be927138c82c0` |

Each row is one prediction from the listed dump (its `sid`, `z_pred` and `cls` appear there),
matched to its best Car ground truth by 3D IoU, with IoU >= 0.05. The only reader is
`tools/decomp/c3_conditioned.py`. It uses `sid`, `z_pred`, `gt_z` (ground-truth depth), `iou3d`,
`occ` and `trunc` (KITTI occlusion level and truncation of that ground truth), and `dist_bin`
(0-15, 15-30, 30-45, 45+ m by `gt_z`). The other columns are not used.

There is no MonoCLUE table, so the analysis leaves MonoCLUE out. The script that built the tables
is not part of this release.

## License

The files are model outputs on KITTI images. KITTI is under CC BY-NC-SA 3.0, so all four assets
are released under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/).

No Waymo or nuScenes predictions are redistributed. The four assets hold only the KITTI val files
listed above, and the repository itself has no Waymo or nuScenes data. The Waymo and nuScenes check
in `tools/crossbench/` needs local copies of those datasets and takes its predictions from the
detectors' own releases or runs (see [`tools/crossbench/README.md`](../tools/crossbench/README.md)).

The complete pools and twelve of the fourteen dumps contain predictions only. Two kinds of files
also carry values derived from the KITTI labels. In `dgp_val.csv` and `official_monocop_val.csv`
these are the nine ground-truth columns above. In the matched tables they are the depth, occlusion
level and truncation of the matched ground-truth box and the 3D IoU to it. The KITTI label files
themselves are not included. Get them from the official KITTI website.

Please cite the paper, KITTI (Geiger et al., CVPR 2012, https://www.cvlibs.net/datasets/kitti/) and the
original detector papers if you use the files.

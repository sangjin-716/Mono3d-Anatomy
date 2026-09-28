# Released data

The paper's analyses run on per-prediction dumps of twelve released monocular 3D detectors on the
KITTI validation split (Chen split, 3,769 images). Each dump was produced from the official released
checkpoint with the detector's own inference code (see `adapters/README.md`) and passed two gates:
it reproduces the published validation number, and it reconstructs the detector's native output box
for box (tap equivalence). The files are too large for git and are attached to the GitHub release
`v1.0` as four zip assets.

## Release assets

| Asset | Content | Zip | Unzipped | SHA-256 of the zip |
|---|---|---:|---:|---|
| `mono3d_anatomy_dumps_v1.zip` | the 14 per-prediction dumps (**required**) | 443 MB | 1.01 GB | `3eb2d46701f481851b89f6f52c2472805d0467c5b8963fdd8b7d861ab4262edd` |
| `mono3d_anatomy_query_complete_pools_v1.zip` | complete candidate pools of the five query-based detectors (every query x class hypothesis before the top-50 flatten) | 276 MB | 977 MB | `0a4696a30a166d4b3a0cbfeab92cbaa09ac91ab6769a15b90e45e90ea462362f` |
| `mono3d_anatomy_m3drpn_complete_pool_v1.zip` | complete candidate pool of M3D-RPN (up to 3,000 boxes per image before its NMS, no score floor) | 1.20 GB | 2.87 GB | `5e0bbce9d2bd06eaea8a5954ad4202aa7541a862e995f7137986b4625aa77497` |
| `mono3d_anatomy_matched_tables_v1.zip` | prediction-to-ground-truth matched tables of six detectors, the input of the cross-detector disagreement analysis | 26.6 MB | 58.3 MB | `47d71716b77a7e018bc9ced001c313448ffe3935a1e92c37ea4e7a2609ba8e83` |

(MB and GB are 10^6 and 10^9 bytes.) Unpack all assets into the same folder, `DUMP_DIR` in
`paths.py` (default `data/dumps/`): several scripts read the complete pools and the matched
tables from `DUMP_DIR` directly.

## Download and verify

```bash
# from the repository root
mkdir -p data/dumps && cd data/dumps
BASE=https://github.com/sangjin-716/Mono3d-Anatomy/releases/download/v1.0
wget $BASE/mono3d_anatomy_dumps_v1.zip                  # required
wget $BASE/mono3d_anatomy_query_complete_pools_v1.zip   # optional, see the table below
wget $BASE/mono3d_anatomy_m3drpn_complete_pool_v1.zip   # optional
wget $BASE/mono3d_anatomy_matched_tables_v1.zip         # optional

# 1) the zips (assets you did not download are skipped)
sha256sum -c --ignore-missing <<'EOF'
3eb2d46701f481851b89f6f52c2472805d0467c5b8963fdd8b7d861ab4262edd  mono3d_anatomy_dumps_v1.zip
0a4696a30a166d4b3a0cbfeab92cbaa09ac91ab6769a15b90e45e90ea462362f  mono3d_anatomy_query_complete_pools_v1.zip
5e0bbce9d2bd06eaea8a5954ad4202aa7541a862e995f7137986b4625aa77497  mono3d_anatomy_m3drpn_complete_pool_v1.zip
47d71716b77a7e018bc9ced001c313448ffe3935a1e92c37ea4e7a2609ba8e83  mono3d_anatomy_matched_tables_v1.zip
EOF

# 2) unpack; each zip also carries its own MD5 list, which is not needed here
for z in mono3d_anatomy_*_v1.zip; do unzip -n "$z" -x 'MD5SUMS*.txt'; done

# 3) the unpacked files against the repository manifest (one OK per file, 26 with all assets)
md5sum -c --ignore-missing ../MD5SUMS.txt
cd ../..
python tools/_release.py        # lists, per asset, whether all of its files are in DUMP_DIR
```

`data/MD5SUMS.txt` holds the full-file MD5 of all 26 files. `reports/gap_exact.txt` and
`reports_orig/gap_exact_orig.txt` also tag each per-prediction dump they read with `dump@<8 hex>`,
the MD5 of the first 1 MiB of the file (`hashlib.md5(open(f, "rb").read(1 << 20)).hexdigest()[:8]`).
All 14 tags match the released files.

## Which scripts need which asset

The per-prediction dumps are read by the analysis scripts and by the real-dump gates of
`evaluator/exact_ap.py`. The other three assets are needed only by these scripts:

| Asset | Scripts |
|---|---|
| query complete pools **and** M3D-RPN complete pool | `tools/decomp/`: `e4_fp_tp_decomp.py`, `exp1_true_ceiling.py`, `c2_iou05_separation.py`, `diffsweep_headroom_sep.py`, `pool_waterfall.py`, `e12_replacement.py`, `gt_state_matrix.py`; `tools/orig/gt_state_matrix_vB.py`; `tools/extensions/`: `budget_matched.py`, `budget_saturation.py`, `pool_census.py`, `nearmiss_panelwide.py`, `difficulty_class_extension.py`, and indirectly `gt_state_recount.py` (it reads the per-GT CSVs written by `gt_state_matrix.py` and `gt_state_matrix_vB.py`) |
| query complete pools only | `tools/decomp/detr_native_sweep.py`; `tools/extensions/class_capability.py` (Part B) |
| M3D-RPN complete pool only | `tools/decomp/e2_budget_truncation.py` |
| matched tables | `tools/decomp/c3_conditioned.py` |

The other `tools/orig/` scripts evaluate only MonoFlex\* and MonoGround\* and need only the
per-prediction dumps. `c3_conditioned.py` skips a matched table that is missing instead of
stopping, so check that its report lists all six detectors and 15 pairs.

## Per-prediction dumps (14 files, `mono3d_anatomy_dumps_v1.zip`)

| Detector | File | Rows | MD5 | Used for |
|---|---|---:|---|---|
| M3D-RPN | `m3drpn_val.csv` | 794,481 | `6565108cf91398cd5e3c597d3bb0526a` | panel |
| MonoDLE | `monodle_val.csv` | 188,450 | `dc7f30f52dac642dc2edc05b4d930531` | panel |
| MonoFlex\* | `monoflex_orig_val.csv` | 105,945 | `a7471ba61d603bc872f2f19df89ff4b6` | panel (original torch-1.4 environment) |
| GUPNet | `gupnet_val.csv` | 114,224 | `1e45419a5e0a4f07e94f6abd40d62503` | panel |
| DEVIANT | `deviant_val.csv` | 128,576 | `9abd7c859a05924761da25d45017bf28` | panel |
| MonoGround\* | `monoground_orig_val.csv` | 104,552 | `8b24830fb47127d45e9d33f6d2b1a395` | panel (original torch-1.4 environment) |
| MonoCon | `monocon_val.csv` | 88,089 | `2f78fc5aaaf6cdddeeebb80f13a088f0` | panel |
| MonoDETR | `monodetr_val.csv` | 185,676 | `ae6c33c3beb3c3d6959be3b938d0bc0f` | panel |
| MonoDGP | `dgp_val.csv` | 188,450 | `5f2d7300f6309b06c014d418045e1fa8` | panel |
| MonoCoP | `official_monocop_val.csv` | 188,450 | `613c40ea9f9dbfc2f20cfe0aacff54ba` | panel |
| MonoCLUE | `monoclue_val.csv` | 188,450 | `8b8162612e1ad829fa1e6cc79feafefd` | panel |
| MonoIA | `monoia_val.csv` | 188,450 | `ca29b8d99d5aa7f0aba382def426bc54` | panel |
| MonoFlex (modern rebuild) | `monoflex_modern_val.csv` | 105,944 | `306eaa334074383459c16a2364e9b994` | frozen reports in `reports/`, auxiliary checks |
| MonoGround (modern rebuild) | `monoground_modern_val.csv` | 104,552 | `878ee0a93903d4c04b9e5a3b08780ec6` | frozen reports in `reports/`, auxiliary checks |

The two detectors that share the MonoFlex codebase reproduce their published numbers only in the
authors' original torch-1.4 environment. The paper's panel uses those outputs (marked \*); the
recomputations on them are in `reports_orig/`. The modern-environment rebuilds are released too,
because the frozen reports in `reports/` and some auxiliary checks (the class and near-miss checks,
the cross-detector disagreement) were computed on them. `tools/_release.py` maps the stems
`monoflex` / `monoground` to the modern files and `monoflex_orig` / `monoground_orig` to the
original-environment files.

### Columns (23 prediction columns in every dump, 10 more in two of them; one row per candidate box, Car class)

| Column | Meaning |
|---|---|
| `sid` | KITTI sample id (the image index in `ImageSets/val.txt`) |
| `pred_idx` | index of the candidate within its image |
| `cls` | the detector's Car classification score |
| `sigma`, `log_sigma_raw` | the detector's depth-uncertainty term as used by its native score, and the raw head output it comes from (constant 1.0 / 0.0 for detectors without an uncertainty head; the exact definition per detector is in `adapters/README.md`) |
| `V` | the detector's native ranking score (the score its own evaluation sorts by) |
| `z_pred` | predicted depth of the 3D centre (m) |
| `bbox_h_pix`, `bbox_w_pix`, `bbox_area_pix` | 2D box height, width and area (pixels) |
| `x_3d`, `y_3d`, `z_3d` | 3D box location in camera coordinates (KITTI convention, m) |
| `h_3d`, `w_3d`, `l_3d` | 3D box dimensions (m) |
| `ry`, `alpha` | rotation around the camera y axis and observation angle (rad) |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | 2D box corners (pixels) |
| `dup_rank` | number of Car candidates in the same image with 2D IoU >= 0.5 and a higher `V` |

`dgp_val.csv` (MonoDGP) and `official_monocop_val.csv` (MonoCoP) have 33 columns: the 23 above,
then 10 columns that their adapter (`adapters/dgp_cop_dump.py`) wrote from a matching against the
KITTI Car labels. Nine of them are derived from the ground truth; `dist_bin_4` only bins the
predicted depth.

| Column | Meaning |
|---|---|
| `max_iou_3d`, `max_iou_bev` | largest 3D and BEV IoU of the box with any Car ground truth of its image (official KITTI IoU kernels; 0 in images without a Car) |
| `matched_gt_idx` | index of the ground-truth object (in the image's KITTI label file) assigned to the box by greedy matching in descending `V` at 3D IoU >= 0.7, one box per object; -1 if none |
| `tp_label_iou07` | 1 if the box was assigned a ground truth in that matching, else 0 |
| `gt_x`, `gt_y`, `gt_z` | 3D location of the assigned ground truth (camera coordinates, m); empty if none |
| `depth_err` | abs(`z_pred` - `gt_z`) (m); empty if none |
| `center_err_bev` | distance between (`x_3d`, `z_3d`) and (`gt_x`, `gt_z`) (m); empty if none |
| `dist_bin_4` | `0-15`, `15-30`, `30-45` or `45+` m, binned on the predicted `z_3d` |

No script that writes a frozen report reads these 10 columns: every analysis recomputes its
matching from the KITTI labels. Only the stand-alone CLI of `tools/decomp/dgp_cop_oracle_matrix.py`,
which writes no cited report, sorts by `max_iou_3d`.

## Complete candidate pools (6 files)

The per-prediction dumps hold what each detector emits after its own candidate budget. The
generation-versus-selection analyses (pool waterfall, matching ceiling, separation decomposition,
cross-detector GT states, matched budgets) need the complete native pools before that budget for
the query-based detectors and M3D-RPN. The CenterNet-style dumps already contain their complete
top-K pools.

| Asset | File | Rows | MD5 |
|---|---|---:|---|
| query complete pools | `monodetr_val_preflatten.csv` | 565,350 | `97792c45db4af69effe8c2a3e201c21c` |
| query complete pools | `monodgp_val_preflatten.csv` | 188,450 | `c228a44fbcc722157d9d1aefcb690b3d` |
| query complete pools | `official_monocop_val_preflatten.csv` | 188,450 | `9564d49b68d958dd34c60e81ecef00b9` |
| query complete pools | `monoclue_val_preflatten.csv` | 565,350 | `1eb46213f3ff53550e39b9c8af59d5de` |
| query complete pools | `monoia_val_preflatten.csv` | 565,350 | `ec629f9ec26163786cd47556e11c13fb` |
| M3D-RPN complete pool | `m3drpn_val_floor0.csv` | 8,012,124 | `c2ca44606c8b582db9d38dc6b7c40990` |

- **Query-based detectors** (`*_val_preflatten.csv`, 31 or 33 columns). Every one of the 50
  inference queries of every image with its decoded box and all class scores, plus the native
  stage flags (`flat_rank` / `flat_rank_car` among the 150 query x class hypotheses by raw class
  score, `native_top50`, `thr_pass`, `in_final`). MonoDETR, MonoCLUE and MonoIA have one row per
  query x class hypothesis (150 per image); MonoDGP and MonoCoP have one row per query (50 per
  image) with the Car hypothesis in `cls_car` / `V_car`. The adapters that wrote them
  (`adapters/monodetr_preflatten_dump.py`, `adapters/detr_preflatten_dump.py`,
  `adapters/detr_preflatten_native.py`) define every column.
- **M3D-RPN** (`m3drpn_val_floor0.csv`). The same 23 columns as the per-prediction dumps, for the
  whole pre-NMS pool (top 3,000 per image, no score floor), written by
  `adapters/m3drpn_dump_floor0.py`. Rebuilding M3D-RPN's native NMS on it reproduces its evaluated
  output.

## Matched tables (6 files, `mono3d_anatomy_matched_tables_v1.zip`)

| File | Detector | Rows | MD5 |
|---|---|---:|---|
| `matched_monodetr_val.csv` | MonoDETR | 26,136 | `d573c708005f3bdeb71da9badcf966a4` |
| `matched_dgp_val.csv` | MonoDGP | 38,362 | `52001dc31640ef8ccaf3ada68ba67011` |
| `matched_cop_val.csv` | MonoCoP | 36,572 | `529839aa01fa0ff178277264e75d9f66` |
| `matched_monoia_val.csv` | MonoIA | 27,899 | `05c78d683b0b889c809d1acb458bf713` |
| `matched_monoflex_val.csv` | MonoFlex (modern rebuild) | 17,521 | `15600437fa94cbf143d929b86ee301f3` |
| `matched_gupnet_val.csv` | GUPNet | 16,541 | `5b037376164bdc91890be927138c82c0` |

Each row is one prediction of the released per-prediction dump (every row's `sid`, `z_pred` and
`cls` occur in `monodetr_val.csv`, `dgp_val.csv`, `official_monocop_val.csv`, `monoia_val.csv`,
`monoflex_modern_val.csv` and `gupnet_val.csv` respectively), matched to its best Car ground truth
by 3D IoU (IoU >= 0.05). `tools/decomp/c3_conditioned.py`, their only reader, uses `sid`,
`z_pred`, `gt_z` (ground-truth depth), `iou3d`, `occ`, `trunc` (KITTI occlusion level and
truncation of that ground truth) and `dist_bin` (0-15, 15-30, 30-45, 45+ m by `gt_z`); the other
columns are carried along unused. MonoCLUE is not included: its matched table predates the
validated MonoCLUE dump, and the analysis excludes it. The script that built the tables is not
part of this release; they are released as the frozen inputs of that analysis.

## License

The files are model outputs on KITTI images. KITTI is distributed under CC BY-NC-SA 3.0, so all
four assets are released under **CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/).
The complete pools and twelve of the fourteen per-prediction dumps contain predictions only.
Two files also carry values derived from the KITTI labels: `dgp_val.csv` and
`official_monocop_val.csv` carry, per box, its best 3D and BEV IoU with a Car ground truth and,
for a box matched at IoU >= 0.7, the 3D location of that ground truth (the columns listed under
"Columns"). The matched tables carry, for each matched prediction, the depth, occlusion level and
truncation of its ground-truth box and the 3D IoU to it. The KITTI label files themselves are not
redistributed; get them from the official KITTI website. Please cite the paper and the original
detector papers when you use the files.

# tools/orig — starred-detector recomputations (MonoFlex\*, MonoGround\*)

The panel of the paper uses dumps from the original torch-1.4 environment for MonoFlex\* and
MonoGround\* (released dump stems `monoflex_orig`, `monoground_orig`; see `adapters/`). The
scripts in this folder re-run the fixed-pool diagnostics of `tools/decomp/` on those two dumps.
Each one is a path-substitution port of the script that wrote the frozen report: the kernels are
the ones in `tools/decomp/`, which they import by module name (`_orig_common.py` puts
`tools/decomp` on `sys.path`). Re-runs write to `reports_rerun/` (`paths.OUT_DIR`) under the
report's file name (without the `reports_orig/` prefix); work directories and caches go to
`cache/orig/` (`paths.CACHE_DIR`).

Run everything from the repository root, e.g. `python tools/orig/gap_exact_orig.py`.

| script | report (frozen copy) | paper item |
|---|---|---|
| `a5_regate_orig.py` | `reports_orig/a5_regate_orig.txt` | native ordering-gap cells of the starred rows (cls>=0.1 native pool) |
| `a5_emh_orig.py` | `reports_orig/emh_orig.txt` (its stdout, captured in the release re-run; the original wrote no file) | Table 1 Easy/Moderate/Hard, starred rows |
| `gap_exact_orig.py` | `reports_orig/gap_exact_orig.txt` | de-quantized gap (all-point/R40/R11), dump hashes |
| `diffsweep_orig.py` | `reports_orig/difficulty_sweep_orig/headroom_sep_by_difficulty.{txt,json}` | separation decomposition (Mod rows), difficulty sweep |
| `exp1_ceiling_orig.py` | `reports_orig/exp1_true_ceiling_orig.txt` | matching ceiling AP\* = M/n_gt |
| `c2_iou05_orig.py` | `reports_orig/c2_iou05_orig.txt` | separation decomposition at IoU 0.5 |
| `anatomy_orig.py` | `reports_orig/oracle_anatomy_orig.txt` | oracle anatomy (pure-z / ray / centre) |
| `diffsweep_anatomy_orig.py` | `reports_orig/difficulty_sweep_orig/anatomy_by_difficulty.{txt,json}` | anatomy by difficulty |
| `bridge_orig.py` | `reports_orig/depth_share_bridge_orig.txt` | accuracy-separation AUROC |
| `bev_transfer_orig.py` | `reports_orig/bev_transfer_orig.txt` | BEV recovery (supplementary BEV table) |
| `gap_metric_robustness_orig.py` | `reports_orig/gap_metric_robustness_orig.txt` | metric robustness (R11 vs R40) |
| `operating_point_sweep_orig.py` | `reports_orig/operating_point_sweep_orig.txt` | operating-point sweep |
| `waterfall_orig.py` | `reports_orig/pool_waterfall_orig.txt` | pool waterfall |
| `e12_orig.py` | `reports_orig/e12_replacement_orig.txt` | replacement share |
| `gt_state_matrix_vB.py` | `reports_orig/gt_state_matrix_vB.txt` | cross-detector GT states (12 detectors) |
| `probe_official_orig.py` | `reports_orig/probe_official_moderate_orig.txt` | progression probe, official moderate set |
| `probe_orig.py` | `reports_orig/probe_orig.txt` | progression probe (Fig. 1 inputs) |
| `transfer_orig.py` | `reports_orig/clean_transfer_orig.txt` | drive-disjoint corrector transfer |
| `transfer_save_orig.py` | prediction dirs in `cache/orig/` | input of the transfer bootstrap |
| `make_ladder_preds_orig.py` | prediction dirs in `cache/orig/` | input of the ladder bootstrap |
| `bootstrap_orig.py` | `reports_orig/bootstrap_floor_orig.txt` | drive-cluster bootstrap floor (23 comparisons) |
| `spearman_gap_base_vB.py` | `reports_orig/spearman_gap_base_vB.txt` | gap-vs-base Spearman (3 bases) |
| `make_vB_reports.py` | `reports_rerun/vB_reports/` (= `figures/data/vB_reports/`) | merged 12-detector view read by the figures |
| no script in this release | `reports_orig/oracle_ladder_o1_orig.txt` | BEV re-sort oracle (O1), starred rows |
| no script in this release | `reports_orig/oracle_ladder_o2_orig.txt` | single-factor geometry oracle (O2), starred rows |

The panel-level numbers that combine these starred rows with the other ten detectors of
`reports/` (O1, O2, the BEV transfer and the far-field census) are recomputed from the printed
values by `tools/extensions/starred_panel_summaries.py`
(`reports/extensions/starred_panel_summaries.txt`).

`make_vB_reports.py` rebuilds the six files of `figures/data/vB_reports/` byte for byte from the
reports shipped in `reports/` and `reports_orig/` (it also writes a `final_run/native_gap_canonical.md`
view that the figures do not read; Fig. 3a takes those cells as literal values).

## Order and shared caches

The scripts share caches exactly as the originals did, so a few must run first:

1. `gap_exact_orig.py` writes `cache/orig/<stem>_bridgecache.npz` (S5-pool oracle IoUs), reused by
   `bridge_orig.py` and `gap_metric_robustness_orig.py` (each rebuilds it if missing).
2. `diffsweep_orig.py` writes `cache/orig/_e4cache_<name>.npz` (native-pool oracle IoUs and TP
   labels). `exp1_ceiling_orig.py` reads the TP labels for its FP-demotion column (nan without the
   cache); `c2_iou05_orig.py` reuses the oracle IoUs.
3. `waterfall_orig.py` before `e12_orig.py` and `gt_state_matrix_vB.py` (they gate on its face-ii
   shares; the frozen `reports_orig/pool_waterfall_orig.txt` is used instead when it is present).
4. Bootstrap: `transfer_save_orig.py`, `make_ladder_preds_orig.py --with-main`, then
   `bootstrap_orig.py --ladder monodle gupnet`, `--ladder deviant monoflex_orig`,
   `--ladder monoflex_orig monoground_orig`, `--ladder monoground_orig monocon`,
   `--transfer monoflex_orig {0,1,2}`, and `--collate`. The collate also reads the sixteen
   main-panel comparison JSONs written by `tools/decomp/bootstrap_floor.py`
   (`cache/decomp/_bootstrap_out/`, or `$MONO3D_MAIN_BOOTSTRAP_OUT`). Each comparison takes
   about 30 minutes (B = 1000 drive resamples; `NBOOT` overrides it).
5. `make_vB_reports.py` last; it reads `reports/probe_detector_progression.txt` for the main-panel
   rows.

The drive-grouped scripts (`transfer_*`, `bootstrap_orig.py`) read the KITTI frame-to-drive
table from `tools/decomp/frame_sequence.py`, which builds it from the KITTI object devkit
mapping files (`<KITTI_ROOT>/mapping/train_mapping.txt`, `train_rand.txt`).

## Inputs

All scripts except `gt_state_matrix_vB.py` need only the per-prediction dumps
(`mono3d_anatomy_dumps_v1.zip`): the two starred dumps, plus the four main-panel neighbours
MonoDLE, GUPNet, DEVIANT and MonoCon for `make_ladder_preds_orig.py --with-main`.
`gt_state_matrix_vB.py` evaluates all twelve detectors on their
complete native pools, so it also needs the two complete-pool assets of release v1.0
(`<detector>_val_preflatten.csv` for the five query detectors, `m3drpn_val_floor0.csv` for
M3D-RPN; see `data/DUMPS.md`). `_orig_common.extra_dump` reads them from `paths.DUMP_DIR`, or from
`$MONO3D_EXTRA_DUMP_DIR` if set. The branches of `native_pool()` / `stages_for()` for the other
detectors are kept verbatim in `exp1_ceiling_orig.py`, `c2_iou05_orig.py`, `diffsweep_orig.py`,
`waterfall_orig.py` and `e12_orig.py`, but these scripts evaluate only the two starred
detectors, so those branches are never executed.

## Release check

In the release check (numpy 2.0.2, shapely 2.0.7, see the main README), every script above that
writes a frozen report reproduces it line by line, apart from the run-timestamp line and, for a
scrubbed public copy, its `# public copy ...` first line, with these exceptions:

- one integer in `oracle_anatomy_orig.txt`: the MonoFlex\* per-bin pure-z recall count in the
  0-15 m bin is 2336 instead of 2337 (the frozen report was made with numpy 1.24.4). All AP values
  and summary lines are identical. The 3D IoU used by the matchers (`ap_corrector_arc.iou3d`,
  shapely/GEOS polygons) can differ in rare degenerate cases across numpy/shapely/GEOS versions;
- `gt_state_matrix_vB.py` reproduces every number of its report; only the `[written]` path line differs;
- `bootstrap_orig.py` was spot-checked on 4 of its 23 comparisons, all identical;
- `reports_orig/emh_orig.txt` is itself the release re-run of `a5_emh_orig.py` (its two `[A5]`
  lines verbatim; the header lines and the READ note were added);
- the two `oracle_ladder_*_orig.txt` reports have no script in this release.

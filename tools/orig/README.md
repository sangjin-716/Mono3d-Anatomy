# tools/orig

These scripts run the `tools/decomp/` analyses on the MonoFlex\* and MonoGround\* dumps from the
authors' original torch-1.4 environment (stems `monoflex_orig` and `monoground_orig`, see
`data/DUMPS.md`). They import the analysis code from `tools/decomp/` (`_orig_common.py` adds it to
`sys.path`), so only the inputs and output paths change.

Run them from the repository root:

```bash
python tools/orig/gap_exact_orig.py
```

Reports go to `reports_rerun/` (`paths.OUT_DIR`) under the same file name they have in
`reports_orig/`. Caches and work directories go to `cache/orig/` (`paths.CACHE_DIR`).

| script | report | paper item |
|---|---|---|
| `a5_regate_orig.py` | `reports_orig/a5_regate_orig.txt` | native ordering-gap cells of the starred rows (cls>=0.1 native pool) |
| `a5_emh_orig.py` | `reports_orig/emh_orig.txt` (saved stdout; the script writes no file) | Table 1 Easy/Moderate/Hard, starred rows |
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
| no script | `reports_orig/oracle_ladder_o1_orig.txt` | BEV re-sort oracle (O1), starred rows |
| no script | `reports_orig/oracle_ladder_o2_orig.txt` | single-factor geometry oracle (O2), starred rows |

The panel-level O1, O2, BEV transfer and far-field numbers combine these starred rows with the
other ten detectors in `reports/`. `tools/extensions/starred_panel_summaries.py` computes them from
the printed values (`reports/extensions/starred_panel_summaries.txt`).

`make_vB_reports.py` rebuilds the six files in `figures/data/vB_reports/` from `reports/` and
`reports_orig/`. It also writes `reports_rerun/vB_reports/final_run/native_gap_canonical.md`. The
figure scripts do not read that file: the Fig. 3a native values are constants in
`figures/repro_make_figs_vB.py`.

## Run order

Some scripts share caches, so run these first:

1. `gap_exact_orig.py` writes `cache/orig/<stem>_bridgecache.npz`, the oracle 3D IoUs of the
   boxes left after a score cut of 0.2 and 2D NMS at IoU 0.5. `bridge_orig.py` and
   `gap_metric_robustness_orig.py` reuse it and rebuild it if it is missing.
2. `diffsweep_orig.py` writes `cache/orig/_e4cache_<name>.npz` (native-pool oracle IoUs and TP
   labels). `exp1_ceiling_orig.py` needs the TP labels for its FP-demotion column, which is nan
   without the cache. `c2_iou05_orig.py` reuses the oracle IoUs.
3. `e12_orig.py` and `gt_state_matrix_vB.py` check their counts against the `suppressed-acc`
   shares in `reports_orig/pool_waterfall_orig.txt`: among GTs that have an IoU 0.7 candidate in the pool, the fraction missed in the final output. If that file is missing they read the re-run output, so run `waterfall_orig.py` first.
4. The bootstrap runs in this order:

   ```bash
   python tools/orig/transfer_save_orig.py
   python tools/orig/make_ladder_preds_orig.py --with-main
   python tools/orig/bootstrap_orig.py --ladder monodle gupnet
   python tools/orig/bootstrap_orig.py --ladder deviant monoflex_orig
   python tools/orig/bootstrap_orig.py --ladder monoflex_orig monoground_orig
   python tools/orig/bootstrap_orig.py --ladder monoground_orig monocon
   python tools/orig/bootstrap_orig.py --transfer monoflex_orig 0
   python tools/orig/bootstrap_orig.py --transfer monoflex_orig 1
   python tools/orig/bootstrap_orig.py --transfer monoflex_orig 2
   python tools/orig/bootstrap_orig.py --collate
   ```

   Each comparison takes about 30 minutes with B = 1000 drive resamples. Set `NBOOT` to change B.

   `--collate` also reads the sixteen main-panel comparison JSONs written by
   `tools/decomp/bootstrap_floor.py`, from `cache/decomp/_bootstrap_out/` or
   `$MONO3D_MAIN_BOOTSTRAP_OUT`. The nine main-panel transfer comparisons (the MonoDGP, GUPNet and
   MonoCoP cells of Table 3) are recomputed from the released dumps with

   ```bash
   python tools/decomp/clean_transfer_save.py
   python tools/decomp/bootstrap_floor.py --transfer dgp 0               # and 1, 2
   python tools/decomp/bootstrap_floor.py --transfer official_monocop 0  # and 1, 2
   python tools/decomp/bootstrap_floor.py --transfer gupnet 0            # and 1, 2
   ```

   before `bootstrap_orig.py --collate`. `clean_transfer_save.py` also writes the folders of the
   modern-environment MonoFlex dump (`monoflex_s*`), which the collate leaves out. The seven
   main-panel ladder comparisons need prediction folders written by a script that is not in this
   release, so their JSONs can only be collated, not recomputed. The transfer rows, the
   confirmatory-set line and the panel floor do not depend on them.
5. `make_vB_reports.py` goes last. It reads `reports/probe_detector_progression.txt` for the
   main-panel rows.

The drive-grouped scripts (`transfer_*`, `bootstrap_orig.py`) get the frame-to-drive table from
`tools/decomp/frame_sequence.py`, which builds it from the KITTI devkit mapping files
`<KITTI_ROOT>/mapping/train_mapping.txt` and `train_rand.txt`.

## Inputs

All scripts except `gt_state_matrix_vB.py` only need `mono3d_anatomy_dumps_v1.zip`: the two
starred dumps, plus MonoDLE, GUPNet, DEVIANT and MonoCon for
`make_ladder_preds_orig.py --with-main`.

`gt_state_matrix_vB.py` evaluates all twelve detectors on their complete native pools, so it also
needs the two complete-pool assets of release v1.0: `mono3d_anatomy_query_complete_pools_v1.zip`
(`<detector>_val_preflatten.csv` for the five query detectors) and
`mono3d_anatomy_m3drpn_complete_pool_v1.zip` (`m3drpn_val_floor0.csv`), see `data/DUMPS.md`. It
reads them from `paths.DUMP_DIR`, or from `$MONO3D_EXTRA_DUMP_DIR` if set.

## Differences from reports_orig/

With numpy 2.0.2 and shapely 2.0.7 the outputs match `reports_orig/` line for line, apart from
the timestamp line, the `# public copy ...` first line of some reports, and these cases:

- `oracle_anatomy_orig.txt`: the MonoFlex\* pure-z recall count in the 0-15 m bin is 2336 instead
  of 2337. All AP values and summary lines are the same. The shipped report was made with numpy
  1.24.4, and the 3D IoU used for matching (`ap_corrector_arc.iou3d`, shapely/GEOS polygons) can
  differ in rare degenerate cases across numpy, shapely and GEOS versions.
- `gt_state_matrix_vB.txt`: only the `[written]` path line differs.
- `bootstrap_floor_orig.txt`: all twelve transfer comparisons (the Table 3 cells) have been re-run
  and match line for line, including the confirmatory-set line and the panel floor (4.021). Their
  JSONs match the original ones in every field but the timestamp, and the prediction folders
  written by `clean_transfer_save.py` are byte-identical to the original ones. Of the ladder
  comparisons, `deviant->monoflex_orig` and `monoflex_orig->monoground_orig` have been re-run and
  match.
- `emh_orig.txt`: only the two `[A5]` lines come from the script. The header lines and the READ
  note were added by hand.

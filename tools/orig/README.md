# tools/orig

The `tools/decomp/` analyses run on the MonoFlex\* and MonoGround\* dumps from the authors' original
torch-1.4 environment (`monoflex_orig`, `monoground_orig`). Run each script from the repository
root, e.g. `python tools/orig/gap_exact_orig.py`. Reports go to `reports_rerun/` under their
`reports_orig/` name, caches to `cache/orig/`. Only `gt_state_matrix_vB.py` needs more than
`mono3d_anatomy_dumps_v1.zip`: the two complete-pool files in [`data/DUMPS.md`](../../data/DUMPS.md).

| script | report (in `reports_orig/` unless noted) | content |
|---|---|---|
| `a5_regate_orig.py` | `a5_regate_orig.txt` | native ordering gap |
| `a5_emh_orig.py` | `emh_orig.txt` (saved stdout, header added by hand) | Table 1 Easy/Moderate/Hard |
| `gap_exact_orig.py` | `gap_exact_orig.txt` | all-point/R40/R11 gap |
| `diffsweep_orig.py` | `difficulty_sweep_orig/headroom_sep_by_difficulty.{txt,json}` | separation by difficulty |
| `exp1_ceiling_orig.py` | `exp1_true_ceiling_orig.txt` | matching ceiling AP\* |
| `c2_iou05_orig.py` | `c2_iou05_orig.txt` | separation at IoU 0.5 |
| `anatomy_orig.py` | `oracle_anatomy_orig.txt` | oracle anatomy |
| `diffsweep_anatomy_orig.py` | `difficulty_sweep_orig/anatomy_by_difficulty.{txt,json}` | anatomy by difficulty |
| `bridge_orig.py` | `depth_share_bridge_orig.txt` | accuracy-separation AUROC |
| `bev_transfer_orig.py` | `bev_transfer_orig.txt` | BEV transfer |
| `gap_metric_robustness_orig.py` | `gap_metric_robustness_orig.txt` | R11 vs R40 |
| `operating_point_sweep_orig.py` | `operating_point_sweep_orig.txt` | operating-point sweep |
| `waterfall_orig.py` | `pool_waterfall_orig.txt` | pool waterfall |
| `e12_orig.py` | `e12_replacement_orig.txt` | replacement share |
| `gt_state_matrix_vB.py` | `gt_state_matrix_vB.txt` | GT states, 12 detectors |
| `probe_official_orig.py` | `probe_official_moderate_orig.txt` | probe, official moderate |
| `probe_orig.py` | `probe_orig.txt` | probe (Fig. 1) |
| `transfer_orig.py` | `clean_transfer_orig.txt` | drive-disjoint transfer |
| `transfer_save_orig.py` | prediction dirs in `cache/orig/` | transfer bootstrap input |
| `make_ladder_preds_orig.py` | prediction dirs in `cache/orig/` | ladder bootstrap input |
| `bootstrap_orig.py` | `bootstrap_floor_orig.txt` | drive-cluster bootstrap |
| `spearman_gap_base_vB.py` | `spearman_gap_base_vB.txt` | gap-vs-base Spearman |
| `make_vB_reports.py` | `reports_rerun/vB_reports/` (= `figures/data/vB_reports/`) | 12-detector view for figures |
| none | `oracle_ladder_o1_orig.txt` | BEV re-sort oracle O1 |
| none | `oracle_ladder_o2_orig.txt` | geometry oracle O2 |

## Run order

1. `gap_exact_orig.py`, then `bridge_orig.py` and `gap_metric_robustness_orig.py` (shared cache).
2. `diffsweep_orig.py`, then `exp1_ceiling_orig.py` and `c2_iou05_orig.py` (shared cache).
3. `waterfall_orig.py`, then `e12_orig.py` and `gt_state_matrix_vB.py`.
4. The bootstrap, about 30 min per comparison (B = 1000, set with `NBOOT`):

   ```bash
   python tools/orig/transfer_save_orig.py
   python tools/orig/make_ladder_preds_orig.py --with-main
   python tools/orig/bootstrap_orig.py --ladder monodle gupnet     # and the 3 pairs below
   python tools/orig/bootstrap_orig.py --transfer monoflex_orig 0  # and 1, 2
   python tools/decomp/clean_transfer_save.py
   python tools/decomp/bootstrap_floor.py --transfer dgp 0  # and 1, 2; same for official_monocop, gupnet
   python tools/orig/bootstrap_orig.py --collate
   ```

   Other ladder pairs: `deviant monoflex_orig`, `monoflex_orig monoground_orig`,
   `monoground_orig monocon`. `--collate` reads the main-panel JSONs from
   `cache/decomp/_bootstrap_out/` or `$MONO3D_MAIN_BOOTSTRAP_OUT`. Table 3 (all twelve transfer
   cells) is fully reproducible this way. The seven main-panel ladder comparisons are not: their
   prediction folders come from a script not in this release, so they can only be collated.
5. `make_vB_reports.py` last.

## Known difference

With numpy 2.0.2 and shapely 2.0.7 the outputs match `reports_orig/` apart from timestamp,
`# public copy` and `[written]` lines. Of the four starred ladder comparisons in the bootstrap report, only deviant to monoflex_orig and monoflex_orig to monoground_orig were re-run. In `oracle_anatomy_orig.txt` the MonoFlex\* pure-z recall
count in the 0-15 m bin is 2336 instead of 2337 (shipped report: numpy 1.24.4). All AP values match.

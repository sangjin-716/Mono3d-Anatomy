# Reproducing the reports

Set up the environment and data first ([README](../README.md), [`data/DUMPS.md`](../data/DUMPS.md)).

## Environment

- Tested: one NVIDIA Quadro RTX 6000 (driver 535.183.01), Python 3.9 or 3.10.
- CUDA: conda `cuda-nvcc` 12.1.66, or a system toolkit via `CUDA_HOME`. Check: `python -c "from numba.cuda.cudadrv import libs; libs.test()"`.
- Pins: `requirements.txt`. Keep numpy at 2.0.x. numpy 1.26 orders M3D-RPN's tied scores differently.
- Figures: matplotlib 3.9.4 or 3.10.8, PyMuPDF, DejaVu Sans Condensed (`fonts-dejavu-extra`), tectonic.
- `adapters/` and `tools/crossbench/` have their own environments. See their READMEs.

## Evaluator check

`python evaluator/exact_ap.py` (`--full` for all twelve) ends as below. Ignore the Numba warnings.

```
toyA perfect-rank: allpoint=100.00 (hand 100.00)  R40 mine/off=2.50/2.50  R11 mine/off=9.09/9.09  [PASS]
toyB tied-scores : allpoint=66.67 (hand 66.67)  R40 mine/off=1.67/1.67  R11 mine/off=6.06/6.06  [PASS]
dgp              R40 off/mine= 22.290/ 22.290  R11 off/mine= 26.215/ 26.215  p101=22.763 p401=22.834 allpoint=22.905 [PASS]
gupnet           R40 off/mine= 16.481/ 16.481  R11 off/mine= 21.865/ 21.865  p101=16.933 p401=17.030 allpoint=17.122 [PASS]
ALL GATES: PASS
```

## Running the scripts

Run from the repository root. Re-runs go to `reports_rerun/` under the same relative path
(`reports/final_run/X` to `reports_rerun/final_run/X`, `reports_orig/Y` to `reports_rerun/Y`) and
caches to `cache/`. `reports/` and `reports_orig/` are never written. Run these in order:

```bash
python tools/decomp/gap_exact.py              # writes cache/decomp/_bridgecache_<det>.npz
python tools/decomp/gap_metric_robustness.py  # reuses it
python tools/decomp/e3_endpoint_gap_boot.py   # reuses it
python tools/decomp/e4_fp_tp_decomp.py        # writes cache/decomp/_e4cache_<det>.npz
python tools/decomp/exp1_true_ceiling.py      # FP-demotion column is nan without it
python tools/decomp/c2_iou05_separation.py    # reuses it
python tools/decomp/diffsweep_headroom_sep.py # needs it
python tools/decomp/gt_state_matrix.py        # writes cache/decomp/gt_state_matrix.csv
python tools/decomp/e2_budget_truncation.py   # needs it
```

Other scripts are independent. Table 3 is the `transfer` rows of `reports_orig/bootstrap_floor_orig.txt`;
its run order and the `tools/orig/` order are in [`tools/orig/README.md`](../tools/orig/README.md).

Runtimes: seconds to 1 h, except `operating_point_sweep.py` 3.5 h, `detr_native_sweep.py` 2 h, `diffsweep_anatomy.py` 1.8 h, bootstrap 30 min per comparison (one core).

## Figures

Input: `figures/data/vB_reports/` (rebuilt into `reports_rerun/vB_reports/` by `tools/orig/make_vB_reports.py`). Output: `reports_rerun/figures/`.

```bash
python figures/make_figs_cr.py        # fig1_progression (Fig. 1), fig2_headroom (Fig. 3), fig3_anatomy (Fig. 4), fig4_waterfall and fig4v2_plane_agreement (supp. Figs. 1, 2)
tectonic figures/fig_unified_src.tex  # Fig. 2
python figures/verify_cr.py           # page size and fonts
python figures/cmp_text.py reports_rerun/figures <figure PDFs from the paper source>
python figures/data_identity.py       # expected diffs: Fig. 1(c) grey series, MonoFlex* waterfall label
```

## Public copies and known differences

Reports headed `# public copy of <source>, sha256 <hash>, scrubbed: ...` keep every number; some in
`reports/extensions/` also drop lines. When diffing, ignore these headers, timestamps, local paths
and package versions. Other differences: one recall count in `reports_orig/oracle_anatomy_orig.txt`
is 2336, not 2337 (APs equal), `reports/final_run/c3_conditioned_analysis.md` has a hand-written
verdict section, and `reports_orig/emh_orig.txt` is hand-annotated stdout of
`tools/orig/a5_emh_orig.py`. No script is released for `reports/depth_share_combo.txt`, the four
`oracle_ladder_o*` reports, the `a5_regate`, `bev_gap_probe`, `a3_farfield_integrity` and
`native_gap_canonical` reports in `reports/final_run/`, the seven ladder comparisons in
`reports_orig/bootstrap_floor_orig.txt` and the supp. B records (e.g. `leave_two_out.txt`, `a4_supremum_lb_SUPPLEMENTARY.md`, `c6_multiplicity_audit.md`, the `prereg_*.md` files). `reports/claim_decision_tree.md`, named in a few reports, is internal and not released. `tools/crossbench/` and
`tools/extensions/difficulty_class_extension.py` were not re-run on the released files.

## Where each result comes from

| Paper item | Report | Script |
|---|---|---|
| Table 1, supp. C: official vs reproduced | `reports/detector_adapters_gates.md`, `adapters/README.md` | `adapters/` |
| Table 1: E/M/H for MonoFlex\*, MonoGround\* | `reports_orig/emh_orig.txt` | `tools/orig/a5_emh_orig.py` |
| Supp. D native gap cells, Table 1 Moderate base for MonoFlex\*, MonoGround\* | `reports_orig/a5_regate_orig.txt`, `reports/final_run/a5_regate.txt` | `tools/orig/a5_regate_orig.py` |
| Table 1, Sec. 5.1: matching ceiling AP\* and ordering gap | `reports/exp1_true_ceiling.txt`, `reports_orig/exp1_true_ceiling_orig.txt` | `tools/decomp/exp1_true_ceiling.py`, `tools/orig/exp1_ceiling_orig.py` |
| Fig. 3b, Sec. 5.1, supp. D: all-point vs R40 vs R11 | `reports/gap_exact.txt`, `reports/gap_metric_robustness.txt`, `reports_orig/gap_exact_orig.txt`, `reports_orig/gap_metric_robustness_orig.txt` | `tools/decomp/gap_exact.py`, `tools/decomp/gap_metric_robustness.py`, `tools/orig/gap_exact_orig.py`, `tools/orig/gap_metric_robustness_orig.py` |
| Fig. 3a, supp. D: re-sort at native operating points | `reports/final_run/native_gap_canonical.md`, `reports/operating_point_sweep.txt`, `reports/detr_native_sweep.txt`, `reports_orig/operating_point_sweep_orig.txt` | `tools/decomp/operating_point_sweep.py`, `tools/decomp/detr_native_sweep.py`, `tools/orig/operating_point_sweep_orig.py` |
| Supp. D: BEV transfer of the re-sort | `reports/final_run/bev_gap_probe.txt`, `reports_orig/bev_transfer_orig.txt`, `reports/extensions/starred_panel_summaries.txt` (Part D) | `tools/orig/bev_transfer_orig.py`, `tools/extensions/starred_panel_summaries.py` |
| Supp. D: oracle O1. Sec. 5.2: oracle O2 | `reports/oracle_ladder_o1.txt`, `reports/oracle_ladder_o2.txt`, `reports_orig/oracle_ladder_o1_orig.txt`, `reports_orig/oracle_ladder_o2_orig.txt`, `reports/extensions/starred_panel_summaries.txt` (Parts B, C) | `tools/extensions/starred_panel_summaries.py` |
| Sec. 5, supp. E: FP demotion, TP re-ordering | `reports/e4_fp_tp_decomp.txt`, `reports/e4_prereg.md` (pre-registration), `reports/final_run/c2_iou05_separation.txt`, `reports_orig/c2_iou05_orig.txt` | `tools/decomp/e4_fp_tp_decomp.py`, `tools/decomp/c2_iou05_separation.py`, `tools/orig/diffsweep_orig.py`, `tools/orig/c2_iou05_orig.py` |
| Fig. 1, Sec. 4: recall, depth error, score quality | `reports/final_run/probe_official_moderate.txt`, `reports/probe_detector_progression.txt`, `reports_orig/probe_official_moderate_orig.txt`, `reports_orig/probe_orig.txt` | `tools/decomp/probe_official_moderate.py`, `tools/decomp/probe_detector_progression.py`, `tools/orig/probe_official_orig.py`, `tools/orig/probe_orig.py` |
| Sec. 4, supp. H: paired common-object control | `reports/paired_perbin_dz.txt`, `reports/paired_common_object.txt` | `tools/decomp/paired_perbin_dz.py`, `tools/decomp/paired_common_object.py` |
| Fig. 4, Sec. 5.2: oracle anatomy | `reports/oracle_anatomy.txt`, `reports_orig/oracle_anatomy_orig.txt` | `tools/decomp/oracle_anatomy.py`, `tools/orig/anatomy_orig.py` |
| Sec. 5.2, supp. H: accuracy-separation AUROC | `reports/depth_share_bridge.txt`, `reports_orig/depth_share_bridge_orig.txt` | `tools/decomp/depth_share_bridge.py`, `tools/orig/bridge_orig.py` |
| Sec. 5.2: score x quality, mean rank | `reports/depth_share_combo.txt` | none |
| Supp. H: far-field recall beyond 45 m | `reports/extensions/starred_panel_summaries.txt` (Part A), `reports/final_run/a3_farfield_integrity.txt` | `tools/extensions/starred_panel_summaries.py` |
| Table 3: drive-disjoint transfer | `reports_orig/bootstrap_floor_orig.txt`, `reports/clean_transfer_strong.txt`, `reports_orig/clean_transfer_orig.txt`, `reports/drive_grouped_oof.md` | `tools/decomp/clean_transfer_save.py`, `tools/decomp/bootstrap_floor.py`, `tools/orig/transfer_save_orig.py`, `tools/orig/bootstrap_orig.py`, `tools/decomp/clean_transfer_strong.py`, `tools/orig/transfer_orig.py`, `tools/decomp/phase1_drive_oof.py` |
| Sec. 7, supp. G: bootstrap floors, endpoint bootstrap | `reports_orig/bootstrap_floor_orig.txt`, `reports/final_run/e3_endpoint_gap_boot.txt` | `tools/decomp/bootstrap_floor.py`, `tools/orig/bootstrap_orig.py`, `tools/decomp/e3_endpoint_gap_boot.py` |
| Supp. F: gap-vs-base Spearman | `reports_orig/spearman_gap_base_vB.txt` | `tools/orig/spearman_gap_base_vB.py` |
| Supp. H: cross-detector disagreement | `reports/final_run/c3_conditioned_analysis.md` | `tools/decomp/c3_conditioned.py` |
| Supp. I, J: GT states, waterfall, replacement, budget | `reports/pool_waterfall.txt`, `reports/e12_replacement.txt`, `reports/gt_state_matrix.txt`, `reports/final_run/e2_budget_truncation.txt`, `reports_orig/pool_waterfall_orig.txt`, `reports_orig/e12_replacement_orig.txt`, `reports_orig/gt_state_matrix_vB.txt`, `reports/extensions/gt_state_recount.txt` | `tools/decomp/pool_waterfall.py`, `tools/decomp/e12_replacement.py`, `tools/decomp/gt_state_matrix.py`, `tools/decomp/e2_budget_truncation.py`, `tools/orig/waterfall_orig.py`, `tools/orig/e12_orig.py`, `tools/orig/gt_state_matrix_vB.py`, `tools/extensions/gt_state_recount.py` |
| Supp. K: by KITTI difficulty | `reports/final_run/difficulty_sweep/`, `reports_orig/difficulty_sweep_orig/` | `tools/decomp/diffsweep_headroom_sep.py`, `tools/decomp/diffsweep_anatomy.py`, `tools/orig/diffsweep_orig.py`, `tools/orig/diffsweep_anatomy_orig.py` |
| Supp. L: matched candidate budgets | `reports/extensions/budget_matched.txt`, `reports/extensions/pool_census.txt`, `reports/extensions/budget_saturation.txt` | `tools/extensions/budget_matched.py`, `tools/extensions/budget_matched_finalize.py`, `tools/extensions/pool_census.py`, `tools/extensions/budget_saturation.py` |
| Supp. M: near misses | `reports/extensions/nearmiss_panelwide.txt` | `tools/extensions/nearmiss_panelwide.py` |
| Supp. N: trend-test power | `reports/extensions/power_analysis.txt` | `tools/extensions/power_analysis.py` |
| Supp. O: Pedestrian and Cyclist | `reports/extensions/difficulty_class_extension.txt`, `reports/extensions/class_gate_audit.txt`, `reports/extensions/class_capability.txt` | `tools/extensions/difficulty_class_extension.py`, `tools/extensions/class_capability.py` |
| Supp. P: Waymo and nuScenes | `reports/extensions/crossbench_audit.md`, `reports/extensions/class_capability.txt` (Part E) | `tools/crossbench/`, `tools/extensions/class_capability.py` |
| Figs. 1, 3, 4, supp. I, J waterfalls | `figures/data/vB_reports/` | `figures/make_figs_cr.py`, `tools/orig/make_vB_reports.py` |
| Fig. 2 | `figures/fig_unified_src.tex` | tectonic |

In the probe reports "matched TP" means the paper's matched boxes (best 3D IoU >= 0.05, uniform pool).

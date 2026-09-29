# Reproducing the reports

This page lists which script and report each table and figure comes from, and how to re-run them.
Set up the environment and download the data first (see the [README](../README.md) and
[`data/DUMPS.md`](../data/DUMPS.md)). `python tools/_release.py` tells you which release assets are
complete in `DUMP_DIR`, and `data/DUMPS.md` says which scripts need which asset.

## Environment

Our tested setup is one NVIDIA Quadro RTX 6000 (driver 535.183.01) with Python 3.9 and
cuda-nvcc 12.1.66 from conda. Python 3.10 also works, and so does a system CUDA toolkit with
`CUDA_HOME` set. Dumping detector outputs (`adapters/`) and the Waymo and nuScenes check
(`tools/crossbench/`) use their own environments, described in their READMEs. To see where numba
found NVVM and libdevice:

```bash
python -c "from numba.cuda.cudadrv import libs; libs.test()"
```

Versions are pinned in `requirements.txt`. Keep numpy at 2.0.x: M3D-RPN has tied scores, and
numpy 1.26 orders the ties differently inside its NMS, so a slightly different set of M3D-RPN boxes
survives.

The figure scripts need a few extra things. No number depends on them.

- PyMuPDF (`pip install pymupdf`) for `figures/verify_cr.py` and `figures/cmp_text.py`
- the DejaVu Sans Condensed font (`fonts-dejavu-extra` on Debian/Ubuntu)
- a LaTeX engine such as [tectonic](https://tectonic-typesetting.github.io) for Fig. 2

## Running the scripts

Run every script from the repository root. The header of each script names the report it
reproduces. Re-runs go to `reports_rerun/` under the same relative path, so `reports/final_run/X`
becomes `reports_rerun/final_run/X` and `reports_orig/Y` becomes `reports_rerun/Y`. Nothing is
written to `reports/` or `reports_orig/`. Caches and temporary KITTI-format files go to `cache/`.

A few scripts in `tools/decomp/` share caches, so run them in this order. The others are
independent.

```bash
python tools/decomp/gap_exact.py              # writes cache/decomp/_bridgecache_<det>.npz
python tools/decomp/gap_metric_robustness.py  # reuses it (rebuilds it if missing)
python tools/decomp/e3_endpoint_gap_boot.py   # reuses it (rebuilds it if missing)

python tools/decomp/e4_fp_tp_decomp.py        # writes cache/decomp/_e4cache_<det>.npz
python tools/decomp/exp1_true_ceiling.py      # FP-demotion column is nan without the e4 cache
python tools/decomp/c2_iou05_separation.py    # reuses the oracle IoUs (recomputes them if missing)
python tools/decomp/diffsweep_headroom_sep.py # needs the e4 cache

python tools/decomp/gt_state_matrix.py        # writes cache/decomp/gt_state_matrix.csv
python tools/decomp/e2_budget_truncation.py   # needs it
```

The run order for `tools/orig/` and the drive-cluster bootstrap (`tools/decomp/bootstrap_floor.py`,
`tools/orig/bootstrap_orig.py`) is in [`tools/orig/README.md`](../tools/orig/README.md).

The twelve cells of Table 3 are the `transfer` rows of `reports_orig/bootstrap_floor_orig.txt`.
Nine of them (MonoDGP, GUPNet, MonoCoP) come from `tools/decomp/`, the three MonoFlex\* cells from
`tools/orig/`:

```bash
python tools/decomp/clean_transfer_save.py          # writes cache/decomp/_bootstrap_preds/
python tools/decomp/bootstrap_floor.py --transfer dgp 0    # also 1, 2, and official_monocop, gupnet
python tools/orig/transfer_save_orig.py             # writes cache/orig/_bootstrap_preds_orig/
python tools/orig/bootstrap_orig.py --transfer monoflex_orig 0   # also 1, 2
python tools/orig/bootstrap_orig.py --collate       # all transfer rows and the panel floor
```

Each `--transfer` run takes about 30 minutes and uses one core, so they can run in parallel.

Runtimes on one Quadro RTX 6000: most scripts take a few seconds to about an hour. The long ones
are `operating_point_sweep.py` (about 3.5 hours), `detr_native_sweep.py` (about 2 hours),
`diffsweep_anatomy.py` (about 1.8 hours), and the bootstrap at about 30 minutes per comparison.

## Figures

The figures are drawn from the reports, so nothing has to be re-run. Their input is
`figures/data/vB_reports/`. "vB" is the paper's twelve-detector view: the reports in `reports/`
with the MonoFlex\* and MonoGround\* rows taken from `reports_orig/`. Files with `vB` in the name
belong to this view. `tools/orig/make_vB_reports.py` rebuilds the folder from those two, into
`reports_rerun/vB_reports/`.

```bash
python figures/make_figs_cr.py         # Figs. 1, 3, 4 and supplementary Figs. 1, 2
tectonic figures/fig_unified_src.tex   # Fig. 2 (TikZ)
```

`make_figs_cr.py` writes five PDFs to `reports_rerun/figures/`: `fig1_progression` (Fig. 1), `fig2_headroom` (Fig. 3), `fig3_anatomy` (Fig. 4), `fig4_waterfall` and `fig4v2_plane_agreement` (supplementary Figs. 1 and 2). These are the checks we used on
them. `cmp_text.py` needs the figure PDFs from the paper's LaTeX source, which are not in this
repository.

```bash
python figures/verify_cr.py        # page size and fonts
python figures/cmp_text.py reports_rerun/figures <figure PDFs from the paper source>
python figures/data_identity.py    # data compared with the older figure scripts
```

With matplotlib 3.9.4 and 3.10.8, the text and vector drawings match the paper's figures.
`data_identity.py` compares against `figures/repro_make_figs_vB.py` and
`figures/repro_make_fig4v2_vB.py` and reports two expected differences: the grey series of
Fig. 1(c) is the native base AP of Table 1, and the MonoFlex\* grey label in the first
supplementary waterfall figure is read from `reports_orig/pool_waterfall_orig.txt`.

## Public copies

Some reports start with a `# public copy of <source>, sha256 <hash>, scrubbed: <what changed>`
line. Every number they keep is unchanged. Some copies in `reports/extensions/` also leave out lines, which
a second header line lists, so compare the numbers there rather than diffing the files. When you
diff any other report, ignore these header lines, the timestamp line, printed local paths (such as
the `[written]` line) and package versions.

## Known differences

Apart from those lines, expect these differences when you diff:

- `reports_orig/oracle_anatomy_orig.txt`: one per-bin recall count is 2336 instead of 2337. All AP
  values are equal. See [`tools/orig/README.md`](../tools/orig/README.md).
- `reports/final_run/c3_conditioned_analysis.md` has a hand-written verdict section that the script
  does not print.
- `reports_orig/emh_orig.txt` is the saved stdout of `tools/orig/a5_emh_orig.py`, with header
  lines and a READ note added by hand. The script writes no file.

`tools/extensions/difficulty_class_extension.py` and `tools/crossbench/` have not been re-run on
the released files.

## Where each result comes from

| Paper item | Report | Script |
|---|---|---|
| Table 1, supp. C: official vs reproduced numbers | `reports/detector_adapters_gates.md` | `adapters/` |
| Table 1: Easy/Moderate/Hard for MonoFlex\*, MonoGround\* | `reports_orig/emh_orig.txt` | `tools/orig/a5_emh_orig.py` |
| Table 1, supp. D: native ordering gap for MonoFlex\*, MonoGround\* | `reports_orig/a5_regate_orig.txt`, `reports/final_run/a5_regate.txt` (modern rebuild) | `tools/orig/a5_regate_orig.py` |
| Table 1, Sec. 5.1: matching ceiling AP\* and ordering gap | `reports/exp1_true_ceiling.txt`, `reports_orig/exp1_true_ceiling_orig.txt` | `tools/decomp/exp1_true_ceiling.py`, `tools/orig/exp1_ceiling_orig.py` |
| Fig. 3b, Sec. 5.1, supp. D: all-point vs AP_R40 vs AP_R11 | `reports/gap_exact.txt`, `reports/gap_metric_robustness.txt`, `reports_orig/gap_exact_orig.txt`, `reports_orig/gap_metric_robustness_orig.txt` | `tools/decomp/gap_exact.py`, `tools/decomp/gap_metric_robustness.py`, `tools/orig/gap_exact_orig.py`, `tools/orig/gap_metric_robustness_orig.py` |
| Fig. 3a, supp. D: true-IoU re-sort at native operating points | `reports/final_run/native_gap_canonical.md`, `reports/operating_point_sweep.txt`, `reports/detr_native_sweep.txt`, `reports_orig/operating_point_sweep_orig.txt` | `tools/decomp/operating_point_sweep.py`, `tools/decomp/detr_native_sweep.py`, `tools/orig/operating_point_sweep_orig.py` |
| Supp. D, BEV table: BEV transfer of the 3D-IoU re-sort | `reports/final_run/bev_gap_probe.txt` (ten non-starred rows), `reports_orig/bev_transfer_orig.txt`, `reports/extensions/starred_panel_summaries.txt` (Part D) | `tools/orig/bev_transfer_orig.py`, `tools/extensions/starred_panel_summaries.py` |
| Supp. D: BEV re-sort oracle O1. Sec. 5.2: geometry oracle O2 | `reports/oracle_ladder_o1.txt`, `reports/oracle_ladder_o2.txt`, `reports_orig/oracle_ladder_o1_orig.txt`, `reports_orig/oracle_ladder_o2_orig.txt`, `reports/extensions/starred_panel_summaries.txt` (Parts B, C) | `tools/extensions/starred_panel_summaries.py` |
| Sec. 5, supp. E: FP demotion and TP-only re-ordering at IoU 0.7 and 0.5 | `reports/e4_fp_tp_decomp.txt`, `reports/e4_prereg.md` (pre-registration), `reports/final_run/c2_iou05_separation.txt`, `reports_orig/c2_iou05_orig.txt` | `tools/decomp/e4_fp_tp_decomp.py`, `tools/decomp/c2_iou05_separation.py`, `tools/orig/diffsweep_orig.py`, `tools/orig/c2_iou05_orig.py` |
| Fig. 1, Sec. 4: recall, depth error, score-quality correlation | `reports/final_run/probe_official_moderate.txt`, `reports/probe_detector_progression.txt`, `reports_orig/probe_official_moderate_orig.txt`, `reports_orig/probe_orig.txt` | `tools/decomp/probe_official_moderate.py`, `tools/decomp/probe_detector_progression.py`, `tools/orig/probe_official_orig.py`, `tools/orig/probe_orig.py` |
| Sec. 4, supp. H: paired common-object control of the depth-error trend | `reports/paired_perbin_dz.txt`, `reports/paired_common_object.txt` | `tools/decomp/paired_perbin_dz.py`, `tools/decomp/paired_common_object.py` |
| Fig. 4, Sec. 5.2: oracle anatomy (pure-z, ray-consistent centre, full centre) | `reports/oracle_anatomy.txt`, `reports_orig/oracle_anatomy_orig.txt` | `tools/decomp/oracle_anatomy.py`, `tools/orig/anatomy_orig.py` |
| Sec. 5.2, supp. H: accuracy-separation AUROC | `reports/depth_share_bridge.txt`, `reports_orig/depth_share_bridge_orig.txt` | `tools/decomp/depth_share_bridge.py`, `tools/orig/bridge_orig.py` |
| Sec. 5.2: score x quality product and mean rank | `reports/depth_share_combo.txt` | none |
| Supp. H: far-field recall beyond 45 m with Wilson CIs | `reports/extensions/starred_panel_summaries.txt` (Part A), `reports/final_run/a3_farfield_integrity.txt` (modern rebuild) | `tools/extensions/starred_panel_summaries.py` |
| Table 3: drive-disjoint transfer of fitted depth correction | `reports_orig/bootstrap_floor_orig.txt` (the `transfer` rows are the twelve cells and their CIs), `reports/clean_transfer_strong.txt`, `reports_orig/clean_transfer_orig.txt`, `reports/drive_grouped_oof.md` | `tools/decomp/clean_transfer_save.py`, `tools/decomp/bootstrap_floor.py`, `tools/orig/transfer_save_orig.py`, `tools/orig/bootstrap_orig.py`, `tools/decomp/clean_transfer_strong.py`, `tools/orig/transfer_orig.py`, `tools/decomp/phase1_drive_oof.py` |
| Sec. 7, supp. G: drive-cluster bootstrap floors, endpoint bootstrap | `reports_orig/bootstrap_floor_orig.txt`, `reports/final_run/e3_endpoint_gap_boot.txt` | `tools/decomp/bootstrap_floor.py`, `tools/orig/bootstrap_orig.py`, `tools/decomp/e3_endpoint_gap_boot.py` |
| Supp. F: gap-vs-base Spearman, three bases | `reports_orig/spearman_gap_base_vB.txt` | `tools/orig/spearman_gap_base_vB.py` |
| Supp. H: cross-detector disagreement | `reports/final_run/c3_conditioned_analysis.md` | `tools/decomp/c3_conditioned.py` |
| Supp. I, J: GT states, pool waterfall, replacement, budget truncation | `reports/pool_waterfall.txt`, `reports/e12_replacement.txt`, `reports/gt_state_matrix.txt`, `reports/final_run/e2_budget_truncation.txt`, `reports_orig/pool_waterfall_orig.txt`, `reports_orig/e12_replacement_orig.txt`, `reports_orig/gt_state_matrix_vB.txt`, `reports/extensions/gt_state_recount.txt` | `tools/decomp/pool_waterfall.py`, `tools/decomp/e12_replacement.py`, `tools/decomp/gt_state_matrix.py`, `tools/decomp/e2_budget_truncation.py`, `tools/orig/waterfall_orig.py`, `tools/orig/e12_orig.py`, `tools/orig/gt_state_matrix_vB.py`, `tools/extensions/gt_state_recount.py` |
| Supp. K: results by KITTI difficulty | `reports/final_run/difficulty_sweep/`, `reports_orig/difficulty_sweep_orig/` | `tools/decomp/diffsweep_headroom_sep.py`, `tools/decomp/diffsweep_anatomy.py`, `tools/orig/diffsweep_orig.py`, `tools/orig/diffsweep_anatomy_orig.py` |
| Supp. L: matched candidate budgets | `reports/extensions/budget_matched.txt`, `reports/extensions/pool_census.txt`, `reports/extensions/budget_saturation.txt` | `tools/extensions/budget_matched.py` with `tools/extensions/budget_matched_finalize.py`, `tools/extensions/pool_census.py`, `tools/extensions/budget_saturation.py` |
| Supp. M: near misses among uncovered GT | `reports/extensions/nearmiss_panelwide.txt` | `tools/extensions/nearmiss_panelwide.py` |
| Supp. N: power of the trend test | `reports/extensions/power_analysis.txt` | `tools/extensions/power_analysis.py` |
| Supp. O: Pedestrian and Cyclist | `reports/extensions/difficulty_class_extension.txt`, `reports/extensions/class_gate_audit.txt`, `reports/extensions/class_capability.txt` | `tools/extensions/difficulty_class_extension.py`, `tools/extensions/class_capability.py` |
| Supp. P: Waymo and nuScenes | `reports/extensions/crossbench_audit.md`, `reports/extensions/class_capability.txt` (Part E) | `tools/crossbench/`, `tools/extensions/class_capability.py` |
| Figs. 1, 3, 4 and the waterfall figures of supp. I and J | `figures/data/vB_reports/` | `figures/make_figs_cr.py`, `tools/orig/make_vB_reports.py` |
| Fig. 2 | `figures/fig_unified_src.tex` (TikZ source) | compiled with tectonic |

Some reports have no script in this release and cannot be regenerated from this repository:
`reports/final_run/a5_regate.txt`, `reports/final_run/bev_gap_probe.txt`,
`reports/final_run/a3_farfield_integrity.txt`, the four per-detector oracle ladder reports, and
`reports/depth_share_combo.txt`, which combines the native score with IoU_z\* from
`tools/decomp/depth_share_bridge.py`. The same holds for the collations and records listed in
supplementary section B, such as `reports/final_run/native_gap_canonical.md`,
`reports/leave_two_out.txt`, `reports/final_run/a4_supremum_lb_SUPPLEMENTARY.md`,
`reports/final_run/c6_multiplicity_audit.md` and the pre-registrations. The seven main-panel ladder comparisons in `reports_orig/bootstrap_floor_orig.txt` cannot be recomputed either, because their prediction folders come from a script that is not in this release. The twelve transfer comparisons (Table 3) can (see [`tools/orig/README.md`](../tools/orig/README.md)). Supplementary section B lists the remaining reports.

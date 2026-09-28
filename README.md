# Where Did Monocular 3D Detection Improve? A Twelve-Detector Anatomy of Progress

**SangJin Jung**<sup>1</sup> and **YongHwan Lee**<sup>2,\*</sup><br>
<sup>1</sup> Dankook University, Republic of Korea · <sup>2</sup> Wonkwang University, Republic of Korea · <sup>\*</sup> corresponding author

ACCV 2026 · Paper and supplementary: link will be added when the proceedings are online.

This repository accompanies the paper. It releases the de-quantized evaluator, the per-prediction
dumps of the twelve released detectors and their complete candidate pools, the frozen reports
behind the paper's numbers, and the scripts that produced them.

## The paper in one paragraph

Under one protocol, twelve strictly monocular 3D detectors from 2019 to 2026 span 11.5 to 25.2
all-point AP3D on KITTI Car (Moderate, IoU 0.7). We extend TIDE3D's ranking-error diagnosis to
their released checkpoints and first ask whether such diagnostics can be trusted. The official
AP_R40 quantizes fixed-pool oracle ceilings onto a 2.5-AP lattice: one MonoIA pool's oracle gain
reads +8.9 under AP_R11 and +13.0 under AP_R40. All-point AP removes this evaluator property. The
fitted post-hoc rescoring we tested reads positive on leaky image-fold splits but fails
drive-disjoint evaluation (-2.6 to -13.6 AP). Higher-performing checkpoints show better near-field
localization, and in fixed-pool accounting most of the missing AP is a coverage limit. All twelve
fixed pools still retain +11.4 to +14.0 AP (median +11.7) of oracle ordering headroom on the
matching-ceiling basis. Whether this headroom narrows as detectors improve is unresolved at twelve
checkpoints, not absent. Results are on KITTI val Car, with preliminary checks on Waymo, nuScenes,
Pedestrian and Cyclist.

## What is released

| Path | Content |
|---|---|
| `evaluator/exact_ap.py` | all-point interpolated AP on the official KITTI matching kernels, self-validated against the official AP_R40 and AP_R11 |
| `evaluator/kitti_eval/` | the official KITTI evaluation (Python port, MIT), vendored so nothing else needs to be cloned |
| GitHub release `v1.0` | four zip assets: the per-prediction dumps of all twelve detectors (plus two modern-environment rebuilds), the complete candidate pools of the query-based detectors and of M3D-RPN, and the matched tables of the cross-detector disagreement analysis, see [`data/DUMPS.md`](data/DUMPS.md) |
| `reports/`, `reports_orig/` | the frozen reports behind the numbers of the main paper and the supplementary |
| `reports/extensions/` | the frozen reports of the camera-ready checks (matched budgets, near misses, power of the trend test, Pedestrian and Cyclist, Waymo and nuScenes) and the panel-level summaries that combine `reports/` with the starred rows of `reports_orig/` |
| `tools/decomp/` | the analysis scripts of the twelve-detector panel |
| `tools/orig/` | the same analyses re-run on the original-environment dumps of MonoFlex\* and MonoGround\* ([`tools/orig/README.md`](tools/orig/README.md)) |
| `tools/extensions/`, `tools/crossbench/` | the scripts of the camera-ready checks |
| `adapters/` | how each dump was produced from the official checkpoint, with the two gates it passed |
| `figures/` | regenerates the paper's figures from the frozen reports (`figures/data/vB_reports/`) |

## Environment

- Linux with an NVIDIA GPU. The official KITTI rotated-IoU kernel runs on the GPU through
  `numba.cuda`, as in the upstream evaluator.
- `numba.cuda` also needs the NVVM library and `libdevice` of the CUDA toolkit, which
  `pip install numba` does not provide. Use a system CUDA toolkit (with `CUDA_HOME` set), or
  install them into a conda environment as the release check did:
  `conda install -c nvidia/label/cuda-12.1.0 cuda-nvcc` (tested: cuda-nvcc 12.1.66, NVIDIA
  driver 535.183.01, Quadro RTX 6000). `python -c "from numba import cuda; print(cuda.is_available())"`
  must print `True`, and `python -c "from numba.cuda.cudadrv import libs; libs.test()"` shows where
  NVVM and libdevice were found.
- Python 3.9 or 3.10. `requirements.txt` pins the versions of the release check (Python 3.9.25,
  CUDA 12.1): numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, scikit-learn 1.6.1, numba 0.60.0,
  shapely 2.0.7, and matplotlib 3.9.4 for the figures.
- **Keep numpy at 2.0.x.** M3D-RPN's scores contain ties (2,450 tied rows within images in its
  uniform `cls >= 0.2` pool). The greedy 2D NMS visits boxes in `np.argsort(-score)` order, and
  numpy 1.26 and numpy 2.0 order tied scores differently, so a slightly different set of M3D-RPN
  boxes survives the NMS and M3D-RPN results can differ slightly from the frozen reports.
  Detectors without tied scores in that pool keep the same boxes under both versions.
- Figures only: `figures/verify_cr.py` and `figures/cmp_text.py` need PyMuPDF (`pip install pymupdf`), and Fig. 2
  (`figures/fig_unified_src.tex`, a standalone TikZ file) needs a LaTeX engine such as
  [tectonic](https://tectonic-typesetting.github.io) (`tectonic figures/fig_unified_src.tex`).
  No number depends on either.

## Setup

```bash
git clone https://github.com/sangjin-716/Mono3d-Anatomy.git && cd Mono3d-Anatomy
pip install -r requirements.txt
cp paths.example.py paths.py        # then point KITTI_ROOT at your KITTI copy
```

`paths.py` needs, under `KITTI_ROOT`:

- `training/label_2/` and `training/calib/` of the KITTI 3D object benchmark;
- the Chen split, `ImageSets/val.txt` (3,769 images) and `ImageSets/train.txt`;
- for the drive-disjoint split, the two mapping files of the KITTI object devkit,
  `mapping/train_mapping.txt` and `mapping/train_rand.txt`. `tools/decomp/frame_sequence.py`
  builds the frame-to-drive table from them.

No path is hardcoded anywhere else. Each entry of `paths.py` can also be set by an environment
variable (see `paths.example.py`).

## Download and verify the data

The GitHub release `v1.0` has four assets. Only the first is needed for most results.

| Asset | Content | Size (zip / unzipped) | Needed by |
|---|---|---|---|
| `mono3d_anatomy_dumps_v1.zip` | 14 per-prediction dumps | 443 MB / 1.01 GB | everything |
| `mono3d_anatomy_query_complete_pools_v1.zip` | 5 complete pools of the query-based detectors | 276 MB / 977 MB | native-pool analyses (see below) |
| `mono3d_anatomy_m3drpn_complete_pool_v1.zip` | complete pre-NMS pool of M3D-RPN | 1.20 GB / 2.87 GB | native-pool analyses (see below) |
| `mono3d_anatomy_matched_tables_v1.zip` | 6 prediction-to-GT matched tables | 26.6 MB / 58.3 MB | `tools/decomp/c3_conditioned.py` only |

The native-pool analyses are `e4_fp_tp_decomp`, `exp1_true_ceiling`, `c2_iou05_separation`,
`diffsweep_headroom_sep`, `pool_waterfall`, `e12_replacement`, `gt_state_matrix` (both pool
assets), `detr_native_sweep` (query pools) and `e2_budget_truncation` (M3D-RPN pool) in
`tools/decomp/`, `tools/orig/gt_state_matrix_vB.py`, and six scripts in `tools/extensions/`
(`class_capability.py` reads only the query pools). The full list is in
[`data/DUMPS.md`](data/DUMPS.md), which also lists the columns. `dgp_val.csv` and
`official_monocop_val.csv` carry 10 ground-truth matching columns besides the 23 prediction
columns. Unpack every asset into `DUMP_DIR` (default `data/dumps/`).

```bash
mkdir -p data/dumps && cd data/dumps
BASE=https://github.com/sangjin-716/Mono3d-Anatomy/releases/download/v1.0
for a in dumps query_complete_pools m3drpn_complete_pool matched_tables; do
  wget $BASE/mono3d_anatomy_${a}_v1.zip          # drop the assets you do not need
done
sha256sum -c --ignore-missing <<'EOF'
3eb2d46701f481851b89f6f52c2472805d0467c5b8963fdd8b7d861ab4262edd  mono3d_anatomy_dumps_v1.zip
0a4696a30a166d4b3a0cbfeab92cbaa09ac91ab6769a15b90e45e90ea462362f  mono3d_anatomy_query_complete_pools_v1.zip
5e0bbce9d2bd06eaea8a5954ad4202aa7541a862e995f7137986b4625aa77497  mono3d_anatomy_m3drpn_complete_pool_v1.zip
47d71716b77a7e018bc9ced001c313448ffe3935a1e92c37ea4e7a2609ba8e83  mono3d_anatomy_matched_tables_v1.zip
EOF
for z in mono3d_anatomy_*_v1.zip; do unzip -n "$z" -x 'MD5SUMS*.txt'; done
md5sum -c --ignore-missing ../MD5SUMS.txt     # one OK per unpacked file, 26 with all assets
cd ../..
python tools/_release.py                      # which assets are complete in DUMP_DIR
```

## Validate the evaluator

```bash
python evaluator/exact_ap.py           # toy cases with hand-computable AP, then equality with the
                                       # official do_eval AP_R40 / AP_R11 on two real dumps
python evaluator/exact_ap.py --full    # the same check on all twelve detectors
```

The quick run takes a few minutes. It first prints many `NumbaPerformanceWarning` blocks
("Grid size ... will likely result in GPU under-utilization"; 36 in our runs) and, with newer numba
versions, a warning about `parallel=True`. These warnings are expected and harmless. The run
ends with:

```
toyA perfect-rank: allpoint=100.00 (hand 100.00)  R40 mine/off=2.50/2.50  R11 mine/off=9.09/9.09  [PASS]
toyB tied-scores : allpoint=66.67 (hand 66.67)  R40 mine/off=1.67/1.67  R11 mine/off=6.06/6.06  [PASS]
dgp              R40 off/mine= 22.290/ 22.290  R11 off/mine= 26.215/ 26.215  p101=22.763 p401=22.834 allpoint=22.905 [PASS]
gupnet           R40 off/mine= 16.481/ 16.481  R11 off/mine= 21.865/ 21.865  p101=16.933 p401=17.030 allpoint=17.122 [PASS]
ALL GATES: PASS
```

The all-point AP evaluates every unique detection score as a threshold and integrates the official
interpolated precision envelope over raw recall. Matching is the official one, unchanged.

## Reproduce the reports

Run every script from the repository root. The header of each script names the frozen report it
produces. Re-runs write to `reports_rerun/` under the report's name relative to `reports/` or
`reports_orig/` (so `reports/final_run/X` is re-run as `reports_rerun/final_run/X` and
`reports_orig/Y` as `reports_rerun/Y`), and never touch `reports/` or `reports_orig/`. Caches and
temporary KITTI-format files go to `cache/`.

**Public copies.** Some frozen reports are scrubbed public copies of the originals. Their first
line reads `# public copy of <source>, sha256 <hash of the original>, scrubbed: <what changed>`,
and in `reports/extensions/` a second header line names any omitted lines. For the copies in
`reports/` and `reports_orig/` that a script writes, the script prints the scrubbed wording, so a
re-run differs from the copy only in the `# public copy` first line and the run-timestamp lines
(apart from printed local paths and package versions; `c3_conditioned_analysis.md` also carries a
hand-written verdict section). The `reports/extensions/` copies are compared number by number
(see the release check below).

A few scripts reuse caches written by another one, so run them in this order:

```bash
python tools/decomp/gap_exact.py              # writes cache/decomp/_bridgecache_<det>.npz (S5-pool oracle IoUs)
python tools/decomp/gap_metric_robustness.py  #   reuses it (rebuilds it if missing)
python tools/decomp/e3_endpoint_gap_boot.py   #   reuses it (rebuilds it if missing)

python tools/decomp/e4_fp_tp_decomp.py        # writes cache/decomp/_e4cache_<det>.npz (native-pool oracle IoUs, TP labels)
python tools/decomp/exp1_true_ceiling.py      #   its FP-demotion column is nan without the e4 cache
python tools/decomp/c2_iou05_separation.py    #   reuses the oracle IoUs (recomputes them if missing)
python tools/decomp/diffsweep_headroom_sep.py #   requires the e4 cache

python tools/decomp/gt_state_matrix.py        # writes cache/decomp/gt_state_matrix.csv
python tools/decomp/e2_budget_truncation.py   #   requires it
```

The other scripts in `tools/decomp/` are independent. The order within `tools/orig/` and the
drive-cluster bootstrap (`tools/decomp/bootstrap_floor.py`, `tools/orig/bootstrap_orig.py`, about
30 minutes per comparison) are described in [`tools/orig/README.md`](tools/orig/README.md). On one
Quadro RTX 6000 most scripts take from a few seconds to about an hour; the longest are
`operating_point_sweep.py` (about 3.5 hours), `detr_native_sweep.py` (about 2 hours) and
`diffsweep_anatomy.py` (about 1.8 hours). The figures are drawn from the frozen reports
(`figures/data/vB_reports/`, which `tools/orig/make_vB_reports.py` rebuilds byte for byte from
`reports/` and `reports_orig/`) and need no re-run: `python figures/make_figs_cr.py` redraws
Figs. 1, 3 and 4 and the two waterfall figures of the supplementary into `reports_rerun/figures/`,
and `python figures/verify_cr.py` checks their page size and fonts. They use the system font
DejaVu Sans Condensed (Debian/Ubuntu package `fonts-dejavu-extra`). In our check, with matplotlib
3.9.4 and 3.10.8, the text and vector drawings of the output equal the camera-ready figures
(`python figures/cmp_text.py reports_rerun/figures <folder with the camera-ready PDFs>`).
Against the submitted figures the output has two data corrections, which
`figures/data_identity.py` reports: the grey series of Fig. 1(c) is the native base AP of Table 1,
and the MonoFlex\* grey label of the first supplementary waterfall figure is read from
`reports_orig/pool_waterfall_orig.txt`.

**Release check.** Before release, the ported scripts were re-run on the released files in a fresh
copy of this repository (the pinned environment above, one NVIDIA Quadro RTX 6000), and each
re-run report was compared line by line with the frozen report it reproduces. Apart from the
`# public copy` header lines of scrubbed copies, the run-timestamp line, printed local paths and
printed package versions:

- every script in `tools/decomp/` that writes a frozen report reproduces it exactly. The frozen
  `c3_conditioned_analysis.md` also carries a hand-written verdict section that the script does
  not print;
- every script in `tools/orig/` that writes a frozen report reproduces it exactly, except one
  integer in `reports_orig/oracle_anatomy_orig.txt` (a per-bin recall count, 2336 instead of
  2337; all AP values identical; see `tools/orig/README.md`). The stdout of `a5_emh_orig.py`,
  which wrote no file originally, is kept as `reports_orig/emh_orig.txt`. `gt_state_matrix_vB.py`
  reproduces every number of its report (only its `[written]` path line differs);
- the drive-cluster bootstrap was spot-checked on 4 of its 23 comparisons, all identical;
- `budget_matched.py`, `budget_saturation.py`, `pool_census.py`, `nearmiss_panelwide.py` and
  `power_analysis.py` in `tools/extensions/` reproduce every number of their public copies. Where
  a copy omits lines, its header lines name them; the `budget_saturation` and `pool_census`
  copies omit nothing. `class_capability.txt` and `starred_panel_summaries.txt` were written by the
  released scripts themselves.
  `difficulty_class_extension.py` and `tools/crossbench/` were not re-run.

## Where each result comes from

| Paper item | Frozen report | Script |
|---|---|---|
| Panel gates, official vs reproduced (Table 1, supp. C) | `reports/detector_adapters_gates.md` | `adapters/` |
| Easy/Moderate/Hard cells of MonoFlex\* and MonoGround\* (Table 1) | `reports_orig/emh_orig.txt` | `tools/orig/a5_emh_orig.py` |
| Native ordering-gap cells of MonoFlex\* and MonoGround\* (Table 1, supp. D) | `reports_orig/a5_regate_orig.txt` (modern-environment counterpart: `reports/final_run/a5_regate.txt`) | `tools/orig/a5_regate_orig.py` |
| Matching ceiling AP\* and ordering gap, per detector (Table 1, Sec. 5.1) | `reports/exp1_true_ceiling.txt`, `reports_orig/exp1_true_ceiling_orig.txt` | `tools/decomp/exp1_true_ceiling.py`, `tools/orig/exp1_ceiling_orig.py` |
| Sampled-recall quantization, all-point vs AP_R40 vs AP_R11 (Fig. 3b, Sec. 5.1, supp. D) | `reports/gap_exact.txt`, `reports/gap_metric_robustness.txt`, `reports_orig/gap_exact_orig.txt`, `reports_orig/gap_metric_robustness_orig.txt` | `tools/decomp/gap_exact.py`, `tools/decomp/gap_metric_robustness.py`, `tools/orig/gap_exact_orig.py`, `tools/orig/gap_metric_robustness_orig.py` |
| True-IoU re-sort at native operating points (Fig. 3a, supp. D) | `reports/final_run/native_gap_canonical.md`, `reports/operating_point_sweep.txt`, `reports/detr_native_sweep.txt`, `reports_orig/operating_point_sweep_orig.txt` | `tools/decomp/operating_point_sweep.py`, `tools/decomp/detr_native_sweep.py`, `tools/orig/operating_point_sweep_orig.py` |
| Cross-metric (BEV) transfer of the 3D-IoU re-sort (supp. D, BEV table) | `reports/final_run/bev_gap_probe.txt` (the paper takes its ten non-starred rows), `reports_orig/bev_transfer_orig.txt` (MonoFlex\*, MonoGround\*), panel summary in `reports/extensions/starred_panel_summaries.txt` (Part D) | `tools/orig/bev_transfer_orig.py`, `tools/extensions/starred_panel_summaries.py`; `bev_gap_probe.txt` has no script in this release |
| BEV-IoU re-sort oracle O1 (supp. D) and single-factor geometry oracle O2 (Sec. 5.2) | `reports/oracle_ladder_o1.txt`, `reports/oracle_ladder_o2.txt`, `reports_orig/oracle_ladder_o1_orig.txt`, `reports_orig/oracle_ladder_o2_orig.txt`; panel medians and ranges in `reports/extensions/starred_panel_summaries.txt` (Parts B, C) | per-detector reports: no script in this release; summaries: `tools/extensions/starred_panel_summaries.py` |
| FP demotion and TP-only re-ordering, IoU 0.7 and 0.5 (Sec. 5, supp. E) | `reports/e4_fp_tp_decomp.txt` (pre-registration `reports/e4_prereg.md`), `reports/final_run/c2_iou05_separation.txt`, `reports_orig/c2_iou05_orig.txt` | `tools/decomp/e4_fp_tp_decomp.py`, `tools/decomp/c2_iou05_separation.py`, `tools/orig/diffsweep_orig.py`, `tools/orig/c2_iou05_orig.py` |
| Progression: recall, depth error, score-quality correlation (Fig. 1, Sec. 4) | `reports/final_run/probe_official_moderate.txt`, `reports/probe_detector_progression.txt`, `reports_orig/probe_official_moderate_orig.txt`, `reports_orig/probe_orig.txt` | `tools/decomp/probe_official_moderate.py`, `tools/decomp/probe_detector_progression.py`, `tools/orig/probe_official_orig.py`, `tools/orig/probe_orig.py` |
| Paired common-object control of the depth-error trend (Sec. 4, supp. H) | `reports/paired_perbin_dz.txt`, `reports/paired_common_object.txt` | `tools/decomp/paired_perbin_dz.py`, `tools/decomp/paired_common_object.py` |
| Oracle anatomy: pure-z, ray-consistent centre, full centre (Fig. 4, Sec. 5.2) | `reports/oracle_anatomy.txt`, `reports_orig/oracle_anatomy_orig.txt` | `tools/decomp/oracle_anatomy.py`, `tools/orig/anatomy_orig.py` |
| Accuracy-separation AUROC (Sec. 5.2, supp. H) | `reports/depth_share_bridge.txt`, `reports_orig/depth_share_bridge_orig.txt` | `tools/decomp/depth_share_bridge.py`, `tools/orig/bridge_orig.py` |
| Score-combination probe, score x quality product and mean rank (Sec. 5.2) | `reports/depth_share_combo.txt` | no script in this release (it combines the native score with the IoU_z\* of `tools/decomp/depth_share_bridge.py`) |
| Far-field recall beyond 45 m with Wilson CIs (supp. H) | `reports/extensions/starred_panel_summaries.txt` (Part A, panel of the paper), `reports/final_run/a3_farfield_integrity.txt` (modern-environment MonoFlex and MonoGround) | `tools/extensions/starred_panel_summaries.py`; `a3_farfield_integrity.txt` has no script in this release |
| Drive-disjoint transfer of fitted rescoring (Table 3) | `reports/clean_transfer_strong.txt`, `reports_orig/clean_transfer_orig.txt`, `reports/drive_grouped_oof.md` | `tools/decomp/clean_transfer_strong.py`, `tools/orig/transfer_orig.py`, `tools/decomp/phase1_drive_oof.py` |
| Drive-cluster bootstrap floors, endpoint bootstrap (Sec. 7, supp. G) | `reports_orig/bootstrap_floor_orig.txt`, `reports/final_run/e3_endpoint_gap_boot.txt` | `tools/decomp/bootstrap_floor.py`, `tools/orig/bootstrap_orig.py`, `tools/decomp/e3_endpoint_gap_boot.py` |
| Gap-vs-base Spearman, three bases (supp. F) | `reports_orig/spearman_gap_base_vB.txt` | `tools/orig/spearman_gap_base_vB.py` |
| Cross-detector disagreement (supp. H) | `reports/final_run/c3_conditioned_analysis.md` | `tools/decomp/c3_conditioned.py` (needs the matched tables) |
| GT states, candidate-pool waterfall, replacement, budget truncation (supp. I, J) | `reports/pool_waterfall.txt`, `reports/e12_replacement.txt`, `reports/gt_state_matrix.txt`, `reports/final_run/e2_budget_truncation.txt`, `reports_orig/pool_waterfall_orig.txt`, `reports_orig/e12_replacement_orig.txt`, `reports_orig/gt_state_matrix_vB.txt`; panel-unreached shares over all pools and over the non-anchor pools in `reports/extensions/gt_state_recount.txt` | `tools/decomp/pool_waterfall.py`, `tools/decomp/e12_replacement.py`, `tools/decomp/gt_state_matrix.py`, `tools/decomp/e2_budget_truncation.py`, `tools/orig/waterfall_orig.py`, `tools/orig/e12_orig.py`, `tools/orig/gt_state_matrix_vB.py`, `tools/extensions/gt_state_recount.py` (reads the per-GT CSVs the two `gt_state_matrix` scripts write) |
| Robustness across KITTI difficulty (supp. K) | `reports/final_run/difficulty_sweep/`, `reports_orig/difficulty_sweep_orig/` | `tools/decomp/diffsweep_headroom_sep.py`, `tools/decomp/diffsweep_anatomy.py`, `tools/orig/diffsweep_orig.py`, `tools/orig/diffsweep_anatomy_orig.py` |
| Matched candidate budgets (supp. L) | `reports/extensions/budget_matched.txt`, `reports/extensions/pool_census.txt`, `reports/extensions/budget_saturation.txt` | `tools/extensions/budget_matched.py` (+ `budget_matched_finalize.py`), `tools/extensions/pool_census.py`, `tools/extensions/budget_saturation.py` |
| Near misses among uncovered GT (supp. M) | `reports/extensions/nearmiss_panelwide.txt` | `tools/extensions/nearmiss_panelwide.py` |
| Power of the trend test (supp. N) | `reports/extensions/power_analysis.txt` | `tools/extensions/power_analysis.py` |
| Pedestrian and Cyclist (supp. O) | `reports/extensions/difficulty_class_extension.txt`, `reports/extensions/class_gate_audit.txt`, `reports/extensions/class_capability.txt` | `tools/extensions/difficulty_class_extension.py`, `tools/extensions/class_capability.py` |
| Waymo and nuScenes preliminary audit (supp. P) | `reports/extensions/crossbench_audit.md`, `reports/extensions/class_capability.txt` (Part E) | `tools/crossbench/`, `tools/extensions/class_capability.py` |
| Figs. 1, 3, 4 and the waterfall figures of supp. I and J | `figures/data/vB_reports/` (the frozen reports with the MonoFlex\* and MonoGround\* rows from `reports_orig/`) | `figures/make_figs_cr.py`, `tools/orig/make_vB_reports.py` |
| Fig. 2 | `figures/fig_unified_src.tex` (TikZ source) | compiled with tectonic |

The supplementary's provenance table (supp. B) lists the remaining reports. Some of them are
collations or records without a script of their own, such as `native_gap_canonical.md`,
`leave_two_out.txt`, `a4_supremum_lb_SUPPLEMENTARY.md`, `c6_multiplicity_audit.md` and the
pre-registrations. The reports marked above as having no script in this release (and
`reports/final_run/a5_regate.txt`, the modern-environment counterpart of `a5_regate_orig.txt`)
are frozen outputs that cannot be re-derived from this repository.

## Notes

- **MonoFlex\* and MonoGround\*.** These two detectors share the MonoFlex codebase and reproduce their
  published numbers only in the authors' original torch-1.4 environment. The panel uses those
  outputs, and `reports_orig/` holds the recomputations on them. The frozen reports in `reports/`
  were computed on modern-environment rebuilds of the same checkpoints, which are released too.
- **What the oracles are.** The matching ceiling and the re-sorts are diagnostic bounds on a fixed
  pool of boxes. They are not realizable gains.
- **Scope.** All claims are about the twelve released checkpoints on KITTI val Car. The Waymo,
  nuScenes, Pedestrian and Cyclist checks are a preliminary audit, not a replication.

## Citation

```bibtex
@inproceedings{jung2026where,
  title     = {Where Did Monocular 3D Detection Improve? A Twelve-Detector Anatomy of Progress},
  author    = {Jung, SangJin and Lee, YongHwan},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## License

Code: Apache-2.0 (`LICENSE`). Frozen reports and figure data: CC BY 4.0 (`reports/LICENSE` for
`reports/` and `reports_orig/`, `figures/LICENSE` for `figures/data/` and
`figures/fig_unified_src.tex`).
Released data (dumps, complete pools, matched tables): CC BY-NC-SA 4.0, because they are model
outputs on KITTI images (`data/DUMPS.md`). The vendored KITTI evaluation and calibration code keep
their MIT licenses (`evaluator/kitti_eval/LICENSE`, `evaluator/LICENSE-MonoDGP`).

We thank the authors of the twelve detectors for releasing their code and checkpoints, and the
KITTI, Waymo Open Dataset and nuScenes teams.

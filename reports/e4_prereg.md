# public copy of reports/e4_prereg.md, sha256 df83974b6d30306cdf92817bb92ba97e8f3c7f880512ea867a6082da395e58c4, scrubbed: approval and review-process wording removed (source lines 2, 5, 58); definitions and numbers verbatim
# E4 pre-registration — FP-demotion vs TP-quality-reordering decomposition
# (FIXED BEFORE RUNNING; 2026-06-12)

All definitions below are frozen before any E4 result is computed. No oracle definition may
change after seeing results. Only existing validated dumps are used; budget
8h; after E4, full analysis freeze.

## Pools (per model, the audited NATIVE pools of the headline range)
- query 5: validated v2 pre-flatten dump, Car-channel rows, native survivors = `in_final`
  flag (assert ≡ flat_rank<50 ∧ cls≥0.2 where both derivable); ranking score = V.
- CenterNet 6: established full dumps, native threshold only (MonoCon 0.4, others 0.2;
  MonoDLE thresholds its cls column, others V — pool_waterfall.py conventions), no box-NMS;
  ranking score = V.
- M3D-RPN: floor-0 dump → native writer stages NMS@0.4 → top-40 → V≥0.75 (E12 code path);
  ranking score = V.

## Evaluator
exact_ap.ap_summaries (validated official kernels), KITTI Car moderate. all-point AP
primary, official R40 secondary. Same GT annos as gap_exact.py.

## Oracle quality o_act
Per-box actual 3D IoU via depth_share_bridge.iou_act_and_zstar on each native pool (same
construction as the published S5 ceilings). Boxes overlapping no GT get o_act=0.

## TP/FP labels (FROZEN from the baseline ordering; used ONLY to build re-orderings)
Per frame: boxes in native-V-descending order, greedy one-to-one match to same-frame Car
GTs (any difficulty) at IoU3D ≥ 0.7 via the arc matcher — the SAME construct that produced
the 12–31% kept-pool FP-mix numbers motivating this experiment. TP = matched, FP = else.
Disclosure: the official evaluator re-matches internally when scoring each re-ordering, so
matched-pair reassignment effects are INCLUDED in every measured AP (set-level measurement,
no per-box bookkeeping is claimed). Same-GT duplicates: only the first-matched box of a GT
is TP; later duplicates are FPs (native query pools carry 6–16% duplicates — disclosed).

## The four orderings (global scores; AP threshold sweep is global)
1. baseline: native V.
2. FP-demotion-only: s' = V + (max V over all models' pool + 1) for TPs; s' = V for FPs.
   → every TP above every FP; TP internal order = native; FP internal order = native.
3. TP-quality-reordering-only: FPs keep native V unchanged; the multiset of TP native-V
   values is reassigned among TPs by o_act-descending order (tie: higher native V first).
   → FP global positions & relative order preserved exactly; TPs permuted within the
   TP-score slots only.
4. full oracle: s' = o_act (existing fixed-pool true-IoU ordering).
No additivity is assumed or claimed between (2), (3), and (4).

## Gates (run aborts interpretation on failure)
- GA baseline: all-point base must reproduce the published native-cell bases
  (21.18/22.82/24.34/24.55/25.18; 15.09/16.27/17.11/17.48/17.47/19.59; 11.49) within ±0.05.
- GB full oracle: all-point gap must reproduce the published native gap cells
  (+11.42/+8.41/+11.58/+9.73/+11.64 query; +14.03/+10.28/+12.04/+11.83/+10.80 CenterNet)
  within ±0.10; MonoCon/M3D-RPN (no exact published native gap cell) within ±0.40 of their
  uniform-pool cells (+11.36/+11.60), labeled. If GB fails from o_act construction drift,
  the pre-declared remedy is to align o_act construction to reproduce the PUBLISHED ceiling
  (a fixed external target) — never to adjust oracles (2)/(3) or to re-pick after results.

## Reporting
Per-model: base / FP-demotion / TP-reorder / full (all-point, R40 in parens) + the three
gaps; family medians; share table gap_FPdem vs gap_TPreorder vs gap_full. Interpretation
rules fixed in advance: TP-reordering contribution substantial → localization-aware
ordering reading stands; FP-demotion dominant → headline narrows to "FP filtering + TP
quality ordering combined"; the full-oracle gap itself is unaffected either way.

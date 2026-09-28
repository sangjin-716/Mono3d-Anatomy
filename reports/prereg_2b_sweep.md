# public copy of reports/prereg_2b_sweep.md, sha256 395a33f1f97e392ee80b41ea453fa2bf02fefe4c2c1344dfe872d55e344566fa, scrubbed: retired internal framing term replaced (source line 67), review-process phrase removed (source line 76); criteria and numbers verbatim
# Pre-registration — #2b de-quantization + operating-point battery (2026-06-11)

Committed BEFORE running #2b. Any deviation requires a dated amendment section at the bottom.
Purpose: fix grids, definitions, primary-vs-secondary designation, and survival criteria so that
no post-hoc selection of favorable settings is possible.

## Fixed grids
- Score threshold sweep: cls-thr ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 0.4} (0.2 = S5 primary cell)
- Top-K per image sweep: K ∈ {20, 50, 100, all-kept}
- NMS IoU sweep: {0.4, 0.5, 0.6, none} (0.5 = S5 primary cell)
- Detectors: operating-point sweep on the 7 non-DETR (DETR family shown near-insensitive in
  p0_5b_sensitivity); metric sweep (R11/R40/dense/exact) on all 12.
- PRIMARY result cell (pre-declared): S5 = thr0.2 + NMS@0.5, AP3D R40 IoU0.7 mod — everything
  else is SECONDARY/diagnostic. Headline numbers come from the primary cell with secondary
  robustness ranges quoted alongside.

## Continuous (de-quantized) AP definition — EXACT, not denser sampling
AP_exact: build the PR curve at EVERY unique score threshold (all detections sorted by score;
cumulative TP/FP under the official matching semantics including ignored/Van/Truck/DontCare
handling), apply the monotonic precision envelope p_interp(r) = max_{r' >= r} p(r'), and
integrate exactly (stepwise) over recall. The sample-point ladder 41 -> 401 -> 801 is used ONLY
as a convergence check toward AP_exact (expected |AP_801 - AP_exact| < 0.1 on the 2 calibration
detectors: MonoDGP, GUPNet); it is not itself the de-quantized estimate.
Implementation must be validated against the official evaluator at 41 points (identical R40
values) before any exact number is used.

## Matched-recall operating points — diagnostic ONLY
Per-detector thresholds tuned (using val GT) to equalize recall across detectors = a GT-derived
DIAGNOSTIC to control score-scale differences across families. Reported in a separate table,
labeled non-deployable; no deployable/achievable claim may cite it. Native operating points
reported alongside.

## Pre-declared survival criteria for the C1 ordering-headroom claim
The claim ("a large, non-closing ordering headroom exists across the panel") SURVIVES iff,
under EACH of {R11, R40, AP_exact} at the primary cell:
  (a) order-ceiling gap > 0 for 12/12 detectors;
  (b) panel median gap >= 6 AP;
  (c) cross-detector gap sd <= 2.5 AP;
  (d) spearman(gap, base AP) in [-0.5, +0.5] (no systematic closing with strength).
AND under the operating-point sweep (7 non-DETR):
  (e) at all four predeclared gap cells [(0.2,0.5)=S5, (0.1,0.5), (0.3,0.5), (0.2,none)],
      gap > 0 and median(non-DETR gaps) >= 6 AP;   [wording corrected, see Amendments]
  (f) the S5-cell gap differs from the per-detector best-predeclared-cell gap by < 3 AP for
      >= 6/7. STATUS NOTE: (f) is retained and reported as the ORIGINAL PREREGISTERED S5
      DIAGNOSTIC; because S5 is not the native operating point for every family, the
      paper-level operating-point robustness claim gate is the native/neighborhood criterion
      defined in reports/claim_decision_tree.md, with (f) always reported in parallel.
IoU0.5 column: if the IoU0.5 gap falls below half of the IoU0.7 gap for >= 6/12, the claim is
re-scoped as IoU0.7-specific (precision-regime claim), not dropped.
If (a)-(f) hold: wording may say "large and stable across metric summaries and operating
points". If any fails: wording downgraded per GATE-2 (claim gate), and the failure itself is
reported as a finding (e.g., "the R40 gap partially reflects recall quantization").

## Known-before-running facts (disclosed)
- All 12 R40 order-ceilings sit exactly on the 2.5-AP grid (recall quantization).
- 3-detector probe: gap_R11 = +10.21/+10.15/+8.85 vs gap_R40 = +13.90/+12.71/+13.01
  (M3D-RPN/MonoDGP/MonoIA) — shrink expected; criteria above were set knowing this.
- #2a (R11/R40/IoU0.5, 12 detectors) launched before this prereg; #2a is treated as part of the
  same battery and its results were NOT seen before fixing these criteria (job still running at
  commit time).

## Amendments
- 2026-06-11 (before any #2b sweep result was read; gap_exact 12-detector run restarted):
  (a) NAMING: "exact-AUC / continuous AP" -> "all-point interpolated AP (all-threshold PR
      area)" everywhere — recall remains discrete, precision uses the official interpolation
      envelope, integration is stepwise; not a physical AUC and not "the only truth": it judges
      GAP SIZE only; depth-relatedness and the missing/suppressed split stay gated on #1/#5.
  (b) BUG DISCLOSURE: a first gap_exact run labeled eval_param(num_sample_pts=11) as "R11" —
      that is NOT official R11 (official = 41-pt envelope sampled at indices 0,4,...,40). Run
      killed, partial report deleted, no number from it was used. Official R11 now computed
      from the same 41-pt envelope as R40 (verified == do_eval on toys + 2 detectors).
  (c) BUG DISCLOSURE: first all-point implementation enveloped recall as well as precision
      (allpoint=9.16 vs param401=22.83 non-convergence caught it); fixed to raw recall +
      explicit recall-tie collapse to max envelope precision; no file artifact ever contained
      the buggy number.
  (d) VALIDATION extended: toy hand-math gates (perfect-rank, tied-score),
      toy R40/R11 == official do_eval, and an embedded per-detector gate in gap_exact.py
      (base-pool ap_summaries must match official do_eval R40 AND R11 <0.01 for ALL 12, else
      abort). Definitions/criteria of this prereg otherwise unchanged.
- 2026-06-12 (during the sweep run; gap cells NOT yet observed):
  (e) wording corrected from "every grid point" to "all four predeclared gap cells" — the
      implemented sweep computes gaps only at the four predeclared cells; the old wording
      promised more than the implementation evaluates. Base-surface cells (no gap) are
      reported descriptively.
  (f) retained, computed, and reported as the ORIGINAL PREREGISTERED S5 DIAGNOSTIC — not
      deleted, not hidden. Reason for adding a separate paper-level gate: a PROTOCOL MISMATCH
      discovered by code audit (S5's 2D-NMS is non-native for the CenterNet family and the
      anchor family's native point is NMS0.4/top-40/thr0.75), NOT any observed result.
      Observation status at this amendment: M3D-RPN base-surface cells had been observed;
      ALL gap cells (every detector) were still unobserved. Both (f) and the native-based
      criterion will be reported side by side to preclude selective use.

# public copy of reports/prereg_2b_detr_sweep.md, sha256 8c566857105f284985edf371862407a7257c3e757b2e65be8d369b10aeadc1e9, scrubbed: instruction wording removed (source lines 8, 117, 124); claim-scope labels neutralized (source lines 29-30, 32, 120); definitions, grids and numbers verbatim
# Pre-registration — #2b-DETR: query-native operating-point robustness (2026-06-12, rev2)

Committed BEFORE running any DETR-native sweep. Amendments require dated entries at the bottom.
Companion to reports/prereg_2b_sweep.md (7 non-DETR) and reports/detr_inference_audit.md
(code-evidence basis). Purpose: extend operating-point robustness to the 5 query-based models
WITHOUT forcing non-native knobs (no NMS sweep; no query-budget sweep — meaningless w/o retrain).

## rev2 (BEFORE any redump or DETR sweep ran)
1. Conditional re-dump REPLACED by **unconditional pre-flatten redump of all 5 models** — the
   trigger was circular (it used quantities the current post-crowding dumps cannot observe).
2. top-K split into TWO axes, never merged: **K_flat** (class-hypothesis budget: how many of the
   150 query×class hypotheses survive by raw cls) and **K_final** (final detection budget by V
   after eligibility). They measure different failure causes.
3. DETR waterfall FIXED (no NMS stage), in the audited NATIVE order:
   all 150 query×class hypotheses → **K_flat budget (top-50 by raw cls)** → **class-score
   threshold 0.2** → V=cls·exp(−σ) ranking → K_final/output.
   (rev3 fix: an earlier revision wrote threshold→K_flat — that is NOT the native order;
   extract_dets_from_outputs applies topk FIRST, decode_detections thresholds AFTER. The
   threshold→K_flat ordering may appear only as a counterfactual SECONDARY diagnostic and must
   never be labeled native.)
4. **Score-path inconsistency diagnostic added**: Native(gate=raw cls, rank=V) vs
   cls-consistent(gate=cls, rank=cls) vs V-consistent(gate=V, rank=V). cls and V scales differ —
   never compare equal numeric thresholds; control by matched candidate count or matched recall
   (GT-derived ⇒ diagnostic-only label). This diagnoses eligibility/ranking score inconsistency;
   it is NOT a method claim.
5. **Native-reconstruction unit-test gate**: final predictions reconstructed from the redump must
   match the in-process native pipeline output on box/class/score/query-id/count per image AND
   AP3D|R40 — any mismatch ABORTS the sweep until fixed.
6. Claim-scope limits: "5/5 studied query-based models" in scope; "five independent DETR
   architectures" and "diverse DETR families" out of scope (one shared MonoDETR-family pipeline).
   DETR-internal results are scoped "within the studied MonoDETR-family pipeline". Cross-family
   wording in scope: "the metric-level ordering headroom is observed across the studied
   query-based and non-query-based families."
7. Any analysis run on the EXISTING post-crowding dumps must be labeled
   "post-crowding 50-candidate diagnostic" and may NOT serve as GATE-2 evidence; final verdicts
   use the pre-flatten redump only.

## Models & native operating point
MonoDETR, MonoDGP, MonoCoP, MonoCLUE, MonoIA (audit: all genuinely query-based, shared pipeline).
NATIVE point (primary): thr 0.2 on raw cls, NO NMS, rank by V = cls·exp(−σ_depth).
S5 (+2D-NMS@0.5) is reported alongside as the uniform-panel diagnostic — labeled non-native.

## Pre-flatten redump (UNCONDITIONAL, all 5 models) — the primary data source
Per image, per query (first 50 = the inference set), store at minimum:
  decoded 3D box (full 23-col geometry, Car mean-size convention), ALL foreground class
  probabilities (cls_car, cls_ped, cls_cyc; sigmoid — no explicit no-object logit exists in
  this family, the implicit no-object is the all-zeros row), depth-uncertainty log_sigma_raw &
  sigma, native ranking score V_car = cls_car·exp(−log_sigma), query_id, argmax class id,
  decoder layer id (last layer for primary; aux layers only under H), flat_rank_car (rank of the
  (q,Car) hypothesis among the 150 by raw cls), native_top50 flag ((q,Car) ∈ native top-50-of-150),
  thr_pass flag (cls_car ≥ 0.2), in_final flag (native_top50 ∧ thr_pass).
Files: experiments/dgp_cop_diag/<f>_val_preflatten.csv. Cost ≈ 1h GPU total, ≈ 400 MB.

## Native-reconstruction unit test (ABORT gate, runs inside the redump) — HYPOTHESIS-LEVEL
The unit of reconstruction is the (query_id, class_id) HYPOTHESIS, not the query: a query's
several class hypotheses compete inside the 150→50 K_flat budget, so query-level flags alone
cannot reproduce native crowding. The dump stores per-query rows carrying ALL class
probabilities + σ, from which every hypothesis state (raw class prob, V_class, flat rank,
native-top-50 membership, threshold pass, final-output membership) is DERIVED deterministically.
Gate: in-process, on the SAME output tensors, reconstruct the full ALL-CLASS native pipeline
(150 → K_flat top-50 by raw cls → thr0.2 → scores cls·exp(−σ)) from the stored columns and
compare against the repo's extract_dets_from_outputs + decode_detections output at hypothesis
level: per image, the (class label, score) multiset must match (<1e-4) and counts must match
for ALL classes; additionally Car rows must match on 3D box (<1e-3). The reconstructed Car
predictions' AP3D|R40 must equal the native-path AP3D|R40 (<0.01). Mismatch ⇒ sweep blocked.

## Fixed grids (PRIMARY analyses; on the PRE-FLATTEN redump) — all in NATIVE stage order
Stage order everywhere: K_flat (top by raw cls over 150) → threshold → V ranking → K_final.
- C. threshold sweep AT native K_flat=50: thr ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 0.4} (0.2 native)
- D-flat. **K_flat sweep** (class-hypothesis budget over 150 by raw cls, BEFORE thr):
  K_flat ∈ {25, 50(native), 100, 150(=no crowding)}, thr fixed at 0.2 — measures Car-hypothesis
  loss to cross-class crowding.
- D-final. **K_final sweep** (final detection budget by V after K_flat50+thr0.2): K_final ∈
  {10, 20, 30, 50(native=all)}. NEVER merged with D-flat.
- F. order-ceiling gap at 4 pre-declared (K_flat, thr) cells: (50, 0.2) = NATIVE primary;
  (150, 0.0) = full pool; (50, 0.3); (150, 0.2) = no-crowding counterfactual.
- A. per-GT accurate-candidate existence in the full 50-query pre-flatten pool (IoU3D ≥ {0.5,0.7}),
  per distance bin; crowding-loss = existence(K_flat150) − existence(K_flat50).
- E. native vs matched-recall operating points (GT-derived ⇒ diagnostic-only, separate table).
- S. score-path inconsistency diagnostic (rev2 item 4): Native / cls-consistent / V-consistent,
  matched-candidate-count controlled; diagnostic-only.
- Metrics: all-point interpolated AP primary, official AP3D|R40 secondary; IoU0.7 primary,
  IoU0.5 robustness column. Evaluator: validated exact_ap.ap_summaries (gates PASS).
- NO NMS sweep (none exists natively); NO query-count sweep (meaningless w/o retrain).
- H. decoder-layer-wise analysis: SECONDARY, only with separate justification; never primary.

## GATE-2-DETR survival criteria (pre-declared)
Under all-point AP at the native cell, and under each grid cell of C and D:
  (a) order-ceiling gap > 0 for 5/5;
  (b) median gap ≥ 6 AP;
  (c) native-cell vs best-cell gap difference < 3 AP for ≥ 4/5;
  (d) IoU0.5 gap ≥ half of IoU0.7 gap for ≥ 3/5 (else claim re-scoped IoU0.7-specific).
Wording on PASS: "the ordering headroom is not an artifact of the threshold/top-K operating
point, and exists at the models' native operating points." On any FAIL: report the failing
cell as a finding; downgrade per the main GATE-2 procedure.

## Claim merging rules (common vs separate)
- MAY be merged across all 12 (after both preregs pass): 4-stage waterfall quantities
  (candidate existence / eligibility / budget / final ranking), order-ceiling gap at native
  points under all-point AP, depth-share results.
- MUST stay family-separate: anything mentioning NMS (7 non-DETR only), query×class crowding
  (DETR only), pre-NMS anchor budget (M3D-RPN only), and any operating-point sweep numbers
  (grids differ by family — merge only at the 4-stage abstraction).
- Architecture-spanning wording is allowed ONLY for quantities computed on all 12 under the
  same metric and native-point definition.

## Cost summary (rev2)
- GPU: pre-flatten redump ≈ 1 h total (5 models), ≈ 400 MB storage — UNCONDITIONAL, primary.
- CPU: oracle caches on the redump pools ≈ 2 h; AP evals C(6)+D-flat(4)+D-final(4)+F(4×2)+S(3)
  ≈ 25 ap_summaries calls × 5 models ≈ 2 h. Total ≈ 4–5 h CPU after redump.
- Execution order: non-DETR sweep finishes → redump → reconstruction gate → DETR CPU sweep →
  merge with non-DETR at the 4-stage abstraction (existence → eligibility → budget → ranking).
- Any interim analysis on the existing post-crowding dumps is labeled
  "post-crowding 50-candidate diagnostic", non-GATE-2.

## Amendments
- 2026-06-12 rev2 (before any redump/DETR sweep ran): conditional re-dump →
  unconditional pre-flatten redump; K_flat/K_final split; score-path consistency diagnostic
  added; native-reconstruction unit-test gate added; family-level claim limits added
  ("5/5 studied query-based models" in scope; "independent/diverse DETR architectures" out of scope;
  internal results scoped "within the studied MonoDETR-family pipeline"). Reason: the
  conditional trigger was circular — it depended on Car-hypothesis losses that the
  post-crowding dumps cannot observe.
- 2026-06-12 rev3 (before any redump/DETR sweep ran): WATERFALL ORDER FIX —
  rev2 wrote "threshold → K_flat"; the audited native order is **K_flat (topk over 150 by raw
  cls) → threshold 0.2** (extract_dets_from_outputs applies topk first; decode_detections
  thresholds after). Primary waterfall, all grids, and the reconstruction test now use the
  native order; threshold→K_flat survives only as an explicitly-labeled counterfactual
  secondary. Unit test upgraded to HYPOTHESIS-LEVEL ((query,class) unit, all classes), since
  query-level Car-only flags cannot reproduce cross-class crowding. Also recorded: the M3D-RPN
  NMS-removal collapse (11.5→2.4) is to be reported ONLY as "duplicate suppression is essential
  to native anchor-family AP", NOT as evidence that NMS is a main cause of accurate-candidate
  suppression; #5 computes total-AP effect and accurate-candidate removal/replacement rates
  separately.

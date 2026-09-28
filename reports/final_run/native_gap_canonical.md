# public copy of reports/final_run/native_gap_canonical.md, sha256 02aee345191d92f1eb9839a0ef573c5ad8fa9f371a59d913472a3adb6e423953, scrubbed: Korean notes translated (source lines 31, 33-34), a manuscript to-do dropped (source line 32), manuscript edit plan dropped (source lines 35-39, section 'What changes in the paper'); numbers verbatim
# Canonical native ordering-gap table — 12/12 single harness (2026-06-13)

Source: tools/decomp/e4_fp_tp_decomp.py (prereg e4_prereg.md, definitions frozen pre-run),
reports/e4_fp_tp_decomp.txt. Baseline gate GA: 12/12 reproduce the published native bases
within ±0.05. Construction-consistency gate GB vs previously published cells: 10/12 within
tolerance; MonoFlex Δ0.18 (older battery thresholded MonoFlex on cls, this harness on V —
near-identical pool, base Δ≤0.01, disclosed); MonoCon Δ0.78 vs a UNIFORM-pool reference
(no native cell existed before — this harness provides the first true native cell;
labeled, not a failure of this table).

Metric: all-point interpolated AP (official R40 in parens), Car moderate IoU3D 0.7,
oracle = fixed-pool true-IoU re-sort (o_act via iou_act_and_zstar on the native pool).

| detector | native pool / operating point | base allpt (R40) | gap allpt | provenance |
|---|---|---|---|---|
| M3D-RPN | floor-0 → NMS@0.4 → top-40 → V≥0.75 (native writer) | 11.51 (11.07) | **+11.68** | e4 row 1, GA✓GB✓ |
| MonoDLE | full dump, cls≥0.2, no box-NMS | 15.09 (14.63) | **+14.03** | GA✓GB✓ |
| MonoFlex | full dump, **cls≥0.1** (released det_threshold), no box-NMS | 16.28 (15.52) | **+10.11** | A5 code-faithful re-gate (replaces V-gated +10.46) |
| GUPNet | full dump, V≥0.2, no box-NMS | 17.11 (16.46) | **+12.04** | GA✓GB✓ |
| DEVIANT | full dump, V≥0.2, no box-NMS | 17.48 (16.71) | **+11.83** | GA✓GB✓ |
| MonoGround | full dump, **cls≥0.1** (released det_threshold), no box-NMS | 17.47 (16.74) | **+10.68** | A5 code-faithful re-gate (replaces V-gated +10.89) |
| MonoCon | full dump, V≥0.4 (native thr), no box-NMS | 19.59 (19.01) | **+10.58** | GA✓; first true native cell (old ref was uniform-pool) |
| MonoDETR | pre-flatten in_final (top-50 ∧ cls≥0.2) | 21.18 (20.91) | **+11.42** | GA✓GB✓ |
| MonoDGP | same | 22.82 (22.02) | **+8.41** | GA✓GB✓ |
| MonoCoP | same | 24.34 (23.84) | **+11.58** | GA✓GB✓ |
| MonoCLUE | same | 24.55 (24.03) | **+9.73** | GA✓GB✓ |
| MonoIA | same | 25.18 (24.41) | **+11.64** | GA✓GB✓ |

## Canonical numbers for abstract / Fig.2 / Sec.5
- **Range: +8.4 to +14.0 all-point AP** (min MonoDGP +8.41, max MonoDLE +14.03) — 12/12 > +8. A5 re-gate of MonoFlex/MonoGround (V→released cls≥0.1) lowered only those two interior cells (+10.46→+10.11, +10.89→+10.68); the headline range is unchanged.
- Medians: overall +11.50; CenterNet +11.36, query +11.42 (anchor n=1: +11.68 — no family inference).
- 12/12 ≥ +6 ✓; 12/12 ≥ +8.4 ✓.
- Uniform-pool battery (+10.7..+14.0, gap_exact.txt): the decomposition pool (the efficiency and
  duplicate-purity decomposition is measured on that pool).

## A5 amendment (2026-06-13)
MonoFlex/MonoGround native cells now use the released raw-cls gate (detector_infer.py L103, DETECTIONS_THRESHOLD=0.1), not V. Both interior; range +8.4–14.0 and base-AP ranks unchanged. Source: reports/final_run/a5_regate.txt.

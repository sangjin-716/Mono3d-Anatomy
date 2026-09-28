# public copy of reports/final_run/a4_supremum_lb_SUPPLEMENTARY.md, sha256 4e46d612dd142a91e118487e51b92efa4418e5079926bd5fcdeef2e295e83b44, scrubbed: Korean internal notes translated to English (source lines 3-11); internal usage rules (source lines 35-41) reduced to the summary sentence they quoted; table and numbers verbatim
# A4 — best-of-tested-orderings supremum lower bound (SUPPLEMENTARY SENSITIVITY ONLY)

Scope: supplementary sensitivity only. The main construct is the **true-IoU
re-ranking gain** (native_gap_canonical, +8.4–14.0). This sup-LB is a **supplementary sensitivity**
and not a "perfect ordering ceiling".

## Rationale
The o_act (true-IoU) re-sort is not the supremum over orderings — in E4 the FP-demotion ordering exceeds the true-IoU
re-sort for some models (MonoDGP +11.59 vs +8.41, MonoCLUE +11.45 vs +9.73). Hence
"oracle ceiling = perfect ordering" is inaccurate. The accurate statement: the maximum over the **orderings we tested** (true-IoU
re-sort, FP-demotion) is a lower bound on the ordering supremum.

## sup-LB = max(true-IoU re-rank gain, FP-demotion gain), all-point AP

| model | true-IoU | FP-demotion | sup-LB |
|---|---|---|---|
| M3D-RPN | +11.68 | +11.68 | +11.68 |
| MonoDLE | +14.03 | +14.06 | +14.06 |
| MonoFlex(A5) | +10.11 | +10.65 | +10.65 |
| GUPNet | +12.04 | +12.09 | +12.09 |
| DEVIANT | +11.83 | +11.88 | +11.88 |
| MonoGround(A5) | +10.68 | +10.92 | +10.92 |
| MonoCon | +10.58 | +11.45 | +11.45 |
| MonoDETR | +11.42 | +11.54 | +11.54 |
| MonoDGP | +8.41 | +11.59 | +11.59 |
| MonoCoP | +11.58 | +11.72 | +11.72 |
| MonoCLUE | +9.73 | +11.45 | +11.45 |
| MonoIA | +11.64 | +11.81 | +11.81 |

sup-LB range = **+10.65 to +14.06** (12/12). FP-demotion is the tighter ordering for the two
low-true-IoU query models (DGP, CLUE) because true-IoU re-sort promotes same-GT duplicates in
those no-NMS native query pools (highest pre-selection dup rates 14–16%), depressing their
true-IoU cell — the sup-LB is unaffected by that artifact.

## Summary
- "the supremum over the orderings we tested is at least
  +10.7–14.1 AP per detector; the headline true-IoU re-sort is a conservative member of that
  set, lower for two query models where it promotes pre-selection duplicates."

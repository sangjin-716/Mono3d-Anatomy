# public copy of reports/final_run/c6_c3_decision_rules.md, sha256 7f43638685a54bc4075076058751de51e23e193fa34225ce8412b2c3e5910f0b, scrubbed: internal experiment labels and a reference to an unreleased note replaced by neutral wording (source lines 4, 20, 23, 29); an internal gate reduced to a scope statement (source lines 24-28); rules and numbers otherwise verbatim
# C6 / C3 decision rules — fixed BEFORE results were read (2026-06-13)

Same discipline as the E3 decision rule (analysis amendment fixed before intervals revealed,
NOT a study preregistration). Written before running c6 (permutation) and c3 (multi-pair disagreement).

## C6 — within-family exact permutation test (lineage-cluster robustness)
- Statistic: for each progression channel x, Spearman rho_s(x, base AP) over the 12 detectors.
- Families (exchangeability blocks): anchor {M3D-RPN} n=1, CenterNet {MonoDLE, MonoFlex,
  GUPNet, DEVIANT, MonoGround, MonoCon} n=6, query {MonoDETR, MonoDGP, MonoCoP, MonoCLUE,
  MonoIA} n=5. Exact null = all 1!*6!*5! = 86,400 within-family permutations of x; recompute
  rho_s each; exact two-sided p = fraction with |rho_s*| >= |rho_s observed|.
- **FIXED RULE (before p-values):** a channel trend may be stated as a *trend* in the main
  text only if its exact within-family p <= 0.05. Channels with p > 0.05 are reported as
  range/endpoint only ("rises from A to B"), NEVER as "rises with AP / trend / monotone".
  The per-detector MAGNITUDE spine is unaffected (it is not a cross-detector statistic).
- Channels tested (the monotonicity table): GT recall, recall per bin, |dz| overall + bins,
  |dx|, dim err, yaw err, rho(V,IoU), frac IoU>=0.7. The within-query rho(V,IoU) sub-trend is
  reported regardless of family-level p (it is a within-family observation).

## C3 — multi-pair cross-detector disagreement, MEASUREMENT ONLY
- Run: per detector pair, correlation between depth disagreement |z_a - z_b| on shared GT
  objects and the depth error |z - z_gt|; report N, Pearson, Spearman per pair + family
  breakdown. CLUE input is a superseded matched dump -> EXCLUDE MonoCLUE.
- **SCOPE (fixed before any output):** NO AP re-scoring / rerank arm is run in this paper.
  We measure only "does disagreement predict error across pairs". Any depth-averaging or
  disagreement-gated AP gain is out of scope and is NOT computed here.
- **FIXED RULE:** the ACCV paper's cross-detector disagreement sentence upgrades from "one pair" to "K pairs
  (panel-scoped)" ONLY if >= 60% of valid pairs show rho >= 0.25 (same-sign). Otherwise it
  stays "one detector pair" and the multi-pair result goes to supplementary as a range.

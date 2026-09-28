# public copy of reports/final_run/c6_multiplicity_audit.md, sha256 d193db67589111ea25f3d909e26084d5ac98efa60408c1f10f2f4082a94cfe9e, scrubbed: instruction wording (source lines 3, 41) removed; the wording rules (source lines 52-56) relabeled as a Scope section; statistics and numbers verbatim
# C6 multiplicity audit — lineage-conditioned exact permutation sensitivity (2026-06-13)

NAME: this is a **within-lineage exact permutation test** /
**lineage-conditioned permutation sensitivity** — NOT "cluster-robust" and NOT a significance
guarantee. It is a confound check: does a cross-detector channel trend survive holding the
lineage composition fixed?

## Specification (all fixed BEFORE p-values were read; reports/final_run/c6_c3_decision_rules.md)
1. **Lineage definition:** anchor {M3D-RPN} n=1; CenterNet {MonoDLE, MonoFlex, GUPNet,
   DEVIANT, MonoGround, MonoCon} n=6; MonoDETR-family {MonoDETR, MonoDGP, MonoCoP, MonoCLUE,
   MonoIA} n=5. (Lineages are contiguous blocks in base-AP order — the confound being tested.)
2. **Tested channels (12):** GT recall; recall 0-15/15-30/30-45/45+; |dz| 0-15/15-30/overall;
   |dx|; dim err; frac IoU>=0.7; rho(V,IoU). (yaw excluded — heading-flip artifact already
   folded; not a clean continuous channel.)
3. **Statistic:** Spearman rho_s(channel, base AP) over the 12 detectors.
4. **Exact null:** all 1!*6!*5! = 86,400 within-lineage permutations of the channel; two-sided
   exact p = fraction with |rho_s*| >= |rho_s observed|.
5. **Pre-fixed rule:** trend language allowed only if p <= 0.05; p=0.055 is NOT a trend.
6. **Exchangeability null assumes:** within a lineage, the channel values are exchangeable
   across detectors — i.e., the ONLY systematic structure is the lineage-level mean. It does
   NOT model within-lineage recipe/seed ordering as signal; a real within-lineage trend
   (e.g., newer CenterNet better) would inflate the observed statistic and is what a low p
   detects. It does not certify independence (the 12 are still non-independent).

## Multiplicity table (12 channels)
| channel | raw p | Bonferroni | BH-FDR | survives |
|---|---|---|---|---|
| **\|dz\| 15-30** | 0.0023 | **0.0276** | **0.0276** | raw, **Bonferroni**, **BH** |
| **recall 0-15** | 0.0079 | 0.0948 | **0.0474** | raw, **BH** |
| recall 15-30 | 0.0166 | 0.1992 | 0.0664 | raw only |
| \|dz\| overall | 0.0442 | 0.5304 | 0.1313 | raw only |
| \|dz\| 0-15 | 0.0547 | — | — | none (not even raw) |
| rho(V,IoU) | 0.0791 | — | — | none |
| GT recall | 0.0862 | — | — | none |
| \|dx\| | 0.2903 | — | — | none |
| dim err | 0.4064 | — | — | none |
| frac IoU>=0.7 | 0.4526 | — | — | none |
| recall 30-45 | 0.5584 | — | — | none |
| recall 45+ | 0.6754 | — | — | none |

## Decision
- **Survive multiplicity (BH-FDR):** **|dz| 15-30 m** and **recall 0-15 m** — these two
  near-field channels are associations NOT explained by lineage composition alone. |dz| 15-30
  also survives the stricter Bonferroni.
- **Raw-p only (exploratory sensitivity):** recall 15-30 (p=0.017), |dz| overall (p=0.044).
- **|dz| 0-15 (p=0.055):** NOT a trend (rule); report as range/endpoint only.
- **Primary reporting = effect sizes** (near-field |dz| -41% at 0-15 m, -26% at 15-30 m).
  The permutation p is reported ONLY as a lineage-confound sensitivity, never as the headline
  statistic. The within-query rho(V,IoU) rise (+0.90, n=5) is **supplementary exploratory
  only** (no main-text claim; n=5).

## Scope
- Supported: "Near-field depth error (15-30 m) and near-field recall (0-15 m) co-vary with base
  AP in a way not explained by lineage composition alone (within-lineage exact permutation,
  BH-FDR controlled); the effect sizes are -26% and the recall rise respectively."
- Not supported: "cluster-robust", "the trend is significant across 12 independent detectors",
  treating the permutation p as the primary evidence, or calling raw-p-only channels trends.

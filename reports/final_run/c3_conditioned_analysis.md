# public copy of reports/final_run/c3_conditioned_analysis.md, sha256 fcc54ed7ae4f4b7600e58b3838b496cba068b515ff9bf726a073234f5a57d625, scrubbed: internal labels and instruction wording (source lines 2-3, 5, 43-48, 61) replaced by neutral text, identically in tools/decomp/c3_conditioned.py; the wording rules of the hand-appended VERDICT (source lines 64-70) turned into a Scope note; all numbers verbatim
# c3_conditioned_analysis 2026-06-13T16:23:04
# gate-valid pairs = pairs over 6 detectors (CLUE excluded, superseded dump). Disagreement vs
# error-of-the-pair-average, conditioned on common difficulty. Existing dumps; no AP re-scoring.

## (4) gate-valid accounting: 7 detectors with matched dumps minus CLUE (excluded) = 6 -> 15 pairs (vs 21 with CLUE).
   detectors used: ['DETR', 'DGP', 'CoP', 'IA', 'Flex', 'GUP']

## (1)+(2)+(3) per gate-valid pair: raw / residual(remove gt_z) / partial(gt_z+occ+trunc)
pair              N  raw_sp   resid  partial
DETR+DGP      88648   0.206   0.109    0.108
DETR+CoP      84003   0.197   0.080    0.077
DETR+IA       65659   0.211   0.101    0.098
DETR+Flex     31652   0.208   0.178    0.161
DETR+GUP      34198   0.224   0.161    0.157
DGP+CoP      132728   0.236   0.121    0.121
DGP+IA        94950   0.277   0.141    0.141
DGP+Flex      44780   0.263   0.226    0.202
DGP+GUP       48103   0.256   0.215    0.208
CoP+IA        91532   0.251   0.086    0.086
CoP+Flex      41486   0.245   0.178    0.161
CoP+GUP       44707   0.255   0.181    0.173
IA+Flex       32983   0.268   0.220    0.190
IA+GUP        35602   0.272   0.199    0.187
Flex+GUP      28298   0.176   0.190    0.159

medians over 15 gate-valid pairs: raw=0.245 resid(no gt_z)=0.178 partial(gt_z+occ+trunc)=0.159
partial>0.10 for: 12/15 pairs; min partial=0.077

## (1) within-distance-bin Spearman (does it survive INSIDE a bin? pooled over pairs)
  bin 0-15   n=305430 within-bin Spearman(d,e) = +0.430
  bin 15-30  n=304847 within-bin Spearman(d,e) = +0.138
  bin 30-45  n=216101 within-bin Spearman(d,e) = -0.149
  bin 45+    n= 72951 within-bin Spearman(d,e) = -0.242

## (5) leave-one-detector-out: median raw Spearman over remaining pairs
  drop DETR : 10 pairs, median raw=0.256 median partial=0.167
  drop DGP  : 10 pairs, median raw=0.234 median partial=0.160
  drop CoP  : 10 pairs, median raw=0.240 median partial=0.160
  drop IA   : 10 pairs, median raw=0.230 median partial=0.160
  drop Flex : 10 pairs, median raw=0.243 median partial=0.131
  drop GUP  : 10 pairs, median raw=0.240 median partial=0.131

Scope, supported: 'Across the gate-valid tested detector pairs, cross-detector
depth disagreement is consistently associated with depth error, and the association persists
after conditioning on distance, occlusion, and truncation.' Not supported: 'information exists
between detectors but not in scores' / 'panel-wide proof' / 'independently replicated' /
'recoverable'. Reason: native score also correlates with quality; 15 pairs share detectors
and GTs (not independent replications); 6-detector subset, not the 12-panel.

## VERDICT (2026-06-13)
- Raw pairwise Spearman(disagreement, average-error) median = 0.245 (15 gate-valid pairs).
- After removing gt_z (distance): 0.178. After partial control gt_z+occ+trunc: **0.159**
  (12/15 pairs >0.10, min 0.077) — association SURVIVES difficulty conditioning but is WEAK.
- WITHIN distance bins it is NEAR-FIELD-CONCENTRATED and INVERTS far: 0-15m +0.430,
  15-30m +0.138, 30-45m **-0.149**, 45+ **-0.242**. So most of the raw association is a
  near-field / distance-driven effect; beyond 30 m more disagreement is NOT associated with
  more error (it reverses).
- Leave-one-detector-out: partial median 0.13-0.17 regardless of which detector dropped
  (not driven by one detector).
- Held-out score-vs-disagreement incremental comparison: NOT RUN (would need a fitted model
  + tuning; C3 is therefore restricted to "error-predictive association",
  NOT "information beyond score").

Scope. Supported C3 statement: "Across the gate-valid tested detector pairs, cross-detector depth
disagreement provides a reproducible depth-error-predictive association that persists after
conditioning on distance, occlusion, and truncation (partial Spearman ~0.16, 12/15 pairs),
though it is concentrated in the near field and weakens or reverses beyond 30 m."
Not supported: information-exists-between-not-in-scores / panel-wide proof / independent
replication / recoverable / signal-absent-from-score. The 6-detector subset and shared GTs
rule out independence/panel claims; the far-field reversal rules out an unqualified panel signal.

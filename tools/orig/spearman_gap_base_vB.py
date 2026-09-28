#!/usr/bin/env python
"""Produces reports_orig/spearman_gap_base_vB.txt.

spearman(gap, base) over the 12-detector vB panel on three bases. The per-detector cells are
copied from the reports (no dump is read):
  CEIL* basis (base, AP*=M/n_gt) : reports/exp1_true_ceiling.txt (ten detectors) and
                                   reports_orig/exp1_true_ceiling_orig.txt (MonoFlex*, MonoGround*)
  true-IoU re-sort, native pools : reports/final_run/native_gap_canonical.md (ten detectors) and
                                   reports_orig/a5_regate_orig.txt NATIVE(cls>=0.1) rows (starred)
  uniform S5 pool (all-point)    : reports/gap_exact.txt (ten detectors) and
                                   reports_orig/gap_exact_orig.txt (starred)
Run from the repository root: python tools/orig/spearman_gap_base_vB.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np
from scipy.stats import spearmanr
import datetime

OUT = out_path("spearman_gap_base_vB.txt")
_lines = []


def emit(*a):
    """print() of the original snippet (stdout was redirected into the report), also kept here."""
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


emit("# spearman(gap, base) over the 12-detector vB panel — three bases (provenance for sec5)")
emit("# generated", datetime.datetime.now().isoformat(timespec='seconds'))
emit("# Flex*/Ground* from _orig recomputations; other 10 frozen. n=12.")
emit()
# CEIL* basis (matching ceiling AP* = M/n_gt) — the headline-magnitude basis
exp1 = {"M3D-RPN":(11.51,23.19),"MonoDLE":(15.09,29.12),"GUPNet":(17.11,29.20),
 "DEVIANT":(17.48,29.36),"MonoCon":(19.59,31.04),"MonoDETR":(21.18,32.64),
 "MonoDGP":(22.82,34.40),"MonoCoP":(24.34,36.03),"MonoCLUE":(24.55,35.94),"MonoIA":(25.18,36.98),
 "MonoFlex*":(18.08,29.57),"MonoGround*":(19.38,31.39)}
b=np.array([v[0] for v in exp1.values()]); c=np.array([v[1] for v in exp1.values()])
r,p=spearmanr(b,c-b); emit(f"CEIL* (AP*=M/n_gt) basis : spearman = {r:+.3f}  p = {p:.4f}   [in-band |r|<0.5]")
# trueIoU A5 native basis
nat = {"M3D-RPN":(11.51,11.68),"MonoDLE":(15.09,14.03),"GUPNet":(17.11,12.04),
 "DEVIANT":(17.48,11.83),"MonoCon":(19.59,10.58),"MonoDETR":(21.18,11.42),
 "MonoDGP":(22.82,8.41),"MonoCoP":(24.34,11.58),"MonoCLUE":(24.55,9.73),"MonoIA":(25.18,11.64),
 "MonoFlex*":(18.11,11.07),"MonoGround*":(19.42,11.87)}
b2=np.array([v[0] for v in nat.values()]); g2=np.array([v[1] for v in nat.values()])
r2,p2=spearmanr(b2,g2); emit(f"true-IoU re-sort (native): spearman = {r2:+.3f}  p = {p2:.4f}   [inflated by MonoDGP +8.41 duplicate]")
# uniform S5 basis
ub={"M3D-RPN":11.49,"MonoDLE":15.12,"MonoFlex*":18.10,"GUPNet":17.12,"DEVIANT":17.50,
 "MonoGround*":19.43,"MonoCon":19.60,"MonoDETR":21.19,"MonoDGP":22.90,"MonoCoP":24.35,
 "MonoCLUE":24.68,"MonoIA":25.20}
ug={"M3D-RPN":11.60,"MonoDLE":14.02,"MonoFlex*":11.56,"GUPNet":12.06,"DEVIANT":11.86,
 "MonoGround*":12.13,"MonoCon":11.36,"MonoDETR":11.51,"MonoDGP":11.24,"MonoCoP":11.69,
 "MonoCLUE":11.20,"MonoIA":11.68}
b3=np.array([ub[k] for k in ug]); g3=np.array([ug[k] for k in ug])
r3,p3=spearmanr(b3,g3); emit(f"uniform S5 pool         : spearman = {r3:+.3f}  p = {p3:.4f}")
emit()
emit("READ: on the matching-ceiling basis (the AP* magnitude the paper quotes) the trend is")
emit("in-band and not significant; the more-negative true-IoU value is basis-artifact from")
emit("MonoDGP duplicate promotion. Reported descriptively; no basis quoted as a finding.")
open(OUT, "w").write("\n".join(_lines) + "\n")

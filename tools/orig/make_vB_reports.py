#!/usr/bin/env python
"""Produces <OUT_DIR>/vB_reports/ (the merged view that is in the repository as
figures/data/vB_reports/).

Build the merged 'vB reports view' consumed by the figure generators (figures/): the main
reports with the MonoFlex/MonoGround rows replaced by the ORIG-env rows (names kept PLAIN so the
figure parsers keyed on detector names work unchanged; the star is a table convention).
Inputs: reports/ (main reports) and the tools/orig reports (reports_orig/ when present, else the
re-run in OUT_DIR). The main-panel detector-progression probe report is read as
reports/probe_detector_progression.txt;
an input that is not present is reported and skipped. Rerun-safe (idempotent overwrite).
Run from the repository root: python tools/orig/make_vB_reports.py
"""
import os, re, shutil, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
from tools.orig._orig_common import frozen_report, orig_report

DST = os.path.join(paths.OUT_DIR, "vB_reports")
os.makedirs(f"{DST}/final_run", exist_ok=True)

def sub_rows(frozen_path, orig_path, dst_path, star_to_plain=True):
    """Replace any line in frozen whose first token is MonoFlex/MonoGround (or starts a known
    row pattern) with the corresponding orig line (star names → plain)."""
    for _p in (frozen_path, orig_path):
        if not os.path.exists(_p):
            print(f"  {os.path.basename(dst_path)}: SKIPPED (missing input {_p})")
            return
    orig_rows = {}
    for ln in open(orig_path):
        t = ln.split()
        if t and t[0] in ("MonoFlex*", "MonoGround*"):
            key = t[0].rstrip("*")
            orig_rows.setdefault(key, []).append(ln.replace(t[0], key + " " * 1, 1))
        m = re.match(r"(\[done\] )(MonoFlex\*|MonoGround\*)(.*)", ln)
        if m:
            key = "[done] " + m.group(2).rstrip("*")
            orig_rows.setdefault(key, []).append(m.group(1) + m.group(2).rstrip("*") + " " + m.group(3).lstrip() + "\n")
    counters = {k: 0 for k in orig_rows}
    out = []
    for ln in open(frozen_path):
        t = ln.split()
        key = None
        if t and t[0] in ("MonoFlex", "MonoGround") and not ln.startswith("#"):
            key = t[0]
        m = re.match(r"\[done\] (MonoFlex|MonoGround)\b", ln)
        if m:
            key = "[done] " + m.group(1)
        if key and key in orig_rows and counters[key] < len(orig_rows[key]):
            out.append(orig_rows[key][counters[key]])
            counters[key] += 1
        else:
            out.append(ln)
    open(dst_path, "w").write("".join(out))
    used = {k: c for k, c in counters.items()}
    print(f"  {os.path.basename(dst_path)}: substituted {used}")

# 1. probe_official (fig1a recall + dz rows)
sub_rows(frozen_report("final_run/probe_official_moderate.txt"), orig_report("probe_official_moderate_orig.txt"),
         f"{DST}/final_run/probe_official_moderate.txt")
# 2. detector-progression probe (fig1b/c: main rows + rec/dz sections)
sub_rows(frozen_report("probe_detector_progression.txt"), orig_report("probe_orig.txt"),
         f"{DST}/probe_detector_progression.txt")
# 3. gap_exact (fig2b metric battery)
sub_rows(frozen_report("gap_exact.txt"), orig_report("gap_exact_orig.txt"), f"{DST}/gap_exact.txt")
# 4. oracle_anatomy (fig3)
sub_rows(frozen_report("oracle_anatomy.txt"), orig_report("oracle_anatomy_orig.txt"), f"{DST}/oracle_anatomy.txt")
# 5. pool_waterfall (fig4v2 panel a)
sub_rows(frozen_report("pool_waterfall.txt"), orig_report("pool_waterfall_orig.txt"), f"{DST}/pool_waterfall.txt")
# 6. gt_state_matrix (fig4v2 panel b): copy the full vB recompute as is (when available)
gs = orig_report("gt_state_matrix_vB.txt")
if os.path.exists(gs):
    shutil.copy(gs, f"{DST}/gt_state_matrix.txt")
    print("  gt_state_matrix.txt: copied vB recompute")
else:
    print("  gt_state_matrix.txt: NOT READY (vB run pending)")
# 7. native_gap_canonical.md (fig2a): replace the two A5 cells (values from a5_regate_orig.txt)
src = open(frozen_report("final_run/native_gap_canonical.md")).read()
subs = [("| MonoFlex |", None), ("| MonoGround |", None)]
lines = src.splitlines(keepends=True)
outl = []
for ln in lines:
    if ln.startswith("| MonoFlex |"):
        outl.append("| MonoFlex | A5 released gate (cls≥0.1, orig-env) | 18.11 (17.29) | **+11.07** | GA✓GB✓ |\n")
    elif ln.startswith("| MonoGround |"):
        outl.append("| MonoGround | A5 released gate (cls≥0.1, orig-env) | 19.42 (18.64) | **+11.87** | GA✓GB✓ |\n")
    else:
        outl.append(ln)
open(f"{DST}/final_run/native_gap_canonical.md", "w").write("".join(outl))
print("  native_gap_canonical.md: A5 cells replaced")
print("[done] vB reports view at", DST)

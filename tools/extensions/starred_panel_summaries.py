"""Panel-level summaries of the twelve-detector panel of the paper, in which MonoFlex* and
MonoGround* are the original-environment runs (reports_orig/) and the other ten detectors are the
rows of the main reports (reports/). Produces
reports/extensions/starred_panel_summaries.txt (a re-run writes
reports_rerun/extensions/starred_panel_summaries.txt).

Each per-detector report below prints the modern-environment MonoFlex and MonoGround rows
and summarises over those. The paper uses the starred rows instead, so the panel-level numbers it
prints (medians, ranges, counts, a rank correlation) appear in no single report. This script
recomputes them from the printed per-detector values only; it runs no detector and reads no dump.
  A. far field beyond 45 m (supplementary Sec. H, far-field census and table: k, Wilson 95% CI,
     count above 3.5%, Spearman of recall with base AP)
       reports/final_run/probe_official_moderate.txt + reports_orig/probe_official_moderate_orig.txt
  B. GT BEV-IoU re-sort, recovery of the 3D-IoU re-sort gain (supplementary Sec. D)
       reports/oracle_ladder_o1.txt + reports_orig/oracle_ladder_o1_orig.txt
  C. single-factor geometry snapping (main Sec. 5.2)
       reports/oracle_ladder_o2.txt + reports_orig/oracle_ladder_o2_orig.txt
  D. cross-metric (BEV) transfer of the 3D-IoU re-ordering (supplementary Sec. D, BEV-transfer table)
       reports/final_run/bev_gap_probe.txt + reports_orig/bev_transfer_orig.txt
Gates: the modern-only summaries recomputed here must equal the summary lines printed in the
reports, medians up to the rounding of the printed inputs (and, for A, the modern rows must
equal reports/final_run/a3_farfield_integrity.txt). A final block compares every panel-level value
with the value printed in the paper.
Run from the repository root: python tools/extensions/starred_panel_summaries.py
"""
import os, re, sys, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ext import ROOT, out_path  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default=out_path("extensions/starred_panel_summaries.txt"))
_args = _ap.parse_args()

PANEL = ["M3D-RPN", "MonoDLE", "MonoFlex*", "GUPNet", "DEVIANT", "MonoGround*", "MonoCon",
         "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]
MODERN = [d.rstrip("*") for d in PANEL]
N45 = 365          # official Moderate Car GT beyond 45 m (KITTI val)
Z = 1.96

out = []
def w(s=""):
    print(s, flush=True); out.append(s)

def rd(rel):
    return open(os.path.join(ROOT, rel)).read().split("\n")

def med(v):
    return float(np.median(np.asarray(v, float)))

def section(lines, start):
    """Lines after the first line that starts with `start`, up to the next blank line."""
    i = next(k for k, l in enumerate(lines) if l.startswith(start))
    blk = []
    for l in lines[i + 1:]:
        if not l.strip():
            break
        blk.append(l)
    return blk

gates = []
def gate(name, ok):
    gates.append(ok)
    w(f"  GATE {name}: {'PASS' if ok else 'FAIL'}")

def wilson(k, n, z=Z):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)

w(f"# starred_panel_summaries {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# script  tools/extensions/starred_panel_summaries.py")
w("# panel   " + ", ".join(PANEL))
w("# inputs  printed per-detector values of the frozen reports named in each part; no dump is read")
w("# note    medians here use the printed (rounded) per-detector values, so they can differ from a")
w("#         report's own median, computed before rounding, by up to the input rounding")
w("")

# ------------------------------------------------------------------ A. far field
w("=" * 100)
w("PART A -- recall at IoU 0.7 beyond 45 m on the official Moderate set (n = 365 GT)")
w("=" * 100)
TAB = "OFFICIAL-MODERATE RECALL@IoU0.7 BY BIN"
def probe_rows(rel):
    rows = {}
    for l in section(rd(rel), TAB):
        t = l.replace("|", " ").split()
        rows[t[0]] = dict(base=float(t[1]), r45=float(t[5]))
    return rows
mod = probe_rows("reports/final_run/probe_official_moderate.txt")
org = probe_rows("reports_orig/probe_official_moderate_orig.txt")
a3 = {}
for l in rd("reports/final_run/a3_farfield_integrity.txt"):
    m = re.match(r"(\S+)\s+([0-9.]+)\s+(\d+)\s+\[\s*([0-9.]+)%,\s*([0-9.]+)%\]\s+(yes|no)\s*$", l)
    if m:
        a3[m.group(1)] = (float(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5)), m.group(6))

def far_rows(src):
    res = []
    for d in src:
        r = org[d] if d.endswith("*") else mod[d]
        k = int(round(r["r45"] * N45))
        lo, hi = wilson(k, N45)
        res.append((d, r["base"], r["r45"], k, lo, hi, "yes" if r["r45"] > 0.035 else "no"))
    return res

rows = far_rows(PANEL)
w("base = the base-AP column of the probe tables (AP_R40 on the uniform pool); k = round(rec45 * 365)")
w(f"{'model':12s} {'base':>6s} {'rec45':>6s} {'k':>4s}   {'Wilson 95% CI':>16s}  >3.5%?")
for d, b, r, k, lo, hi, f in rows:
    w(f"{d:12s} {b:6.2f} {r:6.3f} {k:4d}   [{lo:5.1f}%, {hi:5.1f}%]  {f:>5s}")
nabove = sum(1 for x in rows if x[6] == "yes")
rho = spearmanr([x[1] for x in rows], [x[2] for x in rows]).correlation
w(f"models with rec45 point estimate > 3.5%: {nabove}/12 -> {[x[0] for x in rows if x[6] == 'yes']}")
w(f"Spearman(rec45, base) over the 12 models = {rho:+.3f}")
w("")
w("gates (modern-environment rows):")
ok = True
for d, b, r, k, lo, hi, f in far_rows(MODERN):
    ref = a3[d]
    ok &= (abs(ref[0] - r) < 1e-9 and ref[1] == k and abs(ref[2] - round(lo, 1)) < 1e-9
           and abs(ref[3] - round(hi, 1)) < 1e-9 and ref[4] == f)
gate("every modern row (rec45, k, Wilson CI, >3.5%) equals reports/final_run/a3_farfield_integrity.txt", ok)
rho_m = spearmanr([mod[d]["base"] for d in MODERN], [mod[d]["r45"] for d in MODERN]).correlation
printed = next(l for l in rd("reports/final_run/probe_official_moderate.txt") if l.strip().startswith("recall 45+"))
gate(f"modern-only Spearman {rho_m:+.3f} equals the probe report's '{printed.split(':')[1].split()[0]}'",
     f"{rho_m:+.3f}" == printed.split(":")[1].split()[0])
w("")

# ------------------------------------------------------------------ B. O1
w("=" * 100)
w("PART B -- GT BEV-IoU re-sort: recovery of the 3D-IoU re-sort gain (IoU 0.7, all-point AP3D)")
w("=" * 100)
def o1_rows(rel):
    res = {}
    for l in section(rd(rel), "detector   fam       | base.7")[1:]:
        t = l.replace("|", " ").split()
        res[t[0]] = dict(head=float(t[5]), rec=float(t[6].rstrip("%")))
    return res
o1m = o1_rows("reports/oracle_ladder_o1.txt")
o1o = o1_rows("reports_orig/oracle_ladder_o1_orig.txt")
rec = [(o1o[d] if d.endswith("*") else o1m[d])["rec"] for d in PANEL]
w("rec.7 per model: " + "  ".join(f"{d} {v:.1f}%" for d, v in zip(PANEL, rec)))
w(f"recovery over the 12 models: median {med(rec):.1f}%  range {min(rec):.1f}%..{max(rec):.1f}%  "
  f"above 90%: {sum(v > 90 for v in rec)}/12")
recm = [o1m[d]["rec"] for d in MODERN]
line = next(l for l in rd("reports/oracle_ladder_o1.txt") if l.startswith("IoU0.7: headroom range"))
m = re.search(r"recovery range ([0-9.]+)%\.\.([0-9.]+)% median ([0-9.]+)%", line)
gate(f"modern-only recovery range {min(recm):.1f}%..{max(recm):.1f}% equals the printed "
     f"{m.group(1)}%..{m.group(2)}%, and the median {med(recm):.2f}% is within input rounding (0.1) "
     f"of the printed {m.group(3)}%",
     f"{min(recm):.1f}" == m.group(1) and f"{max(recm):.1f}" == m.group(2)
     and abs(med(recm) - float(m.group(3))) <= 0.1 + 1e-9)
w("")

# ------------------------------------------------------------------ C. O2
w("=" * 100)
w("PART C -- single-factor snapping of matched-box geometry to GT (IoU 0.7, all-point AP3D gain over base)")
w("=" * 100)
KEYS = ["purez", "xz", "centre", "y", "h", "wl", "yaw", "BEVshr"]
def o2_rows(rel):
    res = {}
    for l in section(rd(rel), "detector   fam       |   base")[1:]:
        t = l.replace("|", " ").split()
        v = [float(x.rstrip("%")) for x in t[3:10]] + [float(t[12].rstrip("%"))]
        res[t[0]] = dict(zip(KEYS, v))
    return res
o2m = o2_rows("reports/oracle_ladder_o2.txt")
o2o = o2_rows("reports_orig/oracle_ladder_o2_orig.txt")
SHOW = ["xz", "y", "h", "wl", "yaw"]
w(f"{'lever':8s} {'median':>8s} {'min':>8s} {'max':>8s}   (12 models)")
M = {}
for k in KEYS:
    v = [(o2o[d] if d.endswith("*") else o2m[d])[k] for d in PANEL]
    M[k] = med(v)
    if k in SHOW:
        w(f"{k:8s} {M[k]:+8.3f} {min(v):+8.2f} {max(v):+8.2f}")
w("(xz = ground-plane position x,z; y = vertical position; h = height; wl = dimensions w,l; yaw.")
w(" The pure-z and full-centre oracles of main Sec. 5.2 follow a different protocol, reported in")
w(" reports/oracle_anatomy.txt and reports_orig/oracle_anatomy_orig.txt; the purez and centre")
w(" columns of this table are therefore not summarised here.)")
line = next(l for l in rd("reports/oracle_ladder_o2.txt") if l.startswith("IoU0.7: BEV-share median"))
mm = {k: med([o2m[d][k] for d in MODERN]) for k in KEYS}
pr = dict(BEVshr=r"BEV-share median ([0-9.]+)%", xz=r"xz median \+?([-+0-9.]+)",
          purez=r"purez median \+?([-+0-9.]+)", y=r"y median \+?([-+0-9.]+)", h=r"h median \+?([-+0-9.]+)")
tol = dict(BEVshr=0.1, xz=0.01, purez=0.01, y=0.01, h=0.01)
okc = True
for k, rx in pr.items():
    v = float(re.search(rx, line).group(1))
    okc &= abs(mm[k] - v) <= tol[k] + 1e-9
gate("modern-only medians of " + ", ".join(pr) + " (" + ", ".join(f"{mm[k]:+.3f}" for k in pr)
     + ") are within input rounding of the medians printed in reports/oracle_ladder_o2.txt", okc)
w("")

# ------------------------------------------------------------------ D. BEV transfer
w("=" * 100)
w("PART D -- 3D-IoU re-ordering evaluated under 3D and under BEV (IoU 0.7, all-point)")
w("=" * 100)
bm = {}
for l in rd("reports/final_run/bev_gap_probe.txt"):
    m = re.match(r"\[done\] (\S+)\s+3D base=\s*[0-9.]+ ceil=\s*[0-9.]+ gap=\+\s*([0-9.]+) \| "
                 r"BEV base=\s*[0-9.]+ ceil=\s*[0-9.]+ gap=\+\s*([0-9.]+)", l)
    if m:
        bm[m.group(1)] = (float(m.group(2)), float(m.group(3)))
bo = {}
for l in rd("reports_orig/bev_transfer_orig.txt"):
    m = re.match(r"\[done\] (\S+)\s+3D base=\s*[0-9.]+ gap=\+\s*([0-9.]+) \| BEV base=\s*[0-9.]+ gain=\+\s*([0-9.]+)", l)
    if m:
        bo[m.group(1)] = (float(m.group(2)), float(m.group(3)))
g3 = [(bo[d] if d.endswith("*") else bm[d])[0] for d in PANEL]
gb = [(bo[d] if d.endswith("*") else bm[d])[1] for d in PANEL]
w(f"{'model':12s} {'3D gap':>8s} {'BEV gain':>9s} {'BEV-3D':>8s}")
for d, a, b in zip(PANEL, g3, gb):
    w(f"{d:12s} {a:+8.2f} {b:+9.2f} {b - a:+8.2f}")
dd = [round(b - a, 2) for a, b in zip(g3, gb)]
w(f"3D gap   range {min(g3):+.2f}..{max(g3):+.2f} median {med(g3):+.3f}")
w(f"BEV gain range {min(gb):+.2f}..{max(gb):+.2f} median {med(gb):+.3f}")
w(f"paired BEV-3D range [{min(dd):+.2f}, {max(dd):+.2f}];  both positive for "
  f"{sum(1 for a, b in zip(g3, gb) if a > 0 and b > 0)}/12")
g3m = [bm[d][0] for d in MODERN]; gbm = [bm[d][1] for d in MODERN]
L = rd("reports/final_run/bev_gap_probe.txt")
e1 = f"3D  gap range {min(g3m):+.2f}..{max(g3m):+.2f} median {med(g3m):+.2f}"
e2 = f"BEV gap range {min(gbm):+.2f}..{max(gbm):+.2f} median {med(gbm):+.2f}"
gate(f"modern-only '{e1}' and '{e2}' are printed in reports/final_run/bev_gap_probe.txt",
     any(l.startswith(e1) for l in L) and any(l.startswith(e2) for l in L))
w("")

# ------------------------------------------------------------------ paper check
w("=" * 100)
w("CHECK against the values printed in the paper")
w("=" * 100)
R = {x[0]: x for x in rows}
chk = []
def c(where, printed, value, ok):
    chk.append(ok)
    w(f"  {'MATCH' if ok else 'DIFF '}  {where:38s} printed {printed:26s} recomputed {value}")
for d, pr in (("MonoFlex*", (".049", 18, "[3.1,7.7]")), ("MonoGround*", (".047", 17, "[2.9,7.3]"))):
    x = R[d]
    val = f".{int(round(x[2] * 1000)):03d} / {x[3]} / [{x[4]:.1f},{x[5]:.1f}]"
    c(f"supp. Sec. H far-field table, {d}", f"{pr[0]} / {pr[1]} / {pr[2]}", val, val == f"{pr[0]} / {pr[1]} / {pr[2]}")
c("supp. Sec. H, above 3.5%", "Eight of twelve", f"{nabove}/12", nabove == 8)
c("supp. Sec. H, rho_s", "0.52", f"{rho:.2f}", f"{rho:.2f}" == "0.52")
c("supp. Sec. D, BEV-IoU re-sort", "median 95.9%", f"{med(rec):.1f}%", f"{med(rec):.1f}" == "95.9")
c("supp. Sec. D, range", "94.4 to 97.5%", f"{min(rec):.1f} to {max(rec):.1f}%",
  (f"{min(rec):.1f}", f"{max(rec):.1f}") == ("94.4", "97.5"))
c("main Sec. 5.2, x,z snapping", "median +53", f"{M['xz']:+.0f} ({M['xz']:+.3f})", f"{M['xz']:+.0f}" == "+53")
c("main Sec. 5.2, y, h, w/l, yaw", "median at most 2.5",
  ", ".join(f"{k} {M[k]:+.3f}" for k in ("y", "h", "wl", "yaw")),
  all(M[k] <= 2.5 for k in ("y", "h", "wl", "yaw")))
c("supp. Sec. D BEV table, 3D gap", "+8.4 to +14.0, med +11.6",
  f"{min(g3):+.1f} to {max(g3):+.1f}, med {med(g3):+.1f}",
  (f"{min(g3):+.1f}", f"{max(g3):+.1f}", f"{med(g3):+.1f}") == ("+8.4", "+14.0", "+11.6"))
c("supp. Sec. D BEV table, BEV gain", "+7.1 to +14.4, med +11.1",
  f"{min(gb):+.1f} to {max(gb):+.1f}, med {med(gb):+.1f}",
  (f"{min(gb):+.1f}", f"{max(gb):+.1f}", f"{med(gb):+.1f}") == ("+7.1", "+14.4", "+11.1"))
c("supp. Sec. D BEV table, BEV-3D", "[-1.33,+0.60]", f"[{min(dd):+.2f},{max(dd):+.2f}]",
  f"[{min(dd):+.2f},{max(dd):+.2f}]" == "[-1.33,+0.60]")
w("")
w(f"GATES: {'ALL PASS' if all(gates) else 'FAIL'}   PAPER CHECK: {sum(chk)}/{len(chk)} MATCH")
open(_args.out, "w").write("\n".join(out) + "\n")

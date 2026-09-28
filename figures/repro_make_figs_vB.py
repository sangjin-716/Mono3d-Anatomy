"""Ported from final_camera_ready/tools/figs/repro_make_figs_vB.py for the public release.
Computation unchanged. Produces the data (and, run standalone, the submitted-layout PDFs in
reports_rerun/figures/submitted_layout/) of Figures 1-4; the camera-ready layout is make_figs_cr.py.

Paper figures 1-4 -- parse frozen report outputs only (no recomputation).
Inputs (figures/data/vB_reports/ unless noted): the frozen reports with the MonoFlex*/MonoGround*
rows replaced by their original-environment rows (reports_orig/), i.e. the panel of the paper:
  final_run/probe_official_moderate.txt  Fig.1a recall, official-moderate n=7,874
  probe_detector_progression.txt         Fig.1b |dz|, Fig.1c rho (vB view of the frozen
                                         reports/probe_detector_progression.txt)
  gap_exact.txt + reports_orig/gap_exact_orig.txt   Fig.2b metric battery
  (Fig.2a native cells are the literal values of reports/final_run/native_gap_canonical.md)
  oracle_anatomy.txt                     Fig.3
  pool_waterfall.txt                     Fig.4
Run from the repository root: python figures/repro_make_figs_vB.py"""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tools._release import ROOT, out_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(ROOT, "figures", "data", "vB_reports")
OUT = out_path("figures/submitted_layout")
os.makedirs(OUT, exist_ok=True)
DETS = ["M3D-RPN", "MonoDLE", "GUPNet", "DEVIANT", "MonoFlex", "MonoGround",
        "MonoCon", "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]
YEAR = {"M3D-RPN": 2019, "MonoDLE": 2021, "MonoFlex": 2021, "GUPNet": 2021,
        "DEVIANT": 2022, "MonoGround": 2022, "MonoCon": 2022, "MonoDETR": 2023,
        "MonoDGP": 2025, "MonoCoP": 2025, "MonoCLUE": 2026, "MonoIA": 2026}
FAM = dict(zip(DETS, ["anchor"] + ["cnet"] * 6 + ["query"] * 5))
FCOL = {"anchor": "#b07aa1", "cnet": "#4e79a7", "query": "#e15759"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 7, "pdf.fonttype": 42})


def lines(fn):
    return open(os.path.join(R, fn)).read().splitlines()


# ---------------- parse progression probe ----------------
prog, rec_bin, dz_bin = {}, {}, {}
sect = None
for ln in lines("probe_detector_progression.txt"):   # vB view of reports/probe_detector_progression.txt
    if ln.startswith("GT RECALL"):
        sect = "rec"
    elif ln.startswith("DEPTH ERROR"):
        sect = "dz"
    elif ln.startswith("MONOTONICITY"):
        sect = None
    t = ln.split()
    if not t or t[0] not in DETS:
        continue
    if sect is None and len(t) == 14:          # main table row; t[11] = rho(V,IoU) column
        prog[t[0]] = dict(ap=float(t[1]), rho=float(t[11]))
    elif sect == "rec":
        rec_bin[t[0]] = [float(x) for x in t[3:7]]
    elif sect == "dz":
        dz_bin[t[0]] = [float(x) for x in t[3:7]]

# ---------------- parse OFFICIAL-moderate recall-by-bin (Fig.1a, n=7,874) ----------------
# Source: figures/data/vB_reports/final_run/probe_official_moderate.txt (frozen; same GT-set as Sec.4 body).
# Row format: DET base_ap | r(0-15) r(15-30) r(30-45) r(45+) r(ALL)
rec_off = {}
_in = False
for ln in lines("final_run/probe_official_moderate.txt"):
    if ln.startswith("OFFICIAL-MODERATE RECALL@IoU0.7 BY BIN"):
        _in = True
        continue
    if _in and ln.startswith("OFFICIAL-MODERATE |dz|"):
        break
    t = ln.split()
    if _in and len(t) >= 8 and t[0] in DETS and t[2] == "|":
        rec_off[t[0]] = [float(t[3]), float(t[4]), float(t[5]), float(t[6])]

# ---------------- parse gap_exact (UNIFORM/diagnostic battery: used in panel (b)) ----------------
gap = {}
for ln in lines("gap_exact.txt"):
    m = re.match(r"(\S+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+\+([\d.]+)\s+\|\s+\+([\d.]+)\s+\+([\d.]+)", ln)
    if m and m.group(1) in DETS:
        gap[m.group(1)] = dict(base=float(m.group(2)), ceil=float(m.group(3)),
                               g_ap=float(m.group(4)), g_r40=float(m.group(5)), g_r11=float(m.group(6)))
# [CR repro fix] vB_reports/gap_exact.txt carries the orig-env [done] rows for MonoFlex*/MonoGround*
# but its summary table (parsed above) still holds the pre-vB rows (make_vB_reports.py only swaps rows
# present in the orig file). The submitted Fig.2(b) used the orig-env cells of
# reports_orig/gap_exact_orig.txt, with each gap = ceil - base of the printed (2-decimal) [done] row
# (verified: MonoFlex R11 36.36-22.31 = 14.05 reproduces the submitted path; the unrounded
# gap_metric_robustness_orig.txt value is 14.04).
_KO = os.path.join(ROOT, "reports_orig")
for ln in open(f"{_KO}/gap_exact_orig.txt"):
    m = re.match(r"\[done\] (MonoFlex|MonoGround)\*\s.*base R11/R40/allpt=\s*([\d.]+)/\s*([\d.]+)/\s*([\d.]+)"
                 r"\s+ceil=\s*([\d.]+)/\s*([\d.]+)/\s*([\d.]+)\s+gap_allpt=\+([\d.]+)", ln)
    if m:
        b11, b40, bap, c11, c40, cap, gap_ap = (float(m.group(k)) for k in range(2, 9))
        gap[m.group(1)] = dict(base=bap, ceil=cap, g_ap=gap_ap, g_r40=c40 - b40, g_r11=c11 - b11)
print("[CR repro] orig-env cells:", {k: gap[k] for k in ("MonoFlex", "MonoGround")})

# ---------------- NATIVE ordering-gap cells (headline construct; panel (a)) ----------------
# Source: native_gap_canonical.md (vB view of reports/final_run/native_gap_canonical.md with the
# MonoFlex*/MonoGround* cells replaced by the original-environment released-gate cells, cls>=0.1:
# 18.11/+11.07 and 19.42/+11.87, printed by a5_regate_orig.py; single source of truth).
# (base all-point AP, native gap all-point AP) per detector; ceil = base + gap.
_NATIVE = {"M3D-RPN": (11.51, 11.68), "MonoDLE": (15.09, 14.03), "MonoFlex": (18.11, 11.07),
           "GUPNet": (17.11, 12.04), "DEVIANT": (17.48, 11.83), "MonoGround": (19.42, 11.87),
           "MonoCon": (19.59, 10.58), "MonoDETR": (21.18, 11.42), "MonoDGP": (22.82, 8.41),
           "MonoCoP": (24.34, 11.58), "MonoCLUE": (24.55, 9.73), "MonoIA": (25.18, 11.64)}
ngap = {d: dict(base=b, ceil=b + g, g_ap=g) for d, (b, g) in _NATIVE.items()}

# ---------------- parse oracle anatomy ----------------
anat = {}
for ln in lines("oracle_anatomy.txt"):
    m = re.match(r"\[done\] (\S+)\s+base=\s*([\d.]+) \| Gz=\+([\d.]+) ray=\+([\d.]+) centre=\+([\d.]+)", ln)
    if m:
        anat[m.group(1)] = dict(base=float(m.group(2)), z=float(m.group(3)),
                                ray=float(m.group(4)), centre=float(m.group(5)))

# ---------------- parse waterfall ----------------
wf = {}
cur = None
for ln in lines("pool_waterfall.txt"):
    m = re.match(r"\[done\] (\S+)\s+\((\w+)\) GT=\d+ \| A@0.5=([\d.]+) A@0.7=([\d.]+) B@0.7=([\d.]+) C@0.7=([\d.]+) \| suppressed-acc\(face-ii share\)=([\d.]+)", ln)
    if m:
        cur = m.group(1)
        wf[cur] = dict(a5=float(m.group(3)), a7=float(m.group(4)), b7=float(m.group(5)),
                       c7=float(m.group(6)), face2=float(m.group(7)))
    m2 = re.search(r"NO pool candidate IoU>=0.5: \d+ \(([\d.]+)\)", ln)
    if m2 and cur:
        wf[cur]["miss30"] = float(m2.group(1))

assert all(len(d) == 12 for d in (prog, rec_bin, dz_bin, gap, anat, rec_off)), \
    {k: len(v) for k, v in dict(prog=prog, rec=rec_bin, dz=dz_bin, gap=gap, anat=anat,
                                rec_off=rec_off).items()}
assert len(wf) == 12, len(wf)
ap = [prog[d]["ap"] for d in DETS]
x = range(12)


def style_ax(a):
    a.set_xticks(list(x))
    a.set_xticklabels(DETS, rotation=60, ha="right", fontsize=6.5)
    a.grid(axis="y", lw=0.3, alpha=0.5)


# ============ Fig 1 — progression decomposition ============
fig, axs = plt.subplots(1, 3, figsize=(8.3, 2.5))
BL = ["0-15m", "15-30m", "30-45m", "45m+"]
BC = ["#2a9d8f", "#4e79a7", "#e9a039", "#c0392b"]
for i in range(4):
    axs[0].plot(x, [rec_off[d][i] for d in DETS], "o-", ms=2.5, lw=1, c=BC[i], label=BL[i])
    axs[1].plot(x, [dz_bin[d][i] for d in DETS], "o-", ms=2.5, lw=1, c=BC[i], label=BL[i])
axs[0].set_ylabel("GT recall @ IoU$_{3D}$0.7")
axs[0].set_title("(a) coverage: near-field only")
axs[0].legend(ncol=2, frameon=False)
axs[1].set_ylabel(r"mean $|\Delta z|$ on matched TP (m)")
axs[1].set_title("(b) depth precision: near-field only")
axs[2].plot(x, [prog[d]["rho"] for d in DETS], "o-", ms=3, lw=1, c="#555")
axs[2].set_ylim(0.0, 1.0)
axs[2].set_ylabel(r"$\rho$(native score, IoU$_{3D}$)")
axs[2].set_title("(c) score-quality alignment: flat")
ax2 = axs[2].twinx()
ax2.plot(x, ap, "s--", ms=2.5, lw=0.8, c="#aaa")
ax2.set_ylabel("base AP (grey)", color="#888")
ax2.tick_params(axis="y", colors="#888")
for a in axs:
    style_ax(a)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_progression.pdf")
print("fig1 ok")

# ============ Fig 2 — ordering headroom + de-quantization ============
fig, axs = plt.subplots(1, 2, figsize=(8.3, 2.6))
a = axs[0]
a.fill_between(x, [ngap[d]["base"] for d in DETS], [ngap[d]["ceil"] for d in DETS],
               color="#e15759", alpha=0.18)
a.plot(x, [ngap[d]["base"] for d in DETS], "o-", ms=3, lw=1.2, c="#333", label="base (native all-point AP)")
a.plot(x, [ngap[d]["ceil"] for d in DETS], "^-", ms=3, lw=1.2, c="#e15759", label="true-IoU re-sort (lower bound)")
for i, d in enumerate(DETS):
    a.annotate(f"+{ngap[d]['g_ap']:.1f}", (i, ngap[d]["ceil"] + 0.6), fontsize=5.5,
               ha="center", color="#a33")
a.set_ylabel("all-point AP (Car mod. IoU0.7)")
a.set_title("(a) native re-sort gain: $+8.4$ to $+14.0$ AP (12/12)")
a.legend(frameon=False, loc="upper left")
a.set_ylim(8, 41)
b = axs[1]
b.plot(x, [gap[d]["g_r11"] for d in DETS], "s:", ms=3, lw=1, c="#999", label="$AP_{R11}$ gap (1/11 grid)")
b.plot(x, [gap[d]["g_r40"] for d in DETS], "d--", ms=3, lw=1, c="#4e79a7", label="$AP_{R40}$ gap (2.5-AP lattice)")
b.plot(x, [gap[d]["g_ap"] for d in DETS], "o-", ms=3, lw=1.4, c="#e15759", label="all-point gap (no grid)")
b.set_ylabel("ordering gap (AP)")
b.set_title("(b) de-quantized: all-point gap spread 0.72 AP; R11 is its grid")
b.legend(frameon=False)
for a in axs:
    style_ax(a)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_headroom.pdf")
print("fig2 ok")

# ============ Fig 3 — oracle anatomy (3D-centre, along the ray) ============
fig, a = plt.subplots(figsize=(8.3, 2.5))
w = 0.27
a.bar([i - w for i in x], [anat[d]["z"] for d in DETS], w, color="#4e79a7",
      label="pure-$z$ oracle (GT $z$, $x/y$ frozen): median 44% of centre")
a.bar(list(x), [anat[d]["ray"] for d in DETS], w, color="#e9a039",
      label="ray-consistent-centre oracle: 82%")
a.bar([i + w for i in x], [anat[d]["centre"] for d in DETS], w, color="#e15759",
      label="full-centre oracle (GT $x,y,z$): 100%")
a.set_ylabel("all-point AP gain on matched boxes")
a.set_ylim(0, 80)
a.set_title("oracle anatomy: ray-coupled 3D-centre displacement, not the camera-axis depth coordinate alone")
a.legend(frameon=False, ncol=3, loc="upper left", fontsize=6.5)
style_ax(a)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_anatomy.pdf")
print("fig3 ok")

# ============ Fig 4 — waterfall: generation-to-selection continuum ============
fig, axs = plt.subplots(1, 2, figsize=(8.3, 2.6))
a = axs[0]
for i, d in enumerate(DETS):
    c = FCOL[FAM[d]]
    a.plot([i, i], [wf[d]["c7"], wf[d]["a7"]], lw=3, c=c, alpha=0.35, solid_capstyle="butt")
    a.plot(i, wf[d]["a7"], "^", ms=4, c=c)
    a.plot(i, wf[d]["c7"], "o", ms=4, c=c)
a.set_ylabel("share of moderate GTs with accurate cand.")
a.set_title("(a) exists in complete pool ($\\blacktriangle$) vs in output ($\\bullet$)")
hs = [plt.Line2D([], [], color=FCOL[f], lw=3, alpha=0.5, label=l) for f, l in
      [("anchor", "anchor (selection-side)"), ("cnet", "CenterNet (generation-side)"),
       ("query", "query (intermediate)")]]
a.legend(handles=hs, frameon=False, fontsize=6.5)
b = axs[1]
b.bar(list(x), [wf[d]["face2"] for d in DETS], 0.55,
      color=[FCOL[FAM[d]] for d in DETS])
b.set_ylabel("suppressed-accurate share (non-replaced)")
b.set_title("(b) accurate candidates lost to selection (face-ii)")
for i, d in enumerate(DETS):
    b.annotate(f"{wf[d]['miss30']:.2f}", (i, wf[d]["face2"] + 0.012), fontsize=5.5, ha="center", color="#666")
b.text(0.98, 0.84, "grey = 30m+ misses with\nno IoU$\\geq$0.5 pool cand.", fontsize=6,
       ha="right", transform=b.transAxes, color="#666")
for a in axs:
    style_ax(a)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_waterfall.pdf")
print("fig4 ok")
print("[all figures written to", OUT + "]")

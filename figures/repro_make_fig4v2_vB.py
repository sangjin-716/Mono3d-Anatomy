"""Data (and, run standalone, the submitted-layout PDF in reports_rerun/figures/submitted_layout/)
of the supplementary generation-selection figure. The camera-ready layout is make_figs_cr.py
(fig4v2_plane_agreement.pdf).

(a) generation-selection plane (pool_waterfall aggregates), (b) cross-detector agreement +
panel-unreached set (gt_state_matrix.txt). y-axis = RETENTION RATE (presence layer; no TP matching).
Inputs: figures/data/vB_reports/{pool_waterfall.txt, gt_state_matrix.txt}.
Run from the repository root: python figures/repro_make_fig4v2_vB.py"""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tools._release import ROOT, out_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = out_path("figures/submitted_layout")
os.makedirs(OUT, exist_ok=True)
R = os.path.join(ROOT, "figures", "data", "vB_reports")
DETS = ["M3D-RPN", "MonoDLE", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround",
        "MonoCon", "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]
FAM = dict(zip(DETS, ["anchor"] + ["cnet"] * 6 + ["query"] * 5))
FCOL = {"anchor": "#b07aa1", "cnet": "#4e79a7", "query": "#e15759"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7,
                     "pdf.fonttype": 42})

wf = {}
for ln in open(os.path.join(R, "pool_waterfall.txt")):
    m = re.match(r"\[done\] (\S+)\s+\(\w+\) GT=\d+ \| A@0.5=[\d.]+ A@0.7=([\d.]+) "
                 r"B@0.7=[\d.]+ C@0.7=([\d.]+)", ln)
    if m:
        wf[m.group(1)] = (float(m.group(2)), float(m.group(3)))
agree, hard = None, {}
for ln in open(os.path.join(R, "gt_state_matrix.txt")):
    if ln.startswith("  0:") and agree is None and "11:" in ln:
        agree = [int(x.split(":")[1]) for x in ln.split()]
    m = re.match(r"PANEL HARD CORE .*: (\d+) / 7874 = ([\d.]+)", ln)
    if m and not hard:
        hard = dict(n=int(m.group(1)), share=float(m.group(2)))
assert len(wf) == 12 and agree and sum(agree) == 7874, (len(wf), agree)

fig, axs = plt.subplots(1, 2, figsize=(8.3, 2.7))
a = axs[0]
for d in DETS:
    A, C = wf[d]
    a.scatter(A, C / A, s=22, c=FCOL[FAM[d]], zorder=3)
qx = [wf[d][0] for d in DETS if FAM[d] == "query"]
qy = [wf[d][1] / wf[d][0] for d in DETS if FAM[d] == "query"]
a.plot(qx, qy, "-", lw=0.8, c="#e15759", alpha=0.5, zorder=2)   # same-family trajectory only
for d, dx, dy in [("M3D-RPN", 0.01, 0.02), ("MonoDETR", -0.012, -0.06), ("MonoCoP", 0.005, -0.06),
                  ("MonoIA", 0.005, 0.03), ("MonoDLE", -0.01, 0.015)]:
    A, C = wf[d]
    a.annotate(d, (A + dx, C / A + dy), fontsize=6, color="#444")
a.set_xlabel("candidate existence  A@0.7 (complete native pool)")
a.set_ylabel("retention rate  C@0.7 / A@0.7")
a.set_title("(a) generation vs selection: family corners")
a.set_xlim(0.2, 0.8); a.set_ylim(0.25, 1.05)
hs = [plt.Line2D([], [], marker="o", ls="", color=FCOL[f], label=l) for f, l in
      [("cnet", "CenterNet: keeps ~all, generates least"),
       ("query", "query: generates more, keeps 79–91%"),
       ("anchor", "anchor: generates most, keeps 32%")]]
a.legend(handles=hs, frameon=False, loc="lower left")
a.grid(lw=0.3, alpha=0.5)

b = axs[1]
cols = ["#999999"] + ["#4e79a7"] * 11 + ["#2a9d8f"]
bars = b.bar(range(13), agree, color=cols)
b.set_xticks(range(13))
b.set_xlabel("# of 12 detectors with an accurate candidate in their complete pool (per GT)")
b.set_ylabel("moderate Car GTs")
b.set_title("(b) cross-detector agreement at IoU$_{3D}$0.7")
b.annotate(f"panel-unreached set\n{hard['n']} GTs = {hard['share']*100:.1f}%\n(no candidate in ANY pool)",
           (0, agree[0]), xytext=(1.3, agree[0] * 0.92), fontsize=6.5, color="#555",
           arrowprops=dict(arrowstyle="->", lw=0.6, color="#777"))
b.grid(axis="y", lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4v2_plane_agreement.pdf")
print("fig4v2 ok; hard core", hard)

"""Ported from final_camera_ready/tools/figs/data_identity.py for the public release.
Computation unchanged. Produces a console report: for each of the five figures, whether the
camera-ready figure (make_figs_cr.py) plots exactly the same data arrays and axes texts as the
submitted-layout figure (repro_make_figs_vB.py / repro_make_fig4v2_vB.py).
Run from the repository root: python figures/data_identity.py
Two differences are expected, both data corrections of the camera-ready figures (make_figs_cr.py
docstring): the grey series of fig1_progression (native base AP of Table 1 instead of the
uniform-pool AP_R40 base) and one grey label of fig4_waterfall (MonoFlex* 0.79 -> 0.78).

Compare the plotted data arrays (not pixels) of the submitted-repro figures vs the CR figures."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
def sig(fig):
    out = []
    for ax in fig.axes:
        for l in ax.lines:
            xd, yd = np.asarray(l.get_xdata(), float), np.asarray(l.get_ydata(), float)
            out.append(("line", tuple(np.round(xd, 6)), tuple(np.round(yd, 6))))
        for p in ax.patches:
            if isinstance(p, Rectangle):
                out.append(("bar", round(p.get_x() + p.get_width() / 2, 6), round(p.get_height(), 6)))
        for c in ax.collections:
            if hasattr(c, "get_offsets") and len(c.get_offsets()) and type(c).__name__ == "PathCollection":
                out.append(("scatter", tuple(map(tuple, np.round(c.get_offsets(), 6)))))
            elif type(c).__name__ in ("PolyCollection", "FillBetweenPolyCollection"):
                v = np.round(c.get_paths()[0].vertices, 6)
                out.append(("fill", tuple(map(tuple, v))))
        for t in ax.texts:
            out.append(("text", t.get_text()))
    return sorted(out, key=repr)
# ---- repro (submitted) figures
FIG = {}
src = open(os.path.join(HERE, "repro_make_figs_vB.py")).read()
for n in ["fig1_progression", "fig2_headroom", "fig3_anatomy", "fig4_waterfall"]:
    src = src.replace(f'fig.savefig(f"{{OUT}}/{n}.pdf")', f'FIGS["{n}"]=fig')
g = {"__name__": "r", "FIGS": FIG, "__file__": os.path.join(HERE, "repro_make_figs_vB.py")}; exec(src, g)
src = open(os.path.join(HERE, "repro_make_fig4v2_vB.py")).read().replace('fig.savefig(f"{OUT}/fig4v2_plane_agreement.pdf")', 'FIGS["fig4v2_plane_agreement"]=fig')
g = {"__name__": "r", "FIGS": FIG, "__file__": os.path.join(HERE, "repro_make_fig4v2_vB.py")}; exec(src, g)
orig = {k: sig(v) for k, v in FIG.items()}
# ---- CR figures
import make_figs_cr as M
CR = {}
def _finish(fig, name, ignore=()):
    CR[name] = sig(fig)
M.finish = _finish
for f in ["fig1", "fig2", "fig3", "fig4", "fig4v2"]:
    getattr(M, f)()
for n in orig:
    o, c = orig[n], CR[n]
    # annotation texts in axes are compared as a multiset; line/bar/scatter/fill as data
    od = [s for s in o if s[0] != "text"]; cd = [s for s in c if s[0] != "text"]
    ot = sorted(s[1] for s in o if s[0] == "text"); ct = sorted(s[1] for s in c if s[0] == "text")
    print(f"{n}: data items {len(od)} vs {len(cd)} identical={od == cd}; axes-text identical={ot == ct}")
    if od != cd:
        so, sc = set(od), set(cd)
        print("   only orig:", [s[:2] for s in so - sc][:5]); print("   only CR:", [s[:2] for s in sc - so][:5])
    if ot != ct:
        print("   texts orig:", ot); print("   texts CR:  ", ct)
print("Expected differences (camera-ready data corrections): fig1_progression grey series, "
      "fig4_waterfall MonoFlex* grey label 0.79 -> 0.78.")

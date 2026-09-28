"""Shared helpers for make_figs_cr.py, which builds the camera-ready figures
(reports_rerun/figures/*.pdf). Writes nothing itself.

Data: the data-parsing part of repro_make_figs_vB.py (everything before the '# ============ Fig 1'
marker) and of repro_make_fig4v2_vB.py (everything before the first plt.subplots) is executed as
it is, so the camera-ready figures use exactly the parsed values that reproduced the submitted
PDFs bit-for-bit.
Layout: 4.80 in wide (= \\linewidth, placed at scale 1.0). Fonts: DejaVu Sans Condensed
(TrueType, pdf.fonttype 42). Every glyph >= 6.5 pt including mathtext subscripts.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib._mathtext as _mt
from matplotlib.text import Text, Annotation
from matplotlib.legend import Legend

HERE = os.path.dirname(os.path.abspath(__file__))
W_IN = 4.80

# subscripts: base >= 7.5 pt with shrink 0.87 -> >= 6.53 pt
SUB_SHRINK = 0.87
_mt.SHRINK_FACTOR = SUB_SHRINK

FS_TICK = 6.5
FS_LEG = 6.5
FS_ANN = 6.5
FS_LAB = 7.5
FS_TITLE = 7.5


def load_main():
    src = open(os.path.join(HERE, "repro_make_figs_vB.py")).read()
    cut = src.index("# ============ Fig 1")
    g = {"__name__": "cr_data", "__file__": os.path.join(HERE, "repro_make_figs_vB.py")}
    exec(compile(src[:cut], "repro_make_figs_vB.py[prefix]", "exec"), g)
    return g


def load_fig4v2():
    src = open(os.path.join(HERE, "repro_make_fig4v2_vB.py")).read()
    cut = src.index("fig, axs = plt.subplots")
    g = {"__name__": "cr_data4v2", "__file__": os.path.join(HERE, "repro_make_fig4v2_vB.py")}
    exec(compile(src[:cut], "repro_make_fig4v2_vB.py[prefix]", "exec"), g)
    return g


def set_rc():
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.stretch": "condensed",
        "font.size": FS_TICK, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LAB,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEG,
        "axes.titlepad": 3.0, "axes.labelpad": 2.0,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.3, "legend.labelspacing": 0.25,
        "legend.columnspacing": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


# ------------------------------------------------------------------ overlap checker
def _leaves(fig):
    out = []
    for a in fig.findobj():
        if a is fig:
            continue
        if isinstance(a, (Text,)) or not a.get_children():
            out.append(a)
    return out


def _mask(fig, keep, dpi):
    """alpha mask of the figure with only artists in `keep` inked. Others get alpha 0 but stay
    visible, so layout (axis-label offsets, legend sizes) is identical to the real render."""
    leaves = _leaves(fig)
    keep_ids = {id(a) for a in keep}
    saved = [(a, a.get_alpha()) for a in leaves]
    extra = []
    for a in keep:
        if isinstance(a, Annotation) and a.arrow_patch is not None and getattr(a, "_cr_text_only", False):
            extra.append((a.arrow_patch, a.arrow_patch.get_alpha()))
            a.arrow_patch.set_alpha(0.0)
    for a, al in saved:
        if id(a) not in keep_ids:
            a.set_alpha(0.0)
            if isinstance(a, Annotation) and a.arrow_patch is not None:
                extra.append((a.arrow_patch, a.arrow_patch.get_alpha()))
                a.arrow_patch.set_alpha(0.0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, 3].copy()
    for a, al in saved:
        a.set_alpha(al)
    for p, al in extra:
        p.set_alpha(al)
    return buf > 90  # ignore faint anti-alias halo


def _dilate(m, r):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def check_overlaps(fig, dpi=300, tol=0, ignore_data=(), clear_pt=0.0):
    """clear_pt > 0: also require that much clearance (text masks dilated by clear_pt)."""
    """Pixel-level ink overlap test. Returns list of (kind, a, b, npix)."""
    fig.set_dpi(dpi)
    fp, ap = fig.patch.get_alpha(), [(ax, ax.patch.get_alpha()) for ax in fig.axes]
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
             and t.get_figure() is not None]
    # drop texts that render nothing (e.g. offset texts, outside canvas)
    tmasks = []
    for t in texts:
        if isinstance(t, Annotation):
            t._cr_text_only = True
        m = _mask(fig, [t], dpi)
        if m.sum() > 0:
            tmasks.append((t, m))
    data = []
    for ax in fig.axes:
        data += [l for l in ax.lines if l.get_visible()]
        data += [p for p in ax.patches if p.get_visible()]
        data += [c for c in ax.collections if c.get_visible()]
    data = [d for d in data if not any(d is z for z in ignore_data)]
    spines = [sp for ax in fig.axes for sp in ax.spines.values() if sp.get_visible()]
    smask = _mask(fig, spines, dpi) if spines else None
    dmask = _mask(fig, data, dpi) if data else None
    # legend handles as obstacles for data and texts (handles vs data)
    legs = [l for l in fig.findobj(Legend)]
    hmasks = []
    for lg in legs:
        hs = [h for h in lg.findobj() if h is not lg and not isinstance(h, Text) and not h.get_children()]
        hs = [h for h in hs if h.get_visible()]
        if hs:
            hmasks.append((lg, _mask(fig, hs, dpi)))
    if clear_pt > 0:
        r = max(1, int(round(clear_pt / 72 * dpi)))
        tmasks = [(t, _dilate(m, r)) for t, m in tmasks]
    res = []
    for i in range(len(tmasks)):
        for j in range(i + 1, len(tmasks)):
            n = int((tmasks[i][1] & tmasks[j][1]).sum())
            if n > tol:
                res.append(("text/text", tmasks[i][0].get_text(), tmasks[j][0].get_text(), n))
        if dmask is not None:
            n = int((tmasks[i][1] & dmask).sum())
            if n > tol:
                res.append(("text/data", tmasks[i][0].get_text(), "DATA", n))
        if smask is not None:
            n = int((tmasks[i][1] & smask).sum())
            if n > tol:
                res.append(("text/spine", tmasks[i][0].get_text(), "SPINE", n))
    for lg, hm in hmasks:
        if dmask is not None:
            n = int((hm & dmask).sum())
            if n > tol:
                res.append(("legendhandle/data", "legend", "DATA", n))
        for t, m in tmasks:
            n = int((hm & m).sum())
            if n > tol:
                res.append(("legendhandle/text", "legend", t.get_text(), n))
    fig.patch.set_alpha(fp)
    for ax, a in ap:
        ax.patch.set_alpha(a)
    # text outside canvas
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    for t, _ in tmasks:
        bb = t.get_window_extent(r)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            res.append(("clipped", t.get_text(), f"bbox={bb.bounds}", 0))
    fig.set_dpi(72)
    return res

"""Camera-ready versions of the five paper figures (ACCV 2026 #1015).

Produces reports_rerun/figures/{fig1_progression, fig2_headroom, fig3_anatomy, fig4_waterfall,
fig4v2_plane_agreement}.pdf (paths.OUT_DIR/figures).
Data are parsed from figures/data/vB_reports/ by the data-parsing part of the repro scripts (see cr_common.py),
with the same series, colours, markers and text strings as the submitted PDFs, and a new
size/layout: 4.80 in wide (placed at scale 1.0 at \\linewidth), all glyphs >= 6.5 pt, no ink
overlaps. Two data corrections of the camera-ready version differ from the submitted figures:
  Fig. 1(c) grey series = native base all-point AP of Table 1 (the _NATIVE cells of
            repro_make_figs_vB.py), not the uniform-pool AP_R40 base;
  supplementary waterfall figure, panel (b) grey labels: the 30 m+ share of MonoFlex* and
            MonoGround* is read from reports_orig/pool_waterfall_orig.txt (MonoFlex* 0.78),
            because figures/data/vB_reports/pool_waterfall.txt swaps only their [done] rows.
Run from the repository root: python figures/make_figs_cr.py [fig1 fig2 fig3 fig4 fig4v2]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tools._release import ROOT, out_path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cr_common as C

OUT = out_path("figures")
os.makedirs(OUT, exist_ok=True)
D = C.load_main()
G4 = C.load_fig4v2()          # both data prefixes run BEFORE set_rc (they set the old rcParams)
DETS, x = D["DETS"], list(D["x"])
C.set_rc()


def style_ax(a, rot=90):
    a.set_xticks(x)
    if rot == 90:
        a.set_xticklabels(DETS, rotation=90, ha="center", va="top", fontsize=C.FS_TICK)
    else:
        a.set_xticklabels(DETS, rotation=rot, ha="right", va="top", rotation_mode="anchor",
                          fontsize=C.FS_TICK)
    a.grid(axis="y", lw=0.3, alpha=0.5)
    a.set_xlim(-0.6, 11.6)


def finish(fig, name, ignore=()):
    path = f"{OUT}/{name}.pdf"
    fig.savefig(path)
    probs = C.check_overlaps(fig, ignore_data=ignore)
    w, h = fig.get_size_inches()
    print(f"[{name}] {w:.2f} x {h:.3f} in = {w*72:.1f} x {h*72:.1f} pt ; overlaps: {len(probs)}")
    for p in probs:
        print("   ", p)
    plt.close(fig)


# ============ Fig 1 — progression decomposition ============
def fig1(H=2.20, edge=2.0, clear=(8.0, 8.0), wts=(1.0, 1.0, 1.0), titlepad=5.0,
         xpad=0.4):
    """Relaxed layout: taller canvas, single-line y labels, panel frames placed in points from
    measured label extents (uniform clearance between a panel's frame and its neighbour's labels)."""
    rec_off, dz_bin, prog, ap = D["rec_off"], D["dz_bin"], D["prog"], D["ap"]
    fig = plt.figure(figsize=(C.W_IN, H))
    axs = [fig.add_axes([0.1 + 0.3 * i, 0.3, 0.2, 0.5]) for i in range(3)]
    BL = ["0-15m", "15-30m", "30-45m", "45m+"]
    BC = ["#2a9d8f", "#4e79a7", "#e9a039", "#c0392b"]
    for i in range(4):
        axs[0].plot(x, [rec_off[d][i] for d in DETS], "o-", ms=2.2, lw=0.9, c=BC[i], label=BL[i])
        axs[1].plot(x, [dz_bin[d][i] for d in DETS], "o-", ms=2.2, lw=0.9, c=BC[i], label=BL[i])
    axs[0].set_ylabel("GT recall @ IoU$_{3D}$ 0.7")
    axs[0].set_title("(a) coverage,\nmainly near field", pad=titlepad)
    # distance key sits in the empty top band of (a) (recall axis spans 0-1, data stay <= 0.71)
    axs[0].set_ylim(-0.03, 1.10)
    axs[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0].legend(ncol=2, frameon=False, loc="upper left", handlelength=1.1,
                  columnspacing=0.7, handletextpad=0.35, borderaxespad=0.4, borderpad=0.1,
                  labelspacing=0.35)
    axs[1].set_ylabel("mean $|\\Delta z|$ on matched boxes (m)")
    axs[1].set_title("(b) depth error,\nmainly near field", pad=titlepad)
    axs[1].set_ylim(0.12, 1.90)
    axs[1].set_yticks([0.5, 1.0, 1.5])
    axs[2].plot(x, [prog[d]["rho"] for d in DETS], "o-", ms=2.6, lw=0.9, c="#555")
    axs[2].set_ylim(0.0, 1.0)
    axs[2].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[2].set_ylabel("$\\rho$(native score, IoU$_{3D}$)")
    axs[2].set_title("(c) score\u2013quality\nrank correlation", pad=titlepad)
    ax2 = axs[2].twinx()
    # grey = base AP of Table 1 (native all-point), not the uniform-pool AP_R40 used before
    ax2.plot(x, [D["ngap"][d]["base"] for d in DETS], "s--", ms=2.2, lw=0.7, c="#aaa")
    ax2.set_ylabel("base AP (grey)", color="#888")
    ax2.set_ylim(9.5, 26.5)
    ax2.set_yticks([15, 20, 25])        # same ticks as before (no new tick numbers)
    ax2.tick_params(axis="y", colors="#888", labelsize=C.FS_TICK, pad=1.5, length=2, width=0.5)
    for s_ in ax2.spines.values():
        s_.set_linewidth(0.6)
    for a in axs:
        style_ax(a)
        a.set_xlim(-xpad, 11 + xpad)
    ax2.set_xlim(-xpad, 11 + xpad)
    # ---- place the three frames in points from the measured label extents (3 passes)
    Wpt, Hpt = C.W_IN * 72, H * 72
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        s = 72.0 / fig.dpi
        ext = []
        for i, a in enumerate(axs):
            fr = a.get_window_extent(r)
            tb = a.get_tightbbox(r)
            if i == 2:
                tb2 = ax2.get_tightbbox(r)
                x1 = max(tb.x1, tb2.x1)
            else:
                x1 = tb.x1
            ext.append(((fr.x0 - tb.x0) * s, (x1 - fr.x1) * s, (tb.y1 - fr.y1) * s,
                        (fr.y0 - tb.y0) * s))
        L = edge + ext[0][0]
        g12 = max(ext[0][1], 0) + clear[0] + ext[1][0]
        g23 = max(ext[1][1], 0) + clear[1] + ext[2][0]
        R = ext[2][1] + edge
        avail = Wpt - L - g12 - g23 - R
        ws = [avail * w / sum(wts) for w in wts]
        top = edge + max(e[2] for e in ext)
        bot = edge + max(e[3] for e in ext)
        hh = Hpt - top - bot
        x0 = L
        for i, a in enumerate(axs):
            pos = [x0 / Wpt, bot / Hpt, ws[i] / Wpt, hh / Hpt]
            a.set_position(pos)
            if i == 2:
                ax2.set_position(pos)
            x0 += ws[i] + (g12 if i == 0 else g23)
    print(f"   fig1 frames (pt): L={L:.1f} g12={g12:.1f} g23={g23:.1f} R={R:.1f} "
          f"w={[round(w_, 1) for w_ in ws]} h={hh:.1f} top={top:.1f} bot={bot:.1f} "
          f"tick pitch={ws[0] / (11 + 2 * xpad):.2f}/{ws[2] / (11 + 2 * xpad):.2f} pt")
    finish(fig, "fig1_progression")


# ============ Fig 2 — ordering headroom + de-quantization ============
def fig2(H=1.70, bot=35.0, topm=22.0):
    ngap, gap = D["ngap"], D["gap"]
    Hpt = H * 72
    fig = plt.figure(figsize=(C.W_IN, H))
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.992, bottom=bot / Hpt, top=1 - topm / Hpt,
                          wspace=0.20)
    axs = [fig.add_subplot(gs[0, i]) for i in range(2)]
    a = axs[0]
    band = a.fill_between(x, [ngap[d]["base"] for d in DETS], [ngap[d]["ceil"] for d in DETS],
                          color="#e15759", alpha=0.18, lw=0)
    a.plot(x, [ngap[d]["base"] for d in DETS], "o-", ms=2.6, lw=1.1, c="#333",
           label="base (native all-point AP)")
    a.plot(x, [ngap[d]["ceil"] for d in DETS], "^-", ms=2.6, lw=1.1, c="#e15759",
           label="true-IoU re-sort (conservative)")
    # same "+x.x" strings as submitted; alternate below (even) / above (odd) the re-sort line so
    # neighbouring labels never share a row; offsets follow the local line height (no ink overlap)
    ceil = [ngap[d]["ceil"] for d in DETS]
    import numpy as _np
    def _line_y(u):
        return _np.interp(u, x, ceil)
    for i, d in enumerate(DETS):
        span = _np.linspace(i - 0.8, i + 0.8, 33)   # label half-width ~0.77 units
        if i % 2 == 0:
            # first label starts at its point (ha=left) so it stays clear of the y axis
            ha, span = ("left", _np.linspace(i, i + 1.6, 33)) if i == 0 else ("center", span)
            ytop = min(_line_y(span)) - 1.7
            a.annotate(f"+{ngap[d]['g_ap']:.1f}", (i + (0.05 if i == 0 else 0), ytop),
                       fontsize=C.FS_ANN, ha=ha, va="top", color="#a33")
        else:
            ybot = max(_line_y(span)) + 1.1
            a.annotate(f"+{ngap[d]['g_ap']:.1f}", (i, ybot), fontsize=C.FS_ANN, ha="center",
                       va="bottom", color="#a33")
    a.set_ylabel("all-point AP\n(Car mod. IoU 0.7)")
    a.set_title("(a) native re-sort gain:\n$+8.4$ to $+14.0$ AP (12/12)")
    a.legend(frameon=False, loc="lower right", borderaxespad=0.2, borderpad=0.1)
    a.set_ylim(6.0, 43.5)
    a.set_yticks([10, 20, 30, 40])
    b = axs[1]
    b.plot(x, [gap[d]["g_r11"] for d in DETS], "s:", ms=2.6, lw=0.9, c="#999",
           label="$AP_{R11}$ gap (1/11 grid)")
    b.plot(x, [gap[d]["g_r40"] for d in DETS], "d--", ms=2.6, lw=0.9, c="#4e79a7",
           label="$AP_{R40}$ gap (2.5-AP lattice)")
    b.plot(x, [gap[d]["g_ap"] for d in DETS], "o-", ms=2.6, lw=1.3, c="#e15759",
           label="all-point gap (no grid)")
    b.set_ylabel("ordering gap (AP)")
    b.set_title("(b) de-quantized all-point gap,\nSD 0.72 AP")
    b.legend(frameon=False, loc="upper right", fontsize=C.FS_LAB, borderaxespad=0.2,
             borderpad=0.1)
    b.set_ylim(8.3, 23.6)   # headroom above the MonoDLE peak so the legend clears every line
    b.set_yticks([10, 12, 14, 16, 18, 20])
    for a_ in axs:
        style_ax(a_, rot=45)
    a.set_xlim(-0.6, 12.05)   # room for the last "+11.6" label inside the frame
    finish(fig, "fig2_headroom", ignore=(band,))


# ============ Fig 3 — oracle anatomy (3D-centre, along the ray) ============
def fig3(H=1.585, bot=28.5, topm=21.5):
    anat = D["anat"]
    Hpt = H * 72
    fig = plt.figure(figsize=(C.W_IN, H))
    a = fig.add_axes([0.085, bot / Hpt, 0.992 - 0.085, 1 - (bot + topm) / Hpt])
    w = 0.27
    b1 = a.bar([i - w for i in x], [anat[d]["z"] for d in DETS], w, color="#4e79a7",
               label="pure-$z$ oracle (GT $z$, $x/y$ frozen): median 44% of centre")
    b2 = a.bar(list(x), [anat[d]["ray"] for d in DETS], w, color="#e9a039",
               label="ray-consistent-centre oracle: 82%")
    b3 = a.bar([i + w for i in x], [anat[d]["centre"] for d in DETS], w, color="#e15759",
               label="full-centre oracle (GT $x,y,z$): 100%")
    a.set_ylabel("all-point AP gain\non matched boxes")
    a.set_ylim(0, 97)
    a.set_yticks([0, 20, 40, 60, 80])
    a.set_title("oracle anatomy: ray-coupled 3D-centre displacement,\n"
                "not the camera-axis depth coordinate alone")
    # 2 rows (row 1: pure-z | ray-consistent, row 2: full-centre); was one row of 3 at 6.5 pt
    a.legend(handles=[b1, b3, b2], frameon=False, ncol=2, loc="upper left",
             borderaxespad=0.25, borderpad=0.1, handlelength=1.2, columnspacing=1.2)
    style_ax(a, rot=30)
    a.set_xlim(-0.6, 11.6)
    finish(fig, "fig3_anatomy")


# ============ Fig 4 — waterfall: generation-to-selection continuum (supplementary) ============
def fig4(H=1.72, bot=35.0, topm=22.0):
    wf, FAM, FCOL = D["wf"], D["FAM"], D["FCOL"]
    # the vB waterfall file swaps only the [done] rows of MonoFlex*/MonoGround* to the original
    # environment, so their 30m+ line is still the modern run (MonoFlex 0.79). Take miss30 from the
    # frozen original-environment report (MonoFlex* 1956/2512 = 0.78, MonoGround* 1877/2483 = 0.76).
    import re
    wf = {d: dict(v) for d, v in wf.items()}
    cur = None
    for ln in open(os.path.join(ROOT, "reports_orig", "pool_waterfall_orig.txt")):
        m = re.match(r"\[done\] (MonoFlex|MonoGround)\*", ln)
        if m:
            cur = m.group(1)
        m2 = re.search(r"NO pool candidate IoU>=0.5: \d+ \(([\d.]+)\)", ln)
        if m2 and cur:
            wf[cur]["miss30"] = float(m2.group(1))
            cur = None
    Hpt = H * 72
    fig = plt.figure(figsize=(C.W_IN, H))
    gs = fig.add_gridspec(1, 2, left=0.095, right=0.992, bottom=bot / Hpt, top=1 - topm / Hpt,
                          wspace=0.30)
    axs = [fig.add_subplot(gs[0, i]) for i in range(2)]
    a = axs[0]
    for i, d in enumerate(DETS):
        c = FCOL[FAM[d]]
        a.plot([i, i], [wf[d]["c7"], wf[d]["a7"]], lw=2.4, c=c, alpha=0.35, solid_capstyle="butt")
        a.plot(i, wf[d]["a7"], "^", ms=3.2, c=c)
        a.plot(i, wf[d]["c7"], "o", ms=3.2, c=c)
    a.set_ylabel("share of moderate GTs\nwith accurate cand.")
    a.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    a.set_title("(a) exists in complete pool ($\\blacktriangle$)\nvs in output ($\\bullet$)")
    hs = [plt.Line2D([], [], color=FCOL[f], lw=2.4, alpha=0.5, label=l) for f, l in
          [("anchor", "anchor (selection-side)"), ("cnet", "CenterNet (generation-side)"),
           ("query", "query (intermediate)")]]
    a.legend(handles=hs, frameon=False, loc="upper right", borderaxespad=0.2, borderpad=0.1,
             handlelength=1.4)
    b = axs[1]
    b.bar(x, [wf[d]["face2"] for d in DETS], 0.55, color=[FCOL[FAM[d]] for d in DETS])
    b.set_ylabel("suppressed-accurate share\n(non-replaced)")
    b.set_title("(b) accurate candidates lost\nto selection")
    for i, d in enumerate(DETS):
        # same numbers as submitted; vertical so neighbours (11.5 pt apart) cannot collide
        b.annotate(f"{wf[d]['miss30']:.2f}", (i, wf[d]["face2"] + 0.015), fontsize=C.FS_ANN,
                   ha="center", va="bottom", rotation=90, color="#666")
    b.text(0.98, 0.97, "grey = 30m+ misses with\nno IoU$\\geq$0.5 pool cand.", fontsize=C.FS_ANN,
           ha="right", va="top", transform=b.transAxes, color="#666")
    b.set_ylim(0, 0.9)
    for a_ in axs:
        style_ax(a_, rot=45)
    finish(fig, "fig4_waterfall")


# ============ Fig 4v2 — generation-selection plane + cross-detector agreement (supplementary) ====
def fig4v2(H=1.74, bot=29.0, topm=22.0):
    G = G4
    wf, agree, hard, FAM, FCOL, DETS2 = G["wf"], G["agree"], G["hard"], G["FAM"], G["FCOL"], G["DETS"]
    FS_L = 7.0
    Hpt = H * 72
    fig = plt.figure(figsize=(C.W_IN, H))
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.992, bottom=bot / Hpt, top=1 - topm / Hpt,
                          wspace=0.27)
    axs = [fig.add_subplot(gs[0, i]) for i in range(2)]
    a = axs[0]
    for d in DETS2:
        A, Cc = wf[d]
        a.scatter(A, Cc / A, s=11, c=FCOL[FAM[d]], zorder=3, linewidths=0)
    qx = [wf[d][0] for d in DETS2 if FAM[d] == "query"]
    qy = [wf[d][1] / wf[d][0] for d in DETS2 if FAM[d] == "query"]
    a.plot(qx, qy, "-", lw=0.7, c="#e15759", alpha=0.5, zorder=2)   # same-family trajectory only
    # same five labels as submitted; offsets re-placed for the smaller canvas (no ink overlap)
    for d, dx, dy, ha, va in [("M3D-RPN", -0.004, 0.03, "right", "bottom"),
                              ("MonoDETR", -0.012, 0.0, "right", "center"),
                              ("MonoCoP", 0.011, 0.0, "left", "center"),
                              ("MonoIA", 0.009, 0.018, "left", "bottom"),
                              ("MonoDLE", -0.004, 0.018, "left", "bottom")]:
        A, Cc = wf[d]
        a.annotate(d, (A + dx, Cc / A + dy), fontsize=C.FS_ANN, color="#444", ha=ha, va=va)
    a.set_xlabel("candidate existence  A@0.7\n(complete native pool)", fontsize=FS_L)
    a.set_ylabel("retention rate  C@0.7 / A@0.7", fontsize=FS_L)
    a.set_title("(a) generation vs selection\nby lineage")
    a.set_xlim(0.2, 0.8); a.set_ylim(0.25, 1.1)
    a.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    hs = [plt.Line2D([], [], marker="o", ls="", ms=3.6, color=FCOL[f], label=l) for f, l in
          [("cnet", "CenterNet: keeps ~all, generates least"),
           ("query", "query: generates more, keeps 79\u201391%"),
           ("anchor", "anchor: generates most, keeps 32%")]]
    a.legend(handles=hs, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.215),
             borderaxespad=0.2, borderpad=0.1, handletextpad=0.2, handlelength=1.0)
    a.grid(lw=0.3, alpha=0.5)
    a.tick_params(labelsize=C.FS_TICK)

    b = axs[1]
    cols = ["#999999"] + ["#4e79a7"] * 11 + ["#2a9d8f"]
    b.bar(range(13), agree, color=cols)
    b.set_xticks(range(13))
    b.set_xlim(-0.6, 12.6)
    b.set_xlabel("# of 12 detectors with an accurate\ncandidate in their complete pool (per GT)",
                 fontsize=FS_L)
    b.set_ylabel("moderate Car GTs", fontsize=FS_L)
    b.set_title("(b) cross-detector agreement\nat IoU$_{3D}$ 0.7")
    b.set_ylim(0, 1450)
    ann = b.annotate(f"panel-unreached set\n{hard['n']} GTs = {hard['share']*100:.1f}%\n(no candidate in ANY pool)",
                     (0.0, agree[0] + 15), xytext=(4.6, 1420), fontsize=C.FS_ANN, color="#555",
                     ha="left", va="top",
                     arrowprops=dict(arrowstyle="->", lw=0.6, color="#777", relpos=(0.0, 0.83),
                                     connectionstyle="angle,angleA=180,angleB=90,rad=0",
                                     shrinkA=1.5, shrinkB=0.5))
    b.grid(axis="y", lw=0.3, alpha=0.5)
    b.tick_params(labelsize=C.FS_TICK)
    finish(fig, "fig4v2_plane_agreement")


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig4v2"]
    for w in which:
        globals()[w]()

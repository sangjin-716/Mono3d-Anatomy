"""Ported from tools/decomp/operating_point_sweep.py for the public release. Computation unchanged.
Produces reports/operating_point_sweep.txt.

Operating-point battery (pre-registered: reports/prereg_2b_sweep.md).

7 non-DETR detectors (DETR family shown near-insensitive previously). All cells use the
validated all-point interpolated AP + official R40 from exact_ap.ap_summaries.

  (ii) base AP at thr {0,0.05,0.1,0.2,0.3,0.4} x NMS {0.4,0.5,0.6,none}   (24 cells)
  (iv) covered by the same grid.
  gap (ceiling - base) at 4 pre-declared cells: (0.2,0.5)=S5 primary, (0.1,0.5), (0.3,0.5),
      (0.2,none). Ceiling scores = oracle IoU computed ONCE on each detector's FULL dump pool
      (cache _opcache_<f>.npz, aligned to dump row index) so every cell subset inherits them.
  (iii) top-K sweep at the S5 cell: per-image top-K by V, K in {20,50,100,all}; gap at K=50,all.
M3D-RPN NOTE: its released val dump has a score floor of 0.05 — its thr=0 and 0.05 cells are
identical and flagged.
Run from the repository root: python tools/decomp/operating_point_sweep.py
"""
import os, sys, shutil, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti, apply_nms
import evaluator.kitti_eval.kitti_common as kc
from exact_ap import ap_summaries
from depth_share_bridge import iou_act_and_zstar

DIAG = cache_dir("decomp")   # work dirs + npz caches (dumps are read through dump_path)
DETS7 = [("M3D-RPN", "m3drpn"), ("MonoDLE", "monodle"), ("MonoFlex", "monoflex"),
         ("GUPNet", "gupnet"), ("DEVIANT", "deviant"), ("MonoGround", "monoground"),
         ("MonoCon", "monocon")]
THRS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
NMSS = [0.4, 0.5, 0.6, None]
GAP_CELLS = [(0.2, 0.5), (0.1, 0.5), (0.3, 0.5), (0.2, None)]
TOPKS = [20, 50, 100, None]
val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
WORK = f"{DIAG}/_opsweep"
OUT = out_path("operating_point_sweep.txt")
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def ap_of(df, score):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    DT = kc.get_label_annos(d, val)
    s = ap_summaries(GT, DT)
    return s["allpoint"], s["r40_official"]


def kept_cell(df, thr, nms):
    pool = df[df["cls"].values >= thr].copy() if thr > 0 else df.copy()
    if nms is not None:
        keep = apply_nms(pool, pool["V"].values, nms)
        pool = pool[keep].copy()
    return pool


def topk_per_img(df, k):
    if k is None:
        return df
    return df.sort_values("V", ascending=False).groupby("sid").head(k)


w(f"# operating_point_sweep run {datetime.datetime.now().isoformat(timespec='seconds')}")
w(f"# prereg: reports/prereg_2b_sweep.md (primary cell = thr0.2+NMS0.5; all else secondary)")
w(f"# AP columns: all-point interpolated AP (official R40 in parens)")
w("")

for name, f in DETS7:
    df = pd.read_csv(dump_path(f)).reset_index(drop=True)
    cache = f"{DIAG}/_opcache_{f}.npz"
    if os.path.exists(cache):
        o_full = np.load(cache)["o_full"]
        assert len(o_full) == len(df)
    else:
        o_full, _ = iou_act_and_zstar(df)     # full dump pool, aligned to df.index
        np.savez_compressed(cache, o_full=o_full)
    w(f"== {name} (dump rows={len(df)}; floor note: min cls={df['cls'].min():.3f})")

    # (ii)+(iv): base surface
    for thr in THRS:
        row = []
        for nms in NMSS:
            kc_ = kept_cell(df, thr, nms)
            a, r40 = ap_of(kc_, kc_["V"].values.astype(float))
            row.append(f"{a:6.2f}({r40:5.2f})")
        w(f"  base thr={thr:4.2f} | " + "  ".join(f"NMS{str(n):>4}:{v}" for n, v in zip(NMSS, row)))

    # gap at pre-declared cells
    for thr, nms in GAP_CELLS:
        kc_ = kept_cell(df, thr, nms)
        sc = o_full[kc_.index.values]
        b, b40 = ap_of(kc_, kc_["V"].values.astype(float))
        c, c40 = ap_of(kc_, sc)
        tag = "S5*" if (thr, nms) == (0.2, 0.5) else "   "
        w(f"  gap {tag} thr={thr:4.2f} NMS={str(nms):>4} | base={b:6.2f} ceil={c:6.2f} "
          f"gap_allpt={c-b:+6.2f} (R40 {c40-b40:+.2f})")

    # (iii) top-K at S5 cell
    s5 = kept_cell(df, 0.2, 0.5)
    for k in TOPKS:
        kk = topk_per_img(s5, k)
        b, _ = ap_of(kk, kk["V"].values.astype(float))
        line = f"  topK K={str(k):>4} | base={b:6.2f}"
        if k in (50, None):
            c, _ = ap_of(kk, o_full[kk.index.values])
            line += f"  gap_allpt={c-b:+6.2f}"
        w(line)
    w("")

w("READ: per prereg, the C1 ordering-headroom wording requires (e) gap>0 & median>=6 at every")
w("gap cell and (f) S5-vs-best-cell gap difference <3 AP for >=6/7. M3D-RPN thr<=0.05 cells are")
w("floor-limited until the floor-0 dump lands (flagged above by min cls).")
if os.path.exists(WORK):
    shutil.rmtree(WORK)
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

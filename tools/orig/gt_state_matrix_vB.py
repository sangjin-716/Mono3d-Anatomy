"""Ported from mono3d_crossdataset/tools/gt_state_matrix_vB.py for the public release. Computation
unchanged. Produces reports_orig/gt_state_matrix_vB.txt and <OUT_DIR>/gt_state_matrix_vB.csv.

GT-level state matrix across the 12-detector panel, vB version: the main-panel
tools/decomp/gt_state_matrix.py with MonoFlex*/MonoGround* read from the original-environment
dumps. COPY-EXTENDED from tools/decomp/pool_waterfall.py (left unmodified): moderate-GT
filter, |dz|<8m candidate gate, native stage definitions and IoU3D kernel are verbatim.

States (PRESENCE layer only — official score-ordered one-to-one TP matching is NOT
reproduced here; the terms "realized"/"TP"/"successful detection" are NOT used anywhere
downstream):
  NO_POOL_CANDIDATE       best complete-pool IoU3D < thr
  LOST_BEFORE_FINAL       best complete-pool IoU3D >= thr, best final-output IoU3D < thr
  FINAL_ACCURATE_PRESENT  best final-output IoU3D >= thr
Stored as floats (a_best, c_best per model) so both IoU 0.7 (primary) and 0.5 (secondary)
layers derive from one file.

GT identity = (sid, index among moderate Cars in label-file line order). Stable because
all 12 model passes read the same label files through the same moderate_gts() in the same
order; recorded in the report (gate G1 additionally proves key-set equality).

Gates (ALL must pass; on any failure the script aborts BEFORE writing aggregates):
  G1 GT key set identical across 12 models; zero duplicates; zero missing (expected 7874)
  G2 c_best <= a_best + 1e-9 for every (GT, model)  [final subset of pool]
  G3 per-model A@0.7 / C@0.7 / face-ii reproduce reports/pool_waterfall.txt and
     pool_waterfall_orig.txt to 3 decimals
Extra (query family only): lost-stage typing — for each LOST GT, every accurate candidate
(IoU3D>=0.7, |dz|<8) is classified by which native eligibility predicate it fails:
budget_only (flatrank>=50, cls>=0.2) / thr_only (flatrank<50, cls<0.2) / both. Sanity:
typed candidate count >= 1 per LOST GT (else FAIL).

Inputs: the released per-prediction dumps for the six CenterNet-family detectors, plus the
complete pools of the two complete-pool release assets (data/DUMPS.md):
<detector>_val_preflatten.csv for the five query detectors and m3drpn_val_floor0.csv for M3D-RPN,
read from DUMP_DIR (or $MONO3D_EXTRA_DUMP_DIR, see tools/orig/_orig_common.extra_dump). Its frozen
output is reports_orig/gt_state_matrix_vB.txt.
Run from the repository root: python tools/orig/gt_state_matrix_vB.py
"""
import os, sys, re, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
from tools.orig._orig_common import extra_dump, frozen_report, orig_report
import numpy as np, pandas as pd
import ap_corrector_arc as arc

BINS = [(0, 15), (15, 30), (30, 45), (45, 1e9)]
CSV = out_path("gt_state_matrix_vB.csv")
OUT = out_path("gt_state_matrix_vB.txt")
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


# ---- verbatim copies from frozen pool_waterfall.py (moderate filter extended to also
# return occ/trunc for the secondary descriptive layer; box tuple unchanged) ----
def moderate_gts(sid):
    p = os.path.join(arc.LABEL_DIR, f"{sid:06d}.txt")
    res = []
    if not os.path.exists(p):
        return res
    for ln in open(p):
        t = ln.split()
        if t[0] != "Car":
            continue
        trunc, occ = float(t[1]), int(t[2])
        hpix = float(t[7]) - float(t[5])
        if occ <= 1 and trunc <= 0.3 and hpix > 25:
            res.append(((float(t[8]), float(t[9]), float(t[10]), float(t[11]),
                         float(t[12]), float(t[13]), float(t[14])), occ, trunc))
    return res


def best_iou_per_gt(g, gts):
    if len(g) == 0:
        return [0.0] * len(gts)
    px = g.x_3d.values; py = g.y_3d.values; pz = g.z_3d.values
    ph = g.h_3d.values; pw = g.w_3d.values; pl = g.l_3d.values; pr = g.ry.values
    res = []
    for t in gts:
        gp = arc.poly(t[3], t[5], t[1], t[2], t[6])
        cand = np.where(np.abs(pz - t[5]) < 8.0)[0]
        best = 0.0
        for k in cand:
            pp = arc.poly(px[k], pz[k], pw[k], pl[k], pr[k])
            v = arc.iou3d(pp, py[k], ph[k], pl[k], pw[k], t, gp)
            if v > best:
                best = v
        res.append(best)
    return res


def acc_cand_flags(g, t):
    """query-family lost-stage typing: per accurate candidate (IoU3D>=0.7, |dz|<8m)
    return (flatrank<50, cls>=0.2) flags."""
    px = g.x_3d.values; py = g.y_3d.values; pz = g.z_3d.values
    ph = g.h_3d.values; pw = g.w_3d.values; pl = g.l_3d.values; pr = g.ry.values
    fr = g.flatrank.values; cl = g.cls.values
    gp = arc.poly(t[3], t[5], t[1], t[2], t[6])
    flags = []
    for k in np.where(np.abs(pz - t[5]) < 8.0)[0]:
        pp = arc.poly(px[k], pz[k], pw[k], pl[k], pr[k])
        if arc.iou3d(pp, py[k], ph[k], pl[k], pw[k], t, gp) >= 0.7:
            flags.append((fr[k] < 50, cl[k] >= 0.2))
    return flags


def stages_for(name):
    if name in ("MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"):
        f = {"MonoDETR": "monodetr", "MonoDGP": "monodgp", "MonoCoP": "official_monocop",
             "MonoCLUE": "monoclue", "MonoIA": "monoia"}[name]
        df = pd.read_csv(extra_dump(f"{f}_val_preflatten.csv"))
        if "class_id" in df.columns:
            car = df[df.class_id == df.car_channel].copy()
            car["flatrank"] = df.loc[car.index, "flat_rank"]
        else:
            car = df.copy(); car["flatrank"] = car["flat_rank_car"]
            car["V"] = car["V_car"]; car["cls"] = car["cls_car"]
        pool = car
        elig = lambda d: d[(d.flatrank < 50) & (d.cls >= 0.2)]
        return pool, elig, True
    if name == "M3D-RPN":
        df = pd.read_csv(extra_dump("m3drpn_val_floor0.csv"))
        pool = df

        def elig(d):
            if len(d) == 0:
                return d
            dd = d.sort_values("V", ascending=False)
            b = dd[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            keep = []
            order = np.arange(len(dd))
            while order.size > 0:
                i = order[0]; keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
                wq = np.maximum(0, xx2 - xx1 + 1); hq = np.maximum(0, yy2 - yy1 + 1)
                ovr = wq * hq / (areas[i] + areas[order[1:]] - wq * hq)
                order = order[np.where(ovr <= 0.4)[0] + 1]
            kk = dd.iloc[keep].head(40)
            return kk[kk["V"] >= 0.75]
        return pool, elig, False
    f = {"MonoDLE": "monodle", "MonoFlex*": "monoflex_orig", "GUPNet": "gupnet",
         "DEVIANT": "deviant", "MonoGround*": "monoground_orig", "MonoCon": "monocon"}[name]
    df = pd.read_csv(dump_path(f))
    thr_native = 0.4 if name == "MonoCon" else 0.2
    col = "cls" if name == "MonoDLE" else "V"
    pool = df
    elig = lambda d, c=col, t=thr_native: d[d[c] >= t]
    return pool, elig, False


MODELS = ["M3D-RPN", "MonoDLE", "MonoFlex*", "GUPNet", "DEVIANT", "MonoGround*", "MonoCon",
          "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]
FAMOF = {"M3D-RPN": "anchor", "MonoDETR": "query", "MonoDGP": "query", "MonoCoP": "query",
         "MonoCLUE": "query", "MonoIA": "query"}
QUERY = [m for m in MODELS if FAMOF.get(m) == "query"]
val = [int(x) for x in open(arc.VAL_LIST).read().split()]

w(f"# gt_state_matrix run {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# presence-layer states only; official TP matching NOT reproduced (terminology restricted)")
w("# GT key = (sid, moderate-Car label-file line-order index); kernels verbatim from pool_waterfall.py")
w("")

# ---- per-model pass ----
rows = {}            # (sid, j) -> dict
lost_typing = {}     # model -> counts dict
for sid in val:
    for j, (t, occ, trunc) in enumerate(moderate_gts(sid)):
        rows[(sid, j)] = {"sid": sid, "gt_idx": j, "z_gt": t[5], "occ": occ, "trunc": trunc}

for name in MODELS:
    pool, elig, is_query = stages_for(name)
    bysid = {s: g for s, g in pool.groupby("sid")}
    typ = {"budget_only": 0, "thr_only": 0, "both": 0, "lost_gts": 0, "untyped_gts": 0}
    for sid in val:
        gts_full = moderate_gts(sid)
        if not gts_full:
            continue
        gts = [x[0] for x in gts_full]
        g = bysid.get(sid, pool.iloc[0:0])
        ge = elig(g)
        a = best_iou_per_gt(g, gts)
        c = best_iou_per_gt(ge, gts)
        for j, t in enumerate(gts):
            r = rows[(sid, j)]
            r[f"{name}_a"] = a[j]; r[f"{name}_c"] = c[j]
            if is_query and a[j] >= 0.7 and c[j] < 0.7:
                typ["lost_gts"] += 1
                flags = acc_cand_flags(g, t)
                if not flags:
                    typ["untyped_gts"] += 1
                for fr_ok, cls_ok in flags:
                    if fr_ok and not cls_ok:
                        typ["thr_only"] += 1
                    elif cls_ok and not fr_ok:
                        typ["budget_only"] += 1
                    elif not fr_ok and not cls_ok:
                        typ["both"] += 1
    if is_query:
        lost_typing[name] = typ
    w(f"[pass] {name} done")

df = pd.DataFrame(sorted(rows.values(), key=lambda r: (r["sid"], r["gt_idx"])))

# ---- gates ----
w(""); w("== GATES ==")
fail = False
n = len(df)
g1a = n == 7874
g1b = df.duplicated(subset=["sid", "gt_idx"]).sum() == 0
g1c = all(df[f"{m}_a"].notna().all() and df[f"{m}_c"].notna().all() for m in MODELS)
w(f"G1 key-set: n={n} (expect 7874) dup=0:{g1b} missing=0:{g1c} -> "
  f"{'PASS' if g1a and g1b and g1c else 'FAIL'}")
fail |= not (g1a and g1b and g1c)
g2bad = sum(int((df[f"{m}_c"] > df[f"{m}_a"] + 1e-9).sum()) for m in MODELS)
w(f"G2 final-subset-of-pool violations: {g2bad} -> {'PASS' if g2bad == 0 else 'FAIL'}")
fail |= g2bad != 0
ref = {}
for _wf in (frozen_report("pool_waterfall.txt"), orig_report("pool_waterfall_orig.txt")):
  for ln in open(_wf):
    m = re.match(r"\[done\] (\S+)\s+\(\w+\) GT=\d+ \| A@0.5=[\d.]+ A@0.7=([\d.]+) "
                 r"B@0.7=[\d.]+ C@0.7=([\d.]+) \| suppressed-acc\(face-ii share\)=([\d.]+)", ln)
    if m:
        ref[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
for mname in MODELS:
    A7 = (df[f"{mname}_a"] >= 0.7).mean()
    C7 = (df[f"{mname}_c"] >= 0.7).mean()
    nA = (df[f"{mname}_a"] >= 0.7).sum()
    fii = ((df[f"{mname}_a"] >= 0.7) & (df[f"{mname}_c"] < 0.7)).sum() / max(nA, 1)
    ra, rc, rf = ref[mname]
    ok = abs(round(A7, 3) - ra) < 1e-9 and abs(round(C7, 3) - rc) < 1e-9 and abs(round(fii, 3) - rf) < 1e-9
    w(f"G3 {mname:10s} A7={A7:.3f}/{ra:.3f} C7={C7:.3f}/{rc:.3f} faceii={fii:.3f}/{rf:.3f} "
      f"-> {'PASS' if ok else 'FAIL'}")
    fail |= not ok
untyped = sum(t["untyped_gts"] for t in lost_typing.values())
w(f"G4 query lost-typing coverage: untyped lost GTs = {untyped} -> "
  f"{'PASS' if untyped == 0 else 'FAIL'}")
fail |= untyped != 0
if fail:
    w(""); w("!! GATE FAILURE — aggregates NOT computed; the frozen reports/pool_waterfall.txt stands !!")
    open(OUT, "w").write("\n".join(out) + "\n")
    sys.exit(1)
df.to_csv(CSV, index=False)
w(f"[written] {CSV} rows={len(df)}")

# ---- aggregates (primary; counts only, no AP) ----
for thr, tag in ((0.7, "IoU0.7 (primary)"), (0.5, "IoU0.5 (secondary)")):
    w(""); w(f"== {tag} ==")
    A = np.stack([(df[f"{m}_a"] >= thr).values for m in MODELS], 1)
    C = np.stack([(df[f"{m}_c"] >= thr).values for m in MODELS], 1)
    nA = A.sum(1); nC = C.sum(1)
    w("agreement: GTs by #models with accurate POOL candidate (0..12):")
    w("  " + " ".join(f"{k}:{int((nA == k).sum())}" for k in range(13)))
    w("agreement: GTs by #models with FINAL_ACCURATE_PRESENT (0..12):")
    w("  " + " ".join(f"{k}:{int((nC == k).sum())}" for k in range(13)))
    hard = nA == 0
    w(f"PANEL HARD CORE (no IoU>={thr} candidate in ANY of the 12 complete native pools): "
      f"{hard.sum()} / {len(df)} = {hard.mean():.3f}")
    for lo, hi in BINS:
        sel = (df.z_gt >= lo) & (df.z_gt < hi)
        w(f"  bin [{lo}-{'+' if hi > 1e8 else int(hi)}] n={int(sel.sum())} "
          f"hard-core={hard[sel.values].mean():.3f} "
          f"final-present-nowhere={(nC[sel.values] == 0).mean():.3f}")
    w(f"hard-core occ/trunc vs rest: occ {df.occ[hard].mean():.2f}/{df.occ[~hard].mean():.2f}, "
      f"trunc {df.trunc[hard].mean():.3f}/{df.trunc[~hard].mean():.3f}, "
      f"z {df.z_gt[hard].mean():.1f}/{df.z_gt[~hard].mean():.1f} m")
    nz = (nC == 0) & ~hard
    w(f"outside hard core: candidate exists somewhere but FINAL_ACCURATE_PRESENT nowhere: "
      f"{int(nz.sum())} ({nz.mean():.3f} of all GTs)")

# family state distribution outside the IoU0.7 hard core
A7m = {m: (df[f"{m}_a"] >= 0.7).values for m in MODELS}
C7m = {m: (df[f"{m}_c"] >= 0.7).values for m in MODELS}
hard7 = np.stack(list(A7m.values()), 1).sum(1) == 0
w(""); w("== family state distribution OUTSIDE the IoU0.7 hard core (share of those GTs) ==")
w(f"{'model':10s} {'fam':6s} NO_POOL_CAND  LOST_BEFORE_FINAL  FINAL_ACC_PRESENT  retention(C/A)")
for m in MODELS:
    sel = ~hard7
    a = A7m[m][sel]; c = C7m[m][sel]
    w(f"{m:10s} {FAMOF.get(m, 'cnet'):6s} {(~a).mean():12.3f} {(a & ~c).mean():18.3f} "
      f"{c.mean():18.3f} {c.sum() / max(a.sum(), 1):14.3f}")

w(""); w("== query lost-stage typing (accurate candidates of LOST GTs; candidate-level) ==")
for m, t in lost_typing.items():
    tot = t["budget_only"] + t["thr_only"] + t["both"]
    w(f"{m:10s} lost_gts={t['lost_gts']:4d} acc-cands typed={tot:5d} | "
      f"thr_only={t['thr_only']/max(tot,1):.3f} budget_only={t['budget_only']/max(tot,1):.3f} "
      f"both={t['both']/max(tot,1):.3f}")

w(""); w("SCOPE LOCK: hard-core shares are statements about the complete native pools exposed")
w("by the studied 12-model panel ONLY — no generalization beyond it; no temporal causal")
w("reading of family/generation differences; presence-layer states, not evaluator TPs.")
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

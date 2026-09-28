"""Produces reports/final_run/difficulty_sweep/anatomy_by_difficulty.{txt,json}.

Difficulty robustness sweep, C (geometry anatomy) for Easy/Mod/Hard.

Reuses tools/decomp/oracle_anatomy.py unchanged (S5 pool = pool_mask thr0.2 + NMS0.5; best-IoU GT
match; pure_z / ray-consistent-centre / full-centre oracle geometries). ONLY change: the final
ap_of is looped over difficulty {0,1,2}. The matched-box geometry oracles are difficulty-agnostic;
only the evaluation GT subset changes. Mod(=1) reproduces oracle_anatomy.txt = the gate.

These are SEPARATE oracle interventions, NOT an additive decomposition (construct preserved).
Run from the repository root: python tools/decomp/diffsweep_anatomy.py
"""
import os, sys, shutil, json, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti, pool_mask, apply_nms
from evaluator.kitti_utils import Calibration
import evaluator.kitti_eval.kitti_common as kc
from exact_ap import ap_summaries

DIAG = cache_dir("decomp")   # work dirs + npz caches (dumps are read through dump_path)
DETS = [("M3D-RPN", "m3drpn"), ("MonoDLE", "monodle"), ("MonoFlex", "monoflex"),
        ("GUPNet", "gupnet"), ("DEVIANT", "deviant"), ("MonoGround", "monoground"),
        ("MonoCon", "monocon"), ("MonoDETR", "monodetr"), ("MonoDGP", "dgp"),
        ("MonoCoP", "official_monocop"), ("MonoCLUE", "monoclue"), ("MonoIA", "monoia")]
VARIANTS = ["pure_z", "ray", "centre"]
DIFFS = [("Easy", 0), ("Mod", 1), ("Hard", 2)]
val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
W = f"{DIAG}/_diffsweep_anat"
OUTDIR = os.path.join(paths.OUT_DIR, "final_run", "difficulty_sweep")
os.makedirs(OUTDIR, exist_ok=True)
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def ap_of(df, cols, score, difficulty):  # adapted from oracle_anatomy.ap_of (+ difficulty)
    d2 = df.copy()
    for k, v in cols.items():
        d2[k] = v
    if os.path.exists(W):
        shutil.rmtree(W)
    dd = os.path.join(W, "data")
    write_kitti(d2.reset_index(drop=True), np.asarray(score, float), dd, val)
    s = ap_summaries(GT, kc.get_label_annos(dd, val), difficulty=difficulty, metric=2, min_overlap=0.7)
    return s["allpoint"], s["r40_official"]


def match_and_variants(kept):  # VERBATIM construct from oracle_anatomy.py (3 variants kept)
    cols = {v: {"x_3d": kept.x_3d.values.copy(), "y_3d": kept.y_3d.values.copy(),
                "z_3d": kept.z_3d.values.copy(), "h_3d": kept.h_3d.values.copy(),
                "w_3d": kept.w_3d.values.copy(), "l_3d": kept.l_3d.values.copy(),
                "ry": kept.ry.values.copy()} for v in VARIANTS}
    nmatch = 0
    for sid, g in kept.groupby("sid"):
        gts = arc.read_gt(int(sid))
        if not gts:
            continue
        cal = Calibration(os.path.join(arc.CALIB_DIR, f"{int(sid):06d}.txt"))
        tx, ty = cal.tx, cal.ty
        gps = [arc.poly(t[3], t[5], t[1], t[2], t[6]) for t in gts]
        for k in g.index.values:
            x0, y0, z0 = kept.at[k, "x_3d"], kept.at[k, "y_3d"], kept.at[k, "z_3d"]
            h3, w3, l3, ry = kept.at[k, "h_3d"], kept.at[k, "w_3d"], kept.at[k, "l_3d"], kept.at[k, "ry"]
            if z0 <= 0.1:
                continue
            pp = arc.poly(x0, z0, w3, l3, ry)
            best, bj = 0.0, -1
            for j, t in enumerate(gts):
                v_ = arc.iou3d(pp, y0, h3, l3, w3, t, gps[j])
                if v_ > best:
                    best, bj = v_, j
            if bj < 0 or best <= 0.0:
                continue
            nmatch += 1
            t = gts[bj]
            gh, gw, gl = t[0], t[1], t[2]; gx, gz, gry = t[3], t[5], t[6]; gy = t[4]
            s = gz / z0
            geo = {
                "pure_z": (x0, y0, gz, h3, w3, l3, ry),
                "ray": (tx + (x0 - tx) * s, (h3 / 2 + ty) + (y0 - h3 / 2 - ty) * s, gz, h3, w3, l3, ry),
                "centre": (gx, gy, gz, h3, w3, l3, ry),
            }
            for vn, (vx, vy, vz, vh, vw, vl, vr) in geo.items():
                c = cols[vn]
                c["x_3d"][k] = vx; c["y_3d"][k] = vy; c["z_3d"][k] = vz
                c["h_3d"][k] = vh; c["w_3d"][k] = vw; c["l_3d"][k] = vl; c["ry"][k] = vr
    return cols, nmatch


# Canonical Moderate gate (oracle_anatomy.txt): base, Gz(pure_z), ray, centre  (all-point)
GATE_MOD = {
    "M3D-RPN": (11.49, 18.62, 36.92, 46.29), "MonoDLE": (15.12, 27.69, 53.25, 64.15),
    "MonoFlex": (16.28, 23.18, 44.06, 62.90), "GUPNet": (17.12, 24.24, 46.47, 57.10),
    "DEVIANT": (17.50, 25.13, 47.41, 58.19), "MonoGround": (17.48, 23.33, 41.00, 59.68),
    "MonoCon": (19.60, 25.50, 48.99, 58.88), "MonoDETR": (21.19, 24.16, 43.68, 53.82),
}  # 8 detectors published in oracle_anatomy.txt head; remaining 4 gated by ratio only

w(f"# diffsweep anatomy {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# C=geometry anatomy (separate oracles, NOT additive): pure_z / ray-consistent-centre / full-centre")
w("# S5 pool (pool_mask thr0.2 + NMS0.5). all-point (R40). Mod(=1) reproduces oracle_anatomy.txt (gate).")
w("")
records = {}
for name, f in DETS:
    df = pd.read_csv(dump_path(f)).reset_index(drop=True)
    pre = df[pool_mask(df, "thr0.2")].copy()
    keep = apply_nms(pre, pre["V"].values, 0.5)
    kept = pre[keep].copy().reset_index(drop=True)
    V = kept["V"].values.astype(float)
    cols, nmatch = match_and_variants(kept)
    ap = {}  # ap[tag][diff]=(allpt,r40)
    for tag, c in [("base", {}), ("pure_z", cols["pure_z"]), ("ray", cols["ray"]), ("centre", cols["centre"])]:
        ap[tag] = {}
        for dname, dval in DIFFS:
            ap[tag][dname] = tuple(round(x, 3) for x in ap_of(kept, c, V, dval))
    rec = {"n": int(len(kept)), "nmatch": nmatch, "ap": ap}
    for dname, _ in DIFFS:
        ba = ap["base"][dname][0]
        gz = ap["pure_z"][dname][0] - ba; gray = ap["ray"][dname][0] - ba; gc = ap["centre"][dname][0] - ba
        rec.setdefault("G_purez", {})[dname] = round(gz, 3)
        rec.setdefault("G_ray", {})[dname] = round(gray, 3)
        rec.setdefault("G_centre", {})[dname] = round(gc, 3)
        rec.setdefault("purez_over_centre", {})[dname] = round(gz / gc, 3) if abs(gc) > 1e-6 else None
        rec.setdefault("ray_over_centre", {})[dname] = round(gray / gc, 3) if abs(gc) > 1e-6 else None
    records[name] = rec
    # gate (Mod): ratio bar + (if published) absolute centre
    pzc = rec["purez_over_centre"]["Mod"]; rc = rec["ray_over_centre"]["Mod"]
    gtag = "ratio✓" if (pzc is not None and 0.30 <= pzc <= 0.55 and rc is not None and 0.70 <= rc <= 0.92) else f"ratioCHECK(pz{pzc}/ray{rc})"
    if name in GATE_MOD:
        gc_ref = GATE_MOD[name][3]
        gtag += " centre✓" if abs(rec["G_centre"]["Mod"] - gc_ref) <= 1.5 else f" centreCHECK(Δ{rec['G_centre']['Mod']-gc_ref:+.1f})"
    w(f"[{name:10s}] gate(Mod)={gtag}  n={len(kept)} matched={nmatch}")
    for dname, _ in DIFFS:
        w(f"    {dname:4s} base={ap['base'][dname][0]:6.2f}  pure_z=+{rec['G_purez'][dname]:5.2f}"
          f"  ray=+{rec['G_ray'][dname]:5.2f}  centre=+{rec['G_centre'][dname]:5.2f}"
          f"  | pure_z/centre={rec['purez_over_centre'][dname]}  ray/centre={rec['ray_over_centre'][dname]}")

w("")
w("== panel summary per difficulty (median ratios; separate oracles, not additive) ==")
for dname, _ in DIFFS:
    pz = [records[n]["purez_over_centre"][dname] for n, _ in DETS if records[n]["purez_over_centre"][dname] is not None]
    rr = [records[n]["ray_over_centre"][dname] for n, _ in DETS if records[n]["ray_over_centre"][dname] is not None]
    gc = [records[n]["G_centre"][dname] for n, _ in DETS]
    w(f"{dname:4s}: pure_z/centre median={np.median(pz):.3f}  ray/centre median={np.median(rr):.3f}"
      f"  | full-centre gap min/med/max = +{min(gc):.1f}/+{np.median(gc):.1f}/+{max(gc):.1f}")
if os.path.exists(W):
    shutil.rmtree(W)
open(f"{OUTDIR}/anatomy_by_difficulty.txt", "w").write("\n".join(out) + "\n")
json.dump(records, open(f"{OUTDIR}/anatomy_by_difficulty.json", "w"), indent=2)
w(f"[written] {OUTDIR}/anatomy_by_difficulty.{{txt,json}}")

"""Produces reports/final_run/probe_official_moderate.txt.

E1': progression probe channels re-aggregated on the OFFICIAL KITTI moderate set
(occ<=1 AND trunc<=0.3 AND bbox pixel height>25). Near-copy of
probe_detector_progression.py (matcher/pools/kernels identical); ONLY read_gt_mod
changes + GT-count gate (total must equal 7,874 = pool_waterfall count) + delta table vs
the earlier probe report.
Run from the repository root: python tools/decomp/probe_official_moderate.py
"""
import os, sys, math, re
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import pool_mask, apply_nms
from scipy.stats import spearmanr

DIAG = cache_dir("decomp")   # work dirs + npz caches (dumps are read through dump_path)
LABEL = os.path.join(paths.LABEL_DIR, "{:06d}.txt")
VAL = paths.VAL_LIST
BINS = ["0-15", "15-30", "30-45", "45+"]
DETS = [("M3D-RPN", "m3drpn", 11.10), ("MonoDLE", "monodle", 14.69),
        ("MonoFlex", "monoflex", 15.57), ("GUPNet", "gupnet", 16.48),
        ("DEVIANT", "deviant", 16.74), ("MonoGround", "monoground", 16.82),
        ("MonoCon", "monocon", 19.05), ("MonoDETR", "monodetr", 20.98),
        ("MonoDGP", "dgp", 22.29), ("MonoCoP", "official_monocop", 23.89),
        ("MonoCLUE", "monoclue", 24.23), ("MonoIA", "monoia", 24.49)]
OUT = out_path("final_run/probe_official_moderate.txt")
out_lines = []


def emit(s=""):
    print(s, flush=True); out_lines.append(s)


def dbin(z):
    return "0-15" if z < 15 else "15-30" if z < 30 else "30-45" if z < 45 else "45+"


def read_gt_official_mod(sid):
    """OFFICIAL moderate Car GTs: occ<=1 & trunc<=0.3 & bbox pix height>25."""
    out = []; p = LABEL.format(sid)
    if not os.path.exists(p):
        return out
    for ln in open(p):
        t = ln.split()
        if t[0] != "Car":
            continue
        trunc = float(t[1]); occ = int(t[2]); hpix = float(t[7]) - float(t[5])
        if occ >= 2 or trunc > 0.3 or hpix <= 25:
            continue
        out.append((float(t[8]), float(t[9]), float(t[10]), float(t[11]),
                    float(t[12]), float(t[13]), float(t[14])))
    return out


def match_full(df):
    z3 = df.z_3d.values; px = df.x_3d.values; py = df.y_3d.values
    pw = df.w_3d.values; pl = df.l_3d.values; pry = df.ry.values; ph = df.h_3d.values
    n = len(df)
    best_iou = np.zeros(n); gz = np.full(n, np.nan)
    pos = {ri: i for i, ri in enumerate(df.index)}
    for sid, g in df.groupby("sid"):
        gts = arc.read_gt(int(sid))
        if not gts:
            continue
        gps = [arc.poly(t[3], t[5], t[1], t[2], t[6]) for t in gts]
        gzz = np.array([t[5] for t in gts])
        for k in g.index.values:
            i = pos[k]
            cand = np.where(np.abs(z3[k] - gzz) < 8.0)[0]
            if cand.size == 0:
                continue
            pp = arc.poly(px[k], z3[k], pw[k], pl[k], pry[k]); best = 0.0; bj = -1
            for j in cand:
                io = arc.iou3d(pp, py[k], ph[k], pl[k], pw[k], gts[j], gps[j])
                if io > best:
                    best = io; bj = j
            best_iou[i] = best
            if bj >= 0 and best >= 0.05:
                gz[i] = gts[bj][5]
    return best_iou, gz


def recall_by_bin(kept, val_sids):
    grp = {s: g for s, g in kept.groupby("sid")}
    nGT = {b: 0 for b in BINS}; hit = {b: 0 for b in BINS}
    for sid in val_sids:
        gts = read_gt_official_mod(sid)
        if not gts:
            continue
        g = grp.get(sid)
        if g is not None:
            px = g.x_3d.values; py = g.y_3d.values; pz = g.z_3d.values
            ph = g.h_3d.values; pw = g.w_3d.values; pl = g.l_3d.values; pry = g.ry.values
        for t in gts:
            b = dbin(t[5]); nGT[b] += 1
            if g is None:
                continue
            gp = arc.poly(t[3], t[5], t[1], t[2], t[6])
            cand = np.where(np.abs(pz - t[5]) < 8.0)[0]
            best = 0.0
            for k in cand:
                io = arc.iou3d(arc.poly(px[k], pz[k], pw[k], pl[k], pry[k]),
                               py[k], ph[k], pl[k], pw[k], t, gp)
                if io > best:
                    best = io
            if best >= 0.7:
                hit[b] += 1
    return nGT, hit


def main():
    val_sids = [int(x) for x in open(VAL).read().split()]
    emit("# E1' probe channels on the OFFICIAL moderate set (occ<=1, trunc<=0.3, h>25px)")
    emit("# matcher/pool identical to frozen probe; only the recall GT filter changes")
    # GATE: total official-moderate GT count must equal 7,874
    tot = sum(len(read_gt_official_mod(s)) for s in val_sids)
    emit(f"GATE GT-count: {tot} (expect 7874) -> {'PASS' if tot == 7874 else 'FAIL'}")
    if tot != 7874:
        open(OUT, "w").write("\n".join(out_lines) + "\n"); sys.exit(1)
    rows = []
    for name, f, base_ap in DETS:
        va = pd.read_csv(dump_path(f)).reset_index(drop=True)
        prepool = va[pool_mask(va, "thr0.2")].copy()
        keep = apply_nms(prepool, prepool["V"].values, 0.5)
        kept = prepool[keep].copy().reset_index(drop=True)
        best_iou, gz = match_full(kept)
        mz = best_iou >= 0.05
        dz = np.abs(kept.z_3d.values - gz)
        dz_bin = {}
        for b in BINS:
            sel = mz & np.array([(not math.isnan(z)) and dbin(z) == b for z in gz])
            dz_bin[b] = float(np.nanmean(dz[sel])) if sel.sum() else float("nan")
        nGT, hit = recall_by_bin(kept, val_sids)
        rec_bin = {b: hit[b] / max(1, nGT[b]) for b in BINS}
        rec_all = sum(hit.values()) / max(1, sum(nGT.values()))
        rows.append(dict(name=name, ap=base_ap, rec_all=rec_all, rec_bin=rec_bin,
                         dz_bin=dz_bin, nGT=nGT))
        emit(f"[done] {name:10s} rec_all={rec_all:.3f} rec45={rec_bin['45+']:.3f} "
             f"|dz|45={dz_bin['45+'] if not math.isnan(dz_bin['45+']) else float('nan'):.3f}")
    emit("")
    emit("OFFICIAL-MODERATE RECALL@IoU0.7 BY BIN (n per bin: " +
         " ".join(f"{b}={rows[0]['nGT'][b]}" for b in BINS) + ")")
    for r in rows:
        emit(f"{r['name']:10s} {r['ap']:6.2f} | " +
             " ".join(f"{r['rec_bin'][b]:7.3f}" for b in BINS) + f" {r['rec_all']:7.3f}")
    emit("")
    emit("OFFICIAL-MODERATE |dz| ON MATCHED, BY MATCHED-GT BIN")
    for r in rows:
        emit(f"{r['name']:10s} {r['ap']:6.2f} | " +
             " ".join((f"{r['dz_bin'][b]:7.3f}" if not math.isnan(r['dz_bin'][b]) else "    NA ")
                      for b in BINS))
    ap = np.array([r['ap'] for r in rows])
    emit("")
    emit("SPEARMAN vs base AP (official set):")
    for b in BINS + ["ALL"]:
        v = np.array([r['rec_all'] if b == "ALL" else r['rec_bin'][b] for r in rows])
        emit(f"  recall {b:6s}: {spearmanr(ap, v).correlation:+.3f}   " +
             " ".join(f"{x:.3f}" for x in v))
    # delta vs frozen probe (45+ only — the contested bin)
    emit("")
    emit("DELTA vs frozen probe (45+ recall): probe-set (n=1867, occ<=1/trunc<=0.5/no-h-filter)")
    old = {"M3D-RPN": 0.010, "MonoDLE": 0.033, "MonoFlex": 0.018, "GUPNet": 0.013,
           "DEVIANT": 0.011, "MonoGround": 0.020, "MonoCon": 0.012, "MonoDETR": 0.028,
           "MonoDGP": 0.012, "MonoCoP": 0.022, "MonoCLUE": 0.028, "MonoIA": 0.035}
    for r in rows:
        emit(f"  {r['name']:10s} old(9668-set)={old[r['name']]:.3f}  "
             f"official(7874-set)={r['rec_bin']['45+']:.3f}")
    open(OUT, "w").write("\n".join(out_lines) + "\n")
    emit(f"[written] {OUT}")


if __name__ == "__main__":
    main()

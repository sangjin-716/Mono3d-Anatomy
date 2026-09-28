"""Produces reports_orig/probe_orig.txt.

Detector-progression diagnostic probe (NO retraining, val dumps only) for the
original-environment MonoFlex*/MonoGround* dumps (a copy of the main-panel
tools/decomp/probe_detector_progression.py, with inputs switched; figure 1 inputs).

On the IDENTICAL S5 kept pool (cls>=0.2 + greedy 2D-NMS@0.5 by native V), compute one
CONSISTENT metric set using the SAME arc-style IoU3D matcher for every detector, so the only
thing that changes row-to-row is the detector.

Metrics (S5 kept pool, Car only):
  - n_kept, n_matched_TP (IoU3D>=0.05 to a Car GT, greedy best per box)
  - GT recall = distinct moderate GTs matched / moderate GTs  (overall + dist bins)
  - localization on matched TP: mean|dz| depth (overall+bins), |dx| lateral,
    dim err (mean |dh|+|dw|+|dl|), yaw err folded to [0,90]
  - rank: Spearman(V, max_iou_3d) on kept pool; matched-only Spearman
  - precision: frac kept-matched with IoU3D>=0.7 (cleared gate) vs [0.05,0.7)
  - dup proxy: dup_rank>0 frac on cls>=0.2 pre-NMS pool; kept/matched ratio
The monotonicity section is nan with two rows (it is filled in the 12-detector view built by
make_vB_reports.py).
Run from the repository root: python tools/orig/probe_orig.py
"""
import os, sys, math
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
from tools.orig._orig_common import ORIG
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import pool_mask, apply_nms
from scipy.stats import spearmanr

LABEL = os.path.join(paths.LABEL_DIR, "{:06d}.txt")
VAL = paths.VAL_LIST
BINS = ["0-15", "15-30", "30-45", "45+"]

# ordered weak -> strong base AP (Mod R40 IoU0.7 3D), as given by the caller
DETS = [
    ("MonoFlex*", "monoflex_orig", 17.38),
    ("MonoGround*", "monoground_orig", 18.73),
]


def dbin(z):
    return "0-15" if z < 15 else "15-30" if z < 30 else "30-45" if z < 45 else "45+"


def read_gt_mod(sid):
    """Moderate Car GTs (occ<=1 & trunc<=0.5): h w l x y z ry."""
    out = []; p = LABEL.format(sid)
    if not os.path.exists(p):
        return out
    for ln in open(p):
        t = ln.split()
        if t[0] != "Car":
            continue
        trunc = float(t[1]); occ = int(t[2])
        if occ >= 2 or trunc > 0.5:
            continue
        out.append((float(t[8]), float(t[9]), float(t[10]), float(t[11]),
                    float(t[12]), float(t[13]), float(t[14])))
    return out


def match_full(df):
    """Per kept pred -> best Car GT (IoU3D>=0.05). Returns arrays aligned to df rows:
    best_iou, gt_idx (-1 if none>=0.05), and matched gt geometry h,w,l,x,y,z,ry.
    Uses ALL Car GTs (not only moderate) for the per-box best-IoU/precision metric,
    matching arc.match conventions. Recall uses a separate moderate-GT pass."""
    z3 = df.z_3d.values; px = df.x_3d.values; py = df.y_3d.values
    pw = df.w_3d.values; pl = df.l_3d.values; pry = df.ry.values; ph = df.h_3d.values
    n = len(df)
    best_iou = np.zeros(n); gx = np.full(n, np.nan); gz = np.full(n, np.nan)
    gh = np.full(n, np.nan); gw = np.full(n, np.nan); gl = np.full(n, np.nan)
    gry = np.full(n, np.nan)
    pos = {ri: i for i, ri in enumerate(df.index)}
    for sid, g in df.groupby("sid"):
        gts = arc.read_gt(int(sid))  # all Car GTs: h w l x y z ry
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
                t = gts[bj]
                gh[i], gw[i], gl[i], gx[i] = t[0], t[1], t[2], t[3]
                gz[i] = t[5]; gry[i] = t[6]
    return best_iou, gx, gz, gh, gw, gl, gry


def recall_by_bin(kept, val_sids):
    """Distinct moderate GTs covered by a kept TP (IoU3D>=0.7) / moderate GTs, per bin.
    A GT counts as recalled if any kept-pool box reaches IoU3D>=0.7 with it (standard
    TP-at-0.7 recall, dedup handled by NMS already)."""
    grp = {s: g for s, g in kept.groupby("sid")}
    nGT = {b: 0 for b in BINS}; hit = {b: 0 for b in BINS}
    for sid in val_sids:
        gts = read_gt_mod(sid)
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


def fold_yaw(d):
    """fold yaw error to [0,90] deg to kill heading-flip artifact."""
    a = np.abs(d) % math.pi
    a = np.minimum(a, math.pi - a)  # [0,90]
    return np.degrees(a)


def fmt(x, n=2):
    return "  NA " if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{n}f}"


def main():
    val_sids = [int(x) for x in open(VAL).read().split()]
    rows = []
    out_lines = []

    def emit(s=""):
        print(s, flush=True); out_lines.append(s)

    emit("=" * 110)
    emit("DETECTOR-PROGRESSION DIAGNOSTIC PROBE  (val dumps only, no retraining)")
    emit("S5 kept pool = cls>=0.2 + greedy 2D-NMS@0.5 by native V; arc IoU3D matcher (identical for all 7)")
    emit("=" * 110)

    for name, f, base_ap in DETS:
        va = pd.read_csv(dump_path(f)).reset_index(drop=True)
        prepool = va[pool_mask(va, "thr0.2")].copy()
        n_pre = len(prepool)
        # dup proxy on pre-NMS cls>=0.2 pool
        dup_frac = float((prepool["dup_rank"].values > 0).mean()) if "dup_rank" in prepool else float("nan")
        keep = apply_nms(prepool, prepool["V"].values, 0.5)
        kept = prepool[keep].copy().reset_index(drop=True)
        n_kept = len(kept)

        best_iou, gx, gz, gh, gw, gl, gry = match_full(kept)
        matched = best_iou >= 0.05
        n_tp = int(matched.sum())

        # localization on matched TP
        dz = kept.z_3d.values - gz
        dx = kept.x_3d.values - gx
        dim_err = (np.abs(kept.h_3d.values - gh) + np.abs(kept.w_3d.values - gw)
                   + np.abs(kept.l_3d.values - gl))
        yaw_err = fold_yaw(kept.ry.values - gry)

        mz = matched
        mean_dz = float(np.nanmean(np.abs(dz[mz]))) if n_tp else float("nan")
        mean_dx = float(np.nanmean(np.abs(dx[mz]))) if n_tp else float("nan")
        mean_dim = float(np.nanmean(dim_err[mz])) if n_tp else float("nan")
        mean_yaw = float(np.nanmean(yaw_err[mz])) if n_tp else float("nan")

        # depth |dz| by bin (over matched TP, binned by matched gt z)
        dz_bin = {}
        gzb = gz.copy()
        for b in BINS:
            sel = mz & np.array([(not math.isnan(z)) and dbin(z) == b for z in gzb])
            dz_bin[b] = float(np.nanmean(np.abs(dz[sel]))) if sel.sum() else float("nan")

        # precision gate
        frac_iou07 = float((best_iou >= 0.7).mean())
        frac_found_imprecise = float(((best_iou >= 0.05) & (best_iou < 0.7)).mean())
        # among MATCHED only, frac clearing 0.7
        frac07_of_tp = float((best_iou[mz] >= 0.7).mean()) if n_tp else float("nan")

        # rank quality
        V = kept["V"].values
        rho = spearmanr(V, best_iou).correlation
        rho_m = spearmanr(V[mz], best_iou[mz]).correlation if n_tp > 5 else float("nan")

        # recall by bin
        nGT, hit = recall_by_bin(kept, val_sids)
        nGT_all = sum(nGT.values()); hit_all = sum(hit.values())
        rec_all = hit_all / max(1, nGT_all)
        rec_bin = {b: hit[b] / max(1, nGT[b]) for b in BINS}

        kept_matched_ratio = n_kept / max(1, n_tp)

        rows.append(dict(
            name=name, ap=base_ap, n_pre=n_pre, n_kept=n_kept, n_tp=n_tp,
            rec_all=rec_all, rec_bin=rec_bin, nGT=nGT, hit=hit,
            mean_dz=mean_dz, dz_bin=dz_bin, mean_dx=mean_dx, mean_dim=mean_dim, mean_yaw=mean_yaw,
            frac_iou07=frac_iou07, frac_found_imprecise=frac_found_imprecise, frac07_of_tp=frac07_of_tp,
            rho=rho, rho_m=rho_m, dup_frac=dup_frac, kmr=kept_matched_ratio,
        ))
        emit(f"[done] {name:9s} ap={base_ap:5.2f} n_kept={n_kept:6d} n_tp={n_tp:5d} "
             f"rec={rec_all:.3f} |dz|={mean_dz:.3f} frac>=0.7={frac_iou07:.3f} rho={rho:.3f}")

    # ---------------- MAIN TABLE ----------------
    emit("")
    emit("#" * 110)
    emit("MAIN TABLE  (rows ordered by base AP, weak -> strong)")
    emit("#" * 110)
    hdr = (f"{'detector':9s} {'baseAP':>6} {'n_kept':>6} {'n_TP':>5} "
           f"{'recall':>6} {'|dz|':>6} {'|dx|':>6} {'dimErr':>6} {'yawErr':>6} "
           f"{'fIoU>=.7':>8} {'fIoU.05-.7':>10} {'rho(V,IoU)':>10} {'rho_TP':>6} {'dup%':>5}")
    emit(hdr)
    emit("-" * len(hdr))
    for r in rows:
        emit(f"{r['name']:9s} {r['ap']:6.2f} {r['n_kept']:6d} {r['n_tp']:5d} "
             f"{r['rec_all']:6.3f} {fmt(r['mean_dz'],3):>6} {fmt(r['mean_dx'],3):>6} "
             f"{fmt(r['mean_dim'],3):>6} {fmt(r['mean_yaw'],2):>6} "
             f"{r['frac_iou07']:8.3f} {r['frac_found_imprecise']:10.3f} "
             f"{r['rho']:10.3f} {fmt(r['rho_m'],3):>6} {r['dup_frac']*100:5.1f}")

    # ---------------- RECALL BY DISTANCE ----------------
    emit("")
    emit("GT RECALL (TP@IoU0.7) BY DISTANCE BIN  (moderate Car GTs)")
    emit(f"{'detector':9s} {'baseAP':>6} | " + " ".join(f"{b:>7}" for b in BINS) + f" {'ALL':>7}")
    emit("-" * 70)
    for r in rows:
        emit(f"{r['name']:9s} {r['ap']:6.2f} | "
             + " ".join(f"{r['rec_bin'][b]:7.3f}" for b in BINS)
             + f" {r['rec_all']:7.3f}")
    emit("(moderate GT counts per bin, last row): " +
         " ".join(f"{b}={rows[0]['nGT'][b]}" for b in BINS))

    # ---------------- DEPTH |dz| BY DISTANCE ----------------
    emit("")
    emit("DEPTH ERROR mean|dz| (m) ON MATCHED TP, BY DISTANCE BIN")
    emit(f"{'detector':9s} {'baseAP':>6} | " + " ".join(f"{b:>7}" for b in BINS) + f" {'ALL':>7}")
    emit("-" * 70)
    for r in rows:
        emit(f"{r['name']:9s} {r['ap']:6.2f} | "
             + " ".join(f"{fmt(r['dz_bin'][b],3):>7}" for b in BINS)
             + f" {fmt(r['mean_dz'],3):>7}")

    # ---------------- MONOTONICITY ANALYSIS ----------------
    emit("")
    emit("#" * 110)
    emit("MONOTONICITY vs base AP (Spearman across the 7 detectors; +1=rises with AP, -1=falls)")
    emit("#" * 110)
    ap = np.array([r['ap'] for r in rows])

    def mono(key, getter, invert_good=False):
        vals = np.array([getter(r) for r in rows], float)
        ok = ~np.isnan(vals)
        if ok.sum() < 3:
            return float("nan"), vals
        rho = spearmanr(ap[ok], vals[ok]).correlation
        return rho, vals

    metrics = [
        ("GT recall", lambda r: r['rec_all'], "higher=better"),
        ("recall 0-15", lambda r: r['rec_bin']['0-15'], "higher=better"),
        ("recall 15-30", lambda r: r['rec_bin']['15-30'], "higher=better"),
        ("recall 30-45", lambda r: r['rec_bin']['30-45'], "higher=better"),
        ("recall 45+", lambda r: r['rec_bin']['45+'], "higher=better"),
        ("frac IoU>=0.7 (kept)", lambda r: r['frac_iou07'], "higher=better"),
        ("frac IoU0.7 of TP", lambda r: r['frac07_of_tp'], "higher=better"),
        ("rho(V,IoU) rank", lambda r: r['rho'], "higher=better"),
        ("mean|dz| depth", lambda r: r['mean_dz'], "LOWER=better"),
        ("|dz| 0-15", lambda r: r['dz_bin']['0-15'], "LOWER=better"),
        ("|dz| 15-30", lambda r: r['dz_bin']['15-30'], "LOWER=better"),
        ("|dz| 30-45", lambda r: r['dz_bin']['30-45'], "LOWER=better"),
        ("|dz| 45+", lambda r: r['dz_bin']['45+'], "LOWER=better"),
        ("|dx| lateral", lambda r: r['mean_dx'], "LOWER=better"),
        ("dim err", lambda r: r['mean_dim'], "LOWER=better"),
        ("yaw err", lambda r: r['mean_yaw'], "LOWER=better"),
        ("dup% (pre-NMS)", lambda r: r['dup_frac'], "?"),
    ]
    emit(f"{'metric':22s} {'spearman_vs_AP':>15} {'dir':>13}   values weak->strong")
    emit("-" * 100)
    for mname, getter, direction in metrics:
        rho, vals = mono(mname, getter)
        vs = " ".join(fmt(v, 3) for v in vals)
        emit(f"{mname:22s} {rho:15.3f} {direction:>13}   {vs}")

    emit("")
    emit("READ: |spearman|>=0.85 = (near-)monotone with AP. near 0 or wrong sign = FLAT/non-monotone.")

    OUT = out_path("probe_orig.txt")
    with open(OUT, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")
    emit("")
    emit("[written] " + OUT)


if __name__ == "__main__":
    main()

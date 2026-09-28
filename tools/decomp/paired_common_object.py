"""Produces reports/paired_common_object.txt.

PAIRED COMMON-OBJECT decomposition (val dumps only, NO retraining).

Mono3D diagnostic. Across detectors of increasing AP, on the SAME
(commonly-detected) moderate Car GTs, decompose whether the AP improvement is
PRECISION (boxes get geometrically tighter = lower depth error / higher 3D-IoU)
or RANKING (boxes scored better relative to the detector's own FPs, same geometry).
This removes the "different detectors match different boxes" confound by fixing
the object set to GTs that EVERY detector in the set detected (best 3D-IoU>=0.3).

REUSES machinery from probe_detector_progression.py:
  S5 kept pool = cls>=0.2 + greedy 2D-NMS@0.5 by native V.
  arc.poly / arc.iou3d / arc.read_gt 3D-IoU matcher.
  moderate Car GT = occ<=1 & trunc<=0.5. distance bins.

STEP 1 (detection matrix): per moderate Car GT (keyed sid+gt position), per
detector record: detected(best3D-IoU>=0.3), best-IoU, best-box |dz|, best-box V,
best-box score-percentile (percentile of that box's V within the detector pool).

STEP 2 (COMMON = GTs detected by ALL detectors in the set):
  (a) PRECISION: median best-IoU, median |dz| per detector (paired, confound-free).
  (b) frac of common objects whose best box clears IoU>=0.7 per detector.
  (c) RANK: Spearman(V, best-IoU) per detector; median score-percentile of the
      common good boxes per detector (does it rank these higher vs its FPs?).

STEP 3 (DIFFERENTIAL = detected by NEWEST not OLDEST): count + dist/occ breakdown.
Run from the repository root: python tools/decomp/paired_common_object.py
"""
import os, sys, math
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
DET_IOU = 0.3  # "found at all"

# chronological / AP order
CORE = [
    ("gupnet",   "gupnet",            16.5),
    ("monodetr", "monodetr",          21.0),
    ("dgp",      "dgp",               22.3),
    ("monocop",  "official_monocop",  23.9),
]
# optional extras (front=monoflex, end=monoia); included in detection matrix so an
# extended common-set / differential can also be reported.
EXTRA_FRONT = [("monoflex", "monoflex", 15.6)]
EXTRA_END   = [("monoia",   "monoia",   24.5)]


def dbin(z):
    return "0-15" if z < 15 else "15-30" if z < 30 else "30-45" if z < 45 else "45+"


def read_gt_mod_full(sid):
    """Moderate Car GTs (occ<=1 & trunc<=0.5). Returns list of dicts with geom + occ/trunc."""
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
        out.append(dict(g=(float(t[8]), float(t[9]), float(t[10]), float(t[11]),
                           float(t[12]), float(t[13]), float(t[14])),  # h w l x y z ry
                        occ=occ, trunc=trunc))
    return out


def build_detection_matrix(name, csvfile, val_sids):
    """For one detector: S5 kept pool, then per moderate Car GT capture best-IoU
    box's metrics. Returns dict keyed (sid, gt_pos) -> dict(found,iou,dz,V,pct,bin,z).
    pct = percentile rank of best box's V within this detector's kept pool (0..1)."""
    va = pd.read_csv(dump_path(csvfile)).reset_index(drop=True)
    prepool = va[pool_mask(va, "thr0.2")].copy()
    keep = apply_nms(prepool, prepool["V"].values, 0.5)
    kept = prepool[keep].copy().reset_index(drop=True)
    n_kept = len(kept)

    # global percentile mapping of V across the whole kept pool (FPs included)
    Vall = kept["V"].values
    order = np.argsort(np.argsort(Vall))           # ranks 0..n-1
    pct_all = (order + 1) / max(1, len(Vall))      # in (0,1], higher V -> higher pct

    grp = {s: (g, pct_all[kept.index.get_indexer(g.index)]) for s, g in kept.groupby("sid")}
    # NOTE: kept reset_index(drop=True) so kept.index is RangeIndex; use positional.
    grp = {}
    for s, g in kept.groupby("sid"):
        pos = g.index.values  # positional in kept
        grp[s] = (g, pct_all[pos])

    mat = {}
    for sid in val_sids:
        gts = read_gt_mod_full(sid)
        if not gts:
            continue
        g = pct = None
        if sid in grp:
            g, pct = grp[sid]
            px = g.x_3d.values; py = g.y_3d.values; pz = g.z_3d.values
            ph = g.h_3d.values; pw = g.w_3d.values; pl = g.l_3d.values; pry = g.ry.values
            V = g.V.values
        for gi, rec in enumerate(gts):
            t = rec["g"]; b = dbin(t[5])
            best = 0.0; bk = -1
            if g is not None:
                gp = arc.poly(t[3], t[5], t[1], t[2], t[6])
                cand = np.where(np.abs(pz - t[5]) < 8.0)[0]
                for k in cand:
                    io = arc.iou3d(arc.poly(px[k], pz[k], pw[k], pl[k], pry[k]),
                                   py[k], ph[k], pl[k], pw[k], t, gp)
                    if io > best:
                        best = io; bk = k
            d = dict(found=(best >= DET_IOU), iou=best, bin=b, gtz=t[5],
                     occ=rec["occ"], trunc=rec["trunc"], dz=np.nan, V=np.nan, pct=np.nan)
            if bk >= 0:
                d["dz"] = abs(pz[bk] - t[5])
                d["V"] = float(V[bk])
                d["pct"] = float(pct[bk])
            mat[(sid, gi)] = d
    return mat, n_kept


def median(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    return float(np.median(v)) if v.size else float("nan")


def fmt(x, n=3):
    return "   NA" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{n}f}"


def main():
    val_sids = [int(x) for x in open(VAL).read().split()]
    out_lines = []

    def emit(s=""):
        print(s, flush=True); out_lines.append(s)

    ALLDETS = EXTRA_FRONT + CORE + EXTRA_END
    emit("=" * 118)
    emit("PAIRED COMMON-OBJECT DECOMPOSITION  (val dumps only, NO retraining)")
    emit("S5 kept pool = cls>=0.2 + greedy 2D-NMS@0.5 by native V; arc 3D-IoU matcher; moderate Car GT (occ<=1 & trunc<=0.5)")
    emit(f"per moderate GT -> best 3D-IoU box over kept preds; detected = best-IoU>={DET_IOU}")
    emit("PRECISION = best-IoU / |dz| (geometry of SAME box).  RANK = Spearman(V,IoU) + score-percentile of good box vs FPs.")
    emit("=" * 118)

    # ---------- STEP 1: detection matrix for every detector ----------
    mats = {}; nkept = {}
    for name, f, ap in ALLDETS:
        m, nk = build_detection_matrix(name, f, val_sids)
        mats[name] = m; nkept[name] = nk
        nfound = sum(1 for d in m.values() if d["found"])
        emit(f"[matrix] {name:9s} ap={ap:5.2f} n_kept={nk:6d} mod_GTs={len(m):5d} found(IoU>={DET_IOU})={nfound:5d} ({nfound/max(1,len(m)):.3f})")

    # universe of GT keys = union across detectors (all share the same GT enumeration)
    gt_keys = set()
    for m in mats.values():
        gt_keys |= set(m.keys())
    gt_keys = sorted(gt_keys)
    emit(f"[universe] total distinct moderate Car GT slots = {len(gt_keys)}")

    def common_set(det_names):
        """GT keys found (IoU>=DET_IOU) by ALL detectors in det_names."""
        out = []
        for k in gt_keys:
            ok = True
            for nm in det_names:
                d = mats[nm].get(k)
                if d is None or not d["found"]:
                    ok = False; break
            if ok:
                out.append(k)
        return out

    def report_common(label, det_names, ap_map):
        common = common_set(det_names)
        emit("")
        emit("#" * 118)
        emit(f"{label}: COMMON objects detected by ALL of {det_names}")
        emit(f"   common-set size N = {len(common)} moderate Car GTs (same objects for every row)")
        emit("#" * 118)
        if not common:
            emit("   (empty common set — SKIP)")
            return common
        hdr = (f"{'detector':9s} {'baseAP':>6} | {'medIoU':>7} {'med|dz|':>8} "
               f"{'frac>=.7':>9} | {'rho(V,IoU)':>11} {'medGoodPct':>11}")
        emit(hdr); emit("-" * len(hdr))
        for nm in det_names:
            m = mats[nm]
            ious = [m[k]["iou"] for k in common]
            dzs  = [m[k]["dz"]  for k in common]
            pcts = [m[k]["pct"] for k in common]
            Vs   = [m[k]["V"]   for k in common]
            med_iou = median(ious)
            med_dz = median(dzs)
            frac07 = float(np.mean([1.0 if (i is not None and i >= 0.7) else 0.0 for i in ious]))
            # rank signal: Spearman(V, best-IoU) on the common good boxes
            vv = np.array(Vs, float); ii = np.array(ious, float)
            ok = ~np.isnan(vv) & ~np.isnan(ii)
            rho = spearmanr(vv[ok], ii[ok]).correlation if ok.sum() > 5 else float("nan")
            med_pct = median(pcts)
            emit(f"{nm:9s} {ap_map[nm]:6.2f} | {fmt(med_iou):>7} {fmt(med_dz):>8} "
                 f"{frac07:9.3f} | {fmt(rho):>11} {fmt(med_pct):>11}")
        # decomposition deltas oldest->newest on this common set
        old, new = det_names[0], det_names[-1]
        mo, mn = mats[old], mats[new]
        d_iou = median([mn[k]["iou"] for k in common]) - median([mo[k]["iou"] for k in common])
        d_dz  = median([mn[k]["dz"]  for k in common]) - median([mo[k]["dz"]  for k in common])
        f07o = float(np.mean([1.0 if mo[k]["iou"] >= 0.7 else 0.0 for k in common]))
        f07n = float(np.mean([1.0 if mn[k]["iou"] >= 0.7 else 0.0 for k in common]))
        po = median([mo[k]["pct"] for k in common]); pn = median([mn[k]["pct"] for k in common])
        emit("-" * len(hdr))
        emit(f"  Δ ({old}->{new}) on SAME {len(common)} objects:")
        emit(f"    PRECISION: med-IoU {fmt(d_iou,3)} (+=tighter)   med|dz| {fmt(d_dz,3)} (-=less depth err)   frac>=.7 {fmt(f07n-f07o,3)}")
        emit(f"    RANK:      med good-box score-pct {fmt(pn-po,3)} (+=ranked higher vs own FPs)")
        return common

    ap_map = {nm: ap for nm, f, ap in ALLDETS}

    # ---------- STEP 2: consecutive core pairs + all-4 core common ----------
    core_names = [nm for nm, f, ap in CORE]
    for a, b in zip(core_names[:-1], core_names[1:]):
        report_common(f"PAIR {a} -> {b}", [a, b], ap_map)

    common4 = report_common("ALL-4 CORE", core_names, ap_map)

    # optional extended (monoflex .. monoia) all-6 common
    all6 = [nm for nm, f, ap in ALLDETS]
    report_common("ALL-6 (incl monoflex+monoia)", all6, ap_map)

    # ---------- STEP 3: DIFFERENTIAL objects (newest-not-oldest) ----------
    def differential(old, new):
        diff = []
        for k in gt_keys:
            do, dn = mats[old].get(k), mats[new].get(k)
            if do is None or dn is None:
                continue
            if dn["found"] and not do["found"]:
                diff.append(k)
        return diff

    emit("")
    emit("#" * 118)
    emit("STEP 3 — DIFFERENTIAL (coverage gain): GTs detected by NEWEST but NOT OLDEST")
    emit("#" * 118)
    for old, new in [("gupnet", "monocop"), ("gupnet", "monodetr"),
                     ("monodetr", "dgp"), ("dgp", "monocop")]:
        diff = differential(old, new)
        # also reverse (lost coverage) for honesty
        lost = differential(new, old)
        binc = {b: 0 for b in BINS}; occ = {0: 0, 1: 0}
        for k in diff:
            d = mats[new][k]; binc[d["bin"]] += 1; occ[d["occ"]] = occ.get(d["occ"], 0) + 1
        emit("")
        emit(f"{old} -> {new}:  gained={len(diff)}   (lost: {new} found but {old} not = {len(lost)})")
        emit(f"   dist bins: " + "  ".join(f"{b}={binc[b]}" for b in BINS))
        emit(f"   occlusion: " + "  ".join(f"occ{o}={c}" for o, c in sorted(occ.items())))

    # ---------- CLEAN SUMMARY TABLE on ALL-4 common ----------
    emit("")
    emit("#" * 118)
    emit("CLEAN TABLE — detector x metrics on the ALL-4-CORE COMMON object set (same objects everyone)")
    emit("#" * 118)
    if common4:
        hdr = (f"{'detector':9s} {'baseAP':>6} {'medIoU':>7} {'med|dz|':>8} "
               f"{'frac>=.7':>9} {'rho(V,IoU)':>11} {'medGoodPct':>11}")
        emit(hdr); emit("-" * len(hdr))
        rowvals = {}
        for nm in core_names:
            m = mats[nm]
            ious = [m[k]["iou"] for k in common4]
            dzs  = [m[k]["dz"]  for k in common4]
            pcts = [m[k]["pct"] for k in common4]
            Vs   = [m[k]["V"]   for k in common4]
            med_iou = median(ious); med_dz = median(dzs); med_pct = median(pcts)
            frac07 = float(np.mean([1.0 if i >= 0.7 else 0.0 for i in ious]))
            vv = np.array(Vs, float); ii = np.array(ious, float); ok = ~np.isnan(vv) & ~np.isnan(ii)
            rho = spearmanr(vv[ok], ii[ok]).correlation if ok.sum() > 5 else float("nan")
            rowvals[nm] = dict(iou=med_iou, dz=med_dz, f07=frac07, rho=rho, pct=med_pct)
            emit(f"{nm:9s} {ap_map[nm]:6.2f} {fmt(med_iou):>7} {fmt(med_dz):>8} "
                 f"{frac07:9.3f} {fmt(rho):>11} {fmt(med_pct):>11}")

        # ---------- VERDICT ----------
        ap_arr = np.array([ap_map[nm] for nm in core_names], float)
        def rho_vs_ap(key, invert=False):
            vals = np.array([rowvals[nm][key] for nm in core_names], float)
            ok = ~np.isnan(vals)
            if ok.sum() < 3:
                return float("nan")
            return spearmanr(ap_arr[ok], vals[ok]).correlation
        r_iou = rho_vs_ap("iou")
        r_dz  = rho_vs_ap("dz")    # want NEGATIVE (lower=better)
        r_f07 = rho_vs_ap("f07")
        r_pct = rho_vs_ap("pct")
        r_rho = rho_vs_ap("rho")
        o, n = core_names[0], core_names[-1]
        span_iou = rowvals[n]["iou"] - rowvals[o]["iou"]
        span_dz  = rowvals[n]["dz"]  - rowvals[o]["dz"]
        span_f07 = rowvals[n]["f07"] - rowvals[o]["f07"]
        span_pct = rowvals[n]["pct"] - rowvals[o]["pct"]
        emit("")
        emit("VERDICT INPUTS (spearman of metric vs AP across the 4 core detectors, on the fixed common set):")
        emit(f"  PRECISION  med-IoU vs AP   = {fmt(r_iou)}   (gupnet->monocop span {fmt(span_iou)}, +=tighter)")
        emit(f"  PRECISION  med|dz| vs AP   = {fmt(r_dz)}   (span {fmt(span_dz)}, want NEGATIVE rho / NEGATIVE span)")
        emit(f"  PRECISION  frac>=0.7 vs AP = {fmt(r_f07)}   (span {fmt(span_f07)}, +=more clear 0.7)")
        emit(f"  RANK       good-box pct vs AP = {fmt(r_pct)}   (span {fmt(span_pct)}, +=ranked higher vs FPs)")
        emit(f"  RANK       Spearman(V,IoU) vs AP = {fmt(r_rho)}")

        # decide dominance: precision rises if IoU up & |dz| down & frac07 up
        prec_rises = (r_iou > 0.6) or (r_dz < -0.6) or (r_f07 > 0.6)
        rank_rises = (r_pct > 0.6) or (r_rho > 0.6)
        # magnitude comparison via normalized spans (rough): precision strength vs rank strength
        prec_strength = max(abs(span_iou) / 0.05 if not math.isnan(span_iou) else 0,
                            abs(span_dz) / 0.10 if not math.isnan(span_dz) else 0,
                            abs(span_f07) / 0.05 if not math.isnan(span_f07) else 0)
        rank_strength = abs(span_pct) / 0.05 if not math.isnan(span_pct) else 0
        if prec_rises and rank_rises:
            both = "both rise"
            dom = "precision" if prec_strength >= rank_strength else "rank"
        elif prec_rises:
            both = "only precision rises"; dom = "precision"
        elif rank_rises:
            both = "only rank rises"; dom = "rank"
        else:
            both = "neither clearly rises"; dom = "neither"
        emit("")
        emit("=" * 118)
        emit("ONE-LINE VERDICT")
        emit("=" * 118)
        nums = (f"med-IoU {fmt(rowvals[o]['iou'],3)}->{fmt(rowvals[n]['iou'],3)}, "
                f"med|dz| {fmt(rowvals[o]['dz'],3)}->{fmt(rowvals[n]['dz'],3)}, "
                f"frac>=.7 {fmt(rowvals[o]['f07'],3)}->{fmt(rowvals[n]['f07'],3)}, "
                f"good-box pct {fmt(rowvals[o]['pct'],3)}->{fmt(rowvals[n]['pct'],3)}")
        emit(f"On the same {len(common4)} objects, the improvement is DOMINANTLY {dom}, with {both}: [{nums}]")

    OUT = out_path("paired_common_object.txt")
    with open(OUT, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")
    emit("")
    emit(f"[written] {OUT}")


if __name__ == "__main__":
    main()

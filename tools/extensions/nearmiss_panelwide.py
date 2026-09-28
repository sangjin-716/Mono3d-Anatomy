"""Ported from the camera-ready check nearmiss_panelwide.py (same file name) for the public release.
Computation unchanged. Produces reports/extensions/nearmiss_panelwide.txt (a re-run writes
reports_rerun/extensions/nearmiss_panelwide.{txt,csv}); supplementary Sec. M, Table M, and main
Sec. 5.1 "Coverage in Fixed-Pool Accounting". A re-run also prints the twelve-detector union rows
and their two gate lines, which the public copy omits because the paper does not report them.

Panel-wide near-miss decomposition.

QUESTION:
  For each of the 12 detectors, among moderate Car GTs that have NO prediction at
  IoU3D >= 0.7 in the FINAL output pool, how close does the best prediction get?
    [0.5, 0.7)  near miss     -> "tighten an existing candidate"
    [0.3, 0.5)  loose candidate
    (0.0, 0.3)  weak
    == 0.0      no overlapping candidate at all -> "propose a new candidate"

NO NEW IoU CODE IS WRITTEN HERE.  The per-GT best-IoU3D floats are read straight out of
the paper's own FROZEN artifact

    gt_state_matrix.csv   (default: cache/decomp/gt_state_matrix.csv; override with --gsm PATH)

which was produced by tools/decomp/gt_state_matrix.py, whose moderate-GT filter, |dz|<8 m
candidate gate and arc.iou3d kernel are (per its own docstring) verbatim copies of the
frozen tools/decomp/pool_waterfall.py used for the shipped paper.  Columns:
    {det}_a = best IoU3D over the COMPLETE native pool
    {det}_c = best IoU3D over the FINAL output pool (native eligibility survivors)
This script only RE-BINS those floats.  It computes no geometry of its own.

REPRODUCTION GATES (all must PASS, else the script prints FAILED and refuses to report):
  G1  n_gt == 7874 and distance-bin counts match the frozen validation
  G2  per-detector A@0.7 / C@0.7 / face-ii reproduce reports/pool_waterfall.txt to 3 dp
  G3  per-detector A@0.5 reproduces reports/pool_waterfall.txt to 3 dp
  G4  per-detector "30 m+ missed GTs, of which NO pool candidate IoU>=0.5" reproduces
      reports/pool_waterfall.txt exactly (count and share)
  G5  panel union: 1176 GTs with no _a>=0.7 anywhere, 194 with no _a>=0.5 anywhere, and
      2002 with no _a>=0.7 over the eleven non-anchor pools (reports/gt_state_matrix.txt,
      reports/final_run/e2_budget_truncation.txt; modern-environment MonoFlex/MonoGround)
  G6/G7 the new [0.5,0.7) bin against the frozen panel-union histograms of
      reports/gt_state_matrix.txt at IoU 0.7 and 0.5

MonoFlex and MonoGround enter through their MODERN-environment dumps (released as
monoflex_modern_val.csv / monoground_modern_val.csv, stems monoflex / monoground; labelled
monoflex_val.csv / monoground_val.csv in the report), the same dumps behind
reports/gt_state_matrix.txt and reports/pool_waterfall.txt.

ZERO-BIN SAFETY:  the frozen kernel only evaluates candidates with |z_pred - z_gt| < 8 m.
A stored 0.0 therefore means "no candidate with IoU3D > 0 *inside that gate*".  Part 0
checks, from the raw label files and the raw dumps, that no GT/prediction Car box pair can
overlap at all once |dz| >= 8 m (max half-extent along z of a box is
sqrt((l/2)^2+(w/2)^2)); if gt_max_half + pred_max_half < 8 m for every dump the 0.0 bin is
EXACT, not censored.  Part 0 can be skipped with --skip-dimcheck (it re-reads ~4 GB).

Usage (from the repository root):
  python tools/extensions/nearmiss_panelwide.py [--skip-dimcheck] [--gsm PATH]
Writes: reports_rerun/extensions/nearmiss_panelwide.txt
        reports_rerun/extensions/nearmiss_panelwide.csv
Reads everything else READ-ONLY.  Part 0 reads the released dumps and the auxiliary
complete-pool dumps (see tools/extensions/_ext.py).
"""
import os
import re
import sys
import glob
import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ext import paths, dump_path, cache_dir, out_path, aux_dump_path, repo_rel, ROOT  # noqa: E402

GSM = os.path.join(paths.CACHE_DIR, "decomp", "gt_state_matrix.csv")
if "--gsm" in sys.argv:
    GSM = sys.argv[sys.argv.index("--gsm") + 1]
WATERFALL = os.path.join(ROOT, "reports", "pool_waterfall.txt")
LABEL_DIR = paths.LABEL_DIR
VAL_LIST = paths.VAL_LIST
OUT = out_path("extensions/nearmiss_panelwide.txt")
OUTCSV = out_path("extensions/nearmiss_panelwide.csv")

MODELS = ["M3D-RPN", "MonoDLE", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround", "MonoCon",
          "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]
FAM = {"M3D-RPN": "anchor", "MonoDETR": "query", "MonoDGP": "query", "MonoCoP": "query",
       "MonoCLUE": "query", "MonoIA": "query"}
# dump file used by the frozen gt_state_matrix.py stages_for() for each detector
DUMP = {
    "M3D-RPN": "m3drpn_val_floor0.csv",
    "MonoDLE": "monodle_val.csv", "MonoFlex": "monoflex_val.csv",
    "GUPNet": "gupnet_val.csv", "DEVIANT": "deviant_val.csv",
    "MonoGround": "monoground_val.csv", "MonoCon": "monocon_val.csv",
    "MonoDETR": "monodetr_val_preflatten.csv", "MonoDGP": "monodgp_val_preflatten.csv",
    "MonoCoP": "official_monocop_val_preflatten.csv",
    "MonoCLUE": "monoclue_val_preflatten.csv", "MonoIA": "monoia_val_preflatten.csv",
}
# released stem of each non-auxiliary dump above (modern-environment MonoFlex / MonoGround)
STEM = {"MonoDLE": "monodle", "MonoFlex": "monoflex", "GUPNet": "gupnet", "DEVIANT": "deviant",
        "MonoGround": "monoground", "MonoCon": "monocon"}


def dump_file(m):
    return dump_path(STEM[m]) if m in STEM else aux_dump_path(DUMP[m])


BINS = [(0, 15), (15, 30), (30, 45), (45, 1e9)]
BINLAB = ["0-15", "15-30", "30-45", "45+"]

out = []


def w(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


def flush(status):
    hdr = [f"# nearmiss_panelwide.py  run {datetime.datetime.now().isoformat(timespec='seconds')}",
           f"# STATUS: {status}", ""]
    open(OUT, "w").write("\n".join(hdr + out) + "\n")


def die(msg):
    w("")
    w("!!!! " + msg)
    w("!!!! FAILED — no near-miss numbers are reported.")
    flush("FAILED")
    sys.exit(1)


# ----------------------------------------------------------------------------------
# PART 0 — zero-bin safety: is the frozen |dz| < 8 m gate exact-zero-safe?
# ----------------------------------------------------------------------------------
def part0_dimcheck():
    w("=" * 100)
    w("PART 0 — ZERO-BIN SAFETY CHECK (is the frozen |dz|<8m candidate gate exact-zero-safe?)")
    w("=" * 100)
    w("A 3D box's half-extent along the z axis is at most sqrt((l/2)^2 + (w/2)^2).")
    w("Two boxes can overlap only if |dz| < half_gt + half_pred.  If that sum < 8 m for every")
    w("GT/prediction pair, then a stored best-IoU of exactly 0.0 means genuinely NO overlapping")
    w("candidate anywhere in the pool, not a candidate hidden outside the gate.")
    w("")
    val = [int(x) for x in open(VAL_LIST).read().split()]
    gmax = 0.0
    for sid in val:
        p = os.path.join(LABEL_DIR, f"{sid:06d}.txt")
        if not os.path.exists(p):
            continue
        for ln in open(p):
            t = ln.split()
            if t[0] != "Car":
                continue
            trunc, occ = float(t[1]), int(t[2])
            hpix = float(t[7]) - float(t[5])
            if occ <= 1 and trunc <= 0.3 and hpix > 25:
                ww, ll = float(t[9]), float(t[10])
                gmax = max(gmax, np.hypot(ll / 2.0, ww / 2.0))
    w(f"GT (moderate Car, val)      max half-z-extent = {gmax:.3f} m")
    worst = 0.0
    for m in MODELS:
        f = dump_file(m)
        pmax = 0.0
        for chunk in pd.read_csv(f, usecols=["w_3d", "l_3d"], chunksize=2_000_000):
            v = np.hypot(chunk.l_3d.values / 2.0, chunk.w_3d.values / 2.0)
            if len(v):
                pmax = max(pmax, float(np.nanmax(v)))
        tot = gmax + pmax
        worst = max(worst, tot)
        w(f"  {m:11s} {DUMP[m]:38s} max pred half-z-extent = {pmax:6.3f} m   "
          f"gt+pred = {tot:6.3f} m   {'SAFE' if tot < 8.0 else 'NOT SAFE'}")
    w("")
    if worst < 8.0:
        w(f"=> worst-case gt+pred half-extent = {worst:.3f} m  <  8 m gate.  ZERO BIN IS EXACT.")
    else:
        w(f"=> worst-case gt+pred half-extent = {worst:.3f} m  >= 8 m gate.")
        w("=> The exactly-0 bin would be CENSORED.  Reporting the 0 bin separately is NOT safe;")
        w("   it must be merged into '<0.3'.")
    w("")
    return worst < 8.0


# ----------------------------------------------------------------------------------
# frozen reference parsing
# ----------------------------------------------------------------------------------
def parse_waterfall():
    ref, cur = {}, None
    for ln in open(WATERFALL):
        m = re.match(r"\[done\] (\S+)\s+\(\w+\) GT=(\d+) \| A@0\.5=([\d.]+) A@0\.7=([\d.]+) "
                     r"B@0\.7=[\d.]+ C@0\.7=([\d.]+) \| suppressed-acc\(face-ii share\)=([\d.]+)", ln)
        if m:
            cur = m.group(1)
            ref[cur] = {"ngt": int(m.group(2)), "A5": float(m.group(3)),
                        "A7": float(m.group(4)), "C7": float(m.group(5)),
                        "fii": float(m.group(6))}
            continue
        m = re.match(r"\s+30m\+ missed GTs=(\d+), of which NO pool candidate IoU>=0\.5: "
                     r"(\d+) \(([\d.]+)\)", ln)
        if m and cur:
            ref[cur]["miss30"] = int(m.group(1))
            ref[cur]["noc30"] = int(m.group(2))
            ref[cur]["noc30_share"] = float(m.group(3))
    return ref


# ----------------------------------------------------------------------------------
def main():
    do_dim = "--skip-dimcheck" not in sys.argv
    zero_exact = None
    if do_dim:
        zero_exact = part0_dimcheck()
    else:
        w("PART 0 skipped (--skip-dimcheck).  The exactly-0 bin is reported but its exactness")
        w("was not re-verified in this run.")
        w("")

    df = pd.read_csv(GSM)
    ref = parse_waterfall()

    # ------------------------------------------------------------------ gates
    w("=" * 100)
    w("PART 1 — REPRODUCTION GATES against the FROZEN paper artifacts")
    w("=" * 100)
    w(f"source of best-IoU floats : {repo_rel(GSM)}")
    w(f"frozen reference          : {repo_rel(WATERFALL)}")
    w("")
    ok_all = True

    # G1
    n = len(df)
    binidx = np.select([(df.z_gt >= lo) & (df.z_gt < hi) for lo, hi in BINS], BINLAB, "??")
    df["dist_bin"] = binidx
    counts = {b: int((df.dist_bin == b).sum()) for b in BINLAB}
    exp_counts = {"0-15": 1930, "15-30": 3163, "30-45": 2416, "45+": 365}
    r1 = (n == 7874) and counts == exp_counts
    ok_all &= r1
    w(f"G1  n_gt = {n} (expect 7874); bins {counts} (expect {exp_counts})  -> {'PASS' if r1 else 'FAIL'}")

    # G2 / G3 / G4
    w("")
    w(f"{'detector':11s} {'A@0.5 mine/frozen':>22s} {'A@0.7 mine/frozen':>22s} "
      f"{'C@0.7 mine/frozen':>22s} {'faceii mine/frozen':>22s} {'30m+ no>=0.5 mine/frozen':>28s}  gate")
    for m in MODELS:
        a = df[f"{m}_a"].values
        c = df[f"{m}_c"].values
        A5, A7, C7 = (a >= 0.5).mean(), (a >= 0.7).mean(), (c >= 0.7).mean()
        nA = int((a >= 0.7).sum())
        fii = ((a >= 0.7) & (c < 0.7)).sum() / max(nA, 1)
        far = df.dist_bin.isin(["30-45", "45+"]).values
        miss30 = int((far & (c < 0.7)).sum())
        noc30 = int((far & (a < 0.5)).sum())
        r = ref[m]
        good = (round(A5, 3) == r["A5"] and round(A7, 3) == r["A7"] and
                round(C7, 3) == r["C7"] and round(fii, 3) == r["fii"] and
                miss30 == r["miss30"] and noc30 == r["noc30"] and
                round(noc30 / max(miss30, 1), 2) == r["noc30_share"])
        ok_all &= good
        w(f"{m:11s} {A5:11.3f}/{r['A5']:<10.3f} {A7:11.3f}/{r['A7']:<10.3f} "
          f"{C7:11.3f}/{r['C7']:<10.3f} {fii:11.3f}/{r['fii']:<10.3f} "
          f"{noc30:6d}/{miss30:<6d} vs {r['noc30']}/{r['miss30']:<6d}  "
          f"{'PASS' if good else 'FAIL'}")

    # G5 — panel union figures, checked against the FROZEN artifacts
    #      reports/gt_state_matrix.txt:41,54 and reports/final_run/e2_budget_truncation.txt:10
    w("")
    A7m = np.stack([(df[f"{m}_a"] >= 0.7).values for m in MODELS], 1)
    A5m = np.stack([(df[f"{m}_a"] >= 0.5).values for m in MODELS], 1)
    nonanchor = [i for i, m in enumerate(MODELS) if m != "M3D-RPN"]
    u07 = int((A7m.sum(1) == 0).sum())
    u05 = int((A5m.sum(1) == 0).sum())
    u07na = int((A7m[:, nonanchor].sum(1) == 0).sum())
    r5 = (u07 == 1176 and round(u07 / n, 3) == 0.149 and
          u05 == 194 and round(u05 / n, 3) == 0.025 and
          u07na == 2002 and round(u07na / n, 3) == 0.254)
    ok_all &= r5
    w(f"G5  no _a>=0.7 in ANY of 12 pools  : {u07}/{n} = {u07/n:.3f}   "
      f"frozen reports/gt_state_matrix.txt:41 = 1176/7874 = 0.149")
    w(f"    no _a>=0.5 in ANY of 12 pools  : {u05}/{n} = {u05/n:.3f}   "
      f"frozen reports/gt_state_matrix.txt:54 =  194/7874 = 0.025")
    w(f"    no _a>=0.7 over 11 non-anchor  : {u07na}/{n} = {u07na/n:.3f}   "
      f"frozen reports/final_run/e2_budget_truncation.txt:10 = 2002/7874 = 0.254")
    w(f"    -> {'PASS' if r5 else 'FAIL'}")
    # (These are modern-environment shares. The panel-union shares printed in supplementary
    #  Sec. J use the original-environment MonoFlex*/MonoGround* dumps instead.)

    # G6/G7 — the strongest available cross-check ON THE NEW BIN ITSELF.
    # reports/gt_state_matrix.txt prints the panel-union "k models with an accurate box"
    # histograms at BOTH IoU 0.7 and IoU 0.5.  The k=0 entries are:
    #   final pool     0.7 -> 2492   0.5 -> 1131      (lines 40 and 52)
    #   complete pool  0.7 -> 1176   0.5 ->  194      (lines 38 and 50)
    # A union GT that is uncovered at 0.7 but lands in the new [0.5,0.7) near-miss bin is
    # exactly a GT that IS covered at 0.5.  So  k0(0.7) - nearmiss_[0.5,0.7)  MUST equal
    # k0(0.5) for both pools.  This tests the new bin against an independently printed
    # frozen number, not just the >=0.7 threshold the matrix was validated at.
    w("")
    for suff, k0_07, k0_05, tag, ln in (("_c", 2492, 1131, "FINAL", "40 / 52"),
                                        ("_a", 1176, 194, "COMPLETE", "38 / 50")):
        U = np.stack([df[f"{m}{suff}"].values for m in MODELS], 1).max(1)
        unc07 = int((U < 0.7).sum())
        nm = int(((U >= 0.5) & (U < 0.7)).sum())
        implied = unc07 - nm
        good = (unc07 == k0_07 and implied == k0_05)
        ok_all &= good
        w(f"G{'6' if suff=='_c' else '7'}  {tag:8s} union uncovered@0.7 = {unc07} (frozen {k0_07}); "
          f"minus near-miss[0.5,0.7)={nm} -> implied uncovered@0.5 = {implied} "
          f"(frozen {k0_05}, gt_state_matrix.txt line {ln})  -> {'PASS' if good else 'FAIL'}")

    w("")
    if not ok_all:
        die("REPRODUCTION GATE FAILURE — the pipeline does not reproduce the frozen numbers.")
    w("ALL REPRODUCTION GATES PASS.  The near-miss decomposition below is a pure re-binning of")
    w("the same per-GT best-IoU3D floats that produced the shipped Table/Figure numbers.")
    w("")

    # ------------------------------------------------------- near-miss decomposition
    def decomp(v, mask):
        """v = best IoU3D per GT; mask = the GT subset (uncovered). returns counts dict."""
        x = v[mask]
        return {
            "n": int(mask.sum()),
            "nm_050_070": int(((x >= 0.5) & (x < 0.7)).sum()),
            "lo_030_050": int(((x >= 0.3) & (x < 0.5)).sum()),
            "wk_000_030": int(((x > 0.0) & (x < 0.3)).sum()),
            "zero": int((x == 0.0).sum()),
            "median": float(np.median(x)) if len(x) else float("nan"),
        }

    rows = []
    for pool_tag, suff, pool_name in (("FINAL", "_c", "final output pool (native selection)"),
                                      ("COMPLETE", "_a", "complete native pool (pre-selection)")):
        w("=" * 100)
        w(f"PART 2{'a' if suff == '_c' else 'b'} — NEAR-MISS DECOMPOSITION, {pool_tag} POOL "
          f"({pool_name})")
        w("=" * 100)
        w("Rows: for each detector, the GTs with NO IoU3D>=0.7 box in this pool, split by the")
        w("BEST IoU3D that any of that detector's boxes achieves against the GT.")
        w("")
        w(f"{'detector':11s} {'fam':7s} {'n_unc':>6s} {'unc%':>6s} | {'[0.5,0.7)':>10s} "
          f"{'[0.3,0.5)':>10s} {'(0,0.3)':>10s} {'exactly 0':>10s} | {'med bestIoU':>11s}")
        for m in MODELS:
            v = df[f"{m}{suff}"].values
            mask = v < 0.7
            d = decomp(v, mask)
            nn = d["n"]
            w(f"{m:11s} {FAM.get(m,'cnet'):7s} {nn:6d} {100*nn/n:5.1f}% | "
              f"{d['nm_050_070']:5d} {100*d['nm_050_070']/nn:4.1f}% "
              f"{d['lo_030_050']:5d} {100*d['lo_030_050']/nn:4.1f}% "
              f"{d['wk_000_030']:5d} {100*d['wk_000_030']/nn:4.1f}% "
              f"{d['zero']:5d} {100*d['zero']/nn:4.1f}% | {d['median']:11.3f}")
            rows.append(dict(pool=pool_tag, detector=m, family=FAM.get(m, "cnet"),
                             dist_bin="ALL", **d))
        # union of 12
        U = np.stack([df[f"{m}{suff}"].values for m in MODELS], 1).max(1)
        maskU = U < 0.7
        dU = decomp(U, maskU)
        w("-" * 100)
        w(f"{'UNION-12':11s} {'-':7s} {dU['n']:6d} {100*dU['n']/n:5.1f}% | "
          f"{dU['nm_050_070']:5d} {100*dU['nm_050_070']/dU['n']:4.1f}% "
          f"{dU['lo_030_050']:5d} {100*dU['lo_030_050']/dU['n']:4.1f}% "
          f"{dU['wk_000_030']:5d} {100*dU['wk_000_030']/dU['n']:4.1f}% "
          f"{dU['zero']:5d} {100*dU['zero']/dU['n']:4.1f}% | {dU['median']:11.3f}")
        w("   (UNION-12 = best IoU3D over all twelve detectors' pools jointly; an upper bound on")
        w("    what the whole panel can localize, not any single deployable detector.)")
        rows.append(dict(pool=pool_tag, detector="UNION-12", family="-", dist_bin="ALL", **dU))
        w("")

        # per distance bin
        w(f"-- same decomposition, by GT distance bin ({pool_tag} pool) --")
        w(f"{'detector':11s} {'bin':7s} {'n_unc':>6s} | {'[0.5,0.7)':>10s} {'[0.3,0.5)':>10s} "
          f"{'(0,0.3)':>10s} {'exactly 0':>10s} | {'med':>7s}")
        for m in MODELS + ["UNION-12"]:
            v = (U if m == "UNION-12"
                 else df[f"{m}{suff}"].values)
            for b in BINLAB:
                sel = (df.dist_bin == b).values
                mask = (v < 0.7) & sel
                if mask.sum() == 0:
                    continue
                d = decomp(v, mask)
                nn = d["n"]
                w(f"{m:11s} {b:7s} {nn:6d} | "
                  f"{d['nm_050_070']:5d} {100*d['nm_050_070']/nn:4.1f}% "
                  f"{d['lo_030_050']:5d} {100*d['lo_030_050']/nn:4.1f}% "
                  f"{d['wk_000_030']:5d} {100*d['wk_000_030']/nn:4.1f}% "
                  f"{d['zero']:5d} {100*d['zero']/nn:4.1f}% | {d['median']:7.3f}")
                rows.append(dict(pool=pool_tag, detector=m,
                                 family=FAM.get(m, "cnet") if m != "UNION-12" else "-",
                                 dist_bin=b, **d))
            w("")
        w("")

    # ------------------------------------------------------------------ summary
    w("=" * 100)
    w("PART 3 — HEADLINE RANGES (final output pool; the pool the paper's 63-77% coverage")
    w("          statement is about)")
    w("=" * 100)
    R = pd.DataFrame(rows)
    F = R[(R.pool == "FINAL") & (R.dist_bin == "ALL") & (R.detector != "UNION-12")]
    for k, lab in (("nm_050_070", "[0.5,0.7) near miss"), ("lo_030_050", "[0.3,0.5)"),
                   ("wk_000_030", "(0,0.3)"), ("zero", "exactly 0 (no overlapping candidate)")):
        sh = F[k] / F.n
        lo_d = F.detector.values[sh.values.argmin()]
        hi_d = F.detector.values[sh.values.argmax()]
        w(f"  {lab:42s} {100*sh.min():5.1f}% ({lo_d}) .. {100*sh.max():5.1f}% ({hi_d})   "
          f"median {100*sh.median():5.1f}%")
    w(f"  median best achieved IoU3D over uncovered GTs   "
      f"{F['median'].min():.3f} .. {F['median'].max():.3f}")
    w("")
    w("PART 3b — the same shares in the far field (30 m+), FINAL pool")
    FF = R[(R.pool == "FINAL") & (R.dist_bin.isin(["30-45", "45+"])) & (R.detector != "UNION-12")]
    agg = FF.groupby("detector")[["n", "nm_050_070", "lo_030_050", "wk_000_030", "zero"]].sum()
    for k, lab in (("nm_050_070", "[0.5,0.7) near miss"), ("zero", "exactly 0")):
        sh = agg[k] / agg["n"]
        w(f"  30m+  {lab:36s} {100*sh.min():5.1f}% ({sh.idxmin()}) .. "
          f"{100*sh.max():5.1f}% ({sh.idxmax()})   median {100*sh.median():5.1f}%")
    w("")

    w("=" * 100)
    w("SCOPE / CAVEATS")
    w("=" * 100)
    w("* Universe: KITTI val (Chen split, 3769 images), official Moderate Car filter")
    w("  (occ<=1, trunc<=0.3, 2D box height>25 px), n_gt = 7874.  Same universe as the paper.")
    w("* 'best IoU3D' is a PRESENCE-layer quantity: the max IoU3D over all of a detector's")
    w("  boxes in the pool for that GT.  It is NOT the official score-ordered one-to-one TP")
    w("  assignment, and none of these numbers are AP.  No AP claim is made or implied.")
    w("* FINAL pool = the detector's own native output (the pool behind the paper's 63-77%")
    w("  coverage statement).  COMPLETE pool = the full pre-selection candidate set, which for")
    w("  the dense-anchor exemplar (M3D-RPN) is ~3000 anchors/image and is therefore NOT")
    w("  comparable in budget to the others.")
    w("* UNION-12 is an oracle over twelve detectors run jointly; it bounds the panel, not any")
    w("  single deployable model.")
    w("* The exactly-0 bin is exact only under the Part 0 dimension check "
      f"({'VERIFIED this run' if zero_exact else 'NOT verified this run' if zero_exact is not None else 'SKIPPED this run'}).")
    w("")

    R.to_csv(OUTCSV, index=False)
    w(f"[written] {os.path.basename(OUTCSV)}  rows={len(R)}")
    flush("COMPUTED")
    w(f"[written] {repo_rel(OUT)}")


if __name__ == "__main__":
    main()

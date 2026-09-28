"""Produces reports/extensions/difficulty_class_extension.txt and .csv (a re-run writes
reports_rerun/extensions/); supplementary Sec. O, Table O.

NOT RE-RUNNABLE FROM THE RELEASED DATA ALONE. Part B reads each detector's complete native
multi-class KITTI prediction directory (Car, Pedestrian, Cyclist), which is not released; the
released dumps are Car-only. Part A also reads the auxiliary complete-pool dumps
(tools/extensions/_ext.py). The native directories are expected under paths.UPSTREAM_ROOT
(see NATIVE_DIRS below); regenerate them with each upstream repository's own evaluation script.

DIFFICULTY + CLASS EXTENSION.
  Part A: Easy / Moderate / Hard, all 12 detectors (Car)
  Part B: Pedestrian / Cyclist, for the detectors that can emit them

PIPELINE PROVENANCE
-------------------
Part A is tools/decomp/exp1_true_ceiling.py (the AP* script behind
reports/exp1_true_ceiling.txt) with exactly two changes:
   (a) difficulty swept d in {0,1,2} instead of pinned to 1, in BOTH
       ap_summaries(..., difficulty=d) and _prepare_data(..., difficulty=d);
   (b) FP-demotion dropped (it needs the fixed Car/Moderate labels).
Pools are the NATIVE panel pools, copied from e4_fp_tp_decomp.native_pool -- the same
pools that produce the Table 1 base AP of the paper.

Part B evaluates the four panel detectors that have a COMPLETE native KITTI-format
prediction directory containing non-Car rows, using the official evaluator's own 3D IoU
both for AP and for the oracle re-sort score.

REPRODUCTION GATES (all must pass or the script aborts)
------------------------------------------------------
GATE-1  Moderate base all-point AP, 12/12, vs reports/e4_fp_tp_decomp.txt  (tol 0.05)
GATE-2  Moderate true-IoU re-sort gain, 12/12, vs reports/e4_fp_tp_decomp.txt "full"
GATE-3  Easy AND Hard base all-point AP vs the paper's Table 1 E/H columns, for the ten
        non-starred detectors (MonoFlex*/MonoGround* are original-torch-1.4-env rows and
        legitimately differ from the modern dumps).  <- validates the difficulty sweep
GATE-4  Moderate AP* vs reports/exp1_true_ceiling.txt CEIL* column, 12/12
GATE-5  Part-B machinery cross-check: recompute one detector's Car/Moderate true-IoU
        re-sort AP through the Part-B path (evaluator-IoU o_act + in-memory score swap)
        and compare to the Part-A path (shapely o_act + write_kitti).

READ-ONLY on all inputs.  Writes only reports_rerun/extensions/difficulty_class_extension.{txt,csv}.
MonoFlex and MonoGround enter through their MODERN-environment dumps (stems monoflex /
monoground) and modern-environment native runs.

Run from the repository root: python tools/extensions/difficulty_class_extension.py
"""
import os, sys, shutil, datetime, copy, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ext import paths, dump_path, cache_dir, out_path, aux_dump_path, repo_rel  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import ap_corrector_arc as arc  # noqa: E402
from dgp_cop_oracle_matrix import write_kitti  # noqa: E402
import evaluator.kitti_eval.kitti_common as kc  # noqa: E402
from exact_ap import ap_summaries  # noqa: E402
from evaluator.kitti_eval.eval import calculate_iou_partly, _prepare_data  # noqa: E402
from depth_share_bridge import DIAG, iou_act_and_zstar  # noqa: E402  (DIAG = cache/decomp)

OUT = out_path("extensions/difficulty_class_extension.txt")
CSV = out_path("extensions/difficulty_class_extension.csv")
WORK = os.path.join(cache_dir("extensions"), "_dcext")

val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
DIFFS = [(0, "Easy"), (1, "Moderate"), (2, "Hard")]

out = []


def w(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); out.append(s)
    open(OUT, "w").write("\n".join(out) + "\n")


def annos_for(df, score):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    return kc.get_label_annos(d, val)


def load_native_dir(path, tag):
    """READ-ONLY mirror of a native KITTI prediction dir with blank lines stripped.

    MonoFlex (30 files) and MonoGround (49 files) write a single '\\n' for images with no
    detection; the official kc.get_label_anno then does x[1] on an empty split and raises
    IndexError.  We copy every file verbatim into the scratchpad, dropping ONLY lines that
    are empty after strip(), and report how many were dropped.  The source dirs are never
    touched.  A dropped blank line carries no detection, so this cannot change any AP.
    """
    mirror = os.path.join(WORK, "native", tag)
    if os.path.exists(mirror):
        shutil.rmtree(mirror)
    os.makedirs(mirror)
    nblank = 0
    nline = 0
    for sid in val:
        fn = f"{sid:06d}.txt"
        src = os.path.join(path, fn)
        keep = []
        if os.path.exists(src):
            for ln in open(src):
                if ln.strip() == "":
                    nblank += 1
                else:
                    keep.append(ln.rstrip("\n"))
        nline += len(keep)
        with open(os.path.join(mirror, fn), "w") as fh:
            fh.write("\n".join(keep) + ("\n" if keep else ""))
    return kc.get_label_annos(mirror, val), nblank, nline


# ---------------------------------------------------------------- native pools
def native_pool(name):   # verbatim from tools/decomp/exp1_true_ceiling.py::native_pool
    if name in ("MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"):
        f = {"MonoDETR": "monodetr", "MonoDGP": "monodgp", "MonoCoP": "official_monocop",
             "MonoCLUE": "monoclue", "MonoIA": "monoia"}[name]
        df = pd.read_csv(aux_dump_path(f"{f}_val_preflatten.csv"))
        if "class_id" in df.columns:
            car = df[df.class_id == df.car_channel].copy()
            car["flatrank"] = df.loc[car.index, "flat_rank"]
        else:
            car = df.copy(); car["flatrank"] = car["flat_rank_car"]
            car["V"] = car["V_car"]; car["cls"] = car["cls_car"]
        pred = (car.flatrank < 50) & (car.cls >= 0.2)
        if "in_final" in car.columns:
            assert (pred == (car.in_final == 1)).all(), f"in_final mismatch {name}"
        return car[pred].reset_index(drop=True)
    if name == "M3D-RPN":
        df = pd.read_csv(aux_dump_path("m3drpn_val_floor0.csv"))
        outp = []
        for sid, d in df.groupby("sid"):
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
            outp.append(kk[kk["V"] >= 0.75])
        return pd.concat(outp).reset_index(drop=True)
    f = {"MonoDLE": "monodle", "MonoFlex": "monoflex", "GUPNet": "gupnet",
         "DEVIANT": "deviant", "MonoGround": "monoground", "MonoCon": "monocon"}[name]
    df = pd.read_csv(dump_path(f))
    thr = 0.4 if name == "MonoCon" else 0.2
    col = "cls" if name == "MonoDLE" else "V"
    return df[df[col] >= thr].reset_index(drop=True)


# --------------------------------------------------- max bipartite matching (Kuhn)
def max_matching(rows, ng):
    matchG = [-1] * ng

    def try_kuhn(d, seen):
        for g in rows[d]:
            if not seen[g]:
                seen[g] = True
                if matchG[g] == -1 or try_kuhn(matchG[g], seen):
                    matchG[g] = d
                    return True
        return False

    M = 0
    for d in range(len(rows)):
        if try_kuhn(d, [False] * ng):
            M += 1
    return M


def oriented(ov, ndt, ngt, tag):
    if ov.shape == (ndt, ngt):
        return ov
    if ov.shape == (ngt, ndt):
        return ov.T
    raise AssertionError(f"{tag}: overlaps {ov.shape} vs dt{ndt} gt{ngt}")


def ceiling_recall(overlaps, dt_annos, current_class, difficulty, min_ov, tag):
    """exact max #valid-GT simultaneously matchable to pool preds at IoU3D>=min_ov / #valid-GT."""
    (_, _, ig_gts, ig_dets, _, _, n_valid_gt) = _prepare_data(GT, dt_annos, current_class,
                                                              difficulty)
    sumM = 0
    for i in range(len(GT)):
        ig = np.asarray(ig_gts[i]); idt = np.asarray(ig_dets[i])
        ov = oriented(overlaps[i], len(idt), len(ig), f"{tag} img{i}")
        vdt = np.where(idt == 0)[0]; vgt = np.where(ig == 0)[0]
        if len(vdt) == 0 or len(vgt) == 0:
            continue
        sub = ov[np.ix_(vdt, vgt)] >= min_ov
        rows = [np.where(sub[d])[0].tolist() for d in range(len(vdt))]
        sumM += max_matching(rows, len(vgt))
    return sumM, int(n_valid_gt)


# ------------------------------------------------------------------ frozen refs
# reports/e4_fp_tp_decomp.txt  (base all-point AP on native pool, and 'full' = true-IoU gain)
E4 = {"M3D-RPN": (11.51, 11.68), "MonoDLE": (15.09, 14.03), "MonoFlex": (16.26, 10.46),
      "GUPNet": (17.11, 12.04), "DEVIANT": (17.48, 11.83), "MonoGround": (17.45, 10.89),
      "MonoCon": (19.59, 10.58), "MonoDETR": (21.18, 11.42), "MonoDGP": (22.82, 8.41),
      "MonoCoP": (24.34, 11.58), "MonoCLUE": (24.55, 9.73), "MonoIA": (25.18, 11.64)}
# main Table 1 (E, M, H); starred rows excluded
TAB1 = {"M3D-RPN": (15.14, 11.51, 8.95), "MonoDLE": (18.45, 15.09, 13.09),
        "GUPNet": (23.94, 17.11, 14.48), "DEVIANT": (26.02, 17.48, 14.89),
        "MonoCon": (26.95, 19.59, 16.52), "MonoDETR": (29.12, 21.18, 17.74),
        "MonoDGP": (30.62, 22.82, 19.88), "MonoCoP": (32.75, 24.34, 20.93),
        "MonoCLUE": (34.03, 24.55, 21.04), "MonoIA": (34.79, 25.18, 21.67)}
TAB1_STAR = {"MonoFlex": (25.08, 18.11, 15.52), "MonoGround": (26.18, 19.42, 16.22)}
# reports/exp1_true_ceiling.txt CEIL* column (Moderate)
EXP1_CEIL = {"M3D-RPN": 23.19, "MonoDLE": 29.12, "MonoFlex": 26.90, "GUPNet": 29.20,
             "DEVIANT": 29.36, "MonoGround": 28.36, "MonoCon": 31.04, "MonoDETR": 32.64,
             "MonoDGP": 34.40, "MonoCoP": 36.03, "MonoCLUE": 35.94, "MonoIA": 36.98}

MODELS = ["M3D-RPN", "MonoDLE", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround", "MonoCon",
          "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]

w(f"# difficulty_class_extension  {datetime.datetime.now().isoformat(timespec='seconds')}")
w(f"# script  tools/extensions/difficulty_class_extension.py")
w(f"# env     python {sys.version.split()[0]}")
w(f"# GT      KITTI training/label_2    split ImageSets/val.txt   n_img={len(val)}")
w("# Part A  = tools/decomp/exp1_true_ceiling.py with difficulty swept 0/1/2 (native panel pools)")
w("# Part B  = Pedestrian/Cyclist from complete native KITTI prediction dirs (see PART B header)")
w("")

# ==========================================================================
#  PART A  --  difficulty extension, Car IoU0.7, all 12 detectors
# ==========================================================================
w("=" * 110)
w("PART A -- KITTI Car, IoU3D>=0.7, native panel pools, all-point interpolated AP")
w("=" * 110)

rowsA = {}
gate5_store = {}
for name in MODELS:
    t0 = time.time()
    pool = native_pool(name)
    cache = f"{DIAG}/_e4cache_{name.replace('-', '')}.npz"
    if os.path.exists(cache):
        z = np.load(cache); o_act = z["o_act"]
        assert len(o_act) == len(pool), f"cache len {len(o_act)} vs pool {len(pool)} ({name})"
    else:
        o_act, _ = iou_act_and_zstar(pool)
    V = pool["V"].values.astype(float)
    dt_base = annos_for(pool, V)
    dt_full = annos_for(pool, o_act)
    # overlaps are score-independent -> compute once from dt_base and reuse for every difficulty
    ov_base = calculate_iou_partly(dt_base, GT, 2, 50)[0]
    rec = {}
    for d, dn in DIFFS:
        b = ap_summaries(GT, dt_base, current_class=0, difficulty=d, min_overlap=0.7)
        f_ = ap_summaries(GT, dt_full, current_class=0, difficulty=d, min_overlap=0.7)
        sumM, ngt = ceiling_recall(ov_base, dt_base, 0, d, 0.7, name)
        rec[dn] = dict(base=b["allpoint"], base_r40=b["r40_official"],
                       full=f_["allpoint"], full_r40=f_["r40_official"],
                       ceil=100.0 * sumM / ngt, M=sumM, ngt=ngt)
        w(f"[A] {name:10s} {dn:8s} npool={len(pool):6d} base={rec[dn]['base']:6.2f} "
          f"trueIoU={rec[dn]['full']:6.2f} gap={rec[dn]['full']-rec[dn]['base']:+6.2f} "
          f"AP*={rec[dn]['ceil']:6.2f} headroom={rec[dn]['ceil']-rec[dn]['base']:+6.2f} "
          f"M/n_gt={sumM}/{ngt}={sumM/ngt:.3f}")
    rowsA[name] = rec
    if name == "DEVIANT":
        gate5_store = dict(pool=pool, dt_base=dt_base, ov=ov_base, ref=rec["Moderate"]["full"])
    w(f"    ({time.time()-t0:.0f}s)")

w("")
w("---- GATES ----")
fails = []
for name in MODELS:
    m = rowsA[name]["Moderate"]
    eb, ef = E4[name]
    d1 = abs(m["base"] - eb); d2 = abs((m["full"] - m["base"]) - ef)
    w(f"GATE-1/2 {name:10s} base {m['base']:6.2f} vs e4 {eb:6.2f} (d={d1:.3f})  |  "
      f"gap {m['full']-m['base']:+6.2f} vs e4 full {ef:+6.2f} (d={d2:.3f})  "
      f"[{'PASS' if d1 < 0.05 and d2 < 0.05 else 'FAIL'}]")
    if not (d1 < 0.05 and d2 < 0.05):
        fails.append(f"GATE-1/2 {name}")
for name, (te, tm, th) in TAB1.items():
    ge = rowsA[name]["Easy"]["base"]; gm = rowsA[name]["Moderate"]["base"]
    gh = rowsA[name]["Hard"]["base"]
    de, dm, dh = abs(ge - te), abs(gm - tm), abs(gh - th)
    ok = max(de, dm, dh) < 0.05
    w(f"GATE-3   {name:10s} Table1 E/M/H {te:5.2f}/{tm:5.2f}/{th:5.2f}  got "
      f"{ge:5.2f}/{gm:5.2f}/{gh:5.2f}  maxd={max(de, dm, dh):.3f}  [{'PASS' if ok else 'FAIL'}]")
    if not ok:
        fails.append(f"GATE-3 {name}")
for name, (te, tm, th) in TAB1_STAR.items():
    ge = rowsA[name]["Easy"]["base"]; gm = rowsA[name]["Moderate"]["base"]
    gh = rowsA[name]["Hard"]["base"]
    w(f"GATE-3*  {name:10s} Table1(orig-env, EXCLUDED from gate) E/M/H {te:5.2f}/{tm:5.2f}/{th:5.2f}"
      f"  got(modern) {ge:5.2f}/{gm:5.2f}/{gh:5.2f}")
for name in MODELS:
    c = rowsA[name]["Moderate"]["ceil"]; e = EXP1_CEIL[name]
    ok = abs(c - e) < 0.05
    w(f"GATE-4   {name:10s} AP* {c:6.2f} vs exp1_true_ceiling.txt {e:6.2f} (d={abs(c-e):.3f}) "
      f"[{'PASS' if ok else 'FAIL'}]")
    if not ok:
        fails.append(f"GATE-4 {name}")

w("")
if fails:
    w("!!!! GATE FAILURES: " + ", ".join(fails))
    w("!!!! ABORTING -- no new numbers may be quoted.")
    open(CSV, "w").write("ABORTED\n")
    sys.exit(1)
w("ALL GATES 1-4 PASS.")
w("")

# ---- Part A tables ----
w("=" * 118)
w("TABLE A1 -- base / true-IoU re-sort / ordering gap, by difficulty (Car IoU0.7, all-point AP)")
w("=" * 118)
h = (f"{'detector':11s}| {'E base':>7}{'E resort':>9}{'E gap':>7} | "
     f"{'M base':>7}{'M resort':>9}{'M gap':>7} | {'H base':>7}{'H resort':>9}{'H gap':>7}")
w(h); w("-" * len(h))
gapd = {dn: [] for _, dn in DIFFS}
headd = {dn: [] for _, dn in DIFFS}
recd = {dn: [] for _, dn in DIFFS}
for name in MODELS:
    cells = []
    for _, dn in DIFFS:
        r = rowsA[name][dn]
        gapd[dn].append(r["full"] - r["base"])
        headd[dn].append(r["ceil"] - r["base"])
        recd[dn].append(r["M"] / r["ngt"])
        cells.append(f"{r['base']:>7.2f}{r['full']:>9.2f}{r['full']-r['base']:>+7.2f}")
    w(f"{name:11s}| " + " | ".join(cells))
w("")
w("=" * 118)
w("TABLE A2 -- matching ceiling AP* = 100*M/n_gt (exact max-over-labelings upper bound) "
  "and matching recall")
w("=" * 118)
h2 = (f"{'detector':11s}| {'E AP*':>7}{'E head':>8}{'E M/ngt':>9} | "
      f"{'M AP*':>7}{'M head':>8}{'M M/ngt':>9} | {'H AP*':>7}{'H head':>8}{'H M/ngt':>9}")
w(h2); w("-" * len(h2))
for name in MODELS:
    cells = []
    for _, dn in DIFFS:
        r = rowsA[name][dn]
        cells.append(f"{r['ceil']:>7.2f}{r['ceil']-r['base']:>+8.2f}{r['M']/r['ngt']:>9.3f}")
    w(f"{name:11s}| " + " | ".join(cells))
w("")
w("== PANEL SUMMARY (Car) ==")
for _, dn in DIFFS:
    g = np.array(gapd[dn]); hh = np.array(headd[dn]); rr = np.array(recd[dn])
    ngt = rowsA['MonoIA'][dn]['ngt']
    w(f"{dn:8s} n_valid_gt={ngt:6d} | true-IoU re-sort gap  {g.min():+.2f}..{g.max():+.2f} "
      f"median {np.median(g):+.2f}  >0: {(g > 0).sum()}/12")
    w(f"{'':8s} {'':17s}| AP*-base headroom     {hh.min():+.2f}..{hh.max():+.2f} "
      f"median {np.median(hh):+.2f}  >0: {(hh > 0).sum()}/12")
    w(f"{'':8s} {'':17s}| matching recall M/ngt {rr.min():.3f}..{rr.max():.3f} "
      f"median {np.median(rr):.3f}  -> {100*np.median(rr):.1f}% of GTs are even reachable")
w("")
w("R40-official base/re-sort (secondary, for reference):")
h3 = (f"{'detector':11s}| {'E base':>7}{'E resort':>9} | {'M base':>7}{'M resort':>9} | "
      f"{'H base':>7}{'H resort':>9}")
w(h3); w("-" * len(h3))
for name in MODELS:
    cells = []
    for _, dn in DIFFS:
        r = rowsA[name][dn]
        cells.append(f"{r['base_r40']:>7.2f}{r['full_r40']:>9.2f}")
    w(f"{name:11s}| " + " | ".join(cells))
w("")

# ==========================================================================
#  PART B  --  class extension
# ==========================================================================
w("=" * 110)
w("PART B -- Pedestrian / Cyclist extension")
w("=" * 110)
w("Only detectors whose RELEASED checkpoint can emit non-Car objects AND for which a complete")
w("3769-image native KITTI prediction directory already exists on disk are usable.  Sources:")
U = paths.UPSTREAM_ROOT
NATIVE_DIRS = [
    ("M3D-RPN", os.path.join(U, "M3D-RPN", "results_origsid", "data")),
    ("MonoFlex", os.path.join(U, "MonoFlex", "tmp_eval", "inference", "kitti_train", "data")),
    ("DEVIANT", os.path.join(U, "DEVIANT", "output", "run_221", "result_kitti", "data")),
    ("MonoGround", os.path.join(U, "MonoGround", "tmp", "inference", "kitti_train", "val", "label_2")),
]
for n, p in NATIVE_DIRS:
    w(f"   {n:11s} {repo_rel(p)}   ({len(os.listdir(p))} files)")
w("")

CLASSES = [(0, "Car", 0.7), (0, "Car", 0.5), (1, "Pedestrian", 0.5), (2, "Cyclist", 0.5)]

# ---- GATE-5: validate the Part-B machinery on the Part-A DEVIANT pool ----
w("---- GATE-5 (Part-B machinery cross-check on the Part-A DEVIANT Car pool) ----")
g5 = gate5_store
ov5 = g5["ov"]
gt_is_car = [np.array([nm == "Car" for nm in GT[i]["name"]], bool) for i in range(len(GT))]
dtB = copy.deepcopy(g5["dt_base"])
for i in range(len(GT)):
    ndt = len(dtB[i]["name"])
    ov = oriented(ov5[i], ndt, len(GT[i]["name"]), f"gate5 img{i}")
    m = gt_is_car[i]
    dtB[i]["score"] = (ov[:, m].max(1) if (m.any() and ndt) else np.zeros(ndt)).astype(np.float64)
ap5 = ap_summaries(GT, dtB, current_class=0, difficulty=1, min_overlap=0.7)["allpoint"]
d5 = abs(ap5 - g5["ref"])
w(f"GATE-5   DEVIANT Car/Moderate true-IoU re-sort: PartB path {ap5:.2f} vs PartA frozen path "
  f"{g5['ref']:.2f}  (d={d5:.3f})  [{'PASS' if d5 < 0.5 else 'FAIL'}]")
if d5 >= 0.5:
    w("!!!! GATE-5 FAIL -- Part B numbers are NOT reportable.  Part A stands.")
    w("!!!! (Part A gates 1-4 already passed; only Part B is voided.)")
w("")

rowsB = {}
if d5 < 0.5:
    for name, path in NATIVE_DIRS:
        t0 = time.time()
        dt, nblank, nline = load_native_dir(path, name)
        hist = {}
        for a in dt:
            for nm in a["name"]:
                hist[nm] = hist.get(nm, 0) + 1
        assert sum(hist.values()) == nline, f"{name}: {sum(hist.values())} vs {nline}"
        w(f"[B] {name:11s} native class histogram: " +
          "  ".join(f"{k}={v}" for k, v in sorted(hist.items())) +
          f"   (total {nline}; {nblank} blank lines skipped)")
        ovB = calculate_iou_partly(dt, GT, 2, 50)[0]
        for cc, cname, min_ov in CLASSES:
            gmask = [np.array([nm == cname for nm in GT[i]["name"]], bool)
                     for i in range(len(GT))]
            dtf = copy.deepcopy(dt)
            for i in range(len(GT)):
                ndt = len(dtf[i]["name"])
                ov = oriented(ovB[i], ndt, len(GT[i]["name"]), f"{name} {cname} img{i}")
                m = gmask[i]
                dtf[i]["score"] = ((ov[:, m].max(1) if (m.any() and ndt) else np.zeros(ndt))
                                   .astype(np.float64))
            for d, dn in DIFFS:
                b = ap_summaries(GT, dt, current_class=cc, difficulty=d, min_overlap=min_ov)
                f_ = ap_summaries(GT, dtf, current_class=cc, difficulty=d, min_overlap=min_ov)
                sumM, ngt = ceiling_recall(ovB, dt, cc, d, min_ov,
                                           f"{name}/{cname}")
                key = (name, cname, min_ov, dn)
                rowsB[key] = dict(base=b["allpoint"], base_r40=b["r40_official"],
                                  full=f_["allpoint"], ceil=100.0 * sumM / ngt,
                                  M=sumM, ngt=ngt)
                w(f"    {name:11s} {cname:10s}@{min_ov} {dn:8s} base={b['allpoint']:6.2f} "
                  f"trueIoU={f_['allpoint']:6.2f} gap={f_['allpoint']-b['allpoint']:+6.2f} "
                  f"AP*={100.0*sumM/ngt:6.2f} headroom={100.0*sumM/ngt-b['allpoint']:+6.2f} "
                  f"M/n_gt={sumM}/{ngt}={sumM/ngt:.3f}")
        w(f"    ({time.time()-t0:.0f}s)")

    w("")
    w("=" * 118)
    w("TABLE B1 -- class extension, 4 detectors, native full prediction files, all-point AP")
    w("=" * 118)
    hb = (f"{'detector':11s} {'class':11s} {'IoU':>4} | " +
          " | ".join(f"{dn[0]} base  gap   AP*  head" for _, dn in DIFFS))
    w(hb); w("-" * len(hb))
    for name, _ in NATIVE_DIRS:
        for cc, cname, min_ov in CLASSES:
            cells = []
            for _, dn in DIFFS:
                r = rowsB[(name, cname, min_ov, dn)]
                cells.append(f"{r['base']:6.2f}{r['full']-r['base']:+6.2f}"
                             f"{r['ceil']:6.2f}{r['ceil']-r['base']:+6.2f}")
            w(f"{name:11s} {cname:11s} {min_ov:>4} | " + " | ".join(cells))
    w("")
    w("== PANEL SUMMARY (class extension, n=4 detectors) ==")
    for cc, cname, min_ov in CLASSES:
        for _, dn in DIFFS:
            g = np.array([rowsB[(n, cname, min_ov, dn)]["full"] -
                          rowsB[(n, cname, min_ov, dn)]["base"] for n, _ in NATIVE_DIRS])
            hh = np.array([rowsB[(n, cname, min_ov, dn)]["ceil"] -
                           rowsB[(n, cname, min_ov, dn)]["base"] for n, _ in NATIVE_DIRS])
            rr = np.array([rowsB[(n, cname, min_ov, dn)]["M"] /
                           rowsB[(n, cname, min_ov, dn)]["ngt"] for n, _ in NATIVE_DIRS])
            ngt = rowsB[(NATIVE_DIRS[0][0], cname, min_ov, dn)]["ngt"]
            w(f"{cname:11s}@{min_ov} {dn:8s} n_gt={ngt:6d} | gap {g.min():+.2f}..{g.max():+.2f} "
              f"med {np.median(g):+.2f} (>0 {(g>0).sum()}/4) | AP*-base {hh.min():+.2f}.."
              f"{hh.max():+.2f} med {np.median(hh):+.2f} | M/ngt med {np.median(rr):.3f}")

# ---------------------------------------------------------------- CSV
recs = []
for name in MODELS:
    for _, dn in DIFFS:
        r = rowsA[name][dn]
        recs.append(dict(part="A", detector=name, cls="Car", iou=0.7, difficulty=dn,
                         pool="native_panel_pool", base_allpoint=round(r["base"], 4),
                         base_r40=round(r["base_r40"], 4),
                         trueiou_resort_allpoint=round(r["full"], 4),
                         ordering_gap=round(r["full"] - r["base"], 4),
                         ap_star=round(r["ceil"], 4),
                         headroom_apstar=round(r["ceil"] - r["base"], 4),
                         M=r["M"], n_gt=r["ngt"], matching_recall=round(r["M"] / r["ngt"], 5)))
for (name, cname, min_ov, dn), r in rowsB.items():
    recs.append(dict(part="B", detector=name, cls=cname, iou=min_ov, difficulty=dn,
                     pool="native_prediction_dir", base_allpoint=round(r["base"], 4),
                     base_r40=round(r["base_r40"], 4),
                     trueiou_resort_allpoint=round(r["full"], 4),
                     ordering_gap=round(r["full"] - r["base"], 4),
                     ap_star=round(r["ceil"], 4),
                     headroom_apstar=round(r["ceil"] - r["base"], 4),
                     M=r["M"], n_gt=r["ngt"], matching_recall=round(r["M"] / r["ngt"], 5)))
pd.DataFrame(recs).to_csv(CSV, index=False)
if os.path.exists(WORK):
    shutil.rmtree(WORK)
w("")
w(f"[written] {repo_rel(OUT)}")
w(f"[written] {repo_rel(CSV)}")

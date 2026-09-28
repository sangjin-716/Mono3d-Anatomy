"""Produces reports_orig/difficulty_sweep_orig/headroom_sep_by_difficulty.{txt,json}.

Difficulty robustness sweep, A (headroom) + B (separation) for Easy/Mod/Hard, for the
original-environment MonoFlex*/MonoGround* dumps (a copy of
tools/decomp/diffsweep_headroom_sep.py with inputs switched). Its Mod rows are the starred
separation decomposition (reports/e4_fp_tp_decomp.txt has no starred rows).

Reuses tools/decomp/e4_fp_tp_decomp.py unchanged (native_pool, the cached o_act+lab, and the
base/fpd/tpr/full score construction). The ONLY change: the final ap_summaries is looped over
difficulty in {0(Easy),1(Mod),2(Hard)}. The re-sort orderings and TP/FP labels are
difficulty-AGNOSTIC (fixed any-Car-GT); only the evaluation GT subset changes with difficulty
(official _prepare_data filter).
Writes the cache <CACHE_DIR>/orig/_e4cache_<name>.npz (o_act + TP labels) that
exp1_ceiling_orig.py and c2_iou05_orig.py reuse, so run this script before those two.
The other detectors' native_pool branches are kept but never executed here.
Run from the repository root: python tools/orig/diffsweep_orig.py
"""
import os, sys, shutil, json, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
from tools.orig._orig_common import ORIG, extra_dump
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti
import evaluator.kitti_eval.kitti_common as kc
from exact_ap import ap_summaries
from depth_share_bridge import iou_act_and_zstar

val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
WORK = f"{ORIG}/_diffsweep_orig"
OUTDIR = os.path.join(paths.OUT_DIR, "difficulty_sweep_orig")
os.makedirs(OUTDIR, exist_ok=True)
DIFFS = [("Easy", 0), ("Mod", 1), ("Hard", 2)]
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def annos_for(df, score):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    return kc.get_label_annos(d, val)


def gts_of(sid):
    p = os.path.join(arc.LABEL_DIR, f"{sid:06d}.txt")
    res = []
    if not os.path.exists(p):
        return res
    for ln in open(p):
        t = ln.split()
        if t[0] == "Car":   # any difficulty (prereg: labels vs ALL Car GTs)
            res.append((float(t[8]), float(t[9]), float(t[10]), float(t[11]),
                        float(t[12]), float(t[13]), float(t[14])))
    return res


def tp_labels(df):
    """frozen baseline labels: per frame, V-desc greedy 1-to-1 match at IoU3D>=0.7."""
    lab = np.zeros(len(df), bool)
    pos = {s: g for s, g in df.groupby("sid")}
    for sid, g in pos.items():
        gts = gts_of(int(sid))
        if not gts:
            continue
        order = g.index.values[np.argsort(-g.V.values, kind="stable")]
        used = [False] * len(gts)
        px = df.x_3d; py = df.y_3d; pz = df.z_3d
        ph = df.h_3d; pw = df.w_3d; pl = df.l_3d; pr = df.ry
        gps = [arc.poly(t[3], t[5], t[1], t[2], t[6]) for t in gts]
        for i in order:
            best, barg = 0.0, -1
            for j, t in enumerate(gts):
                if used[j] or abs(pz[i] - t[5]) >= 8.0:
                    continue
                pp = arc.poly(px[i], pz[i], pw[i], pl[i], pr[i])
                v = arc.iou3d(pp, py[i], ph[i], pl[i], pw[i], t, gps[j])
                if v > best:
                    best, barg = v, j
            if best >= 0.7:
                used[barg] = True; lab[df.index.get_loc(i)] = True
    return lab


def native_pool(name):  # VERBATIM from e4_fp_tp_decomp.py
    if name in ("MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"):
        f = {"MonoDETR": "monodetr", "MonoDGP": "monodgp", "MonoCoP": "official_monocop",
             "MonoCLUE": "monoclue", "MonoIA": "monoia"}[name]
        df = pd.read_csv(extra_dump(f"{f}_val_preflatten.csv"))
        if "class_id" in df.columns:
            car = df[df.class_id == df.car_channel].copy(); car["flatrank"] = df.loc[car.index, "flat_rank"]
        else:
            car = df.copy(); car["flatrank"] = car["flat_rank_car"]; car["V"] = car["V_car"]; car["cls"] = car["cls_car"]
        return car[(car.flatrank < 50) & (car.cls >= 0.2)].reset_index(drop=True)
    if name == "M3D-RPN":
        df = pd.read_csv(extra_dump("m3drpn_val_floor0.csv")); outp = []
        for sid, d in df.groupby("sid"):
            dd = d.sort_values("V", ascending=False)
            b = dd[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]; areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            keep = []; order = np.arange(len(dd))
            while order.size > 0:
                i = order[0]; keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
                wq = np.maximum(0, xx2 - xx1 + 1); hq = np.maximum(0, yy2 - yy1 + 1)
                ovr = wq * hq / (areas[i] + areas[order[1:]] - wq * hq)
                order = order[np.where(ovr <= 0.4)[0] + 1]
            kk = dd.iloc[keep].head(40); outp.append(kk[kk["V"] >= 0.75])
        return pd.concat(outp).reset_index(drop=True)
    f = {"MonoFlex*": "monoflex_orig", "MonoGround*": "monoground_orig"}[name]
    df = pd.read_csv(dump_path(f)); thr = 0.2
    col = "V"
    return df[df[col] >= thr].reset_index(drop=True)


# Canonical Moderate gate values (e4_fp_tp_decomp.txt): base_allpt, full_gap, FPdem_gap, TPreord_gap
GATE_MOD = {
    "M3D-RPN": (11.51, 11.68, 11.68, -0.33), "MonoDLE": (15.09, 14.03, 14.06, -1.34),
    "MonoFlex": (16.26, 10.46, 10.65, -2.13), "GUPNet": (17.11, 12.04, 12.09, -0.92),
    "DEVIANT": (17.48, 11.83, 11.88, -1.23), "MonoGround": (17.45, 10.89, 10.92, -1.60),
    "MonoCon": (19.59, 10.58, 11.45, -1.37), "MonoDETR": (21.18, 11.42, 11.54, -1.16),
    "MonoDGP": (22.82, 8.41, 11.59, -1.33), "MonoCoP": (24.34, 11.58, 11.72, -1.07),
    "MonoCLUE": (24.55, 9.73, 11.45, -1.27), "MonoIA": (25.18, 11.64, 11.81, -1.46),
}
MODELS = ["MonoFlex*", "MonoGround*"]

w(f"# diffsweep headroom+separation {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# A=headroom(full-resort gap), B=separation(FPdem/TPreord) | allpoint (R40) | Easy/Mod/Hard")
w("# Mod(=1) reproduces e4_fp_tp_decomp.txt (gate); re-sort orderings difficulty-agnostic.")
w("")
records = {}
for name in MODELS:
    pool = native_pool(name)
    cache = f"{ORIG}/_e4cache_{name.replace('-', '').replace('*', 'X')}.npz"
    if os.path.exists(cache):
        z = np.load(cache); o_act = z["o_act"]; lab = z["lab"].astype(bool)
        assert len(o_act) == len(pool), f"cache len mismatch {name}"
    else:
        o_act, _ = iou_act_and_zstar(pool)
        lab = tp_labels(pool)
        np.savez_compressed(cache, o_act=o_act, lab=lab)
    V = pool["V"].values.astype(float); BIG = V.max() + 1.0
    s_fpd = np.where(lab, V + BIG, V)
    s_tpr = V.copy(); tp_idx = np.where(lab)[0]
    if len(tp_idx):
        slot = np.sort(V[tp_idx])[::-1]
        order = tp_idx[np.lexsort((-V[tp_idx], -o_act[tp_idx]))]
        s_tpr[order] = slot
    # build annos once per score, evaluate at all three difficulties
    rec = {"TPshare": float(lab.mean()), "n": int(len(pool))}
    ap = {}  # ap[tag][diff] = (allpoint, r40)
    for tag, sc in [("base", V), ("fpd", s_fpd), ("tpr", s_tpr), ("full", o_act)]:
        annos = annos_for(pool, sc)
        ap[tag] = {}
        for dname, dval in DIFFS:
            s = ap_summaries(GT, annos, difficulty=dval, metric=2, min_overlap=0.7)
            ap[tag][dname] = (round(s["allpoint"], 3), round(s["r40_official"], 3))
    rec["ap"] = ap
    # per-difficulty gaps
    for dname, _ in DIFFS:
        base_a = ap["base"][dname][0]
        rec.setdefault("gap_all", {})[dname] = round(ap["full"][dname][0] - base_a, 3)
        rec.setdefault("gap_r40", {})[dname] = round(ap["full"][dname][1] - ap["base"][dname][1], 3)
        rec.setdefault("fpdem", {})[dname] = round(ap["fpd"][dname][0] - base_a, 3)
        rec.setdefault("tpreord", {})[dname] = round(ap["tpr"][dname][0] - base_a, 3)
        full_g = ap["full"][dname][0] - base_a
        rec.setdefault("fpdem_over_full", {})[dname] = round((ap["fpd"][dname][0] - base_a) / full_g, 3) if abs(full_g) > 1e-6 else None
    records[name] = rec
    # gate check (Mod)
    gate = "NEW(orig-env; no frozen reference)"
    w(f"[{name:10s}] gate(Mod)={gate} TPshare={rec['TPshare']:.3f}")
    for dname, _ in DIFFS:
        w(f"    {dname:4s} base={ap['base'][dname][0]:6.2f}({ap['base'][dname][1]:5.2f})"
          f"  full-gap=+{rec['gap_all'][dname]:5.2f}(R40 +{rec['gap_r40'][dname]:5.2f})"
          f"  FPdem=+{rec['fpdem'][dname]:5.2f}  TPreord={rec['tpreord'][dname]:+5.2f}"
          f"  FPdem/full={rec['fpdem_over_full'][dname]}")

w("")
w("== panel summary per difficulty (all-point) ==")
for dname, _ in DIFFS:
    gaps = [records[n]["gap_all"][dname] for n in MODELS]
    tpr = [records[n]["tpreord"][dname] for n in MODELS]
    tpr_nonpos = sum(1 for t in tpr if t <= 0.0)
    w(f"{dname:4s}: headroom min/med/max = +{min(gaps):.2f}/+{np.median(gaps):.2f}/+{max(gaps):.2f}"
      f" | 12/12 ≥+8? {sum(1 for g in gaps if g >= 8.0)}/12"
      f" | TPreord ≤0: {tpr_nonpos}/12 (range {min(tpr):+.2f}..{max(tpr):+.2f})")
if os.path.exists(WORK):
    shutil.rmtree(WORK)
open(f"{OUTDIR}/headroom_sep_by_difficulty.txt", "w").write("\n".join(out) + "\n")
json.dump(records, open(f"{OUTDIR}/headroom_sep_by_difficulty.json", "w"), indent=2)
w(f"[written] {OUTDIR}/headroom_sep_by_difficulty.{{txt,json}}")

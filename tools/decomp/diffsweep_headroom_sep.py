"""Ported from tools/decomp/diffsweep_headroom_sep.py for the public release. Computation unchanged.
Produces reports/final_run/difficulty_sweep/headroom_sep_by_difficulty.{txt,json}.

Difficulty robustness sweep — A (headroom) + B (separation) for Easy/Mod/Hard.

REUSES tools/decomp/e4_fp_tp_decomp.py VERBATIM (native_pool, the cached o_act+lab, and the
base/fpd/tpr/full score construction). The ONLY change: the final ap_summaries is looped over
difficulty in {0(Easy),1(Mod),2(Hard)}. The re-sort orderings and TP/FP labels are
difficulty-AGNOSTIC (frozen any-Car-GT); only the evaluation GT subset changes with difficulty
(official _prepare_data filter). So moderate(=1) reproduces e4_fp_tp_decomp.txt exactly = the gate.
Run e4_fp_tp_decomp.py first: it writes the _e4cache_<model>.npz files this script requires.
Native pools need the pre-flatten query dumps and the floor-0 M3D-RPN dump in DUMP_DIR.
Run from the repository root: python tools/decomp/diffsweep_headroom_sep.py
"""
import os, sys, shutil, json, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti
import evaluator.kitti_eval.kitti_common as kc
from exact_ap import ap_summaries
from depth_share_bridge import iou_act_and_zstar, DIAG

val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
WORK = f"{DIAG}/_diffsweep_hs"
OUTDIR = os.path.join(paths.OUT_DIR, "final_run", "difficulty_sweep")
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


def native_pool(name):  # VERBATIM from e4_fp_tp_decomp.py
    if name in ("MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"):
        f = {"MonoDETR": "monodetr", "MonoDGP": "monodgp", "MonoCoP": "official_monocop",
             "MonoCLUE": "monoclue", "MonoIA": "monoia"}[name]
        df = pd.read_csv(os.path.join(paths.DUMP_DIR, f"{f}_val_preflatten.csv"))
        if "class_id" in df.columns:
            car = df[df.class_id == df.car_channel].copy(); car["flatrank"] = df.loc[car.index, "flat_rank"]
        else:
            car = df.copy(); car["flatrank"] = car["flat_rank_car"]; car["V"] = car["V_car"]; car["cls"] = car["cls_car"]
        return car[(car.flatrank < 50) & (car.cls >= 0.2)].reset_index(drop=True)
    if name == "M3D-RPN":
        df = pd.read_csv(os.path.join(paths.DUMP_DIR, "m3drpn_val_floor0.csv")); outp = []
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
    f = {"MonoDLE": "monodle", "MonoFlex": "monoflex", "GUPNet": "gupnet",
         "DEVIANT": "deviant", "MonoGround": "monoground", "MonoCon": "monocon"}[name]
    df = pd.read_csv(dump_path(f)); thr = 0.4 if name == "MonoCon" else 0.2
    col = "cls" if name == "MonoDLE" else "V"
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
MODELS = ["M3D-RPN", "MonoDLE", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround", "MonoCon",
          "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]

w(f"# diffsweep headroom+separation {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# A=headroom(full-resort gap), B=separation(FPdem/TPreord) | allpoint (R40) | Easy/Mod/Hard")
w("# Mod(=1) reproduces e4_fp_tp_decomp.txt (gate); re-sort orderings difficulty-agnostic.")
w("")
records = {}
for name in MODELS:
    pool = native_pool(name)
    cache = f"{DIAG}/_e4cache_{name.replace('-', '')}.npz"
    assert os.path.exists(cache), f"missing cache {cache} (do NOT recompute; gate fail)"
    z = np.load(cache); o_act = z["o_act"]; lab = z["lab"].astype(bool)
    assert len(o_act) == len(pool), f"cache len mismatch {name}"
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
    gb, gfull, gfpd, gtpr = GATE_MOD[name]
    db = abs(ap["base"]["Mod"][0] - gb); dfull = abs(rec["gap_all"]["Mod"] - gfull)
    dfpd = abs(rec["fpdem"]["Mod"] - gfpd); dtpr = abs(rec["tpreord"]["Mod"] - gtpr)
    gate = "PASS" if (db <= 0.05 and dfull <= 0.10 and dfpd <= 0.15 and dtpr <= 0.15) else f"CHECK(b{db:.2f}/f{dfull:.2f}/fp{dfpd:.2f}/tp{dtpr:.2f})"
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

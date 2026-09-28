"""Ported from tools/decomp/detr_native_sweep.py for the public release. Computation unchanged.
Produces reports/detr_native_sweep.txt.

DETR-native operating-point sweep (reports/prereg_2b_detr_sweep.md rev3) on the validated
pre-flatten dumps (<f>_val_preflatten.csv in DUMP_DIR; not part of the 14 released val dumps).

NATIVE stage order everywhere: K_flat (top-K of the 150 query×class hypotheses by raw cls)
→ cls threshold → V=cls·exp(−σ) ranking → K_final. NO NMS (none exists natively).
Car-hypothesis universe per model: 50 rows/img (each query's Car hypothesis).
  C      thr ∈ {0,.05,.1,.2,.3,.4} at K_flat=50           (native thr = 0.2)
  D-flat K_flat ∈ {25,50,100,150} at thr=0.2              (150 = no-crowding counterfactual)
  D-fin  K_final ∈ {10,20,30,50} after (K_flat50, thr0.2) (by V, per image)
  F gap  (K_flat,thr) ∈ {(50,.2)=NATIVE,(150,0),(50,.3),(150,.2)}: ceiling = rank by oracle
         IoU3D (cache _pfcache_<f>.npz via iou_act_and_zstar on the 50/img Car pool)
  S diag cls-consistent (gate cls≥.2, rank cls) / V-consistent (per-image count matched to
         native, select+rank by V) — DIAGNOSTIC-ONLY labels
  IoU0.5 robustness: base+ceiling at the native cell with min_overlap=0.5
Metrics: all-point interpolated AP primary + official R40 (same call), Car moderate.
Run from the repository root: python tools/decomp/detr_native_sweep.py
"""
import os, sys, shutil, datetime
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
from depth_share_bridge import iou_act_and_zstar

DIAG = cache_dir("decomp")   # work dirs + npz caches (dumps are read through dump_path)
MODELS = [("MonoDETR", "monodetr", "hyp"), ("MonoDGP", "monodgp", "qry"),
          ("MonoCoP", "official_monocop", "qry"), ("MonoCLUE", "monoclue", "hyp"),
          ("MonoIA", "monoia", "hyp")]
val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)
W = f"{DIAG}/_detrsweep"
OUT = out_path("detr_native_sweep.txt")
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def ap2(df, score, min_overlap=0.7):
    if os.path.exists(W):
        shutil.rmtree(W)
    d = os.path.join(W, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    DT = kc.get_label_annos(d, val)
    s = ap_summaries(GT, DT, min_overlap=min_overlap)
    return s["allpoint"], s["r40_official"]


def load_car(f, kind):
    df = pd.read_csv(os.path.join(paths.DUMP_DIR, f"{f}_val_preflatten.csv"))
    if kind == "hyp":
        car = df[df.class_id == df.car_channel].copy()
        car["flatrank"] = car["flat_rank"]
    else:  # generic per-query schema (DGP/CoP)
        car = df.copy()
        car["flatrank"] = car["flat_rank_car"]
        car["V"] = car["V_car"]; car["cls"] = car["cls_car"]
    return car.reset_index(drop=True)


w(f"# detr_native_sweep {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# prereg rev3; native order K_flat->thr->V->K_final; validated preflatten dumps only")
w("# cells: allpoint(R40); native cell = K_flat50, thr0.2")
w("")

for name, f, kind in MODELS:
    car = load_car(f, kind)
    cache = f"{DIAG}/_pfcache_{f}.npz"
    if os.path.exists(cache):
        o_act = np.load(cache)["o_act"]; assert len(o_act) == len(car)
    else:
        o_act, o_z = iou_act_and_zstar(car)
        np.savez_compressed(cache, o_act=o_act, o_z=o_z)
    car["oact"] = o_act
    w(f"== {name} (Car-hypothesis rows={len(car)})")

    def pool(kflat, thr):
        p = car[car.flatrank < kflat]
        return p[p["cls"] >= thr] if thr > 0 else p

    # C: thr sweep @K_flat50
    for thr in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
        p = pool(50, thr)
        a, r = ap2(p, p["V"].values)
        w(f"  C thr={thr:4.2f} | base={a:6.2f} ({r:5.2f})" + ("   <-NATIVE" if thr == 0.2 else ""))
    # D-flat: K_flat sweep @thr0.2
    for kf in (25, 50, 100, 150):
        p = pool(kf, 0.2)
        a, r = ap2(p, p["V"].values)
        w(f"  Dflat K_flat={kf:3d} | base={a:6.2f} ({r:5.2f})")
    # D-final
    p0 = pool(50, 0.2)
    for kfin in (10, 20, 30, 50):
        p = p0.sort_values("V", ascending=False).groupby("sid").head(kfin)
        a, r = ap2(p, p["V"].values)
        w(f"  Dfin K_final={kfin:3d} | base={a:6.2f} ({r:5.2f})")
    # F gap cells
    for kf, thr in ((50, 0.2), (150, 0.0), (50, 0.3), (150, 0.2)):
        p = pool(kf, thr)
        b, br = ap2(p, p["V"].values)
        c_, cr = ap2(p, p["oact"].values)
        tag = "NATIVE" if (kf, thr) == (50, 0.2) else "      "
        w(f"  F {tag} K_flat={kf:3d} thr={thr:3.1f} | base={b:6.2f} ceil={c_:6.2f} "
          f"gap={c_-b:+6.2f} (R40 {cr-br:+.2f})")
    # IoU0.5 at native cell
    p = pool(50, 0.2)
    b5, _ = ap2(p, p["V"].values, min_overlap=0.5)
    c5, _ = ap2(p, p["oact"].values, min_overlap=0.5)
    w(f"  IoU0.5 native    | base={b5:6.2f} ceil={c5:6.2f} gap={c5-b5:+6.2f}")
    # S diagnostics (diagnostic-only)
    pc = pool(50, 0.2)
    a, r = ap2(pc, pc["cls"].values)
    w(f"  S cls-consistent | AP={a:6.2f} ({r:5.2f})  [diagnostic-only]")
    nat_counts = pc.groupby("sid").size()
    pv = pool(50, 0.0).sort_values("V", ascending=False).groupby("sid", group_keys=False) \
        .apply(lambda g: g.head(int(nat_counts.get(g.name, 0))))
    a, r = ap2(pv, pv["V"].values)
    w(f"  S V-consistent   | AP={a:6.2f} ({r:5.2f})  [diagnostic-only; count-matched]")
    w("")

w("READ: grading per prereg rev3 + claim_decision_tree rubric happens at GATE-2 (combined")
w("with the 7 non-DETR results). Structural/counterfactual cells (K_flat=150) are diagnostics.")
if os.path.exists(W):
    shutil.rmtree(W)
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

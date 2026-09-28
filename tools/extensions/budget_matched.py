"""Produces the per-run cell files reports_rerun/extensions/_bm_run{A,B}.txt, which
budget_matched_finalize.py cross-checks and turns into reports/extensions/budget_matched.txt
(supplementary Sec. L, Table L).

BUDGET-MATCHED coverage-vs-ordering split.

Question. The Sec. 5 split of missing AP into a COVERAGE share (no >=0.7 box exists in the fixed
pool for 63-77% of GTs) and an ORDERING share is computed on each detector's NATIVE final pool.
Native pools have unequal candidate budgets, so the split could be a budget artifact.

This script recomputes, per detector, on pools with an IDENTICAL candidate budget N:
    base    = all-point AP, Car Moderate, IoU3D 0.7, ranked by the detector's native score V
    AP*     = 100 * M / n_gt  (M = maximum bipartite matching of pool predictions to valid
              Moderate GTs at IoU3D >= 0.7 -- the exact max-over-labelings ceiling)
    missing AP     = 100 - base
    ordering share = (AP* - base) / (100 - base)
    coverage share = (100 - AP*) / (100 - base)
The two shares sum to 1 by construction (same fixed-pool accounting as Sec. 5).

Protocols (both reported):
  P1  complete pool -> top-N by V per image                      (pure budget match)
  P2  complete pool -> 2D-NMS@0.5 by V -> top-N by V per image   (matched protocol + budget)
Neither applies a score threshold, so the budget is exactly N wherever the pool can supply it.

Budgets N = 5, 10, 20. Justification (reports/extensions/budget_saturation.txt): the fraction of
the 3769 val images whose COMPLETE pool holds >= N candidates is >= 0.992 (N=5), >= 0.972
(N=10), >= 0.869 (N=20, worst = MonoCon); at N=30 MonoCon collapses to 0.017 (its native
top-k is 30) and at N=50 five CenterNet detectors are at 0.000. N=20 is therefore the largest
budget every detector in the panel can actually supply.

GATE. With pool = native_pool() (copied from tools/decomp/e4_fp_tp_decomp.py) the script must
reproduce reports/exp1_true_ceiling.txt base / recall*. MonoFlex/MonoGround are gated against
reports_orig/exp1_true_ceiling_orig.txt, because the panel uses their original-environment
(torch-1.4) dumps (stems monoflex_orig / monoground_orig).

Kernels (max_matching, ceiling_recall, annos_for, native_pool) are copied unchanged from
tools/decomp/exp1_true_ceiling.py / e4_fp_tp_decomp.py -- those are scripts, not libraries;
importing them re-runs and overwrites reports.

Inputs: the released dumps (dump_path) and the auxiliary complete-pool dumps (aux_dump_path:
the five *_val_preflatten.csv and m3drpn_val_floor0.csv, see tools/extensions/_ext.py).
Runtime: about one hour per run on one GPU. Two independent runs are required by the finalize
step (reproducibility gate):
  python tools/extensions/budget_matched.py --run --tag A
  python tools/extensions/budget_matched.py --run --tag B
  python tools/extensions/budget_matched_finalize.py
"""
import os, sys, shutil, datetime, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ext import paths, dump_path, cache_dir, out_path, aux_dump_path  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import ap_corrector_arc as arc  # noqa: E402
from dgp_cop_oracle_matrix import write_kitti, greedy_nms_keep  # noqa: E402
import evaluator.kitti_eval.kitti_common as kc  # noqa: E402
from exact_ap import ap_summaries  # noqa: E402
from evaluator.kitti_eval.eval import calculate_iou_partly, _prepare_data  # noqa: E402

sys.setrecursionlimit(100000)
CACHE = cache_dir("extensions")
NIMG = 3769
val = [int(x) for x in open(arc.VAL_LIST).read().split()]
assert len(val) == NIMG, len(val)
GT = kc.get_label_annos(arc.LABEL_DIR, val)
BOXCOLS = ["sid", "V", "alpha", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
           "h_3d", "w_3d", "l_3d", "x_3d", "y_3d", "z_3d", "ry"]
out = []
WORK = None


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def annos_for(df, score):                       # verbatim exp1_true_ceiling.py
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    return kc.get_label_annos(d, val)


def max_matching(rows, ng):                     # verbatim exp1_true_ceiling.py
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


def ceiling_recall(dt_annos, name):             # verbatim exp1_true_ceiling.py
    overlaps, _, _, _ = calculate_iou_partly(dt_annos, GT, 2)
    (_, _, ig_gts, ig_dets, _, _, n_valid_gt) = _prepare_data(GT, dt_annos, 0, 1)
    sumM = 0
    for i in range(len(GT)):
        ov = overlaps[i]
        ig = np.asarray(ig_gts[i]); idt = np.asarray(ig_dets[i])
        if ov.shape == (len(idt), len(ig)):
            pass
        elif ov.shape == (len(ig), len(idt)):
            ov = ov.T
        else:
            raise AssertionError(f"{name} img{i}: overlaps {ov.shape} vs dt{len(idt)} gt{len(ig)}")
        vdt = np.where(idt == 0)[0]; vgt = np.where(ig == 0)[0]
        if len(vdt) == 0 or len(vgt) == 0:
            continue
        sub = ov[np.ix_(vdt, vgt)] >= 0.7
        rows = [np.where(sub[d])[0].tolist() for d in range(len(vdt))]
        sumM += max_matching(rows, len(vgt))
    return sumM, int(n_valid_gt)


# ---------------------------------------------------------------- pools
QUERY = {"MonoDETR": "monodetr", "MonoDGP": "monodgp", "MonoCoP": "official_monocop",
         "MonoCLUE": "monoclue", "MonoIA": "monoia"}
CNET = {"MonoDLE": "monodle", "MonoFlex": "monoflex", "GUPNet": "gupnet",
        "DEVIANT": "deviant", "MonoGround": "monoground", "MonoCon": "monocon"}
# the paper's panel uses the ORIGINAL-environment (torch-1.4) dumps for these two (Table 1, starred)
VB_ORIG = {"MonoFlex": dump_path("monoflex_orig"), "MonoGround": dump_path("monoground_orig")}
M3D_PRECAP = 500        # pre-truncate the 3000-deep anchor pool before uniform NMS


def native_pool(name, vb=True):     # verbatim from tools/decomp/e4_fp_tp_decomp.py
    if name in QUERY:
        df = pd.read_csv(aux_dump_path(f"{QUERY[name]}_val_preflatten.csv"))
        if "class_id" in df.columns:
            car = df[df.class_id == df.car_channel].copy()
            car["flatrank"] = df.loc[car.index, "flat_rank"]
        else:
            car = df.copy(); car["flatrank"] = car["flat_rank_car"]
            car["V"] = car["V_car"]; car["cls"] = car["cls_car"]
        pred = (car.flatrank < 50) & (car.cls >= 0.2)
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
    src = VB_ORIG[name] if (vb and name in VB_ORIG) else dump_path(CNET[name])
    df = pd.read_csv(src)
    thr = 0.4 if name == "MonoCon" else 0.2
    col = "cls" if name == "MonoDLE" else "V"
    return df[df[col] >= thr].reset_index(drop=True)


def complete_pool(name):
    """COMPLETE pre-selection pool = stage-A pool of tools/decomp/pool_waterfall.py."""
    if name in QUERY:
        df = pd.read_csv(aux_dump_path(f"{QUERY[name]}_val_preflatten.csv"))
        if "class_id" in df.columns:
            car = df[df.class_id == df.car_channel].copy()
        else:
            car = df.copy(); car["V"] = car["V_car"]
        return car[BOXCOLS].reset_index(drop=True)
    if name == "M3D-RPN":
        cache = os.path.join(CACHE, f"_m3drpn_top{M3D_PRECAP}.csv")
        if os.path.exists(cache):
            return pd.read_csv(cache)
        keep = []
        for ch in pd.read_csv(aux_dump_path("m3drpn_val_floor0.csv"), usecols=BOXCOLS,
                              chunksize=2_000_000):
            keep.append(ch.sort_values("V", ascending=False, kind="stable")
                          .groupby("sid").head(M3D_PRECAP))
        df = (pd.concat(keep).sort_values("V", ascending=False, kind="stable")
                .groupby("sid").head(M3D_PRECAP).reset_index(drop=True))
        df.to_csv(cache, index=False)
        return df
    src = VB_ORIG.get(name, dump_path(CNET[name]))
    return pd.read_csv(src)[BOXCOLS].reset_index(drop=True)


def topN(df, N):
    return (df.sort_values("V", ascending=False, kind="stable")
              .groupby("sid").head(N).reset_index(drop=True))


def nms_pool(df, thr=0.5):
    kept = []
    for sid, g in df.groupby("sid", sort=False):
        kept.append(g[greedy_nms_keep(g, g["V"].values, thr)])
    return pd.concat(kept).reset_index(drop=True)


def measure(pool, name):
    V = pool["V"].values.astype(float)
    dt = annos_for(pool, V)
    base = ap_summaries(GT, dt)["allpoint"]
    sumM, n_gt = ceiling_recall(dt, name)
    assert n_gt == 7874, f"n_valid_gt drifted to {n_gt} on {name}"   # GT-corruption guard
    return base, 100.0 * sumM / n_gt, sumM / n_gt


MODELS = ["M3D-RPN", "MonoDLE", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround", "MonoCon",
          "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]
# frozen: reports/exp1_true_ceiling.txt (2026-06-20)
GATE_REF = {"MonoDLE": (15.09, 0.291), "GUPNet": (17.11, 0.292), "DEVIANT": (17.48, 0.294),
            "MonoCon": (19.59, 0.310), "MonoDETR": (21.18, 0.326), "MonoDGP": (22.82, 0.344),
            "MonoCoP": (24.34, 0.360), "MonoCLUE": (24.55, 0.359), "MonoIA": (25.18, 0.370),
            "M3D-RPN": (11.51, 0.232)}
# frozen: reports_orig/exp1_true_ceiling_orig.txt (2026-07-03)
GATE_REF.update({"MonoFlex": (18.08, 0.296), "MonoGround": (19.38, 0.314)})
NS = [5, 10, 20]
LAB = {"MonoFlex": "MonoFlex*", "MonoGround": "MonoGround*"}


def main():
    global WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="(accepted for compatibility)")
    ap.add_argument("--tag", default="A")
    ap.add_argument("--out", default=None,
                    help="default: reports_rerun/extensions/_bm_run<TAG>.txt")
    ap.add_argument("--gate", nargs="*", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = out_path(f"extensions/_bm_run{a.tag}.txt")
    WORK = os.path.join(CACHE, f"_bmwork{a.tag}")

    w(f"# budget_matched run {a.tag} {datetime.datetime.now().isoformat(timespec='seconds')}")
    w("# script tools/extensions/budget_matched.py; kernels copied verbatim from")
    w("#   tools/decomp/exp1_true_ceiling.py (annos_for/max_matching/ceiling_recall/native_pool)")
    w("# metric: Car Moderate IoU3D 0.7, all-point interpolated AP, 3769 val images, 7874 valid GTs")
    w("")

    w("== GATE: native_pool() reproduces frozen exp1_true_ceiling ==")
    w(f"{'detector':12s} {'base':>7s} {'ref':>7s} {'recall*':>8s} {'ref':>7s}  verdict")
    gate_names = a.gate if a.gate else MODELS
    nfail = 0
    for name in gate_names:
        ref = GATE_REF[name]
        b, c, r = measure(native_pool(name), name)
        ok = abs(b - ref[0]) < 0.02 and abs(r - ref[1]) < 0.0015
        nfail += (not ok)
        w(f"{LAB.get(name,name):12s} {b:7.2f} {ref[0]:7.2f} {r:8.3f} {ref[1]:7.3f}  "
          f"{'PASS' if ok else 'FAIL'}")
    w(f"gate: {len(gate_names)-nfail}/{len(gate_names)} PASS")
    w("")
    if a.gate is not None:
        open(a.out, "w").write("\n".join(out) + "\n")
        return

    w("# COVERAGE vs ORDERING share of missing AP at an IDENTICAL candidate budget N.")
    w("#   base = all-point AP ranked by native score V")
    w("#   AP*  = 100*M/n_gt, M = max bipartite matching (pool pred x valid Moderate GT) @ IoU3D>=0.7")
    w("#   ordering share = (AP*-base)/(100-base) ; coverage share = (100-AP*)/(100-base)")
    w("# P1 = complete pool -> top-N by V             (pure budget match; no threshold, no NMS)")
    w("# P2 = complete pool -> 2D-NMS@0.5 -> top-N by V (matched protocol AND budget)")
    w("# MonoFlex*/MonoGround* use the torch-1.4 original-environment dumps.")
    w(f"# M3D-RPN's 3000-deep pool is pre-capped at top-{M3D_PRECAP}/img by V before NMS.")
    w("")
    rows = {}
    for name in MODELS:
        cp = complete_pool(name)
        npl = nms_pool(cp)
        cps = len(cp) / NIMG; nps = len(npl) / NIMG
        w(f"# {LAB.get(name,name):12s} complete={cps:8.2f}/img  after-uniform-NMS={nps:6.2f}/img")
        for N in NS:
            for tag, src in (("P1", cp), ("P2", npl)):
                pool = topN(src, N)
                sz = len(pool) / NIMG
                b, c, r = measure(pool, name)
                miss = 100.0 - b
                rows[(name, N, tag)] = (b, c, r, sz, (c - b) / miss, (100.0 - c) / miss)
                w(f"[cell] {LAB.get(name,name):12s} N={N:3d} {tag} pool/img={sz:5.2f} "
                  f"base={b:6.2f} AP*={c:6.2f} recall*={r:.4f} "
                  f"order_sh={(c-b)/miss:.4f} cover_sh={(100.0-c)/miss:.4f}")
        open(a.out, "w").write("\n".join(out) + "\n")     # incremental
    w("")
    for tag in ("P1", "P2"):
        for N in NS:
            w("=" * 100)
            w(f"PROTOCOL {tag}   BUDGET N={N}  (identical for all twelve)")
            w(f"{'detector':12s} {'pool/img':>8s} {'base':>7s} {'AP*':>7s} {'recall*':>8s} "
              f"{'missAP':>7s} {'order_AP':>9s} {'order_sh':>9s} {'cover_sh':>9s}")
            va = []
            for name in MODELS:
                b, c, r, sz, osh, csh = rows[(name, N, tag)]
                w(f"{LAB.get(name,name):12s} {sz:8.2f} {b:7.2f} {c:7.2f} {r:8.4f} {100-b:7.2f} "
                  f"{c-b:+9.2f} {osh:9.4f} {csh:9.4f}")
                va.append((c - b, osh, csh, r))
            va = np.array(va)
            w(f"band   : recall* {va[:,3].min():.3f}-{va[:,3].max():.3f} | ordering gap "
              f"{va[:,0].min():+.2f}..{va[:,0].max():+.2f} AP | ordering share "
              f"{va[:,1].min():.3f}-{va[:,1].max():.3f} | coverage share "
              f"{va[:,2].min():.3f}-{va[:,2].max():.3f}")
            w(f"medians: ordering gap {np.median(va[:,0]):+.2f} AP | ordering share "
              f"{np.median(va[:,1]):.3f} | coverage share {np.median(va[:,2]):.3f} | "
              f"coverage miss 1-recall* {1-np.median(va[:,3]):.3f}")
            w(f"coverage share > ordering share: {int((va[:,2] > va[:,1]).sum())}/12")
    w("")
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    open(a.out, "w").write("\n".join(out) + "\n")
    print(f"[written] {a.out}")


main()

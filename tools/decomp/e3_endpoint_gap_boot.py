"""Produces reports/final_run/e3_endpoint_gap_boot.txt.

E3: endpoint ordering-gap difference under paired drive-cluster bootstrap.
Stat per replicate (identical drive resample across all four arms):
    dGap = (AP_ceil(B) - AP_base(B)) - (AP_ceil(A) - AP_base(A))
Pairs: (MonoDLE -> MonoIA) and (M3D-RPN -> MonoIA). S5 uniform pools (the battery where
ceiling scores o_act exist), dense-401 primary / official R40 secondary, B=1000,
RandomState(0), the same machinery class as bootstrap_floor.py.
The o_act cache (_bridgecache_<f>.npz) is the one gap_exact.py writes; if it is absent it is
built here with the same call (depth_share_bridge.iou_act_and_zstar on the S5 pool).
Drive map: frame_sequence.py (built from the KITTI devkit mapping).
Run from the repository root: python tools/decomp/e3_endpoint_gap_boot.py
"""
import os, sys, shutil, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import pool_mask, apply_nms, write_kitti
import evaluator.kitti_eval.kitti_common as kc
from fast_subset_dense import CachedEvalDense

DIAG = cache_dir("decomp")   # work dirs + npz caches (dumps are read through dump_path)
OUT = out_path("final_run/e3_endpoint_gap_boot.txt")
NBOOT = 1000
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


from frame_sequence import load_frame_sequence
fsq = load_frame_sequence()
drive_of = dict(zip(fsq.frame_idx.astype(int), fsq.drive.astype(str)))
val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)


def arms_for(f):
    """build (base_annos, ceil_annos) for detector dump key f on the S5 pool."""
    df = pd.read_csv(dump_path(f)).reset_index(drop=True)
    pre = df[pool_mask(df, "thr0.2")].copy()
    keep = apply_nms(pre, pre["V"].values, 0.5)
    kept = pre[keep].copy().reset_index(drop=True)
    cache_f = f"{DIAG}/_bridgecache_{f}.npz"
    if not os.path.exists(cache_f):   # release: build the cache exactly as gap_exact.py does
        from depth_share_bridge import iou_act_and_zstar
        o_act_, o_z_ = iou_act_and_zstar(kept)
        np.savez_compressed(cache_f, o_act=o_act_, o_z=o_z_)
    z = np.load(cache_f)
    o_act = z["o_act"]
    assert len(o_act) == len(kept), f"cache misaligned {f}"
    annos = {}
    for tag, sc in (("base", kept["V"].values.astype(float)), ("ceil", o_act)):
        d = f"{DIAG}/_e3work/{f}_{tag}"
        if os.path.exists(d):
            shutil.rmtree(d)
        write_kitti(kept, np.asarray(sc, float), os.path.join(d, "data"), val)
        annos[tag] = kc.get_label_annos(os.path.join(d, "data"), val)
    return annos


w(f"# e3_endpoint_gap_boot run {datetime.datetime.now().isoformat(timespec='seconds')}")
w(f"# dGap = gap(B) - gap(A) per identical drive resample; B={NBOOT}; S5 pools; dense-401 (R40)")
w("")
pos_of = {}
for i, s in enumerate(val):
    pos_of.setdefault(drive_of.get(int(s), "NONE"), []).append(i)
drives = sorted(pos_of)
PAIRS = [("monodle", "monoia", "MonoDLE->MonoIA"), ("m3drpn", "monoia", "M3D-RPN->MonoIA")]
cache = {}
for fa, fb, tag in PAIRS:
    for f in (fa, fb):
        if f not in cache:
            annos = arms_for(f)
            cache[f] = {t: CachedEvalDense(GT, annos[t]) for t in ("base", "ceil")}
            ab = cache[f]["base"].eval_subset_pair(range(len(val)))
            ac = cache[f]["ceil"].eval_subset_pair(range(len(val)))
            w(f"[arm] {f:10s} base={ab[0]:.2f}({ab[1]:.2f}) ceil={ac[0]:.2f}({ac[1]:.2f}) "
              f"gap={ac[0]-ab[0]:+.2f}({ac[1]-ab[1]:+.2f})")
    ces = (cache[fa], cache[fb])
    pntA = [ces[0][t].eval_subset_pair(range(len(val))) for t in ("base", "ceil")]
    pntB = [ces[1][t].eval_subset_pair(range(len(val))) for t in ("base", "ceil")]
    point = (pntB[1][0] - pntB[0][0]) - (pntA[1][0] - pntA[0][0])
    point40 = (pntB[1][1] - pntB[0][1]) - (pntA[1][1] - pntA[0][1])
    rng = np.random.RandomState(0)
    dd = np.zeros(NBOOT); dr = np.zeros(NBOOT)
    for b in range(NBOOT):
        sample = rng.choice(len(drives), size=len(drives), replace=True)
        pos = []
        for k in sample:
            pos.extend(pos_of[drives[k]])
        gA = [ces[0][t].eval_subset_pair(pos) for t in ("base", "ceil")]
        gB = [ces[1][t].eval_subset_pair(pos) for t in ("base", "ceil")]
        dd[b] = (gB[1][0] - gB[0][0]) - (gA[1][0] - gA[0][0])
        dr[b] = (gB[1][1] - gB[0][1]) - (gA[1][1] - gA[0][1])
        if b % 100 == 0:
            print(f"  {tag} rep {b}/{NBOOT}", flush=True)
    lo, hi = np.percentile(dd, [2.5, 97.5]); lo4, hi4 = np.percentile(dr, [2.5, 97.5])
    w(f"[done] {tag:18s} point dGap={point:+.3f} CI[{lo:+.3f},{hi:+.3f}] "
      f"(R40 {point40:+.3f} CI[{lo4:+.3f},{hi4:+.3f}]) zero-in-CI={'YES' if lo <= 0 <= hi else 'NO'}")
w("")
w("READ: zero-in-CI=YES -> an endpoint gap CHANGE is not resolvable under drive resampling;")
w("the matching reading is then 'no resolvable endpoint change', not 'did not shrink'.")
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

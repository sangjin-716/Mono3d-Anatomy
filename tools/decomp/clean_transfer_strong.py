"""Ported from tools/decomp/clean_transfer_strong.py for the public release. Computation unchanged.
Produces reports/clean_transfer_strong.txt.

Clean drive-disjoint cal->test corrector transfer, NO retraining (val is already out-of-sample
for every detector). Replicates the clean-transfer result on STRONG detectors + a non-DETR family,
with DRIVE-grouped (not image-random) folds and 3 seeds.

For each detector: split val DRIVES drive-disjoint into cal(~20%)/test(~80%), train the canonical
HGB depth corrector on cal matched-TPs, apply to test, eval AP on the test frames. transfer ΔAP ≤ 0
across detectors/families/seeds = strong replication of "post-hoc did not transfer (drive-clean)".
Drive map: frame_sequence.py (built from the KITTI devkit mapping).
Run from the repository root: python tools/decomp/clean_transfer_strong.py
"""
import os, sys, shutil
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti, pool_mask, apply_nms
from evaluator.kitti_utils import Calibration
import evaluator.kitti_eval.kitti_common as kc
from evaluator.kitti_eval.eval import do_eval
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

DIAG = cache_dir("decomp")   # work dirs + npz caches (dumps are read through dump_path)
DET = arc.DET
ov07 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7]] * 3)
ov05 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
MO = np.stack([ov07, ov05], 0)[:, :, [0]]
WORK = f"{DIAG}/_cleantr"
from frame_sequence import load_frame_sequence
fs = load_frame_sequence()
drive_of = dict(zip(fs.frame_idx, fs.drive))
DETS = [("MonoDGP", "dgp", "DETR"), ("MonoCoP", "official_monocop", "DETR"),
        ("GUPNet", "gupnet", "CenterNet"), ("MonoFlex", "monoflex", "CenterNet")]


def gbm():
    return HGB(max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=40,
               random_state=0, early_stopping=False)


def ev(df, score, znew, tx, ty, x0, y0, z0, h3, frames):
    s = znew / z0
    d = df.copy(); d["x_3d"] = tx + (x0 - tx) * s; d["y_3d"] = (h3 / 2. + ty) + (y0 - h3 / 2. - ty) * s; d["z_3d"] = znew
    fset = set(frames)
    d = d[d.sid.isin(fset)]
    base = d[pool_mask(d, "thr0.2")].copy()
    keep = apply_nms(base, base["V"].values, 0.5); base = base[keep].copy()
    sc = score[base.index.values] if hasattr(score, "__len__") and len(score) == len(df) else base["V"].values
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    dd = os.path.join(WORK, "data"); write_kitti(base.reset_index(drop=True), base["V"].values, dd, frames)
    GTf = kc.get_label_annos(arc.LABEL_DIR, frames)
    return float(do_eval(GTf, kc.get_label_annos(dd, frames), [0], MO, compute_aos=False, DIForDIS=True)[6][0, 1, 0])


print("=" * 86)
print("CLEAN drive-disjoint cal->test corrector transfer (no retraining; strong detectors; 3 seeds)")
print("=" * 86)
print(f"  {'detector':12s} {'family':10s} {'base(test)':>11} {'transfer dAP (mean±sd over 3 seeds)':>34}")
allrows = []
for name, f, fam in DETS:
    vp = dump_path(f)
    if not os.path.exists(vp):
        print(f"  {name}: dump missing"); continue
    df = pd.read_csv(vp).reset_index(drop=True)
    gtz, dz = arc.match(df); mt = ~np.isnan(dz)
    X = df[DET].astype(float).values; z = df.z_3d.values
    tx = np.zeros(len(df)); ty = np.zeros(len(df))
    for sid, g in df.groupby("sid"):
        cal = Calibration(os.path.join(arc.CALIB_DIR, f"{int(sid):06d}.txt")); i = g.index.values
        tx[i] = cal.tx; ty[i] = cal.ty
    x0 = df.x_3d.values; y0 = df.y_3d.values; z0 = df.z_3d.values; h3 = df.h_3d.values
    sids = df.sid.values.astype(int)
    drv = np.array([str(drive_of.get(s, "NONE")) for s in sids])
    udrv = np.unique(drv[drv != "NONE"])
    transfers = []; bases = []
    for seed in range(3):
        rng = np.random.RandomState(seed); perm = rng.permutation(udrv)
        ncal = max(1, int(0.2 * len(udrv)))
        cal_d = set(perm[:ncal].tolist()); test_d = set(perm[ncal:].tolist())
        cal_mask = mt & np.isin(drv, list(cal_d))
        test_frames = sorted({int(s) for s in sids[np.isin(drv, list(test_d))]})
        if cal_mask.sum() < 200 or len(test_frames) < 100:
            continue
        m = gbm().fit(X[cal_mask], dz[cal_mask])
        dz_hat = m.predict(X)
        b = ev(df, df["V"].values, z, tx, ty, x0, y0, z0, h3, test_frames)
        c = ev(df, df["V"].values, z - dz_hat, tx, ty, x0, y0, z0, h3, test_frames)
        transfers.append(c - b); bases.append(b)
    if transfers:
        t = np.array(transfers)
        print(f"  {name:12s} {fam:10s} {np.mean(bases):>11.2f} {np.mean(t):>+18.2f} ± {np.std(t):.2f}   (seeds: {', '.join(f'{x:+.2f}' for x in t)})")
        allrows.append((name, fam, np.mean(t), np.std(t)))

print("\n  VERDICT:")
if allrows and all(r[2] <= 0.05 for r in allrows):
    print("  All transfers <= ~0 across detectors/families/seeds -> clean post-hoc transfer FAILS broadly")
    print("  (drive-grouped, strong detectors, no retraining) -> the earlier clean-transfer result REPLICATES; the image-fold")
    print("  OOF confound is removed (this split is drive-disjoint).")
else:
    print("  Some transfer > 0 -> NOT a clean broad failure; re-examine.")
if os.path.exists(WORK):
    shutil.rmtree(WORK)
OUT = out_path("clean_transfer_strong.txt")
with open(OUT, "w") as fh:
    for name, fam, mt_, sd in allrows:
        fh.write(f"{name} ({fam}): transfer dAP = {mt_:+.2f} ± {sd:.2f} (3 drive-disjoint seeds)\n")
print(f"\n[written] {OUT}")

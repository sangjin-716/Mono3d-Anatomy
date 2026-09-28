"""Ported from mono3d_crossdataset/tools/transfer_save_orig.py for the public release.
Computation unchanged. Produces the prediction dirs consumed by bootstrap_orig.py --transfer
(no report of its own).

Bootstrap prerequisite: refit the clean-transfer correctors for the original-environment
MonoFlex*/MonoGround* dumps (2 detectors x 3 seeds, identical recipe to transfer_orig.py /
tools/decomp/clean_transfer_strong.py) and SAVE the base / corrected S5 prediction dirs +
test-frame lists for the paired drive-cluster bootstrap (a copy of
tools/decomp/clean_transfer_save.py with inputs switched).
Out: <CACHE_DIR>/orig/_bootstrap_preds_orig/<f>_s<seed>_{base,corr}/data + <f>_s<seed>_frames.txt
     + provenance.json (script, dump sha8, recipe, timestamp).
Run from the repository root: python tools/orig/transfer_save_orig.py
"""
import os, sys, shutil, json, hashlib, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
from tools.orig._orig_common import ORIG
import numpy as np, pandas as pd
import ap_corrector_arc as arc
from dgp_cop_oracle_matrix import write_kitti, pool_mask, apply_nms
from evaluator.kitti_utils import Calibration
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from frame_sequence import frame_sequence_csv

OUTB = f"{ORIG}/_bootstrap_preds_orig"
DET = arc.DET
fs = pd.read_csv(frame_sequence_csv())
drive_of = dict(zip(fs.frame_idx, fs.drive))
DETS = [("MonoFlex*", "monoflex_orig"), ("MonoGround*", "monoground_orig")]


def gbm():
    return HGB(max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=40,
               random_state=0, early_stopping=False)


def sha8(p):
    return hashlib.sha256(open(p, "rb").read(1 << 22)).hexdigest()[:8]


os.makedirs(OUTB, exist_ok=True)
prov = {"script": "tools/orig/transfer_save_orig.py (ported verbatim)",
        "recipe": "identical to clean_transfer_strong.py (HGB seed0, drive-disjoint 20/80)",
        "time": datetime.datetime.now().isoformat(timespec="seconds"), "sets": {}}
for name, f in DETS:
    vp = dump_path(f)
    df = pd.read_csv(vp).reset_index(drop=True)
    gtz, dz = arc.match(df); mt = ~np.isnan(dz)
    X = df[DET].astype(float).values; z0 = df.z_3d.values
    tx = np.zeros(len(df)); ty = np.zeros(len(df))
    for sid, g in df.groupby("sid"):
        cal = Calibration(os.path.join(arc.CALIB_DIR, f"{int(sid):06d}.txt"))
        i = g.index.values; tx[i] = cal.tx; ty[i] = cal.ty
    x0 = df.x_3d.values; y0 = df.y_3d.values; h3 = df.h_3d.values
    sids = df.sid.values.astype(int)
    drv = np.array([str(drive_of.get(s, "NONE")) for s in sids])
    udrv = np.unique(drv[drv != "NONE"])
    for seed in range(3):
        rng = np.random.RandomState(seed); perm = rng.permutation(udrv)
        ncal = max(1, int(0.2 * len(udrv)))
        cal_d = set(perm[:ncal].tolist()); test_d = set(perm[ncal:].tolist())
        cal_mask = mt & np.isin(drv, list(cal_d))
        test_frames = sorted({int(s) for s in sids[np.isin(drv, list(test_d))]})
        if cal_mask.sum() < 200 or len(test_frames) < 100:
            continue
        m = gbm().fit(X[cal_mask], dz[cal_mask])
        znew = z0 - m.predict(X)
        s = znew / z0
        for variant, dz_apply in (("base", False), ("corr", True)):
            d2 = df.copy()
            if dz_apply:
                d2["x_3d"] = tx + (x0 - tx) * s
                d2["y_3d"] = (h3 / 2. + ty) + (y0 - h3 / 2. - ty) * s
                d2["z_3d"] = znew
            fset = set(test_frames)
            d2 = d2[d2.sid.isin(fset)]
            base = d2[pool_mask(d2, "thr0.2")].copy()
            keep = apply_nms(base, base["V"].values, 0.5); base = base[keep].copy()
            tag = f"{f}_s{seed}_{variant}"
            dd = os.path.join(OUTB, tag, "data")
            if os.path.exists(os.path.dirname(dd)):
                shutil.rmtree(os.path.dirname(dd))
            write_kitti(base.reset_index(drop=True), base["V"].values.astype(float), dd, test_frames)
            prov["sets"][tag] = {"dump_sha8": sha8(vp), "n_frames": len(test_frames)}
        with open(os.path.join(OUTB, f"{f}_s{seed}_frames.txt"), "w") as fh:
            fh.write("\n".join(str(x) for x in test_frames))
    print(f"[saved] {name} (3 seeds x base/corr)", flush=True)
json.dump(prov, open(os.path.join(OUTB, "provenance.json"), "w"), indent=1)
print("[done]", flush=True)

"""Produces reports/drive_grouped_oof.md (+ drive_grouped_oof_results.json next to it).

Phase 1: drive-grouped OOF vs image-fold OOF (no detector retraining; existing dumps).
Same canonical settings (S5, IoU3D>=.05/|dz|<8 match, DET15, GBM OMP=1, calib-ray reproj, official eval,
native V fixed). HEADLINE = POOLED cross-fitted AP: each val box predicted by the fold where it is held out,
all concatenated -> ONE official KITTI eval. NOT the mean of per-fold APs.
image fold = rank(sid)%5 ; drive fold = GroupKFold(group=raw_drive, n=5).
The per-box depth residual dz (ap_corrector_arc.match) is cached in <CACHE_DIR>/decomp/_canon_cache
keyed by the dump's sha256 prefix; it is built on first use.
Drive map: frame_sequence.py (built from the KITTI devkit mapping).
Run from the repository root: python tools/decomp/phase1_drive_oof.py [<dump stem>]
"""
import os, sys, json, hashlib, shutil
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

DET = arc.DET; DIAG = cache_dir("decomp"); CACHE = f"{DIAG}/_canon_cache"
DETS = [("MonoDGP", "dgp"), ("MonoDETR", "monodetr"), ("MonoCoP", "official_monocop"),
        ("MonoFlex", "monoflex"), ("GUPNet", "gupnet")]
val_list = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val_list)
ov07 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7]] * 3)
ov05 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
MO = np.stack([ov07, ov05], 0)[:, :, [0]]
from frame_sequence import load_frame_sequence
drive_of = dict(zip(load_frame_sequence().frame_idx,
                    load_frame_sequence().drive))
def gbm(): return HGB(max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=40, random_state=0, early_stopping=False)
def sha8(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

def run_det(name, f):
    vp = dump_path(f); va = pd.read_csv(vp).reset_index(drop=True)
    cf = f"{CACHE}/{f}_val_{sha8(vp)}.npz"
    if not os.path.exists(cf):   # release: build the dz cache as the canonical harness did
        os.makedirs(CACHE, exist_ok=True); gtz_, dz_ = arc.match(va); np.savez(cf, gtz=gtz_, dz=dz_)
    dv = np.load(cf)["dz"]; mv = ~np.isnan(dv)
    tx = np.zeros(len(va)); ty = np.zeros(len(va))
    for sid, g in va.groupby("sid"):
        cal = Calibration(os.path.join(arc.CALIB_DIR, f"{int(sid):06d}.txt")); i = g.index.values; tx[i] = cal.tx; ty[i] = cal.ty
    h3 = va.h_3d.values; x0 = va.x_3d.values; y0 = va.y_3d.values; z0 = va.z_3d.values
    def reproj(z): s = z / z0; return tx + (x0 - tx) * s, (h3 / 2. + ty) + (y0 - h3 / 2. - ty) * s, z
    WORK = f"{DIAG}/_p1_{f}"
    def ev(znew):
        vx, vy, vz = reproj(znew); d = va.copy(); d["x_3d"] = vx; d["y_3d"] = vy; d["z_3d"] = vz
        base = d[pool_mask(d, "thr0.2")].copy(); keep = apply_nms(base, base["V"].values, 0.5); base = base[keep].copy()
        if os.path.exists(WORK): shutil.rmtree(WORK)
        dd = os.path.join(WORK, "data"); write_kitti(base, base["V"].values, dd, val_list)
        return float(do_eval(GT, kc.get_label_annos(dd, val_list), [0], MO, compute_aos=False, DIForDIS=True)[6][0, 1, 0])
    X = va[DET].astype(float).values; z = va.z_3d.values
    sids = va["sid"].values
    base_ap = ev(z)
    def pooled_oof(fold_id):
        pred = np.zeros(len(va))
        for k in np.unique(fold_id):
            tr = mv & (fold_id != k); te = (fold_id == k)
            pred[te] = gbm().fit(X[tr], dv[tr]).predict(X[te])
        return pred
    # image fold
    uniq = np.unique(sids); rmap = {s: i for i, s in enumerate(uniq)}
    img_fold = np.array([rmap[s] % 5 for s in sids])
    # drive fold (GroupKFold by drive)
    drives = np.array([drive_of[int(s)] for s in sids])
    gkf = GroupKFold(n_splits=5); drv_fold = np.zeros(len(va), int)
    for fi, (_, te) in enumerate(gkf.split(X, dv, groups=drives)): drv_fold[te] = fi
    pio = pooled_oof(img_fold); pdo = pooled_oof(drv_fold)
    img_ap = ev(z - pio); drv_ap = ev(z - pdo)
    r2_img = r2_score(dv[mv], pio[mv]); r2_drv = r2_score(dv[mv], pdo[mv])
    # per-fold stats (drive fold)
    fold_stats = []
    for k in range(5):
        te = drv_fold == k
        fold_stats.append(dict(fold=k, n_drives=int(len(np.unique(drives[te]))),
                               n_images=int(len(np.unique(sids[te]))), n_matched=int((mv & te).sum())))
    dsz = pd.Series(drives).value_counts().values
    return dict(detector=name, dump=f, val_sha=sha8(vp), base_AP=round(base_ap, 3),
                image_OOF_dAP=round(img_ap - base_ap, 3), drive_OOF_dAP=round(drv_ap - base_ap, 3),
                image_OOF_R2=round(r2_img, 4), drive_OOF_R2=round(r2_drv, 4),
                n_drives=int(len(np.unique(drives))), n_matched_val=int(mv.sum()),
                drive_size_max=int(dsz.max()), drive_size_median=int(np.median(dsz)), drive_size_min=int(dsz.min()),
                fold_stats=fold_stats)

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    dets = [d for d in DETS if (only == "all" or d[1] == only)]
    rows = []
    for name, f in dets:
        print(f"== {name} ==", flush=True); r = run_det(name, f); rows.append(r)
        print(f"  base {r['base_AP']} | image-OOF {r['image_OOF_dAP']:+} (R2 {r['image_OOF_R2']}) | "
              f"drive-OOF {r['drive_OOF_dAP']:+} (R2 {r['drive_OOF_R2']}) | drives {r['n_drives']}", flush=True)
        json.dump({"config": "OMP=1, GBM(300,4,0.05,40,rs0,es=False), S5, DET15, pooled-cross-fit AP", "results": rows},
                  open(out_path("drive_grouped_oof_results.json"), "w"), indent=2)
    # markdown
    with open(out_path("drive_grouped_oof.md"), "w") as m:
        m.write("# Phase 1 — drive-grouped vs image-fold OOF (pooled cross-fitted AP)\n\n")
        m.write("Headline = pooled cross-fitted AP (all held-out fold predictions concatenated -> one official KITTI eval). Canonical OMP=1.\n\n")
        m.write("| detector | base AP | image-OOF ΔAP | drive-OOF ΔAP | image R² | drive R² | Δ(img−drive) | #drives |\n")
        m.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            m.write(f"| {r['detector']} | {r['base_AP']} | {r['image_OOF_dAP']:+} | {r['drive_OOF_dAP']:+} | "
                    f"{r['image_OOF_R2']} | {r['drive_OOF_R2']} | {r['image_OOF_dAP']-r['drive_OOF_dAP']:+.3f} | {r['n_drives']} |\n")
        m.write("\n## drive size imbalance (val) & per-fold\n")
        for r in rows:
            m.write(f"\n**{r['detector']}** drives={r['n_drives']} size max/med/min={r['drive_size_max']}/{r['drive_size_median']}/{r['drive_size_min']}\n")
            for fsx in r["fold_stats"]:
                m.write(f"  - fold{fsx['fold']}: {fsx['n_drives']} drives, {fsx['n_images']} imgs, {fsx['n_matched']} matched\n")
    print(f"\n-> {out_path('drive_grouped_oof.md')} + {out_path('drive_grouped_oof_results.json')}", flush=True)

"""Prints the "score AND geometry at 2 dp -> 17.14" line of
reports/extensions/class_gate_audit.txt (supplementary Sec. O). Needs only the released DEVIANT dump.
Run from the repository root: python tools/extensions/partb_anchor_mech2.py

Mechanism, step 2: reproduce the DEVIANT Part-B anchor 17.14 from the Part-A pool
by applying BOTH roundings the 2-dp txt dump imposes (score AND box geometry).

Step 1 (partb_anchor_gate.py) showed score-only rounding gives 17.20 vs Part-B 17.14.
write_kitti emits geometry at 4 dp and score at 6 dp; the Part-B dump is 2 dp throughout.
"""
import os, sys, shutil
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ext import dump_path, cache_dir, ROOT  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import ap_corrector_arc as arc  # noqa: E402
import evaluator.kitti_eval.kitti_common as kc  # noqa: E402
from exact_ap import ap_summaries  # noqa: E402

WORK = os.path.join(cache_dir("extensions"), "_pbmech2")
val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)

df = pd.read_csv(dump_path("deviant"))
pool = df[df["V"] >= 0.2].reset_index(drop=True)


def write2dp(pool, score, ddir):
    if os.path.exists(ddir):
        shutil.rmtree(ddir)
    os.makedirs(ddir)
    p = pool.assign(scoreval=np.asarray(score, float))
    for sid, g in p.groupby("sid"):
        with open(os.path.join(ddir, f"{int(sid):06d}.txt"), "w") as f:
            for r in g.itertuples(index=False):
                f.write(f"Car 0.0 0 {r.alpha:.2f} {r.bbox_x1:.2f} {r.bbox_y1:.2f} "
                        f"{r.bbox_x2:.2f} {r.bbox_y2:.2f} {r.h_3d:.2f} {r.w_3d:.2f} "
                        f"{r.l_3d:.2f} {r.x_3d:.2f} {r.y_3d:.2f} {r.z_3d:.2f} "
                        f"{r.ry:.2f} {r.scoreval:.2f}\n")
    for sid in val:
        q = os.path.join(ddir, f"{sid:06d}.txt")
        if not os.path.exists(q):
            open(q, "w").close()


d = os.path.join(WORK, "data")
write2dp(pool, pool["V"].values.astype(float), d)
dt = kc.get_label_annos(d, val)
ap = ap_summaries(GT, dt, current_class=0, difficulty=1, min_overlap=0.7)["allpoint"]
REF_B = 17.14
print("---- MECHANISM GATE, step 2 (both roundings, as the 2-dp dump writes them) ----")
print(f"score AND geometry at 2 dp -> base {ap:.2f}  vs Part-B frozen {REF_B:.2f}  "
      f"(d={abs(ap-REF_B):.3f})  [{'PASS' if abs(ap-REF_B) < 0.05 else 'FAIL'}]")
print(f"full-precision Part-A reference = 17.48 ; total quantisation cost {ap-17.48:+.2f} AP")

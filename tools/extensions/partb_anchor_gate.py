"""Ported from the camera-ready check partb_anchor_gate.py (same file name) for the public release.
Computation unchanged. Prints (stdout) the mechanism and per-detector Car-anchor lines of
reports/extensions/class_gate_audit.txt (supplementary Sec. O: DEVIANT 17.14 vs 17.48; three
kept detectors within 0.01). Needs only the released DEVIANT dump and the frozen
reports/extensions/difficulty_class_extension.csv (override with --csv).

Part-B Car-anchor gate + score-quantisation mechanism test.

A check found DEVIANT's Part-B Car/Moderate base = 17.14 against the frozen
Part-A / Table-1 value 17.48 (d = 0.34 AP), while the other three Part-B pools
reproduce their anchors to <= 0.01 AP.

Diagnosis established before this script ran:
  the Part-B directory DEVIANT/output/run_221/result_kitti/data holds EXACTLY the
  same 10,186 Car predictions as the Part-A pool (identical per-image counts), but
  written with the score field at 2 decimal places -> only 69 distinct score values
  among 10,186 boxes (99.32% tie ratio).  The other three dirs carry 6-20 decimals.

MECHANISM GATE (this script):
  take the Part-A full-precision DEVIANT pool, round ONLY the score to 2 dp, and
  recompute base all-point AP through the Part-A path.  If the quantisation is the
  cause, this must land on 17.14 -- a value produced by a DIFFERENT process
  (difficulty_class_extension.py Part B) reading a DIFFERENT file (the txt dump).

Also emits the per-detector Part-B Car-anchor gate line, and
recomputes the Pedestrian/Cyclist bands over the pools that pass their anchor.

READ-ONLY on all inputs.  Writes nothing but stdout.
Run from the repository root: python tools/extensions/partb_anchor_gate.py
"""
import os, sys, shutil, copy, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ext import dump_path, cache_dir, ROOT  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import ap_corrector_arc as arc  # noqa: E402
from dgp_cop_oracle_matrix import write_kitti
import evaluator.kitti_eval.kitti_common as kc  # noqa: E402
from exact_ap import ap_summaries  # noqa: E402

WORK = os.path.join(cache_dir("extensions"), "_pbanchor")
CSV = os.path.join(ROOT, "reports", "extensions", "difficulty_class_extension.csv")
if "--csv" in sys.argv:
    CSV = sys.argv[sys.argv.index("--csv") + 1]
val = [int(x) for x in open(arc.VAL_LIST).read().split()]
GT = kc.get_label_annos(arc.LABEL_DIR, val)


def annos_for(df, score):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    d = os.path.join(WORK, "data")
    write_kitti(df.reset_index(drop=True), np.asarray(score, float), d, val)
    return kc.get_label_annos(d, val)


print("=" * 100)
print("Part-B Car-anchor gate and score-quantisation mechanism")
print("=" * 100)

# ---------------------------------------------------------------- mechanism
df = pd.read_csv(dump_path("deviant"))
pool = df[df["V"] >= 0.2].reset_index(drop=True)
V = pool["V"].values.astype(float)
print(f"\nPart-A DEVIANT pool: {len(pool)} Car boxes, "
      f"{len(np.unique(V))} distinct full-precision scores")

dt_full = annos_for(pool, V)
ap_full = ap_summaries(GT, dt_full, current_class=0, difficulty=1,
                       min_overlap=0.7)["allpoint"]

Vq = np.round(V, 2)
print(f"scores rounded to 2 dp: {len(np.unique(Vq))} distinct values "
      f"(tie ratio {1 - len(np.unique(Vq)) / len(Vq):.4f})")
dt_q = annos_for(pool, Vq)
ap_q = ap_summaries(GT, dt_q, current_class=0, difficulty=1,
                    min_overlap=0.7)["allpoint"]

REF_A = 17.48   # Table 1 / Part A frozen
REF_B = 17.14   # Part B, difficulty_class_extension.txt:207, independent process
print("\n---- MECHANISM GATE ----")
print(f"full-precision score  -> base {ap_full:.2f}   vs Part-A/Table-1 frozen {REF_A:.2f}"
      f"   (d={abs(ap_full-REF_A):.3f})  [{'PASS' if abs(ap_full-REF_A)<0.05 else 'FAIL'}]")
print(f"score rounded to 2 dp -> base {ap_q:.2f}   vs Part-B  frozen {REF_B:.2f}"
      f"   (d={abs(ap_q-REF_B):.3f})  [{'PASS' if abs(ap_q-REF_B)<0.05 else 'FAIL'}]")
print(f"quantisation cost = {ap_q - ap_full:+.2f} AP  "
      f"(explains the {REF_B-REF_A:+.2f} anchor miss)")

# ------------------------------------------------- per-detector anchor table
print("\n---- Part-B Car-anchor gate, per detector ----")
ANCH = [("M3D-RPN", 11.51, 11.51), ("MonoFlex", 16.26, 16.27),
        ("DEVIANT", 17.48, 17.14), ("MonoGround", 17.45, 17.46)]
DEC = {"M3D-RPN": "6", "MonoFlex": "1-19", "DEVIANT": "2 (quantised)", "MonoGround": "1-20"}
NDIST = {"M3D-RPN": 6567, "MonoFlex": 7496, "DEVIANT": 69, "MonoGround": 7165}
NCAR = {"M3D-RPN": 12472, "MonoFlex": 17142, "DEVIANT": 10186, "MonoGround": 16719}
ok = []
for n, a, b in ANCH:
    d = abs(b - a)
    v = "PASS" if d <= 0.05 else "FAIL"
    if v == "PASS":
        ok.append(n)
    print(f"  {n:11s} PartB Car/Mod base {b:6.2f} vs PartA/Table-1 {a:6.2f}  d={d:.2f}  "
          f"[{v}]   score decimals={DEC[n]:14s} distinct={NDIST[n]:5d}/{NCAR[n]}")
print(f"  anchors reproducing: {len(ok)}/4  -> {', '.join(ok)}")

# ------------------------------------- class bands over anchor-passing pools
print("\n---- Pedestrian / Cyclist bands, restricted to anchor-passing pools ----")
csv = pd.read_csv(CSV)
sub = csv[csv["part"] == "B"]
for cname in ("Pedestrian", "Cyclist"):
    for dn in ("Easy", "Moderate", "Hard"):
        m = sub[(sub["cls"] == cname) & (sub["difficulty"] == dn)]
        allr = list(zip(m["detector"], m["ordering_gap"]))
        keep = [(d, g) for d, g in allr if d in ok]
        if not keep:
            continue
        g4 = [g for _, g in allr]
        g3 = [g for _, g in keep]
        drop = [d for d, _ in allr if d not in ok]
        print(f"  {cname:10s} {dn:8s} all-{len(g4)} {min(g4):+.2f}..{max(g4):+.2f} "
              f"med {np.median(g4):+.2f}   |  anchor-passing-{len(g3)} "
              f"{min(g3):+.2f}..{max(g3):+.2f} med {np.median(g3):+.2f}"
              f"   (dropped: {','.join(drop) if drop else '-'})")
print("\nDONE")

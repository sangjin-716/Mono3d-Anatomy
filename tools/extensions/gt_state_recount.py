"""Produces reports/extensions/gt_state_recount.txt (a re-run writes
reports_rerun/extensions/gt_state_recount.txt);
supplementary Secs. I and J (panel-unreached 14.8% over all twelve pools, 25.1% over the eleven
non-anchor pools, 2.4% at IoU 0.5, a further 16.3%, CenterNet-style pool existence 0.294 to 0.317).

Recounts the presence-layer shares directly from the two per-GT state matrices:
  vB     : original-environment panel (MonoFlex*, MonoGround*), the CSV written by
           tools/orig/gt_state_matrix_vB.py (default reports_rerun/gt_state_matrix_vB.csv)
  pre-vB : modern-environment panel, the CSV written by tools/decomp/gt_state_matrix.py
           (default cache/decomp/gt_state_matrix.csv)
Its closing adjudication note is not printed (reports/extensions/gt_state_recount.txt leaves it
out too). Read-only. Run from the repository root after the two scripts above:
    python tools/extensions/gt_state_recount.py [--vb CSV] [--modern CSV] [--out TXT]
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ext import cache_dir, out_path, repo_rel  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--vb", default=out_path("gt_state_matrix_vB.csv"))
_ap.add_argument("--modern", default=os.path.join(cache_dir("decomp"), "gt_state_matrix.csv"))
_ap.add_argument("--out", default=out_path("extensions/gt_state_recount.txt"))
_args = _ap.parse_args()

FILES = [
    ("vB  (orig-env MonoFlex*/MonoGround*) 2026-07-03", _args.vb),
    ("pre-vB (modern-env)                  2026-06-12", _args.modern),
]

out = []
def w(s=""):
    print(s, flush=True); out.append(s)

for label, path in FILES:
    rows = list(csv.DictReader(open(path)))
    hdr = rows[0].keys()
    acols = [c for c in hdr if c.endswith("_a")]
    ccols = [c for c in hdr if c.endswith("_c")]
    anchor_a = [c for c in acols if c.startswith("M3D-RPN")]
    non_anchor_a = [c for c in acols if not c.startswith("M3D-RPN")]
    n = len(rows)
    w("=" * 78)
    w(label)
    w("  file : %s" % repo_rel(path))
    w("  n_gt = %d ; pools(_a) = %d ; anchor col = %s" % (n, len(acols), anchor_a))
    for thr in (0.7, 0.5):
        hc12 = sum(1 for r in rows if max(float(r[c]) for c in acols) < thr)
        hc11 = sum(1 for r in rows if max(float(r[c]) for c in non_anchor_a) < thr)
        w("  IoU>=%.1f  hard-core 12 pools = %5d / %d = %.5f -> %.1f%%"
          % (thr, hc12, n, hc12 / n, 100 * hc12 / n))
        w("  IoU>=%.1f  hard-core 11 non-anchor = %5d / %d = %.5f -> %.1f%%"
          % (thr, hc11, n, hc11 / n, 100 * hc11 / n))
    # 'further X%': outside 12-pool hard core, no FINAL accurate box anywhere
    thr = 0.7
    further = sum(1 for r in rows
                  if max(float(r[c]) for c in acols) >= thr
                  and max(float(r[c]) for c in ccols) < thr)
    w("  IoU>=0.7  outside hard core, FINAL accurate nowhere = %d / %d = %.5f -> %.1f%%"
      % (further, n, further / n, 100 * further / n))
    # CenterNet-family pool-existence band
    cnet = [c for c in acols if c.split("_")[0] in
            ("MonoDLE", "MonoFlex", "MonoFlex*", "GUPNet", "DEVIANT",
             "MonoGround", "MonoGround*", "MonoCon")]
    band = sorted(sum(1 for r in rows if float(r[c]) >= 0.7) / n for c in cnet)
    w("  IoU>=0.7  CenterNet-family pool existence band = %.3f..%.3f -> %d--%d%%"
      % (band[0], band[-1], round(100 * band[0]), round(100 * band[-1])))

open(_args.out, "w").write("\n".join(out) + "\n")

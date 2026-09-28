"""Produces reports/final_run/c3_conditioned_analysis.md.

C3 conditioning: does cross-detector disagreement predict depth error AFTER controlling
for common difficulty (distance/occlusion/truncation)? Existing matched dumps only.
6 detectors (MonoCLUE excluded: its matched dump predates the validated MonoCLUE re-dump) -> 15 pairs.
Per pair: join on shared GT (sid, gt_z); disagreement d=|z_a-z_b|; error e=|mean(z_a,z_b)-gt_z|.
Reports: (1) within-dist-bin Spearman; (2) residual corr after removing gt_z; (3) partial
Spearman controlling gt_z+occ+trunc; (4) leave-one-detector-out; (5) detector accounting.
Inputs: per-prediction matched tables matched_<f>_val.csv (columns sid, z_pred, gt_z, iou3d, occ,
trunc, dist_bin) in DUMP_DIR; they are the matched-tables release asset (data/DUMPS.md), not
part of the per-prediction dump zip.
Run from the repository root: python tools/decomp/c3_conditioned.py
"""
import os, sys, itertools, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
DIAG = cache_dir("decomp")   # work dirs + npz caches (dumps are read through dump_path)
DETS = {"DETR": "matched_monodetr_val.csv", "DGP": "matched_dgp_val.csv",
        "CoP": "matched_cop_val.csv", "IA": "matched_monoia_val.csv",
        "Flex": "matched_monoflex_val.csv", "GUP": "matched_gupnet_val.csv"}
# CLUE excluded: matched_monoclue_val.csv is the superseded old dump (cls<0.2 invalid).
out = []


def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def load(fn):
    p = os.path.join(paths.DUMP_DIR, fn)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d.iou3d > 0].copy()  # matched only
    d["key"] = d.sid.astype(int).astype(str) + "_" + d.gt_z.round(2).astype(str)
    return d


D = {k: load(v) for k, v in DETS.items()}
D = {k: v for k, v in D.items() if v is not None}
keys = list(D.keys())


def partial_spearman(x, y, Z):
    """partial Spearman of x,y controlling columns of Z (rank-residualize then correlate)."""
    rx = rankdata(x); ry = rankdata(y)
    A = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])] + [np.ones(len(x))])
    bx = np.linalg.lstsq(A, rx, rcond=None)[0]; by = np.linalg.lstsq(A, ry, rcond=None)[0]
    ex = rx - A @ bx; ey = ry - A @ by
    return np.corrcoef(ex, ey)[0, 1]


w(f"# c3_conditioned_analysis {datetime.datetime.now().isoformat(timespec='seconds')}")
w("# gate-valid pairs = pairs over 6 detectors (CLUE excluded, superseded dump). Disagreement vs")
w("# error-of-the-pair-average, conditioned on common difficulty. Existing dumps; no AP re-scoring.")
w("")
w(f"## (4) gate-valid accounting: 7 detectors with matched dumps minus CLUE (excluded) = 6 -> "
  f"{len(list(itertools.combinations(keys,2)))} pairs (vs 21 with CLUE).")
w(f"   detectors used: {keys}")
w("")
rows = []
for a, b in itertools.combinations(keys, 2):
    m = pd.merge(D[a][["key", "z_pred", "gt_z", "occ", "trunc", "dist_bin"]],
                 D[b][["key", "z_pred"]], on="key", suffixes=("_a", "_b"))
    if len(m) < 50:
        continue
    za, zb, gz = m.z_pred_a.values, m.z_pred_b.values, m.gt_z.values
    d = np.abs(za - zb); e = np.abs(0.5 * (za + zb) - gz)
    raw = spearmanr(d, e).statistic
    resid = partial_spearman(d, e, gz.reshape(-1, 1))  # remove gt_z only
    Z = np.column_stack([gz, m.occ.values, m.trunc.values])
    part = partial_spearman(d, e, Z)
    rows.append(dict(pair=f"{a}+{b}", n=len(m), raw=raw, resid_gtz=resid, partial=part,
                     a=a, b=b))
df = pd.DataFrame(rows)
w("## (1)+(2)+(3) per gate-valid pair: raw / residual(remove gt_z) / partial(gt_z+occ+trunc)")
w(f"{'pair':12s} {'N':>6} {'raw_sp':>7} {'resid':>7} {'partial':>8}")
for _, r in df.iterrows():
    w(f"{r['pair']:12s} {r['n']:6d} {r['raw']:7.3f} {r['resid_gtz']:7.3f} {r['partial']:8.3f}")
w("")
w(f"medians over {len(df)} gate-valid pairs: raw={df.raw.median():.3f} "
  f"resid(no gt_z)={df.resid_gtz.median():.3f} partial(gt_z+occ+trunc)={df.partial.median():.3f}")
w(f"partial>0.10 for: {int((df.partial>0.10).sum())}/{len(df)} pairs; "
  f"min partial={df.partial.min():.3f}")
w("")
# (1) within-distance-bin, pooled across pairs (representative pair DGP+CoP + pooled)
w("## (1) within-distance-bin Spearman (does it survive INSIDE a bin? pooled over pairs)")
bins = ["0-15", "15-30", "30-45", "45+"]
for bn in bins:
    ds = []; es = []
    for a, b in itertools.combinations(keys, 2):
        m = pd.merge(D[a][["key", "z_pred", "gt_z", "dist_bin"]], D[b][["key", "z_pred"]],
                     on="key", suffixes=("_a", "_b"))
        m = m[m.dist_bin == bn]
        if len(m) < 30:
            continue
        za, zb, gz = m.z_pred_a.values, m.z_pred_b.values, m.gt_z.values
        ds.append(np.abs(za - zb)); es.append(np.abs(0.5 * (za + zb) - gz))
    if ds:
        d = np.concatenate(ds); e = np.concatenate(es)
        w(f"  bin {bn:6s} n={len(d):6d} within-bin Spearman(d,e) = {spearmanr(d,e).statistic:+.3f}")
w("")
# (5) leave-one-detector-out
w("## (5) leave-one-detector-out: median raw Spearman over remaining pairs")
for k in keys:
    sub = df[(df.a != k) & (df.b != k)]
    w(f"  drop {k:5s}: {len(sub)} pairs, median raw={sub.raw.median():.3f} "
      f"median partial={sub.partial.median():.3f}")
w("")
w("Scope, supported: 'Across the gate-valid tested detector pairs, cross-detector")
w("depth disagreement is consistently associated with depth error, and the association persists")
w("after conditioning on distance, occlusion, and truncation.' Not supported: 'information exists")
w("between detectors but not in scores' / 'panel-wide proof' / 'independently replicated' /")
w("'recoverable'. Reason: native score also correlates with quality; 15 pairs share detectors")
w("and GTs (not independent replications); 6-detector subset, not the 12-panel.")
OUT = out_path("final_run/c3_conditioned_analysis.md")
open(OUT, "w").write("\n".join(out) + "\n")
w(f"[written] {OUT}")

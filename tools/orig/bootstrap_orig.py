"""Produces reports_orig/bootstrap_floor_orig.txt (--collate) and one JSON per
comparison in <OUT_DIR>/_bootstrap_out_orig/.

Paired drive-cluster bootstrap (GATE-3) for the comparisons that involve the
original-environment MonoFlex*/MonoGround* dumps (a copy of tools/decomp/bootstrap_floor.py
with inputs switched). Floor definition: floor_c = half-width of the 95% percentile bootstrap
interval of paired dAP (identical drive resamples both arms); panel floor = median over the
confirmatory set (the 12 transfer comparisons). Metric: dense-401 AP primary (validated
surrogate of all-point; bias cancels in paired d), official R40 secondary. B = NBOOT (default 1000).

Modes (CLI):
  --transfer f seed   one transfer comparison (base vs corr on its test frames;
                      prediction dirs from transfer_save_orig.py)
  --ladder fA fB      one adjacent-ladder comparison (full val, S5 prediction dirs from
                      make_ladder_preds_orig.py)
  --collate           gather all jsons -> bootstrap_floor_orig.txt + GATE-3 inputs
The collate merges the main-panel JSONs written by tools/decomp/bootstrap_floor.py
(<CACHE_DIR>/decomp/_bootstrap_out, or $MONO3D_MAIN_BOOTSTRAP_OUT), minus the seven
comparisons that used the modern-environment MonoFlex/MonoGround dumps, with the JSONs of
this script. The comparisons run for the paper:
  --ladder monodle gupnet ; --ladder deviant monoflex_orig ;
  --ladder monoflex_orig monoground_orig ; --ladder monoground_orig monocon ;
  --transfer monoflex_orig {0,1,2} ; --collate
Run from the repository root: python tools/orig/bootstrap_orig.py <mode>
"""
import os, sys, json, argparse, datetime
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths, dump_path, cache_dir, out_path
from tools.orig._orig_common import ORIG, DECOMP
import numpy as np, pandas as pd
import ap_corrector_arc as arc
import evaluator.kitti_eval.kitti_common as kc
from fast_subset_dense import CachedEvalDense
from frame_sequence import frame_sequence_csv

PRED = f"{ORIG}/_bootstrap_preds_orig"
OUTD = os.path.join(paths.OUT_DIR, "_bootstrap_out_orig")
NBOOT = int(os.environ.get("NBOOT", "1000"))
fsq = pd.read_csv(frame_sequence_csv())
drive_of = dict(zip(fsq.frame_idx.astype(int), fsq.drive.astype(str)))


def paired_boot(sids, dtA, dtB):
    GT = kc.get_label_annos(arc.LABEL_DIR, sids)
    ceA = CachedEvalDense(GT, dtA)
    ceB = CachedEvalDense(GT, dtB)
    pos_of = {}
    for i, s in enumerate(sids):
        pos_of.setdefault(drive_of.get(int(s), "NONE"), []).append(i)
    drives = sorted(pos_of)
    aA = ceA.eval_subset_pair(range(len(sids)))
    aB = ceB.eval_subset_pair(range(len(sids)))
    d_dense = np.zeros(NBOOT); d_r40 = np.zeros(NBOOT)
    rng = np.random.RandomState(0)
    for b in range(NBOOT):
        sample = rng.choice(len(drives), size=len(drives), replace=True)
        pos = []
        for k in sample:
            pos.extend(pos_of[drives[k]])
        xa = ceA.eval_subset_pair(pos); xb = ceB.eval_subset_pair(pos)
        d_dense[b] = xb[0] - xa[0]; d_r40[b] = xb[1] - xa[1]
        if b % 100 == 0:
            print(f"  rep {b}/{NBOOT}", flush=True)
    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5]); return float(lo), float(hi), float((hi - lo) / 2)
    res = {"n_drives": len(drives), "B": NBOOT,
           "point_dense": aB[0] - aA[0], "point_r40": aB[1] - aA[1],
           "dense_ci": ci(d_dense), "r40_ci": ci(d_r40),
           "baseA_dense": aA[0], "baseB_dense": aB[0],
           "time": datetime.datetime.now().isoformat(timespec="seconds")}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", nargs=2)
    ap.add_argument("--ladder", nargs=2)
    ap.add_argument("--collate", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    if a.transfer:
        f, seed = a.transfer
        sids = [int(x) for x in open(f"{PRED}/{f}_s{seed}_frames.txt")]
        dtA = kc.get_label_annos(f"{PRED}/{f}_s{seed}_base/data", sids)
        dtB = kc.get_label_annos(f"{PRED}/{f}_s{seed}_corr/data", sids)
        res = paired_boot(sids, dtA, dtB)
        res["kind"] = "transfer"; res["tag"] = f"{f}_s{seed}"
        json.dump(res, open(f"{OUTD}/transfer_{f}_s{seed}.json", "w"), indent=1)
        print("[done]", res["tag"], res["point_dense"], res["dense_ci"], flush=True)
    elif a.ladder:
        fA, fB = a.ladder
        sids = [int(x) for x in open(arc.VAL_LIST).read().split()]
        def _ldir(f):
            wp = f"{ORIG}/_ladder_preds_orig/{f}/data"
            return wp if os.path.exists(wp) else f"{DECOMP}/_ladder_preds/{f}/data"
        dtA = kc.get_label_annos(_ldir(fA), sids)
        dtB = kc.get_label_annos(_ldir(fB), sids)
        res = paired_boot(sids, dtA, dtB)
        res["kind"] = "ladder"; res["tag"] = f"{fA}->{fB}"
        json.dump(res, open(f"{OUTD}/ladder_{fA}__{fB}.json", "w"), indent=1)
        print("[done]", res["tag"], res["point_dense"], res["dense_ci"], flush=True)
    elif a.collate:
        out = [f"# bootstrap_floor collate {datetime.datetime.now().isoformat(timespec='seconds')}",
               f"# floor_c = CI half-width of paired dense-401 dAP; B per file; drives resampled"]
        floors = []; n_excl05 = 0; n_signeg = 0; npos = 0; nt = 0
        FROZEN_OUT = os.environ.get("MONO3D_MAIN_BOOTSTRAP_OUT", f"{DECOMP}/_bootstrap_out")
        DROP = {"ladder_monodle__monoflex.json", "ladder_monoflex__gupnet.json",
                "ladder_deviant__monoground.json", "ladder_monoground__monocon.json",
                "transfer_monoflex_s0.json", "transfer_monoflex_s1.json", "transfer_monoflex_s2.json"}
        files = [(FROZEN_OUT, fn) for fn in sorted(os.listdir(FROZEN_OUT)) if fn not in DROP]
        files += [(OUTD, fn) for fn in sorted(os.listdir(OUTD))]
        for dpath, fn in files:
            if not fn.endswith(".json"):
                continue
            r = json.load(open(f"{dpath}/{fn}"))
            lo, hi, half = r["dense_ci"]
            out.append(f"{r['kind']:8s} {r['tag']:24s} point={r['point_dense']:+6.3f} "
                       f"CI[{lo:+6.3f},{hi:+6.3f}] floor={half:.3f} (R40 point {r['point_r40']:+.3f})")
            if r["kind"] == "transfer":
                nt += 1; floors.append(half)
                if r["point_dense"] > 0:
                    npos += 1
                if hi < 0.5:
                    n_excl05 += 1
                if hi < 0:
                    n_signeg += 1
        if floors:
            out.append("")
            out.append(f"confirmatory transfer set: n={nt}; positive point estimates={npos}/{nt}; "
                       f"exclude dAP>=+0.5: {n_excl05}/{nt}; significantly negative: {n_signeg}/{nt}")
            out.append(f"PANEL FLOOR (median CI half-width, dense-401) = {np.median(floors):.3f} AP")
        txt = "\n".join(out)
        print(txt)
        open(out_path("bootstrap_floor_orig.txt"), "w").write(txt + "\n")
    else:
        ap.error("choose a mode")


if __name__ == "__main__":
    main()

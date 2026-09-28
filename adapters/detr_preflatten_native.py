"""Pre-flatten dump v2 — NATIVE-extract + NATIVE-decode based (for MonoDETR-family repos whose

Ported from tools/decomp/adapters/detr_preflatten_native.py for the public release. Computation
unchanged. Produces the auxiliary per-hypothesis pools monoclue_val_preflatten.csv and
monoia_val_preflatten.csv (v2; release asset mono3d_anatomy_query_complete_pools_v1.zip; used for the query-lineage native
pools). Config: the original runs used copies of the repo configs whose only edits were recorded
as sed commands: MonoCLUE configs/monoclue.yaml with an absolute dataset.root_dir; MonoIA
config/monoia_val.yaml with an absolute dataset.root_dir and model.focal_embedding_path pointing at
the repo's focal_list_feat.npy. This port applies the same edits via --root_dir and
--focal_embedding_path. The `lib.*` imports are the detector repo passed as --repo.
raw-head re-decode is unsafe: MonoCLUE, MonoIA; MonoDETR's v1 already validated 0-err).

Geometry path: extract_dets_from_outputs(topk=Q*C) -> the repo's OWN decode_detections with
threshold=-1e9 (only skip in this family is the threshold -> rows stay 1:1 aligned with dets,
and repo-specific corrections [e.g. IA trans_inv unwarp, any focal handling] are inherited).
Order/(q,c) recovered via torch.topk(flat, Q*C).indices (same tie-break as extract).

Unit test (abort): independent extract(topk=50)+decode(thr=0.2) Car rows must equal the rows
reconstructed from flags (count, score <1e-4, z <1e-3) per image.

Run: python detr_preflatten_native.py --repo R --cfg C --ckpt K --out O  (repo env, cwd=repo)
"""
import os, sys, time, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths  # noqa: E402,F401
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--cfg", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--root_dir", default=None, help="optional override of cfg['dataset']['root_dir']")
ap.add_argument("--focal_embedding_path", default=None,
                help="MonoIA only: override of cfg['model']['focal_embedding_path']")
args = ap.parse_args()
sys.path.insert(0, args.repo)

import numpy as np, pandas as pd, torch, yaml
from torch.utils.data import DataLoader
from lib.helpers.model_helper import build_model
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset

THR = 0.2
cfg = yaml.safe_load(open(args.cfg))
if args.root_dir:
    cfg["dataset"]["root_dir"] = args.root_dir
if args.focal_embedding_path:
    cfg["model"]["focal_embedding_path"] = args.focal_embedding_path
device = torch.device("cuda")
model, _ = build_model(cfg["model"])
model = model.to(device)
st = torch.load(args.ckpt, map_location=device)
sd = st.get("model_state", st)
miss = model.load_state_dict(sd, strict=False)
print(f"[load] missing={len(getattr(miss,'missing_keys',[]))} "
      f"unexpected={len(getattr(miss,'unexpected_keys',[]))}", flush=True)
assert not getattr(miss, "missing_keys", []), "ckpt/arch mismatch"
model.eval()

try:
    ds = KITTI_Dataset(split="val", cfg=cfg["dataset"], SAM=False)
except TypeError:
    ds = KITTI_Dataset(split="val", cfg=cfg["dataset"])
if hasattr(ds, "data_augmentation"):
    ds.data_augmentation = False
loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=False,
                    collate_fn=getattr(ds, "collate_fn", None))
cms = ds.cls_mean_size
CAR = 1

rows = []; n_mismatch = 0; t0 = time.time()
with torch.no_grad():
    for bi, (inputs, calibs, targets, info) in enumerate(loader):
        inputs = inputs.to(device); calibs_t = calibs.to(device)
        img_sizes = info["img_size"].to(device)
        for k in targets:
            if torch.is_tensor(targets[k]):
                targets[k] = targets[k].to(device)
        try:
            outputs = model(inputs, calibs_t, targets, img_sizes, dn_args=0)
        except TypeError:
            outputs = model(inputs, calibs_t, img_sizes, dn_args=0)
        probs = outputs["pred_logits"].sigmoid()
        B, Q, C = probs.shape
        order = torch.topk(probs.reshape(B, Q * C), Q * C, dim=1).indices
        d_all = extract_dets_from_outputs(outputs=outputs, K=50, topk=Q * C)
        d_all_n = d_all.detach().cpu().numpy()
        d50 = extract_dets_from_outputs(outputs=outputs, K=50, topk=50).detach().cpu().numpy()
        order_n = order.cpu().numpy(); probs_n = probs.cpu().numpy()
        info_np = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
                   for k, v in info.items()}
        ids = info_np["img_id"]
        calib_objs = [ds.get_calib(int(s)) for s in ids]
        dec_all = decode_detections(dets=d_all_n, info=info_np, calibs=calib_objs,
                                    cls_mean_size=cms, threshold=-1e9)
        nat = decode_detections(dets=d50, info=info_np, calibs=calib_objs,
                                cls_mean_size=cms, threshold=THR)
        for i in range(B):
            sid = int(ids[i])
            rowsd = dec_all[ids[i]]
            assert len(rowsd) == Q * C, f"decode row misalign sid={sid}: {len(rowsd)}"
            rec_final = []
            for j in range(Q * C):
                fidx = int(order_n[i, j]); q, c = fidx // C, fidx % C
                r = rowsd[j]
                lab = int(r[0])
                assert lab == c, f"label/order mismatch sid={sid} j={j}"
                clsc = float(d_all_n[i, j, 1]); sg = max(float(d_all_n[i, j, -1]), 1e-6)
                V = float(r[-1])                       # native final score (cls*sigma)
                nt50 = j < 50; thr_ok = clsc >= THR
                infin = bool(nt50 and thr_ok)
                p = probs_n[i, q]
                rows.append(dict(
                    sid=sid, query_id=q, class_id=c, flat_rank=j,
                    native_top50=int(nt50), thr_pass=int(thr_ok), in_final=int(infin),
                    cls=clsc, cls_c0=float(p[0]), cls_c1=float(p[1]),
                    cls_c2=float(p[2]) if C > 2 else 0.0, car_channel=CAR,
                    sigma=sg, log_sigma_raw=float(-np.log(sg)), V=V,
                    z_pred=float(r[11]),
                    bbox_x1=float(r[2]), bbox_y1=float(r[3]),
                    bbox_x2=float(r[4]), bbox_y2=float(r[5]),
                    bbox_h_pix=float(r[5] - r[3]), bbox_w_pix=float(r[4] - r[2]),
                    bbox_area_pix=float((r[5] - r[3]) * (r[4] - r[2])),
                    h_3d=float(r[6]), w_3d=float(r[7]), l_3d=float(r[8]),
                    x_3d=float(r[9]), y_3d=float(r[10]), z_3d=float(r[11]),
                    ry=float(r[12]), alpha=float(r[1])))
                if infin and c == CAR:
                    rec_final.append((V, float(r[11])))
            nat_rows = [r for r in nat[ids[i]] if int(r[0]) == CAR]
            ok = len(nat_rows) == len(rec_final)
            if ok and rec_final:
                ns = sorted(float(r[-1]) for r in nat_rows); rs = sorted(x[0] for x in rec_final)
                ok = all(abs(a - b) < 1e-4 for a, b in zip(ns, rs))
                nz = sorted(float(r[11]) for r in nat_rows); rz = sorted(x[1] for x in rec_final)
                ok = ok and all(abs(a - b) < 1e-3 for a, b in zip(nz, rz))
            if not ok:
                n_mismatch += 1
                if n_mismatch <= 5:
                    print(f"[MISMATCH] sid={sid} nat={len(nat_rows)} rec={len(rec_final)}", flush=True)
        if bi % 50 == 0:
            print(f"  batch {bi}/{len(loader)} rows={len(rows)} mism={n_mismatch} "
                  f"{time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
df.to_csv(args.out, index=False)
print(f"[dump] wrote {args.out} ({len(df)} rows) mismatches={n_mismatch}", flush=True)
if n_mismatch:
    print("UNIT TEST FAILED", flush=True); sys.exit(1)
print("UNIT TEST PASS", flush=True)

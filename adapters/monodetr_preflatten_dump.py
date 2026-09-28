"""MonoDETR PRE-FLATTEN dump (reports/prereg_2b_detr_sweep.md, rev3).

Produces the complete per-hypothesis pool monodetr_val_preflatten.csv (release asset
mono3d_anatomy_query_complete_pools_v1.zip), used for the native pools of the query-based detectors.
Config: the repo's configs/monodetr.yaml with dataset.root_dir set from --root_dir
(default paths.KITTI_ROOT). The `lib.*` imports are the upstream MonoDETR package (--repo).

Design: call the repo's NATIVE extract_dets_from_outputs with topk=150, which returns EVERY
(query, class) hypothesis with the native-faithful geometry decode (the 20.13-vs-20.83 issue
of per-query re-decoding is avoided entirely). Rows are per-HYPOTHESIS (150/img, label col
included); the Car analysis selects label==1 rows (exactly 50/img: every query's Car
hypothesis present, no 150→50 crowding censoring).

Columns: sid, query_id, class_id, flat_rank (0..149, raw-cls order), native_top50
(flat_rank<50), thr_pass (cls>=0.2), in_final (both), cls (this hypothesis's sigmoid prob),
cls_c0/c1/c2 (all channels of the query), sigma, log_sigma_raw, V (=cls*sigma), z_pred,
full box geometry (native decode), alpha, ry.

UNIT TEST (abort gate): per image, rows with in_final==1 ∧ class_id==1 must equal the
INDEPENDENT native run extract(topk=50)+decode_detections(thr=0.2) Car output on count,
score (<1e-4) and z (<1e-3); plus consistency assert label[j] == order[j]%3 for all rows.
Run (MonoDETR env, torch 1.9 in our runs; any cwd): python adapters/monodetr_preflatten_dump.py
"""
import os, sys, time, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--repo", default=os.path.join(paths.UPSTREAM_ROOT, "MonoDETR"))
REPO = os.path.abspath(_pre.parse_known_args()[0].repo)
sys.path.insert(0, REPO)
import numpy as np, torch, yaml
import pandas as pd
from torch.utils.data import DataLoader
from lib.helpers.model_helper import build_model
from lib.helpers.decode_helper import (extract_dets_from_outputs, decode_detections,
                                       get_heading_angle)
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset

CAR = 1   # cls2id: Pedestrian0 Car1 Cyclist2 (verified by adapter + unit test)
THR = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--ckpt", default=os.path.join(paths.UPSTREAM_ROOT, "ckpts", "monodetr", "monodetr_best.pth"))
    ap.add_argument("--cfg", default=os.path.join(REPO, "configs", "monodetr.yaml"))
    ap.add_argument("--root_dir", default=paths.KITTI_ROOT, help="dataset.root_dir override (as in the original run)")
    ap.add_argument("--out", default=os.path.join(paths.OUT_DIR, "dumps", "monodetr_val_preflatten.csv"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    cfg["dataset"]["root_dir"] = args.root_dir
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = torch.device("cuda")
    model, _ = build_model(cfg["model"]); model = model.to(device)
    st = torch.load(args.ckpt, map_location=device); sd = st.get("model_state", st)
    miss = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(getattr(miss,'missing_keys',[]))} "
          f"unexpected={len(getattr(miss,'unexpected_keys',[]))}", flush=True)
    model.eval()

    ds = KITTI_Dataset(split="val", cfg=dict(cfg["dataset"]))
    if hasattr(ds, "data_augmentation"):
        ds.data_augmentation = False
    loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=False,
                        collate_fn=getattr(ds, "collate_fn", None))
    cms = ds.cls_mean_size
    rows = []; n_mismatch = 0; t0 = time.time()
    with torch.no_grad():
        for bi, (inputs, calibs, targets, info) in enumerate(loader):
            inputs = inputs.to(device); calibs_t = calibs.to(device)
            img_sizes = info["img_size"].to(device)
            for k in targets:
                if torch.is_tensor(targets[k]):
                    targets[k] = targets[k].to(device)
            outputs = model(inputs, calibs_t, targets, img_sizes, dn_args=0)
            logits = outputs["pred_logits"]; probs = logits.sigmoid()
            B, Q, C = probs.shape
            # use torch.topk (not argsort): must follow the SAME tie-break order as the
            # repo's extract_dets_from_outputs, which uses torch.topk internally
            order = torch.topk(probs.reshape(B, Q * C), Q * C, dim=1).indices
            d150 = extract_dets_from_outputs(outputs=outputs, K=50, topk=Q * C)
            d150 = d150.detach().cpu().numpy()
            d50 = extract_dets_from_outputs(outputs=outputs, K=50, topk=50).detach().cpu().numpy()
            order_n = order.cpu().numpy(); probs_n = probs.cpu().numpy()
            info_np = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
                       for k, v in info.items()}
            imgsz = info_np["img_size"]; ids = info_np["img_id"]
            calib_objs = [ds.get_calib(int(s)) for s in ids]
            nat = decode_detections(dets=d50, info=info_np, calibs=calib_objs,
                                    cls_mean_size=cms, threshold=THR)
            for i in range(B):
                sid = int(ids[i]); W, H = float(imgsz[i][0]), float(imgsz[i][1])
                cal = calib_objs[i]
                rec_final = []
                for j in range(d150.shape[1]):
                    fidx = int(order_n[i, j]); q, c = fidx // C, fidx % C
                    lab = int(d150[i, j, 0])
                    assert lab == c, f"label/order mismatch sid={sid} j={j}: {lab} vs {c}"
                    clsc = float(d150[i, j, 1]); sg = max(float(d150[i, j, 36]), 1e-6)
                    V = clsc * sg
                    x = float(d150[i, j, 2] * W); y = float(d150[i, j, 3] * H)
                    w_ = float(d150[i, j, 4] * W); h2 = float(d150[i, j, 5] * H)
                    bbox = [x - w_/2, y - h2/2, x + w_/2, y + h2/2]
                    depth = float(d150[i, j, 6])
                    dims = d150[i, j, 31:34] + cms[c]
                    h3, w3, l3 = float(dims[0]), float(dims[1]), float(dims[2])
                    x3 = float(d150[i, j, 34] * W); y3 = float(d150[i, j, 35] * H)
                    loc = cal.img_to_rect(np.array([x3]), np.array([y3]), np.array([depth])).reshape(-1)
                    lx, ly, lz = float(loc[0]), float(loc[1]) + h3/2.0, float(loc[2])
                    alpha = float(get_heading_angle(d150[i, j, 7:31]))
                    ry = float(cal.alpha2ry(alpha, x))
                    nt50 = j < 50; thr_ok = clsc >= THR
                    infin = bool(nt50 and thr_ok)
                    p = probs_n[i, q]
                    rows.append(dict(
                        sid=sid, query_id=q, class_id=c, flat_rank=j,
                        native_top50=int(nt50), thr_pass=int(thr_ok), in_final=int(infin),
                        cls=clsc, cls_c0=float(p[0]), cls_c1=float(p[1]), cls_c2=float(p[2]),
                        car_channel=CAR, sigma=sg, log_sigma_raw=float(-np.log(sg)),
                        V=V, z_pred=depth,
                        bbox_h_pix=bbox[3]-bbox[1], bbox_w_pix=bbox[2]-bbox[0],
                        bbox_area_pix=(bbox[3]-bbox[1])*(bbox[2]-bbox[0]),
                        x_3d=lx, y_3d=ly, z_3d=lz, h_3d=h3, w_3d=w3, l_3d=l3,
                        ry=ry, alpha=alpha,
                        bbox_x1=bbox[0], bbox_y1=bbox[1], bbox_x2=bbox[2], bbox_y2=bbox[3]))
                    if infin and c == CAR:
                        rec_final.append((V, lz))
                # unit test vs independent native run
                nat_rows = [r for r in nat[ids[i]] if int(r[0]) == CAR]
                ok = len(nat_rows) == len(rec_final)
                if ok and rec_final:
                    ns = sorted(float(r[-1]) for r in nat_rows); rs = sorted(r[0] for r in rec_final)
                    ok = all(abs(a - b) < 1e-4 for a, b in zip(ns, rs))
                    nz = sorted(float(r[11]) for r in nat_rows); rz = sorted(r[1] for r in rec_final)
                    ok = ok and all(abs(a - b) < 1e-3 for a, b in zip(nz, rz))
                if not ok:
                    n_mismatch += 1
                    if n_mismatch <= 5:
                        print(f"[MISMATCH] sid={sid} nat={len(nat_rows)} rec={len(rec_final)}", flush=True)
            if bi % 50 == 0:
                print(f"  batch {bi}/{len(loader)} rows={len(rows)} mism={n_mismatch} "
                      f"{time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[dump] wrote {args.out} ({len(df)} rows) mismatches={n_mismatch}", flush=True)
    if n_mismatch:
        print("UNIT TEST FAILED", flush=True); sys.exit(1)
    print("UNIT TEST PASS", flush=True)


if __name__ == "__main__":
    main()

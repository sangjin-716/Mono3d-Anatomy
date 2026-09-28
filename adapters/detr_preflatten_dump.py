"""PRE-FLATTEN per-query dump for MonoDETR-family models (DGP/CoP/CLUE/IA variant; MonoDETR
itself uses adapters/monodetr_preflatten_dump.py because its forward signature differs).

Produces the complete per-hypothesis pools monodgp_val_preflatten.csv and
official_monocop_val_preflatten.csv (release asset mono3d_anatomy_query_complete_pools_v1.zip),
used for the native pools of the query-based detectors. Do not use it for MonoCLUE or MonoIA:
this generic raw-head decode skips their repo-specific decode corrections, so use
adapters/detr_preflatten_native.py for those two. --label_dir (default paths.LABEL_DIR) is only
used for the AP printed at the end, and --root_dir optionally overrides cfg["dataset"]["root_dir"].

Following reports/prereg_2b_detr_sweep.md (rev2): for every val image and each of the 50 inference
queries, store the decoded 3D box (Car mean-size convention, same math as adapters/dgp_cop_dump.py), ALL
foreground class sigmoid probs (cls_car/ped/cyc; family has no explicit no-object logit),
log_sigma/sigma, V_car = cls_car*exp(-log_sigma), V_maxcls (old-dump compat check), query_id,
argmax_cls, layer_id ('last'), flat_rank_car (rank of the (q,Car) hypothesis among the 150
query x class raw-cls hypotheses), native_top50 ((q,Car) in native top-50-of-150), thr_pass
(cls_car >= 0.2), in_final (native_top50 AND thr_pass).

NATIVE STAGE ORDER (audited; prereg rev3): 150 query x class hypotheses -> K_flat top-50 by
RAW cls (extract_dets_from_outputs topk) -> threshold 0.2 (decode_detections) -> V ranking.
The stored per-query rows carry ALL class probabilities + sigma, so every (query,class)
HYPOTHESIS state (raw prob, V_class, flat rank, top-50 membership, thr pass, final membership)
is derivable deterministically; Car-hypothesis states are additionally materialized as columns.

EMBEDDED HYPOTHESIS-LEVEL UNIT TEST (abort gate): per image, the full ALL-CLASS native
pipeline is reconstructed from stored quantities (K_flat -> thr -> cls*sigma scores) and
compared against the repo's OWN extract_dets_from_outputs + decode_detections output on the
SAME tensors: (class,score) multiset (<2e-4), counts for all classes, and Car 3D boxes
(<1e-3). Any mismatching image aborts with nonzero exit. At the end the reconstruction's
Car-mod-R40 is computed with the repo's official evaluator for the record.

The `lib.*` imports are the detector repo passed as --repo.
Run (repo env, cwd=repo): python <Mono3d-Anatomy>/adapters/detr_preflatten_dump.py \
  --repo <repo> --cfg <yaml> --ckpt <pth> --out <csv> --tag <tag> [--car_idx auto]
"""
from __future__ import annotations
import os, sys, time, shutil, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--car_idx", default="auto")
    ap.add_argument("--label_dir", default=paths.LABEL_DIR, help="KITTI training/label_2 (record-only AP)")
    ap.add_argument("--root_dir", default=None, help="optional override of cfg['dataset']['root_dir']")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import numpy as np, pandas as pd, torch, yaml
    from lib.helpers.model_helper import build_model
    from lib.helpers.decode_helper import (get_heading_angle, extract_dets_from_outputs,
                                           decode_detections)
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    from torch.utils.data import DataLoader

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    if args.root_dir:
        cfg["dataset"]["root_dir"] = args.root_dir
    device = torch.device("cuda")
    model, _ = build_model(cfg["model"])
    model = model.to(device)
    st = torch.load(args.ckpt, map_location=device)
    sd = st.get("model_state", st)
    miss = model.load_state_dict(sd, strict=False)
    nmiss = len(getattr(miss, "missing_keys", [])); nunexp = len(getattr(miss, "unexpected_keys", []))
    print(f"[load] missing={nmiss} unexpected={nunexp}", flush=True)
    assert nmiss == 0, "checkpoint does not match model arch"
    model.eval()

    try:    # MonoCLUE's dataset takes SAM=True by default (train-time aux input; off at inference)
        ds = KITTI_Dataset(split="val", cfg=cfg["dataset"], SAM=False)
    except TypeError:
        ds = KITTI_Dataset(split="val", cfg=cfg["dataset"])
    ds.data_augmentation = False
    if hasattr(ds, "istrain"):
        ds.istrain = False
    loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=False,
                        pin_memory=True, collate_fn=getattr(ds, "collate_fn", None))
    cms = ds.cls_mean_size

    # Car channel index: detect from dataset class list when possible
    if args.car_idx == "auto":
        names = None
        for attr in ("class_name", "class_names", "CLASSES", "writelist"):
            if hasattr(ds, attr):
                names = [str(x).lower() for x in getattr(ds, attr)]
                break
        assert names and "car" in names, f"cannot auto-detect Car index from {names}"
        CAR = names.index("car")
    else:
        CAR = int(args.car_idx)
    print(f"[classmap] CAR channel index = {CAR}", flush=True)

    kdir_rec = os.path.join(os.path.dirname(args.out), f"_kitti_{args.tag}_rec", "data")
    if os.path.exists(kdir_rec):
        shutil.rmtree(kdir_rec)
    os.makedirs(kdir_rec)

    rows = []
    n_mismatch = 0
    t0 = time.time()
    THR = 0.2
    with torch.no_grad():
        for bi, (inputs, calibs, targets, info) in enumerate(loader):
            inputs = inputs.to(device); calibs_t = calibs.to(device)
            img_sizes = info["img_size"].to(device)
            for k in targets:
                if torch.is_tensor(targets[k]):
                    targets[k] = targets[k].to(device)
            try:
                outputs = model(inputs, calibs_t, targets, img_sizes, dn_args=0)
            except TypeError:   # MonoCLUE: forward(images, calibs, img_sizes, dn_args)
                outputs = model(inputs, calibs_t, img_sizes, dn_args=0)
            logits = outputs["pred_logits"]                      # (B,Q,C)
            probs = logits.sigmoid()
            B, Q, C = probs.shape
            flat = probs.reshape(B, Q * C)
            top50 = torch.topk(flat, 50, dim=1).indices          # native budget (= topk cfg 50)
            order = torch.argsort(flat, dim=1, descending=True)
            rank_of = torch.empty_like(order)
            rank_of.scatter_(1, order, torch.arange(Q * C, device=device).expand(B, -1))
            log_sigma = outputs["pred_depth"][..., 1]
            sigma = torch.exp(-log_sigma)
            pred_z = outputs["pred_depth"][..., 0]
            boxes = outputs["pred_boxes"]
            H = img_sizes[:, 1].clamp(min=1.0).unsqueeze(1)
            W = img_sizes[:, 0].clamp(min=1.0).unsqueeze(1)
            bbox_h = ((boxes[..., 4] + boxes[..., 5]) * H).clamp(min=1.0)
            bbox_w = ((boxes[..., 2] + boxes[..., 3]) * W).clamp(min=1.0)
            dim = outputs["pred_3d_dim"]
            h3d = dim[..., 0].cpu().numpy() + cms[0][0]
            w3d = dim[..., 1].cpu().numpy() + cms[0][1]
            l3d = dim[..., 2].cpu().numpy() + cms[0][2]
            heading = outputs["pred_angle"].cpu().numpy()
            xs = (boxes[..., 0] * W).cpu().numpy()
            ys = (boxes[..., 1] * H).cpu().numpy()
            probs_n = probs.cpu().numpy(); sigma_n = sigma.cpu().numpy()
            ls_n = log_sigma.cpu().numpy(); z_n = pred_z.cpu().numpy()
            bh_n = bbox_h.cpu().numpy(); bw_n = bbox_w.cpu().numpy()
            top50_n = top50.cpu().numpy(); rank_n = rank_of.cpu().numpy()

            # native path on the SAME tensors (repo code) for the unit test
            dets_nat = extract_dets_from_outputs(outputs=outputs, K=50, topk=50)
            dets_nat = dets_nat.detach().cpu().numpy()
            info_np = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
                       for k, v in info.items()}
            calib_objs = [ds.get_calib(int(i)) for i in info_np["img_id"]]
            nat = decode_detections(dets=dets_nat, info=info_np, calibs=calib_objs,
                                    cls_mean_size=cms, threshold=THR)

            for b in range(B):
                sid = int(info_np["img_id"][b])
                calib = ds.get_calib(sid)
                t50 = set(int(x) for x in top50_n[b])
                rec_final = []   # reconstruction of native Car output from flags
                lines = []
                for q in range(Q):
                    x_img = float(xs[b, q]); y_img = float(ys[b, q])
                    w_img = float(bw_n[b, q]); h_img = float(bh_n[b, q])
                    bbox = [x_img - w_img / 2, y_img - h_img / 2,
                            x_img + w_img / 2, y_img + h_img / 2]
                    depth = float(z_n[b, q])
                    loc = calib.img_to_rect(np.array([x_img]), np.array([y_img]),
                                            np.array([depth])).reshape(-1)
                    hh, ww, ll = float(h3d[b, q]), float(w3d[b, q]), float(l3d[b, q])
                    loc[1] += hh / 2
                    alpha = get_heading_angle(heading[b, q])
                    ry = calib.alpha2ry(alpha, x_img)
                    p = probs_n[b, q]
                    cls_car = float(p[CAR])
                    sg = float(sigma_n[b, q])
                    V_car = cls_car * sg
                    fidx = q * C + CAR
                    nt50 = fidx in t50
                    thr_ok = cls_car >= THR
                    infin = bool(nt50 and thr_ok)
                    rows.append(dict(
                        sid=sid, query_id=q, layer_id="last",
                        cls_car=cls_car,
                        cls_c0=float(p[0]), cls_c1=float(p[1]),
                        cls_c2=float(p[2]) if C > 2 else 0.0,
                        car_channel=CAR,
                        argmax_cls=int(p.argmax()),
                        log_sigma_raw=float(ls_n[b, q]), sigma=sg,
                        V_car=V_car, V_maxcls=float(p.max() * sg),
                        flat_rank_car=int(rank_n[b, q * C + CAR]),
                        native_top50=int(nt50), thr_pass=int(thr_ok), in_final=int(infin),
                        z_pred=depth,
                        bbox_h_pix=h_img, bbox_w_pix=w_img, bbox_area_pix=h_img * w_img,
                        x_3d=float(loc[0]), y_3d=float(loc[1]), z_3d=float(loc[2]),
                        h_3d=hh, w_3d=ww, l_3d=ll, ry=float(ry), alpha=float(alpha),
                        bbox_x1=bbox[0], bbox_y1=bbox[1], bbox_x2=bbox[2], bbox_y2=bbox[3]))
                    if infin:
                        rec_final.append((V_car, loc[0], loc[1], loc[2], bbox, hh, ww, ll, ry, alpha))
                        lines.append(f"Car 0.0 0 {alpha:.4f} "
                                     f"{bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f} "
                                     f"{hh:.4f} {ww:.4f} {ll:.4f} "
                                     f"{loc[0]:.4f} {loc[1]:.4f} {loc[2]:.4f} {ry:.4f} {V_car:.6f}\n")
                with open(os.path.join(kdir_rec, f"{sid:06d}.txt"), "w") as fo:
                    fo.writelines(lines)

                # ---- HYPOTHESIS-LEVEL unit test vs native path (all classes) ----
                # reconstruct the full native pipeline from stored quantities:
                # 150 hypotheses -> K_flat top-50 by raw cls -> thr0.2 -> score cls*sigma
                key = info_np["img_id"][b]
                rec_hyp = []   # (class_id, score) for ALL classes, native order semantics
                for fi in t50:                       # K_flat survivors (native top-50)
                    q_, c_ = fi // C, fi % C
                    pc = float(probs_n[b, q_, c_])
                    if pc < THR:                     # threshold AFTER K_flat (native order)
                        continue
                    rec_hyp.append((c_, pc * float(sigma_n[b, q_])))
                nat_all = list(nat[key])
                ok = len(nat_all) == len(rec_hyp)
                if ok:
                    ns = sorted((int(r[0]), round(float(r[-1]), 4)) for r in nat_all)
                    rs = sorted((c_, round(s_, 4)) for c_, s_ in rec_hyp)
                    ok = all(a[0] == c[0] and abs(a[1] - c[1]) < 2e-4 for a, c in zip(ns, rs))
                # Car rows additionally checked on 3D box (z) vs our decoded geometry
                nat_rows = [r for r in nat_all if int(r[0]) == CAR]
                if ok:
                    ok = len(nat_rows) == len(rec_final)
                if ok and rec_final:
                    nz = sorted(float(r[11]) for r in nat_rows)   # loc z in decode output
                    rz = sorted(r[3] for r in rec_final)
                    ok = all(abs(a - c) < 1e-3 for a, c in zip(nz, rz))
                if not ok:
                    n_mismatch += 1
                    if n_mismatch <= 5:
                        print(f"[MISMATCH] sid={sid} nat_all={len(nat_all)} rec_hyp={len(rec_hyp)} "
                              f"natCar={len(nat_rows)} recCar={len(rec_final)}", flush=True)
            if bi % 50 == 0:
                print(f"  batch {bi}/{len(loader)} rows={len(rows)} mismatches={n_mismatch} "
                      f"{time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[dump] wrote {args.out} ({len(df)} rows) mismatches={n_mismatch}", flush=True)

    # AP of the reconstruction (repo's own evaluator), for the record
    try:
        from lib.datasets.kitti.kitti_eval_python import kitti_common as kc
        from lib.datasets.kitti.kitti_eval_python.eval import do_eval
        LAB = args.label_dir
        sids = sorted(int(x) for x in ds.idx_list)
        GT = kc.get_label_annos(LAB, sids)
        DT = kc.get_label_annos(os.path.dirname(kdir_rec) + "/data", sids)
        ov07 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7]] * 3)
        ov05 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                         [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
        MO = np.stack([ov07, ov05], 0)[:, :, [0]]
        r = do_eval(GT, DT, [0], MO, compute_aos=False, DIForDIS=True)
        print(f"[recon-AP] Car mod R40 = {float(r[6][0,1,0]):.3f}", flush=True)
    except Exception as e:
        print(f"[recon-AP] skipped ({e})", flush=True)

    if n_mismatch:
        print(f"UNIT TEST FAILED: {n_mismatch} images mismatch native path", flush=True)
        sys.exit(1)
    print("UNIT TEST PASS: reconstruction == native pipeline on all images", flush=True)


if __name__ == "__main__":
    main()

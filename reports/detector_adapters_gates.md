<!-- public copy of reports/detector_adapters_gates.md, sha256 e09cc08cb28c0c6e756e8fb8c28c5e1ad24ba240e148624a4ca70960d4cf9ce9, scrubbed: an absolute checkpoint path shortened to ./ckpts (item 6) -->
# Detector Adapter Gates — Panel Expansion 7→12 (2026-06-10/11)

Purpose: provenance for the 5 added detectors. Gate order per adapter: (1) native repo
eval must reproduce the published/verified number, (2) 23-col dump must reconstruct the
native detections (tap-equivalence), (3) S5 protocol number recorded for the ladder.
All canonical grading = MonoDGP kitti_eval_python do_eval, AP3D R40 IoU0.7 Car E/M/H,
val Chen split (identical to panel baseline). Env: monoflex_accv (torch 1.10+cu113).

## Gate results

| Detector | Claimed (E/M/H) | Our native run | Tap-equivalence | S5 (E/M/H) |
|---|---|---|---|---|
| DEVIANT (ECCV22) | 24.63/16.54/14.52 (run_221 shipped preds, graded by our do_eval) | 24.65/16.49/14.51 | boxes+scores exact (native writer rounds to 2dp; our full-precision 25.41/16.71/14.74) | 25.43/16.74/14.77 |
| MonoDLE (CVPR21) | 17.94/13.72/12.10 (repo README, released ckpt) | 18.26/14.57/13.00 | box-level exact (writer 2dp rounding); thr-recon 18.30/14.63/13.05 | 18.36/14.69/12.48 |
| MonoCon (AAAI22, 2gunsu) | 26.03/19.02/15.92 (2gunsu ckpt) | 26.03/19.02/15.92 (exact) | 26.031/19.015/15.912 vs native (Δ≤0.007) | 26.09/19.05/15.94 |
| M3D-RPN (ICCV19) | 14.53/11.07/8.65 (R40 re-eval of released val1 model; paper 17.06 is R11) | 14.531/11.073/8.646 (exact) | native-NMS recon 14.538/11.099/8.670 (Δ≤0.03) | 14.54/11.10/8.65 |
| MonoGround (CVPR22) | 25.24/18.69/15.58 (repo README) | 23.34/16.79/13.91 (−1.90 mod) | (dump in flight) | (pending) |

## Notes / honesty items

1. **MonoDLE +0.85 mod drift (favorable)**: same ckpt, same repo code, same split; our
   canonical do_eval matches the repo-internal eval exactly (18.259/14.566/12.999), so the
   drift is inference-env (torch 1.1-era → 1.10, Pillow) — NOT an eval-protocol difference.
2. **MonoFlex-codebase −1.9 family drift**: MonoGround −1.90 AND a control run of
   MonoFlex official ckpt natively in our env gives 22.29/**15.54**/13.38 R40 vs paper-val
   17.51 mod (−1.97). The two repos sharing the recompiled DCNv2 fork (+inplace-abn stack)
   are the only ones that drift; GUPNet (16.48 vs official 16.46), DEVIANT, MonoCon,
   M3D-RPN reproduce (near-)exactly in the same env. Panel treatment: both enter at
   reproduced strength (MonoFlex 15.57 S5 was already the panel value — consistent);
   provenance reports official+reproduced. Root cause (DCNv2 recompile suspected) not
   isolated further — flagged as known limitation, do not claim official parity for these two.
3. **M3D-RPN specifics**: torch-0.4→1.10 inference port = py_cpu_nms swap (+1-offset IoU
   convention preserved; torchvision.ops.nms NOT used to keep NMS math identical) + visdom
   install. Split images are renumbered by the repo; sid mapping in ckpts/m3drpn/val_id_map.csv.
   Dump = pre-NMS top-3000 pool, Car score≥0.05, cap 300/img (2016 imgs hit cap), decode incl
   native hill-climb; no uncertainty head → sigma≡1.0, log_sigma_raw≡0.0 (constant).
   Native-NMS reconstruction from the dump (NMS@0.4 +1-offset, top-40, V≥0.75) reproduces
   the native AP (Δ≤0.03) — pool is faithful.
4. **MonoCon**: K=30 native top-k (vs 50 for other CenterNets) — native pool definition kept.
   cls=V=combined score (heatmap×exp(−d1)), native threshold 0.4.
5. **DEVIANT/MonoDLE writers round scores to 2dp** in their KITTI files; our dumps keep full
   precision (tap-equivalence shown at box level; small AP deltas are purely this rounding).
6. Checkpoint files + md5 under ./ckpts/{deviant,monodle,monocon,m3drpn,monoground}/.
   Adapters: tools/{deviant,monodle,monocon,monoground}_dump.py, M3D-RPN/scripts/m3drpn_dump_accv.py.
   Dumps: experiments/dgp_cop_diag/{deviant,monodle,monocon,m3drpn,monoground}_val.csv (23-col).

## Rejected / deferred (from reports/detector_addition_shortlist.md)
DID-M3D (LiDAR-supervised — not strict mono), MonoUNI / GUPNet++ (no public KITTI val ckpt —
numbers unverifiable, DO NOT add), SMOKE (true R40 mod 7.09, below panel band; mm-env cost),
MonoLSS (public ckpt is trainval-trained = val-leaked).

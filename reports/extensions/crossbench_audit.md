# public summary of the camera-ready checks XDSFIX_RESULT.md, sha256 6186c18d470fd7916ccd3a678a6b092691aa5590fbe53d7b4e43fda15d7ae96c, and GATE_CERTIFICATE.md, sha256 10b3fe7ba027401a992432cecae09a6a103a431c999c2a084dc1cd4773de192d, scrubbed: rewritten as a summary that leaves out review-process material; every number is quoted verbatim with a source tag resolved in the appendix

Cross-benchmark preliminary audit behind supplementary Sec. P and the "Other Benchmarks" paragraph of main Sec. 5.1. This is a **preliminary audit, not a replication**: one released checkpoint per detector (on Waymo, its released predictions), no confidence intervals, and only the reproduction step has an external reference. Its magnitudes are not compared with the KITTI band or across detectors. Scripts: `tools/crossbench/` (see its README). Every number below carries a source tag `[Xn]` / `[Gn]` resolved in the appendix (X = XDSFIX_RESULT.md, G = GATE_CERTIFICATE.md, O = other source file, see appendix).

## Detectors, checkpoints, splits

| Detector | Benchmark | Checkpoint / predictions | Split |
|---|---|---|---|
| GUPNet | Waymo | released Waymo run `run_1050` of the DEVIANT release (authors' predictions, byte-identical) [G165-166][O1] | Waymo val, 39,848 front-camera frames [X170][G161] |
| DEVIANT | Waymo | released Waymo run `run_1051` of the DEVIANT release [G165-166][O1] | same |
| FCOS3D | nuScenes | `fcos3d_..._finetune_20210717_095645-8d806dc2.pth` (mmdetection3d metafile row by weights URL) [X54] | nuScenes v1.0-trainval val, 6,019 samples, `detection_cvpr_2019` protocol [X132] |
| PGD | nuScenes | `pgd_..._finetune_20211114_162135-5ec7c1cd.pth` [X55] | same |
| CenterNet | nuScenes | CenterTrack model zoo, e140 checkpoint (CenterNet-e140) [X150] | same |
| EPro-PnP-Det | nuScenes | released checkpoint, gated against its official README [G75][X151] | same |
| MonoRCNN++ | Waymo | released predictions; excluded (fails its own published numbers) [G80][G175] | Waymo val |

FCOS3D and PGD share the mmdet3d FCOS3D lineage [X230].

## Reproduction gate on the unmodified official evaluators (Table P1)

| Detector | Metric | Ours | Published | Delta | Source |
|---|---|---|---|---|---|
| GUPNet | Waymo APH-L1 All, IoU 0.7 / 0.5 | 2.25 / 9.88 | 2.27 / 9.94 | -0.02 / -0.06 | [G70][G71][G173][X176] |
| DEVIANT | Waymo APH-L1 All, IoU 0.7 / 0.5 | 2.65 / 10.82 | 2.67 / 10.89 | -0.02 / -0.07 | [G72][G73][G174][X177] |
| FCOS3D | nuScenes mAP / NDS | 32.13 / 39.48 | 32.1 / 39.3 | +0.03 / +0.18 | [X116][X148][X75] |
| PGD | nuScenes mAP / NDS | 35.84 / 42.89 | 35.8 / 42.5 | +0.04 / +0.39 | [X117][X149][X75] |
| CenterNet | nuScenes mAP | 30.265 | 30.27 | -0.005 | [X150][G74] |
| EPro-PnP-Det | nuScenes NDS | 42.43 | 0.425 (README) | -0.07 | [X151][G75] |
| MonoRCNN++ | Waymo AP3D-L1 / L2, IoU 0.7 | 1.70 / 1.59 | 4.28 / 4.05 | fails | [G80][G175] |
| MonoRCNN++ | Waymo AP3D-L1 / L2, IoU 0.5 | 7.09 / 6.65 | 11.37 / 10.79 | fails (4/4) | [G80][G175] |

Waymo predictions of GUPNet and DEVIANT are byte-identical to the authors' released outputs (39,848/39,848) [G194]; the Waymo gate is |Delta| <= 0.07 [G187-188]. The reference is the authors' own 2022 official-evaluator output shipped in the released archives [G164-166]. We did not re-run inference on Waymo, so this gate checks our ground-truth conversion and evaluator.

NDS notes. NDS is secondary on nuScenes; the FCOS3D and PGD gates are on mAP, which is inside the metafile's 1-decimal rounding [X75]. PGD's +0.39 NDS exceeds rounding; it is localised to a TP-error budget of +0.039 (a term mAP does not use) and attributed, without proof, to the mmdet3d v1.0.0.dev0 -> 1.4.0 attribute/velocity plumbing [X75][X223]. FCOS3D's NDS residual is +0.18 [X75]. EPro-PnP-Det is gated on NDS against its README value 0.425 [X151][G75].

## True-IoU re-sort on the gated pools, Car, IoU3D 0.7, all-point AP (Table P2)

| Detector | Data | base | re-sort | gain | boxes/img | Source |
|---|---|---|---|---|---|---|
| GUPNet | Waymo | 7.46 | 12.37 | +4.90 | 3.97 | [X186][X100] |
| DEVIANT | Waymo | 8.13 | 13.04 | +4.91 | 4.02 | [X188][X100] |
| FCOS3D | nuScenes | 4.23 | 12.01 | +7.78 | 4.8 | [X157][X99] |
| PGD | nuScenes | 6.33 | 13.07 | +6.75 | 5.0 | [X159][X99] |
| EPro-PnP-Det | nuScenes | 8.06 | 14.18 | +6.13 | 4.5 | [X161][X42][X99] |
| CenterNet | nuScenes | 5.07 | 11.06 | +5.99 | not measured | [X163][X166] |

The re-sort raises AP for all six, by +4.90 to +7.78 at IoU 0.7 [X193][X211]. Base AP here comes from our own instrument, not from the official evaluators, and on Waymo uses our conversion of the Car ground truth. No oracle gate exists: the re-sort has no external reference on Waymo or nuScenes [X221]. CenterNet's pool density was never measured and its ranked pool still carries duplicates, so it is not pool-comparable with the other three nuScenes pools [X166][X227].

## Appendix: sources (file:line)

- X42: XDSFIX_RESULT.md:42 — EPro-PnP-Det IoU 0.7 cell base 8.06, re-sort 14.18, gain +6.13 (reproduced in both oracle sessions).
- X54, X55: XDSFIX_RESULT.md:54-55 — loaded FCOS3D / PGD checkpoint names; published 32.1 / 39.3 and 35.8 / 42.5.
- X75: XDSFIX_RESULT.md:75 — residuals vs published: FCOS3D +0.03 mAP / +0.18 NDS; PGD +0.04 mAP / +0.39 NDS; mAP inside 1-decimal rounding; PGD TP-error budget +0.039.
- X99: XDSFIX_RESULT.md:99 — nuScenes car predictions per image: FCOS3D 4.8, PGD 5.0, EPro-PnP 4.5.
- X100: XDSFIX_RESULT.md:100 — Waymo boxes per image: GUPNet 3.97, DEVIANT 4.02.
- X116, X117: XDSFIX_RESULT.md:116-117 — FCOS3D 32.13 / 39.48 vs 32.1 / 39.3; PGD 35.84 / 42.89 vs 35.8 / 42.5.
- X132: XDSFIX_RESULT.md:132 — nuScenes v1.0-trainval val, 6,019 samples, `detection_cvpr_2019`, one checkpoint per detector, no CIs.
- X148-X151: XDSFIX_RESULT.md:148-151 — nuScenes gate table (FCOS3D, PGD, CenterNet-e140 30.265 vs 30.27 = -0.005, EPro-PnP 42.43 vs NDS 0.425 = -0.07).
- X157, X159, X161, X163: XDSFIX_RESULT.md:157,159,161,163 — nuScenes IoU 0.7 re-sort rows (FCOS3D, PGD, EPro-PnP, CenterNet-e140).
- X166: XDSFIX_RESULT.md:166 — CenterNet-e140 not pool-comparable; duplicates remain; density never measured.
- X170: XDSFIX_RESULT.md:170 — Waymo val, 39,848 front-camera frames.
- X176, X177: XDSFIX_RESULT.md:176-177 — GUPNet / DEVIANT Waymo gate PASS (Delta -0.02 / -0.06 and -0.02 / -0.07).
- X186, X188: XDSFIX_RESULT.md:186,188 — Waymo IoU 0.7 re-sort rows (GUPNet, DEVIANT).
- X193, X211: XDSFIX_RESULT.md:193,211 — six detectors, IoU 0.7 gains +4.90 to +7.78.
- X221: XDSFIX_RESULT.md:221 — no Waymo oracle gate exists; same on nuScenes.
- X223: XDSFIX_RESULT.md:223 — PGD +0.39 NDS localised to a +0.039 TP-error budget, not proven.
- X227: XDSFIX_RESULT.md:227 — CenterNet-e140 oracle pool density never measured; residual duplication.
- X230: XDSFIX_RESULT.md:230 — FCOS3D and PGD share the mmdet3d FCOS3D lineage.
- G70-G73: GATE_CERTIFICATE.md:70-73 — Waymo APH-L1 All: 2.25 vs 2.27, 9.88 vs 9.94 (GUPNet); 2.65 vs 2.67, 10.82 vs 10.89 (DEVIANT).
- G74, G75: GATE_CERTIFICATE.md:74-75 — CenterNet mAP 30.265 vs 30.27; EPro-PnP NDS 42.43 vs published 0.425 (official README), Delta -0.07.
- G80: GATE_CERTIFICATE.md:80 — MonoRCNN++ AP3D-L1 All 1.70 vs 4.28, L2 1.59 vs 4.05, IoU 0.5 7.09 vs 11.37 and 6.65 vs 10.79, 4/4 FAIL.
- G161: GATE_CERTIFICATE.md:161 — full val 39,848 frames.
- G164-G166: GATE_CERTIFICATE.md:164-166 — reference = authors' 2022 official-evaluator logs in `gupnet_waymo_run_1050.zip` / `deviant_waymo_run_1051.zip`.
- G173-G175: GATE_CERTIFICATE.md:173-175 — Waymo gate table (GUPNet, DEVIANT gated; MonoRCNN++ failed 4/4).
- G187-G188: GATE_CERTIFICATE.md:187-188 — all 39,848 front-camera val frames, |Delta| <= 0.07.
- G194: GATE_CERTIFICATE.md:194 — Waymo predictions byte-identical to the authors' released outputs, 39,848/39,848.
- O1: gate_audit_closure.txt:80-81 (not one of the two summarised files) — the Waymo checkpoints are Waymo-trained official releases (run_1050 / run_1051), not KITTI-to-Waymo transfer.

Not found verbatim in either summarised file: the statement that EPro-PnP-Det's released README reports only NDS (the files give only "NDS 0.425 (README)"), and the assignment of EPro-PnP-Det to the FCOS3D lineage. Both are in the upstream EPro-PnP-Det README (model tables list NDS only, L15 "extends the one-stage detector FCOS3D"), see reports/extensions/class_capability.txt Part E.

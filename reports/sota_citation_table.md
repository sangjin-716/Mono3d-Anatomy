# SOTA Citation Table — KITTI Car 3D AP (author-reported)

Compiled 2026-06-11 for the ACCV 2026 mono3D diagnosis paper. All numbers below are
**author-reported** (paper tables / official KITTI test-server submissions), **NOT reproduced by us**.
Every row was extracted with a verbatim quote from the cited source; rows without a verbatim
quote carry no numbers (see "Flagged / excluded"). Where independent second-source verification
was run, no corrections were required (`verify.corrected_m` was never set; no row had
`verify.confirmed = false`).

---

## Table 1 — KITTI **test** Car 3D AP|R40, IoU 0.7 (author-reported, NOT reproduced by us)

Sorted by year, then moderate AP. `strict-mono` = trained and tested on single RGB + KITTI 3D box
labels only (no LiDAR / dense-depth supervision, no external depth estimator, no CAD, no temporal
frames). `no` / `yes*` rows are detailed in the "Flagged" section below.

| Model | Venue | Family | Easy | Mod | Hard | strict-mono | source |
|---|---|---|---:|---:|---:|---|---|
| MonoDIS | ICCV 2019 | anchor | 10.37 | 7.94 | 6.40 | yes | [arXiv:1905.12365](https://ar5iv.labs.arxiv.org/html/1905.12365) |
| M3D-RPN | ICCV 2019 | anchor | 14.76 | 9.71 | 7.42 | yes | [MonoFlex Tab.1, arXiv:2104.02323](https://arxiv.org/pdf/2104.02323) [a] |
| SMOKE | CVPRW 2020 | centernet | 14.03 | 9.76 | 7.84 | yes | [arXiv:2002.10111](https://ar5iv.labs.arxiv.org/html/2002.10111) |
| MonoPair | CVPR 2020 | centernet | 13.04 | 9.99 | 8.65 | yes | [arXiv:2003.00504](https://arxiv.org/abs/2003.00504) |
| RTM3D | ECCV 2020 | centernet | 14.41 | 10.34 | 8.77 | yes* | [ECVA PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480647.pdf) [b] |
| MonoDLE | CVPR 2021 | centernet | 17.23 | 12.26 | 10.29 | yes | [arXiv:2103.16237](https://ar5iv.labs.arxiv.org/html/2103.16237) |
| MonoFlex | CVPR 2021 | centernet | 19.94 | 13.89 | 12.07 | yes | [arXiv:2104.02323](https://ar5iv.labs.arxiv.org/html/2104.02323) |
| GUPNet | ICCV 2021 | centernet | 20.11 | 14.20 | 11.77 | yes | [arXiv:2107.13774](https://arxiv.org/abs/2107.13774) |
| DD3D | ICCV 2021 | other | 23.22 | 16.34 | 14.20 | **no** | [arXiv:2108.06417](https://arxiv.org/abs/2108.06417) |
| MonoGround | CVPR 2022 | centernet | 21.37 | 14.36 | 12.62 | yes | [arXiv:2206.07372](https://arxiv.org/abs/2206.07372) |
| DEVIANT | ECCV 2022 | centernet | 21.88 | 14.46 | 11.89 | yes | [official repo](https://github.com/abhi1kumar/DEVIANT) (= paper arXiv:2207.10758 Tab.3) |
| MonoDTR | CVPR 2022 | detr | 21.99 | 15.39 | 12.73 | **no** | [arXiv:2203.10981](https://ar5iv.labs.arxiv.org/html/2203.10981) |
| DID-M3D | ECCV 2022 | centernet | 24.40 | 16.29 | 13.75 | **no** | [arXiv:2207.08531](https://arxiv.org/abs/2207.08531) |
| MonoCon | AAAI 2022 | centernet | 22.50 | 16.46 | 13.95 | yes | [arXiv:2112.04628](https://arxiv.org/abs/2112.04628) |
| MonoDETR | ICCV 2023 | detr | 25.00 | 16.47 | 13.58 | yes | [arXiv:2203.13310v5](https://arxiv.org/html/2203.13310v5) [c] |
| MonoUNI | NeurIPS 2023 | centernet | 24.75 | 16.73 | 13.49 | yes | [NeurIPS 2023 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/2703a0e3c2b33506295a77762338cf24-Paper-Conference.pdf) |
| MonoATT | CVPR 2023 | detr | 24.72 | 17.37 | 15.00 | yes | [arXiv:2303.13018](https://ar5iv.labs.arxiv.org/html/2303.13018) |
| GUPNet++ | TPAMI 2024 | centernet | 24.99 | 16.48 | 14.58 | yes | [arXiv:2310.15624v2](https://arxiv.org/html/2310.15624v2) |
| MonoCD | CVPR 2024 | centernet | 25.53 | 16.59 | 14.53 | **no** | [arXiv:2404.03181v1](https://arxiv.org/html/2404.03181v1) |
| MonoLSS | 3DV 2024 | centernet | 26.11 | 19.15 | 16.94 | yes | [arXiv:2312.14474](https://arxiv.org/abs/2312.14474) |
| MonoDGP | CVPR 2025 | detr | 26.35 | 18.72 | 15.97 | yes | [arXiv:2410.19590v1](https://arxiv.org/html/2410.19590v1) [d] |
| MonoCLUE | AAAI 2026 | detr | 27.94 | 19.70 | 16.69 | **no** | [arXiv:2511.07862](https://arxiv.org/abs/2511.07862) |
| MonoIA | CVPR 2026 | detr | 29.52 | 20.29 | 17.93 | yes* | [arXiv:2603.27059](https://arxiv.org/abs/2603.27059) [e] |

[a] M3D-RPN's original ICCV 2019 paper predates the Oct-2019 R40 metric change and reports R11
only (test R11: 20.65/15.70/13.32). The R40 test numbers 14.76/9.71/7.42 are the official KITTI
leaderboard values for the authors' test-server submission, quoted verbatim from MonoFlex (CVPR
2021) Table 1 and uniformly cited across later papers (MonoFlex, GrooMeD-NMS, MonoDLE).
[b] RTM3D paper Table 4 labels the metric "AP3D IoU=0.7"; the KITTI test server has used AP|R40
since Oct 2019, and later papers (MonoFlex Tab.1) cite the identical 14.41/10.34/8.77 under R40.
[c] ICCV 2023 camera-ready / arXiv v5 numbers. Older arXiv versions (v1–v3) of MonoDETR report
different/lower test numbers (e.g. 23.65/15.92/12.99 cited by S3-MonoDETR) — do not mix.
[d] Official MonoDGP GitHub README additionally reports a test-server resubmission of the released
checkpoint (26.12/18.87/16.79); the paper-reported numbers used here remain 26.35/18.72/15.97.
[e] MonoIA paper Table 3 values. The official README lists test Mod 19.11 (equal to MonoCoP's
paper value — likely a README typo or a different checkpoint submission); paper values used.

---

## Table 2 — KITTI **val** (Chen split, 3712/3769) Car 3D AP|R40, IoU 0.7 (author-reported, where available)

SMOKE (val reported in R11 only), RTM3D (val reported in R11 only), and DD3D (no Car 3D-AP val
E/M/H published) have no R40 val numbers and are omitted.

| Model | Venue | Easy | Mod | Hard | source |
|---|---|---:|---:|---:|---|
| MonoDIS | ICCV 2019 | 11.06 | 7.60 | 6.37 | [arXiv:1905.12365](https://ar5iv.labs.arxiv.org/html/1905.12365) |
| M3D-RPN | ICCV 2019 | 14.53 | 11.07 | 8.65 | [MonoFlex Tab.1, arXiv:2104.02323](https://arxiv.org/pdf/2104.02323) [f] |
| MonoPair | CVPR 2020 | 16.28 | 12.30 | 10.42 | [arXiv:2003.00504](https://arxiv.org/abs/2003.00504) |
| MonoDLE | CVPR 2021 | 17.45 | 13.66 | 11.68 | [arXiv:2103.16237](https://ar5iv.labs.arxiv.org/html/2103.16237) |
| GUPNet | ICCV 2021 | 22.76 | 16.46 | 13.72 | [arXiv:2107.13774](https://arxiv.org/abs/2107.13774) |
| MonoFlex | CVPR 2021 | 23.64 | 17.51 | 14.83 | [arXiv:2104.02323](https://ar5iv.labs.arxiv.org/html/2104.02323) |
| DID-M3D | ECCV 2022 | 22.98 | 16.12 | 14.03 | [arXiv:2207.08531](https://arxiv.org/abs/2207.08531) |
| DEVIANT | ECCV 2022 | 24.63 | 16.54 | 14.52 | [official repo](https://github.com/abhi1kumar/DEVIANT) |
| MonoDTR | CVPR 2022 | 24.52 | 18.57 | 15.51 | [arXiv:2203.10981](https://ar5iv.labs.arxiv.org/html/2203.10981) |
| MonoGround | CVPR 2022 | 25.24 | 18.69 | 15.58 | [arXiv:2206.07372](https://arxiv.org/abs/2206.07372) |
| MonoCon | AAAI 2022 | 26.33 | 19.01 | 15.98 | [arXiv:2112.04628](https://arxiv.org/abs/2112.04628) [g] |
| MonoUNI | NeurIPS 2023 | 24.51 | 17.18 | 14.01 | [NeurIPS 2023 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/2703a0e3c2b33506295a77762338cf24-Paper-Conference.pdf) [h] |
| MonoDETR | ICCV 2023 | 28.84 | 20.61 | 16.38 | [arXiv:2203.13310v5](https://arxiv.org/html/2203.13310v5) |
| MonoATT | CVPR 2023 | 29.01 | 23.49 | 19.60 | [arXiv:2303.13018](https://ar5iv.labs.arxiv.org/html/2303.13018) |
| MonoCD | CVPR 2024 | 24.22 | 18.27 | 15.42 | [arXiv:2404.03181v1](https://arxiv.org/html/2404.03181v1) [i] |
| MonoLSS | 3DV 2024 | 25.91 | 18.29 | 15.94 | [arXiv:2312.14474](https://arxiv.org/abs/2312.14474) [j] |
| GUPNet++ | TPAMI 2024 | 29.03 | 20.45 | 17.89 | [arXiv:2310.15624v2](https://arxiv.org/html/2310.15624v2) |
| MonoDGP | CVPR 2025 | 30.76 | 22.34 | 19.02 | [arXiv:2410.19590v1](https://arxiv.org/html/2410.19590v1) |
| MonoCLUE | AAAI 2026 | 33.74 | 24.10 | 20.58 | [arXiv:2511.07862](https://arxiv.org/abs/2511.07862) |
| MonoIA | CVPR 2026 | 33.61 | 24.40 | 20.80 | [arXiv:2603.27059](https://arxiv.org/abs/2603.27059) [k] |

[f] Not in M3D-RPN's original paper (R11 era; original val1 R11: 20.27/17.06/15.21). R40 Chen-split
val numbers are the community-standard re-evaluation (originally from Kinematic3D, same first
author), quoted verbatim from MonoFlex Table 1.
[g] Paper Table 5 values. Official MonoCon README lists the "paper" val row as 26.33/19.03/16.00 —
a minor discrepancy with the paper's own Table 5 (26.33/19.01/15.98); paper values used.
[h] MonoUNI's paper does not explicitly label its val split as Chen split (standard KITTI val
assumed); val row is the full-model ablation row (Table 5, row f).
[i] MonoCD val row = MonoFlex+CD variant (the released "MonoCD" model per official README). Best
val variant in paper Table 2 is MonoCon+CD = 26.45/19.37/16.38.
[j] MonoLSS has no dedicated val SOTA table; val row is the full-method ablation row (Table 5),
median over 5 seeds at 150 epochs.
[k] Single-dataset (KITTI-only) training. MonoIA's multi-dataset (KITTI+nuScenes+Waymo) val result
is higher (Mod 28.91, paper Table 4) and is NOT the number reported here.

---

## Flagged / excluded

### Flagged: not strictly monocular (`strict_mono != yes`) — kept in tables, marked **no**

These methods are RGB-only at inference but use extra signals at training time, violating the
strict "single RGB + KITTI 3D box labels only" criterion:

- **DD3D** (ICCV 2021): large-scale dense-depth **pre-training on DDAD15M (~15M frames) with
  LiDAR-projected ground-truth depth** ("L1 loss between predicted depth and projective
  ground-truth depth"), plus COCO 2D-detection pre-initialization.
- **MonoDTR** (CVPR 2022): **LiDAR-projected sparse depth maps** as auxiliary depth supervision
  during training ("We project the LiDAR signals into the image plane to generate the sparse
  ground truth depth map"); official repo data prep requires KITTI `velodyne/`.
- **DID-M3D** (ECCV 2022): **LiDAR-projected, completion-densified depth maps** as visual-depth
  supervision at training; official repo requires `depth_dense` files.
- **MonoCD** (CVPR 2024): paper claims Extra Data = "None", but the official repo's training
  requires **KITTI road-plane files (LiDAR-fitted ground planes**, OpenMMLab `train_planes.zip`)
  to supervise the Horizon Heatmap.
- **MonoCLUE** (AAAI 2026): **external pretrained SAM (Segment Anything) segmentation masks** as
  training-time supervision for its region/clustering heads, despite the paper's "without extra
  information" claim. No LiDAR/depth/CAD/temporal signals.

### Flagged: `yes*` — strictly monocular with a train-time caveat

- **RTM3D** (ECCV 2020): no LiDAR/depth/CAD/temporal anywhere, but train-time keypoint-label
  **augmentation uses KITTI right-camera (stereo-pair) images** ("We project the 3D bounding box
  of Ground Truths in the left and right images to obtain Ground Truth keypoints").
- **MonoIA** (CVPR 2026): paper Extra Data = "None" (no LiDAR/depth/CAD/temporal), but uses
  **frozen ChatGPT-4o-generated text prompts + a CLIP ViT-H/14 text encoder** to embed camera
  intrinsics at both train and test — a language prior on intrinsics rather than a depth/geometry
  signal.

### Excluded: extraction/verification failed — NO numbers reported

- **Unnamed entry (arXiv 2025, detr family)**: the extraction pass returned only
  `{venue: arXiv, year: 2025, family: detr}` — no model name, no numbers, no verbatim quote, no
  source URL, and no `found=true` flag. Per policy (no number without a quote), this entry is
  excluded entirely and carries no numbers. It must be re-extracted before any use.

No row in this batch had `verify.confirmed = false`, and no `verify.corrected_m` correction was
issued; all second-source checks that ran (MonoPair, SMOKE, RTM3D, MonoDLE, MonoFlex, GUPNet,
MonoDTR, MonoGround, DEVIANT, DID-M3D, MonoDETR, MonoATT, MonoUNI, MonoCD) matched the claimed
moderate AP exactly (delta 0.00). Rows with `verify = null` (M3D-RPN, MonoDIS, MonoCon, DD3D,
MonoLSS, GUPNet++, MonoDGP, MonoCLUE, MonoIA) are quote-backed single-source extractions that did
not receive an independent second-source pass.

---

## Footer notes

1. **Author-reported, not reproduced.** All test numbers are the authors' official KITTI
   test-server (leaderboard) results as reported in the cited papers/tables; all val numbers are
   the authors' reported Chen-split results. We reproduce **only** our 12-model analysis panel
   separately — see `reports/detector_adapters_gates.md`.
2. **Metric.** KITTI test server = Car 3D AP|R40 at IoU 0.7 since Oct 2019. Legacy AP|R11 numbers
   (pre-2020 papers, some val tables) are never mixed into the tables above; where a paper reports
   only R11 on val (SMOKE: 14.76/12.85/11.50; RTM3D val1: 20.77/16.86/16.63), the R40 val cell is
   omitted rather than substituted.
3. **Column-order traps caught during extraction** (do not re-introduce): MonoPair AP_BEV
   (19.28/14.83/12.89) vs AP_3D; MonoATT search snippets conflating BEV/3D columns; MonoDGP
   Table 1 ordering Test-BEV / Test-3D / Val-BEV / Val-3D; RTM3D's val-R11 numbers sitting in the
   same row as test-R40.
4. **Released-checkpoint vs paper numbers**: GUPNet (later checkpoint test 22.26/15.02/13.12),
   MonoDGP (README resubmission 26.12/18.87/16.79), MonoDLE (repo val 17.94/13.72/12.10), SMOKE
   (README 14.17/9.88/8.63), MonoCD (README reproduced val 25.99/19.12/16.03), MonoCLUE (README
   warns ±1 AP3D|R40 training variance) — tables above always use the **paper-reported** numbers.
5. **Verification context.** PapersWithCode is defunct (redirects to HuggingFace), so second-source
   verification used comparison tables in independent peer-reviewed follow-up papers (MonoFlex,
   MonoDTR, MonoCon, MonoCD, MonoDGP, MonoNeRD, OBMO, GUPNet++, MonoCLUE, MonoDETRNext, SPAN) and
   official GitHub READMEs.

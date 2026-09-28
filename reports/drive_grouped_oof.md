# Phase 1 — drive-grouped vs image-fold OOF (pooled cross-fitted AP)

Headline = pooled cross-fitted AP (all held-out fold predictions concatenated -> one official KITTI eval). Canonical OMP=1.

| detector | base AP | image-OOF ΔAP | drive-OOF ΔAP | image R² | drive R² | Δ(img−drive) | #drives |
|---|---|---|---|---|---|---|---|
| MonoDGP | 22.29 | +3.858 | -1.091 | 0.1876 | 0.0192 | +4.949 | 45 |
| MonoDETR | 20.978 | +0.147 | -2.748 | 0.1804 | 0.0307 | +2.895 | 45 |
| MonoCoP | 23.886 | +1.119 | -3.396 | 0.1978 | -0.0165 | +4.515 | 45 |
| MonoFlex | 15.571 | +0.971 | -1.666 | 0.1803 | 0.0068 | +2.637 | 45 |
| GUPNet | 16.481 | +1.749 | -0.703 | 0.1677 | 0.0037 | +2.452 | 45 |

## drive size imbalance (val) & per-fold

**MonoDGP** drives=45 size max/med/min=28850/550/50
  - fold0: 8 drives, 755 imgs, 3866 matched
  - fold1: 9 drives, 754 imgs, 8452 matched
  - fold2: 10 drives, 754 imgs, 6631 matched
  - fold3: 8 drives, 753 imgs, 7678 matched
  - fold4: 10 drives, 753 imgs, 11735 matched

**MonoDETR** drives=45 size max/med/min=27820/550/50
  - fold0: 10 drives, 766 imgs, 2197 matched
  - fold1: 8 drives, 754 imgs, 5521 matched
  - fold2: 9 drives, 750 imgs, 4530 matched
  - fold3: 8 drives, 749 imgs, 7580 matched
  - fold4: 10 drives, 750 imgs, 6308 matched

**MonoCoP** drives=45 size max/med/min=28850/550/50
  - fold0: 8 drives, 755 imgs, 3805 matched
  - fold1: 9 drives, 754 imgs, 7237 matched
  - fold2: 10 drives, 754 imgs, 6737 matched
  - fold3: 8 drives, 753 imgs, 7493 matched
  - fold4: 10 drives, 753 imgs, 11300 matched

**MonoFlex** drives=45 size max/med/min=16476/278/28
  - fold0: 8 drives, 769 imgs, 1453 matched
  - fold1: 10 drives, 739 imgs, 4106 matched
  - fold2: 8 drives, 714 imgs, 4547 matched
  - fold3: 10 drives, 855 imgs, 3055 matched
  - fold4: 9 drives, 692 imgs, 4360 matched

**GUPNet** drives=45 size max/med/min=20436/339/24
  - fold0: 8 drives, 665 imgs, 951 matched
  - fold1: 10 drives, 846 imgs, 3235 matched
  - fold2: 9 drives, 712 imgs, 3554 matched
  - fold3: 9 drives, 769 imgs, 4275 matched
  - fold4: 9 drives, 777 imgs, 4526 matched

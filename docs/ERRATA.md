# Errata

## 2026-10-05: where the dumps differ from the repositories' own output

Found after acceptance, by comparing each dump with the files that the repository's own tester
writes. The camera-ready supplementary states the same points (Sec. C, "Deviations Found after
Acceptance"). The released dumps and every number in the paper are unchanged.

### 1. Yaw of MonoDGP and MonoCoP

- `adapters/dgp_cop_dump.py` and `adapters/detr_preflatten_dump.py` compute
  `rotation_y = alpha2ry(alpha, u)` with `u` at the projected 3D centre. Both repositories use
  the centre of the 2D box (`lib/helpers/decode_helper.py`).
- The 2D box in the dump has the native width and height but is centred on the projected 3D
  centre. This does not enter 3D AP.
- Counts, scores, 3D centres, dimensions and alpha agree with the repository output to its
  two-decimal precision. Yaw differs by more than 0.011 rad for about a fifth of the boxes, by at
  most 0.12 rad.
- Taking only yaw from the repository output reproduces its AP_R40 exactly.
- Effect on base AP (Car Moderate, all-point, native pool):

  | Detector | dump | with the repository yaw |
  |---|---|---|
  | MonoDGP | 22.82 | 23.31 |
  | MonoCoP | 24.34 | 24.69 |

- Every result for these two detectors in the paper uses the adapter's yaw. Their ceilings and gaps
  have not been recomputed.
- A smaller difference of the same kind (at most 0.05 rad, 0.07 AP_R40) affects the 23-field dump
  of MonoIA (`monoia_val.csv`). Its pre-flatten dump matches the repository output to rounding.

### 2. MonoCLUE

- MonoCLUE inference is stochastic: a random k-means initialisation, and an attention dropout that
  stays active at test time. The repository's tester sets the seed. The adapter does not.
- The dump is therefore a separate run of the same pipeline, not a box-for-box copy of the
  repository output. All-point AP is 24.55 for the dump and 24.75 for the repository output.

### 3. Reproduction column

In supplementary Table 2 and in [`adapters/README.md`](../adapters/README.md), "ours" is the
repository's own evaluation for six detectors only (M3D-RPN, MonoDLE, MonoFlex\*, DEVIANT,
MonoGround\*, MonoCon). For the other six it is:

- GUPNet, MonoDGP, MonoCLUE, MonoIA: AP_R40 of the dump on the uniform pool.
- MonoCoP: AP_R40 of the dump at the native operating point.
- MonoDETR: the README value.

The repository's own evaluation of the released checkpoints gives (Car Moderate AP_R40):

| Detector | printed "ours" | repository's own evaluation | README, released checkpoint |
|---|---|---|---|
| GUPNet | 16.48 | 16.23 | 16.23 |
| MonoDETR | 20.83 | 20.78 | 20.83 |
| MonoDGP | 22.29 | 22.50 | 22.50 |
| MonoCoP | 23.84 | 24.06 | 24.05 |
| MonoCLUE | 24.20 | 24.12 | 24.11 |
| MonoIA | 24.48 | 24.48 | 24.40 |

- Measured with the unmodified upstream code at the commits listed in `adapters/README.md`, the
  released checkpoints (hash-identical to the published files), each repository's own tester, and
  the official evaluator vendored in `evaluator/kitti_eval`.
- MonoDETR and MonoDGP need two import aliases on torch 2: their version check takes the
  torch < 1.9 branch there.
- The MonoCLUE value is from a June 2026 run. The others were re-run on 2026-10-05.
- The repositories' writers round every field to two decimals, and the dumps keep full
  precision. This alone moves AP_R40 by up to about 0.25. GUPNet's output scores 16.46 at full
  precision and 16.23 as written.

### Status

A corrected dump for MonoDGP and MonoCoP and a re-run of their numbers are planned.

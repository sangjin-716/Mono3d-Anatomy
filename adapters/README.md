# Adapters

These scripts wrote the per-prediction dumps. You only need them to regenerate a dump. The paper
uses the released dumps, listed with their columns in [`data/DUMPS.md`](../data/DUMPS.md).

## Running

- Run each adapter in the detector's own repo and environment (torch 1.4 for MonoFlex\* and
  MonoGround\*, 1.9 for GUPNet and the MonoDETR family, 1.10 for the rest).
- Most take `--repo` and `--ckpt`, with defaults under `paths.UPSTREAM_ROOT`. Each script header
  lists its arguments. Output goes to `<paths.OUT_DIR>/dumps/`. The MonoFlex and modern MonoGround
  scripts write `monoflex_val.csv` or `monoground_val.csv`. Released names add `_orig` or `_modern`.
- `m3drpn_*.py`, `test_rpn_3d_accv.py`, `dgp_cop_dump.py` and `detr_preflatten_*.py` must be run
  from the upstream repo root.
- The scripts are cleaned up and not re-run since. Please open an issue if one fails.

```bash
python adapters/deviant_dump.py --repo <UPSTREAM_ROOT>/DEVIANT --ckpt <checkpoint>
```

## Per detector

Car Moderate AP_R40 on KITTI val. Checkpoints are linked from each repo's README unless noted.
Each dump passed a reproduction check (G1) and a match to the detector's native output (G2), see [`reports/detector_adapters_gates.md`](../reports/detector_adapters_gates.md). G2 in short:

- M3D-RPN: re-running the native NMS on the dump gives 14.538/11.099/8.670 (native 14.531/11.073/8.646).
- MonoDLE and DEVIANT match box for box (the native writer rounds to 2 decimals).
- MonoCon gives 26.031/19.015/15.912, at most 0.007 from native.
- The MonoDETR family reproduces the native top-50 in counts, scores, 3D centres and dimensions.
  Yaw differs for MonoDGP and MonoCoP, and MonoCLUE's dump is a separate stochastic run. See
  [`docs/ERRATA.md`](../docs/ERRATA.md).
- MonoFlex\* and MonoGround\* are the output of each repo's own post-processor.

| Detector | Commit | Adapter | Checkpoint | Published / ours | Repo's own eval |
|---|---|---|---|---|---|
| [M3D-RPN](https://github.com/garrickbrazil/M3D-RPN) | `bf204e3f95f6` | `m3drpn_dump_accv.py` | `M3D-RPN-Release.zip` (val1) | 11.07 / 11.07 | 11.07 |
| [MonoDLE](https://github.com/xinzhuma/monodle) | `e426aa65fdc7` | `monodle_dump.py` | README | 13.72 / 14.57 | 14.57 |
| [MonoFlex](https://github.com/zhangyp15/MonoFlex)\* | `ec6da017c325` | `monoflex_dump_orig.py` | README | 17.51 / 17.34 | 17.34 |
| [GUPNet](https://github.com/SuperMHP/GUPNet) | `d0e02cad228f` | `gupnet_dump.py` | README | 16.46 / 16.48 | 16.23 |
| [DEVIANT](https://github.com/abhi1kumar/DEVIANT) | `2e6eca6e27d7` | `deviant_dump.py` | `run_221` | 16.54 / 16.49 | 16.49 |
| [MonoGround](https://github.com/cfzd/MonoGround)\* | `05b3baf73228` | `monoground_dump_orig.py` | README | 18.69 / 18.69 | 18.69 |
| [MonoCon](https://github.com/2gunsu/monocon-pytorch) | `908807bdd8d4` | `monocon_dump.py` | README | 19.02 / 19.02 | 19.02 |
| [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR) | `6994b9f51240` | `monodetr_dump.py` | README | 20.83 / 20.83 | 20.78 |
| [MonoDGP](https://github.com/PuFanqi23/MonoDGP) | `aa059a18214a` | `dgp_cop_dump.py` | README | 22.34 / 22.29 | 22.50 |
| [MonoCoP](https://alanzhangcs.github.io/monocop-page) | not recorded | `dgp_cop_dump.py` | [HF](https://huggingface.co/zhihao406/MonoCoP) | 23.98 / 23.84 | 24.06 |
| [MonoCLUE](https://github.com/SungHunYang/MonoCLUE) | `016d3e8d3c99` | `monoclue_dump.py` | README | 24.10 / 24.20 | 24.12 |
| [MonoIA](https://github.com/alanzhangcs/MonoIA) | `69d6ee30ca5e` | `monoia_dump.py` | [HF](https://huggingface.co/zhihao406/MonoIA) `MonoIA_KITTI_Val.pth` | 24.40 / 24.48 | 24.48 |

- M3D-RPN publishes AP_R11, so 11.07 is our AP_R40. Apply
  `patches/M3D-RPN_rpn_util_py_cpu_nms.patch`. `test_rpn_3d_accv.py` is the native run.
  `LICENSE-M3D-RPN` covers `test_rpn_3d_accv.py`, a copy of the upstream `test_rpn_3d.py`.
- MonoDLE is 0.85 above published with the same code and data, due to the environment.
- "Ours" is the value printed in the paper. "Repo's own eval" is the output of each repository's
  own tester on the released checkpoint. They differ for six detectors, see
  [`docs/ERRATA.md`](../docs/ERRATA.md).
- MonoCoP's published value is that of its paper. Its README lists 24.05 for the released Car
  checkpoint.
- `gupnet_native_eval.py` grades GUPNet's own tester output with the same evaluator.
- MonoFlex\* and MonoGround\*: `build_monoflex_orig_env.sh` builds the original torch 1.4
  environment. `monoflex_dump.py` and `monoground_dump.py` give the modern builds (15.54, 16.79).
  Both read KITTI in the MonoFlex layout (`--kitti_dir`, or `DATA_DIR` in MonoGround's config).
- MonoCon uses the 2gunsu/monocon-pytorch re-implementation. Its published value is from that
  README. Apply `patches/MonoCon_base_engine_map_location.patch`.
- Full candidate pools: `m3drpn_dump_floor0.py`, `monodetr_preflatten_dump.py`,
  `detr_preflatten_dump.py` (MonoDGP, MonoCoP), `detr_preflatten_native.py` (MonoCLUE, MonoIA).

#!/usr/bin/env python3
"""Waymo official-evaluator reproduction gate, v2 (adds MonoRCNN++).

Reads the evaluator tables written by run_gate.sh (crossbench/waymo_gate/gate_<det>_fullval_<iou>.txt
in the re-run output directory) and prints the gate deltas against the published references
transcribed below.

INSTRUMENT
  python 3.9, tensorflow 2.11.0, waymo-open-dataset-tf-2-11-0 1.6.1 (pip wheel, prebuilt
  waymo_open_dataset/metrics/ops/metrics_ops.so = the official C++ kernels
  that also back compute_detection_metrics_main).
  Harness = DEVIANT's own data/waymo/waymo_eval{,_0_5}.py with ONE hunk changed
  (hardcoded pd_set/gt_set/pd_dir/gt_dir -> four WEVAL_* env vars), see
  waymo_eval_patched.py / waymo_eval_0_5_patched.py. Metric config, Hungarian matcher,
  breakdown generators, printing untouched.
  Full val: 39,848 front-camera frames, 569,234 GT boxes (all classes).

REFERENCES (values transcribed in the dictionaries below)
  A. DEVIANT README.md Model Zoo, "Waymo Val" rows, metric column APH-L1.
     -> GUPNet run_1050, DEVIANT run_1051. Columns All/Easy/Med/Hard at 0.7 and 0.5.
  B. Shi, Chen, Kim. "Multivariate Probabilistic Monocular 3D Object Detection."
     WACV 2023, Table 3. Metric = AP3D (NOT APH), IoU>0.5 and IoU>0.7,
     LEVEL 1 and LEVEL 2, columns Overall/0-30m/30-50m/50m-inf.
     -> this is MonoRCNN++'s OWN published Waymo table, and it also carries
        GUPNet and DEVIANT rows.
  C. Yang et al. "MonoCLUE." arXiv:2511.07862v1, SUPPLEMENTARY Table 4.
     AP3D and APH3D, IoU 0.7, Level 1 and Level 2, All/0-30/30-50/50-inf.
  D. The evaluator log included in the DEVIANT release archives (see AUTHORS_LOG below).
  Independence caveat: B and C report values numerically identical to A where they
  overlap, i.e. they are transcriptions of the original DEVIANT/GUPNet tables, not
  independent re-runs. They add coverage (LEVEL_2, AP3D) but not independent
  measurement for GUPNet/DEVIANT. The MonoRCNN++ row of B is its own published reference.

TOLERANCE
  |delta| <= 0.15 absolute APH on the All column, for GUPNet and DEVIANT, at each IoU
  threshold. For MonoRCNN++ the requirement is "its own published Waymo number reproduced";
  we apply the same 0.15 absolute on the All column, on AP3D because AP3D is the metric its
  paper reports.
"""
import re, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import out_path
HERE = os.path.dirname(out_path(os.path.join('crossbench', 'waymo_gate', 'gate_deltas_v2.txt')))
TOL = 0.15
COLS = ('All', '[0,30)', '[30,50)', '[50,inf)')

# ---- Reference D: the AUTHORS' OWN official-evaluator stdout, 2022 ---------
# extracted from the released archives
#   checkpoints/waymo/gupnet_waymo_run_1050.zip  -> run_1050/log/20220718_224309
#   checkpoints/waymo/deviant_waymo_run_1051.zip -> run_1051/log/20220718_224400
# (same waymo_eval.py, same val.txt / val_org.txt, run on the authors' machine)
# (det, thr) -> {level: (AP 4 cols, APH 4 cols)}
AUTHORS_LOG = {
    ('gupnet', 'iou07'):  {1: ((2.28, 6.15, 0.81, 0.03), (2.27, 6.11, 0.80, 0.03)),
                           2: ((2.14, 6.13, 0.78, 0.02), (2.12, 6.08, 0.77, 0.02))},
    ('gupnet', 'iou05'):  {1: ((10.02, 24.78, 4.84, 0.22), (9.94, 24.59, 4.78, 0.22)),
                           2: ((9.39, 24.69, 4.67, 0.19), (9.31, 24.50, 4.62, 0.19))},
    ('deviant', 'iou07'): {1: ((2.69, 6.95, 0.99, 0.02), (2.67, 6.90, 0.98, 0.02)),
                           2: ((2.52, 6.93, 0.95, 0.02), (2.50, 6.87, 0.94, 0.02))},
    ('deviant', 'iou05'): {1: ((10.98, 26.85, 5.13, 0.18), (10.89, 26.64, 5.08, 0.18)),
                           2: ((10.29, 26.75, 4.95, 0.16), (10.20, 26.54, 4.90, 0.16))},
}

# ---- Reference A: DEVIANT README, APH LEVEL_1 -----------------------------
README_APH_L1 = {
    ('gupnet',  'iou07'): (2.27, 6.11, 0.80, 0.03),
    ('deviant', 'iou07'): (2.67, 6.90, 0.98, 0.02),
    ('gupnet',  'iou05'): (9.94, 24.59, 4.78, 0.22),
    ('deviant', 'iou05'): (10.89, 26.64, 5.08, 0.18),
}
# ---- Reference B: MonoRCNN++ WACV23 Table 3, AP3D --------------------------
WACV_AP3D = {
    ('gupnet',     'iou07', 1): (2.28, 6.15, 0.81, 0.03),
    ('gupnet',     'iou07', 2): (2.14, 6.13, 0.78, 0.02),
    ('deviant',    'iou07', 1): (2.69, 6.95, 0.99, 0.02),
    ('deviant',    'iou07', 2): (2.52, 6.93, 0.95, 0.02),
    ('monorcnnpp', 'iou07', 1): (4.28, 9.84, 0.91, 0.09),
    ('monorcnnpp', 'iou07', 2): (4.05, 9.81, 0.89, 0.08),
    ('gupnet',     'iou05', 1): (10.02, 24.78, 4.84, 0.22),
    ('gupnet',     'iou05', 2): (9.39, 24.69, 4.67, 0.19),
    ('deviant',    'iou05', 1): (10.98, 26.85, 5.13, 0.18),
    ('deviant',    'iou05', 2): (10.29, 26.75, 4.95, 0.16),
    ('monorcnnpp', 'iou05', 1): (11.37, 27.95, 4.07, 0.42),
    ('monorcnnpp', 'iou05', 2): (10.79, 27.88, 3.98, 0.39),
}
# ---- Reference C: MonoCLUE suppl. Table 4, IoU 0.7 -------------------------
MONOCLUE = {  # (det, level) -> (AP3D 4 cols, APH3D 4 cols)
    ('gupnet', 1):  ((2.28, 6.15, 0.81, 0.03), (2.27, 6.11, 0.80, 0.03)),
    ('gupnet', 2):  ((2.14, 6.13, 0.78, 0.02), (2.12, 6.08, 0.77, 0.02)),
    ('deviant', 1): ((2.69, 6.95, 0.99, 0.02), (2.67, 6.90, 0.98, 0.02)),
    ('deviant', 2): ((2.52, 6.93, 0.95, 0.02), (2.50, 6.87, 0.94, 0.02)),
}

ROW = re.compile(r'^VEHICLE\s+\|\s+(\d)\s+\|((?:\s+[\d.]+){4})\s*\|((?:\s+[\d.]+){4})\s*\|')

def parse(path):
    out = {}
    for line in open(path):
        m = ROW.match(line)
        if m:
            lvl = int(m.group(1))
            out[lvl] = ([float(x) for x in m.group(2).split()],
                        [float(x) for x in m.group(3).split()])
    return out

OURS = {}
for det in ('gupnet', 'deviant', 'monorcnnpp'):
    for thr in ('iou07', 'iou05'):
        f = f'{HERE}/gate_{det}_fullval_{thr}.txt'
        if os.path.exists(f):
            OURS[(det, thr)] = parse(f)

bar = '=' * 100
print(bar)
print('WAYMO OFFICIAL-EVALUATOR REPRODUCTION GATE v2   (waymo-open-dataset-tf-2-11-0 1.6.1 / TF 2.11.0)')
print('full val, 39,848 front-camera frames, 569,234 GT boxes fed to the evaluator, vehicle/Car class')
print(bar)

fails, passes = [], []

print('\n--- GATE 0  vs the AUTHORS\' OWN 2022 official-evaluator stdout (strongest reference)')
print('    all VEHICLE cells, AP_3D and APH_3D, LEVEL_1 and LEVEL_2, both IoU thresholds')
print(f'{"det":11s} {"IoU":6s} {"L":2s} {"metric":7s} {"col":9s} {"ours":>7s} {"authors":>8s} {"delta":>8s}  verdict')
for det in ('gupnet', 'deviant'):
    for thr in ('iou07', 'iou05'):
        if (det, thr) not in OURS: continue
        for lvl in (1, 2):
            ap, aph = OURS[(det, thr)][lvl]
            rap, raph = AUTHORS_LOG[(det, thr)][lvl]
            for name, mine, ref in (('AP_3D', ap, rap), ('APH_3D', aph, raph)):
                for i, c in enumerate(COLS):
                    d = mine[i] - ref[i]
                    v = ('PASS' if abs(d) <= TOL else 'FAIL') if i == 0 else 'not gated'
                    print(f'{det:11s} {thr:6s} {lvl:<2d} {name:7s} {c:9s} '
                          f'{mine[i]:7.2f} {ref[i]:8.2f} {d:+8.2f}  {v}')

print('\n--- GATE 1  APH LEVEL_1 vs DEVIANT README Model Zoo   [tol |d|<=0.15 on All]')
print(f'{"det":11s} {"IoU":6s} {"col":9s} {"ours":>7s} {"published":>10s} {"delta":>8s}  verdict')
for det in ('gupnet', 'deviant'):
    for thr in ('iou07', 'iou05'):
        if (det, thr) not in OURS: continue
        aph = OURS[(det, thr)][1][1]
        ref = README_APH_L1[(det, thr)]
        for i, c in enumerate(COLS):
            d = aph[i] - ref[i]
            if i == 0:
                v = 'PASS' if abs(d) <= TOL else 'FAIL'
                (passes if v == 'PASS' else fails).append((det, thr, 'APH-L1 All', d))
            else:
                v = 'not gated'
            print(f'{det:11s} {thr:6s} {c:9s} {aph[i]:7.2f} {ref[i]:10.2f} {d:+8.2f}  {v}')

print('\n--- GATE 2  AP3D LEVEL_1/LEVEL_2 vs MonoRCNN++ (WACV 2023) Table 3   [tol |d|<=0.15 on All]')
print('    for MonoRCNN++ this is its OWN published table (self-gate)')
print(f'{"det":11s} {"IoU":6s} {"L":2s} {"col":9s} {"ours":>7s} {"published":>10s} {"delta":>8s}  verdict')
for det in ('gupnet', 'deviant', 'monorcnnpp'):
    for thr in ('iou07', 'iou05'):
        if (det, thr) not in OURS: continue
        for lvl in (1, 2):
            ap = OURS[(det, thr)][lvl][0]
            ref = WACV_AP3D[(det, thr, lvl)]
            for i, c in enumerate(COLS):
                d = ap[i] - ref[i]
                if i == 0:
                    v = 'PASS' if abs(d) <= TOL else 'FAIL'
                    if det == 'monorcnnpp':
                        (passes if v == 'PASS' else fails).append(
                            (det, thr, f'AP3D-L{lvl} All', d))
                else:
                    v = 'not gated'
                print(f'{det:11s} {thr:6s} {lvl:<2d} {c:9s} {ap[i]:7.2f} {ref[i]:10.2f} {d:+8.2f}  {v}')

print('\n--- GATE 3  AP3D + APH3D, IoU 0.7, L1/L2 vs MonoCLUE (arXiv 2511.07862v1) suppl. Table 4')
print('    corroboration only; its GUPNet/DEVIANT values are transcriptions of A/B, not a re-run')
print(f'{"det":11s} {"L":2s} {"metric":7s} {"col":9s} {"ours":>7s} {"published":>10s} {"delta":>8s}')
for det in ('gupnet', 'deviant'):
    if (det, 'iou07') not in OURS: continue
    for lvl in (1, 2):
        ap, aph = OURS[(det, 'iou07')][lvl]
        rap, raph = MONOCLUE[(det, lvl)]
        for name, mine, ref in (('AP3D', ap, rap), ('APH3D', aph, raph)):
            for i, c in enumerate(COLS):
                print(f'{det:11s} {lvl:<2d} {name:7s} {c:9s} {mine[i]:7.2f} {ref[i]:10.2f} {mine[i]-ref[i]:+8.2f}')

print('\n' + bar)
print(f'GATED CELLS (All column only): {len(passes)} PASS, {len(fails)} FAIL')
for p in passes: print(f'  PASS  {p[0]:11s} {p[1]:6s} {p[2]:14s} delta {p[3]:+.2f}')
for f_ in fails: print(f'  FAIL  {f_[0]:11s} {f_[1]:6s} {f_[2]:14s} delta {f_[3]:+.2f}')
print(bar)
sys.exit(1 if fails else 0)

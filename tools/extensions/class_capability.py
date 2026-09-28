"""Class-capability census of the twelve released checkpoints (supplementary Sec. O) and checks
of the upstream release documents for the cross-benchmark detectors (supplementary Sec. P).

It backs statements of supplementary Secs. O and P and main Sec. 5.1 that had no report of their
own, and changes no number of the paper. Produces reports/extensions/class_capability.txt (a
re-run writes reports_rerun/extensions/class_capability.txt). Read-only on every input.

  Part A  Released per-prediction dumps (data/dumps). Class-label column, rows, images and
          the Car-selection line of the adapter that wrote each dump. No GPU.
  Part B  Complete pools of the five query-based detectors (auxiliary dumps, see _ext.py),
          which keep the sigmoid probability of every class channel for every query. No GPU.
  Part C  Native full-validation prediction directories written by the upstream repositories'
          own test scripts at their native thresholds. Not released, read from
          paths.UPSTREAM_ROOT. No GPU.
  Part D  Checkpoint probe of GUPNet and MonoCon, the two multi-class checkpoints without a
          native full-validation directory in the paper's runs. GPU, run in each detector's own
          environment (the environment of its adapter). The repository's own decode runs on the
          first val batches and the class histogram is cached for the report pass.
  Part E  Release documents of the upstream repositories (README files and configs).

Run from the repository root:
  python tools/extensions/class_capability.py --probe gupnet     # GUPNet environment, GPU
  python tools/extensions/class_capability.py --probe monocon    # MonoCon environment, GPU
  python tools/extensions/class_capability.py                    # report pass (numpy, pandas)
Parts C to E print a note and are skipped where their inputs are absent.

Layout under paths.UPSTREAM_ROOT (the one adapters/ and difficulty_class_extension.py use):
  M3D-RPN/results_origsid/data                         native val predictions (M3D-RPN)
  MonoFlex/tmp_eval/inference/kitti_train/data         native val predictions (modern env)
  DEVIANT/output/run_221/result_kitti/data             native val predictions
  MonoGround/tmp/inference/kitti_train/val/label_2     native val predictions (modern env)
  MonoDLE/outputs/data                                 native val predictions
  MonoCLUE/outputs/monoclue/outputs/data               native val predictions
  MonoIA/outputs/monoia_car/monoia_car/outputs/data    native val predictions
  GUPNet/{code,data,README.md}, ckpts/gupnet/gupnet_val.pth
  MonoCon/, ckpts/monocon/pretrained/{config.yaml,best.pth}
  DEVIANT/{README.md,experiments/run_1050.yaml,experiments/run_1051.yaml}
  EPro-PnP/EPro-PnP-Det/README.md
"""
import os, re, sys, json, glob, hashlib, argparse, datetime
from collections import Counter
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ext import paths, dump_path, cache_dir, out_path, aux_dump_path, ROOT  # noqa: E402

U = paths.UPSTREAM_ROOT
PANEL = ["M3D-RPN", "MonoDLE", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround", "MonoCon",
         "MonoDETR", "MonoDGP", "MonoCoP", "MonoCLUE", "MonoIA"]
# Supplementary Sec. O: "Six of the twelve released checkpoints emit Pedestrian and Cyclist,
# namely M3D-RPN, MonoFlex, GUPNet, DEVIANT, MonoGround and MonoCon."
PAPER_MULTICLASS = ["M3D-RPN", "MonoFlex", "GUPNet", "DEVIANT", "MonoGround", "MonoCon"]

# Part A: released dump stem -> (detector, adapter file, pattern of its Car-selection line)
DUMPS = [
    ("m3drpn", "M3D-RPN", "m3drpn_dump_accv.py", r"!= 'Car'"),
    ("monodle", "MonoDLE", "monodle_dump.py", r"!= CAR"),
    ("monoflex_orig", "MonoFlex*", "monoflex_dump_orig.py", r"# Car only"),
    ("monoflex", "MonoFlex (modern)", "monoflex_dump.py", r"# Car only"),
    ("gupnet", "GUPNet", "gupnet_dump.py", r"!= 1:"),
    ("deviant", "DEVIANT", "deviant_dump.py", r"!= CAR"),
    ("monoground_orig", "MonoGround*", "monoground_dump_orig.py", r"# Car only"),
    ("monoground", "MonoGround (modern)", "monoground_dump.py", r"# Car only"),
    ("monocon", "MonoCon", "monocon_dump.py", r'!= "Car"'),
    ("monodetr", "MonoDETR", "monodetr_dump.py", r"!= CAR"),
    ("dgp", "MonoDGP", "dgp_cop_dump.py", r"sigmoid\(\)\.max\(-1\)"),
    ("official_monocop", "MonoCoP", "dgp_cop_dump.py", r"sigmoid\(\)\.max\(-1\)"),
    ("monoclue", "MonoCLUE", "monoclue_dump.py", r"!= CAR"),
    ("monoia", "MonoIA", "monoia_dump.py", r"!=CAR"),
]
LABEL_COLS = {"name", "type", "class", "class_name", "label", "category", "cls_name", "class_id"}

# Part B: auxiliary complete pools of the query-based detectors
QUERY = [("MonoDETR", "monodetr"), ("MonoDGP", "monodgp"), ("MonoCoP", "official_monocop"),
         ("MonoCLUE", "monoclue"), ("MonoIA", "monoia")]

# Part C: native full-validation prediction directories (relative to UPSTREAM_ROOT)
NATIVE = [
    ("M3D-RPN", "M3D-RPN/results_origsid/data"),
    ("MonoFlex", "MonoFlex/tmp_eval/inference/kitti_train/data"),
    ("DEVIANT", "DEVIANT/output/run_221/result_kitti/data"),
    ("MonoGround", "MonoGround/tmp/inference/kitti_train/val/label_2"),
    ("MonoDLE", "MonoDLE/outputs/data"),
    ("MonoCLUE", "MonoCLUE/outputs/monoclue/outputs/data"),
    ("MonoIA", "MonoIA/outputs/monoia_car/monoia_car/outputs/data"),
]
NATIVE_NOTE = {"MonoFlex": "modern environment", "MonoGround": "modern environment",
               "DEVIANT": "scores and geometry at 2 decimals"}

# Part E: (topic, file relative to UPSTREAM_ROOT, upstream source, regex patterns; a pattern given
# as (regex, n) prints its first n matching lines, otherwise up to 4)
DOCS = [
    ("GUPNet checkpoint classes", "GUPNet/README.md", "github.com/SuperMHP/GUPNet README.md",
     [r"best multi-category", (r"Pedestrian@IoU=0\.5", 1), (r"Cyclist@IoU=0\.5", 1)]),
    ("GUPNet checkpoint classes", "GUPNet/code/experiments/config.yaml",
     "github.com/SuperMHP/GUPNet code/experiments/config.yaml", [r"writelist"]),
    ("MonoCon checkpoint classes", "MonoCon/utils/kitti_convert_utils.py",
     "github.com/2gunsu/monocon-pytorch utils/kitti_convert_utils.py", [r"^CLASSES\s*="]),
    ("MonoCon checkpoint classes", "ckpts/monocon/pretrained/config.yaml",
     "config.yaml shipped with the monocon-pytorch checkpoint", [r"NUM_CLASSES"]),
    ("Waymo-trained checkpoints (Sec. P)", "DEVIANT/README.md",
     "github.com/abhi1kumar/DEVIANT README.md", [r"^\|\s*Waymo Val"]),
    ("Waymo-trained checkpoints (Sec. P)", "DEVIANT/experiments/run_1050.yaml",
     "github.com/abhi1kumar/DEVIANT experiments/run_1050.yaml",
     [r"^\s*type:\s*'waymo'", r"^\s*train_split_name"]),
    ("Waymo-trained checkpoints (Sec. P)", "DEVIANT/experiments/run_1051.yaml",
     "github.com/abhi1kumar/DEVIANT experiments/run_1051.yaml",
     [r"^\s*type:\s*'waymo'", r"^\s*train_split_name"]),
    ("EPro-PnP-Det (Sec. P)", "EPro-PnP/EPro-PnP-Det/README.md",
     "github.com/tjiiv-cprg/EPro-PnP EPro-PnP-Det/README.md",
     [r"extends the one-stage detector FCOS3D", r"^\|\s*Config\s*\|", r"epropnp_det_basic\]"]),
]

PROBE_CACHE = lambda det: os.path.join(cache_dir("class_capability"), f"probe_{det}.json")  # noqa: E731
CLS3 = ("Car", "Pedestrian", "Cyclist")


def tag(path):
    """dump@ tag of the frozen reports: MD5 of the first 1 MiB, 8 hex."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read(1 << 20)).hexdigest()[:8]


def md5_file(path, n=12):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def sha_file(path, n=12):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:n]


# ============================================================================ Part D probes
class _Log:
    def info(self, *a, **k):
        pass


def probe_gupnet(a):
    """GUPNet's own decode (extract_dets_from_outputs, K=50) on the first val batches.
    Class-id histogram of all top-k peaks and of the peaks at the tester threshold."""
    import time
    code = os.path.abspath(a.gupnet_code)
    sys.path.insert(0, code)
    os.chdir(code)
    import numpy as np, torch, yaml
    from torch.utils.data import DataLoader
    from lib.datasets.kitti import KITTI
    from lib.helpers.model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint
    from lib.helpers.decode_helper import extract_dets_from_outputs
    cfg = yaml.load(open(a.gupnet_cfg), Loader=yaml.Loader)
    cfg["dataset"]["root_dir"] = os.path.abspath(a.gupnet_root)
    thr = float(cfg.get("tester", {}).get("threshold", 0.2))
    ds = KITTI(root_dir=cfg["dataset"]["root_dir"], split="val", cfg=cfg["dataset"])
    ds.data_augmentation = False
    dl = DataLoader(ds, batch_size=16, num_workers=4, shuffle=False)
    model = build_model(cfg["model"], ds.cls_mean_size)
    load_checkpoint(model=model, optimizer=None, filename=a.gupnet_ckpt, logger=_Log(),
                    map_location="cuda:0")
    model.cuda().eval()
    names = list(getattr(ds, "class_name", ["c0", "c1", "c2"]))
    cnt = np.zeros(3, int); cnt_thr = np.zeros(3, int); mx = np.zeros(3); nimg = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, (inputs, calibs, coord_ranges, _, info) in enumerate(dl):
            if bi >= a.nbatch:
                break
            outputs = model(inputs.cuda(), coord_ranges.cuda(), calibs.cuda(), K=50, mode="test")
            dets = extract_dets_from_outputs(outputs=outputs, K=50).cpu().numpy()
            nimg += dets.shape[0]
            cid = dets[:, :, 0].astype(int).ravel(); sc = dets[:, :, 1].ravel()
            for c in range(3):
                m = cid == c
                cnt[c] += int(m.sum()); cnt_thr[c] += int((sc[m] >= thr).sum())
                if m.any():
                    mx[c] = max(mx[c], float(sc[m].max()))
    res = dict(detector="GUPNet", images=nimg, topk=50, threshold=thr,
               writelist=cfg["dataset"].get("writelist"),
               counts_all={names[c]: int(cnt[c]) for c in range(3)},
               counts_thr={names[c]: int(cnt_thr[c]) for c in range(3)},
               max_score={names[c]: round(float(mx[c]), 6) for c in range(3)},
               ckpt_md5=md5_file(a.gupnet_ckpt), torch=torch.__version__,
               seconds=round(time.time() - t0))
    return res


def probe_monocon(a):
    """MonoCon's own evaluation formatter (_get_eval_formats, top-k 30) on the first val
    batches. Class-name histogram at the native test threshold and with no threshold."""
    import time
    repo = os.path.abspath(a.monocon_repo)
    sys.path.insert(0, repo)
    os.chdir(repo)
    import torch
    from engine.monocon_engine import MonoconEngine
    from utils.engine_utils import load_cfg, move_data_device
    cfg = load_cfg(os.path.abspath(a.monocon_cfg))
    cfg.DATA.ROOT = os.path.abspath(a.kitti_root)
    cfg.GPU_ID = 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine = MonoconEngine(cfg, auto_resume=False, is_test=True)
    engine.load_checkpoint(os.path.abspath(a.monocon_ckpt), verbose=False)
    model = engine.model
    model.eval()
    thr = float(model.head.test_thres)
    dev = engine.current_device
    c_thr = Counter(); c_all = Counter(); nimg = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, td in enumerate(engine.test_loader):
            if bi >= a.nbatch:
                break
            td = move_data_device(td, dev)
            for an in model.head._get_eval_formats(td, model(td, return_loss=False))["img_bbox"]:
                nimg += 1
                c_thr.update(list(an["name"]))
            model.head.test_thres = -1e9
            for an in model.head._get_eval_formats(td, model(td, return_loss=False))["img_bbox"]:
                c_all.update(list(an["name"]))
            model.head.test_thres = thr
    res = dict(detector="MonoCon", images=nimg, topk=int(model.head.topk), threshold=thr,
               num_classes=int(cfg.MODEL.HEAD.NUM_CLASSES),
               counts_all={k: int(c_all.get(k, 0)) for k in CLS3},
               counts_thr={k: int(c_thr.get(k, 0)) for k in CLS3},
               ckpt_md5=md5_file(a.monocon_ckpt), torch=torch.__version__,
               seconds=round(time.time() - t0))
    return res


# ============================================================================ report pass
def report():
    import numpy as np, pandas as pd
    OUT = out_path("extensions/class_capability.txt")
    out = []

    def w(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        out.append(s)

    rule = "=" * 100
    nval = len(open(paths.VAL_LIST).read().split())
    w(f"# class_capability  {datetime.datetime.now().isoformat(timespec='seconds')}")
    w("# script  tools/extensions/class_capability.py")
    w(f"# KITTI val (ImageSets/val.txt) n_img={nval}")
    w("# Question: which of the twelve released checkpoints emit Pedestrian and Cyclist, and which")
    w("# classes the full-validation outputs used by the paper contain (supplementary Sec. O,")
    w("# main Sec. 5.1). Part E also checks three release-document statements of Sec. P.")
    w("")
    ev = {d: [] for d in PANEL}     # detector -> list of (part, verdict yes/no, text)

    # ------------------------------------------------------------------ Part A
    w(rule)
    w("PART A -- released per-prediction dumps (data/dumps)")
    w(rule)
    w("A dump row is one candidate box. The 23-column schema (data/DUMPS.md) has no class-label")
    w("column, the column `cls` is a score, and each adapter keeps Car candidates only.")
    w("")
    w(f"{'detector':20s} {'file':27s} {'dump@':8s} {'rows':>8s} {'imgs':>5s} {'rows/img':>9s}"
      f"  label col  adapter Car selection")
    a_ok = True
    for stem, det, adapter, pat in DUMPS:
        p = dump_path(stem)
        cols = list(pd.read_csv(p, nrows=0).columns)
        lab = sorted(set(c.lower() for c in cols) & LABEL_COLS)
        df = pd.read_csv(p, usecols=["sid", "cls"])
        per = df.groupby("sid").size()
        src = os.path.join(ROOT, "adapters", adapter)
        hit = "NOT FOUND"
        for i, line in enumerate(open(src), 1):
            if re.search(pat, line):
                hit = f"adapters/{adapter}:{i}  {line.strip()[:80]}"
                break
        w(f"{det:20s} {os.path.basename(p):27s} {tag(p):8s} {len(df):8d} {per.size:5d} "
          f"{per.min():4d}-{per.max():<4d}  {'none' if not lab else ','.join(lab):9s}  {hit}")
        a_ok = a_ok and not lab and hit != "NOT FOUND"
        if stem == "monodle" and per.size == nval and per.min() == per.max() == 50:
            ev["MonoDLE"].append(("A", "no", "all 50 top-k slots of every val image are Car peaks "
                                  f"({len(df)} rows = {nval} x 50, adapter keeps Car, no score cut)"))
    w("dgp_cop_dump.py keeps every query, with its maximum class probability as `cls`, and decodes")
    w("it with the Car mean size. Part B shows that the MonoDGP and MonoCoP checkpoints are Car-only.")
    w("MonoDLE: the adapter writes the K=50 heatmap top-k with no score threshold and drops non-Car")
    w("peaks, so 50 rows in every image means no Pedestrian or Cyclist peak in any val top-50.")
    if a_ok:
        w("RESULT A: every released dump holds Car candidates only, including the full-validation")
        w("dumps of GUPNet (gupnet_val.csv) and MonoCon (monocon_val.csv).")
    else:
        w("RESULT A: CHECK FAILED (a class-label column or an adapter Car-selection line, see above)")
    w("")

    # ------------------------------------------------------------------ Part B
    w(rule)
    w("PART B -- complete pools of the five query-based detectors (auxiliary dumps)")
    w(rule)
    w("Every query of every val image with the sigmoid probability of each class channel.")
    w("A query counts once even where the pool stores one row per query and class.")
    w("")
    w(f"{'detector':9s} {'file':37s} {'dump@':8s} {'rows':>7s} {'queries':>7s} car_ch "
      f"{'max p(Car)':>10s} {'max p(other1)':>13s} {'max p(other2)':>13s} {'q: other>=0.05':>14s}")
    b_other, b_car = [], []
    for det, f in QUERY:
        p = aux_dump_path(f"{f}_val_preflatten.csv")
        if not os.path.exists(p):
            w(f"{det:9s} {os.path.basename(p)}  absent (see tools/extensions/_ext.py), skipped")
            continue
        d = pd.read_csv(p, usecols=["sid", "query_id", "cls_c0", "cls_c1", "cls_c2", "car_channel"])
        q = d.drop_duplicates(["sid", "query_id"])
        car = int(q.car_channel.iloc[0])
        assert (q.car_channel == car).all()
        P = q[["cls_c0", "cls_c1", "cls_c2"]].to_numpy()
        oth = [c for c in range(3) if c != car]
        mo = P[:, oth].max(axis=0)
        n05 = int((P[:, oth].max(axis=1) >= 0.05).sum())
        w(f"{det:9s} {os.path.basename(p):37s} {tag(p):8s} {len(d):7d} {len(q):7d} {car:6d} "
          f"{P[:, car].max():10.6f} {mo[0]:13.6f} {mo[1]:13.6f} {n05:14d}")
        b_other.append(float(mo.max())); b_car.append(float(P[:, car].max()))
        if mo.max() < 0.01:
            ev[det].append(("B", "no", f"non-Car channels never exceed {mo.max():.6f} on any of "
                            f"{len(q)} val queries"))
    if b_other:
        w(f"RESULT B: over {len(b_other)} checkpoints the largest non-Car probability of any val query is "
          f"{max(b_other):.6f}, while the Car")
        w(f"channel reaches at least {min(b_car):.3f} in each. "
          + ("These are Car-only checkpoints." if max(b_other) < 0.01 else "CHECK the non-Car channels."))
    w("")

    # ------------------------------------------------------------------ Part C
    w(rule)
    w("PART C -- native full-validation prediction directories (upstream test scripts)")
    w(rule)
    w("Written by each repository's own test script at its native threshold during the paper's")
    w("runs. Not released. Class histogram of the first token of every non-empty line, and a")
    w("content hash (MD5 of the files concatenated in name order, 12 hex).")
    w("")
    ref = {}
    rp = os.path.join(ROOT, "reports", "extensions", "difficulty_class_extension.txt")
    if os.path.exists(rp):
        for line in open(rp):
            m = re.match(r"\[B\]\s+(\S+)\s+native class histogram:\s+(.*?)\s+\(total", line)
            if m:
                ref[m.group(1)] = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", m.group(2)))
    have_native = False
    for det, rel in NATIVE:
        d = os.path.join(U, rel)
        if not os.path.isdir(d):
            w(f"{det:11s} <UPSTREAM_ROOT>/{rel}  absent, skipped")
            continue
        have_native = True
        files = sorted(glob.glob(os.path.join(d, "*.txt")))
        h = hashlib.md5(); c = Counter()
        for fp in files:
            b = open(fp, "rb").read()
            h.update(b)
            for line in b.decode().splitlines():
                if line.strip():
                    c[line.split()[0]] += 1
        hist = "  ".join(f"{k}={c.get(k, 0)}" for k in CLS3)
        extra = {k: v for k, v in c.items() if k not in CLS3}
        chk = ""
        if det in ref:
            chk = "  [= difficulty_class_extension.txt Part B]" if all(
                ref[det].get(k, 0) == c.get(k, 0) for k in CLS3) else "  [DIFFERS from difficulty_class_extension.txt]"
        note = f" ({NATIVE_NOTE[det]})" if det in NATIVE_NOTE else ""
        w(f"{det:11s} <UPSTREAM_ROOT>/{rel}  files={len(files)}  md5={h.hexdigest()[:12]}{note}")
        w(f"{'':11s} {hist}{'  other=' + str(extra) if extra else ''}{chk}")
        if len(files) == nval:
            if c.get("Pedestrian", 0) > 0 and c.get("Cyclist", 0) > 0:
                ev[det].append(("C", "yes", f"native full-val output holds {c['Pedestrian']} "
                                f"Pedestrian and {c['Cyclist']} Cyclist boxes{note}"))
            elif c.get("Pedestrian", 0) == 0 and c.get("Cyclist", 0) == 0:
                ev[det].append(("C", "no", f"native full-val output holds Car only ({c.get('Car', 0)} boxes)"))
    if not have_native:
        w("(no native directory found under paths.UPSTREAM_ROOT; Part C skipped)")
    w("No native full-validation directory of GUPNet or MonoCon is kept from the paper's runs.")
    w("Their full-validation outputs are the released dumps of Part A, which hold Car only.")
    w("")

    # ------------------------------------------------------------------ Part D
    w(rule)
    w("PART D -- checkpoint probe of GUPNet and MonoCon (repository's own decode, GPU)")
    w(rule)
    for det, key in (("GUPNet", "gupnet"), ("MonoCon", "monocon")):
        cp = PROBE_CACHE(key)
        if not os.path.exists(cp):
            w(f"{det}: probe cache absent. Run  python tools/extensions/class_capability.py "
              f"--probe {key}  in the {det} environment.")
            continue
        r = json.load(open(cp))
        w(f"{det}: released checkpoint md5 {r['ckpt_md5']}, first {r['images']} val images, "
          f"top-k {r['topk']}, native threshold {r['threshold']}, torch {r['torch']}")
        if det == "GUPNet":
            w(f"   config writelist {r['writelist']}")
            for k in r["counts_all"]:
                w(f"   {k:11s} top-k peaks {r['counts_all'][k]:6d}   at >= {r['threshold']}: "
                  f"{r['counts_thr'][k]:5d}   max score {r['max_score'][k]:.6f}")
        else:
            w(f"   head NUM_CLASSES {r['num_classes']}")
            w(f"   at the native threshold {r['threshold']}: " +
              "  ".join(f"{k}={r['counts_thr'][k]}" for k in CLS3))
            w("   with no threshold (all top-k):  " +
              "  ".join(f"{k}={r['counts_all'][k]}" for k in CLS3))
        ped, cyc = r["counts_thr"].get("Pedestrian", 0), r["counts_thr"].get("Cyclist", 0)
        ev[det].append(("D", "yes" if ped > 0 and cyc > 0 else "no",
                        f"{ped} Pedestrian and {cyc} Cyclist detections above the native threshold "
                        f"on {r['images']} val images"))
    w("")

    # ------------------------------------------------------------------ Part E
    w(rule)
    w("PART E -- release documents of the upstream repositories")
    w(rule)
    for topic, rel, src, pats in DOCS:
        p = os.path.join(U, rel)
        if not os.path.exists(p):
            w(f"[{topic}] <UPSTREAM_ROOT>/{rel}  absent, skipped")
            continue
        lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        w(f"[{topic}] {src}  (sha256 {sha_file(p)})")
        for pat in pats:
            pat, nmax = pat if isinstance(pat, tuple) else (pat, 4)
            hits = [(i, l) for i, l in enumerate(lines, 1) if re.search(pat, l)]
            for i, l in hits[:nmax]:
                w(f"   L{i:<4d} {re.sub(r'<[^>]+>', ' ', l).strip()[:88]}")
            if not hits:
                w(f"   pattern {pat!r} NOT FOUND")
        if rel.endswith("EPro-PnP-Det/README.md"):
            heads = [l for l in lines if re.match(r"^\|\s*Config\s*\|", l)]
            metr = set()
            for hl in heads:
                cells = [x.strip() for x in hl.strip().strip("|").split("|")]
                metr.update(cells[cells.index("TTA") + 1:] if "TTA" in cells else cells[2:])
            n_map = sum(len(re.findall(r"\bmAP\b", l)) for l in lines)
            w(f"   model tables: {len(heads)}, metric columns: {sorted(metr)}, "
              f"occurrences of 'mAP' in the README: {n_map}")
    w("")

    # ------------------------------------------------------------------ summary
    w(rule)
    w("SUMMARY -- twelve panel checkpoints")
    w(rule)
    w(f"{'detector':11s} {'emits Ped+Cyc':13s} {'full-val outputs of the paper':46s} evidence")
    yes = []
    for det in PANEL:
        vs = {v for _, v, _ in ev[det]}
        verdict = "CONFLICT" if len(vs) > 1 else (vs.pop() if vs else "no evidence")
        if verdict == "yes":
            yes.append(det)
        nat = [t for part, _, t in ev[det] if part == "C"]
        fv = "released dump: Car only"
        if det in ("GUPNet", "MonoCon"):
            fv += "; no native dir"
        elif any(det == n for n, _ in NATIVE):
            fv += "; native dir: " + ("Car, Ped, Cyc" if any("Pedestrian and" in t for t in nat)
                                      else "Car only" if nat else "absent")
        w(f"{det:11s} {verdict:13s} {fv:46s} " + "; ".join(f"{p}" for p, _, _ in ev[det]))
    w("")
    for det in PANEL:
        for part, v, t in ev[det]:
            w(f"  {det:11s} [{part}] {v:3s} {t}")
    w("")
    w(f"Checkpoints that emit Pedestrian and Cyclist: {len(yes)} of {len(PANEL)}: {', '.join(yes)}")
    missing = [d for d in PANEL if not ev[d]]
    state = ("MATCH" if yes == PAPER_MULTICLASS else "DIFFERS") if not missing else \
        f"INCOMPLETE (no evidence for {', '.join(missing)}; inputs of Parts C or D absent)"
    w(f"Supplementary Sec. O list ({', '.join(PAPER_MULTICLASS)}): {state}")
    w("GUPNet and MonoCon full-validation dumps: Car only (Part A). Sec. O and main Sec. 5.1 use "
      "the multi-class native directories of Part C for M3D-RPN, MonoFlex and MonoGround.")
    w(f"[written] {os.path.relpath(OUT, ROOT)}")
    with open(OUT, "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", choices=["gupnet", "monocon"], default=None,
                    help="run one checkpoint probe (GPU, that detector's environment) and cache it")
    ap.add_argument("--nbatch", type=int, default=40,
                    help="val batches per probe (GUPNet 16 images, MonoCon config batch size)")
    ap.add_argument("--gupnet_code", default=os.path.join(U, "GUPNet", "code"))
    ap.add_argument("--gupnet_cfg", default=None, help="default <gupnet_code>/experiments/config.yaml")
    ap.add_argument("--gupnet_root", default=None, help="GUPNet data root, default <gupnet_code>/../data")
    ap.add_argument("--gupnet_ckpt", default=os.path.join(U, "ckpts", "gupnet", "gupnet_val.pth"))
    ap.add_argument("--monocon_repo", default=os.path.join(U, "MonoCon"))
    ap.add_argument("--monocon_cfg", default=os.path.join(U, "ckpts", "monocon", "pretrained", "config.yaml"))
    ap.add_argument("--monocon_ckpt", default=os.path.join(U, "ckpts", "monocon", "pretrained", "best.pth"))
    ap.add_argument("--kitti_root", default=paths.KITTI_ROOT)
    a = ap.parse_args()
    if a.probe is None:
        report()
        return
    a.gupnet_cfg = a.gupnet_cfg or os.path.join(a.gupnet_code, "experiments", "config.yaml")
    a.gupnet_root = a.gupnet_root or os.path.join(os.path.dirname(os.path.abspath(a.gupnet_code)), "data")
    for k in ("gupnet_code", "gupnet_cfg", "gupnet_root", "gupnet_ckpt", "monocon_repo",
              "monocon_cfg", "monocon_ckpt", "kitti_root"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    cp = PROBE_CACHE(a.probe)
    res = probe_gupnet(a) if a.probe == "gupnet" else probe_monocon(a)
    with open(cp, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print(f"[cached] {cp}")


if __name__ == "__main__":
    main()

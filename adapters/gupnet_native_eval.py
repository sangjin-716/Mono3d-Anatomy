"""Faithfulness cross-check: run GUPNet's OWN tester (native decode + threshold 0.2)
to write its predictions, then evaluate them with OUR python do_eval (the official KITTI
evaluator vendored in evaluator/kitti_eval). GUPNet's own output scores 16.23, the value its
README lists for the released checkpoint. The dump at the same threshold holds the same boxes
and scores 16.46, because it keeps full precision while GUPNet's writer rounds every field,
the score included, to two decimals (see docs/ERRATA.md).

It writes no report; the numbers are printed (gate G1/G2 cross-check for GUPNet).
Run in a GUPNet env (torch 1.9 in our runs): python adapters/gupnet_native_eval.py
"""
import os, sys, shutil, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from tools._release import paths, cache_dir  # noqa: E402
sys.path.remove(_ROOT); sys.path.append(_ROOT)   # upstream repo packages take precedence
_ap = argparse.ArgumentParser()
_ap.add_argument("--code", default=os.path.join(paths.UPSTREAM_ROOT, "GUPNet", "code"))
_ap.add_argument("--ckpt", default=os.path.join(paths.UPSTREAM_ROOT, "ckpts", "gupnet", "gupnet_val.pth"))
_ap.add_argument("--root", default=None, help="GUPNet data root (<root>/KITTI/...); default <code>/../data")
_args = _ap.parse_args()
GUP = os.path.abspath(_args.code)
sys.path.insert(0, GUP)
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from lib.datasets.kitti import KITTI
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.tester_helper import Tester

ROOT = os.path.abspath(_args.root) if _args.root else os.path.join(os.path.dirname(GUP), "data")
CKPT = os.path.abspath(_args.ckpt)
WORK = cache_dir("_gupnet_native")
LABEL_DIR = paths.LABEL_DIR
VAL_LIST = paths.VAL_LIST


class _Log:
    def info(self, *a, **k): pass


def main():
    cfg = yaml.safe_load(open(os.path.join(GUP, "experiments/config.yaml")))
    cfg['dataset']['root_dir'] = ROOT
    cfg['tester']['threshold'] = 0.2          # GUPNet native threshold
    cfg['tester']['resume_model'] = CKPT
    device = torch.device("cuda")
    val = KITTI(root_dir=ROOT, split='val', cfg=dict(cfg['dataset'])); val.data_augmentation = False
    cms = val.cls_mean_size
    model = build_model(cfg['model'], cms)
    loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=4, drop_last=False)
    os.makedirs(WORK, exist_ok=True); os.chdir(WORK)     # Tester writes ./outputs/data
    Tester(cfg['tester'], model, loader, _Log()).test()
    pred_dir = os.path.join(WORK, "outputs", "data")
    n = len([f for f in os.listdir(pred_dir) if f.endswith('.txt')])
    print(f"[native] GUPNet wrote {n} prediction files", flush=True)

    # evaluate GUPNet's own predictions with OUR python do_eval
    import evaluator.kitti_eval.kitti_common as kc
    from evaluator.kitti_eval.eval import do_eval
    val_ids = [int(x) for x in open(VAL_LIST).read().split()]
    gt = kc.get_label_annos(LABEL_DIR, val_ids)
    dt = kc.get_label_annos(pred_dir, val_ids)
    ov07 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7]] * 3)
    ov05 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                     [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    min_ov = np.stack([ov07, ov05], 0)[:, :, [0]]
    r = do_eval(gt, dt, [0], min_ov, compute_aos=False, DIForDIS=True)
    m3 = r[6]
    print(f"[native] GUPNet-own predictions via OUR do_eval: Car 3d mod@.7 = {m3[0,1,0]:.2f} "
          f"(easy {m3[0,0,0]:.2f}, hard {m3[0,2,0]:.2f})", flush=True)
    print(f"  compare: our dump S2 mod@.7 = 16.46 ; README(C++ ap40) = 16.23", flush=True)
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()

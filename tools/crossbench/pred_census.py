"""Ported from camera-ready checks/xdsfix_pgd_oracle/pred_census.py for the public release.
Computation unchanged. Produces the car-prediction census and per-depth-bin coverage of a converted oracle pkl
(console); in the paper it gives the PGD and EPro-PnP-Det boxes per image (pred_car per image).

pred_census.py — car-class prediction census on a converted oracle pkl, plus
coverage (bipartite M) broken down by GT camera-frame depth bin.
Usage: python pred_census.py <pkl> <tag>
"""
import pickle, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools._release import paths
sys.path.insert(0, os.path.join(paths.UPSTREAM_ROOT, 'mmdetection3d'))   # optional source checkout
os.environ.setdefault('PYTHONPATH', os.path.join(paths.UPSTREAM_ROOT, 'mmdetection3d'))
from mmdet3d.structures import CameraInstance3DBoxes
from scipy.optimize import linear_sum_assignment

CAR = 0
BINS = [(0, 30), (30, 50), (50, 1e9)]


def _cpu(b):
    return CameraInstance3DBoxes(b.tensor.detach().cpu(), box_dim=b.tensor.shape[-1],
                                 origin=(0.5, 0.5, 0.5)) if hasattr(b, 'tensor') else b


def main():
    path, tag = sys.argv[1], sys.argv[2]
    data = pickle.load(open(path, 'rb'))
    n_img = len(data)
    n_pred_all = 0
    zc = np.zeros(len(BINS), dtype=np.int64)
    n_pred_car = 0
    for s in data:
        p = s['pred_instances_3d']
        pl = np.asarray(p['labels_3d'])
        n_pred_all += len(pl)
        m = pl == CAR
        n_pred_car += int(m.sum())
        if m.sum():
            z = p['bboxes_3d'].tensor.cpu().numpy()[m][:, 2]
            for i, (a, b) in enumerate(BINS):
                zc[i] += int(((z >= a) & (z < b)).sum())
    print(f'[census:{tag}] imgs={n_img} pred_all={n_pred_all} '
          f'({n_pred_all/n_img:.1f}/img) pred_car={n_pred_car} ({n_pred_car/n_img:.1f}/img)',
          flush=True)
    for i, (a, b) in enumerate(BINS):
        print(f'    car preds z[{a},{b}) : {zc[i]} ({100.0*zc[i]/max(n_pred_car,1):.1f}%)', flush=True)

    for thr in (0.7, 0.5):
        gtz_tot = np.zeros(len(BINS), dtype=np.int64)
        gtz_cov = np.zeros(len(BINS), dtype=np.int64)
        for s in data:
            p = s['pred_instances_3d']; g = s['eval_ann_info']
            pl = np.asarray(p['labels_3d']); gl = np.asarray(g['gt_labels_3d'])
            gm = gl == CAR
            if gm.sum() == 0:
                continue
            gb = g['gt_bboxes_3d'][gm]
            gz = (gb.tensor.cpu().numpy() if hasattr(gb, 'tensor') else np.asarray(gb))[:, 2]
            bidx = np.zeros(len(gz), dtype=np.int64)
            for i, (a, b) in enumerate(BINS):
                bidx[(gz >= a) & (gz < b)] = i
            for i in range(len(BINS)):
                gtz_tot[i] += int((bidx == i).sum())
            pm = pl == CAR
            if pm.sum() == 0:
                continue
            iou = CameraInstance3DBoxes.overlaps(_cpu(p['bboxes_3d'][pm]), _cpu(gb), mode='iou').numpy()
            edge = (iou >= thr).astype(np.float64)
            if edge.size == 0:
                continue
            r, c = linear_sum_assignment(edge, maximize=True)
            for ri, ci in zip(r, c):
                if edge[ri, ci] > 0:
                    gtz_cov[bidx[ci]] += 1
        print(f'[cover:{tag}] IoU{thr}: M={gtz_cov.sum()}/{gtz_tot.sum()}', flush=True)
        for i, (a, b) in enumerate(BINS):
            print(f'    gt z[{a},{b}) : M={gtz_cov[i]}/{gtz_tot[i]} = '
                  f'{100.0*gtz_cov[i]/max(gtz_tot[i],1):.1f}%', flush=True)


if __name__ == '__main__':
    main()

# Ported from mono3d_crossdataset/configs/pgd_dump_full.py for the public release.
# Official mmdet3d PGD 2x finetune config with only the evaluator replaced by DumpResults, so
# inference writes the per-camera prediction pkl used by reformat_official.py and the oracle.
# Edit the two placeholders: your mmdetection3d checkout and the cache directory of this
# repository (paths.CACHE_DIR/crossbench/nusc). Run from the mmdetection3d root:
#   python tools/test.py <this file> <checkpoint>
# checkpoint: pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune_20211114_162135-5ec7c1cd.pth
_base_ = '/path/to/mmdetection3d/configs/pgd/pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d_finetune.py'
test_evaluator = dict(_delete_=True, type='DumpResults',
                      out_file_path='/path/to/cache/crossbench/nusc/preds_pgd_full.pkl')
val_evaluator = test_evaluator

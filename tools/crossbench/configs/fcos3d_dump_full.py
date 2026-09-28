# Ported from mono3d_crossdataset/configs/fcos3d_dump_full.py for the public release.
# Official mmdet3d FCOS3D finetune config with only the evaluator replaced by DumpResults, so
# inference writes the per-camera prediction pkl used by nusc_official_reformat.py and the oracle.
# Edit the two placeholders: your mmdetection3d checkout and the cache directory of this
# repository (paths.CACHE_DIR/crossbench/nusc). Run from the mmdetection3d root:
#   python tools/test.py <this file> <checkpoint>
# checkpoint: fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
_base_ = '/path/to/mmdetection3d/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py'
test_evaluator = dict(_delete_=True, type='DumpResults',
                      out_file_path='/path/to/cache/crossbench/nusc/preds_fcos3d_full.pkl')
val_evaluator = test_evaluator

"""Patched MonoFlex paths catalog: DATA_DIR points at a local KITTI copy in the MonoFlex layout
(<DATA_DIR>/training/{image_2,calib,label_2,ImageSets}). Used by setting cfg.PATHS_CATALOG to this
file (the repo's config/paths_catalog.py stays untouched).

DATA_DIR is read from the environment variable MONOFLEX_KITTI_DIR, which
adapters/monoflex_dump.py and adapters/monoflex_dump_orig.py set from their --kitti_dir argument.
"""
import os


class DatasetCatalog():
    DATA_DIR = os.path.join(os.environ.get("MONOFLEX_KITTI_DIR", "kitti_monoflex"), "")
    DATASETS = {
        "kitti_train": {"root": "training/"},
        "kitti_test": {"root": "testing/"},
    }

    @staticmethod
    def get(name):
        if "kitti" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(root=os.path.join(data_dir, attrs["root"]))
            return dict(factory="KITTIDataset", args=args)
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog():
    IMAGENET_MODELS = {
        "DLA34": "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"
    }

    @staticmethod
    def get(name):
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_imagenet_pretrained(name)

    @staticmethod
    def get_imagenet_pretrained(name):
        name = name[len("ImageNetPretrained/"):]
        return ModelCatalog.IMAGENET_MODELS[name]

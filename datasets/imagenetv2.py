import os

from dassl.data.datasets import DATASET_REGISTRY, DatumMT, DatasetBaseMT
from dassl.utils import listdir_nohidden

from .imagenet import ImageNet


@DATASET_REGISTRY.register()
class ImageNetV2(DatasetBaseMT):
    """ImageNetV2.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenetv2"
    taskid2type = ['sl']
    taskid2metric = ["overall_acc"]

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        image_dir = "imagenetv2-matched-frequency-format-val"
        self.image_dir = os.path.join(self.dataset_dir, image_dir)
        self.taskdesc_file = os.path.join(self.dataset_dir, "task_desc.txt")
        self.taskname_file = os.path.join(self.dataset_dir, "task_name.txt")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)
        
        taskid2desc = [open(self.taskdesc_file, 'r').read().strip()]
        taskid2tname = [open(self.taskname_file, 'r').read().strip()]
        taskid2cname = classnames
        
        super().__init__(taskid2tname, taskid2cname, taskid2desc, self.taskid2type, self.taskid2metric, train_x=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = list(classnames.keys())
        items = []

        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            imnames = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                item = DatumMT(impath=impath, label=[label], task=[0])
                items.append(item)

        return items

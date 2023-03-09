import os

from dassl.data.datasets import DATASET_REGISTRY, DatumMT, DatasetBaseMT
from dassl.utils import listdir_nohidden

from .imagenet import ImageNet

TO_BE_IGNORED = ["README.txt"]


@DATASET_REGISTRY.register()
class ImageNetA(DatasetBaseMT):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-adversarial"
    taskid2type = ['sl']
    taskid2metric = ["overall_acc"]

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "imagenet-a")
        self.taskdesc_file = os.path.join(self.dataset_dir, "task_desc.txt")
        self.taskname_file = os.path.join(self.dataset_dir, "task_name.txt")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)
        
        data = self.read_data(classnames)
        
        taskid2desc = [open(self.taskdesc_file, 'r').read().strip()]
        taskid2tname = [open(self.taskname_file, 'r').read().strip()]
        taskid2cname = [classnames]

        super().__init__(taskid2tname, taskid2cname, taskid2desc, self.taskid2type, self.taskid2metric, 
                         train_x=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = DatumMT(impath=impath, label=[label], task=[0])
                items.append(item)

        return items

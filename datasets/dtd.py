import os
import pickle
import random

from dassl.data.datasets import DATASET_REGISTRY, DatumMT, DatasetBaseMT
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class DescribableTextures(DatasetBaseMT):

    dataset_dir = "dtd"
    taskid2type = ['sl']
    taskid2metric = ["overall_acc"]

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_dk_DescribableTextures.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_mt")
        self.taskdesc_file = os.path.join(self.dataset_dir, "task_desc.txt")
        self.taskname_file = os.path.join(self.dataset_dir, "task_name.txt")
        mkdir_if_missing(self.split_fewshot_dir)
        
        if os.path.exists(self.split_path):
            train, val, test, meta = OxfordPets.read_split(self.split_path, self.image_dir)
            
            taskid2cname = meta["taskid2cname"]
            taskid2desc = meta["taskid2desc"]
            taskid2tname = meta["taskid2tname"]
        else:
            train, val, test, taskid2cname = self.read_and_split_data(self.image_dir)
            
            taskid2desc = [open(self.taskdesc_file, 'r').read().strip()]
            taskid2tname = [open(self.taskname_file, 'r').read().strip()]
            
            meta = {
                "taskid2cname": taskid2cname,
                "taskid2desc": taskid2desc,
                "taskid2type": self.taskid2type,
                "taskid2tname": taskid2tname,
                "taskid2metric": self.taskid2metric
            }
            
            OxfordPets.save_split(train, val, test, meta, self.split_path, self.image_dir)
        
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test, taskid2cname = OxfordPets.subsample_classes(train, val, test, taskid2cname, subsample=subsample)
        
        super().__init__(taskid2tname, taskid2cname, taskid2desc, self.taskid2type, self.taskid2metric, 
                         train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=[]):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     dog/
        #     cat/
        #     horse/
        # =============
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()
        
        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = DatumMT(impath=im, label=[y], task=[0])  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train : n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val :], label, category))

        return train, val, test, [categories]
    
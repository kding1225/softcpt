import os
import pickle
import math
import random
from collections import defaultdict, OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, DatumMT, DatasetBaseMT
from dassl.utils import read_json, write_json, mkdir_if_missing


@DATASET_REGISTRY.register()
class OxfordPets(DatasetBaseMT):

    dataset_dir = "oxford_pets"
    taskid2type = ['sl']
    taskid2metric = ["mean_acc"]

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_dk_OxfordPets.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_mt")
        self.taskdesc_file = os.path.join(self.dataset_dir, "task_desc.txt")
        self.taskname_file = os.path.join(self.dataset_dir, "task_name.txt")
        mkdir_if_missing(self.split_fewshot_dir)
        
        if os.path.exists(self.split_path):
            train, val, test, meta = self.read_split(self.split_path, self.image_dir)
            
            taskid2cname = meta["taskid2cname"]
            taskid2desc = meta["taskid2desc"]
            taskid2tname = meta["taskid2tname"]
        else:
            trainval, taskid2cname = self.read_data(split_file="trainval.txt")
            test, _ = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            
            taskid2desc = [open(self.taskdesc_file, 'r').read().strip()]
            taskid2tname = [open(self.taskname_file, 'r').read().strip()]
            
            meta = {
                "taskid2cname": taskid2cname,
                "taskid2desc": taskid2desc,
                "taskid2type": self.taskid2type,
                "taskid2tname": taskid2tname,
                "taskid2metric": self.taskid2metric
            }
            
            self.save_split(train, val, test, meta, self.split_path, self.image_dir)
        
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
        train, val, test, taskid2cname = self.subsample_classes(train, val, test, taskid2cname, subsample=subsample)
        
        super().__init__(taskid2tname, taskid2cname, taskid2desc, self.taskid2type, self.taskid2metric, 
                         train_x=train, val=val, test=test)

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []
        M = {}

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = DatumMT(impath=impath, label=[label], task=[0])
                items.append(item)
                M[label] = breed
        
        keys = list(M.keys())
        keys.sort()
        
        return items, [[M[k] for k in keys]]

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = OrderedDict()
        for idx, item in enumerate(trainval):
            for t, l in zip(item.task, item.label):
                if (t, l) in tracker:
                    tracker[(t, l)].append(idx)
                else:
                    tracker[(t, l)] = [idx]
        
        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, meta, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                task = item.task
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, task))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test, "meta": meta}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, task in items:
                impath = os.path.join(path_prefix, impath)
                item = DatumMT(impath=impath, label=label, task=task)
                out.append(item)
            return out
        
        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])
        meta = split["meta"]

        return train, val, test, meta
    
    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]
        
        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label[0])
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args[:-1]:
            dataset_new = []
            for item in dataset:
                ls = []
                for i in item.label:
                    if i in selected:
                        ls.append(relabeler[i])
                if len(ls):
                    item.label = ls
                    dataset_new.append(item)
            output.append(dataset_new)
        
        taskid2cname = [[args[-1][0][i] for i in selected]]
        
        return output+[taskid2cname]

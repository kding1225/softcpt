import os
import pickle
import random
import numpy as np

from dassl.data.datasets import DATASET_REGISTRY, DatumMT, DatasetBaseMT
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets

@DATASET_REGISTRY.register()
class Fashion20(DatasetBaseMT):

    dataset_dir = "fashion20"
    taskid2type = ['sl']*20
    taskid2metric = ["mean_acc"]*20

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_dk_Fashion20.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_mt")
        self.taskinfo_file = os.path.join(self.dataset_dir, "task_info.csv")  # task_name\ttask_desc\tlabel_name, label_name is sep by comma
        mkdir_if_missing(self.split_fewshot_dir)

        task_info = []
        with open(self.taskinfo_file, 'r') as f:
            for idx, line in enumerate(f):
                if idx==0: continue
                _, task_name, _, label_name = line.strip().split("\t")
                task_name = task_name.strip().replace(" ", "_")
                label_name = label_name.strip().split(",")
                label_name = [x.strip() for x in label_name]
                task_desc = [f"cassify {task_name} into {len(label)} categories" for label in label_name]
                task_info.append({
                    "task_name": task_name,
                    "task_desc": task_desc,
                    "label_name": label_name
                })
        self.task_info = task_info

        if os.path.exists(self.split_path):
            train, val, test, _ = OxfordPets.read_split(self.split_path, self.image_dir)
            taskid2cname = [item["label_name"] for item in task_info]
            taskid2desc = [item["task_desc"] for item in task_info]
            taskid2tname = [item["task_name"] for item in task_info]
        else:
            train, val, test, _ = self.read_and_split_data(self.image_dir, self.task_info)
            taskid2cname = [item["label_name"] for item in task_info]
            taskid2desc = [item["task_desc"] for item in task_info]
            taskid2tname = [item["task_name"] for item in task_info]
            
            assert len(self.taskid2type) == len(taskid2desc)
            assert len(self.taskid2metric) == len(taskid2desc)

            meta = {
                "taskid2cname": taskid2cname,
                "taskid2desc": taskid2desc,
                "taskid2type": self.taskid2type,
                "taskid2tname": taskid2tname,
                "taskid2metric": self.taskid2metric
            }
            
            OxfordPets.save_split(train, val, test, meta, self.split_path, self.image_dir)
            
        self.taskid2cname = taskid2cname
        self.taskid2tname = taskid2tname
        self.taskid2desc = taskid2desc

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

        train, val, test = self.subsample_tasks(train, val, test, taskid=cfg.DATASET.SUBSAMPLE_TASKS)
        # do not support subsample classes
        assert cfg.DATASET.SUBSAMPLE_CLASSES == "all"
        super().__init__(self.taskid2tname, self.taskid2cname, self.taskid2desc, self.taskid2type, self.taskid2metric, 
                         train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(image_dir, task_info, p_trn=0.5, p_val=0.2, ignored=[]):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #    0/
        #      0/
        #         xxxxx.jpg
        #         yyyyy.jpg
        #      1/
        #     
        # =============

        image_list = []
        task_cls = []
        taskids = [x for x in os.listdir(image_dir) if not x.startswith('.')]
        taskids.sort()
        for taskid in taskids:
            path1 = os.path.join(image_dir, taskid)
            clsids = [x for x in os.listdir(path1) if not x.startswith('.')]
            clsids.sort()
            for clsid in clsids:
                path2 = os.path.join(path1, clsid)
                files = [x for x in os.listdir(path2) if not x.startswith('.')]
                for file in files:
                    file = os.path.join(path2, file)
                    image_list.append(file)
                    task_cls.append([int(taskid), int(clsid)])
        task_cls = np.array(task_cls)
        assert len(task_cls) == len(image_list)
        
        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, ys, t):
            items = []
            for im,y in zip(ims, ys):
                item = DatumMT(impath=im, label=[y], task=[t])  # is already 0-based
                items.append(item)
            return items
        
        # import pdb; pdb.set_trace()
        train, val, test = [], [], []
        categories = []
        for tid, item in enumerate(task_info):  # for per task
            indices = np.where(task_cls[:, 0] == tid)[0].tolist()
            random.shuffle(indices)
            images = [image_list[j] for j in indices]
            labels = task_cls[indices, 1].tolist()

            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            train.extend(_collate(images[:n_train], labels[:n_train], tid))
            val.extend(_collate(images[n_train : n_train + n_val], labels[n_train : n_train + n_val], tid))
            test.extend(_collate(images[n_train + n_val :], labels[n_train + n_val :], tid))

            categories.append(item["label_name"])

        return train, val, test, categories

    def subsample_tasks(self, train, val, test, taskid="all"):
        if taskid == "all":
            return train, val, test

        train, val, test = None, None, None
        if taskid.startswith("task"):
            taskid = int(taskid[4:])
            assert taskid < len(self.taskid2cname), "taskid out of bound, max taskid {}".format(
                len(self.taskid2cname) - 1)
            train = [item for item in train if taskid == item.task[0]]
            val = [item for item in val if taskid == item.task[0]]
            test = [item for item in test if taskid == item.task[0]]

            for item in train + val + test:
                item.task = [0]

            self.taskid2cname = [self.taskid2cname[taskid]]
            self.taskid2tname = [self.taskid2tname[taskid]]
            self.taskid2desc = [self.taskid2desc[taskid]]
            self.taskid2type = [self.taskid2type[taskid]]
            self.taskid2metric = [self.taskid2metric[taskid]]
        else:
            raise NotImplementedError
        
        return train, val, test
import os
import pickle
import math
import random
import tqdm
import numpy as np
from collections import defaultdict
import multiprocessing
from functools import partial
from copy import deepcopy
from addict import Dict

from dassl.data.datasets import DATASET_REGISTRY, DatumMT, DatasetBaseMT
from dassl.utils import read_json, write_json, mkdir_if_missing

from .fruit_vegetable import FruitVegetable
from .kaggle_flower import KaggleFlower
from .kaggle_mushroom import KaggleMushroom
from .kaggle_vegetable import KaggleVegetable
from .oxford_flowers import OxfordFlowers
from .plant_seedling import PlantSeedlingV1
from .plant_village import PlantVillage


def init_dataset(cls, cfg):
    return cls(cfg)


@DATASET_REGISTRY.register()
class Plant6(DatasetBaseMT):
    
    dataset_dir = "plant6"
    data_sources = [
        FruitVegetable,
        KaggleFlower, 
        KaggleMushroom, 
        KaggleVegetable, 
        PlantSeedlingV1,
        PlantVillage
    ]
    
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.anno_dir = self.dataset_dir
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_mt")
        
        mkdir_if_missing(self.split_fewshot_dir)
        
        train, val, test = None, None, None
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val, test = data["train"], data["val"], data["test"]
                    
                    self.taskid2cname = data["taskid2cname"]
                    self.taskid2tname = data["taskid2tname"]
                    self.taskid2desc = data["taskid2desc"]
                    
                    taskid2type, taskid2metric = [], []
                    for data_cls in self.data_sources:
                        taskid2type.extend(data_cls.taskid2type)
                        taskid2metric.extend(data_cls.taskid2metric)
                    self.taskid2type = taskid2type
                    self.taskid2metric = taskid2metric
            else:
                cfg_tmp = Dict(deepcopy(cfg))
                cfg_tmp.DATASET.SUBSAMPLE_CLASSES = "all"
                with multiprocessing.Pool(len(self.data_sources)) as p:
                    self.datasets = p.map(
                        partial(init_dataset, cfg=cfg_tmp), 
                        self.data_sources
                )
                
                train, val, test = self.merge_tasks(self.datasets)

                data = {
                    "train": train, 
                    "val": val, 
                    "test": test,
                    "taskid2cname": self.taskid2cname,
                    "taskid2tname": self.taskid2tname,
                    "taskid2desc": self.taskid2desc,
                    "taskid2type": self.taskid2type,
                    "taskid2metric": self.taskid2metric
                }
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        else:  # read all data
            cfg_tmp = Dict(deepcopy(cfg))
            cfg_tmp.DATASET.NUM_SHOTS = -1
            cfg_tmp.DATASET.SUBSAMPLE_CLASSES = "all"
            
            with multiprocessing.Pool(len(self.data_sources)) as p:
                    self.datasets = p.map(
                        partial(init_dataset, cfg=cfg_tmp), 
                        self.data_sources
                )
            train, val, test = self.merge_tasks(self.datasets)
        
        num_classes = [len(x) for x in self.taskid2cname]
        print("before subsample task and class:")
        print("taskid2tname:")
        print(self.taskid2tname)
        print("num_classes:")
        print(num_classes)
        print("taskid2cname:")
        print(self.taskid2cname)

        train, val, test = self.subsample_tasks(train, val, test, taskid=cfg.DATASET.SUBSAMPLE_TASKS)
        train, val, test = self.subsample_classes(train, val, test, subsample=cfg.DATASET.SUBSAMPLE_CLASSES)
        super().__init__(self.taskid2tname, self.taskid2cname, self.taskid2desc, self.taskid2type, self.taskid2metric, 
                         train_x=train, val=val, test=test)
        
        num_classes = [len(x) for x in self.taskid2cname]
        print("after subsample task and class:")
        print("taskid2tname:")
        print(self.taskid2tname)
        print("num_classes:")
        print(num_classes)
        print("taskid2cname:")
        print(self.taskid2cname)
        
        write_json(
            {
                "taskid2tname": self.taskid2tname, 
                "taskid2desc": self.taskid2desc,
                "taskid2cname": self.taskid2cname,
                "taskid2type": self.taskid2type,
                "taskid2metric": self.taskid2metric,
                "cls_range": self.compute_cls_range(num_classes),
                "num_classes": num_classes, 
                "train_statics": self.get_statics(train),
                "val_statics": self.get_statics(val),
                "test_statics": self.get_statics(test)
            },
            "info.json"
        )

    def merge_tasks(self, datasets):
        
        taskid2tname = []
        taskid2desc = []
        taskid2cname = []
        taskid2type = []
        taskid2metric = []
        taskid_map = {}  # from old taskid to new taskid
        idx = 0
        for i, dataset in enumerate(datasets):
            taskid2tname.extend(dataset.taskid2tname)
            taskid2desc.extend(dataset.taskid2desc)
            taskid2cname.extend(dataset.taskid2cname)
            taskid2type.extend(dataset.taskid2type)
            taskid2metric.extend(dataset.taskid2metric)
            num_tasks = len(dataset.taskid2tname)
            for j in range(num_tasks):
                taskid_map[(i, j)] = idx
                idx += 1
                
        train = []
        for i, dataset in enumerate(datasets):
            for item in dataset.train_x:
                item.task = [taskid_map[(i, t)] for t in item.task]
                train.append(item)
        val = []
        for i, dataset in enumerate(datasets):
            for item in dataset.val:
                item.task = [taskid_map[(i, t)] for t in item.task]
                val.append(item)
        test = []
        for i, dataset in enumerate(datasets):
            for item in dataset.test:
                item.task = [taskid_map[(i, t)] for t in item.task]
                test.append(item)
                
        self.taskid2cname = taskid2cname
        self.taskid2tname = taskid2tname
        self.taskid2desc = taskid2desc
        self.taskid2type = taskid2type
        self.taskid2metric = taskid2metric
        
        return train, val, test
        
    def subsample_tasks(self, *args, taskid="all"):
        if taskid == "all":
            return args
        
        def filter_by_taskid(data_split, selected, relabeler):
            selected = set(selected)
            items = []
            for item in data_split:
                if item.task[0] in selected:
                    item.task = [relabeler[item.task[0]]]
                    items.append(item)
            return items
        
        selected = None
        if taskid.startswith("task"):
            taskid = int(taskid[4:])
            assert taskid < len(self.taskid2cname), "taskid out of bound, max taskid {}".format(len(self.taskid2cname)-1)
            selected = [taskid]
        elif taskid in ["base", "new"]:
            m = math.ceil(len(self.taskid2cname) / 2)
            taskids = list(range(len(self.taskid2cname)))
            if taskid == "base":
                selected = taskids[:m]
            else:
                selected = taskids[m:]
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = [filter_by_taskid(x, selected, relabeler) for x in args]
        
        self.taskid2cname = [self.taskid2cname[i] for i in selected]
        self.taskid2tname = [self.taskid2tname[i] for i in selected]
        self.taskid2desc = [self.taskid2desc[i] for i in selected]
        self.taskid2type = [self.taskid2type[i] for i in selected]
        self.taskid2metric = [self.taskid2metric[i] for i in selected]
        
        return output
    
    def subsample_classes(self, *args, subsample):
        
        def filter_by_class(data_split, selecteds, relabelers):
            items = []
            for item in data_split:
                task = item.task[0]
                ls = []
                for l in item.label:
                    if l in selecteds[task]:
                        ls.append(relabelers[task][l])
                if len(ls):
                    item.label = ls
                    items.append(item)
            return items
        
        assert subsample in ["all", "base", "new"]
        if subsample=="all": return args
        
        selecteds = []
        relabelers = []
        for i in range(len(self.taskid2cname)):
            n = len(self.taskid2cname[i])
            # Divide classes into two halves
            m = math.ceil(n / 2)
            print(f"SUBSAMPLE {subsample.upper()} CLASSES for task {i}!")
            labels = list(range(n))
            if subsample == "base":
                selected = labels[:m]  # take the first half
            else:
                selected = labels[m:]  # take the second half
            relabeler = {y: y_new for y_new, y in enumerate(selected)}
            
            selecteds.append(selected)
            relabelers.append(relabeler)
            
        output = [filter_by_class(x, selecteds, relabelers) for x in args]
        
        taskid2cname = []
        for i, names in enumerate(self.taskid2cname):
            selected = selecteds[i]
            taskid2cname.append([names[j] for j in selected])
        self.taskid2cname = taskid2cname
        
        return output
    
    def compute_cls_range(self, num_classes):
        cls_range = []
        tt = 0
        for i, c in enumerate(num_classes):
            cls_range.append((tt, tt+c))
            tt += c
        return cls_range
    
    def get_statics(self, data_source):
        
        output = self.split_dataset_by_label(data_source)
        keys = list(output.keys())
        
        # number of samples per task-cls
        per_label_cnt = {i:{} for i in range(len(self.taskid2tname))}
        per_task_cnt = {i:0 for i in range(len(self.taskid2tname))}
        tol = 0
        for key in keys:
            taskid, clsid = key
            num = len(output[key])
            per_label_cnt[taskid][clsid] = num
            per_task_cnt[taskid] += num
            tol += num
        res = {
            "tol": tol,
            "per_task_cnt": per_task_cnt,
            "per_label_cnt": per_label_cnt
        }
        
        return res
        
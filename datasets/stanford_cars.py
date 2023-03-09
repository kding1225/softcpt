import os
import pickle
from scipy.io import loadmat

from dassl.data.datasets import DATASET_REGISTRY, DatumMT, DatasetBaseMT
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class StanfordCars(DatasetBaseMT):

    dataset_dir = "stanford_cars"
    taskid2type = ['sl']
    taskid2metric = ["overall_acc"]

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_dk_StanfordCars.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_mt")
        self.taskdesc_file = os.path.join(self.dataset_dir, "task_desc.txt")
        self.taskname_file = os.path.join(self.dataset_dir, "task_name.txt")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test, meta = OxfordPets.read_split(self.split_path, self.dataset_dir)
            
            taskid2cname = meta["taskid2cname"]
            taskid2desc = meta["taskid2desc"]
            taskid2tname = meta["taskid2tname"]
        else:
            trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
            test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
            meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")
            trainval, taskid2cname = self.read_data("cars_train", trainval_file, meta_file)
            test, _ = self.read_data("cars_test", test_file, meta_file)
            train, val = OxfordPets.split_trainval(trainval)
            
            taskid2desc = [open(self.taskdesc_file, 'r').read().strip()]
            taskid2tname = [open(self.taskname_file, 'r').read().strip()]
            
            meta = {
                "taskid2cname": taskid2cname,
                "taskid2desc": taskid2desc,
                "taskid2type": self.taskid2type,
                "taskid2tname": taskid2tname,
                "taskid2metric": self.taskid2metric
            }
            
            OxfordPets.save_split(train, val, test, meta, self.split_path, self.dataset_dir)

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

    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        items = []
        
        M = {}
        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            item = DatumMT(impath=impath, label=[label], task=[0])
            items.append(item)
            M[label] = classname
        
        keys = list(M.keys())
        keys.sort()
        
        return items, [[M[k] for k in keys]]

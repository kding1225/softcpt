import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, DatumMT, DatasetBaseMT
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetBaseMT):

    dataset_dir = "caltech-101"
    taskid2type = ['sl']
    taskid2metric = ['mean_acc']
    
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_dk_Caltech101.json")
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
            train, val, test, taskid2cname = DTD.read_and_split_data(self.image_dir, ignored=IGNORED)
            
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
            
        taskid2cname = self.update_cnames(taskid2cname)

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
    def update_cnames(taskid2cname):
        
        new_taskid2cname = []
        for names in taskid2cname:
            new_names = [NEW_CNAMES[name] if name in NEW_CNAMES else name for name in names]
            new_taskid2cname.append(new_names)
        
        return new_taskid2cname
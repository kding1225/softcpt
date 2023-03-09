import os
import pickle
import random

from dassl.data.datasets import DATASET_REGISTRY, DatumMT, DatasetBaseMT
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "Apple___Apple_scab": "apple leaf with scab disease",
    "Grape___Esca_(Black_Measles)": "grape leaf with black measles",
    "Squash___Powdery_mildew": "squash with powdery mildew",
    "Apple___Black_rot": "apple leaf with black rot",
    "Grape___healthy": "healthy grape leaf",
    "Strawberry___healthy": "healthy strawberry leaf",
    "Apple___Cedar_apple_rust": "apple leaf with cedar apple rust",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape leaf with leaf blight",
    "Strawberry___Leaf_scorch": "strawberry leaf with scorch disease",
    "Apple___healthy": "healthy apple leaf",
    "Orange___Haunglongbing_(Citrus_greening)": "orange leaf with citrus greening disease",
    "Tomato___Bacterial_spot": "tomato leaf with bacterial spot",
    "Background_without_leaves": "background without leaves",
    "Peach___Bacterial_spot": "peach leaf with bacterial spot",
    "Tomato___Early_blight": "tomato leaf with early blight",
    "Blueberry___healthy": "healthy blueberry leaf",
    "Peach___healthy": "healthy peach leaf",
    "Tomato___healthy": "healthy tomato leaf",
    "Cherry___healthy": "healthy cherry leaf",
    "Pepper,_bell___Bacterial_spot": "pepper with bacterial spot",
    "Tomato___Late_blight": "tomato leaf with late blight",
    "Cherry___Powdery_mildew": "cherry leaf with powdery mildew",
    "Pepper,_bell___healthy": "healthy pepper leaf",
    "Tomato___Leaf_Mold": "tomato leaf with mold",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "corn leaf with cercospora leaf spot or gray leaf spot",
    "Potato___Early_blight": "potato leaf with early blight",
    "Tomato___Septoria_leaf_spot": "tomato leaf with septoria leaf spot",
    "Corn___Common_rust": "corn leaf with common rust",
    "Potato___healthy": "healthy potato leaf",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato leaf with spider mites or two-spotted spider mite",
    "Corn___healthy": "healthy corn leaf",
    "Potato___Late_blight": "potato leaf with late blight",
    "Tomato___Target_Spot": "tomato leaf with target spot",
    "Corn___Northern_Leaf_Blight": "corn leaf with northern leaf blight",
    "Raspberry___healthy": "healthy raspberry leaf",
    "Tomato___Tomato_mosaic_virus": "tomato leaf with tomato mosaic virus",
    "Grape___Black_rot": "grape leaf with black rot",
    "Soybean___healthy": "healthy soybean leaf",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato leaf with tomato yellow leaf curl virus"
}


@DATASET_REGISTRY.register()
class PlantVillage(DatasetBaseMT):

    dataset_dir = "PlantVillage"
    taskid2type = ['sl']
    taskid2metric = ["overall_acc"]
    
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "Plant_leave_diseases_dataset_without_augmentation")
        self.split_path = os.path.join(self.dataset_dir, "split_dk_PlantVillage.json")
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
            train, val, test, taskid2cname = DTD.read_and_split_data(self.image_dir)
            self.process_class_names(taskid2cname)
            print(taskid2cname)
            
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
    
    def process_class_names(self, taskid2cname):
        
        for names in taskid2cname:
            for i, name in enumerate(names):
                names[i] = NEW_CNAMES[name]
import itertools
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import os.path as osp

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.evaluation import build_evaluator

from clip import clip
from clip.model import convert_weights

from .mtcoop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "FruitVegetable": "a photo of a {}, a type of fruit or vegetable.",
    "KaggleFlower": "a photo of a {}, a type of flower.",
    "KaggleMushroom": "a photo of a {}, a type of mushroom.",
    "KaggleVegetable": "a photo of a {}, a type of vegetable.",
    "PlantSeedlingV1": "a photo of a {}, a type of seedling.",
    "PlantVillage": "a photo of a plant leaf with {} disease.",
    "General10": {
        0: "a photo of a {}.",  # caltech101
        1: "{} texture.",  # dtd
        2: "a centered satellite photo of {}.",  # eurosat
        3: "a photo of a {}, a type of aircraft.",  # fgvc_aircraft
        4: "a photo of {}, a type of food.",  # food101 
        5: "a photo of a {}, a type of flower.",  # oxford_flowers
        6: "a photo of a {}, a type of pet.", # oxford_pets
        7: "a photo of a {}.", # stanford_cars
        8: "a photo of a {}.", # sun397
        9: "a photo of a person doing {}.",  # ucf101
    },
    "Plant6": {
        0: "a photo of a {}, a type of fruit or vegetable.",  # fruit_vegetable
        1: "a photo of a {}, a type of flower.",  # kaggle_flower
        2: "a photo of a {}, a type of mushroom.",  # kaggle_mushroom
        3: "a photo of a {}, a type of vegetable.",  # kaggle_vegetable
        4: "a photo of a {}, a type of seedling.",  # plant_seedling
        5: "a photo of a plant leaf with {} disease.",  # plant_village
    },
    "Fashion20": {
        0: "a photo of {}, a type of pants.",
        1: "a photo of {}, a type of pants.",
        2: "a photo of {}, a type of pants.",
        3: "a photo of {}, a type of tops.",
        4: "a photo of {}, a type of tops.",
        5: "a photo of {}, a type of tops.",
        6: "a photo of {}, a type of tops.",
        7: "a photo of {}, a type of shoes.",
        8: "a photo of {}, a type of shoes.",
        9: "a photo of {}, a type of shoes.",
        10: "a photo of {}, a type of shoes.",
        11: "a photo of {}, a type of shoes.",
        12: "a photo of {}, a type of shoes.",
        13: "a photo of {}, a type of shoes.",
        14: "a photo of {}, a type of hat.",
        15: "a photo of {}, a type of socks.",
        16: "a photo of {}, a type of socks.",
        17: "a photo of {}, a type of skirt.",
        18: "a photo of {}, a type of tops.",
        19: "a photo of {}, a type of underwear.",
    }
}


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    
    def __init__(self, cfg):
        
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(
            cfg, 
            lab2cname=self.taskid2cname, 
            task2tname=self.taskid2tname,
            task2type=self.taskid2type,
            task2metric=self.taskid2metric,
        )
        self.best_result = -np.inf
    
    def build_model(self):
        cfg = self.cfg
        
        classnames = self.dm.dataset.taskid2cname
        self.taskid2tname = self.dm.dataset.taskid2tname
        self.taskid2cname = self.dm.dataset.taskid2cname
        self.taskid2type = self.dm.dataset.taskid2type
        self.taskid2metric = self.dm.dataset.taskid2metric
        self.num_tasks = len(self.taskid2tname)
        self.cls_range = self.compute_cls_range([len(x) for x in self.taskid2cname])
        
        if isinstance(classnames[0], (list, tuple)):
            classnames = list(itertools.chain.from_iterable(classnames))
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)
        
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        if isinstance(temp, dict):
            taskid = cfg.DATASET.SUBSAMPLE_TASKS
            assert taskid.startswith("task")
            temp = temp[int(taskid[4:])]
        
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
    
    def compute_cls_range(self, num_classes):
        cls_range = []
        tt = 0
        for i, c in enumerate(num_classes):
            cls_range.append((tt, tt+c))
            tt += c
        return cls_range


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(TrainerX):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def __init__(self, cfg):
        
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(
            cfg, 
            lab2cname=self.taskid2cname, 
            task2tname=self.taskid2tname,
            task2type=self.taskid2type,
            task2metric=self.taskid2metric,
        )
        self.best_result = -np.inf
    
    def build_model(self):
        cfg = self.cfg
        
        classnames = self.dm.dataset.taskid2cname
        self.taskid2tname = self.dm.dataset.taskid2tname
        self.taskid2cname = self.dm.dataset.taskid2cname
        self.taskid2type = self.dm.dataset.taskid2type
        self.taskid2metric = self.dm.dataset.taskid2metric
        self.num_tasks = len(self.taskid2tname)
        self.cls_range = self.compute_cls_range([len(x) for x in self.taskid2cname])

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model
    
    def compute_cls_range(self, num_classes):
        cls_range = []
        tt = 0
        for i, c in enumerate(num_classes):
            cls_range.append((tt, tt+c))
            tt += c
        return cls_range
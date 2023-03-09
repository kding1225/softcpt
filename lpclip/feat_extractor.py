import os, argparse
import numpy as np
import torch
import sys
import tqdm

sys.path.append(os.path.abspath(".."))

from datasets.oxford_pets import OxfordPets
from datasets.oxford_flowers import OxfordFlowers
from datasets.fgvc_aircraft import FGVCAircraft
from datasets.dtd import DescribableTextures
from datasets.eurosat import EuroSAT
from datasets.stanford_cars import StanfordCars
from datasets.food101 import Food101
from datasets.sun397 import SUN397
from datasets.caltech101 import Caltech101
from datasets.ucf101 import UCF101
from datasets.imagenet import ImageNet
from datasets.imagenetv2 import ImageNetV2
from datasets.imagenet_sketch import ImageNetSketch
from datasets.imagenet_a import ImageNetA
from datasets.imagenet_r import ImageNetR

from datasets.fruit_vegetable import FruitVegetable
from datasets.kaggle_flower import KaggleFlower
from datasets.kaggle_mushroom import KaggleMushroom
from datasets.kaggle_vegetable import KaggleVegetable
from datasets.general10 import General10
from datasets.plant6 import Plant6
from datasets.fashion20 import Fashion20
from datasets.plant_seedling import PlantSeedlingV1
from datasets.plant_village import PlantVillage

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.data.transforms import build_transform
from dassl.data import DatasetWrapper

from clip import clip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.old_data_path:
        cfg.DATALOADER.OLD_DATA_PATH = args.old_data_path

    if args.new_data_path:
        cfg.DATALOADER.NEW_DATA_PATH = args.new_data_path


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.OURS = CN()
    cfg.TRAINER.OURS.N_CTX = 10  # number of context vectors
    cfg.TRAINER.OURS.CSC = False  # class-specific context
    cfg.TRAINER.OURS.CTX_INIT = ""  # initialize context vectors with given words
    cfg.TRAINER.OURS.WEIGHT_U = 0.1  # weight for the unsupervised loss

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.SUBSAMPLE_TASKS = "all"  # [taski, all, base, new]

    cfg.DATALOADER.OLD_DATA_PATH = ""  # old data path in the split files
    cfg.DATALOADER.NEW_DATA_PATH = ""  # new data path in your file system


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)
    
    # 4. From optional input arguments
    print(args.opts)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    ######################################
    #   Setup DataLoader
    ######################################
    dataset = eval(cfg.DATASET.NAME)(cfg)

    if args.split == "train":
        dataset_input = dataset.train_x
    elif args.split == "val":
        dataset_input = dataset.val
    else:
        dataset_input = dataset.test

    tfm_train = build_transform(cfg, is_train=False)
    data_loader = torch.utils.data.DataLoader(
        DatasetWrapper(cfg, dataset_input, transform=tfm_train, is_train=False, taskid2cname=dataset.taskid2cname),
        batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        sampler=None,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )

    ########################################
    #   Setup Network
    ########################################
    print("***************** backbone type: *****************")
    print(cfg.MODEL.BACKBONE.NAME)
    clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME, "cuda", jit=False)
    clip_model.eval()
    ###################################################################################################################
    # Start Feature Extractor
    feature_list = []
    label_list = []
    train_dataiter = iter(data_loader)
    for train_step in tqdm.tqdm(range(1, len(train_dataiter) + 1)):
        batch = next(train_dataiter)
        data = batch["img"].cuda()
        feature = clip_model.visual(data.half())
        feature = feature.cpu()
        for idx in range(len(data)):
            feature_list.append(feature[idx].tolist())
        
        label = []
        for row in batch["label"]:  # assume only one task
            label.append(torch.nonzero(row)[:, 0].tolist())
        
        label_list.extend(label)
    name = os.path.splitext(os.path.split(args.dataset_config_file)[-1])[0]
    # output_dir = os.path.join(cfg.OUTPUT_DIR, name)
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    save_filename = f"{args.split}"
    np.savez(
        os.path.join(output_dir, save_filename),
        feature_list=feature_list,
        label_list=label_list
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--num-shot", type=int, default=1, help="number of shots")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="which split")
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--old-data-path", type=str, default="")
    parser.add_argument("--new-data-path", type=str, default="")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    
    args = parser.parse_args()
    main(args)

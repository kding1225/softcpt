import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import datasets.fruit_vegetable
import datasets.kaggle_flower
import datasets.kaggle_mushroom
import datasets.kaggle_vegetable
import datasets.general10
import datasets.plant6
import datasets.plant_seedling
import datasets.plant_village
import datasets.fashion20

import trainers.mtcoop_hard
import trainers.mtcoop
import trainers.zsclip


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

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

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
        
    cfg.TEST.PER_TASK_RESULT = False

    cfg.TRAINER.MTCOOPH = CN()
    cfg.TRAINER.MTCOOPH.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MTCOOPH.N_CTX = 16  # number of context vectors
    cfg.TRAINER.MTCOOPH.CSC = False  # class-specific context
    cfg.TRAINER.MTCOOPH.CTX_INIT = ""  # initialization words
    cfg.TRAINER.MTCOOPH.TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.MTCOOPH.TSC = False # task-specific context

    cfg.TRAINER.MTCOOP = CN()
    cfg.TRAINER.MTCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MTCOOP.CLASS_N_CTX = 16  # number of context vectors
    cfg.TRAINER.MTCOOP.CLASS_CSC = False  # class-specific context
    cfg.TRAINER.MTCOOP.CLASS_CTX_INIT = ""  # initialization words
    cfg.TRAINER.MTCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.MTCOOP.TASK_N_CTX = 16
    cfg.TRAINER.MTCOOP.TASK_CTX_INIT = ""
    cfg.TRAINER.MTCOOP.TASK_CSC = False
    cfg.TRAINER.MTCOOP.TASK_TOKEN_POSITION = "end"
    cfg.TRAINER.MTCOOP.N_CTX = 16
    cfg.TRAINER.MTCOOP.CTX_INIT = ""
    cfg.TRAINER.MTCOOP.CSC = False
    cfg.TRAINER.MTCOOP.TOKEN_POSITION = "end"
    cfg.TRAINER.MTCOOP.PG_TYPE = ""
    cfg.TRAINER.MTCOOP.PG_RATIO = 1
    cfg.TRAINER.MTCOOP.CLS_SAMPLE_RATE = 0.0
    cfg.TRAINER.MTCOOP.FREEZE_CLS_PROMPT = 0

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.SUBSAMPLE_TASKS = "all"  # [taski, all, base, new]

    cfg.DATALOADER.OLD_DATA_PATH = ""  # old data path in the split files
    cfg.DATALOADER.NEW_DATA_PATH = ""  # new data path in your file system


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)  # add extra cfg

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

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
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

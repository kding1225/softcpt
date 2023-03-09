#!/bin/bash
sh set_env.sh

cd ..

# custom config
DATA=$YOUR_DATA_PATH
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
TID=$3  # choose a task from this dataset
CLASS=$4 # base or new

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/ZeroShot/${CFG}.yaml \
--output-dir "output/base2new/${TRAINER}/${CFG}/${CLASS}/${DATASET}-${TID}" \
--eval-only \
DATASET.NUM_SHOTS 1 \
DATASET.SUBSAMPLE_TASKS "task${TID}" \
DATASET.SUBSAMPLE_CLASSES ${CLASS}

#!/bin/bash
sh set_env.sh

cd ..

# custom config
DATA=$YOUR_DATA_PATH
TRAINER=MTCoOp

DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
CLASS_N_CTX=$4
CLASS_CSC=$5
TASK_N_CTX=$6
TASK_CSC=$7
N_CTX=$8
CSC=$9
PG_TYPE=${10}
PG_RATIO=${11}

for SEED in 1 2 3
do
    DIR=output/base2new/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/cnctx${CLASS_N_CTX}_ccsc${CLASS_CSC}_tnctx${TASK_N_CTX}_tcsc${TASK_CSC}_nctx${N_CTX}_csc${CSC}_type${PG_TYPE}_ratio${PG_RATIO}_loss${LOSS}/base/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.MTCOOP.CLASS_N_CTX ${CLASS_N_CTX} \
        TRAINER.MTCOOP.CLASS_CSC ${CLASS_CSC} \
        TRAINER.MTCOOP.TASK_N_CTX ${TASK_N_CTX} \
        TRAINER.MTCOOP.TASK_CSC ${TASK_CSC} \
        TRAINER.MTCOOP.N_CTX ${N_CTX} \
        TRAINER.MTCOOP.CSC ${CSC} \
        TRAINER.MTCOOP.PG_TYPE ${PG_TYPE} \
        TRAINER.MTCOOP.PG_RATIO ${PG_RATIO} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES "base"
    fi
done

#!/bin/bash
sh set_env.sh

cd ..

# custom config
DATA=$YOUR_DATA_PATH
TRAINER=MTCoOpHard

DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
NCTX=$4  # number of context tokens
CSC=$5  # class-specific context (False or True)
TSC=$6 # task-specific

for SEED in 1 2 3
do 
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_tsc${TSC}/seed${SEED}
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
        TRAINER.MTCOOPH.N_CTX ${NCTX} \
        TRAINER.MTCOOPH.CSC ${CSC} \
        TRAINER.MTCOOPH.TSC ${TSC} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done

#!/bin/bash
sh set_env.sh
cd ../lpclip

DATA=$YOUR_DATA_PATH
DATASET=$1
MODEL=$2
MAX_TASK_IDX=$3  # max task id

sh feat_extractor.sh ${DATA} ${DATASET} ${MODEL} ${MAX_TASK_IDX}
sh linear_probe.sh ${DATASET} ${MODEL} ${MAX_TASK_IDX}

echo "done"

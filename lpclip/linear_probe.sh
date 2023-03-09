DATASET=$1
MODEL=$2
MAX_TASK_IDX=$3  # max index of taskid
FEAT_DIR=../output/lpclip/${DATASET}/${MODEL}

for TID in $(seq 0 ${MAX_TASK_IDX})
do
    python linear_probe.py \
    --dataset ${DATASET} \
    --feature_dir ${FEAT_DIR}/${TID} \
    --num_step 8 \
    --num_run 3 \
    --task-id ${TID}
done

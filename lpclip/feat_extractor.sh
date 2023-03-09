# sh feat_extractor.sh
DATA=$1
DATASET=$2
MODEL=$3
MAX_TASK_IDX=$4  # max index of taskid
OUTPUT=../output/lpclip/${DATASET}/${MODEL}
SEED=1

for TID in $(seq 0 ${MAX_TASK_IDX})
do
    for SPLIT in train val test
    do
        if [ -f "${OUTPUT}/${TID}/${SPLIT}.npz" ]; then
            echo "skip ${OUTPUT}/${TID}/${SPLIT}.npz"
        else
            python feat_extractor.py \
                --split ${SPLIT} \
                --root ${DATA} \
                --seed ${SEED} \
                --dataset-config-file ../configs/datasets/${DATASET}.yaml \
                --config-file ../configs/trainers/LP/${MODEL}_val.yaml \
                --output-dir ${OUTPUT}/${TID} \
                --eval-only \
                DATASET.SUBSAMPLE_TASKS "task${TID}"
        fi
    done
done
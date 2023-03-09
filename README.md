# [Prompt Tuning with Soft Context Sharing for Vision-Language Models](https://arxiv.org/abs/2208.13474)

Vision-language models have recently shown great potential on many computer vision tasks. Meanwhile,
prior work demonstrates prompt tuning designed for vision-language models could acquire superior performance on few-shot
image recognition compared to linear probe, a strong baseline. In real-world applications, many few-shot tasks are correlated,
particularly in a specialized area. However, such information is ignored by previous work. Inspired by the fact that modeling task
relationships by multi-task learning can usually boost performance, we propose a novel method SoftCPT (Soft Context Sharing for Prompt Tuning)
to fine-tune pre-trained vision-language models on multiple target few-shot tasks, simultaneously. Specifically, we design a task-shared meta
network to generate prompt vector for each task using pre-defined task name together with a learnable meta prompt as input. As such,
the prompt vectors of all tasks will be shared in a soft manner. The parameters of this shared meta network as well as the meta prompt
vector are tuned on the joint training set of all target tasks. Extensive experiments on three multi-task few-shot datasets show that SoftCPT
outperforms the representative single-task prompt tuning method CoOp by a large margin, implying the effectiveness of multi-task learning
in vision-language prompt tuning. The source code and data will be made publicly available.

## Installation
Our code is based on [CoOp](https://github.com/KaiyangZhou/CoOp), please refer to it for the installation.

## Datasets
Please refer to CoOp for downloading the datasets used in General-10. The datasets should be put into the subfolders according to the
`dataset_dir` class attribute in each dataset class. For downloading the datasets used in Plant-6, please refer to the link provided in
our original paper. The Fashion-20 dataset will be released soon.

## How to Run
Set environment variables in scripts/set_env.sh first.

### Multi-task Learning on All Classes

The navie hard sharing is implemented in trainers/mtcoop_hard.py, it also supports CoOp. The examples for use are listed below:

(1) hard shared CoOp, all classes of all tasks share a prompt
```commandline
sh main_mtcoop_hard.sh general10 rn50_ep50 1 16 False False
sh main_mtcoop_hard.sh general10 rn50_ep50 2 16 False False
sh main_mtcoop_hard.sh general10 rn50_ep50 4 16 False False
sh main_mtcoop_hard.sh general10 rn50_ep50 8 16 False False
sh main_mtcoop_hard.sh general10 rn50_ep50 16 16 False False
```

(2) CoOp, one prompt per task
```commandline
sh main_mtcoop_hard.sh general10 rn50_ep50 1 16 False True
sh main_mtcoop_hard.sh general10 rn50_ep50 2 16 False True
sh main_mtcoop_hard.sh general10 rn50_ep50 4 16 False True
sh main_mtcoop_hard.sh general10 rn50_ep50 8 16 False True
sh main_mtcoop_hard.sh general10 rn50_ep50 16 16 False True
```

(3) one prompt per class
```commandline
sh main_mtcoop_hard.sh general10 rn50_ep50 1 16 True False
sh main_mtcoop_hard.sh general10 rn50_ep50 2 16 True False
sh main_mtcoop_hard.sh general10 rn50_ep50 4 16 True False
sh main_mtcoop_hard.sh general10 rn50_ep50 8 16 True False
sh main_mtcoop_hard.sh general10 rn50_ep50 16 16 True False
```

Our soft sharing is implemented in trainers/mtcoop.py. The examples for use are listed below:

(1) SoftCPT-NATA
```commandline
sh main_mtcoop.sh general10 rn50_ep50 1 0 False 4 False 16 False lin 1
sh main_mtcoop.sh general10 rn50_ep50 2 0 False 4 False 16 False lin 1
sh main_mtcoop.sh general10 rn50_ep50 4 0 False 4 False 16 False lin 1
sh main_mtcoop.sh general10 rn50_ep50 8 0 False 4 False 16 False lin 1
sh main_mtcoop.sh general10 rn50_ep50 16 0 False 4 False 16 False lin 1
```

(2) SoftCPT-NATS
```commandline
sh main_mtcoop.sh general10 rn50_ep50 1 0 False 4 True 16 False lin 1
sh main_mtcoop.sh general10 rn50_ep50 2 0 False 4 True 16 False lin 1
sh main_mtcoop.sh general10 rn50_ep50 4 0 False 4 True 16 False lin 1
sh main_mtcoop.sh general10 rn50_ep50 8 0 False 4 True 16 False lin 1
sh main_mtcoop.sh general10 rn50_ep50 16 0 False 4 True 16 False lin 1
```

(3) SoftCPT-CATA

On General:
```commandline
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 1 4 False 4 False 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 2 4 False 4 False 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 4 4 False 4 False 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 8 4 False 4 False 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 16 4 False 4 False 16 True lin 1 0.1
```

On Plant-6 and Fshion-20:
```commandline
sh main_mtcoop.sh plant6 rn50_ep50 1 4 False 4 False 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 2 4 False 4 False 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 4 4 False 4 False 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 8 4 False 4 False 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 16 4 False 4 False 16 True lin 1
```

(4) SoftCPT-CSTA

On General:
```commandline
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 1 4 True 4 False 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 2 4 True 4 False 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 4 4 True 4 False 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 8 4 True 4 False 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 16 4 True 4 False 16 True lin 1 0.1
```

On Plant-6 and Fshion-20:
```commandline
sh main_mtcoop.sh plant6 rn50_ep50 1 4 True 4 False 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 2 4 True 4 False 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 4 4 True 4 False 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 8 4 True 4 False 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 16 4 True 4 False 16 True lin 1
```

(5) SoftCPT-CATS

On General:
```commandline
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 1 4 False 4 True 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 2 4 False 4 True 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 4 4 False 4 True 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 8 4 False 4 True 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 16 4 False 4 True 16 True lin 1 0.1
```

On Plant-6 and Fshion-20:
```commandline
sh main_mtcoop.sh plant6 rn50_ep50 1 4 False 4 True 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 2 4 False 4 True 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 4 4 False 4 True 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 8 4 False 4 True 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 16 4 False 4 True 16 True lin 1
```

(6) SoftCPT-CSTS

On General:
```commandline
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 1 4 True 4 True 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 2 4 True 4 True 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 4 4 True 4 True 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 8 4 True 4 True 16 True lin 1 0.1
sh main_mtcoop_cls_sample.sh general10 rn50_ep50 16 4 True 4 True 16 True lin 1 0.1
```

On Plant-6 and Fshion-20:
```commandline
sh main_mtcoop.sh plant6 rn50_ep50 1 4 True 4 True 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 2 4 True 4 True 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 4 4 True 4 True 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 8 4 True 4 True 16 True lin 1
sh main_mtcoop.sh plant6 rn50_ep50 16 4 True 4 True 16 True lin 1
```

### Base to New Class Experiments

train SoftCPT-NATA on base classes:
```commandline
sh main_mtcoop_base2new_train.sh general10 rn50_ep50 1 0 False 4 False 16 False lin 1
sh main_mtcoop_base2new_train.sh general10 rn50_ep50 2 0 False 4 False 16 False lin 1
sh main_mtcoop_base2new_train.sh general10 rn50_ep50 4 0 False 4 False 16 False lin 1
sh main_mtcoop_base2new_train.sh general10 rn50_ep50 8 0 False 4 False 16 False lin 1
sh main_mtcoop_base2new_train.sh general10 rn50_ep50 16 0 False 4 False 16 False lin 1
```

test SoftCPT-NATA on new classes:
```commandline
sh main_mtcoop_base2new_test.sh general10 rn50_ep50 1 0 False 4 False 16 False lin 1
sh main_mtcoop_base2new_test.sh general10 rn50_ep50 2 0 False 4 False 16 False lin 1
sh main_mtcoop_base2new_test.sh general10 rn50_ep50 4 0 False 4 False 16 False lin 1
sh main_mtcoop_base2new_test.sh general10 rn50_ep50 8 0 False 4 False 16 False lin 1
sh main_mtcoop_base2new_test.sh general10 rn50_ep50 16 0 False 4 False 16 False lin 1
```

### zero shot learning

on all classes for a task:
```commandline
# taskid in [0, 9] on general10, in [0, 5] on plant6, in [0, 19] on fashion20
sh zeroshot.sh plant6 rn50 $taskid
```

on base classes for a task:
```commandline
sh zeroshot_base2new.sh plant6 rn50 $taskid base
```

on new classes for a task:
```commandline
sh zeroshot_base2new.sh plant6 rn50 $taskid new
```

### linear probe

```commandline
sh linearprob.sh general10 rn50 9
sh linearprob.sh plant6 rn50 5
sh linearprob.sh fashion20 rn50 19
```

### parse results

```commandline
python parse_test_res.py PATH_TO_RESULTS
```

## Citation
If you use this code in your research, please kindly cite the following paper

```
@article{softcpt,
    title={Prompt Tuning with Soft Context Sharing for Vision-Language Models},
    author={Kun Ding, Ying Wang, Pengzhang Liu, Qiang Yu, Haojian Zhang, Shiming Xiang and Chunhong Pan},
    journal={arXiv preprint arXiv:2208.13474},
    year={2022}
}
```

## Acknowledgments
We would like to thank [@KaiyangZhou](https://github.com/KaiyangZhou/CoOp) for sharing the code.

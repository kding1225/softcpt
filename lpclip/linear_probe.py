"""
this code only support single label
"""
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import argparse
import itertools


def compute_metric(preds, gts, num_classes=None, typ="overall_acc"):
    """
    typ: overall_acc, mean_acc
    """
    
    # weird trick with bincount
    def confusion_matrix_2_numpy(y_true, y_pred, N):
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1) 
        if (N is None):
            N = max(max(y_true), max(y_pred)) + 1
        y = N * y_true + y_pred
        y = np.bincount(y, minlength=N*N)
        y = y.reshape(N, N)
        return y
    
    assert typ in ["overall_acc", "mean_acc"]
    
    if typ == "overall_acc":
        confusion_matrix = confusion_matrix_2_numpy(gts.astype(np.int64), preds.astype(np.int64), num_classes)
        per_class_accs = np.diag(confusion_matrix)/confusion_matrix.sum(1)
        return np.mean(per_class_accs)
    elif typ == "mean_acc":
        return np.sum(preds==gts)/len(preds)


def get_metric_type(data_name, task_id):
    
    name_to_metric_type = {
        "oxford_pets": "mean_acc",
        "oxford_flowers": "mean_acc",
        "fgvc_aircraft": "mean_acc",
        "dtd": "overall_acc",
        "eurosat": "overall_acc",
        "stanford_cars": "overall_acc",
        "food101": "overall_acc",
        "sun397": "overall_acc",
        "caltech101": "mean_acc",
        "ucf101": "overall_acc",
        "fruit_vegetable": "overall_acc",
        "kaggle_flower": "overall_acc",
        "kaggle_mushroom": "overall_acc",
        "kaggle_vegetable": "overall_acc",
        "plant_seedling": "overall_acc",
        "plant_village": "overall_acc",
        "general10": [
            "mean_acc",
            "overall_acc",
            "overall_acc",
            "mean_acc",
            "overall_acc",
            "mean_acc",
            "mean_acc",
            "overall_acc",
            "overall_acc",
            "overall_acc"
        ],
        "plant6": ["overall_acc"]*6,
        "fashion20": ["mean_acc"]*20,
    }
    metric = name_to_metric_type[data_name]
    if task_id != -1:
        metric = metric[task_id]
    return metric


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="", help="path to dataset")
parser.add_argument("--num_step", type=int, default=8, help="number of steps")
parser.add_argument("--num_run", type=int, default=10, help="number of runs")
parser.add_argument("--feature_dir", type=str, default="clip_feat", help="feature dir path")
parser.add_argument("--task-id", type=int, default=-1, help="select a task")
args = parser.parse_args()

dataset = args.dataset
dataset_path = args.feature_dir
metric_type = get_metric_type(dataset, args.task_id)

train_file = np.load(os.path.join(dataset_path, "train.npz"))
train_feature, train_label = train_file["feature_list"], train_file["label_list"]
val_file = np.load(os.path.join(dataset_path, "val.npz"))
val_feature, val_label = val_file["feature_list"], val_file["label_list"]
test_file = np.load(os.path.join(dataset_path, "test.npz"))
test_feature, test_label = test_file["feature_list"], test_file["label_list"]

train_label = list(itertools.chain.from_iterable(train_label))
val_label = list(itertools.chain.from_iterable(val_label))
test_label = list(itertools.chain.from_iterable(test_label))

train_label = np.array(train_label)
val_label = np.array(val_label)
test_label = np.array(test_label)

# os.makedirs("report", exist_ok=True)
val_shot_list = {1: 1, 2: 2, 4: 4, 8: 4, 16: 4}

for num_shot in [1, 2, 4, 8, 16]:
    test_acc_step_list = np.zeros([args.num_run, args.num_step])
    for seed in range(1, args.num_run + 1):
        np.random.seed(seed)
        print(f"-- Seed: {seed} --------------------------------------------------------------")
        # Sampling
        all_label_list = np.unique(train_label)
        selected_idx_list = []
        for label in all_label_list:
            label_collection = np.where(train_label == label)[0]
            selected_idx = np.random.choice(label_collection, size=num_shot, replace=False)
            selected_idx_list.extend(selected_idx)

        fewshot_train_feature = train_feature[selected_idx_list]
        fewshot_train_label = train_label[selected_idx_list]

        val_num_shot = val_shot_list[num_shot]
        val_selected_idx_list = []
        for label in all_label_list:
            label_collection = np.where(val_label == label)[0]
            selected_idx = np.random.choice(label_collection, size=val_num_shot, replace=False)
            val_selected_idx_list.extend(selected_idx)
        
        fewshot_val_feature = val_feature[val_selected_idx_list]
        fewshot_val_label = val_label[val_selected_idx_list]
        
        # search initialization
        search_list = [1e6, 1e4, 1e2, 1, 1e-2, 1e-4, 1e-6]
        acc_list = []
        for c_weight in search_list:
            clf = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_weight).fit(
                fewshot_train_feature, fewshot_train_label
            )
            pred = clf.predict(fewshot_val_feature)
            
            acc_val = compute_metric(pred, fewshot_val_label, typ=metric_type)
            acc_list.append(acc_val)
        
        print(acc_list, flush=True)
        
        # binary search
        peak_idx = np.argmax(acc_list)
        c_peak = search_list[peak_idx]
        c_left, c_right = 1e-1 * c_peak, 1e1 * c_peak
        
        def binary_search(c_left, c_right, seed, step, test_acc_step_list):
            clf_left = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_left).fit(fewshot_train_feature, fewshot_train_label)
            pred_left = clf_left.predict(fewshot_val_feature)
            acc_left = compute_metric(pred_left, fewshot_val_label, typ=metric_type)
            print("Val accuracy (Left): {:.2f}".format(100 * acc_left), flush=True)

            clf_right = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_right).fit(fewshot_train_feature, fewshot_train_label)
            pred_right = clf_right.predict(fewshot_val_feature)
            acc_right = compute_metric(pred_right, fewshot_val_label, typ=metric_type)
            print("Val accuracy (Right): {:.2f}".format(100 * acc_right), flush=True)

            # find maximum and update ranges
            if acc_left < acc_right:
                c_final = c_right
                clf_final = clf_right
                # range for the next step
                c_left = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_right = np.log10(c_right)
            else:
                c_final = c_left
                clf_final = clf_left
                # range for the next step
                c_right = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_left = np.log10(c_left)

            pred = clf_final.predict(test_feature)
            test_acc = 100 * compute_metric(pred, test_label, typ=metric_type)
            print("Test Accuracy: {:.2f}".format(test_acc), flush=True)
            test_acc_step_list[seed - 1, step] = test_acc

            saveline = "{}, seed {}, {} shot, weight {}, test_acc {:.2f}\n".format(dataset, seed, num_shot, c_final, test_acc)
            with open(
                "{}/lp_s{}r{}_details.txt".format(dataset_path, args.num_step, args.num_run),
                "a+",
            ) as writer:
                writer.write(saveline)
            return (
                np.power(10, c_left),
                np.power(10, c_right),
                seed,
                step,
                test_acc_step_list,
            )

        for step in range(args.num_step):
            print(
                f"{dataset}, {num_shot} Shot, Round {step}: {c_left}/{c_right}",
                flush=True,
            )
            c_left, c_right, seed, step, test_acc_step_list = binary_search(c_left, c_right, seed, step, test_acc_step_list)
    # save results of last step
    test_acc_list = test_acc_step_list[:, -1]
    acc_mean = np.mean(test_acc_list)
    acc_std = np.std(test_acc_list)
    save_line = "{}, {} Shot; {}; Test acc stat: {:.2f} ({:.2f})\n".format(
        dataset, 
        num_shot, 
        ",".join(map(str, test_acc_list)),
        acc_mean, 
        acc_std
    )
    print(save_line, flush=True)
    with open(
        "{}/lp_s{}r{}.txt".format(dataset_path, args.num_step, args.num_run),
        "a+",
    ) as writer:
        writer.write(save_line)

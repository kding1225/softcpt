import itertools
import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import average_precision_score

from .build import EVALUATOR_REGISTRY


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def compute_mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return ap.mean()


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.2f}%\n"
            f"* error: {err:.2f}%\n"
            f"* macro_f1: {macro_f1:.2f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    "* class: {} ({})\t"
                    "total: {:,}\t"
                    "correct: {:,}\t"
                    "acc: {:.2f}%".format(
                        label, classname, total, correct, acc
                    )
                )
            mean_acc = np.mean(accs)
            print("* average: {:.2f}%".format(mean_acc))

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print('Confusion matrix is saved to "{}"'.format(save_path))

        return results


@EVALUATOR_REGISTRY.register()
class ClassificationMT(EvaluatorBase):
    """
    overall metric is the unweighted mean of metrics from all tasks
    """
    def __init__(self, cfg, lab2cname=None, task2tname=None, task2type=None, task2metric=None, **kwargs):
        """
        lab2cname: lab id to cname for per task
        task2tname: task id to task name
        task2type: taskid to task type
        task2metric: taskid to metric type
        """
        self.cfg = cfg
        self._lab2cname = lab2cname
        self._task2tname = task2tname
        self._task2type = task2type
        self._task2metric = task2metric
        self._y_true = []
        self._y_score = []
        
        self._num_classes = [len(x) for x in lab2cname]
        self._cls_range = self.compute_cls_range(self._num_classes)
        
    def compute_cls_range(self, num_classes):
        cls_range = []
        tt = 0
        for i, c in enumerate(num_classes):
            cls_range.append((tt, tt+c))
            tt += c
        return cls_range
    
    def reset(self):
        self._y_true = []
        self._y_score = []
        
    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        self._y_true.append(gt.cpu().numpy())
        self._y_score.append(self.compute_scores(mo))
        
    def compute_scores(self, logits):
        scores = logits.clone()
        for i, rg in enumerate(self._cls_range):
            st, ed = rg
            typ = self._task2type[i]
            scores[:, st:ed] = logits[:, st:ed].softmax(dim=-1)
        
        if scores.ndim == 3:
            scores = scores.mean(dim=0)
        
        return scores.cpu().numpy()
        
    def compute_metric(self, y_true, y_score, N, typ, metric):
        
        if metric == "overall_acc":
            assert typ in ["sl"]
            
            pred = y_score.argmax(1)
            gt = y_true.argmax(1)
            matches = (pred==gt).astype(float)
            correct = int(matches.sum())
            total = len(gt)
            metric = 100. * correct / total
            
        elif metric == "mean_acc":
            assert typ in ["sl"]
            
            y_pred = y_score.argmax(1)
            y_true = y_true.argmax(1)
            
            y = N * y_true + y_pred
            y = np.bincount(y, minlength=N*N)
            confusion_matrix = y.reshape(N, N)  # confusion matrix
            
            per_class_accs = np.diag(confusion_matrix)/confusion_matrix.sum(1)
            metric = 100. * np.mean(per_class_accs)

        else:
            raise NotImplementedError
            
        return metric

    def evaluate(self):
        results = OrderedDict()
        
        y_true = np.concatenate(self._y_true, axis=0)
        y_score = np.concatenate(self._y_score, axis=0)
        mask = y_true != -1
        
        metrics = []
        cnts = []
        cc = 0
        for i, num in enumerate(self._num_classes):
            cur_y_true = y_true[:, cc:cc+num]
            cur_y_score = y_score[:, cc:cc+num]
            cur_mask = mask[:, cc:cc+num]
            cur_type = self._task2type[i]
            cur_metric = self._task2metric[i]
            valid = np.sum(cur_mask, axis=1) > 0
            
            cur_y_true = cur_y_true[valid, :]
            cur_y_score = cur_y_score[valid, :]
            metric = self.compute_metric(cur_y_true, cur_y_score, num, cur_type, cur_metric)
            
            cc += num
            metrics.append(metric)
            cnts.append(valid.sum())
            
        metrics = np.array(metrics)
        cnts = np.array(cnts)
        
        metric = np.mean(metrics)  # overall metric
        results["metric"] = metric  # The first value will be returned by trainer.test()
        
        total = np.sum(cnts)
        print(
            "=> result\n"
            f"* total: {total:,}\n"
            f"* overall_metric: {metric:.2f}%"
        )
        
        # print per task metric
        if self.cfg.TEST.PER_TASK_RESULT:
            
            for i, metric in enumerate(metrics):
                print(
                    "* task: {} ({})\t"
                    "total: {:,}\t"
                    "metric: {:.2f}%".format(
                        i, self._task2tname[i], cnts[i], metric
                    )
                )
            
            mean_metric = np.mean(metrics)
            print("* average: {:.2f}%".format(mean_metric))
            results["pertask_metric"] = mean_metric  # unweighted
        
        return results
    
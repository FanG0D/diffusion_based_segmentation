# Author: Bingxin Ke
# Last modified: 2024-02-15


import pandas as pd
import torch
import numpy as np


# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)



def pixel_accuracy(output, target):
    """像素准确率"""
    with torch.no_grad():
        output = torch.argmax(output, dim=1)
        correct = torch.eq(output, target).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(output, target, num_classes):
    """平均交并比"""
    with torch.no_grad():
        output = torch.argmax(output, dim=1)
        intersection = torch.zeros(num_classes)
        union = torch.zeros(num_classes)
        
        for cls in range(num_classes):
            pred_mask = (output == cls)
            target_mask = (target == cls)
            intersection[cls] = (pred_mask & target_mask).sum()
            union[cls] = (pred_mask | target_mask).sum()
        
        iou = intersection / (union + 1e-6)  # 添加小值避免除零
        return iou.mean()


# def dice_coefficient(output, target, num_classes):
#     """Dice系数"""
#     with torch.no_grad():
#         output = torch.argmax(output, dim=1)
#         dice_scores = torch.zeros(num_classes)
        
#         for cls in range(num_classes):
#             pred_mask = (output == cls)
#             target_mask = (target == cls)
#             intersection = (pred_mask & target_mask).sum()
#             total = pred_mask.sum() + target_mask.sum()
#             dice_scores[cls] = 2.0 * intersection / (total + 1e-6)
            
#         return dice_scores.mean()


def precision_recall_f1(output, target, num_classes):
    """精确率、召回率和F1分数"""
    with torch.no_grad():
        output = torch.argmax(output, dim=1)
        precision = torch.zeros(num_classes)
        recall = torch.zeros(num_classes)
        
        for cls in range(num_classes):
            pred_mask = (output == cls)
            target_mask = (target == cls)
            
            tp = (pred_mask & target_mask).sum().float()
            fp = (pred_mask & ~target_mask).sum().float()
            fn = (~pred_mask & target_mask).sum().float()
            
            precision[cls] = tp / (tp + fp + 1e-6)
            recall[cls] = tp / (tp + fn + 1e-6)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        return precision.mean(), recall.mean(), f1.mean()


def boundary_iou(output, target, num_classes, boundary_width=1):
    """边界IoU"""
    with torch.no_grad():
        output = torch.argmax(output, dim=1)
        boundary_iou_scores = torch.zeros(num_classes)
        
        for cls in range(num_classes):
            pred_mask = (output == cls)
            target_mask = (target == cls)
            
            # 使用形态学操作找到边界
            from torch.nn.functional import max_pool2d
            pred_boundary = pred_mask ^ max_pool2d(pred_mask.float(), boundary_width*2+1, 
                                                 stride=1, padding=boundary_width)
            target_boundary = target_mask ^ max_pool2d(target_mask.float(), boundary_width*2+1,
                                                     stride=1, padding=boundary_width)
            
            intersection = (pred_boundary & target_boundary).sum()
            union = (pred_boundary | target_boundary).sum()
            boundary_iou_scores[cls] = intersection / (union + 1e-6)
            
        return boundary_iou_scores.mean()

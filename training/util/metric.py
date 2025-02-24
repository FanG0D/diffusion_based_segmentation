# Author: Bingxin Ke
# Last modified: 2024-02-15
import torch.nn.functional as F

import pandas as pd
import torch
import numpy as np
from torchvision.transforms.functional import resize


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


def _batch_miou_fscore(output, target, nclass, T, beta2=0.3):
    """batch mIoU and Fscore"""
    # output: [BF, C, H, W],
    # target: [BF, H, W]
    mini = 1
    maxi = nclass
    nbins = nclass
    # predict = torch.argmax(output, 1) + 1
    predict = output.float() + 1
    target = target.float() + 1
    # pdb.set_trace()
    predict = predict.float() * (target > 0).float() # [BF, H, W]
    intersection = predict * (predict == target).float() # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    batch_size = target.shape[0] // T
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    fscores = torch.zeros(nclass).float()

    # vid_miou_list = torch.zeros(target.shape[0]).float()
    vid_miou_list = []
    for i in range(target.shape[0]):
        area_inter = torch.histc(intersection[i].cpu(), bins=nbins, min=mini, max=maxi) # TP
        area_pred = torch.histc(predict[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FP
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FN
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        # iou[torch.isnan(iou)] = 1.
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1

        precision = area_inter / area_pred
        recall = area_inter / area_lab
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[torch.isnan(fscore)] = 0.
        fscores += fscore

        vid_miou_list.append(torch.sum(iou) / (torch.sum( iou != 0 ).float()))

    return ious, fscores, cls_count, vid_miou_list

def mIoU(output, target):

    # output: mask [H, W]    target: annotation [H, W] 

    with torch.no_grad():
        # print(target.shape)
        # print(output.shape)
        target_size = target.shape
        output = output.unsqueeze(0)
        target = target.unsqueeze(0)
        output = resize(
            output,
            target_size[-2:],
            antialias=True,
        )
        output = output.squeeze()
        target = target.squeeze()
        output = output.unsqueeze(0)
        target = target.unsqueeze(0)

        result = _batch_miou_fscore(output, target, 151, 1)
        batch_iou = result[0] / result[2]
        batch_iou[torch.isnan(batch_iou)] = 0
        batch_iou = torch.sum(batch_iou) / torch.sum(result[2] != 0)

        return batch_iou

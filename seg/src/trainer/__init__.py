# Author: Bingxin Ke
# Last modified: 2024-05-17

from .diff_based_seg_trainer import DiffBasedSegTrainer


trainer_cls_name_dict = {
    "DiffBasedSegTrainer": DiffBasedSegTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]

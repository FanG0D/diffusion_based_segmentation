# Author: Bingxin Ke
# Last modified: 2024-03-30

import io
import os
import random
import tarfile
from enum import Enum

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize


class DatasetMode(Enum):
    RGB_ONLY = "rgb_only"
    EVAL = "evaluate"
    TRAIN = "train"


def read_image_from_tar(tar_obj, img_rel_path):
    image = tar_obj.extractfile("./" + img_rel_path)
    image = image.read()
    image = Image.open(io.BytesIO(image))


class BaseSegDataset(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        dataset_dir: str,
        disp_name: str,
        name_mode,
        augmentation_args: dict = None,
        resize_to_hw=None,
        rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        # dataset info
        self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        self.disp_name = disp_name
        self.name_mode: SegFileNameMode = name_mode


        # training arguments
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw
        self.rgb_transform = rgb_transform

        # Load filenames
        with open(self.filename_ls_path, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ] 

        # Tar dataset
        self.tar_obj = None
        self.is_tar = (
            True
            if os.path.isfile(dataset_dir) and tarfile.is_tarfile(dataset_dir)
            else False
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        rgb_rel_path, seg_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Seg data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            seg_data = self._load_seg_data(
                seg_rel_path=seg_rel_path,
            )
            
            rasters.update(seg_data)

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other

    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        # whether to norminalize!!!!!
        outputs = {
            "rgb_int": torch.from_numpy(rgb).int(),
        }
        return outputs
    
    def _load_seg_data(self, seg_rel_path):
        # Read seg data
        outputs = {}
        seg_raw = self._read_seg_file(seg_rel_path)
        outputs["anotation_raw"] = seg_raw.clone()

        return outputs

    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        rgb_rel_path = filename_line[0]

        seg_rel_path = None
        if DatasetMode.RGB_ONLY != self.mode:
            seg_rel_path = filename_line[1]

        return rgb_rel_path, seg_rel_path


    def _read_image(self, img_rel_path) -> np.ndarray:
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            image = self.tar_obj.extractfile("./" + img_rel_path)
            image = image.read()
            image = Image.open(io.BytesIO(image))  # [H, W, rgb]
        else:
            img_path = os.path.join(self.dataset_dir, img_rel_path)
            image = Image.open(img_path)
        image = np.asarray(image)
        return image

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        rgb = self._read_image(rel_path)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
        return rgb
 
    def _read_seg_file(self, rel_path):
        seg_in = self._read_image(rel_path)
        seg_rgb = seg_in.astype(int) 
        seg_decoded = seg_rgb

        return seg_decoded


    def __del__(self):
        if self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None


# Prediction file naming modes
class SegFileNameMode(Enum):
    id = 1  # id.png
    rgb_id = 2  # rgb_id.png
    i_d_rgb = 3  # i_d_1_rgb.png
    rgb_i_d = 4


def get_pred_name(rgb_basename, name_mode, suffix=".png"):
    if SegFileNameMode.rgb_id == name_mode:
        pred_basename = "pred_" + rgb_basename.split("_")[1]
    elif SegFileNameMode.i_d_rgb == name_mode:
        pred_basename = rgb_basename.replace("_rgb.", "_pred.")
    elif SegFileNameMode.id == name_mode:
        pred_basename = "pred_" + rgb_basename
    elif SegFileNameMode.rgb_i_d == name_mode:
        pred_basename = "pred_" + "_".join(rgb_basename.split("_")[1:])
    else:
        raise NotImplementedError
    # change suffix
    pred_basename = os.path.splitext(pred_basename)[0] + suffix

    return pred_basename

# @GonzaloMartinGarcia
# This file houses our dataset mixer and training dataset classes.

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import random
import pandas as pd
import cv2

#################
# Dataset Mixer
#################

class MixedDataLoader:
    def __init__(self, loader1, loader2, split1=9, split2=1):
        self.loader1 = loader1
        self.loader2 = loader2
        self.split1 = split1
        self.split2 = split2
        self.frac1, self.frac2 = self.get_split_fractions()
        self.randchoice1=None

    def __iter__(self):
        self.loader_iter1 = iter(self.loader1)
        self.loader_iter2 = iter(self.loader2)
        self.randchoice1 = self.create_split()
        self.indx = 0
        return self
    
    def get_split_fractions(self):
        size1 = len(self.loader1)
        size2 = len(self.loader2)
        effective_fraction1 = min((size2/size1) * (self.split1/self.split2), 1) 
        effective_fraction2 = min((size1/size2) * (self.split2/self.split1), 1) 
        print("Effective fraction for loader1: ", effective_fraction1)
        print("Effective fraction for loader2: ", effective_fraction2)
        return effective_fraction1, effective_fraction2

    def create_split(self):
        randchoice1 = [True]*int(len(self.loader1)*self.frac1) + [False]*int(len(self.loader2)*self.frac2)
        np.random.shuffle(randchoice1)
        return randchoice1

    def __next__(self):
        if self.indx == len(self.randchoice1):
            raise StopIteration
        if self.randchoice1[self.indx]:
            self.indx += 1
            return next(self.loader_iter1)
        else:
            self.indx += 1
            return next(self.loader_iter2)
        
    def __len__(self):
        return int(len(self.loader1)*self.frac1) + int(len(self.loader2)*self.frac2)
    

#################
# Transforms 
#################

# ADE20K
class SynchronizedTransform_ADE20K:
    def __init__(self, H, W, crop_size=(512, 512), initial_size=(512, 1024)):
        self.H = H
        self.W = W
        self.crop_size = crop_size
        self.initial_size = initial_size  
        self.resize = transforms.Resize((H,W))
        self.initial_resize = transforms.Resize(initial_size)  
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.to_tensor = transforms.ToTensor()

    def random_crop(self, rgb_image, seg_img):

        w, h = rgb_image.size
        th, tw = self.crop_size            
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)   
        rgb_crop = rgb_image.crop((x1, y1, x1 + tw, y1 + th))
        seg_crop = seg_img.crop((x1, y1, x1 + tw, y1 + th))
        
        return rgb_crop, seg_crop
    
    def __call__(self, rgb_image, seg_img):
        # initial resize
        rgb_image = self.initial_resize(rgb_image)
        
        # seg_img one-hot resize
        seg_in_origin = np.array(seg_img)
        num_classes = 151
        h, w = seg_in_origin.shape
        one_hot = np.zeros((num_classes, h, w), dtype=np.float32)
        for c in range(num_classes):
            one_hot[c][seg_in_origin == c] = 1
        
        # resize one-hot
        resized_one_hot = np.array([
            cv2.resize(one_hot[c], self.initial_size, 
                    interpolation=cv2.INTER_NEAREST)
            for c in range(num_classes)
        ])
        seg_in_origin = np.argmax(resized_one_hot, axis=0)
        seg_img = Image.fromarray(seg_in_origin.astype(np.uint8))
        
        # random crop
        rgb_image, seg_img = self.random_crop(rgb_image, seg_img)
        annotation_in = seg_img
        seg_in_origin = np.array(seg_img)

        # palette
        color_map = self.ade_palette()
        color_map = np.array(color_map).astype(np.uint8)
        annotation_in.putpalette(color_map)
        seg_in_rgb_copy = annotation_in.convert(mode='RGB')
        seg_in_rgb = np.asarray(seg_in_rgb_copy)
        seg_in_rgb = np.transpose(seg_in_rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
        seg_in_rgb = seg_in_rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        # h-flip
        if random.random() > 0.5:
            rgb_image = self.horizontal_flip(rgb_image)
            seg_in_origin = np.flip(seg_in_origin, axis=1)
            seg_in_rgb = np.flip(seg_in_rgb, axis=2) 

        # to tensor
        rgb_tensor = self.to_tensor(rgb_image) * 2.0 - 1.0
        seg_in_origin = seg_in_origin.copy()
        annotation_tensor = torch.from_numpy(seg_in_origin)
        seg_in_rgb = seg_in_rgb.copy()
        seg_tensor = torch.from_numpy(seg_in_rgb)

        return rgb_tensor, seg_tensor, annotation_tensor 

    def ade_palette(self):
        """ADE20K palette for external use."""
        return  [[222, 222, 145], [18, 30, 7], [8, 23, 47], [30, 6, 96],
                 [1, 13, 164], [12, 28, 191], [25, 52, 32], [29, 48, 52],
                 [15, 51, 95], [25, 56, 167], [25, 42, 210], [27, 81, 31],
                 [9, 88, 54], [27, 92, 113], [11, 99, 151], [26, 110, 183],
                 [24, 130, 26], [4, 122, 75], [3, 132, 98], [26, 147, 167],
                 [17, 132, 197], [5, 169, 28], [19, 184, 67], [0, 190, 122],
                 [12, 167, 147], [6, 161, 196], [2, 205, 3], [5, 220, 61],
                 [23, 225, 107], [7, 217, 157], [25, 208, 191], [74, 10, 8],
                 [69, 30, 69], [56, 4, 98], [61, 29, 164], [60, 10, 194],
                 [60, 52, 19], [74, 69, 52], [65, 68, 116], [81, 41, 161],
                 [70, 60, 197], [66, 81, 14], [55, 107, 61], [76, 110, 108],
                 [74, 104, 162], [72, 94, 197], [60, 133, 16], [69, 128, 67],
                 [59, 148, 104], [65, 133, 154], [68, 128, 183], [79, 181, 11],
                 [76, 170, 56], [71, 175, 103], [53, 162, 137], [53, 182, 183],
                 [51, 229, 26], [51, 202, 51], [69, 213, 122], [63, 213, 161],
                 [71, 203, 197], [120, 11, 31], [124, 3, 68], [131, 2, 98],
                 [113, 1, 162], [102, 13, 209], [109, 50, 30], [126, 41, 47],
                 [107, 46, 118], [112, 49, 147], [109, 41, 189], [103, 83, 15],
                 [126, 99, 70], [124, 101, 104], [131, 103, 159],
                 [128, 110, 183], [119, 148, 9], [112, 137, 50], [123, 127, 116],
                 [107, 124, 167], [102, 148, 203], [124, 180, 15],
                 [116, 168, 65], [104, 182, 102], [111, 164, 163],
                 [105, 174, 191], [102, 218, 20], [126, 203, 64],
                 [108, 215, 109], [110, 221, 157], [107, 230, 192],
                 [160, 25, 11], [165, 12, 65], [153, 2, 117], [182, 21, 141],
                 [160, 19, 188], [176, 58, 19], [175, 58, 56], [170, 69, 93],
                 [176, 42, 146], [157, 44, 211], [157, 105, 2], [180, 98, 73],
                 [182, 85, 92], [169, 93, 152], [156, 89, 202], [157, 144, 22],
                 [180, 151, 77], [154, 146, 118], [162, 136, 143],
                 [171, 134, 184], [170, 174, 15], [178, 180, 65],
                 [176, 183, 120], [175, 169, 147], [181, 165, 197],
                 [156, 227, 3], [167, 218, 61], [160, 216, 119],
                 [164, 251, 141], [177, 201, 251], [231, 30, 13], [219, 6, 59],
                 [211, 26, 122], [216, 16, 153], [209, 12, 192], [216, 70, 15],
                 [215, 46, 60], [234, 61, 112], [224, 53, 157], [227, 49, 207],
                 [221, 108, 8], [220, 93, 73], [230, 111, 113], [218, 89, 143],
                 [231, 90, 195], [227, 144, 22], [208, 137, 49], [210, 128, 116],
                 [225, 135, 157], [221, 135, 193], [211, 174, 18],
                 [222, 185, 50], [229, 183, 93], [233, 162, 155],
                 [255, 167, 205], [211, 215, 15], [232, 225, 71], [0, 0, 0],
                 [255, 255, 255], [215, 216, 196]]
    
#####################
# Training Datasets
#####################

class ADE20K(Dataset):
    def __init__(self, root_dir, transform=True):
        self.root_dir = root_dir
        self.transform = SynchronizedTransform_ADE20K(H=480, W=480) if transform else None
        self.pairs = self._find_pairs()

    def _find_pairs(self):
        # root_dir/
        # ade20k_train/images    # RGB images
        # ade20k_train/annotations    # Segmentation masks
        image_dir = os.path.join(self.root_dir,"images")
        annot_dir = os.path.join(self.root_dir,"annotations")
        pairs = []
        
        for img_name in os.listdir(image_dir):
            if img_name.endswith('.jpg'):
                ann_name = img_name.replace('.jpg', '.png')
                img_path = os.path.join(image_dir, img_name)
                ann_path = os.path.join(annot_dir, ann_name)
                
                if os.path.exists(img_path) and os.path.exists(ann_path):
                    pairs.append({
                        'rgb_path': img_path,
                        'seg_path': ann_path
                    })
        return pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pairs = self.pairs[idx]
        
        rgb_path = pairs['rgb_path']
        rgb_image = Image.open(rgb_path).convert('RGB')
        
        seg_path = pairs['seg_path']
        seg_image = Image.open(seg_path)

        # transfrom
        if self.transform is not None:
            rgb_tensor, seg_tensor, annotation_tensor = self.transform(rgb_image, seg_image)
            return {
                "rgb": rgb_tensor.float(),
                "seg": seg_tensor.float(),
                "annotation": annotation_tensor.long()  
            }
        else:
            rgb_tensor    = transforms.ToTensor()(rgb_image) * 2.0 - 1.0
            annotation_tensor  = torch.from_numpy(np.array(seg_image))#.squeeze(0)
            return {
                "rgb": rgb_tensor.float(),
                "annotation": annotation_tensor.long()  
            }
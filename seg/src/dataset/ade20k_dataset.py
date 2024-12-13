import torch
from PIL import Image
import numpy as np

from .base_seg_dataset import BaseSegDataset, SegFileNameMode


class ADE20KDataset(BaseSegDataset):
    def __init__(
        self,
        # kitti_bm_crop,  # Crop to ADE20K benchmark size
        # valid_mask_crop,  # Evaluation mask. [None, garg or eigen]
        **kwargs,
    ) -> None:
        super().__init__(
            # ADE20K data parameter
            min_depth=1e-5,
            max_depth=80,
            has_filled_depth=False,
            name_mode=SegFileNameMode.id,
            **kwargs,
        )
        # self.kitti_bm_crop = kitti_bm_crop
        # self.valid_mask_crop = valid_mask_crop
        # assert self.valid_mask_crop in [
        #     None,
        #     "garg",  # set evaluation mask according to Garg  ECCV16
        #     "eigen",  # set evaluation mask according to Eigen NIPS14
        # ], f"Unknown crop type: {self.valid_mask_crop}"

        # Filter out empty annotation
        self.filenames = [f for f in self.filenames if "None" != f[1]]

    def _read_seg_file(self, rel_path):
        annotation_in = super()._read_image(rel_path)
        # shangse
        color_map = self.ade_palette()
        color_map = np.array(color_map).astype(np.uint8)
        seg_in = annotation_in.putpalette(color_map)
        seg_in.convert('RGB')
        seg_in = np.asarray(seg_in)
        return seg_in

    def _load_rgb_data(self, rgb_rel_path):
        rgb_data = super()._load_rgb_data(rgb_rel_path)
        return rgb_data
    
    def _load_seg_data(self, seg_rel_path):
        seg_raw = self._read_seg_file(seg_rel_path)
        seg_raw = np.transpose(seg_raw, (2, 0, 1)).astype(int)  # [rgb, H, W]
        seg_norm = seg_raw / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        outputs = {
            "seg_int": torch.from_numpy(seg_raw).int(),
            "seg_norm": torch.from_numpy(seg_norm).float(),
        }

        return outputs

    def ade_palette():
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
    

    

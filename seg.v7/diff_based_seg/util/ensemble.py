
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch

from .image_util import get_tv_resample_method, resize_max_res


def inter_distances(tensors: torch.Tensor):
    """
    To calculate the distance between each two seg maps.
    """
    distances = []
    for i, j in torch.combinations(torch.arange(tensors.shape[0])):
        arr1 = tensors[i : i + 1]
        arr2 = tensors[j : j + 1]
        distances.append(arr1 - arr2)
    dist = torch.concatenate(distances, dim=0)
    return dist


def ensemble_seg(
    seg: torch.Tensor,
    # scale_invariant: bool = True,
    # shift_invariant: bool = True,
    output_uncertainty: bool = False,
    reduction: str = "median",
    regularizer_strength: float = 0.02,
    max_iter: int = 2,
    tol: float = 1e-3,
    max_res: int = 1024,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:


    if seg.dim() != 4 or seg.shape[1] != 3:
        raise ValueError(f"Expecting 4D tensor of shape [B,3,H,W]; got {seg.shape}.")
    if reduction not in ("mean", "median"):
        raise ValueError(f"Unrecognized reduction method: {reduction}.")


    def ensemble(
        seg_aligned: torch.Tensor, return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        uncertainty = None
        if reduction == "mean":
            prediction = torch.mean(seg_aligned, dim=0, keepdim=True)
            if return_uncertainty:
                uncertainty = torch.std(seg_aligned, dim=0, keepdim=True)
        elif reduction == "median":
            prediction = torch.median(seg_aligned, dim=0, keepdim=True).values
            if return_uncertainty:
                uncertainty = torch.median(
                    torch.abs(seg_aligned - prediction), dim=0, keepdim=True
                ).values
        else:
            raise ValueError(f"Unrecognized reduction method: {reduction}.")
        return prediction, uncertainty


    maskiage, uncertainty = ensemble(seg, return_uncertainty=output_uncertainty)

    return maskiage, uncertainty  # [1,3,H,W], [1,3,H,W]
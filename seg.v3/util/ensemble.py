# ����Ԥ��֮����ڽ�����м���



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


    if seg.dim() != 4 or seg.shape[1] != 1:
        raise ValueError(f"Expecting 4D tensor of shape [B,1,H,W]; got {seg.shape}.")
    if reduction not in ("mean", "median"):
        raise ValueError(f"Unrecognized reduction method: {reduction}.")

    # def init_param(depth: torch.Tensor):
    #     init_min = depth.reshape(ensemble_size, -1).min(dim=1).values
    #     init_max = depth.reshape(ensemble_size, -1).max(dim=1).values

    #     if scale_invariant and shift_invariant:
    #         init_s = 1.0 / (init_max - init_min).clamp(min=1e-6)
    #         init_t = -init_s * init_min
    #         param = torch.cat((init_s, init_t)).cpu().numpy()
    #     elif scale_invariant:
    #         init_s = 1.0 / init_max.clamp(min=1e-6)
    #         param = init_s.cpu().numpy()
    #     else:
    #         raise ValueError("Unrecognized alignment.")

    #     return param

    # def align(depth: torch.Tensor, param: np.ndarray) -> torch.Tensor:
    #     if scale_invariant and shift_invariant:
    #         s, t = np.split(param, 2)
    #         s = torch.from_numpy(s).to(depth).view(ensemble_size, 1, 1, 1)
    #         t = torch.from_numpy(t).to(depth).view(ensemble_size, 1, 1, 1)
    #         out = depth * s + t
    #     elif scale_invariant:
    #         s = torch.from_numpy(param).to(depth).view(ensemble_size, 1, 1, 1)
    #         out = depth * s
    #     else:
    #         raise ValueError("Unrecognized alignment.")
    #     return out

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

    # def cost_fn(param: np.ndarray, depth: torch.Tensor) -> float:
    #     cost = 0.0
    #     depth_aligned = align(depth, param)

    #     for i, j in torch.combinations(torch.arange(ensemble_size)):
    #         diff = depth_aligned[i] - depth_aligned[j]
    #         cost += (diff**2).mean().sqrt().item()

    #     if regularizer_strength > 0:
    #         prediction, _ = ensemble(depth_aligned, return_uncertainty=False)
    #         err_near = (0.0 - prediction.min()).abs().item()
    #         err_far = (1.0 - prediction.max()).abs().item()
    #         cost += (err_near + err_far) * regularizer_strength

    #     return cost

    # def compute_param(depth: torch.Tensor):
    #     import scipy

    #     depth_to_align = depth.to(torch.float32)
    #     if max_res is not None and max(depth_to_align.shape[2:]) > max_res:
    #         depth_to_align = resize_max_res(
    #             depth_to_align, max_res, get_tv_resample_method("nearest-exact")
    #         )

    #     param = init_param(depth_to_align)

    #     res = scipy.optimize.minimize(
    #         partial(cost_fn, depth=depth_to_align),
    #         param,
    #         method="BFGS",
    #         tol=tol,
    #         options={"maxiter": max_iter, "disp": False},
    #     )

    #     return res.x

    # requires_aligning = scale_invariant or shift_invariant
    # ensemble_size = seg.shape[0]

    # if requires_aligning:
    #     param = compute_param(depth)
    #     depth = align(depth, param)

    maskiage, uncertainty = ensemble(seg, return_uncertainty=output_uncertainty)

    return maskiage, uncertainty  # [1,3,H,W], [1,3,H,W]
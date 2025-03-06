import pandas as pd
import torch

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


def mIoU(pred, target):
    """
    Calculate mean IoU only for classes present in the target
    
    Args:
        pred (torch.Tensor): Prediction tensor, shape [B, H, W] or [H, W]
        target (torch.Tensor): Ground truth tensor, shape [B, H, W] or [H, W]
        
    Returns:
        float: Mean IoU of classes present in target
    """
    # Ensure tensors are on the same device
    pred = pred.to(target.device)
    
    # Ensure correct input dimensions
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
        
    # Get classes that are actually present in target (excluding background class 0)
    present_classes = torch.unique(target)
    present_classes = present_classes[present_classes != 0]
    
    if present_classes.numel() == 0:
        return torch.tensor(0.0, device=target.device)
    
    # Calculate IoU for each class
    ious = []
    for cls in present_classes:
        # Create binary masks
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        # Calculate intersection and union
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        # Calculate IoU for current class
        iou = intersection / (union + 1e-8)  # Add small value to prevent division by zero
        ious.append(iou)
    
    # Convert to tensor and calculate mean
    mean_iou = torch.stack(ious).mean()
    
    return mean_iou

import torch
import torch.nn as nn

#########
# Losses
#########
    
class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=0, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, pred, seg_gt):
        # pred: [N, 151, H, W]
        # seg_gt:  [N, H, W]
        if seg_gt.dim() == 4:
            seg_gt = seg_gt.squeeze(1)
        
        # if pred.dtype != torch.float32:
        #     pred = pred.float()  
        seg_gt = seg_gt.long()

        with torch.autocast(device_type='cuda', enabled=False):
            # Adjust pred size to match seg_gt size
            if pred.size(2) != seg_gt.size(1) or pred.size(3) != seg_gt.size(2):
                pred = nn.functional.interpolate(pred, size=(seg_gt.size(1), seg_gt.size(2)), mode='bilinear', align_corners=False)
            
            # Compute the loss
            loss = self.loss_fn(pred, seg_gt)
        return loss
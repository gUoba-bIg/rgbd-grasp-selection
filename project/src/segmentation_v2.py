"""Isolated center-regression diagnostic loss; previous experiments stay frozen.

Gaussian center regression is motivated by Panoptic-DeepLab's public design:
https://github.com/facebookresearch/detectron2/tree/main/projects/Panoptic-DeepLab
This variant keeps the pilot's sigmoid heatmap, normalized offsets, architecture,
and decoder. It is not a reproduction of that method or a new research claim.
"""

import torch.nn.functional as F


def segmentation_loss(outputs, targets):
    foreground = targets["foreground"].float()
    logits = outputs["foreground"].float()
    probabilities = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, foreground)
    dice = 1 - (2 * (probabilities * foreground).sum() + 1) / (
        probabilities.sum() + foreground.sum() + 1)
    centers = F.mse_loss(outputs["centers"].float().sigmoid(), targets["centers"].float())
    per_pixel = F.smooth_l1_loss(outputs["offsets"].float(), targets["offsets"].float(),
                               reduction="none", beta=0.05)
    offsets = (per_pixel * foreground).sum() / (2 * foreground.sum()).clamp(min=1)
    total = bce + dice + 200 * centers + 5 * offsets
    return total, {"foreground_bce": bce.detach(), "foreground_dice_loss": dice.detach(),
                   "center_mse": centers.detach(), "center_weighted": (200 * centers).detach(),
                   "offset_smooth_l1": offsets.detach()}

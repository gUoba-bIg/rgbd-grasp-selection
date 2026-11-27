"""Compact category-agnostic RGB-D instance segmentation pilot, no pretrained weights."""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


def group_norm(channels):
    return nn.GroupNorm(8, channels)


class PilotInstanceNet(nn.Module):
    """ResNet18 with GroupNorm, 48-channel FPN and foreground/center/offset heads.

    The depth and validity maps use early fusion. This is a deliberately separate
    pilot architecture, not the proposed quality-aware two-stream final method.
    """

    def __init__(self, mode="rgbd", fpn_channels=48):
        super().__init__()
        if mode not in ("rgb", "rgbd"):
            raise ValueError("Unsupported input mode")
        encoder = resnet18(weights=None, norm_layer=group_norm)
        if mode == "rgbd":
            encoder.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(encoder.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.stages = nn.ModuleList([encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4])
        self.laterals = nn.ModuleList([nn.Conv2d(channels, fpn_channels, 1) for channels in (64, 128, 256, 512)])
        self.refine = nn.Sequential(nn.Conv2d(fpn_channels, 32, 3, padding=1), group_norm(32), nn.ReLU())
        self.head = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), group_norm(32), nn.ReLU())
        self.foreground = nn.Conv2d(32, 1, 1)
        self.centers = nn.Conv2d(32, 1, 1)
        self.offsets = nn.Conv2d(32, 2, 1)
        nn.init.constant_(self.centers.bias, -2.19)

    def forward(self, inputs):
        size = inputs.shape[-2:]
        features = []
        hidden = self.stem(inputs)
        for stage in self.stages:
            hidden = stage(hidden)
            features.append(hidden)
        pyramid = self.laterals[-1](features[-1])
        for index in range(2, -1, -1):
            pyramid = self.laterals[index](features[index]) + F.interpolate(pyramid, size=features[index].shape[-2:], mode="nearest")
        hidden = self.head(F.interpolate(self.refine(pyramid), size=size, mode="nearest"))
        return {"foreground": self.foreground(hidden), "centers": self.centers(hidden), "offsets": self.offsets(hidden)}


def segmentation_loss(outputs, targets):
    """BCE + Dice foreground, Gaussian center focal loss, foreground-only offsets."""
    fg = targets["foreground"].float()
    prediction = outputs["foreground"].float()
    probabilities = prediction.sigmoid()
    bce = F.binary_cross_entropy_with_logits(prediction, fg)
    dice = 1 - (2 * (probabilities * fg).sum() + 1) / (probabilities.sum() + fg.sum() + 1)
    heatmap = targets["centers"].float()
    center_probability = outputs["centers"].float().sigmoid().clamp(1e-5, 1 - 1e-5)
    positive = (heatmap == 1).float()
    negative = (heatmap < 1).float()
    positive_loss = -(center_probability.log() * (1 - center_probability).pow(2) * positive).sum()
    negative_loss = -((1 - center_probability).log() * center_probability.pow(2) * (1 - heatmap).pow(4) * negative).sum()
    centers = (positive_loss + negative_loss) / positive.sum().clamp(min=1)
    per_pixel = F.smooth_l1_loss(outputs["offsets"].float(), targets["offsets"].float(), reduction="none", beta=0.05)
    offsets = (per_pixel * fg).sum() / (2 * fg.sum()).clamp(min=1)
    total = bce + dice + centers + 5 * offsets
    return total, {"foreground_bce": bce.detach(), "foreground_dice_loss": dice.detach(),
                   "center_focal": centers.detach(), "offset_smooth_l1": offsets.detach()}


def decode_instances(outputs, foreground_threshold=0.5, center_threshold=0.35,
                     min_pixels=20, max_centers=64):
    """Decode one image without ground-truth labels or ground-truth object counts."""
    foreground = outputs["foreground"][0, 0].float().sigmoid().detach().cpu()
    centers = outputs["centers"][0, 0].float().sigmoid().detach().cpu()
    offsets = outputs["offsets"][0].float().detach().cpu().numpy()
    height, width = foreground.shape
    instance_map = np.zeros((height, width), dtype=np.int32)
    peaks = (centers == F.max_pool2d(centers[None, None], 9, stride=1, padding=4)[0, 0]) & (centers >= center_threshold)
    coordinates = np.argwhere(peaks.numpy())
    if not len(coordinates):
        return instance_map
    scores = centers.numpy()[coordinates[:, 0], coordinates[:, 1]]
    order = np.argsort(-scores, kind="stable")
    selected = []
    # Plateaus can produce adjacent peaks. Deterministic suppression merges them.
    for index in order:
        candidate = coordinates[index]
        if all(np.linalg.norm(candidate - other) >= 5 for other in selected):
            selected.append(candidate)
            if len(selected) == max_centers:
                break
    coordinates = np.asarray(selected)
    yy, xx = np.nonzero(foreground.numpy() >= foreground_threshold)
    if not len(yy):
        return instance_map
    destinations = np.stack([yy + offsets[1, yy, xx] * height,
                             xx + offsets[0, yy, xx] * width], axis=1)
    # Bound temporary allocations for larger images.
    nearest = np.empty(len(yy), dtype=np.int32)
    for start in range(0, len(yy), 4096):
        distance = ((destinations[start:start + 4096, None] - coordinates[None]) ** 2).sum(axis=2)
        nearest[start:start + 4096] = distance.argmin(axis=1)
    label = 1
    for center_index in range(len(coordinates)):
        mask = nearest == center_index
        if int(mask.sum()) >= min_pixels:
            instance_map[yy[mask], xx[mask]] = label
            label += 1
    return instance_map


def instance_counts(predicted, actual, iou_threshold=0.5):
    """Maximum-cardinality IoU-threshold matching; returns TP/FP/FN, not mask AP."""
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    pred_ids = np.unique(predicted[predicted > 0])
    true_ids = np.unique(actual[actual > 0])
    overlaps = np.zeros((len(pred_ids), len(true_ids)), dtype=np.float64)
    for i, pred_id in enumerate(pred_ids):
        pred_mask = predicted == pred_id
        for j, true_id in enumerate(true_ids):
            true_mask = actual == true_id
            intersection = int((pred_mask & true_mask).sum())
            union = int((pred_mask | true_mask).sum())
            overlaps[i, j] = intersection / max(union, 1)
    matches = 0
    if overlaps.size:
        valid = overlaps >= iou_threshold
        # A valid-match bonus exceeds the entire possible secondary IoU reward.
        bonus = min(overlaps.shape) + 1
        rows, columns = linear_sum_assignment(-(valid.astype(np.float64) * bonus + overlaps))
        matches = int(valid[rows, columns].sum())
    return {"tp": matches, "fp": len(pred_ids) - matches, "fn": len(true_ids) - matches}


def summarize_counts(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return dict(counts, precision_iou50=tp / (tp + fp) if tp + fp else 0.0,
                recall_iou50=tp / (tp + fn) if tp + fn else 0.0,
                f1_iou50=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)

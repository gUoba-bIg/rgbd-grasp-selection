"""Unfrozen 480x640 dual-encoder prototype; no research experiment is implied.

The fixed-fusion and quality-gated arms have identical trainable parameters.
Quality supervision and the gate-strength schedule are intentionally not defined
here. Extra output keys require a new dedicated training/evaluation entry point.
"""
import copy
import math
import numbers

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


TARGET_INPUT_SIZE = (480, 640)
FPN_CHANNELS = 128
STAGE_CHANNELS = (64, 128, 256, 512)
ARMS = ('fixed', 'quality')


def group_norm(channels):
    return nn.GroupNorm(8, channels)


class SeparateEncoder(nn.Module):
    """A scratch ResNet18 encoder without its unused classification layer."""
    def __init__(self, input_channels):
        super().__init__()
        if input_channels not in (2, 3):
            raise ValueError('Encoder requires RGB or depth/validity input channels')
        encoder = resnet18(weights=None, norm_layer=group_norm)
        if input_channels == 2:
            encoder.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # A zero independent depth stream can be blocked by normalization/ReLU.
            nn.init.kaiming_normal_(encoder.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.stages = nn.ModuleList((encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4))

    def forward(self, inputs):
        hidden = self.stem(inputs)
        features = []
        for stage in self.stages:
            hidden = stage(hidden)
            features.append(hidden)
        return features


class QualityHead(nn.Module):
    """Predict one bounded quality map from detached features at one scale."""
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(2*channels, 32, 1), group_norm(32), nn.ReLU(), nn.Conv2d(32, 1, 1))

    def forward(self, rgb_feature, depth_feature):
        detached = torch.cat((rgb_feature.detach(), depth_feature.detach()), dim=1)
        return self.layers(detached).sigmoid()


class DualStreamInstanceNet(nn.Module):
    """Independent RGB/depth encoders, four additive fusion levels, 128-channel FPN.

    Inputs are preprocessed N x 5 x H x W tensors, with RGB in the first three
    channels and normalized depth/validity in the last two. The target research
    resolution is 480 x 640; smaller tensors support component tests only.

    For the quality arm, depth is multiplied by 1-s*(1-q.detach()). At s=0 both
    arms therefore use exactly the same arithmetic and initial weights. The
    fixed arm always uses a unit gate while still computing its quality heads.
    Both arms require an explicit keyword gate_strength; omitted values cannot
    silently disable quality gating in a future inference entry point.
    No gate schedule, quality target, auxiliary loss or training budget is set.
    """
    def __init__(self, arm='fixed', fpn_channels=FPN_CHANNELS):
        super().__init__()
        if arm not in ARMS or fpn_channels != FPN_CHANNELS:
            raise ValueError('Prototype requires fixed/quality arm and 128 FPN channels')
        self.arm = arm
        self.fpn_channels = fpn_channels
        self.rgb_encoder = SeparateEncoder(3)
        self.depth_encoder = SeparateEncoder(2)
        self.rgb_laterals = nn.ModuleList(nn.Conv2d(channels, fpn_channels, 1) for channels in STAGE_CHANNELS)
        self.depth_laterals = nn.ModuleList(nn.Conv2d(channels, fpn_channels, 1) for channels in STAGE_CHANNELS)
        self.quality_heads = nn.ModuleList(QualityHead(channels) for channels in STAGE_CHANNELS)
        self.refine = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                                    group_norm(fpn_channels), nn.ReLU())
        self.head = nn.Sequential(nn.Conv2d(fpn_channels, 64, 3, padding=1), group_norm(64), nn.ReLU())
        self.foreground = nn.Conv2d(64, 1, 1)
        self.centers = nn.Conv2d(64, 1, 1)
        self.offsets = nn.Conv2d(64, 2, 1)
        nn.init.constant_(self.centers.bias, -2.19)

    @staticmethod
    def checked_strength(gate_strength):
        if (isinstance(gate_strength, bool) or not isinstance(gate_strength, numbers.Real)
                or not math.isfinite(gate_strength) or not 0 <= gate_strength <= 1):
            raise ValueError('Gate strength must be a finite scalar in [0, 1]')
        return float(gate_strength)

    def forward(self, inputs, *, gate_strength):
        strength = self.checked_strength(gate_strength)
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 4 or inputs.shape[0] < 1
                or inputs.shape[1] != 5 or min(inputs.shape[-2:]) < 32 or not inputs.is_floating_point()):
            raise ValueError('Expected floating N x 5 x H x W input with H,W at least 32')
        rgb_features = self.rgb_encoder(inputs[:, :3])
        depth_features = self.depth_encoder(inputs[:, 3:])
        fused, quality_maps, gates = [], [], []
        effective_strength = strength if self.arm == 'quality' else 0.0
        for index, (rgb_feature, depth_feature) in enumerate(zip(rgb_features, depth_features)):
            quality = self.quality_heads[index](rgb_feature, depth_feature)
            # Both arms execute this same expression; no segmentation gradient reaches quality.
            gate = 1.0-effective_strength*(1.0-quality.detach())
            fused.append(self.rgb_laterals[index](rgb_feature)+gate*self.depth_laterals[index](depth_feature))
            quality_maps.append(quality)
            gates.append(gate)
        pyramid = fused[-1]
        for index in range(2, -1, -1):
            pyramid = fused[index]+F.interpolate(pyramid, size=fused[index].shape[-2:], mode='nearest')
        hidden = self.refine(pyramid)
        hidden = self.head(F.interpolate(hidden, size=inputs.shape[-2:], mode='nearest'))
        return {'foreground': self.foreground(hidden), 'centers': self.centers(hidden), 'offsets': self.offsets(hidden),
                'quality_maps': tuple(quality_maps), 'effective_gates': tuple(gates)}


def make_paired_models(seed=20261022):
    """Copy one scratch CPU reference without consuming the caller's CPU RNG.

    No CUDA generator is inspected or seeded. A future trainer owns its device
    placement, post-construction RNG reset and complete checkpoint contract.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Expected a nonnegative integer initialization seed')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        fixed = DualStreamInstanceNet(arm='fixed')
        quality = copy.deepcopy(fixed)
        quality.arm = 'quality'
    return fixed, quality

"""v37 identity-preserving quality-conditioned FPN residual; frozen parents unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from src.segmentation import PilotInstanceNet
from src.p5_quality_v23 import QualityNet


class QualityResidualV37(nn.Module):
    """Same-capacity paired arms. Quality never rescales metric depth or validity."""
    def __init__(self, arm, residual_scale=0.25):
        super().__init__()
        if arm not in ('fixed', 'quality'):
            raise ValueError('Unknown paired arm')
        self.arm = arm
        self.ablation = 'learned'
        self.residual_scale = float(residual_scale)
        self.backbone = PilotInstanceNet('rgbd')
        self.quality = QualityNet()
        self.adapter = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1),
                                     nn.GroupNorm(4, 16), nn.SiLU(),
                                     nn.Conv2d(16, 48, 1))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.quality.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.quality.eval()
        return self

    def forward(self, inputs):
        with torch.no_grad():
            quality = self.quality(inputs)
            reliability = (quality['valid_logits'].sigmoid()
                           * torch.exp(-quality['log_sigma'].exp() / .005)
                           * (1 - .5 * quality['anomaly_logits'].sigmoid()))
        if self.arm == 'fixed' or self.ablation == 'unit':
            reliability = torch.ones_like(reliability)
        elif self.ablation == 'mean':
            valid = inputs[:, 4:5]
            value = ((reliability * valid).sum((2, 3), keepdim=True)
                     / valid.sum((2, 3), keepdim=True).clamp_min(1))
            reliability = value.expand_as(reliability)
        elif self.ablation != 'learned':
            raise ValueError('Unknown quality ablation')
        # The exact original five-channel tensor reaches the existing encoder.
        hidden = self.backbone.stem(inputs)
        features = []
        for stage in self.backbone.stages:
            hidden = stage(hidden)
            features.append(hidden)
        pyramid = self.backbone.laterals[-1](features[-1])
        for index in range(2, -1, -1):
            pyramid = (self.backbone.laterals[index](features[index])
                       + F.interpolate(pyramid, size=features[index].shape[-2:], mode='nearest'))
        gate = F.interpolate(reliability, size=pyramid.shape[-2:], mode='area').to(pyramid.dtype)
        # tanh bounds the added feature correction, not physical geometry.
        correction = self.residual_scale * torch.tanh(self.adapter(pyramid)) * gate
        pyramid = pyramid + correction
        hidden = self.backbone.head(F.interpolate(self.backbone.refine(pyramid),
                                                  size=inputs.shape[-2:], mode='nearest'))
        return {'foreground': self.backbone.foreground(hidden),
                'centers': self.backbone.centers(hidden),
                'offsets': self.backbone.offsets(hidden),
                'quality': quality, 'effective_gate': reliability,
                'residual_max_abs': correction.detach().abs().amax()}

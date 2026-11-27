"""Deterministic depth degradation and auxiliary targets for a paired study.

The target measures availability and fidelity to the preprocessed input
reference. It is not calibrated sensor accuracy or a grasp-success probability.
"""
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from src.graspnet_frames import GraspNetFrames


SEED = 20261022
CONDITIONS = ('clean', 'noise', 'missing', 'joint')
NOISE_SIGMA_M = 0.02
QUALITY_DECAY_M = 0.02
DEPTH_CLIP_M = 2.0
QUALITY_LOSS_WEIGHT = 1.0


def _integer(value, label, low=0, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError('Invalid '+label)
    return value


def gate_strength_for_update(step):
    """One-based successful-update schedule; retries keep the same step."""
    _integer(step, 'update', 1, 800)
    return min(1.0, max(0.0, (step-80)/160.0))


def corruption_condition(index, epoch):
    """Canonical frame index, not its shuffled minibatch position."""
    _integer(index, 'sample index', 0, 319)
    _integer(epoch, 'epoch', 0, 9)
    return CONDITIONS[(index+epoch) % len(CONDITIONS)]


def _record_number(value, label):
    # Existing manifests can represent frame IDs as zero-padded decimal strings.
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    return _integer(value, label)


def corruption_metadata(index, epoch, record=None, condition=None):
    """Pure expected identity used by training-history verification."""
    _integer(index, 'sample index', 0, 319)
    _integer(epoch, 'epoch', 0, 9)
    selected = corruption_condition(index, epoch) if condition is None else condition
    if selected not in CONDITIONS:
        raise ValueError('Unknown depth condition')
    if record is None:
        scene, frame = 0, index
    else:
        scene = _record_number(record['scene'], 'scene')
        frame = _record_number(record['frame'], 'frame')
    identity = 'R18|{}|{}|{}|{}|{}'.format(SEED, epoch, scene, frame, selected)
    seed = int(hashlib.sha256(identity.encode('ascii')).hexdigest()[:8], 16)
    return {'condition': selected, 'seed': seed, 'index': index, 'epoch': epoch,
            'scene': scene, 'frame': frame}


def corrupt_depth(inputs, index, epoch, *, record=None, condition=None):
    """Return CPU five-channel input, one-channel target and scalar metadata.

    Input depth is already clipped to two metres and divided by two. Original
    invalidity comes from the uncorrupted validity channel. The target is never
    appended to the model input. Smaller images support synthetic CPU tests;
    their rectangular hole is also exactly floor(H/4) by floor(W/4).
    """
    if (not isinstance(inputs, torch.Tensor) or inputs.device.type != 'cpu'
            or inputs.dtype != torch.float32 or inputs.requires_grad
            or inputs.ndim != 3 or inputs.shape[0] != 5 or min(inputs.shape[-2:]) < 4
            or not bool(torch.isfinite(inputs).all())):
        raise ValueError('Expected finite CPU float32 input [5,H,W] without gradients')
    clean = inputs.detach().numpy()
    validity = clean[4]
    if not bool(np.all((validity == 0) | (validity == 1))):
        raise ValueError('Clean validity must be binary')
    valid = validity == 1
    if (not bool(np.all((clean[3] >= 0) & (clean[3] <= 1)))
            or np.any(clean[3][~valid] != 0) or np.any(clean[3][valid] <= 0)):
        raise ValueError('Clean depth and validity are inconsistent')
    metadata = corruption_metadata(index, epoch, record, condition)
    selected = metadata['condition']
    rng = np.random.RandomState(metadata['seed'])
    height, width = clean.shape[-2:]
    reference = clean[3]*np.float32(DEPTH_CLIP_M)
    altered = reference.copy()
    noise_zero = np.zeros_like(valid)
    noise_upper = np.zeros_like(valid)
    if selected in ('noise', 'joint'):
        noise = rng.normal(0.0, NOISE_SIGMA_M, size=(height, width)).astype(np.float32)
        candidate = reference+noise
        noise_zero = valid & (candidate <= 0)
        noise_upper = valid & (candidate > DEPTH_CLIP_M)
        altered = np.where(valid, np.clip(candidate, 0.0, DEPTH_CLIP_M), 0).astype(np.float32)
    added_missing = np.zeros_like(valid)
    rectangle = None
    if selected in ('missing', 'joint'):
        block_h, block_w = height//4, width//4
        top = int(rng.randint(0, height-block_h+1))
        left = int(rng.randint(0, width-block_w+1))
        added_missing[top:top+block_h, left:left+block_w] = valid[top:top+block_h, left:left+block_w]
        rectangle = [top, left, block_h, block_w]
    # Count added missing first; remaining noise-zero pixels form a disjoint class.
    surviving_noise_zero = noise_zero & ~added_missing
    final_valid = valid & ~added_missing & ~noise_zero
    altered[~final_valid] = 0.0
    target = np.zeros((height, width), dtype=np.float32)
    residual = np.abs(altered[final_valid]-reference[final_valid])
    target[final_valid] = np.exp(-residual/np.float32(QUALITY_DECAY_M))
    modified = inputs.detach().clone()
    modified[3].copy_(torch.from_numpy(altered/np.float32(DEPTH_CLIP_M)))
    modified[4].copy_(torch.from_numpy(final_valid.astype(np.float32)))
    metadata.update(height=height, width=width, missing_rectangle=rectangle,
        noise_sigma_m=NOISE_SIGMA_M if selected in ('noise', 'joint') else 0.0,
        quality_decay_m=QUALITY_DECAY_M, source_valid_pixels=int(valid.sum()),
        original_invalid_pixels=int((~valid).sum()), added_missing_valid_pixels=int(added_missing.sum()),
        noise_clipped_zero_pixels=int(surviving_noise_zero.sum()),
        noise_clipped_zero_before_missing_pixels=int(noise_zero.sum()),
        noise_clipped_upper_pixels=int(noise_upper.sum()), final_valid_pixels=int(final_valid.sum()),
        mean_quality=float(target.mean(dtype=np.float64)))
    return modified, torch.from_numpy(target[None]), metadata


def quality_loss(quality_maps, qtarget):
    """Mean over pixels, then four scales, then samples; target has no gradient."""
    if not isinstance(quality_maps, (tuple, list)) or len(quality_maps) != 4:
        raise ValueError('Expected four quality maps')
    if not isinstance(qtarget, torch.Tensor) or not qtarget.is_floating_point():
        raise ValueError('Quality target must be a floating tensor')
    target = qtarget.detach().float()
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if (target.ndim != 4 or target.shape[0] < 1 or target.shape[1] != 1
            or not bool(torch.isfinite(target).all()) or bool(((target < 0) | (target > 1)).any())):
        raise ValueError('Expected bounded quality target [N,1,H,W]')
    losses = []
    for quality in quality_maps:
        if (not isinstance(quality, torch.Tensor) or not quality.is_floating_point()
                or quality.ndim != 4 or quality.shape[:2] != target.shape[:2]
                or quality.device != target.device or min(quality.shape[-2:]) < 1
                or any(a > b for a, b in zip(quality.shape[-2:], target.shape[-2:]))):
            raise ValueError('Quality map geometry or device differs')
        smaller = F.interpolate(target, size=quality.shape[-2:], mode='area')
        losses.append((quality.float()-smaller).square().mean(dim=(1, 2, 3)))
    return torch.stack(losses).mean()


class QualityGraspNetFrames(GraspNetFrames):
    """Reuse existing clean preprocessing, add depth-only degradation afterward."""
    def __init__(self, data_root, records, size=(480, 640), seed=SEED, augment=True, condition=None):
        if tuple(size) != (480, 640) or seed != SEED:
            raise ValueError('Research wrapper requires the fixed resolution and seed')
        if condition is not None and condition not in CONDITIONS:
            raise ValueError('Unknown evaluation condition')
        if condition is not None and augment:
            raise ValueError('Explicit evaluation condition requires augmentation disabled')
        if condition is None and not augment:
            raise ValueError('Evaluation requires an explicit depth condition')
        super().__init__(data_root, records, size=size, mode='rgbd', seed=seed, augment=augment,
                         depth_clip_m=DEPTH_CLIP_M)
        self.condition = condition

    def epoch_order(self, epoch):
        _integer(epoch, 'epoch', 0, 9)
        return np.random.RandomState((self.seed+epoch) % 2**32).permutation(len(self.records)).tolist()

    def get(self, index, epoch=0):
        if self.condition is not None and epoch != 0:
            raise ValueError('Evaluation condition requires epoch zero')
        clean, targets = super().get(index, epoch)
        modified, target, metadata = corrupt_depth(clean, index, epoch,
            record=self.records[index], condition=self.condition)
        targets = dict(targets)
        targets['quality_target'] = target
        targets['quality_metadata'] = metadata
        return modified, targets

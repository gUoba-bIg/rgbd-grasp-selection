"""Read-only CPU decomposition of the preserved 100-step fitting diagnostic.

Oracle substitutions intentionally use training labels for fault isolation only.
They are not model performance and never select a held-out decoder threshold.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from src.checkpoints import atomic_json, file_sha256
from src.graspnet_frames import GraspNetFrames
from src.segmentation import PilotInstanceNet, decode_instances, instance_counts, summarize_counts


def differentiable_losses(outputs, targets):
    fg = targets['foreground'].float()
    p = outputs['foreground'].sigmoid()
    bce = F.binary_cross_entropy_with_logits(outputs['foreground'], fg)
    dice = 1 - (2 * (p * fg).sum() + 1) / (p.sum() + fg.sum() + 1)
    heatmap = targets['centers']
    cp = outputs['centers'].sigmoid().clamp(1e-5, 1 - 1e-5)
    positive = (heatmap == 1).float()
    negative = (heatmap < 1).float()
    pos = -(cp.log() * (1 - cp).pow(2) * positive).sum() / positive.sum().clamp(min=1)
    neg = -((1 - cp).log() * cp.pow(2) * (1 - heatmap).pow(4) * negative).sum() / positive.sum().clamp(min=1)
    per_pixel = F.smooth_l1_loss(outputs['offsets'], targets['offsets'], reduction='none', beta=0.05)
    offset = 5 * (per_pixel * fg).sum() / (2 * fg.sum()).clamp(min=1)
    return {'foreground_bce_dice': bce + dice, 'center_positive': pos,
            'center_negative': neg, 'offset_weighted': offset}


def gradients(model, inputs, targets):
    outputs = model(inputs[None])
    parts = differentiable_losses(outputs, targets)
    params = list(model.named_parameters())
    result = {}
    group_keys = ('stem', 'stages', 'laterals', 'refine', 'head', 'foreground', 'centers', 'offsets')
    for name, loss in parts.items():
        grads = torch.autograd.grad(loss, [p for _, p in params], retain_graph=True, allow_unused=True)
        sums = {key: 0.0 for key in group_keys}
        for (param_name, _), grad in zip(params, grads):
            if grad is not None:
                sums[param_name.split('.')[0]] += float(grad.double().square().sum())
        result[name] = {'loss': float(loss.detach()), 'gradient_norm_total': float(sum(sums.values()) ** 0.5),
                        'gradient_norm_by_group': {k: v ** 0.5 for k, v in sums.items()}}
    total_grads = torch.autograd.grad(sum(parts.values()), [p for _, p in params], allow_unused=True)
    total_norm = sum(float(g.double().square().sum()) for g in total_grads if g is not None) ** 0.5
    result['total'] = {'gradient_norm': total_norm, 'clip_multiplier_at_limit_1': min(1, 1 / total_norm)}
    return result


def main():
    torch.set_num_threads(4)
    directory = ROOT / 'experiments/segmentation_diagnostics/rgbd_overfit_v1'
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    manifest = json.loads((directory / 'data_manifest.json').read_text(encoding='utf-8'))
    pointer = json.loads((directory / 'checkpoints/latest.json').read_text(encoding='utf-8'))
    checkpoint_manifest = directory / 'checkpoints' / pointer['manifest']
    assert file_sha256(checkpoint_manifest) == pointer['manifest_sha256']
    cm = json.loads(checkpoint_manifest.read_text(encoding='utf-8'))
    checkpoint = checkpoint_manifest.parent / cm['file']
    assert file_sha256(checkpoint) == cm['sha256']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['extra']['config'] == config
    torch.manual_seed(config['seed'])
    model = PilotInstanceNet(mode=config['mode']).eval()
    data = GraspNetFrames(ROOT / 'data/raw/graspnet', manifest['records']['train'],
                         size=config['input_size'], mode=config['mode'], seed=config['seed'], augment=False)
    augmented_data = GraspNetFrames(ROOT / 'data/raw/graspnet', manifest['records']['train'],
                         size=config['input_size'], mode=config['mode'], seed=config['seed'], augment=True)
    first_input, first_targets = data.get(0)
    target_batch = {k: v[None] for k, v in first_targets.items() if k != 'labels'}
    initial_gradient = gradients(model, first_input, target_batch)
    model.load_state_dict(payload['model'])
    final_gradient = gradients(model, first_input, target_batch)
    oracle_names = ('predicted', 'oracle_all', 'oracle_foreground', 'oracle_centers', 'oracle_offsets',
                    'predicted_foreground_only', 'predicted_centers_only', 'predicted_offsets_only')
    totals = {name: {'tp': 0, 'fp': 0, 'fn': 0} for name in oracle_names}
    thresholds = {str(th): {'tp': 0, 'fp': 0, 'fn': 0} for th in (0.05, 0.1, 0.2, 0.35)}
    frames = []
    with torch.no_grad():
        for i, record in enumerate(data.records):
            inputs, target = data.get(i)
            output = model(inputs[None])
            labels = target['labels'].numpy()
            oracle = {'foreground': torch.where(target['foreground'][None] > 0, 12.0, -12.0),
                      'centers': torch.logit(target['centers'][None].clamp(1e-5, 1 - 1e-5)),
                      'offsets': target['offsets'][None]}
            variants = {'predicted': output, 'oracle_all': oracle}
            for key in ('foreground', 'centers', 'offsets'):
                variants['oracle_' + key] = dict(output, **{key: oracle[key]})
                variants['predicted_' + key + '_only'] = dict(oracle, **{key: output[key]})
            counts_by_variant = {}
            for name, prediction in variants.items():
                counts = instance_counts(decode_instances(prediction, **config['decoder']), labels)
                counts_by_variant[name] = summarize_counts(counts)
                for key in counts:
                    totals[name][key] += counts[key]
            for threshold in thresholds:
                decoder = dict(config['decoder'], center_threshold=float(threshold))
                counts = instance_counts(decode_instances(output, **decoder), labels)
                for key in counts:
                    thresholds[threshold][key] += counts[key]
            fg = labels > 0
            height, width = labels.shape
            difference = (output['offsets'][0].numpy() - target['offsets'].numpy()) * np.array([width, height])[:, None, None]
            error = np.linalg.norm(difference, axis=0)[fg]
            center_probability = output['centers'][0, 0].sigmoid()
            gt_centers = torch.nonzero(target['centers'][0] == 1)
            center_at_gt = center_probability[gt_centers[:, 0], gt_centers[:, 1]].numpy()
            pred_fg = output['foreground'][0, 0].sigmoid().numpy() >= 0.5
            aug_input, aug_target = augmented_data.get(i, epoch=49)
            aug_output = model(aug_input[None])
            aug_pred_fg = aug_output['foreground'][0, 0].sigmoid().numpy() >= 0.5
            aug_fg = aug_target['labels'].numpy() > 0
            frames.append({'scene': record['scene'], 'frame': record['frame'], 'oracle_decomposition': counts_by_variant,
                           'foreground_iou': float((pred_fg & fg).sum() / (pred_fg | fg).sum()),
                           'last_epoch_augmented_foreground_iou': float((aug_pred_fg & aug_fg).sum() / (aug_pred_fg | aug_fg).sum()),
                           'center_probability_at_true_centers': center_at_gt.tolist(),
                           'offset_endpoint_error_pixels': {'mean': float(error.mean()), 'median': float(np.median(error)),
                                                            'p90': float(np.quantile(error, 0.9))}})
    history = json.loads((directory / 'history.json').read_text(encoding='utf-8'))
    report = {'scope': 'Training-only read-only CPU fault isolation, not model training or held-out performance',
              'checkpoint_sha256': cm['sha256'], 'source_sha256': {name: file_sha256(ROOT / name) for name in ('src/segmentation.py', 'src/graspnet_frames.py')},
              'oracle_warning': 'Oracle substitutions use ground truth intentionally and are not deployable predictions',
              'threshold_warning': 'Training-only sensitivity, no validation threshold chosen',
              'training_frames': frames, 'oracle_totals': {k: summarize_counts(v) for k, v in totals.items()},
              'threshold_sensitivity_training_only': {k: summarize_counts(v) for k, v in thresholds.items()},
              'gradients_no_update_cpu_fp32': {'initial_same_seed_first_unaugmented_frame': initial_gradient,
                                             'step100_first_unaugmented_frame': final_gradient},
              'learning_rate': {'after_step50': history[49]['learning_rate_after_step'],
                                'after_step80': history[79]['learning_rate_after_step'],
                                'after_step100': history[99]['learning_rate_after_step']}}
    output_path = ROOT / 'results/P2/segmentation_readonly_audit.json'
    if output_path.exists():
        raise RuntimeError('Audit already exists; refusing to replace')
    atomic_json(output_path, report)
    print(json.dumps({'output': str(output_path), 'oracle_totals': report['oracle_totals'],
                      'threshold_sensitivity_training_only': report['threshold_sensitivity_training_only'],
                      'initial_gradient_norms': {k: v.get('gradient_norm_total', v.get('gradient_norm')) for k, v in initial_gradient.items()},
                      'final_gradient_norms': {k: v.get('gradient_norm_total', v.get('gradient_norm')) for k, v in final_gradient.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Read-only training-evidence review; never repeats production optimization.

This checks recorded execution consistency, exact configuration and frozen fit
normalization/prior. It does not replay all512 gradients, nor establish input
scene provenance without the runner's frozen row/source manifest.
"""
import hashlib
import json
import numpy as np
import torch
from src import p6_local_correspondence_v50 as core

require = core.require
STAGES = ('forward', 'backward', 'update')
EVENTS = ('intent', 'returned')


def _finite_scalar(value, nonnegative=False):
    return (isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value) and (not nonnegative or value >= 0))


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _independent_order(n, updates, batch_size, seed):
    generator = torch.Generator(device='cpu'); generator.manual_seed(seed)
    batches = []
    while len(batches) < updates:
        permutation = torch.randperm(n, generator=generator).tolist()
        for start in range(0, n, batch_size):
            if len(batches) == updates: break
            batches.append(permutation[start:start+batch_size])
    return batches


def _events_contract(events, order):
    require(isinstance(events, list) and len(events) == len(order)*6, 'Wrong count of training intent/return events')
    require(all(isinstance(row, dict) and type(row.get('id')) is int and type(row.get('cases')) is int
                and isinstance(row.get('batch_indices'), list)
                and all(type(v) is int for v in row['batch_indices']) for row in events),
            'Event identifiers and batch positions must be exact integers')
    expected = []
    for update, ids in enumerate(order, 1):
        for stage in STAGES:
            for outcome in EVENTS:
                expected.append(dict(stage=stage, event=outcome, id=update, batch_indices=ids, cases=len(ids)))
    require(events == expected, 'Training event order, batch identity or pairing differs')
    return {stage+'_'+event: len(order) for stage in STAGES for event in EVENTS}


def _trace_contract(trace, order, final_state_sha):
    require(isinstance(trace, list) and len(trace) == len(order) and _sha(final_state_sha), 'Incomplete update trace/final hash')
    required = {'update', 'batch_indices', 'nll_before_update', 'gradient_l2', 'state_sha256', 'forward_returned', 'update_returned'}
    for i, (row, ids) in enumerate(zip(trace, order), 1):
        require(isinstance(row, dict) and set(row) == required and row['update'] == i
                and type(row['update']) is int and type(row['forward_returned']) is int and type(row['update_returned']) is int
                and isinstance(row['batch_indices'], list) and all(type(v) is int for v in row['batch_indices'])
                and row['batch_indices'] == ids and row['forward_returned'] == i and row['update_returned'] == i
                and _finite_scalar(row['nll_before_update'], True) and _finite_scalar(row['gradient_l2'], True)
                and _sha(row['state_sha256']), 'Invalid finite update trace or exact returned counters')
    require(trace[-1]['state_sha256'] == final_state_sha, 'Returned final checkpoint is not the last update')


def _persisted_contract(persisted, internal, name):
    require(isinstance(persisted, list) and len(persisted) == len(internal), 'Incomplete persisted '+name)
    normalized = []
    for row, reference in zip(persisted, internal):
        require(isinstance(row, dict) and set(row) in (set(reference), set(reference)|{'at'}),
                'Unexpected persisted '+name+' payload fields')
        normalized.append({key: value for key, value in row.items() if key != 'at'})
    require(normalized == internal, 'Persisted '+name+' differs from returned evidence')


def _normalizer_contract(head, x, y):
    a = np.asarray(x, np.float64); target = np.asarray(y)
    require(a.ndim == 5 and a.shape[1:] == (81, 4, 4, 28) and np.isfinite(a).all()
            and target.shape == (len(a),) and np.issubdtype(target.dtype, np.integer)
            and np.all((target >= 0) & (target < 81)), 'Invalid supplied training observations/labels')
    center = a.mean(axis=(0, 1, 2, 3)); scale = a.std(axis=(0, 1, 2, 3))
    scale = np.where(scale > 1e-8, scale, 1.)
    normalizer = head.get('normalizer', {})
    require(normalizer.get('source_scope') == 'training_only' and normalizer.get('fitted_cases') == len(a)
            and normalizer.get('clip') == [-8., 8.]
            and np.array_equal(np.asarray(normalizer.get('center')), center)
            and np.array_equal(np.asarray(normalizer.get('scale')), scale),
            'Saved normalizer is not exactly the supplied training-only normalizer')
    counts = np.bincount(target, minlength=81).astype(np.float64)+1/81
    prior = counts/counts.sum()
    require(np.array_equal(np.asarray(head.get('empirical_prior')), prior),
            'Saved prior is not exactly the supplied training-label smoothed prior')


def _optimizer_contract(state, network, completed_updates):
    parameters = list(network.parameters())
    require(isinstance(state, dict) and set(state) == {str(i) for i in range(len(parameters))}, 'Incomplete final Adam states')
    for i, parameter in enumerate(parameters):
        row = state[str(i)]
        require(isinstance(row, dict) and set(row) == {'step', 'exp_avg', 'exp_avg_sq'}
                and _finite_scalar(row['step']) and row['step'] == completed_updates, 'Wrong final Adam step counter')
        first, second = np.asarray(row['exp_avg']), np.asarray(row['exp_avg_sq'])
        require(first.shape == second.shape == tuple(parameter.shape) and np.isfinite(first).all()
                and np.isfinite(second).all() and np.all(second >= 0), 'Invalid final Adam moment tensors')
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def validate_training_report(head, report, x, y, *, events=None, updates=None):
    """Validate the complete864/512 production contract without retraining."""
    require(isinstance(head, dict) and isinstance(report, dict), 'Training head/report must be dictionaries')
    expected = dict(completed=True, training_role='production', parameter_count=1193, updates_requested=512,
        returned_updates=512, batch_size=32, seed=20261102, cpu_threads=2, optimizer='Adam', learning_rate=.001,
        weight_decay=.001, adam_betas=[.9, .999], adam_epsilon=1e-8, foreach=False,
        loss='mean_proper_categorical_NLL_with_fixed_training_prior', gradient_clipping=False,
        early_stopping=False, checkpoint_selection=False, hyperparameter_search=False, calibration_used=False,
        normalizer_fitted_cases=864, prior_fitted_cases=864, automatic_retry=False)
    for key, value in expected.items():
        require(report.get(key) == value and (type(report.get(key)) is type(value) if type(value) in (bool, int) else True),
                'Production training configuration mismatch: '+key)
    require(np.asarray(x).shape == (864, 81, 4, 4, 28) and np.asarray(y).shape == (864,),
            'Production review requires all864 training cases')
    expected_architecture = dict(token_widths=[16, 8], conv_channels=8, conv_kernel=3, conv_padding=1,
        pool='fixed_16_region_mean', activation='ReLU', candidate_specific_biases=0)
    require(head.get('architecture') == expected_architecture and head.get('final_checkpoint_only') is True
            and head.get('head_refitted_for_deployment') is False and head.get('spatial_shape') == [4, 4],
            'Head architecture or checkpoint rule changed')
    network = core.load_network(head, core.FEATURE_NAMES, require_production=True)
    require(report.get('final_state_sha256') == core.state_sha256(network), 'Final report/state identity mismatch')
    initializer = core.build_model(28, seed=20261102)
    require(report.get('initialization_sha256') == core.state_sha256(initializer), 'Seed initialization identity mismatch')
    _normalizer_contract(head, x, y)
    order = _independent_order(864, 512, 32, 20261102)
    require(report.get('batch_order') == order, 'Training order differs from fixed seeded epochs')
    counts = _events_contract(report.get('events'), order)
    require(report.get('counts') == counts, 'Actual stage counters do not match paired events')
    require(report.get('training_case_forwards_returned') == 512*32
            and report.get('training_candidate_branches_returned') == 512*32*81,
            'Returned batch/candidate cardinalities differ from recorded calls')
    _trace_contract(report.get('update_trace'), order, head['state_sha256'])
    if events is not None: _persisted_contract(events, report['events'], 'events')
    if updates is not None: _persisted_contract(updates, report['update_trace'], 'updates')
    optimizer_hash = _optimizer_contract(report.get('optimizer_state'), network, 512)
    return dict(passed=True, reviewed_training_cases=864, returned_training_forward_calls=512,
        returned_backward_calls=512, returned_optimizer_updates=512, returned_batch_case_evaluations=512*32,
        candidate_branch_cardinality=512*32*81, candidate_branches_are_not_separate_forward_calls=True,
        all512_data_order_and_finite_trace_checked=True, normalizer_and_prior_recomputed_training_only=True,
        seed_initialization_checked=True, final_state_sha256=head['state_sha256'],
        final_optimizer_state_sha256=optimizer_hash, external_events_matched=events is not None,
        external_updates_matched=updates is not None, production_training_repeated=False,
        all512_gradient_values_replayed=False, input_scene_provenance_requires_frozen_runner_manifest=True)

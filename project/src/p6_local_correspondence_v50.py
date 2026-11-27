"""Small spatial correspondence head on fixed observable regional features.

Production training is one fixed final-checkpoint run. No image, source-id,
noise-mode, true warp vector or instance annotation is accepted by prediction.
The target candidate index is accepted only by supervised training.
"""
import copy
import hashlib
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from scipy.special import logsumexp
from src import p6_joint_repair_v45 as base

FEATURE_COUNT = 28
FEATURE_NAMES = tuple(base.WARP_FEATURES)+(
    'instance_confidence', 'instance_soft_boundary', 'instance_boundary_proximity',
    'instance_boundary_depth_product', 'instance_proximity_depth_product', 'instance_interior_depth_product',
    'observed_valid_fraction3', 'instance_interior_normal_variation', 'instance_interior_plane_residual',
    'instance_confidence_surface_valid', 'instance_confidence_depth_edge', 'foreground_depth_valid')
SEED = 20261102
TRAIN_UPDATES = 512
BATCH_SIZE = 32
LEARNING_RATE = .001
WEIGHT_DECAY = .001
CPU_THREADS = 2
MODEL_FAMILY = 'spatial_local_correspondence_v50'
require = base.require
CANDIDATES = base.CANDIDATES
warp_statistics = base.warp_statistics


@contextmanager
def cpu_runtime():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(CPU_THREADS)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


class LocalCorrespondence(nn.Module):
    def __init__(self, n_features=FEATURE_COUNT):
        super().__init__()
        require(type(n_features) is int and n_features == FEATURE_COUNT, 'Frozen28-channel architecture required')
        self.token = nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.spatial = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.output = nn.Linear(8, 1)

    def forward(self, features):
        require(features.ndim == 5 and tuple(features.shape[1:]) == (81, 4, 4, FEATURE_COUNT),
                'Expected batch by81 by4 by4 by28 observable tokens')
        n = features.shape[0]
        token = self.token(features).reshape(n*81, 4, 4, 8).permute(0, 3, 1, 2)
        spatial = torch.relu(self.spatial(token))
        return self.output(spatial.mean(dim=(2, 3))).reshape(n, 81)


def build_model(n_features=FEATURE_COUNT, seed=SEED):
    require(type(seed) is int, 'Seed must be an integer')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LocalCorrespondence(n_features).cpu()
    require(sum(p.numel() for p in model.parameters()) == 1193, 'Architecture parameter budget changed')
    return model


def state_arrays(network):
    return {name: tensor.detach().cpu().tolist() for name, tensor in network.state_dict().items()}


def state_sha256(network):
    digest = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        a = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode('utf-8')); digest.update(str(a.shape).encode('ascii'))
        digest.update(a.tobytes())
    return digest.hexdigest()


def _features(features, feature_names):
    a = np.asarray(features)
    names = tuple(feature_names)
    require(names == FEATURE_NAMES, 'Frozen28 observable feature names/order required')
    require(a.ndim == 5 and a.shape[1:] == (81, 4, 4, FEATURE_COUNT) and len(a) > 0
            and np.issubdtype(a.dtype, np.number) and np.isfinite(a).all(), 'Invalid local feature tensor')
    return a, names


def fit_normalizer(features):
    a = np.asarray(features, np.float64)
    require(a.ndim == 5 and a.shape[1:] == (81, 4, 4, FEATURE_COUNT) and len(a) > 0
            and np.isfinite(a).all(), 'Invalid normalizer fit inputs')
    mean, scale = a.mean(axis=(0, 1, 2, 3)), a.std(axis=(0, 1, 2, 3))
    scale = np.where(scale > 1e-8, scale, 1.)
    return dict(center=mean.tolist(), scale=scale.tolist(), clip=[-8., 8.], fitted_cases=len(a),
                source_scope='training_only')


def normalize_features(features, normalizer):
    a = np.asarray(features)
    center, scale = np.asarray(normalizer['center']), np.asarray(normalizer['scale'])
    require(a.ndim == 5 and a.shape[1:] == (81, 4, 4, FEATURE_COUNT)
            and np.isfinite(a).all() and center.shape == scale.shape == (FEATURE_COUNT,)
            and np.isfinite(center).all() and np.isfinite(scale).all() and np.all(scale > 0)
            and normalizer.get('clip') == [-8., 8.], 'Invalid fixed normalization')
    result = np.clip((a-center)/scale, -8., 8.).astype(np.float32)
    require(np.isfinite(result).all(), 'Nonfinite normalized local features')
    return result


def batch_order(n_cases, updates=TRAIN_UPDATES, batch_size=BATCH_SIZE, seed=SEED):
    require(type(n_cases) is int and type(updates) is int and type(batch_size) is int
            and n_cases >= batch_size > 0 and n_cases % batch_size == 0 and updates > 0,
            'Batch order requires full fixed-size batches')
    generator = torch.Generator(device='cpu'); generator.manual_seed(seed)
    result = []; order = []; cursor = 0
    for _ in range(updates):
        if cursor == len(order):
            order = torch.randperm(n_cases, generator=generator).tolist(); cursor = 0
        result.append(order[cursor:cursor+batch_size]); cursor += batch_size
    return result


def _optimizer_arrays(optimizer):
    return {str(index): {key: value.detach().cpu().tolist() if torch.is_tensor(value) else copy.deepcopy(value)
                         for key, value in entry.items()}
            for index, entry in optimizer.state_dict()['state'].items()}


def _train(features, targets, feature_names, updates, batch_size, role, guard, on_event, on_update):
    a, names = _features(features, feature_names); y = np.asarray(targets)
    require(y.shape == (len(a),) and np.issubdtype(y.dtype, np.integer) and np.all((y >= 0) & (y < 81)),
            'Invalid supervised candidate labels')
    order = batch_order(len(a), updates, batch_size)
    guard(); normalizer = fit_normalizer(a); x = normalize_features(a, normalizer)
    prior = base.empirical_prior(y, n_candidates=81, alpha=1/81)
    events, trace = [], []
    counts = {stage+'_'+event: 0 for stage in ('forward', 'backward', 'update') for event in ('intent', 'returned')}
    examples_returned = 0
    network = build_model(); initialization_hash = state_sha256(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                                 betas=(.9, .999), eps=1e-8, foreach=False)
    input_tensor = torch.from_numpy(x); label_tensor = torch.from_numpy(y.astype(np.int64))
    log_prior = torch.from_numpy(np.log(prior).astype(np.float32))

    def event(stage, outcome, update, ids):
        counts[stage+'_'+outcome] += 1
        row = dict(stage=stage, event=outcome, id=update, batch_indices=list(ids), cases=len(ids))
        events.append(row); on_event(copy.deepcopy(row))

    def answer(completed):
        model = dict(family=MODEL_FAMILY, feature_names=list(names), candidate_count=81,
            candidate_order=CANDIDATES.tolist(), spatial_shape=[4, 4], feature_count=FEATURE_COUNT,
            architecture=dict(token_widths=[16, 8], conv_channels=8, conv_kernel=3, conv_padding=1,
                              pool='fixed_16_region_mean', activation='ReLU', candidate_specific_biases=0),
            parameter_count=1193, state_dict=state_arrays(network), state_sha256=state_sha256(network),
            normalizer=copy.deepcopy(normalizer), empirical_prior=prior.tolist(), fitted_cases=len(a),
            training_role=role, completed=completed, returned_updates=counts['update_returned'],
            requested_updates=updates, seed=SEED, final_checkpoint_only=True, head_refitted_for_deployment=False)
        return dict(model=model, training_report=dict(completed=completed, training_role=role,
            counts=copy.deepcopy(counts), training_case_forwards_returned=examples_returned,
            training_candidate_branches_returned=examples_returned*81,
            parameter_count=1193, updates_requested=updates, returned_updates=counts['update_returned'],
            batch_size=batch_size, seed=SEED, cpu_threads=CPU_THREADS, optimizer='Adam', learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY, adam_betas=[.9, .999], adam_epsilon=1e-8, foreach=False,
            loss='mean_proper_categorical_NLL_with_fixed_training_prior', gradient_clipping=False,
            early_stopping=False, checkpoint_selection=False, hyperparameter_search=False,
            calibration_used=False, initialization_sha256=initialization_hash,
            final_state_sha256=model['state_sha256'], batch_order=copy.deepcopy(order),
            events=copy.deepcopy(events), update_trace=copy.deepcopy(trace), optimizer_state=_optimizer_arrays(optimizer),
            normalizer_fitted_cases=len(a), prior_fitted_cases=len(a), automatic_retry=False))
    try:
        with cpu_runtime():
            network.train()
            for update, ids in enumerate(order, 1):
                guard(); optimizer.zero_grad(set_to_none=True)
                event('forward', 'intent', update, ids)
                scores = network(input_tensor[ids])
                examples_returned += len(ids); event('forward', 'returned', update, ids)
                require(bool(torch.isfinite(scores).all()), 'Nonfinite training forward')
                logp = torch.log_softmax(scores+log_prior, dim=1)
                loss = -logp[torch.arange(len(ids)), label_tensor[ids]].mean()
                require(bool(torch.isfinite(loss)), 'Nonfinite training NLL')
                guard(); event('backward', 'intent', update, ids)
                loss.backward(); event('backward', 'returned', update, ids)
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in network.parameters()),
                        'Nonfinite/missing local-model gradient')
                gradient_norm = float(torch.sqrt(sum(torch.sum(p.grad*p.grad) for p in network.parameters())))
                require(np.isfinite(gradient_norm), 'Nonfinite aggregate gradient norm')
                guard(); event('update', 'intent', update, ids)
                optimizer.step(); event('update', 'returned', update, ids)
                require(all(bool(torch.isfinite(p).all()) for p in network.parameters()), 'Nonfinite updated parameters')
                row = dict(update=update, batch_indices=list(ids), nll_before_update=float(loss.detach()),
                           gradient_l2=gradient_norm, state_sha256=state_sha256(network),
                           forward_returned=counts['forward_returned'], update_returned=counts['update_returned'])
                trace.append(row); on_update(copy.deepcopy(row))
            return answer(True)
    except BaseException as error:
        error.partial_report = answer(False)
        raise


def fit_local_model(features, targets, feature_names, guard=lambda: None,
                    on_event=lambda row: None, on_update=lambda row: None):
    a, _ = _features(features, feature_names)
    require(len(a) == 864, 'Production fit requires exactly the frozen eight-scene864 cases')
    return _train(a, targets, feature_names, TRAIN_UPDATES, BATCH_SIZE, 'production', guard, on_event, on_update)


def fit_synthetic_model(features, targets, feature_names, updates=2, batch_size=4,
                        guard=lambda: None, on_event=lambda row: None, on_update=lambda row: None):
    """Explicit toy-only path; its outputs cannot pass production validation."""
    require(type(updates) is int and 0 < updates <= 8 and len(features) <= 64,
            'Synthetic helper has a separate small immutable cap')
    return _train(features, targets, feature_names, updates, batch_size, 'synthetic_only', guard, on_event, on_update)


def load_network(model, feature_names, require_production=True):
    require(isinstance(model, dict) and model.get('family') == MODEL_FAMILY
            and model.get('feature_names') == list(feature_names) and model.get('feature_count') == FEATURE_COUNT
            and model.get('candidate_order') == CANDIDATES.tolist() and model.get('parameter_count') == 1193,
            'Local model feature/candidate/architecture identity mismatch')
    if require_production:
        require(model.get('training_role') == 'production' and model.get('completed') is True
                and model.get('fitted_cases') == 864 and model.get('returned_updates') == TRAIN_UPDATES
                and model.get('requested_updates') == TRAIN_UPDATES and model.get('seed') == SEED,
                'Incomplete/synthetic training cannot be used as production model')
    network = build_model()
    expected = network.state_dict(); supplied = model.get('state_dict', {})
    require(set(supplied) == set(expected), 'Incomplete serialized model state')
    tensors = {}
    for name, prototype in expected.items():
        a = np.asarray(supplied[name], np.float32)
        require(a.shape == tuple(prototype.shape) and np.isfinite(a).all(), 'Invalid serialized parameter tensor')
        tensors[name] = torch.from_numpy(a)
    network.load_state_dict(tensors, strict=True)
    require(state_sha256(network) == model.get('state_sha256'), 'Serialized model weights changed')
    network.eval(); return network


def predict_warp(model, features, feature_names, guard=lambda: None, on_event=lambda row: None,
                 require_production=True):
    a, names = _features(features, feature_names)
    network = load_network(model, names, require_production=require_production)
    prior = base.old.probability_vector(model['empirical_prior'], 81)
    require(np.all(prior > 0), 'Frozen prior must retain81 positive support')
    x = normalize_features(a, model['normalizer']); output = []
    with cpu_runtime(), torch.no_grad():
        for index, start in enumerate(range(0, len(x), BATCH_SIZE)):
            guard(); count = min(BATCH_SIZE, len(x)-start)
            row = dict(stage='forward', id=index, start=start, cases=count, scope='prediction')
            on_event(dict(row, event='intent'))
            scores = network(torch.from_numpy(x[start:start+count])).numpy().astype(np.float64)
            on_event(dict(row, event='returned'))
            require(np.isfinite(scores).all(), 'Nonfinite deployed candidate scores')
            logits = scores+np.log(prior)[None]
            p = np.exp(logits-logsumexp(logits, axis=1, keepdims=True))
            require(np.isfinite(p).all() and np.all(p > 0), 'Numerical underflow violates81 positive support')
            output.append(p)
    return np.concatenate(output)

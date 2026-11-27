"""One finite supervised mixture calibration for the same frozen eight-scene head.

The auxiliary four scenes are excluded from this head's training; they are not
claimed to be unseen by the whole research project or frozen depth/H models.
This module neither refits a head nor dispatches experiments nor reads files.
"""
import copy
import numpy as np
from scipy.special import logsumexp
from src import p6_joint_repair_v45 as base
from src import p6_probability_calibration_v46 as previous

MIXTURE_GRID = tuple(i/20 for i in range(21))
FIXED_FOLD_INDEX = 0
AUXILIARY_SCENES = previous.FOLDS[0]
TRAIN_SCENES = tuple(s for s in previous.FIT_SCENES if s not in AUXILIARY_SCENES)
LEVELS = (.5, .8, .95)
OTHER_GATES = ('warp_nll_gain', 'warp_brier_noninferior', 'warp95_coverage', 'warp_set_sharper', 'worst_scene_warp')
FAMILY = 'fixed_eight_scene_head_prior_mixture_selection_v49'
CANDIDATES = base.CANDIDATES
require = base.require
score_warp = previous.score_warp
warp_statistics = base.warp_statistics


def _probabilities(probability, name):
    p = np.asarray(probability, np.float64)
    require(p.ndim == 2 and p.shape[1] == 81 and len(p) > 0 and np.isfinite(p).all()
            and np.all(p > 0) and np.allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0.),
            'Invalid positive81 '+name)
    return p


def _prior(priors, n):
    p = _probabilities(priors, 'prior')
    require(len(p) == n and np.array_equal(p, np.broadcast_to(p[0], p.shape)),
            'Every case must use the same frozen head prior')
    return p


def baseline_probabilities(scores, priors):
    s = np.asarray(scores, np.float64)
    require(s.ndim == 2 and s.shape[1] == 81 and len(s) > 0 and np.isfinite(s).all(), 'Invalid raw head scores')
    p = _prior(priors, len(s)); logits = s+np.log(p)
    return _probabilities(np.exp(logits-logsumexp(logits, axis=1, keepdims=True)), 'head probabilities')


def mix_probabilities(lambda_value, head_probabilities, priors):
    require(np.isscalar(lambda_value) and not isinstance(lambda_value, (bool, np.bool_))
            and np.isfinite(lambda_value) and 0 <= lambda_value <= 1, 'Mixture weight outside [0,1]')
    head = _probabilities(head_probabilities, 'head probabilities'); prior = _prior(priors, len(head))
    if lambda_value == 0: return head.copy()
    if lambda_value == 1: return prior.copy()
    return _probabilities((1-float(lambda_value))*head+float(lambda_value)*prior, 'mixed probabilities')


def _labels(targets, scenes, n):
    y, ss = np.asarray(targets), np.asarray(scenes)
    require(y.shape == ss.shape == (n,) and np.issubdtype(y.dtype, np.integer)
            and np.issubdtype(ss.dtype, np.integer) and np.all((0 <= y) & (y < 81))
            and np.all(ss >= 0), 'Invalid candidate labels or scene identifiers')
    return y, ss


def _statistics(probabilities, targets, scenes):
    # Reuse the frozen per-case stable sort and selected probability mass.
    rr = [base.warp_statistics(p, int(y), levels=LEVELS) for p, y in zip(probabilities, targets)]
    levels = {str(level): {field: float(np.mean([r['credible_sets'][j][key] for r in rr]))
                          for field, key in (('covered', 'covered'), ('mass', 'probability'), ('size', 'size'))}
              for j, level in enumerate(LEVELS)}
    worst = min(float(np.mean([r['credible_sets'][2]['covered'] for r, s in zip(rr, scenes) if s == scene]))
                for scene in sorted(set(scenes.tolist())))
    return dict(nll=float(np.mean([r['nll'] for r in rr])), brier=float(np.mean([r['brier'] for r in rr])),
                levels=levels, worst_scene_warp95=worst)


def candidate_metrics(probabilities, priors, targets, scenes):
    method = _probabilities(probabilities, 'method probabilities'); prior = _prior(priors, len(method))
    y, ss = _labels(targets, scenes, len(method))
    m, p = _statistics(method, y, ss), _statistics(prior, y, ss)
    levels = {str(level): dict(method=m['levels'][str(level)], prior=p['levels'][str(level)]) for level in LEVELS}
    gaps = {level: abs(row['method']['covered']-row['method']['mass']) for level, row in levels.items()}
    gates = dict(warp_nll_gain=p['nll']-m['nll'] >= .05, warp_brier_noninferior=m['brier'] <= p['brier'],
        warp95_coverage=levels['0.95']['method']['covered'] >= .90,
        warp_set_sharper=levels['0.95']['method']['size'] <= .9*levels['0.95']['prior']['size'],
        warp_multilevel_reliability=all(gap <= .05 for gap in gaps.values()), worst_scene_warp=m['worst_scene_warp95'] >= .80)
    return dict(summary=dict(split='fit_auxiliary_calibration', warp_nll=dict(method=m['nll'], prior=p['nll']),
        warp_brier=dict(method=m['brier'], prior=p['brier']), warp_levels=levels,
        worst_scene_warp95=m['worst_scene_warp95'], case_count=len(method)),
        gates={k: bool(v) for k, v in gates.items()}, other_five_passed=bool(all(gates[g] for g in OTHER_GATES)),
        reliability_gap_by_level=gaps, max_reliability_gap=max(gaps.values()))


def select_candidate(candidates):
    """Exact tie rule; infeasible candidates cannot win on reliability alone."""
    require(isinstance(candidates, list) and len(candidates) == 21, 'All fixed21 candidates must return before selection')
    feasible = []
    for index, row in enumerate(candidates):
        require(row.get('index') == index and row.get('lambda_value') == MIXTURE_GRID[index], 'Fixed mixture grid order changed')
        gates, gaps = row.get('gates', {}), row.get('reliability_gap_by_level', {})
        require(set(gates) == set(OTHER_GATES)|{'warp_multilevel_reliability'}
                and all(type(v) is bool for v in gates.values()) and set(gaps) == {str(l) for l in LEVELS}
                and all(np.isfinite(v) and 0 <= v <= 1 for v in gaps.values()), 'Invalid candidate metric record')
        gap = max(gaps.values()); feasible_flag = all(gates[g] for g in OTHER_GATES)
        require(row.get('max_reliability_gap') == gap and row.get('other_five_passed') is feasible_flag
                and gates['warp_multilevel_reliability'] == (gap <= .05), 'Candidate gate summary inconsistent')
        if feasible_flag: feasible.append(row)
    return None if not feasible else min(feasible, key=lambda row: (row['max_reliability_gap'], row['lambda_value']))


def calibrate_fixed_mixture(scores, priors, targets, scenes, guard=lambda: None, on_candidate=lambda record: None):
    head = baseline_probabilities(scores, priors); prior = _prior(priors, len(head))
    y, ss = _labels(targets, scenes, len(head))
    require(len(head) == 432 and tuple(sorted(set(ss.tolist()))) == AUXILIARY_SCENES
            and all(int(np.sum(ss == scene)) == 108 for scene in AUXILIARY_SCENES),
            'Auxiliary calibration must be exactly F0 four scenes times108 cases')
    candidates = []
    def report(completed):
        selected = select_candidate(candidates) if completed else None
        supported = bool(selected is not None and selected['max_reliability_gap'] <= .05)
        return dict(family=FAMILY, completed=completed, fixed_fold_index=FIXED_FOLD_INDEX,
            auxiliary_scenes=list(AUXILIARY_SCENES), frozen_head_training_scenes=list(TRAIN_SCENES),
            auxiliary_case_count=len(head), empirical_prior=prior[0].tolist(), mixture_grid=list(MIXTURE_GRID),
            candidates=copy.deepcopy(candidates), candidate_evaluations=len(candidates), candidate_evaluation_cap=21,
            selected_index=None if selected is None else selected['index'],
            selected_lambda=None if selected is None else selected['lambda_value'], selected=copy.deepcopy(selected),
            supported=supported, passed=supported, supervised_calibration_selections=1,
            selection_objective='other_five_gates_then_minimum_maximum_three_level_mass_coverage_gap',
            tie_rule='exact_gap_ties_choose_smallest_lambda', reliability_limit=.05,
            threshold_search=False, grid_refinement=False, automatic_retry=False,
            new_head_fits=0, depth_or_H_refits=0, base_head_changed_between_calibration_and_deployment=False,
            prior_estimated_from_auxiliary_labels=False, scientific_adoption=False,
            auxiliary_unseen_by_this_warp_head=True, auxiliary_unseen_by_whole_research=False,
            auxiliary_unseen_by_depth_or_H=False, original_calibration_or_evaluation_used_for_selection=False)
    try:
        for index, lambda_value in enumerate(MIXTURE_GRID):
            guard()
            metrics = candidate_metrics(mix_probabilities(lambda_value, head, prior), prior, y, ss)
            row = dict(index=index, lambda_value=lambda_value, **metrics)
            candidates.append(row); on_candidate(copy.deepcopy(row))
        return report(True)
    except BaseException as error:
        error.partial_report = report(False)
        raise


def _accepted_report(report):
    require(isinstance(report, dict) and report.get('family') == FAMILY and report.get('completed') is True
            and report.get('passed') is True and report.get('supported') is True
            and report.get('fixed_fold_index') == 0 and report.get('mixture_grid') == list(MIXTURE_GRID)
            and report.get('candidate_evaluations') == 21, 'Mixture selection did not pass its auxiliary gate')
    selected = select_candidate(report['candidates'])
    require(selected is not None and selected['max_reliability_gap'] <= .05
            and report.get('selected_index') == selected['index']
            and report.get('selected_lambda') == selected['lambda_value']
            and report.get('selected') == selected, 'Saved selected candidate does not reproduce fixed rule')
    return selected['lambda_value']


def wrap_model(base_model, calibration_report, head_identity=None):
    _, _, _, prior = previous._base_parameters(base_model)
    require(base_model.get('fitted_cases') == 864, 'Deployment must preserve the fixed eight-scene head')
    _accepted_report(calibration_report)
    require(np.array_equal(np.asarray(calibration_report.get('empirical_prior'), float), prior),
            'Deployment prior differs from auxiliary selection prior')
    return dict(family='frozen_eight_scene_head_prior_mixture_v49', candidate_count=81, fixed_fold_index=0,
        base_model=copy.deepcopy(base_model), calibration=copy.deepcopy(calibration_report),
        base_head_identity=copy.deepcopy(head_identity), raw_tau1_base_offset=True, added_temperature_layers=0,
        prior_reestimated=False, original_base_head_refitted=False)


def predict_warp(wrapper, features):
    require(isinstance(wrapper, dict) and wrapper.get('family') == 'frozen_eight_scene_head_prior_mixture_v49'
            and wrapper.get('candidate_count') == 81 and wrapper.get('fixed_fold_index') == 0
            and wrapper.get('raw_tau1_base_offset') is True and wrapper.get('added_temperature_layers') == 0,
            'Wrong frozen-head mixture wrapper')
    model = wrapper['base_model']; lambda_value = _accepted_report(wrapper['calibration'])
    _, _, _, prior = previous._base_parameters(model)
    require(model.get('fitted_cases') == 864
            and np.array_equal(np.asarray(wrapper['calibration']['empirical_prior'], float), prior),
            'Frozen eight-scene head/prior mismatch')
    x = np.asarray(features, float); single = x.ndim == 2
    if single: x = x[None]
    head = base.predict_warp(model, x)
    result = mix_probabilities(lambda_value, head, np.broadcast_to(prior, head.shape))
    return result[0] if single else result

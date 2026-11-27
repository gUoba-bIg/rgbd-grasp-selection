"""Finite diagnostics of frozen fit/calibration caches; no fitting or I/O."""
import numpy as np
from scipy.special import stdtr, stdtrit
from scipy.stats import rankdata

MODES = ('nominal', 'noise5', 'noise5_missing15')
LEVELS = (.5, .8, .95)
CDF_GRID = (-10., -5., -3., -2., -1., 0., 1., 2., 3., 5., 10.)
FEATURES = ('depth_m', 'valid_fraction3', 'valid_fraction7', 'median_deviation_mm',
            'local_range_mm', 'global_interior_noise_proxy_mm')
PROTOCOL = dict(version=44, purpose='frozen_cache_mechanism_audit_only', fit_point_case_cap=256,
                levels=list(LEVELS), standardized_cdf_grid=list(CDF_GRID), reliability_bins=10,
                feature_groups='mode plus fixed fit-derived quartiles of features 1,3,4,5',
                quantitative_flags=dict(central_coverage_absolute_gap=.05, absolute_scale_residual_correlation=.3,
                                        outside_supervised_mass=.05),
                new_parameter_fits=0, new_neural_forward=0, new_official_AP=0, raw_replay=0,
                evaluation_rows_used_for_diagnosis=False,
                continuation='evidence flags permit only proposal of one prespecified repair; no automatic distribution selection')


def need(value, message):
    if not bool(value):
        raise ValueError(message)


def weights_normalized(weights):
    w = np.asarray(weights, np.float64)
    need(w.ndim == 1 and len(w) and np.isfinite(w).all() and np.all(w > 0), 'Invalid diagnostic weights')
    return w/w.sum()


def pearson(x, y, weights):
    x, y, w = np.asarray(x, float), np.asarray(y, float), weights_normalized(weights)
    need(x.shape == y.shape == w.shape and np.isfinite(x).all() and np.isfinite(y).all(), 'Invalid correlation arrays')
    a, b = x-np.dot(w, x), y-np.dot(w, y)
    va, vb = np.dot(w, a*a), np.dot(w, b*b)
    if va <= 1e-30 or vb <= 1e-30:
        return dict(value=None, reason='constant_or_degenerate_variable')
    return dict(value=float(np.clip(np.dot(w, a*b)/np.sqrt(va*vb), -1., 1.)), reason=None)


def reconstructed_case_membership(weights, fit_case_rows):
    """Recover concatenation boundaries from frozen per-case uniform 1/n weights.

    No source IDs or per-point surface labels were cached. This only establishes
    case membership under the checked v43 producer contract, not surface strata.
    """
    w = np.asarray(weights, float)
    need(w.ndim == 1 and np.isfinite(w).all() and np.all(w > 0), 'Invalid saved sample weights')
    need(len(fit_case_rows) and all(r['split'] == 'fit' and r['mode'] in MODES for r in fit_case_rows), 'Nonfit rows in training audit')
    need(len({r['key'] for r in fit_case_rows}) == len(fit_case_rows), 'Duplicate fit case')
    membership = np.empty(len(w), np.int32); cursor = 0; counts = []
    for i, row in enumerate(fit_case_rows):
        need(cursor < len(w), 'Fit cases exceed depth cache')
        n = int(np.rint(1/w[cursor]))
        need(8 < n <= 256 and cursor+n <= len(w) and np.allclose(w[cursor:cursor+n], 1/n, rtol=0., atol=1e-15),
             'Cannot reconstruct exact per-case depth support')
        membership[cursor:cursor+n] = i; counts.append(n); cursor += n
    need(cursor == len(w), 'Unassigned depth samples remain')
    return membership, counts


def frozen_scale(depth_model, features):
    """Evaluate the saved seven-coefficient formula; never fit any coefficient."""
    need(depth_model['family'] == 'same_source_zero_location_student_t' and depth_model['features'] == list(FEATURES)
         and depth_model['df'] == 3. and depth_model['scale_bounds_m'] == [.0001, .05]
         and depth_model['calibration_multiplier'] == 1. and depth_model['location_m'] == 0., 'Unexpected frozen depth head')
    x = np.asarray(features, float); center = np.asarray(depth_model['center'], float)
    normalization = np.asarray(depth_model['scale'], float); theta = np.asarray(depth_model['log_scale_coefficients'], float)
    need(x.ndim == 2 and x.shape[1] == 6 and center.shape == normalization.shape == (6,) and theta.shape == (7,)
         and all(np.isfinite(a).all() for a in (x, center, normalization, theta)) and np.all(normalization > 0), 'Invalid frozen scale arrays')
    standardized = np.clip((x-center)/normalization, -8., 8.)
    linear = theta[0]+standardized@theta[1:]
    scale = np.exp(np.clip(linear, np.log(.0001), np.log(.05)))
    return scale, linear


def residual_summary(residual, scale, weights, linear=None):
    y, s, w = np.asarray(residual, float), np.asarray(scale, float), weights_normalized(weights)
    need(y.shape == s.shape == w.shape and np.isfinite(y).all() and np.isfinite(s).all() and np.all(s > 0), 'Invalid residual diagnostics')
    u, absolute = y/s, np.abs(y)
    def fraction(mask): return float(np.clip(np.dot(w, mask), 0., 1.))
    empirical = np.array([fraction(u <= value) for value in CDF_GRID])
    expected = stdtr(3., np.array(CDF_GRID))
    covered = {str(level): fraction(np.abs(u) <= stdtrit(3., (1+level)/2)) for level in LEVELS}
    correlation = pearson(s, absolute, w)
    rank_correlation = pearson(rankdata(s), rankdata(absolute), w)
    return dict(n=len(y), positive_fraction=fraction(y > 0), negative_fraction=fraction(y < 0),
                zero_fraction=fraction(y == 0), mean_signed_residual_m=float(w@y), mean_abs_residual_m=float(w@absolute),
                standardized_mean=float(w@u), standardized_second_moment=float(w@(u*u)),
                scale_mean_m=float(w@s), scale_min_m=float(s.min()), scale_max_m=float(s.max()),
                scale_lower_saturation_fraction=fraction(linear <= np.log(.0001)) if linear is not None else None,
                scale_upper_saturation_fraction=fraction(linear >= np.log(.05)) if linear is not None else None,
                scale_abs_residual_pearson=correlation, scale_abs_residual_spearman=rank_correlation,
                interval_coverage=covered, interval_width95_mean_m=float(2*stdtrit(3., .975)*(w@s)),
                standardized_cdf=dict(grid=list(CDF_GRID), observed=empirical.tolist(), expected_student_t3=expected.tolist(),
                                      maximum_grid_absolute_gap=float(np.max(np.abs(empirical-expected)))),
                effective_weight_count=float(1/np.dot(w, w)),
                dependent_samples=True, confidence_intervals_or_iid_tests_claimed=False)


def analyze_training(depth_X, depth_y, weights, fit_case_rows, depth_model):
    x, y, w = np.asarray(depth_X, float), np.asarray(depth_y, float), np.asarray(weights, float)
    need(x.shape == (len(y), 6) and w.shape == y.shape, 'Training cache shape mismatch')
    membership, counts = reconstructed_case_membership(w, fit_case_rows)
    scale, linear = frozen_scale(depth_model, x)
    modes = np.array([r['mode'] for r in fit_case_rows])[membership]
    grouped = {}
    for mode in MODES:
        selected = modes == mode
        need(selected.any(), 'Missing prespecified training mode')
        grouped[mode] = residual_summary(y[selected], scale[selected], w[selected], linear[selected])
    bins = {}
    for column in (1, 3, 4, 5):
        edges = np.quantile(x[:, column], [.25, .5, .75])
        assignment = np.searchsorted(edges, x[:, column], side='right')
        rows = []
        for group in range(4):
            select = assignment == group
            rows.append(dict(bin=group, n=int(select.sum()), summary=residual_summary(y[select], scale[select], w[select], linear[select]) if select.any() else None,
                             empty_reason=None if select.any() else 'duplicate_quantile_edges_or_no_members'))
        bins[FEATURES[column]] = dict(fit_derived_edges=edges.tolist(), boundary_rule='searchsorted side right', groups=rows)
    feature_correlations = {name: dict(with_abs_residual=pearson(x[:, col], np.abs(y), w), with_predicted_scale=pearson(x[:, col], scale, w))
                            for col, name in enumerate(FEATURES)}
    return dict(scope='fit_cache_only_resubstitution_diagnostics', point_count=len(y), case_count=len(fit_case_rows),
                case_point_counts=counts, case_membership_reconstructed_from_weights=True,
                all=residual_summary(y, scale, w, linear), by_mode=grouped, by_fit_feature_quartile=bins,
                feature_correlations=feature_correlations, fitted_coefficients_unchanged=True,
                fit_scenes=sorted({r['scene'] for r in fit_case_rows}),
                cannot_identify=['out_of_sample_conditional_depth_CDF', 'same_surface_membership', 'causal_noise_family'],
                feature_conditioning_can_distort_standardized_residual_shape=True)


def categorical_summary(probability, targets, supported_indices):
    p, y = np.asarray(probability, float), np.asarray(targets)
    need(p.ndim == 2 and p.shape[1] == 81 and y.shape == (len(p),) and len(p) and np.issubdtype(y.dtype, np.integer)
         and np.all((y >= 0) & (y < 81)) and np.isfinite(p).all() and np.all(p >= 0)
         and np.allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0), 'Invalid calibration posterior')
    support = np.asarray(supported_indices)
    need(support.shape == (9,) and len(np.unique(support)) == 9 and np.isin(y, support).all(), 'Supervised nine-warp support differs')
    truth_p = p[np.arange(len(p)), y]
    zero = truth_p == 0
    nll = None if zero.any() else float(-np.log(truth_p).mean())
    brier = np.sum(p*p, axis=1)-2*truth_p+1
    predicted, confidence = np.argmax(p, axis=1), np.max(p, axis=1)
    hit = predicted == y
    edges = np.linspace(0., 1., 11); assignment = np.minimum((confidence*10).astype(int), 9)
    reliability = []; ece = 0.
    for i in range(10):
        keep = assignment == i; n = int(keep.sum())
        acc, conf = (float(hit[keep].mean()), float(confidence[keep].mean())) if n else (None, None)
        if n: ece += n/len(p)*abs(acc-conf)
        reliability.append(dict(bin=i, lower=float(edges[i]), upper=float(edges[i+1]), n=n, accuracy=acc, confidence=conf))
    sorted_ids = np.argsort(-p, axis=1, kind='stable'); sorted_p = np.take_along_axis(p, sorted_ids, axis=1)
    cumulative = np.cumsum(sorted_p, axis=1); levels = {}
    for level in LEVELS:
        sizes = np.minimum((cumulative < level).sum(axis=1)+1, 81)
        ranks = np.argmax(sorted_ids == y[:, None], axis=1)
        levels[str(level)] = dict(coverage=float(np.mean(ranks < sizes)), size_mean=float(sizes.mean()), size_max=int(sizes.max()),
                                 size_fraction_of_81=float(sizes.mean()/81), full_81_fraction=float(np.mean(sizes == 81)),
                                 at_least_9_fraction=float(np.mean(sizes >= 9)))
    outside = 1-p[:, support].sum(axis=1)
    supported_sorted = np.sort(p[:, support], axis=1)[:, ::-1]
    support_count95 = (np.cumsum(supported_sorted, axis=1) < .95).sum(axis=1)+1
    return dict(n=len(p), nll=nll, zero_true_probability_count=int(zero.sum()), brier=float(brier.mean()),
                top1_accuracy=float(hit.mean()), true_class_probability_mean=float(truth_p.mean()), reliability=reliability,
                top_label_fixed_bin_ece=float(ece), credible_sets=levels,
                outside_supervised72_mass_mean=float(outside.mean()), outside_supervised72_mass_max=float(outside.max()),
                outside_supervised72_mass_over_05_fraction=float(np.mean(outside > .05)),
                supervised9_total_mass_mean=float(1-outside.mean()), supervised9_can_reach95_fraction=float(np.mean(support_count95 <= 9)),
                saturation_does_not_imply_no_information=True, candidate_truncation_or_renormalization_performed=False)


def optional_mean(values):
    available = [float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(value=float(np.mean(available)) if available else None, included=len(available), missing=len(values)-len(available))


def analyze_calibration(probability, cal_rows, result_cal_rows, prior, supported_indices):
    need(len(cal_rows) == len(result_cal_rows) and len(cal_rows) > 0
         and all(r['split'] == 'calibration' and r['mode'] in MODES for r in cal_rows+result_cal_rows), 'Evaluation rows cannot enter calibration audit')
    need([r['key'] for r in cal_rows] == [r['key'] for r in result_cal_rows], 'Calibration metadata/result join differs')
    scenes = sorted({r['scene'] for r in cal_rows})
    scene_mode_counts = {str(scene): {mode: sum(r['scene'] == scene and r['mode'] == mode for r in cal_rows) for mode in MODES} for scene in scenes}
    need(len({count for modes in scene_mode_counts.values() for count in modes.values()}) == 1,
         'Unequal scene/mode case counts would invalidate equal-case scene weighting')
    p = np.asarray(probability, float); targets = np.array([r['target_index'] for r in cal_rows], np.int64)
    prior = np.asarray(prior, float); need(prior.shape == (81,), 'Invalid prior shape')
    baseline = np.broadcast_to(prior, p.shape)
    all_warp = {name: categorical_summary(value, targets, supported_indices) for name, value in (('method', p), ('prior', baseline))}
    groups = {}
    fields_A = ('nll', 'fixed5mm_nll', 'width95', 'coverage0.5', 'coverage0.8', 'coverage0.95', 'joint_coverage0.95')
    fields_B = ('joint_nll', 'missing_brier', 'predicted_missing', 'oracle_missing', 'conditional_width95',
                'conditional_coverage0.5', 'conditional_coverage0.8', 'conditional_coverage0.95',
                'union_width95', 'union_pieces95', 'union_coverage95', 'conditional_undefined_count')
    for mode in MODES:
        keep = np.array([r['mode'] == mode for r in cal_rows]); need(keep.any(), 'Missing calibration mode')
        selected = [r for r, yes in zip(result_cal_rows, keep) if yes]
        point = {}
        for region in ('main', 'boundary'):
            a = {field: optional_mean([r['A'][region].get(field) for r in selected]) for field in fields_A}
            b = {name: {field: optional_mean([r['B'][name][region].get(field) for r in selected]) for field in fields_B}
                 for name in ('method', 'prior_same_scale', 'prior_fixed5mm', 'uniform_fixed5mm', 'single_mode')}
            point[region] = dict(A=a, B=b, aggregation='equal case means of already cached aggregates')
        groups[mode] = dict(warp={name: categorical_summary(value[keep], targets[keep], supported_indices) for name, value in (('method', p), ('prior', baseline))}, point_aggregates=point)
    evidence = dict(per_target_component_centers=False, per_target_component_scales=False, per_target_source_membership=False,
                    calibration_standardized_residuals=False, calibration_aggregate_coverage_and_width=True)
    return dict(scope='calibration_only_no_parameter_fitting', case_count=len(cal_rows),
                calibration_scenes=scenes, scene_mode_case_counts=scene_mode_counts,
                equal_case_weights_equal_scene_weights=True, warp=all_warp, by_mode=groups,
                B_evidence_availability=evidence,
                B_mechanism='aggregate comparisons can establish co-occurrence, not separate component width, surface spacing and posterior causes',
                calibration_point_CDF_computed=False, evaluation_used_for_diagnosis=False,
                no_hard72_truncation=True, no_temperature_or_distribution_fit=True)


def mechanism_assessment(training, calibration):
    flags = {}; central = []
    for mode in ('noise5', 'noise5_missing15'):
        fit = training['by_mode'][mode]
        cal = calibration['by_mode'][mode]['point_aggregates']['main']['A']
        fit_gaps = [fit['interval_coverage'][str(q)]-q for q in LEVELS]
        cal_gaps = [None if cal['coverage'+str(q)]['value'] is None else cal['coverage'+str(q)]['value']-q for q in LEVELS]
        common = any(abs(fg) > .05 and cg is not None and abs(cg) > .05 and fg*cg > 0 for fg, cg in zip(fit_gaps, cal_gaps))
        central.append(common)
        correlation = fit['scale_abs_residual_pearson']['value']
        flags[mode] = dict(fit_and_calibration_central_mismatch_same_direction=common,
                           scale_absolute_residual_association=correlation,
                           strong_conditioning_association=correlation is not None and abs(correlation) >= .3)
    warp = calibration['warp']; method, prior = warp['method'], warp['prior']
    better = method['nll'] is not None and prior['nll'] is not None and method['nll'] < prior['nll'] and method['brier'] < prior['brier']
    depth_signal = all(central)
    candidate_signal = method['outside_supervised72_mass_mean'] > .05
    return dict(status='mechanism_evidence_available_for_one_prespecified_proposal' if depth_signal or candidate_signal else 'no_identified_single_repair_from_available_cache',
                training_and_calibration_flags=flags, learned_warp_better_than_empirical_prior_on_both_proper_scores=better,
                outside_supervised_mass_flag=candidate_signal, persistent_depth_coverage_mismatch_flag=depth_signal,
                exactly_one_repair_is_identified=False, automatic_repair_or_new_fit_authorized_by_this_report=False,
                continuation_requires=['state one falsifiable mechanism and retain competing explanation',
                                       'freeze one repair and matched baselines before any fit',
                                       'do not consult evaluation rows to choose the repair',
                                       'retain all 81 candidates and original negative decisions',
                                       'stop if evidence cannot distinguish the proposed mechanism within the allowed scope'],
                unsupported_inferences=['Student-t must be replaced by Gaussian', 'nine-class credible saturation proves no information',
                                        'A same-source residual calibration proves B fixed-target calibration',
                                        'aggregate B width identifies point-level surface mixture causes'])

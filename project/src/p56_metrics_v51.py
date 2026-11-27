"""Prespecified fixed-pool GraspNet geometry metrics; no fitting or IO."""
import numpy as np

GROUPS = ('A', 'B', 'C', 'D', 'R', 'C0', 'D0')
CONDITIONS = ('clean', 'noise5', 'missing15', 'shift3', 'joint')
SCENES = (2, 6, 7, 11, 13, 15, 18, 22, 33, 39)
FRICTIONS = (.2, .4, .6, .8, 1., 1.2)
TOP_K = (1, 5, 10, 20, 50)
GATE_THRESHOLDS = dict(joint_AP_gain=.01, quality_main_AP_gain=.005, uncertainty_main_AP_gain=.005,
    joint_over_rgb_AP_gain=.005, clean_AP_loss_max=.01, positive_degraded_scenes_min=6,
    conditional_concordance_min_exclusive=.5, conditional_unique_pairs_min=50,
    Top5_04_gain_min=0., learned_over_fixed_uncertainty_AP_min=0.,
    quality_increment_with_U_AP_min=0., uncertainty_increment_with_quality_AP_min=0.)


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def normalize_ranking(candidates, scores, original_indices):
    """Fixed geometry, stable frozen-pool-order ties, unique ordinal confidences."""
    a = np.asarray(candidates)
    s = np.asarray(scores, np.float64)
    ids = np.asarray(original_indices)
    require(a.ndim == 2 and a.shape[1] == 17 and a.dtype == np.float32
            and len(a) <= 20 and np.isfinite(a).all(), 'Expected at most twenty float32 fixed grasps')
    require(s.shape == ids.shape == (len(a),) and np.isfinite(s).all()
            and np.issubdtype(ids.dtype, np.integer) and len(np.unique(ids)) == len(ids), 'Invalid scores/candidate identities')
    order = np.argsort(-s, kind='stable')
    result = a[order].copy()
    result[:, 1] = np.clip(result[:, 1], 0., .1)
    result[:, 0] = np.arange(len(a), 0, -1, dtype=np.float32)
    return result, order.astype(np.int64)


def grasp_metrics(scores, collisions):
    """Official TOP50 x six-friction denominator, retaining failed/absent slots."""
    s, raw_collision = np.asarray(scores, np.float64), np.asarray(collisions)
    co = np.asarray(collisions, np.bool_)
    require(s.ndim == 1 and co.shape == s.shape and (raw_collision.dtype == np.bool_ or not len(s)) and len(s) <= 50
            and np.isfinite(s).all() and np.isin(s, [-1., *FRICTIONS]).all(), 'Invalid official geometric labels')
    success = (s[:, None] > 0) & (s[:, None] <= np.asarray(FRICTIONS)[None]) & ~co[:, None]
    counts = np.zeros((50, 6), np.float64)
    n = len(s)
    if n:
        counts[:n] = np.cumsum(success, axis=0)
        counts[n:] = counts[n-1]
    prefix = counts/np.arange(1, 51)[:, None]
    return dict(AP=float(prefix.mean()), AP_04=float(prefix[:, 1].mean()),
        prefix=prefix.tolist(), selected=n,
        topk={str(k):dict(mu04=float(prefix[k-1, 1]), six_average=float(prefix[k-1].mean()),
                         available=min(k, n), denominator=k,
                         any_mu04=bool(success[:min(k, n), 1].any())) for k in TOP_K})


def candidate_support(ranking, group, selected, physical):
    """Descriptive counts on the complete fixed pool; unknown never means reject.

    Official selection is NMS/object/global support selection. Physical labels
    are separate from model missing-support flags. Empty overlaps collision;
    negative_mu04 means an official failure at mu=.4, regardless of model flags.
    """
    ids = ranking['pool_indices']; candidates = ranking['candidates']
    require(group in GROUPS and len(ids) == len(candidates) == 20 and len(set(ids)) == 20
            and set(physical) == set(ids), 'Support accounting requires all twenty frozen candidates')
    chosen = {r['index']:r for r in selected}
    require(len(chosen) == len(selected) and set(chosen) <= set(ids), 'Invalid official selected support')
    scores = np.asarray(ranking['scores'][group], float)
    require(scores.shape == (20,) and np.isfinite(scores).all(), 'Invalid support ranking scores')
    details = []
    for slot, (idx, candidate) in enumerate(zip(ids, candidates)):
        require(candidate['pool_slot'] == slot and candidate['pool_index'] == idx, 'Candidate support order differs')
        arm = candidate['arms'][group]; label = physical[idx]
        unknown, reasons = arm['unknown'], arm['unknown_reasons']
        contact, score = float(arm['contact_support']), float(arm['rank_score'])
        require(type(unknown) is bool and isinstance(reasons, list) and all(isinstance(r, str) for r in reasons)
                and unknown == bool(reasons), 'Invalid model unknown flag')
        require(np.isfinite(contact) and 0 <= contact <= 1+1e-12 and score == scores[slot], 'Support values differ from ranking')
        require(label['index'] == idx and type(label['collision']) is bool and type(label['empty']) is bool
                and label['score'] in [-1., *FRICTIONS] and (not label['empty'] or label['collision']),
                'Invalid complete singleton physical labels')
        if idx in chosen:
            require(chosen[idx]['score'] == label['score'] and chosen[idx]['collision'] == label['collision'],
                    'Selected official physical label differs')
        positive = bool(0 < label['score'] <= .4 and not label['collision'])
        details.append(dict(pool_slot=slot, index=idx, model_unknown=unknown, unknown_reasons=list(reasons),
            zero_contact=contact == 0., zero_rank_score=score == 0., official_selected=idx in chosen,
            collision=label['collision'], empty=label['empty'], score_negative=label['score'] < 0,
            positive_mu04=positive, negative_mu04=not positive))
    flags = ('collision', 'empty', 'score_negative', 'positive_mu04', 'negative_mu04')
    selected_details = [r for r in details if r['official_selected']]
    return dict(candidate_denominator=20, selected_count=len(selected), selected_over_pool=len(selected)/20.,
        unselected_count=20-len(selected),
        model_counts=dict(unknown=sum(r['model_unknown'] for r in details),
                          zero_contact=sum(r['zero_contact'] for r in details),
                          zero_score=sum(r['zero_rank_score'] for r in details),
                          unknown_and_selected=sum(r['model_unknown'] for r in selected_details)),
        official_pool_counts={key:sum(r[key] for r in details) for key in flags},
        official_selected_counts={key:sum(r[key] for r in selected_details) for key in flags},
        candidates=details)


def summarize_candidate_support(rows, scenes=SCENES, conditions=CONDITIONS):
    expected = {(int(s), c, g) for s in scenes for c in conditions for g in GROUPS}
    keys = {(int(r['scene']), r['condition'], r['group']) for r in rows}
    require(len(keys) == len(rows) and keys == expected, 'Incomplete descriptive candidate support scope')
    by_group = {}
    for group in GROUPS:
        units = [r for r in rows if r['group'] == group]
        require(all(r['candidate_denominator'] == 20 and len(r['candidates']) == 20 for r in units),
                'Candidate denominator changed')
        denominator = 20*len(units); selected = sum(r['selected_count'] for r in units)
        totals = {field:{key:sum(r[field][key] for r in units) for key in units[0][field]}
                  for field in ('model_counts', 'official_pool_counts', 'official_selected_counts')}
        by_group[group] = dict(cases=len(units), candidate_denominator=denominator, selected_count=selected,
            selected_over_pool=selected/denominator, mean_selected_per_case=selected/len(units),
            unselected_count=denominator-selected,
            **totals, rates_per_complete_pool={field:{key:value/denominator for key,value in counts.items()}
                                              for field,counts in totals.items()})
    return dict(by_group=by_group, unknown_is_rejection=False, unknown_candidates_removed=False,
        count_semantics='All cases and all 20 candidates remain. Model flags overlap; official empty is also collision. '
            'score_negative is official score<0; negative_mu04 is not an official positive at mu=.4. '
            'Selected counts refer to the official ranked support, not a model unknown filter. '
            'All reported rates retain the complete pool denominator; AP retains TOP50 x 6 separately.')


def conditional_concordance(baseline_scores, penalty_fixed, penalty_quality, positive, original_indices):
    """Association within four fixed baseline-score bins, not causal evidence.

    Each positive/negative pair is a unique support unit. The two penalties
    A-C and B-D use that same pair set; their support is never double-counted.
    Lower penalty predicts success; equal penalties contribute one half.
    """
    a = np.asarray(baseline_scores, np.float64)
    p, q = np.asarray(penalty_fixed, np.float64), np.asarray(penalty_quality, np.float64)
    y, ids = np.asarray(positive), np.asarray(original_indices)
    require(a.shape == p.shape == q.shape == y.shape == ids.shape == (20,)
            and y.dtype == np.bool_ and np.issubdtype(ids.dtype, np.integer)
            and len(np.unique(ids)) == 20 and np.isfinite(a).all()
            and np.isfinite(p).all() and np.isfinite(q).all(), 'Concordance requires the complete twenty-candidate pool')
    bins = np.array_split(np.argsort(-a, kind='stable'), 4)
    rows = []
    for number, idx in enumerate(bins):
        yes, no = idx[y[idx]], idx[~y[idx]]
        pairs = [(int(i), int(j)) for i in yes for j in no]
        wins = [sum(float(v[i] < v[j]) + .5*float(v[i] == v[j]) for i, j in pairs) for v in (p, q)]
        rows.append(dict(bin=number, original_indices=ids[idx].tolist(), positive_count=len(yes),
                         negative_count=len(no), unique_pairs=len(pairs),
                         concordant_pair_credit_fixed=float(wins[0]), concordant_pair_credit_quality=float(wins[1])))
    n = sum(row['unique_pairs'] for row in rows)
    credits = [sum(row[key] for row in rows) for key in ('concordant_pair_credit_fixed', 'concordant_pair_credit_quality')]
    return dict(unique_pairs=n, concordant_pair_credit_fixed=credits[0], concordant_pair_credit_quality=credits[1],
        concordance_fixed=credits[0]/n if n else None, concordance_quality=credits[1]/n if n else None,
        concordance=(credits[0]+credits[1])/(2*n) if n else None, bins=rows)


def combine_concordance(cases):
    n = sum(r['unique_pairs'] for r in cases)
    fixed = sum(r['concordant_pair_credit_fixed'] for r in cases)
    quality = sum(r['concordant_pair_credit_quality'] for r in cases)
    value = (fixed+quality)/(2*n) if n else None
    adequate = n >= GATE_THRESHOLDS['conditional_unique_pairs_min']
    supported = bool(adequate and value > GATE_THRESHOLDS['conditional_concordance_min_exclusive'])
    return dict(cases=len(cases), unique_pairs=n, concordance_fixed=fixed/n if n else None,
                concordance_quality=quality/n if n else None, concordance=value,
                status='inconclusive' if not adequate else ('supported_association' if supported else 'not_supported'),
                supported=supported, scientific_meaning='conditional association only; not causality, significance or calibrated safety')


def summarize(rows, concordance_cases, scenes=SCENES, conditions=CONDITIONS):
    """Equal scene and condition weighting; all seven prespecified arms retained."""
    expected = {(int(s), c, g) for s in scenes for c in conditions for g in GROUPS}
    indexed = {(int(r['scene']), r['condition'], r['group']):r['metrics'] for r in rows}
    require(len(indexed) == len(rows) and set(indexed) == expected, 'Missing, duplicate, or extra official units')
    require(set(conditions) == set(CONDITIONS), 'The clean plus four-degradation scope is fixed')
    diagnostic_index = {(int(r['scene']), r['condition']):r for r in concordance_cases}
    require(len(diagnostic_index) == len(concordance_cases)
            and set(diagnostic_index) == {(int(s), c) for s in scenes for c in conditions}, 'Incomplete concordance scope')
    degraded = [c for c in conditions if c != 'clean']
    means = {g:{c:float(np.mean([indexed[(int(s), c, g)]['AP'] for s in scenes])) for c in conditions} for g in GROUPS}
    by_scene = {str(s):{g:float(np.mean([indexed[(int(s), c, g)]['AP'] for c in degraded])) for g in GROUPS} for s in scenes}
    overall = {g:float(np.mean([by_scene[str(s)][g] for s in scenes])) for g in GROUPS}
    a, b, c, d = (overall[g] for g in ('A', 'B', 'C', 'D'))
    contrast = dict(joint=d-a, quality_main=(b-a+d-c)/2., uncertainty_main=(c-a+d-b)/2.,
        interaction=d-c-b+a, versus_pure_RGB=d-overall['R'], quality_increment_with_U=d-c,
        U_increment_with_quality=d-b, learned_vs_fixed_U=((c-overall['C0'])+(d-overall['D0']))/2.)
    clean_loss = means['A']['clean']-means['D']['clean']
    scene_gains = {s:values['D']-values['A'] for s, values in by_scene.items()}
    positive_scenes = sum(v > 0 for v in scene_gains.values())
    top5_gain = float(np.mean([np.mean([indexed[(int(s), c, 'D')]['topk']['5']['mu04']
                     - indexed[(int(s), c, 'A')]['topk']['5']['mu04'] for c in degraded]) for s in scenes]))
    diagnostic = combine_concordance(concordance_cases)
    diagnostic_degraded = combine_concordance([r for r in concordance_cases if r['condition'] != 'clean'])
    threshold = GATE_THRESHOLDS
    gates = dict(joint_AP_gain=contrast['joint'] >= threshold['joint_AP_gain'],
        quality_main_AP_gain=contrast['quality_main'] >= threshold['quality_main_AP_gain'],
        uncertainty_main_AP_gain=contrast['uncertainty_main'] >= threshold['uncertainty_main_AP_gain'],
        pure_RGB_AP_gain=contrast['versus_pure_RGB'] >= threshold['joint_over_rgb_AP_gain'],
        quality_increment_with_U=contrast['quality_increment_with_U'] >= threshold['quality_increment_with_U_AP_min'],
        U_increment_with_quality=contrast['U_increment_with_quality'] >= threshold['uncertainty_increment_with_quality_AP_min'],
        clean_AP_preserved=clean_loss <= threshold['clean_AP_loss_max'],
        positive_degraded_scenes=positive_scenes >= threshold['positive_degraded_scenes_min'],
        top5_mu04_noninferior=top5_gain >= threshold['Top5_04_gain_min'],
        learned_vs_fixed_U_noninferior=contrast['learned_vs_fixed_U'] >= threshold['learned_over_fixed_uncertainty_AP_min'],
        conditional_geometric_association=diagnostic['supported'])
    return dict(scope='Fixed20-candidate previously exposed development geometry evaluation, not full official benchmark or robot success',
        condition_AP=means, degraded_AP_by_scene=by_scene, mean_degraded_AP=overall,
        contrasts=contrast, clean_D_vs_A_loss=clean_loss, degraded_D_vs_A_scene_gains=scene_gains,
        positive_degraded_scene_count=positive_scenes, degraded_top5_mu04_D_minus_A=top5_gain,
        conditional_geometry_association=diagnostic, degraded_association_description=diagnostic_degraded,
        interaction_is_descriptive_not_required_positive=True, gate_thresholds=dict(threshold), gates={k:bool(v) for k,v in gates.items()},
        proposal_expand=bool(all(gates.values())), automatic_expansion=False,
        independent_significance_claim=False, user_confirmation_required_before_expansion=True)

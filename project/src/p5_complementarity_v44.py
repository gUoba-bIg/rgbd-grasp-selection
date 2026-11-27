"""Finite cache-only instance complementarity; no model, optimizer, or AP API.

Label-informed union coverage is an empirical property of two cached candidate
sets, not a realizable merger or a bound on the whole RGB-D method.
"""
import hashlib
import numpy as np

CONDITIONS = ('clean', 'noise5', 'missing15', 'shift3', 'joint')


def need(value, message):
    if not value:
        raise RuntimeError(message)


def nearest(array, height, width):
    yy = ((2*np.arange(height)+1)*array.shape[0])//(2*height)
    xx = ((2*np.arange(width)+1)*array.shape[1])//(2*width)
    return array[np.ix_(yy, xx)]


def overlaps(pred, gt, independent=False):
    need(pred.shape == gt.shape and pred.ndim == 2, 'Mask grid differs')
    pids = np.unique(pred); gids = np.unique(gt)
    need(np.all(pids >= 0) and np.all(gids >= 0), 'Negative instance identifier')
    if independent:
        base=int(gids.max())+1
        keys, counts = np.unique(pred.astype(np.int64).ravel()*base+gt.ravel(),return_counts=True)
        pairs=np.column_stack((keys//base,keys%base))
        table = np.zeros((len(pids), len(gids)), np.int64)
        table[np.searchsorted(pids, pairs[:, 0]), np.searchsorted(gids, pairs[:, 1])] = counts
    else:
        pi = np.searchsorted(pids, pred.ravel()); gi = np.searchsorted(gids, gt.ravel())
        table = np.bincount(pi*len(gids)+gi, minlength=len(pids)*len(gids)).reshape(len(pids),len(gids))
    pa = table.sum(1); ga = table.sum(0)
    iou = table / np.maximum(pa[:, None]+ga[None, :]-table, 1)
    pp = pids > 0; gg = gids > 0
    return pids[pp], gids[gg], iou[np.ix_(pp, gg)]


def match(pred, gt, scores, independent=False):
    pids, gids, iou = overlaps(pred, gt, independent)
    need(set(int(x) for x in pids) == {int(x['instance_id']) for x in scores}, 'Score/instance support mismatch')
    need(len(scores) == len(pids), 'Duplicate score identifiers')
    ordered = sorted(scores, key=lambda r: (-float(r['score']), int(r['instance_id'])))
    available = set(int(x) for x in gids); assigned = {}
    for item in ordered:
        pid = int(item['instance_id']); idx = int(np.searchsorted(pids, pid))
        options = [(float(iou[idx, k]), int(g)) for k, g in enumerate(gids) if int(g) in available and iou[idx,k] >= .5]
        if options:
            # Exact tie uses smallest GT identifier, fixed before cache access.
            _, gid = min(options, key=lambda x: (-x[0], x[1]))
            assigned[pid] = gid; available.remove(gid)
    return assigned, set(int(x) for x in gids)


def observed_depth(raw, factor, record, condition, seed=20261023, epoch=10000):
    """NumPy replay of frozen corruption, solely to recover observed validity.

    No RGB image or label is an input. The returned depth hash is checked against
    old v24 receipts wherever those receipts contain it.
    """
    raw = np.asarray(raw, np.float32)/factor
    valid = np.isfinite(raw) & (raw > 0)
    reference = np.where(valid, np.minimum(raw,2.)/2.,0).astype(np.float32)*2
    z = reference.copy(); v = valid.copy(); h,w = v.shape
    ident = f"P5v23|{seed}|{epoch}|{record['scene']}|{record['frame']}|{condition}"
    rng = np.random.RandomState(int(hashlib.sha256(ident.encode()).hexdigest()[:8],16))
    if condition in ('shift3','joint'):
        z[:,3:] = reference[:,:-3]; z[:,:3] = 0
        v[:,3:] = valid[:,:-3]; v[:,:3] = False
    if condition in ('noise5','joint'):
        z = np.where(v,np.clip(z+rng.normal(0,.005,z.shape).astype(np.float32),0,2),0).astype(np.float32)
        v &= z > 0
    if condition in ('missing15','joint'):
        v &= rng.uniform(size=z.shape) >= .15
        hh,ww = h//4,w//4; y=int(rng.randint(0,h-hh+1)); x=int(rng.randint(0,w-ww+1))
        v[y:y+hh,x:x+ww] = False
    z[~v] = 0
    return z,v


def analyze_case(rgb, depth, gt, rgb_scores, depth_scores, quality, valid, identity, independent=False):
    from PIL import Image
    resize = (lambda a: np.asarray(Image.fromarray(a.astype(np.int32)).resize((gt.shape[1],gt.shape[0]),Image.Resampling.NEAREST))) if independent else (lambda a: nearest(a,*gt.shape))
    rm, gids = match(resize(rgb),gt,rgb_scores,independent)
    dm, dgids = match(resize(depth),gt,depth_scores,independent)
    need(gids == dgids and len(gids)>0,'GT scope differs or empty')
    rg=set(rm.values()); dg=set(dm.values()); rescue=dg-rg; lost=rg-dg
    sigma=quality['sigma']; anomaly=quality['anomaly_probability']
    need(sigma.shape == depth.shape == valid.shape == anomaly.shape,'Quality/mask grid mismatch')
    need(np.isfinite(sigma).all() and np.all(sigma>0) and np.isfinite(anomaly).all() and np.all((anomaly>=0)&(anomaly<=1)),'Invalid quality arrays')
    extra=np.exp(-sigma.astype(np.float64)/.005)*(1-.5*anomaly.astype(np.float64))
    candidates=[]
    for item in sorted(depth_scores,key=lambda r:int(r['instance_id'])):
        pid=int(item['instance_id']); mask=depth==pid; support=mask&valid
        label='rescue' if dm.get(pid) in rescue else ('false_positive' if pid not in dm else 'redundant_true_positive')
        # The region is the observed predicted mask. GT assigns audit labels only.
        candidates.append(dict(instance_id=pid,kind=label,matched_gt=dm.get(pid),pixels=int(mask.sum()),
            observed_valid_fraction=float(np.mean(valid[mask])),valid_pixels=int(support.sum()),
            negative_sigma=float(-np.mean(sigma[support],dtype=np.float64)) if support.any() else None,
            negative_anomaly=float(-np.mean(anomaly[support],dtype=np.float64)) if support.any() else None,
            extra=float(np.mean(extra[support])) if support.any() else None,score=float(item['score'])))
    return dict(**identity,ground_truth=len(gids),rgb_tp=len(rg),rgb_fp=len(rgb_scores)-len(rg),
        depth_tp=len(dg),depth_fp=len(depth_scores)-len(dg),rgb_fn=len(gids)-len(rg),
        rescued=len(rescue),lost=len(lost),both_missed=len(gids-(rg|dg)),
        rescued_gt=sorted(rescue),lost_gt=sorted(lost),candidates=candidates)


def matched_quality(rows):
    """One fixed nearest-validity match per positive/negative, without fitting.

    This is matched-pair concordance (not an ordinary global AUROC). Correlated
    masks/frames and residual validity/scene confounding remain explicit.
    """
    pairs=[]
    for scene in sorted({r['scene'] for r in rows}):
        for condition in CONDITIONS:
            pool=[]
            for r in rows:
                if r['scene']==scene and r['condition']==condition:
                    pool += [dict(c,scene=scene,condition=condition,frame=r['frame']) for c in r['candidates'] if c['kind'] in ('rescue','false_positive') and c['extra'] is not None]
            pos=sorted([c for c in pool if c['kind']=='rescue'],key=lambda c:(c['observed_valid_fraction'],c['frame'],c['instance_id']))
            neg=[c for c in pool if c['kind']=='false_positive']; unused=set(range(len(neg)))
            for p in pos:
                options=[(abs(p['observed_valid_fraction']-neg[k]['observed_valid_fraction']),neg[k]['frame'],neg[k]['instance_id'],k) for k in unused]
                if not options:continue
                delta,_,_,k=min(options)
                if delta>.01:continue
                n=neg[k]; unused.remove(k)
                outcomes={key:float(p[key]>n[key])+.5*float(p[key]==n[key]) for key in ('extra','negative_sigma','negative_anomaly','observed_valid_fraction')}
                pairs.append(dict(scene=scene,condition=condition,positive=[p['frame'],p['instance_id']],negative=[n['frame'],n['instance_id']],valid_fraction_difference=float(delta),outcomes=outcomes))
    return pairs


def summarize(rows):
    scenes=sorted({r['scene'] for r in rows}); condition_summary={}
    fields=('ground_truth','rgb_tp','rgb_fp','depth_tp','depth_fp','rgb_fn','rescued','lost','both_missed')
    for condition in CONDITIONS:
        blocks=[]
        for scene in scenes:
            selected=[r for r in rows if r['scene']==scene and r['condition']==condition]
            need(len(selected)==16,'Expected16frames per scene/condition')
            sums={k:sum(r[k] for r in selected) for k in fields}
            blocks.append(dict(scene=scene,counts=sums,
                rescue_recall_gain=float(np.mean([r['rescued']/r['ground_truth'] for r in selected])),
                lost_recall=float(np.mean([r['lost']/r['ground_truth'] for r in selected])),
                net_depth_minus_RGB_recall=float(np.mean([(r['rescued']-r['lost'])/r['ground_truth'] for r in selected])),
                rgb_recall=float(np.mean([r['rgb_tp']/r['ground_truth'] for r in selected])),
                union_recall=float(np.mean([(r['rgb_tp']+r['rescued'])/r['ground_truth'] for r in selected])),
                rescued_fraction_of_rgb_misses=sums['rescued']/sums['rgb_fn'] if sums['rgb_fn'] else None))
        condition_summary[condition]=dict(scenes=blocks,scene_macro_rescue_recall_gain=float(np.mean([b['rescue_recall_gain'] for b in blocks])),counts={k:sum(b['counts'][k] for b in blocks) for k in fields})
    by_scene={str(sc):float(np.mean([next(b for b in condition_summary[d]['scenes'] if b['scene']==sc)['rescue_recall_gain'] for d in CONDITIONS[1:]])) for sc in scenes}
    gain=float(np.mean(list(by_scene.values())))
    pairs=matched_quality(rows); degraded=[p for p in pairs if p['condition']!='clean']; scene_quality={}
    for sc in scenes:
        items=[p for p in degraded if p['scene']==sc]
        if items:
            scene_quality[str(sc)]={k:float(np.mean([p['outcomes'][k] for p in items])) for k in ('extra','negative_sigma','negative_anomaly','observed_valid_fraction')}
    concordance={k:float(np.mean([s[k] for s in scene_quality.values()])) if scene_quality else None for k in ('extra','negative_sigma','negative_anomaly','observed_valid_fraction')}
    gates=dict(complementarity_mean_at_least_2pp=gain>=.02,positive_complementarity_scenes_at_least_6=sum(v>0 for v in by_scene.values())>=6,
        matched_degraded_pairs_at_least_100=len(degraded)>=100,quality_identifiable_scenes_at_least_6=len(scene_quality)>=6,
        extra_concordance_at_least_55pct=concordance['extra'] is not None and concordance['extra']>=.55,
        majority_identifiable_scenes_positive=sum(v['extra']>.5 for v in scene_quality.values())>len(scene_quality)/2)
    return dict(conditions=condition_summary,degraded_scene_macro_rescue_recall_gain=gain,degraded_rescue_by_scene=by_scene,
        quality=dict(matched_pairs=pairs,degraded_pair_count=len(degraded),degraded_scene_concordance=scene_quality,degraded_scene_macro_concordance=concordance),
        screening_gates=gates,training_candidate_worthwhile=all(gates.values()),scientific_adoption_supported=False,
        interpretation='Exposed-development empirical cached-candidate complementarity only; no AP, realized fusion, oracle upper bound, causal quality contribution, calibrated probability, or independent generalization claim.',
        validity_control_limitation='Known-valid coverage matched within1percentage point by scene/condition; all remaining confounding and dependence remain. Concordance is not standard global AUROC. GT only assigns descriptive labels; no router fitted.')

"""Fixed-pool P5/P6 use validation; model proxies, never safety probabilities.

No labels, source IDs, true warp, clean depth, model fitting or neural forward
exist in this module. A changes RGB association; B changes a fixed RGB ray's
depth query. Observed obstacle coordinates and membership are never warped.
"""
from copy import deepcopy
import numpy as np
from src import uncertainty_mixture_v43 as mixture
from src import p6_joint_repair_v45 as calibrated


FROZEN_METHOD = dict(
    version='v51', obstacle_voxel_m=.01, obstacle_max_observed_depth_m=2.,
    finger_length_m=.06, finger_thickness_m=.01, approach_m=.05,
    collision_reference_occupancy=.01, anchors_per_side=32,
    anchor_selection='fixed geometric half-closing pixels; sorted row-major floor(k*n/count)',
    obstacle_weight='every original point retained; reciprocal original 1cm voxel population',
    obstacle_coordinates='original camera XYZ and rays; never shifted by correspondence warp',
    contact='sum_g p(g) * association_A(g) * min(left_B(g),right_B(g))',
    collision='sum_original_points voxel_weight * continuous ray-box union mass / box_voxel_volume',
    collision_ranking_penalty='clip(occupancy_ratio / .01, 0, 1)',
    fixed_u_scale_m=.005, fixed_u_shape='student_t_df3',
    fixed_u_warp='unchanged frozen eight-scene v50 training empirical prior',
    learned_u_warp='unchanged v50 posterior; shared between all anchors and P5 arms',
    learned_u_depth='unchanged v45 depth scale and symmetric H',
    no_u='zero warp and point depth', ray_parallel_tolerance=1e-15,
    invalid_depth='explicit counts; no missing mass reassignment; no claimed unseen-space safety',
    ranking='raw_score * contact_support * (1-collision_proxy)',
    tie='stable descending ranking score then original pool order',
    arms=dict(A=['fixed','none'],B=['quality','none'],C=['fixed','learned'],
              D=['quality','learned'],R=['rgb','none'],C0=['fixed','fixed'],D0=['quality','fixed']),
    scientific_calibration_passed=False,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _guard(guard):
    if guard is not None:
        guard()


def _observations(depth, valid):
    z, ok = mixture.observations(depth, valid)
    # Inherit the P7 observed-cloud domain. Counts disclose its exclusions.
    usable = ok & (z > 0) & (z < FROZEN_METHOD['obstacle_max_observed_depth_m'])
    return z, ok, usable


def pixel_rays(pixel_indices, image_shape, K, original_size):
    ids = np.asarray(pixel_indices)
    h, w = image_shape
    k, size = np.asarray(K, float), np.asarray(original_size)
    require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer)
            and np.all((ids >= 0) & (ids < h*w)), 'Invalid ray pixel indices')
    require(k.shape == (3,3) and np.isfinite(k).all() and k[0,0] > 0 and k[1,1] > 0,
            'Invalid camera intrinsics')
    require(size.shape == (2,) and np.isfinite(size).all() and np.all(size > 0)
            and np.equal(size,np.floor(size)).all(), 'Invalid original image size')
    x, y = ids % w, ids // w
    # Original integer-nearest source sampling, exactly the v43/P7 convention.
    ox = ((2*x+1)*int(size[0])) // (2*w)
    oy = ((2*y+1)*int(size[1])) // (2*h)
    return np.column_stack(((ox-k[0,2])/k[0,0], (oy-k[1,2])/k[1,1], np.ones(len(ids))))


def prepare_scene_geometry(depth, valid, K, original_size):
    z, ok, usable = _observations(depth, valid)
    ids = np.flatnonzero(usable.ravel())
    rays = pixel_rays(ids, z.shape, K, original_size)
    observed_z = z.ravel()[ids]
    points = rays * observed_z[:,None]
    if len(points):
        voxels = np.floor(points/FROZEN_METHOD['obstacle_voxel_m']).astype(np.int64)
        _, inverse, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
        weights = 1./counts[inverse]
        voxel_count = len(counts)
    else:
        weights, voxel_count = np.empty(0), 0
    return dict(points=points, rays=rays, observed_z=observed_z, pixel_indices=ids,
                voxel_weights=weights, image_shape=z.shape, K=np.asarray(K,float).copy(),
                original_size=np.asarray(original_size).copy(), observed_depth=z,
                observed_valid=ok, usable=usable,
                counts=dict(total_pixels=int(z.size), original_valid=int(ok.sum()),
                            retained_obstacle_points=len(ids), original_occupied_voxels=voxel_count,
                            invalid_or_missing_pixels=int((~ok).sum()),
                            valid_outside_inherited_depth_domain=int((ok & ~usable).sum()),
                            targets_removed=0, obstacle_points_warped=0))


def depth_scale_map(depth, valid, depth_model, guard=None):
    """One frozen CPU depth-head evaluation, no new fit or NN forward."""
    z, ok, usable = _observations(depth, valid)
    result = np.zeros(z.shape, dtype=np.float64)
    idx = np.flatnonzero(usable.ravel())
    if len(idx):
        _guard(guard)
        x = mixture.depth_features(z, ok, idx)
        _guard(guard)
        result.ravel()[idx] = mixture.predict_depth(depth_model, x)
        _guard(guard)
    return result


def _candidate(candidate):
    row = np.asarray(candidate, dtype=float)
    require(row.shape == (17,) and np.isfinite(row).all(), 'Invalid candidate row')
    require(row[0] >= 0 and np.all(row[1:4] > 0), 'Nonpositive candidate geometry or negative score')
    rotation, translation = row[4:13].reshape(3,3), row[13:16]
    require(np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=1e-4)
            and abs(np.linalg.det(rotation)-1.) <= 1e-4, 'Invalid candidate rotation')
    return row, rotation, translation


def gripper_boxes(candidate):
    row, _, _ = _candidate(candidate)
    width, height, depth = row[1:4]
    length, thick, approach = .06, .01, .05
    # Boxes have disjoint interiors. x upper faces of palm/approach belong to
    # those boxes for the point-mass no-U baseline, as in the pinned detector.
    lo = np.array([[depth-length,-width/2-thick,-height/2],
                   [depth-length,width/2,-height/2],
                   [depth-length-thick,-width/2-thick,-height/2],
                   [depth-length-thick-approach,-width/2-thick,-height/2]])
    hi = np.array([[depth,-width/2,height/2], [depth,width/2+thick,height/2],
                   [depth-length,width/2+thick,height/2],
                   [depth-length-thick,width/2+thick,height/2]])
    closing_lo = np.array([[depth-length,-width/2,-height/2], [depth-length,0.,-height/2]])
    closing_hi = np.array([[depth,0.,height/2], [depth,width/2,height/2]])
    return dict(lower=lo, upper=hi, upper_x_closed=[False,False,True,True],
                closing_lower=closing_lo, closing_upper=closing_hi,
                volume_m3=float(np.prod(hi-lo,axis=1).sum()), names=['left','right','palm','approach'])


def _fixed_subset(ids, count):
    ids = np.asarray(ids,dtype=np.int64)
    n = min(count,len(ids))
    return ids[np.arange(n)*len(ids)//n] if n else ids.copy()


def candidate_geometry(candidate, scene):
    _, rotation, translation = _candidate(candidate)
    q = (scene['points']-translation)@rotation
    boxes = gripper_boxes(candidate)
    lower, upper = boxes['closing_lower'], boxes['closing_upper']
    # y=0 is assigned once to the right half; global exterior faces are open.
    overall = (q[:,0] > lower[0,0]) & (q[:,0] < upper[0,0])
    overall &= (q[:,1] > lower[0,1]) & (q[:,1] < upper[1,1])
    overall &= (q[:,2] > lower[0,2]) & (q[:,2] < upper[0,2])
    ids = scene['pixel_indices']
    side_ids = [ids[overall & (q[:,1] < 0)], ids[overall & (q[:,1] >= 0)]]
    anchors = [_fixed_subset(a,FROZEN_METHOD['anchors_per_side']) for a in side_ids]
    return dict(closing_pixels=ids[overall], side_anchor_pixels=anchors,
                side_closing_counts=[len(a) for a in side_ids], forbidden_boxes=boxes,
                local_rays=scene['rays']@rotation, local_origin=-translation@rotation,
                local_observed=q)


def ray_box_intervals(local_rays, local_origin, lower, upper, lower_closed=None):
    """Exact positive-z interval along r*z+origin through an open box."""
    rays, origin = np.asarray(local_rays,float), np.asarray(local_origin,float)
    low, high = np.asarray(lower,float), np.asarray(upper,float)
    closed = np.zeros(3,bool) if lower_closed is None else np.asarray(lower_closed,bool)
    require(rays.ndim == 2 and rays.shape[1] == 3 and origin.shape == low.shape == high.shape == (3,)
            and np.isfinite(rays).all() and np.isfinite(origin).all()
            and np.isfinite(low).all() and np.isfinite(high).all() and np.all(high>low)
            and closed.shape == (3,),
            'Invalid ray or box')
    lo, hi = np.zeros(len(rays)), np.full(len(rays),np.inf)
    hit = np.ones(len(rays),bool)
    for axis in range(3):
        r = rays[:,axis]
        parallel = np.abs(r) < FROZEN_METHOD['ray_parallel_tolerance']
        above = origin[axis] >= low[axis] if closed[axis] else origin[axis] > low[axis]
        hit &= ~parallel | (above & (origin[axis] < high[axis]))
        a = np.divide(low[axis]-origin[axis],r,out=np.full(len(r),-np.inf),where=~parallel)
        b = np.divide(high[axis]-origin[axis],r,out=np.full(len(r),np.inf),where=~parallel)
        lo = np.maximum(lo,np.minimum(a,b)); hi = np.minimum(hi,np.maximum(a,b))
    hit &= hi > lo
    return lo, hi, hit


def continuous_interval_mass(observed_z, scale, lo, hi, hit, shape_model=None):
    z,s,a,b = np.broadcast_arrays(np.asarray(observed_z,float), np.asarray(scale,float),
                                  np.asarray(lo,float),np.asarray(hi,float))
    good = np.broadcast_to(np.asarray(hit,bool),z.shape)
    require(np.isfinite(z).all() and np.isfinite(s).all() and np.all(s>0)
            and not np.isnan(a).any() and not np.isnan(b).any(), 'Invalid continuous event input')
    answer = np.zeros(z.shape)
    if good.any():
        answer[good] = (calibrated.residual_cdf(z[good]-a[good],s[good],shape_model)
                        - calibrated.residual_cdf(z[good]-b[good],s[good],shape_model))
    require(np.isfinite(answer).all() and np.all(answer >= -1e-14)
            and np.all(answer <= 1+1e-14), 'Invalid CDF interval probability')
    return np.clip(answer,0.,1.)


def collision_proxies(scene, geometry, scale_map, shape_model, guard=None):
    """Three U definitions on every original observed obstacle, no mask input."""
    smap = np.asarray(scale_map,float)
    require(smap.shape == scene['image_shape'] and np.isfinite(smap).all(), 'Invalid scale map')
    sigma = smap.ravel()[scene['pixel_indices']]
    require(np.all(sigma>0), 'Nonpositive observed scale')
    totals = dict(none=0.,fixed=0.,learned=0.)
    boxes, q = geometry['forbidden_boxes'], geometry['local_observed']
    details = []
    for b, (lower,upper) in enumerate(zip(boxes['lower'],boxes['upper'])):
        _guard(guard)
        lo,hi,hit = ray_box_intervals(geometry['local_rays'],geometry['local_origin'],lower,upper)
        point = np.all((q > lower) & (q < upper),axis=1)
        if boxes['upper_x_closed'][b]:
            point = (q[:,0]>lower[0]) & (q[:,0]<=upper[0]) & np.all((q[:,1:]>lower[1:]) & (q[:,1:]<upper[1:]),axis=1)
        weights = scene['voxel_weights']
        found = np.flatnonzero(hit)
        per = dict(none=float(weights@point),fixed=0.,learned=0.)
        if len(found):
            z, a, c, w = scene['observed_z'][found],lo[found],hi[found],weights[found]
            per['fixed'] = float(w@continuous_interval_mass(z,np.full(len(found),.005),a,c,np.ones(len(found),bool)))
            per['learned'] = float(w@continuous_interval_mass(z,sigma[found],a,c,np.ones(len(found),bool),shape_model))
        for key in totals: totals[key] += per[key]
        details.append(dict(box=boxes['names'][b],intersecting_original_rays=len(found),weighted_occupancy=per))
    divisor = boxes['volume_m3']/FROZEN_METHOD['obstacle_voxel_m']**3
    result = {}
    for key,value in totals.items():
        ratio = value/divisor
        result[key] = dict(occupancy_ratio=ratio, collision_proxy=float(np.clip(ratio/.01,0.,1.)),
                           collision_expected_occupancy_proxy=float(np.clip(ratio/.01,0.,1.)),
                           expected_weighted_occupied_voxels=value,forbidden_voxel_volume=divisor,
                           unknown=len(scene['points'])==0,obstacle_points_retained=len(scene['points']))
    result['boxes'] = details
    return result


def association_by_warp(instances, foreground_probability, seed_pixel, closing_pixels):
    labels, fg = np.asarray(instances), np.asarray(foreground_probability,float)
    require(labels.ndim==2 and np.issubdtype(labels.dtype,np.integer) and np.all(labels>=0)
            and fg.shape==labels.shape and np.isfinite(fg).all() and np.all((fg>=0)&(fg<=1)),
            'Invalid predicted instance association inputs')
    require(isinstance(seed_pixel,(int,np.integer)) and 0<=int(seed_pixel)<labels.size, 'Invalid candidate seed')
    ids=np.asarray(closing_pixels)
    require(ids.ndim==1 and np.issubdtype(ids.dtype,np.integer) and np.all((ids>=0)&(ids<labels.size)), 'Invalid closing pixels')
    h,w=labels.shape; target=int(labels.ravel()[int(seed_pixel)])
    values=np.zeros(len(mixture.CANDIDATES)); missing=np.zeros_like(values)
    if not len(ids):
        return dict(target_id=target,unknown=True,association=values,missing_fraction=np.ones_like(values),closing_count=0)
    x,y=ids%w,ids//w
    for k,(dx,dy) in enumerate(mixture.CANDIDATES):
        sx,sy=x-dx,y-dy
        inside=(sx>=0)&(sx<w)&(sy>=0)&(sy<h)
        missing[k]=1-float(inside.sum())/len(ids)
        if target and inside.any():
            values[k]=float(np.sum(fg[sy[inside],sx[inside]]*(labels[sy[inside],sx[inside]]==target)))/len(ids)
    return dict(target_id=target,unknown=target==0,association=values,missing_fraction=missing,closing_count=len(ids))


def anchor_support_by_warp(scene, geometry, candidate, scale_map, shape_model):
    _,rotation,translation=_candidate(candidate)
    depth,valid=scene['observed_depth'],scene['usable']
    h,w=depth.shape; smap=np.asarray(scale_map,float)
    require(smap.shape==depth.shape and np.isfinite(smap).all() and np.all(smap[valid]>0),'Invalid anchor scale')
    result={key:np.zeros((2,len(mixture.CANDIDATES))) for key in ('none','fixed','learned')}
    missing=np.ones((2,len(mixture.CANDIDATES)))
    for side,ids in enumerate(geometry['side_anchor_pixels']):
        if not len(ids):continue
        world_rays=pixel_rays(ids,depth.shape,scene['K'],scene['original_size'])
        rays=world_rays@rotation
        lo,hi,hit=ray_box_intervals(rays,-translation@rotation,
                                  geometry['forbidden_boxes']['closing_lower'][side],
                                  geometry['forbidden_boxes']['closing_upper'][side],
                                  lower_closed=[False,side==1,False])
        x,y=ids%w,ids//w
        for k,(dx,dy) in enumerate(mixture.CANDIDATES):
            sx,sy=x+dx,y+dy
            inside=(sx>=0)&(sx<w)&(sy>=0)&(sy<h)
            cx,cy=np.clip(sx,0,w-1),np.clip(sy,0,h-1)
            good=inside&valid[cy,cx]
            missing[side,k]=1-float(good.sum())/len(ids)
            if not good.any():continue
            z=depth[cy[good],cx[good]]; a,b,active=lo[good],hi[good],hit[good]
            # A point mass can lie exactly on the owned y=0 half boundary.
            # CDF endpoints have zero continuous mass, but the no-U event must
            # use the same explicit closed-right/open-exterior geometry as
            # anchor selection, including nonparallel rays.
            local_points=(world_rays[good]*z[:,None]-translation)@rotation
            lower=geometry['forbidden_boxes']['closing_lower'][side]
            upper=geometry['forbidden_boxes']['closing_upper'][side]
            point_inside=np.all((local_points>lower)&(local_points<upper),axis=1)
            if side==1:
                point_inside=((local_points[:,0]>lower[0])&(local_points[:,0]<upper[0])
                              &(local_points[:,1]>=lower[1])&(local_points[:,1]<upper[1])
                              &(local_points[:,2]>lower[2])&(local_points[:,2]<upper[2]))
            result['none'][side,k]=float(point_inside.sum())/len(ids)
            result['fixed'][side,k]=float(continuous_interval_mass(z,np.full(len(z),.005),a,b,active).sum())/len(ids)
            result['learned'][side,k]=float(continuous_interval_mass(z,smap[cy[good],cx[good]],a,b,active,shape_model).sum())/len(ids)
    return dict(side_support=result,missing_fraction=missing,
                anchor_counts=[len(a) for a in geometry['side_anchor_pixels']],
                anchor_pixel_indices=[a.tolist() for a in geometry['side_anchor_pixels']])


def aggregate_contact(association, side_support, probabilities):
    a,b,p=np.asarray(association,float),np.asarray(side_support,float),np.asarray(probabilities,float)
    require(a.shape==p.shape==(81,) and b.shape==(2,81) and np.isfinite(a).all()
            and np.isfinite(b).all() and np.all((a>=0)&(a<=1)) and np.all((b>=0)&(b<=1)), 'Invalid contact components')
    mixture.probability_vector(p,81)
    return float(p@(a*np.minimum(b[0],b[1])))


def build_features(case, *, learned_posterior, fixed_prior, depth_scale_map, shape_model, guard=None):
    """Return all seven rankings and replayable details, in fixed pool order.

    The caller owns inference/ledger/resource control. This performs no NN call.
    Pool size may be below 20; it is preserved, never padded/dropped/reselected.
    """
    _guard(guard)
    learned=mixture.probability_vector(learned_posterior,81)
    prior=mixture.probability_vector(fixed_prior,81)
    require(np.all(learned>0)&np.all(prior>0), 'Learned and prior support must retain all 81 candidates')
    zero=np.zeros(81);zero[int(np.flatnonzero(np.all(mixture.CANDIDATES==0,axis=1))[0])]=1.
    probabilities=dict(none=zero,fixed=prior,learned=learned)
    pool=np.asarray(case['pool_indices']);candidates=np.asarray(case['candidates'])
    seeds=np.asarray(case.get('seed_pixels',case.get('pixels')))
    require(candidates.ndim==2 and candidates.shape[1]==17 and seeds.shape==(len(candidates),)
            and np.issubdtype(seeds.dtype,np.integer), 'Invalid fixed candidate pool inputs')
    require(pool.ndim==1 and np.issubdtype(pool.dtype,np.integer) and len(pool)<=20
            and len(np.unique(pool))==len(pool) and np.all((pool>=0)&(pool<len(candidates))), 'Invalid fixed pool indices')
    scene=prepare_scene_geometry(case['observed_depth'],case['observed_valid'],case['K'],case['original_size'])
    arms=case['arms']; rows=[];scores={name:[] for name in FROZEN_METHOD['arms']}
    for slot,idx in enumerate(pool):
        _guard(guard)
        candidate=candidates[idx];geo=candidate_geometry(candidate,scene)
        collision=collision_proxies(scene,geo,depth_scale_map,shape_model,guard)
        anchors=anchor_support_by_warp(scene,geo,candidate,depth_scale_map,shape_model)
        association={name:association_by_warp(arms[name]['instances'],arms[name]['foreground_probability'],int(seeds[idx]),geo['closing_pixels'])
                     for name in ('fixed','quality','rgb')}
        group={}
        for name,(mask,uncertainty) in FROZEN_METHOD['arms'].items():
            a=association[mask];p=probabilities[uncertainty]
            support=aggregate_contact(a['association'],anchors['side_support'][uncertainty],p)
            cp=collision[uncertainty]
            score=float(candidate[0])*support*(1-cp['collision_proxy'])
            reasons=[]
            if a['target_id']==0: reasons.append('seed_has_no_predicted_target')
            if not all(anchors['anchor_counts']): reasons.append('empty_geometric_contact_half')
            if cp['unknown']:reasons.append('no_observed_obstacles_in_inherited_domain')
            missing=(anchors['missing_fraction']@p).tolist()
            if max(missing)>0:reasons.append('represented_contact_depth_mass_is_missing')
            group[name]=dict(mask=mask,uncertainty=uncertainty,target_id=a['target_id'],
                            contact_support=support,occupancy_ratio=cp['occupancy_ratio'],
                            collision_proxy=cp['collision_proxy'],collision_expected_occupancy_proxy=cp['collision_proxy'],
                            rank_score=score,unknown=bool(reasons),unknown_reasons=reasons,
                            contact_missing_mass_by_side=missing,association_out_of_image_mass=float(a['missing_fraction']@p))
            scores[name].append(score)
        rows.append(dict(pool_slot=slot,pool_index=int(idx),seed_pixel=int(seeds[idx]),raw_score=float(candidate[0]),
                         closing_count=len(geo['closing_pixels']),side_closing_counts=geo['side_closing_counts'],
                         contact_anchor_pixels=anchors['anchor_pixel_indices'],anchor_counts=anchors['anchor_counts'],
                         association_by_warp={key:dict(target_id=a['target_id'],unknown=a['unknown'],
                             values=a['association'].tolist(),missing_fraction=a['missing_fraction'].tolist()) for key,a in association.items()},
                         side_support_by_warp={key:value.tolist() for key,value in anchors['side_support'].items()},
                         contact_missing_by_warp=anchors['missing_fraction'].tolist(),collision=collision,arms=group))
    _guard(guard)
    return dict(method=deepcopy(FROZEN_METHOD),scores=scores,candidates=rows,
                rankings={name:np.argsort(-np.asarray(values),kind='stable').tolist() for name,values in scores.items()},
                learned_risk={name:[r['arms'][name] for r in rows] for name in scores},
                geometry_counts=scene['counts'],candidate_count=len(pool),pool_indices=pool.tolist(),
                learned_posterior=learned.tolist(),fixed_prior=prior.tolist(),
                actual_neural_forward_calls=0,new_fits=0,uncalibrated_model_proxy=True,
                scientific_calibration_passed=False,unseen_space_claimed_safe=False)

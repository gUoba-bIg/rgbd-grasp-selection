"""Pure v45 A/B/joint evaluation; immutable v43 definitions and17 gates retained.

No filesystem access, runner state, fit, command-line entry point or mutation of
frozen module globals occurs here. Frame arrays and fitted kernel are supplied.
conditional_width95 is the within-component average, not a mixture quantile width.
The original union_width95 criterion remains the actual B sharpness gate.
"""
import numpy as np
from src import uncertainty_mixture_v43 as core
from src import correspondence_v40 as old
from src import p6_joint_repair_v45 as repair
from scripts.p6_mixture_v43 import (observation,independent_source,fixed_query,
    strata,summarize_point_values,t_logpdf,masked_mean)
need=core.require
MODES=("nominal","noise5","noise5_missing15")
WARPS=old.WARPS
LEVELS=(.5,.8,.95)

def _scope_rows(rows,split):
    need(split in ("calibration","evaluation"),"Only fixed calibration/evaluation scopes")
    selected=[r for r in rows if r["split"]==split]
    scenes=(30,31,34,38) if split=="calibration" else (42,43,57,93)
    need(len(selected)==432 and tuple(sorted({r["scene"] for r in selected}))==scenes,"Expected432 cases in the fixed four-scene split")
    need(all(sum(r["scene"]==s for r in selected)==108 for s in scenes),"Unequal fixed scene contribution")
    need(all("key" in r for r in selected) and len({r["key"] for r in selected})==432,"Missing/duplicate case identity")
    return selected,scenes

def aggregate_warp(rows,split="calibration"):
    """Original six warp gates, with no depth/GT feature access or fitting."""
    rr,scenes=_scope_rows(rows,split)
    wp={k:mean(rr,lambda r:r["warp_"+k]["nll"]) for k in ("method","prior")}
    wb={k:mean(rr,lambda r:r["warp_"+k]["brier"]) for k in ("method","prior")}
    levels={str(l):{k:{f:mean(rr,lambda r:r["warp_"+k]["levels"][str(l)][f]) for f in ("covered","mass","size")} for k in ("method","prior")} for l in LEVELS}
    worst=min(mean([r for r in rr if r["scene"]==s],lambda r:r["warp_method"]["levels"]["0.95"]["covered"]) for s in scenes)
    gates=dict(warp_nll_gain=wp["prior"]-wp["method"]>=.05,warp_brier_noninferior=wb["method"]<=wb["prior"],
        warp95_coverage=levels["0.95"]["method"]["covered"]>=.90,
        warp_set_sharper=levels["0.95"]["method"]["size"]<=.9*levels["0.95"]["prior"]["size"],
        warp_multilevel_reliability=all(abs(v["method"]["covered"]-v["method"]["mass"])<=.05 for v in levels.values()),
        worst_scene_warp=worst>=.80)
    return dict(split=split,warp_nll=wp,warp_brier=wb,warp_levels=levels,worst_scene_warp95=worst,case_count=432),{k:bool(v) for k,v in gates.items()}

def evaluate_case(frame,r,delta,mode,p,depth_model,prior,kernel):
    z,ok,sid=observation(frame,r,delta,mode)
    query=fixed_query(frame);idx=query["ids"];h,w=z.shape;v,u=np.divmod(idx,w);uv=np.column_stack([u,v])
    did=int(np.flatnonzero(np.all(core.CANDIDATES==delta,axis=1))[0])
    ou=u+delta[0];ov=v+delta[1];visible=ok[ov,ou]
    need(np.array_equal(sid[ov[visible],ou[visible]],idx[visible]),"Source tracking differs")
    # Only observed candidate samples supply predicted scales.
    cx=u[:,None]+core.CANDIDATES[None,:,0];cy=v[:,None]+core.CANDIDATES[None,:,1]
    read_ids=np.unique(np.r_[((cy*w+cx).ravel()),(ov*w+ou)[visible]])
    read_ids=read_ids[ok.ravel()[read_ids]]
    sf=core.depth_features(z,ok,read_ids);scale_map=np.full(z.shape,.005)
    scale_map.ravel()[read_ids]=core.predict_depth(depth_model,sf)
    masks=strata(frame,query,delta,"A");bmasks=strata(frame,query,delta,"B")
    original=old.xyz_at(uv,z[v,u],frame["K"],frame["original_size"],(w,h))
    before=z.copy()
    A=repair.point_mixture_A(np.column_stack([ou[visible],ov[visible]]),z[ov[visible],ou[visible]],
        scale_map[ov[visible],ou[visible]],p,shape_model=kernel)
    original_A=old.xyz_at(np.column_stack([ou[visible],ov[visible]]),z[ov[visible],ou[visible]],frame["K"],frame["original_size"],(w,h))
    propagated=repair.propagate_components(A,frame["K"],original_size=frame["original_size"],grid_size=(w,h))
    need(np.array_equal(propagated["observed_xyz"],original_A),"Original observation geometry changed")
    reference=frame["depth"].ravel()[idx]
    residual=z[ov[visible],ou[visible]]-reference[visible];sc=scale_map[ov[visible],ou[visible]]
    values={"nll":-repair.residual_logpdf(residual,sc,kernel),"fixed5mm_nll":-t_logpdf(residual,np.full_like(sc,.005)),
            "width95":2*repair.residual_ppf(.975,sc,kernel)}
    for level in LEVELS:values["coverage"+str(level)]=(np.abs(residual)<=repair.residual_ppf((1+level)/2,sc,kernel)).astype(float)
    ajoint=repair.hierarchical_statistics(A,reference[visible],did)
    for row in ajoint:
        values["joint_coverage"+str(row["level"])]=row["joint_source_covered"].astype(float)
        values["joint_probability"+str(row["level"])]=row["latent_joint_continuous_mass"]
    values["joint_nll"]=values["nll"]-np.log(p[did])
    amasks={k:m[visible] for k,m in masks.items()}
    Ametrics=summarize_point_values(values,amasks,{k:int(m.sum()) for k,m in masks.items()})
    Bmetrics={}
    distributions={"method":(p,scale_map,kernel),"prior_same_scale":(prior,scale_map,kernel),"prior_fixed5mm":(prior,np.full(z.shape,.005),None),
                   "uniform_fixed5mm":(np.ones(81)/81,np.full(z.shape,.005),None),
                   "single_mode":(np.eye(81)[int(np.argmax(p))],scale_map,kernel)}
    for name,(weights,sm,distribution_kernel) in distributions.items():
        B=repair.target_mixture_B(z,ok,sm,uv,weights,shape_model=distribution_kernel)
        stats=repair.mixture_statistics(B,reference)
        missing=np.asarray(B["missing_probability"])
        logpdf=np.asarray(stats["logpdf_subdensity"])
        # Observable outcome is (oracle candidate visible, source reference depth).
        nll=np.where(visible,-logpdf,-np.log(np.maximum(missing,1e-300)))
        zero_event=np.where(visible,~np.isfinite(logpdf),missing==0)
        vals=dict(joint_nll_capped_diagnostic=np.minimum(nll,1e6),missing_brier=(missing-(~visible))**2,
                  predicted_missing=missing,oracle_missing=(~visible).astype(float),
                  zero_mass_event=zero_event.astype(float))
        group=summarize_point_values(vals,bmasks)
        for k,m in bmasks.items():
            group[k]["zero_probability_event_count"]=int(np.sum(m&zero_event))
            group[k]["joint_nll"]=masked_mean(nll,m) if not np.any(m&zero_event) else None
            group[k]["uncapped_joint_nll_is_infinite"]=bool(np.any(m&zero_event))
        # Conditional numeric depth coverage never silently drops the missing count.
        for k,m in bmasks.items():
            selected=m&visible
            group[k]["visible_n"]=int(selected.sum());group[k]["missing_n"]=int((m&~visible).sum())
            for j,level in enumerate(LEVELS):group[k]["conditional_coverage"+str(level)]=masked_mean(stats["conditional_covered"][:,j].astype(float),selected)
            group[k]["conditional_width95"]=masked_mean(stats["within_component_mean_width95"],selected)
        joint=repair.hierarchical_statistics(B,reference,did,levels=(.95,))
        jrow=joint[0]
        for k,m in bmasks.items():
            selected=m&visible
            group[k]["conditional_undefined_count"]=int(np.sum(selected&~stats["conditional_eligible"]))
            group[k]["union_width95"]=masked_mean(jrow["numerical_union_width"],selected)
            group[k]["union_coverage95"]=masked_mean(jrow["numerical_union_covered"].astype(float),selected)
            group[k]["union_full_outcome_coverage95"]=masked_mean(
                np.where(visible,jrow["numerical_union_covered"],jrow["selected_missing_probability"]>0).astype(float),m)
            group[k]["union_pieces95"]=masked_mean(jrow["numerical_union_piece_count"],selected)
        Bmetrics[name]=group
    need(np.array_equal(z,before) and np.array_equal(original,old.xyz_at(uv,z[v,u],frame["K"],frame["original_size"],(w,h))),"Observation overwritten")
    def warp_stats(weights):
        value=core.warp_statistics(weights,did)
        value["levels"]={str(v["level"]):dict(covered=v["covered"],size=v["size"],mass=v["probability"]) for v in value.pop("credible_sets")}
        return value
    return dict(warp_method=warp_stats(p),warp_prior=warp_stats(prior),
        A=Ametrics,B=Bmetrics,total_reference_query=len(idx),oracle_visible=int(visible.sum()),oracle_missing=int((~visible).sum()),
        candidate_support_checked=True,supported_visible_candidates=int(visible.sum()),missing_true_candidates=int((~visible).sum()),
        fixed_domain_count=query["domain_n"],excluded_border=query["excluded_border"],original_xyz_unchanged=True,
        geometry_covariance_finite=bool(np.isfinite(propagated["conditional_covariance_xyz"]).all()))

def mean(items,fn):
    values=[fn(v) for v in items];values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None

def aggregate(rows,split="evaluation"):
    rr,expected_scenes=_scope_rows(rows,split)
    wp={k:mean(rr,lambda r:r["warp_"+k]["nll"]) for k in ("method","prior")}
    wb={k:mean(rr,lambda r:r["warp_"+k]["brier"]) for k in ("method","prior")}
    levels={}
    for l in LEVELS:
        tag=str(l)
        levels[tag]={k:{field:mean(rr,lambda r:r["warp_"+k]["levels"][tag][field]) for field in ("covered","mass","size")} for k in ("method","prior")}
    # Equal frame/warp/mode means; every scene contributes equal cases.
    groups={}
    for mode in MODES:
        subset=[r for r in rr if r["mode"]==mode]
        groups[mode]={}
        for region in ("main","boundary","same_instance","cross_instance","depth_jump_proxy"):
            groups[mode][region]={
                "A":{field:mean(subset,lambda r:r["A"][region].get(field)) for field in ("nll","joint_nll","fixed5mm_nll","width95","coverage0.5","coverage0.8","coverage0.95","joint_coverage0.5","joint_coverage0.8","joint_coverage0.95","joint_probability0.5","joint_probability0.8","joint_probability0.95")},
                "B":{name:{field:mean(subset,lambda r:r["B"][name][region].get(field)) for field in
                    ("joint_nll","joint_nll_capped_diagnostic","zero_probability_event_count","missing_brier","predicted_missing","oracle_missing","conditional_coverage0.5","conditional_coverage0.8","conditional_coverage0.95","conditional_width95","union_width95","union_coverage95","union_full_outcome_coverage95","union_pieces95","conditional_undefined_count")}
                    for name in ("method","prior_same_scale","prior_fixed5mm","single_mode","uniform_fixed5mm")}}
    for mode in MODES:
        subset=[r for r in rr if r["mode"]==mode]
        for region in groups[mode]:
            groups[mode][region]["A_denominators"]={
                "reference_n":sum(r["A"][region]["reference_n"] for r in subset),
                "observed_n":sum(r["A"][region]["n"] for r in subset),
                "nonempty_cases":sum(r["A"][region]["n"]>0 for r in subset)}
            for name in groups[mode][region]["B"]:
                target=groups[mode][region]["B"][name]
                source=[r["B"][name][region] for r in subset]
                target["denominators"]={key:sum(v[key] for v in source) for key in ("n","visible_n","missing_n","conditional_undefined_count","zero_probability_event_count")}
                target["nonempty_cases"]=sum(v["n"]>0 for v in source)
                if any(v["uncapped_joint_nll_is_infinite"] for v in source):
                    target["joint_nll"]=None;target["uncapped_joint_nll_is_infinite"]=True
                else:target["uncapped_joint_nll_is_infinite"]=False
    worst_warp=min(mean([r for r in rr if r["scene"]==s],lambda r:r["warp_method"]["levels"]["0.95"]["covered"]) for s in expected_scenes)
    worst_depth=min(mean([r for r in rr if r["scene"]==s and r["mode"]!="nominal"],lambda r:r["A"]["main"]["coverage0.95"]) for s in expected_scenes)
    numeric=[groups[m] for m in MODES[1:]]
    method_B=mean(rr,lambda r:r["B"]["method"]["main"]["joint_nll"])
    prior_B=mean(rr,lambda r:r["B"]["prior_same_scale"]["main"]["joint_nll"])
    method_zero=sum(r["B"]["method"]["main"]["zero_probability_event_count"] for r in rr)
    prior_zero=sum(r["B"]["prior_same_scale"]["main"]["zero_probability_event_count"] for r in rr)
    if method_zero:method_B=None
    if prior_zero:prior_B=None
    gates=dict(warp_nll_gain=wp["prior"]-wp["method"]>=.05,warp_brier_noninferior=wb["method"]<=wb["prior"],
        warp95_coverage=levels["0.95"]["method"]["covered"]>=.90,
        warp_set_sharper=levels["0.95"]["method"]["size"]<=.9*levels["0.95"]["prior"]["size"],
        warp_multilevel_reliability=all(abs(v["method"]["covered"]-v["method"]["mass"])<=.05 for v in levels.values()),
        worst_scene_warp=worst_warp>=.80,worst_scene_depth=worst_depth>=.80,
        source_depth95=all(.90<=v["main"]["A"]["coverage0.95"]<=.98 and v["boundary"]["A"]["coverage0.95"]>=.85 for v in numeric),
        source_depth_sharpness=all(v["main"]["A"]["width95"]<=.04 and v["boundary"]["A"]["width95"]<=.08 for v in numeric),
        source_depth_nll_gain=np.mean([v["main"]["A"]["fixed5mm_nll"]-v["main"]["A"]["nll"] for v in numeric])>=.05,
        target_joint_nll_noninferior=method_zero==0 and prior_zero==0 and method_B is not None and prior_B is not None and method_B<=prior_B,
        target_missing_brier_noninferior=mean(rr,lambda r:r["B"]["method"]["main"]["missing_brier"])<=mean(rr,lambda r:r["B"]["prior_same_scale"]["main"]["missing_brier"])+1e-12,
        target_conditional_coverage=all(.90<=v["main"]["B"]["method"]["conditional_coverage0.95"]<=.98 and v["boundary"]["B"]["method"]["conditional_coverage0.95"]>=.85 for v in numeric),
        source_joint95=all(v["main"]["A"]["joint_coverage0.95"]>=.90 and v["boundary"]["A"]["joint_coverage0.95"]>=.85 for v in numeric),
        target_union_sharpness=np.mean([v["main"]["B"]["method"]["union_width95"] for v in numeric])<=np.mean([v["main"]["B"]["prior_same_scale"]["union_width95"] for v in numeric]),
        target_union95=all(v["main"]["B"]["method"]["union_coverage95"]>=.90 and v["boundary"]["B"]["method"]["union_coverage95"]>=.85 for v in numeric),
        observation_preserved=all(r["original_xyz_unchanged"] and r["geometry_covariance_finite"] for r in rr))
    gates={k:bool(v) for k,v in gates.items()}
    summary=dict(split=split,warp_nll=wp,warp_brier=wb,warp_levels=levels,worst_scene_warp95=worst_warp,worst_scene_depth95=worst_depth,
        groups=groups,target_joint_nll_method=method_B,target_joint_nll_prior_same_scale=prior_B,target_zero_probability_events=dict(method=method_zero,prior_same_scale=prior_zero),
        evaluated_cases=432,calibration_cases=432,total_reference_queries=sum(r["total_reference_query"] for r in rr),
        oracle_visible_queries=sum(r["oracle_visible"] for r in rr),oracle_missing_queries=sum(r["oracle_missing"] for r in rr),
        numeric_coverage_is_conditional_on_oracle_visibility=True,candidate_support_checked=all(r["candidate_support_checked"] for r in rr),
        candidate_support_scope="All oracle-visible fixed queries have exact source-ID in the 81 grid; missing remains missing; only known 9 synthetic warps")
    return summary,gates

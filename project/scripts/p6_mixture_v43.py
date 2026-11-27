"""Finite P6 v43 development experiment. Ground truth is restricted to supervision/statistics."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.p6_mixture_common_v43 import *
import argparse,math
import numpy as np
from scipy.special import stdtr,stdtrit,gammaln
from src import uncertainty_mixture_v43 as core
from src import correspondence_v40 as old
from scripts.p6_correspondence_v40 import load_frame
from scripts.pipeline import project_lock
MODES=("nominal","noise5","noise5_missing15")
WARPS=old.WARPS
LEVELS=(.5,.8,.95)
DATA=ROOT/"logs/P6/correspondence_v40/data_audit.json"

def arrays_save(path,**values):
    need(not path.exists(),"Existing immutable array output")
    with path.open("xb") as f:np.savez_compressed(f,**values)
def arrays_read(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def seed_for(r,delta,mode):
    return int(hashlib.sha256(f"P6v43|20261101|{r['scene']}|{r['frame']}|{delta}|{mode}".encode()).hexdigest()[:8],16)
def observation(frame,r,delta,mode):
    z,ok,sid=old.warp_integer(frame["depth"],frame["valid"],delta)
    z=z.astype(np.float64)
    rng=np.random.RandomState(seed_for(r,delta,mode))
    if mode!="nominal":
        z[ok]+=rng.normal(0,.005,z.shape)[ok]
        ok&=z>0
    if mode=="noise5_missing15":
        ok&=rng.uniform(size=z.shape)>=.15
        h,w=z.shape;y=int(rng.randint(0,h-h//4+1));x=int(rng.randint(0,w-w//4+1))
        ok[y:y+h//4,x:x+w//4]=False
    z[~ok]=0;sid[~ok]=-1
    return z,ok,sid
def independent_source(frame,r,delta,mode):
    h,w=frame["depth"].shape;v,u=np.indices((h,w));x=u-delta[0];y=v-delta[1]
    inside=(x>=0)&(x<w)&(y>=0)&(y<h)
    x=x.clip(0,w-1);y=y.clip(0,h-1)
    ok=inside&frame["valid"][y,x];z=frame["depth"][y,x].astype(float);z[~ok]=0
    rng=np.random.RandomState(seed_for(r,delta,mode))
    if mode!="nominal":
        noise=rng.normal(0,.005,z.shape);z=np.where(ok,z+noise,0);ok=ok&(z>0)
    if mode=="noise5_missing15":
        keep=rng.uniform(size=z.shape)>=.15
        top=int(rng.randint(0,h-h//4+1));left=int(rng.randint(0,w-w//4+1))
        keep[top:top+h//4,left:left+w//4]=False;ok&=keep
    return np.where(ok,z,0),ok,np.where(ok,y*w+x,-1)
def fixed_query(frame,main_limit=512,boundary_limit=128):
    domain=frame["eligible"].copy()
    domain[:8]=domain[-8:]=False;domain[:,:8]=domain[:,-8:]=False
    labels=frame["labels"];boundary=np.zeros(labels.shape,bool)
    for axis in (0,1):
        boundary|=labels!=np.roll(labels,1,axis)
        boundary|=labels!=np.roll(labels,-1,axis)
    from scipy.ndimage import binary_dilation
    boundary=binary_dilation(boundary,iterations=1)
    main=old.spaced(domain,main_limit);extra=old.spaced(domain&boundary,boundary_limit);ids=np.union1d(main,extra)
    need(len(main)>0 and len(extra)>0,"Empty prespecified evaluation stratum")
    return dict(ids=ids,main=np.isin(ids,main),boundary=boundary.ravel()[ids],domain_n=int(domain.sum()),
                excluded_border=int(frame["eligible"].sum()-domain.sum()))
def key_for(split,r,delta,mode):return f"{split}/{r['scene']:04d}/{r['frame']}/{delta[0]}_{delta[1]}/{mode}"
def iter_frames():
    data=read(DATA)
    for split,records in data["records"].items():
        for r in records:yield split,r,load_frame(data,r,labels=True)
def fit_samples(frame,z,ok,sid,limit=256):
    eligible=ok.copy();ids=sid[ok];eligible[ok]=frame["eligible"].ravel()[ids]
    # Sampling for supervised fit only; these masks never enter observable_case_features.
    selected=old.spaced(eligible,limit)
    need(len(selected)>8,"Too few supervised depth samples")
    f=core.depth_features(z,ok,selected)
    residual=z.ravel()[selected]-frame["depth"].ravel()[sid.ravel()[selected]]
    return f,residual,np.full(len(selected),1/len(selected))
def collect(guard,replay=False,expected=None):
    xx=[];yy=[];df=[];dy=[];dw=[];rows=[]
    for split,r,frame in iter_frames():
        for delta in WARPS:
            index=int(np.flatnonzero(np.all(core.CANDIDATES==delta,axis=1))[0])
            for mode in MODES:
                z,ok,sid=observation(frame,r,delta,mode)
                key=key_for(split,r,delta,mode)
                x=call("feature_replay" if replay else "feature_generate",key,lambda:core.observable_case_features(frame["rgb"],z,ok))
                if replay:
                    iz,iv,ii=independent_source(frame,r,delta,mode)
                    need(np.array_equal(z,iz) and np.array_equal(ok,iv) and np.array_equal(sid,ii),"Independent source/noise/missing replay mismatch")
                    need(np.allclose(x,expected["X"][len(rows)],rtol=0,atol=1e-12),"Observable feature replay differs")
                xx.append(x);yy.append(index)
                rows.append(dict(key=key,split=split,scene=r["scene"],frame=r["frame"],warp=list(delta),mode=mode,
                    observed_count=int(ok.sum()),missing_original_valid=int(frame["valid"].sum()-ok.sum()),
                    source_sha=hashlib.sha256(np.ascontiguousarray(sid).tobytes()).hexdigest()))
                if split=="fit":
                    f,y,w=fit_samples(frame,z,ok,sid);df.append(f);dy.append(y);dw.append(w)
                guard()
    return dict(X=np.array(xx),y=np.array(yy,np.int64),depth_X=np.concatenate(df),depth_y=np.concatenate(dy),depth_weights=np.concatenate(dw)),rows
def generate(c,guard):
    a,rows=collect(guard)
    need(len(rows)==2160 and {k:sum(v["split"]==k for v in rows) for k in ("fit","calibration","evaluation")}==dict(fit=1296,calibration=432,evaluation=432),"Wrong case count")
    arrays_save(OUT/"generated.npz",**a)
    once(OUT/"generated.json",dict(at=now(),rows=rows,array_sha256=sha(OUT/"generated.npz"),calibration_is_diagnostic_only=True))
def generated():
    meta=read(OUT/"generated.json");need(sha(OUT/"generated.npz")==meta["array_sha256"],"Generated arrays changed")
    return arrays_read(OUT/"generated.npz"),meta["rows"]
def fit(c,guard):
    a,rows=generated();keep=np.array([r["split"]=="fit" for r in rows])
    warp=call("warp_fit","one_prespecified_fit",lambda:core.fit_warp(a["X"][keep],a["y"][keep],guard=guard))
    depth=call("depth_fit","one_prespecified_fit",lambda:core.fit_depth(a["depth_X"],a["depth_y"],a["depth_weights"],guard=guard))
    prior=core.empirical_prior(a["y"][keep])
    once(OUT/"model.json",dict(at=now(),warp=warp,depth=depth,empirical_prior=prior.tolist(),
        generated_sha256=sha(OUT/"generated.npz"),fit_scenes=read(DATA)["scenes"]["fit"],calibration_fits=0,GT_in_predictor=False))
def t_logpdf(residual,scale):
    return gammaln(2)-gammaln(1.5)-.5*np.log(3*np.pi)-np.log(scale)-2*np.log1p((residual/scale)**2/3)
def masked_mean(a,m):
    return float(np.mean(a[m])) if np.any(m) and np.isfinite(a[m]).all() else None
def strata(frame,query,delta,interface="A"):
    idx=query["ids"];h,w=frame["depth"].shape;v,u=np.divmod(idx,w)
    sign=1 if interface=="A" else -1
    ou=u+sign*delta[0];ov=v+sign*delta[1]
    label=frame["labels"];zr=frame["depth"]
    same=label[v,u]==label[ov,ou]
    jump=np.abs(zr[v,u]-zr[ov,ou])>.02
    return dict(main=query["main"],boundary=query["boundary"],same_instance=same,cross_instance=~same,
                depth_jump_proxy=jump)
def summarize_point_values(values,masks,denominators=None):
    ans={}
    for name,m in masks.items():
        ans[name]=dict(n=int(m.sum()),**{k:masked_mean(np.asarray(v),m) for k,v in values.items()})
        if denominators and name in denominators:ans[name]["reference_n"]=int(denominators[name])
    return ans
def evaluate_case(frame,r,delta,mode,p,depth_model,prior):
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
    A=core.point_mixture_A(np.column_stack([ou[visible],ov[visible]]),z[ov[visible],ou[visible]],
        scale_map[ov[visible],ou[visible]],p)
    original_A=old.xyz_at(np.column_stack([ou[visible],ov[visible]]),z[ov[visible],ou[visible]],frame["K"],frame["original_size"],(w,h))
    propagated=core.propagate_components(A,frame["K"],original_size=frame["original_size"],grid_size=(w,h))
    need(np.array_equal(propagated["observed_xyz"],original_A),"Original observation geometry changed")
    reference=frame["depth"].ravel()[idx]
    residual=z[ov[visible],ou[visible]]-reference[visible];sc=scale_map[ov[visible],ou[visible]]
    values={"nll":-t_logpdf(residual,sc),"fixed5mm_nll":-t_logpdf(residual,np.full_like(sc,.005)),
            "width95":2*stdtrit(3,.975)*sc}
    for level in LEVELS:values["coverage"+str(level)]=(np.abs(residual)<=stdtrit(3,(1+level)/2)*sc).astype(float)
    ajoint=core.hierarchical_statistics(A,reference[visible],did)
    for row in ajoint:
        values["joint_coverage"+str(row["level"])]=row["joint_source_covered"].astype(float)
        values["joint_probability"+str(row["level"])]=row["latent_joint_continuous_mass"]
    values["joint_nll"]=values["nll"]-np.log(p[did])
    amasks={k:m[visible] for k,m in masks.items()}
    Ametrics=summarize_point_values(values,amasks,{k:int(m.sum()) for k,m in masks.items()})
    Bmetrics={}
    distributions={"method":(p,scale_map),"prior_same_scale":(prior,scale_map),"prior_fixed5mm":(prior,np.full(z.shape,.005)),
                   "uniform_fixed5mm":(np.ones(81)/81,np.full(z.shape,.005)),
                   "single_mode":(np.eye(81)[int(np.argmax(p))],scale_map)}
    for name,(weights,sm) in distributions.items():
        B=core.target_mixture_B(z,ok,sm,uv,weights)
        stats=core.mixture_statistics(B,reference)
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
        joint=core.hierarchical_statistics(B,reference,did,levels=(.95,))
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
def evaluate(c,guard,replay=False):
    a,rows=generated();model=read(OUT/"model.json");prior=np.array(model["empirical_prior"])
    need(model["generated_sha256"]==sha(OUT/"generated.npz"),"Fit data changed")
    predictions=core.predict_warp(model["warp"],a["X"])
    index={r["key"]:j for j,r in enumerate(rows)};result=[]
    for split,r,frame in iter_frames():
        if split=="fit":continue
        for delta in WARPS:
            for mode in MODES:
                key=key_for(split,r,delta,mode);j=index[key]
                metrics=call("evaluation_replay" if replay else "evaluation",key,
                    lambda:evaluate_case(frame,r,delta,mode,predictions[j],model["depth"],prior))
                result.append(dict(key=key,split=split,scene=r["scene"],frame=r["frame"],warp=list(delta),mode=mode,**metrics));guard()
    summary,gates=aggregate(result)
    answer=dict(at=now(),scope="Known global integer warp and synthetic noise/missing on 20 exposed development scenes; not real sensor calibration or independent test",
        rows=result,summary=summary,gates=gates,development_scientific_passed=all(gates.values()),
        engineering_acceptance_pending=True,model_sha256=sha(OUT/"model.json"),
        geometry_auxiliary=read(OUT/"geometry/result.json"),new_neural_forward=0,new_official_AP=0,progress_before=[45,56.25])
    if not replay:
        arrays_save(OUT/"posterior.npz",probability=predictions)
        once(OUT/"result.json",answer)
    return answer
def mean(items,fn):
    values=[fn(v) for v in items];values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None
def aggregate(rows):
    rr=[r for r in rows if r["split"]=="evaluation"]
    need(len(rr)==432,"Expected 432 evaluation cases")
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
    worst_warp=min(mean([r for r in rr if r["scene"]==s],lambda r:r["warp_method"]["levels"]["0.95"]["covered"]) for s in (42,43,57,93))
    worst_depth=min(mean([r for r in rr if r["scene"]==s and r["mode"]!="nominal"],lambda r:r["A"]["main"]["coverage0.95"]) for s in (42,43,57,93))
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
    summary=dict(warp_nll=wp,warp_brier=wb,warp_levels=levels,worst_scene_warp95=worst_warp,worst_scene_depth95=worst_depth,
        groups=groups,target_joint_nll_method=method_B,target_joint_nll_prior_same_scale=prior_B,target_zero_probability_events=dict(method=method_zero,prior_same_scale=prior_zero),
        evaluated_cases=432,calibration_cases=432,total_reference_queries=sum(r["total_reference_query"] for r in rr),
        oracle_visible_queries=sum(r["oracle_visible"] for r in rr),oracle_missing_queries=sum(r["oracle_missing"] for r in rr),
        numeric_coverage_is_conditional_on_oracle_visibility=True,candidate_support_checked=all(r["candidate_support_checked"] for r in rr),
        candidate_support_scope="All oracle-visible fixed queries have exact source-ID in the 81 grid; missing remains missing; only known 9 synthetic warps")
    return summary,gates
def close_numeric(a,b,path="result"):
    if isinstance(a,dict):
        need(set(a)==set(b),"Keys differ "+path)
        for k in a:close_numeric(a[k],b[k],path+"/"+k)
    elif isinstance(a,list):
        need(len(a)==len(b),"List length differs "+path)
        for i,(x,y) in enumerate(zip(a,b)):close_numeric(x,y,path+"/"+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        need(np.isclose(a,b,rtol=1e-10,atol=1e-12),"Numeric difference "+path)
    else:need(a==b,"Value differs "+path)
def review(c,guard):
    a,rows=generated();rebuilt,rrows=collect(guard,replay=True,expected=a)
    need(rows==rrows,"Case metadata changed")
    for k in a:need(np.array_equal(a[k],rebuilt[k]),"Full generated arrays differ "+k)
    del rebuilt
    fresh=evaluate(c,guard,replay=True);old_result=read(OUT/"result.json")
    for k in ("rows","summary","gates","development_scientific_passed"):close_numeric(old_result[k],fresh[k],k)
    for fname in ("model.json","generated.npz","posterior.npz"):need((OUT/fname).exists(),"Missing artifact")
    probs=arrays_read(OUT/"posterior.npz")["probability"]
    p=core.predict_warp(read(OUT/"model.json")["warp"],a["X"])
    need(np.array_equal(probs,p),"Stored posterior differs")
    need(not ledger()["unknown"],"Unknown scientific calls")
    protected={relative(p):identity(p) for p in OUT.rglob("*") if p.is_file()}
    once(LOG/"review.json",dict(at=now(),passed=True,scope="Full deterministic replay; independent corruption/source replay; numerical module independently reviewed/tested",
        reviewed_outputs=protected,development_scientific_passed=old_result["development_scientific_passed"],ledger=ledger()))
def cold(c,guard):
    proof=read(LOG/"review.json");need(proof["passed"],"Review not passed")
    for name,value in proof["reviewed_outputs"].items():need(identity(ROOT/name)==value,"Read-only artifact changed "+name)
    result=read(OUT/"result.json");summary,gates=aggregate(result["rows"])
    close_numeric(summary,result["summary"]);need(gates==result["gates"],"Gate recomputation differs")
    need(ledger()==proof["ledger"] and not ledger()["unknown"],"Ledger changed after review")
    once(LOG/"cold.json",dict(at=now(),passed=True,fresh_process_pid=os.getpid(),review_sha256=sha(LOG/"review.json"),
        development_scientific_passed=result["development_scientific_passed"],no_new_research=True))
def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("generate","fit","evaluate","review","cold"));args=p.parse_args()
    c=verify()
    with project_lock(ROOT):
        guard=Guard(args.stage)
        try:globals()[args.stage](c,guard)
        finally:guard.close()
if __name__=="__main__":main()


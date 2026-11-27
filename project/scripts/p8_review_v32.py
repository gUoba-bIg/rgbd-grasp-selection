"""Read-only numerical review, then cold hash reentry in a distinct process."""
import os,sys,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/build/graspnetAPI_eval_v7'))
from scripts.p8_common_v32 import *
def independent_selection(a,keep,scores,geo,state):
 import numpy as np
 from scripts.verify_grasp_geometry_v10 import exact_indices
 indices=sorted(np.flatnonzero(keep).tolist(),key=lambda i:(-float(scores[i]),i));normalized=a[indices].copy();normalized[:,0]=range(len(indices),0,-1)
 if not indices:return dict(indices=[],slots=[],confidence=[])
 nms=geo.Group(normalized).nms(.03,np.pi/6).grasp_group_array;original=[indices[i] for i in exact_indices(normalized,nms,np)];assigned=geo.assignments(a,original,state);selected=[]
 for slot in range(len(state['ids'])):
  chosen=sorted([j for j,s in enumerate(assigned) if s==slot],key=lambda j:-float(nms[j,0]))[:10];selected.extend((original[j],slot,float(nms[j,0])) for j in chosen)
 selected=sorted(selected,key=lambda v:-v[2])[:50]
 return dict(indices=[v[0] for v in selected],slots=[v[1] for v in selected],confidence=[v[2] for v in selected])
def independent_probability(y,z):
 import numpy as np
 from scipy.special import expit
 n=len(y)
 if not n:return dict(n=0,positives=0,nll=None,brier=None,ece15=None,bins=[],aurc=None)
 p=[float(expit(float(v))) for v in z];ordered=sorted(range(n),key=lambda i:(p[i],i));q,rem=divmod(n,15);start=0;bins=[];ece=0.
 for k in range(15):
  size=q+(k<rem);ix=ordered[start:start+size];start+=size;conf=sum(p[i] for i in ix)/size if size else None;freq=sum(float(y[i]) for i in ix)/size if size else None
  if size:ece+=size/n*abs(conf-freq)
  bins.append(dict(n=size,confidence=conf,frequency=freq))
 failures=0;risks=[]
 for k,i in enumerate(sorted(range(n),key=lambda i:(-p[i],i)),1):failures+=1-float(y[i]);risks.append(failures/k)
 return dict(n=n,positives=int(sum(y)),nll=sum(max(float(v),0)+math.log1p(math.exp(-abs(float(v))))-float(t)*float(v) for t,v in zip(y,z))/n,brier=sum((v-float(t))**2 for v,t in zip(p,y))/n,ece15=ece,bins=bins,aurc=sum(risks)/n)
def run(cold=False):
 import numpy as np
 c=verify()
 if cold:
  r=load(LOG/'readonly.json');need(r['passed'],'Missing numerical acceptance')
  for p,s in r['files'].items():need(stat(ROOT/p)==s,'Cold immutable output changed '+p)
  for p,s in read(ROOT/'logs/P7/risk_v31/cold.json')['files'].items():need(stat(ROOT/p)==s,'P7 parent changed on cold reentry '+p)
  sealed(LOG/'cold.json',dict(passed=True,at=now(),files=r['files'],new_model_forwards=0,new_official_scores=0,new_training=0));return
 from src.grasp_observed_filter_v29 import independent_mask,pinned_detector
 from src.p8_math_v32 import input_sample,REMOVED,ranked_candidates
 from src.risk_observations_v31 import points
 from scripts.p8_geometry_v32 import observations,sources,weights,method_scores,Geometry,check_oracle
 from scripts.p7_review_v31 import independent_features,close_values
 from scripts.p8_metrics_v32 import datasets,calculate
 from scipy.optimize import minimize_scalar
 b=Budget(c,'review');load(LOG/'mask_readonly.json');geo=Geometry(c);w,mean,scale,penalty=weights();ledger=counts();returned={e['call_id'] for e in events() if e['kind'].endswith('_returned')};oldreturned={e['call_id'] for e in [readrow for readrow in map(__import__('json').loads,(ROOT/'logs/P7/risk_v31/calls.jsonl').read_text().splitlines())] if e['kind'].endswith('_returned')};oracles=0;reviewed=0
 for r in cases(c):
  aold=np.load(oldobservations(r));gen=load(folder(r)/'generation.json');new=np.load(artifact(folder(r)/'foreground.npz'));cloud,px,nfg=input_sample(aold,c['seed'],r['scene']);need(np.array_equal(new['sampled_pixels'],px) and gen['foreground_points']==nfg,'Foreground input sample changed');need(set(gen['forward_call_ids']+gen['identity_call_ids'])<=returned,'Missing generator return');need(np.all(aold['instances'].ravel()[new['pixels']]>0),'Candidate origin not foreground')
  yy,xx=np.unravel_index(new['pixels'],(480,640));z=aold['observed_depth'][yy,xx].astype(float);K=aold['K'];width,height=aold['original_size'];oy=((2*yy+1)*height)//960;ox=((2*xx+1)*width)//1280;xyz=np.column_stack(((ox-K[0,2])*z/K[0,0],(oy-K[1,2])*z/K[1,1],z));need(np.allclose(new['candidates'][:,13:16],xyz,rtol=2e-7,atol=1e-7),'Candidate original-pixel mapping differs');aold.close();new.close();state=geo.scene(r)
  for source in sources(r):
   receipt=LOG/'reviewed_units'/r['split']/(key(r)+'_'+source+'.json')
   if receipt.exists():
    accepted=load(receipt)
    for p,s in accepted['files'].items():need(stat(ROOT/p)==s,'Previously reviewed unit changed')
    reviewed+=1;oracles+=accepted['oracles'];continue
   b.guard();d=folder(r)/source;a=observations(r,source);saved=np.load(artifact(d/'features.npz'));cand=a['candidates'];cloud,_=points(a['observed_depth'],a['observed_valid'],a['K'],a['original_size'])
   for resolution,name in [(.01,'voxels10'),(.005,'voxels05')]:need(np.array_equal(pinned_detector()(cloud,voxel_size=resolution).scene_points,saved[name]),'Full observed cloud filtering changed')
   keep=independent_mask(cand,saved['voxels10']);need(np.array_equal(keep,saved['keep']),'Independent collision eligibility differs')
   # Full raw-candidate feature coverage: a discarded candidate could have changed ranking.
   for start in range(0,len(cand),32):
    b.guard();expected=independent_features(a,list(range(start,min(start+32,len(cand)))),saved['voxels05']);need(np.allclose(expected,saved['features'][start:start+32],rtol=2e-6,atol=2e-7),'Independent P8 features differ')
   scores=method_scores(saved['features'],source,w,mean,scale,penalty,audit=True)
   for k,v in scores.items():need(np.array_equal(v,saved[k]),'Risk/reduced-feature score differs')
   sel=load(d/'selection.json');pool=geo.pool(cand,keep) if source=='foreground' else [];need(sel['pool']==pool,'Observable calibration pool differs')
   methods={k:independent_selection(cand,keep,v,geo,state) for k,v in scores.items()} if r['split']=='validation' else {};need(methods==sel['methods'],'Independent official selection differs');required=sorted(set(pool).union(*(set(x['indices']) for x in methods.values())));need(required==sel['required_indices'] and geo.assignments(cand,required,state)==sel['required_slots'],'Coverage/assignment mismatch')
   labs=load(d/'labels.json');need(labs['indices']==required,'Label coverage differs');labels={v['index']:v for v in labs['records']}
   for i,slot in zip(required,sel['required_slots']):
    rec=load(d/'labels'/f'{i:04d}.json');need(rec==labels[i] and rec['grasp']==cand[i].tolist() and rec['object_slot']==slot,'Label identity mismatch');need(rec['positive']==bool(0<rec['score']<=.4 and not rec['collision']),'Success target differs')
    if rec['origin']=='P7_reuse':prior=load(ROOT/rec['origin_path']);need(prior['grasp'][1:]==rec['grasp'][1:] and prior['score']==rec['score'] and prior['collision']==rec['collision'] and rec['call_id'] in oldreturned,'Invalid P7 reuse')
    else:need(rec['call_id'] in returned,'Unknown score used')
   if (d/'oracle.json').exists():
    o=load(d/'oracle.json');need(o['call_id'] in returned,'Unknown oracle used');sc=cand[:,0] if r['split']=='calibration' else scores['D0' if source=='whole' else 'D4'];norm,orig=ranked_candidates(cand,keep,sc)
    if r['split']=='calibration':mask=np.isin(orig,pool);norm=norm[mask];orig=orig[mask];expected=independent_selection(cand,np.isin(np.arange(len(cand)),pool),sc,geo,state)
    else:expected=methods['D0' if source=='whole' else 'D4']
    need(expected==o['expected'],'Oracle selection trace differs');check_oracle(o['value'],norm,orig,labels,expected);oracles+=1
   saved.close();sealed(receipt,dict(files={p.relative_to(ROOT).as_posix():stat(p) for p in d.rglob('*') if p.is_file()},oracles=int((d/'oracle.json').exists())));reviewed+=1;status('full_readonly',units_completed=reviewed,total_units=225)
 need(oracles==3,'Required independent upstream references absent');cal=load(BASE/'calibration.json');data=datasets(c);result=load(BASE/'metrics.json')
 for split,d in data.items():
  close_values(independent_probability(d['y'],d['z']),result['probability'][split]['uncalibrated'])
  if cal['adequate']:close_values(independent_probability(d['y'],d['z']/cal['temperature']),result['probability'][split]['calibrated'])
 if cal['adequate']:
  def audit_temp():
   d=data['calibration'];fit=minimize_scalar(lambda z:independent_probability(d['y'],d['z']/np.exp(z))['nll'],bounds=(np.log(.05),np.log(20.)),method='bounded',options=dict(maxiter=100,xatol=1e-6));return dict(T=float(np.exp(fit.x)),nfev=int(fit.nfev),success=bool(fit.success))
  tp=LOG/'temperature_identity.json'
  if tp.exists():tmp=load(tp)['value']
  else:tmp,cid=invoke('temperature_audit',audit_temp);sealed(tp,dict(value=tmp,call_id=cid))
  need(tmp['success'] and np.isclose(tmp['T'],cal['temperature'],rtol=1e-4,atol=1e-5),'Independent temperature differs')
 close_values(calculate(c,cal['temperature'],cal['adequate']),result)
 for row in result['rows']:
  r={k:row[k] for k in ['scene','split','condition','frame']};source='whole' if row['method'] in ['D0','D6'] else 'foreground';d=folder(r)/source;sel=load(d/'selection.json')['methods'][row['method']];labs={v['index']:v for v in load(d/'labels.json')['records']};prefix=[]
  for k in range(1,51):prefix.append([sum(0<labs[i]['score']<=mu and not labs[i]['collision'] for i in sel['indices'][:k])/k for mu in [.2,.4,.6,.8,1.,1.2]])
  need(np.allclose(prefix,row['prefix'],rtol=0,atol=0) and np.isclose(np.mean(prefix),row['AP']),'Independent grasp AP differs')
 for p,s in read(ROOT/'logs/P7/risk_v31/cold.json')['files'].items():need(stat(ROOT/p)==s,'P7 historical artifact changed '+p)
 for p,s in c['reference_inputs'].items():need(stat(ROOT/p)==s,'Reference input changed '+p)
 for k,v in counts().items():need(v['intents']<=c['call_limits'][k],'Call budget mismatch');need(not v['unknown_ids'] or k=='upstream_oracle','Unresolved required computation '+k)
 files={p.relative_to(ROOT).as_posix():stat(p) for d in [BASE,ROOT/'experiments/p8_ablation_v32'] for p in d.rglob('*') if p.is_file()};files.update({p.relative_to(ROOT).as_posix():stat(p) for p in [CONFIG,LOG/'calls.jsonl',LOG/'mask_readonly.json']});b.save('complete');sealed(LOG/'readonly.json',dict(passed=True,at=now(),files=files,units=225,upstream_references=3,all_raw_candidate_features_verified=True,scientific_usable=result['scientific_usable'],new_official_scores=0,new_training=0))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run('--cold' in sys.argv)

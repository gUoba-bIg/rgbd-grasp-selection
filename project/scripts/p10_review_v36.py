"""Full observed-feature/selection/metric checks without refitting or rescoring."""
import sys,hashlib
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p10_common_v36 import *
environment()
def independent_depth(c,r):
 import numpy as np
 from PIL import Image
 from scipy.io import loadmat
 d=ROOT/c['data_root']/f"scenes/scene_{r['scene']:04d}/realsense";meta=loadmat(d/'meta/0064.mat',variable_names=['factor_depth','intrinsic_matrix'])
 with Image.open(d/'depth/0064.png') as im:size=im.size;z=np.asarray(im.resize((640,480),Image.Resampling.NEAREST),np.float32)/float(meta['factor_depth'].item())
 valid=np.isfinite(z)&(z>0);z=np.where(valid,np.minimum(z,2.)/2,0).astype(np.float32)*2;original=z.copy();v=valid.copy();cond=r['condition'];rng=np.random.RandomState(int(hashlib.sha256(f"P5v23|20261023|{c['quality_screen_epoch']}|{r['scene']}|0064|{cond}".encode()).hexdigest()[:8],16))
 shift=int(cond[5:]) if cond.startswith('shift') else (3 if cond=='joint' else 0)
 if shift:z[:,shift:]=original[:,:-shift];z[:,:shift]=0;v[:,shift:]=valid[:,:-shift];v[:,:shift]=False
 sigma=int(cond[5:])/1000 if cond.startswith('noise') else (.005 if cond=='joint' else 0.)
 if sigma:z=np.where(v,np.clip(z+rng.normal(0,sigma,z.shape).astype(np.float32),0,2),0).astype(np.float32);v&=z>0
 rate=int(cond[7:])/100 if cond.startswith('missing') else (.15 if cond=='joint' else 0.)
 if rate:v&=rng.uniform(size=z.shape)>=rate
 if cond in ['missing15','joint','block']:
  y=int(rng.randint(0,361));x=int(rng.randint(0,481));v[y:y+120,x:x+160]=False
 z[~v]=0;return z,v,meta['intrinsic_matrix'],size
def review(scene):
 import numpy as np
 from scripts.p10_geometry_v36 import Geometry,configured,observations,check_oracle
 from scripts.p7_review_v31 import independent_features
 from scripts.p8_review_v32 import independent_selection
 from src.grasp_observed_filter_v29 import independent_mask
 from src.p8_math_v32 import logits,ranked_candidates
 c=verify();b=Budget(c,'review');load(BASE/'scoring_complete.json');geo=Geometry(configured(c));ev=events();returns={x['call_id'] for x in ev if x['kind'].endswith('_returned')};intents={x['call_id']:x for x in ev if x['kind'].endswith('_intent')};w={}
 with np.load(artifact(ROOT/'results/P7/risk_v31/predictions/train.npz')) as a:mean=a['mean'];scale=a['scale']
 for s in c['seeds']:
  with np.load(artifact(ROOT/f'results/P9/risk_seeds_v35/seeds/{s}/weights.npz')) as a:w[str(s)]={k:a[k].copy() for k in a.files}
 for r in cases(c,scene):
  d=folder(r);rp=LOG/'reviewed_cases'/(key(r)+'.json')
  if rp.exists():
   for p,s in load(rp)['files'].items():need(stat(ROOT/p)==s,'Reviewed case changed')
   continue
  b.guard();verify_case_inputs(c,r);gen=load(d/'generation.json');need(all(cid in returns for cid in gen['forward_call_ids']+gen['identity_call_ids']+[gen['corruption_call_id']]),'Unknown generation used');need(all(intents[cid]['case']==r for cid in gen['forward_call_ids']+[gen['corruption_call_id']]),'Generation case provenance differs');a=observations(r);cand=a['candidates'];depth,valid,K,size=invoke('corruption_audit',lambda:independent_depth(c,r),case=r)[0];need(np.array_equal(depth,a['observed_depth']) and np.array_equal(valid,a['observed_valid']) and np.array_equal(K,a['K']) and tuple(a['original_size'])==size,'Independent observation replay differs')
  with np.load(artifact(d/'features.npz')) as q:f={k:q[k].copy() for k in q.files}
  need(np.array_equal(independent_mask(cand,f['voxels10']),f['keep']),'Independent observed collision mask differs')
  for start in range(0,len(cand),32):
   b.guard();expected=independent_features(a,np.arange(start,min(start+32,len(cand))),f['voxels05']);need(np.allclose(expected,f['features'][start:start+32],rtol=2e-6,atol=2e-7),'Independent observable feature differs')
  need(np.array_equal(f['D0'],cand[:,0]),'D0 raw confidence differs');sel=load(d/'selection.json');state=geo.scene(r)
  for s in c['seeds']:
   z,cid=invoke('risk_numpy_audit',lambda:logits(f['features'],w[str(s)],mean,scale),seed=s,case=r);need(np.allclose(z,f[str(s)],rtol=2e-5,atol=2e-5),'Independent risk export differs');need(sel['call_ids'][str(s)] in returns and intents[sel['call_ids'][str(s)]]['case']==r,'Risk forward provenance differs')
  methods={m:independent_selection(cand,f['keep'],f[m],geo,state) for m in ['D0']+[str(s) for s in c['seeds']]};pool=geo.pool(cand,f['keep']);need(methods==sel['methods'] and pool==sel['pool'],'Independent ranked selection differs');required=sorted(set(pool).union(*(set(v['indices']) for v in methods.values())));slots=geo.assignments(cand,required,state);need(required==sel['required_indices'] and slots==sel['required_slots'],'Required label support differs');labs=load(d/'labels.json');need(labs['indices']==required,'Label coverage differs');labels={v['index']:v for v in labs['records']}
  for ix,slot in zip(required,slots):
   rec=load(d/'labels'/f'{ix:04d}.json');need(rec==labels[ix] and rec['grasp']==cand[ix].tolist() and rec['object_slot']==slot and rec['object_id']==state['ids'][slot],'Label geometry binding differs');need(rec['positive']==bool(0<rec['score']<=.4 and not rec['collision']),'Probability event differs');cid=rec['call_id'];need(cid in returns and intents[cid]['kind']=='official_score_intent' and intents[cid]['case']==r and intents[cid]['index']==ix,'Unknown/misassigned official score')
  for m in methods:
   op=d/f'oracle_{m}.json'
   if op.exists():
    o=load(op);norm,orig=ranked_candidates(cand,f['keep'],f[m]);need(o['selection']==methods[m],'Oracle selection mismatch')
    if o['empty_reference']:need(len(norm)==0 and o['call_id'] is None and not methods[m]['indices'],'Empty reference contract differs')
    else:need(o['call_id'] in returns,'Unknown oracle');check_oracle(o['value'],norm,orig,labels,methods[m])
  sealed(rp,dict(at=now(),passed=True,files={p.relative_to(ROOT).as_posix():stat(p) for p in d.rglob('*') if p.is_file()}));status('review',scene=scene,condition=r['condition'])
 b.save('complete');sealed(LOG/'reviewed_scenes'/f'{scene:04d}.json',dict(at=now(),passed=True,scene=scene,conditions=12))
def finalize(cold=False):
 from scripts.p7_review_v31 import close_values
 from scripts.p10_metrics_v36 import calculate
 c=verify();b=Budget(c,'cold' if cold else 'review');b.guard()
 if cold:
  ro=load(LOG/'readonly.json');need(ro['passed'],'Full acceptance absent')
  for n,(p,s) in enumerate(ro['files'].items()):
   if n%100==0:b.guard()
   need(stat(ROOT/p)==s,'Cold artifact changed '+p)
  parents(c);sealed(LOG/'cold.json',dict(at=now(),passed=True,files=ro['files'],new_training=0,new_forwards=0,new_official_scores=0));b.save('complete');return
 for s in sum(c['groups'].values(),[]):need(load(LOG/'reviewed_scenes'/f'{s:04d}.json')['passed'],'Scene review absent')
 r=load(BASE/'metrics.json');close_values(r,calculate(c,True));parents(c);ledger=counts();need(all(not v['unknown'] for v in ledger.values()),'New unknown call prevents acceptance');oracles=[load(folder(dict(scene=s,condition='clean'))/f'oracle_{m}.json') for s in [100,130,160] for m in ['D0']+[str(s) for s in c['seeds']]];need(ledger['upstream_oracle']['known_returns']==sum(not o['empty_reference'] for o in oracles),'Missing official reference comparisons')
 for p,s in asset_inputs().items():need(stat(ROOT/p)==s,'Asset changed')
 files={p.relative_to(ROOT).as_posix():stat(p) for base in [BASE,LOG/'reviewed_cases',LOG/'reviewed_scenes'] for p in base.rglob('*') if p.is_file()};files.update(asset_inputs());files.update({p.relative_to(ROOT).as_posix():stat(p) for p in [CONF,ACONF,LOG/'calls.jsonl',ROOT/c['parent_binding']]});sealed(LOG/'readonly.json',dict(at=now(),passed=True,files=files,calls=ledger,ranking_scientific_usable=r['ranking_scientific_usable'],full_cases_verified=1080,all_parents_unchanged=True));b.save('complete')
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):review(int(sys.argv[2])) if sys.argv[1]=='scene' else finalize(sys.argv[1]=='cold')

"""Outcome-blind three-seed selections then deduplicated physical labels."""
import os,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/build/graspnetAPI_eval_v7'))
from scripts.p9_common_v35 import *
from scripts.p8_geometry_v32 import Geometry,observations,check_oracle
def predict():
 import numpy as np,torch
 from scripts.p7_fit_v31 import model
 c=verify();b=Budget(c,'prediction');load(BASE/'training_complete.json');torch.set_num_threads(2);torch.use_deterministic_algorithms(True);geo=Geometry(c);train=pool('train');mean=train['mean'];scale=train['scale'];nets={}
 for seed in c['seeds']:
  fit=load(seedpath(seed)/'fit.json');net=model();state=torch.load(EXP/str(seed)/f"epoch_{fit['selected_epoch']:02d}.pt",map_location='cpu',weights_only=False);net.load_state_dict(state['model']);net.eval();nets[seed]=net
 for number,r in enumerate(cases(c),1):
  b.guard();d=BASE/'cases'/key(r);p=d/'selection.json'
  if p.exists():load(p);artifact(d/'predictions.npz');continue
  a=observations(r,'whole');cand=a['candidates'];fp=P8/'cases/validation'/key(r)/'whole/features.npz'
  with np.load(artifact(fp)) as f:features=f['features'].copy();keep=f['keep'].copy()
  X=torch.from_numpy((features-mean)/scale);scores={};ids={}
  for seed,net in nets.items():
   def ff():
    with torch.inference_mode():return net(X).reshape(-1).numpy().copy()
   z,cid=invoke('risk_forward',ff,seed=seed,phase='all_observable_candidates',case=r,n=len(cand));need(np.isfinite(z).all(),'Invalid risk scores');scores[str(seed)]=z;ids[str(seed)]=cid
  payload(d/'predictions.npz',**scores);state=geo.scene(r);methods={s:geo.select(cand,keep,z,state)[0] for s,z in scores.items()};required=sorted(set().union(*(set(v['indices']) for v in methods.values())));slots=geo.assignments(cand,required,state)
  sealed(p,dict(case=r,methods=methods,required_indices=required,required_slots=slots,call_ids=ids,input_features=stat(fp),labels_used_for_model_scores=False,official_GT_used_only_for_metric_assignment=True));status('seed_selections',completed=number,total=75)
 b.save('complete');sealed(BASE/'selection_complete.json',dict(at=now(),scene_conditions=75,seed_conditions=225,labels_used_for_selection=False))
def score():
 import numpy as np
 from src.p8_math_v32 import ranked_candidates
 c=verify();b=Budget(c,'scoring');load(BASE/'selection_complete.json');geo=Geometry(c);oldreturns=set()
 for dirname in ['P7/risk_v31','P8/ablation_v32']:
  oldreturns.update(x['call_id'] for x in map(json.loads,(ROOT/'logs'/dirname/'calls.jsonl').read_text().splitlines()) if x['kind'].endswith('_returned'))
 for number,r in enumerate(cases(c),1):
  b.guard();d=BASE/'cases'/key(r)
  if (d/'labels.json').exists():load(d/'labels.json');continue
  sel=load(d/'selection.json');a=observations(r,'whole');cand=a['candidates'];state=geo.scene(r);oldpath=P8/'cases/validation'/key(r)/'whole';prior={x['index']:x for x in load(oldpath/'labels.json')['records']};records=[]
  for ix,slot in zip(sel['required_indices'],sel['required_slots']):
   b.guard();p=d/'labels'/f'{ix:04d}.json'
   if p.exists():rec=load(p)
   elif ix in prior:
    old=prior[ix];op=oldpath/'labels'/f'{ix:04d}.json';need(old==load(op) and old['grasp'][1:]==cand[ix,1:].tolist() and old['object_slot']==slot and old['call_id'] in oldreturns,'Inherited label contract differs');rec=dict(old,grasp=cand[ix].tolist(),origin='accepted_P8',origin_path=op.relative_to(ROOT).as_posix(),source_receipt=stat(op));sealed(p,rec)
   else:
    value,cid=invoke('official_score',lambda:geo.score(cand[ix],slot,state),case=r,index=ix);need(value['score'] in [-1.,.2,.4,.6,.8,1.,1.2],'Invalid official label');rec=dict(index=ix,grasp=cand[ix].tolist(),object_slot=slot,object_id=int(state['ids'][slot]),**value,positive=bool(0<value['score']<=.4 and not value['collision']),call_id=cid,origin='P9');sealed(p,rec)
   need(rec['grasp']==cand[ix].tolist() and rec['object_slot']==slot,'Scored geometry differs');records.append(rec)
  if r['scene']==c['groups']['validation'][0] and r['condition']=='clean':
   with np.load(artifact(P8/'cases/validation'/key(r)/'whole/features.npz')) as f:keep=f['keep'].copy()
   with np.load(artifact(d/'predictions.npz')) as f:
    for seed in c['seeds']:
     op=d/f'oracle_{seed}.json';norm,original=ranked_candidates(cand,keep,f[str(seed)]);need(len(norm)>0,'Required upstream reference empty')
     if not op.exists():value,cid=invoke('upstream_oracle',lambda:geo.oracle(norm,state),seed=seed,case=r);sealed(op,dict(value=value,call_id=cid,selection=sel['methods'][str(seed)]))
     o=load(op);check_oracle(o['value'],norm,original,{x['index']:x for x in records},sel['methods'][str(seed)])
  sealed(d/'labels.json',dict(indices=sel['required_indices'],records=records));status('official_scoring',completed=number,total=75,new_known_scores=counts()['official_score']['known_returns'])
 b.save('complete');sealed(BASE/'scoring_complete.json',dict(at=now(),scene_conditions=75,seed_conditions=225))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):predict() if sys.argv[1]=='predict' else score()

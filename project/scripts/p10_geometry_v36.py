"""Same accepted physical metric, scene-isolated workers and small mesh cache."""
import sys,functools,gc
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p10_common_v36 import *
environment()
from scripts.p8_geometry_v32 import Geometry as AcceptedGeometry,check_oracle
class Geometry(AcceptedGeometry):
 def __init__(self,c):super().__init__(c)
 def modelpath(self,oid,tail):return ROOT/f'data/raw/prerequisites_v26/models/{oid:03d}/{tail}' if oid<71 else DATA/f'models/{oid:03d}/{tail}'
 @functools.lru_cache(maxsize=40)
 def model(self,oid):
  p=self.modelpath(oid,'nontextured.ply');need(stat(p)==self.refs[p.relative_to(ROOT).as_posix()],'Model input changed');return self.official.voxel_sample_points(self.np.asarray(self.o3d.io.read_point_cloud(str(p)).points),.008)
 @functools.lru_cache(maxsize=4)
 def dex(self,oid):
  for ext in ['obj','sdf']:
   p=self.modelpath(oid,'textured.'+ext);need(stat(p)==self.refs[p.relative_to(ROOT).as_posix()],'Mesh input changed')
  return self.official.load_dexnet_model(str(self.modelpath(oid,'textured')))
 @functools.lru_cache(maxsize=1)
 def _scene(self,scene):return super().scene(dict(scene=scene))
 def scene(self,r):return self._scene(r['scene'])
def observations(r):
 import numpy as np
 with np.load(artifact(folder(r)/'observations.npz')) as a:v={k:a[k].copy() for k in a.files}
 v['candidates'][:,1]=np.clip(v['candidates'][:,1],0,.1);return v
def configured(c):return dict(c,reference_inputs=asset_inputs())
def features(scene):
 import numpy as np,torch
 from scripts.p7_fit_v31 import model
 from src.risk_observations_v31 import points,base_features
 from src.grasp_observed_filter_v29 import filter_candidates,pinned_detector
 c=verify();b=Budget(c,'features');load(BASE/'generation_complete.json');geo=Geometry(configured(c));torch.set_num_threads(2);torch.use_deterministic_algorithms(True);nets={}
 with np.load(artifact(ROOT/'results/P7/risk_v31/predictions/train.npz')) as a:mean=a['mean'];scale=a['scale']
 for seed in c['seeds']:
  n=model();fit=load(ROOT/f'results/P9/risk_seeds_v35/seeds/{seed}/fit.json');ck=torch.load(ROOT/f"experiments/risk_seeds_v35/{seed}/epoch_{fit['selected_epoch']:02d}.pt",map_location='cpu',weights_only=False);n.load_state_dict(ck['model']);n.eval();nets[seed]=n
 for r in cases(c,scene):
  d=folder(r);p=d/'selection.json'
  if p.exists():load(p);artifact(d/'features.npz');continue
  b.guard();a=observations(r);cand=a['candidates'];cloud,_=points(a['observed_depth'],a['observed_valid'],a['K'],a['original_size']);keep,v10=filter_candidates(cand,cloud,geo.Group);detector=pinned_detector()(cloud,voxel_size=.005);local=[]
  for start in range(0,len(cand),32):
   b.guard();co,em,ious=detector.detect(geo.Group(cand[start:start+32]),approach_dist=.05,collision_thresh=.01,return_empty_grasp=True,empty_thresh=.01,return_ious=True);local.append(np.column_stack([co,em]+ious).astype(np.float32))
  local=np.concatenate(local) if local else np.empty((0,7),np.float32);base=base_features(cand,a['pixels'],a['sigma'],a['quality_valid'],a['quality_anomaly'],a['foreground_probability'],a['instances']);f=np.concatenate([base,local],axis=1);need(f.shape==(len(cand),19) and np.isfinite(f).all(),'Invalid features');X=torch.from_numpy((f-mean)/scale);scores={'D0':cand[:,0].copy()};ids={}
  for seed,n in nets.items():
   def ff():
    with torch.inference_mode():return n(X).reshape(-1).numpy().copy()
   z,cid=invoke('risk_forward',ff,seed=seed,case=r);need(np.isfinite(z).all(),'Invalid risk score');scores[str(seed)]=z;ids[str(seed)]=cid
  payload(d/'features.npz',features=f,keep=keep,voxels10=v10,voxels05=detector.scene_points,**scores)
  state=geo.scene(r);methods={m:geo.select(cand,keep,z,state)[0] for m,z in scores.items()};pool=geo.pool(cand,keep);required=sorted(set(pool).union(*(set(v['indices']) for v in methods.values())));slots=geo.assignments(cand,required,state)
  sealed(p,dict(case=r,methods=methods,pool=pool,required_indices=required,required_slots=slots,call_ids=ids,feature_GT_used=False,scores_GT_used=False,official_metric_GT_assignment=True));status('features',scene=scene,condition=r['condition']);del a,f,cloud,X;gc.collect()
 b.save('complete');sealed(BASE/'features_done'/f'{scene:04d}.json',dict(at=now(),scene=scene,conditions=12))
def score(scene):
 import numpy as np
 from src.p8_math_v32 import ranked_candidates
 c=verify();b=Budget(c,'scoring');load(BASE/'selection_complete.json');geo=Geometry(configured(c))
 for r in cases(c,scene):
  d=folder(r)
  if (d/'labels.json').exists():load(d/'labels.json');continue
  b.guard();sel=load(d/'selection.json');a=observations(r);cand=a['candidates'];state=geo.scene(r);records={}
  # Group score calls by object slot to bound mesh reloads; output order unchanged.
  for ix,slot in sorted(zip(sel['required_indices'],sel['required_slots']),key=lambda v:(v[1],v[0])):
   b.guard();p=d/'labels'/f'{ix:04d}.json'
   if p.exists():rec=load(p)
   else:
    value,cid=invoke('official_score',lambda:geo.score(cand[ix],slot,state),case=r,index=ix);need(value['score'] in [-1.,.2,.4,.6,.8,1.,1.2],'Invalid official label');rec=dict(index=ix,grasp=cand[ix].tolist(),object_slot=slot,object_id=int(state['ids'][slot]),**value,positive=bool(0<value['score']<=.4 and not value['collision']),call_id=cid);sealed(p,rec)
   need(rec['grasp']==cand[ix].tolist() and rec['object_slot']==slot,'Candidate geometry differs');records[ix]=rec
  if r['condition']=='clean' and scene in [100,130,160]:
   with np.load(artifact(d/'features.npz')) as f:
    for m in ['D0']+[str(s) for s in c['seeds']]:
     op=d/f'oracle_{m}.json';norm,original=ranked_candidates(cand,f['keep'],f[m])
     if not op.exists():
      if len(norm):value,cid=invoke('upstream_oracle',lambda:geo.oracle(norm,state),case=r,method=m);sealed(op,dict(value=value,call_id=cid,selection=sel['methods'][m],empty_reference=False))
      else:sealed(op,dict(value=None,call_id=None,selection=sel['methods'][m],empty_reference=True,reason='Pinned upstream eval_grasp has undefined top50 cutoff for an empty input; no official call made'))
     o=load(op)
     if o['empty_reference']:need(len(norm)==0 and not sel['methods'][m]['indices'],'Invalid empty reference')
     else:check_oracle(o['value'],norm,original,records,sel['methods'][m])
  sealed(d/'labels.json',dict(indices=sel['required_indices'],records=[records[i] for i in sel['required_indices']]));status('scoring',scene=scene,condition=r['condition'],known_scores=counts()['official_score']['known_returns']);del a,records;gc.collect()
 b.save('complete');sealed(BASE/'scoring_done'/f'{scene:04d}.json',dict(at=now(),scene=scene,conditions=12))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):features(int(sys.argv[2])) if sys.argv[1]=='features' else score(int(sys.argv[2]))

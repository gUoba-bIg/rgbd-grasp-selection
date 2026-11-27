"""Observable features/selections, followed by deduplicated official labels."""
import os,sys,functools
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/build/graspnetAPI_eval_v7'))
from scripts.p8_common_v32 import *
def observations(r,source):
 import numpy as np
 with np.load(oldobservations(r)) as v:a={k:v[k] for k in v.files}
 if source=='foreground':
  with np.load(artifact(folder(r)/'foreground.npz')) as v:a.update(candidates=v['candidates'],pixels=v['pixels'])
 a['candidates']=a['candidates'].copy();a['candidates'][:,1]=np.clip(a['candidates'][:,1],0,.1);return a
def sources(r):return ['foreground'] if r['split']=='calibration' else ['whole','foreground']
def weights():
 import numpy as np
 out={}
 for name,p in [('full',OLD/'selected_weights.npz'),('no_quality',BASE/'models/no_quality/weights.npz'),('no_instance',BASE/'models/no_instance/weights.npz')]:
  # P7's selected_weights receipt has the same artifact field.
  with np.load(artifact(p)) as a:out[name]={k:a[k] for k in a.files}
 with np.load(artifact(OLD/'predictions/train.npz')) as a:mean=a['mean'];scale=a['scale'];penalty=float(np.ptp(a['features'][:,0])+1.)
 return out,mean,scale,penalty
def method_scores(features,source,w,mean,scale,penalty,audit=False):
 import numpy as np
 from src.p8_math_v32 import logits,REMOVED
 def infer(name):return invoke('risk_numpy_audit' if audit else 'risk_numpy_forward',lambda:logits(features,w[name],mean,scale,REMOVED[name]),model=name,source=source,samples=len(features))[0]
 raw=features[:,0].astype(float);full=infer('full')
 if source=='whole':return dict(D0=raw,D6=full)
 return dict(D1=raw,D2=raw-penalty*np.clip((features[:,5]-mean[5])/scale[5],0,3)/3,D3=infer('no_quality'),D4=full,D5=full.copy(),D7=infer('no_instance'))
class Geometry:
 def __init__(self,c):
  import numpy as np,open3d as o3d
  from graspnetAPI import GraspGroup
  from graspnetAPI.utils import eval_utils as official
  from graspnetAPI.utils.config import get_config
  from graspnetAPI.utils.xmlhandler import xmlReader
  self.np=np;self.o3d=o3d;self.Group=GraspGroup;self.official=official;self.get_config=get_config;self.xmlReader=xmlReader;self.root=ROOT/c['data_root'];self.refs=c['reference_inputs']
  from scripts.verify_grasp_geometry_v10 import score_configs
  self.friction,self.configs=score_configs((np,o3d,GraspGroup,official,get_config,xmlReader))
 @functools.lru_cache(maxsize=40)
 def model(self,oid):
  p=self.root/f'models/{oid:03d}/nontextured.ply';need(stat(p)==self.refs[p.relative_to(ROOT).as_posix()],'Reference model changed');return self.official.voxel_sample_points(self.np.asarray(self.o3d.io.read_point_cloud(str(p)).points),.008)
 @functools.lru_cache(maxsize=16)
 def dex(self,oid):
  for ext in ['obj','sdf']:
   p=self.root/f'models/{oid:03d}/textured.{ext}';need(stat(p)==self.refs[p.relative_to(ROOT).as_posix()],'Reference mesh changed')
  return self.official.load_dexnet_model(str(self.root/f'models/{oid:03d}/textured'))
 def scene(self,r):
  np=self.np;o=self.official;sd=self.root/f"scenes/scene_{r['scene']:04d}/realsense"
  for tail in ['annotations/0064.xml','cam0_wrt_table.npy','camera_poses.npy']:
   p=sd/tail;need(stat(p)==self.refs[p.relative_to(ROOT).as_posix()],'Reference scene changed')
  parsed=[o.parse_posevector(v) for v in self.xmlReader(str(sd/'annotations/0064.xml')).getposevectorlist()];ids=[p[0] for p in parsed];poses=[p[1] for p in parsed];models=[self.model(i) for i in ids];transformed=[o.transform_points(p,t) for p,t in zip(models,poses)];scene=np.concatenate(transformed);slots=np.concatenate([np.full(len(p),i,np.int32) for i,p in enumerate(transformed)]);table=o.transform_points(o.create_table_points(1.,1.,.05,dx=-.5,dy=-.5,dz=-.05,grid_size=.008),np.linalg.inv(np.load(sd/'cam0_wrt_table.npy')@np.load(sd/'camera_poses.npy')[64]))
  return dict(ids=ids,poses=poses,models=models,transformed=transformed,scene=scene,slots=slots,table=table,world=np.concatenate([scene,table]))
 def select(self,candidates,keep,scores,state):
  import numpy as np
  from src.p8_math_v32 import ranked_candidates
  from scripts.verify_grasp_geometry_v10 import select_groups,exact_indices
  normalized,original=ranked_candidates(candidates,keep,scores)
  if not len(original):return dict(indices=[],slots=[],confidence=[]),normalized
  groups,_,_,detail=select_groups(normalized,state['models'],state['poses'],self.Group,self.official,np);rows=[(int(original[j]),i,float(g[0])) for i,group in enumerate(groups) for j,g in zip(exact_indices(normalized,group,np),group)];rows.sort(key=lambda t:-t[2]);need(len(rows)<=50,'Unique confidence must enforce <=50')
  return dict(indices=[r[0] for r in rows],slots=[r[1] for r in rows],confidence=[r[2] for r in rows]),normalized
 def pool(self,candidates,keep):
  from src.p8_math_v32 import ranked_candidates
  from scripts.verify_grasp_geometry_v10 import exact_indices
  a,original=ranked_candidates(candidates,keep,candidates[:,0])
  if not len(a):return []
  g=self.Group(a).nms(.03,self.np.pi/6);g.sort_by_score();return original[exact_indices(a,g[:20].grasp_group_array,self.np)].tolist()
 def assignments(self,candidates,indices,state):
  return state['slots'][self.official.compute_closest_points(candidates[indices,13:16],state['scene'])].tolist() if indices else []
 def score(self,g,slot,state):
  o=self.official;co,em,dg=o.collision_detection([g[None]],[state['transformed'][slot]],[None],[state['poses'][slot]],state['world'],outlier=.05,empty_thresh=10,return_dexgrasps=True);collision=bool(co[0][0]);value=-1.
  if not collision and dg[0][0] is not None:value=float(o.get_grasp_score(dg[0][0],self.dex(state['ids'][slot]),self.friction,self.configs))
  return dict(collision=collision,empty=bool(em[0][0]),score=value)
 def oracle(self,normalized,state):
  geom=self
  class Dex:
   def __getitem__(self,i):return geom.dex(state['ids'][i])
  g,s,c=self.official.eval_grasp(self.Group(normalized.copy()),state['models'],Dex(),state['poses'],self.get_config(),table=state['table'],voxel_size=.008,TOP_K=50)
  return dict(groups=[self.np.asarray(x).tolist() for x in g],scores=[self.np.asarray(x).tolist() for x in s],collisions=[self.np.asarray(x).tolist() for x in c])
def prepare():
 import numpy as np
 from src.risk_observations_v31 import points,base_features
 from src.grasp_observed_filter_v29 import filter_candidates,pinned_detector
 c=verify();b=Budget(c,'features');load(BASE/'generation_complete.json');load(BASE/'fitting_complete.json');geo=Geometry(c);w,mean,scale,penalty=weights();done=0
 for r in cases(c):
  for source in sources(r):
   d=folder(r)/source;p=d/'selection.json'
   if p.exists():load(p);artifact(d/'features.npz');done+=1;continue
   b.guard();a=observations(r,source);cand=a['candidates'];cloud,_=points(a['observed_depth'],a['observed_valid'],a['K'],a['original_size']);keep,v10=filter_candidates(cand,cloud,geo.Group);detector=pinned_detector()(cloud,voxel_size=.005);local=[]
   for start in range(0,len(cand),32):
    b.guard();co,em,ious=detector.detect(geo.Group(cand[start:start+32]),approach_dist=.05,collision_thresh=.01,return_empty_grasp=True,empty_thresh=.01,return_ious=True);local.append(np.column_stack([co,em]+ious).astype(np.float32))
   local=np.concatenate(local) if local else np.empty((0,7),np.float32);base=base_features(cand,a['pixels'],a['sigma'],a['quality_valid'],a['quality_anomaly'],a['foreground_probability'],a['instances']);f=np.concatenate([base,local],axis=1);need(f.shape==(len(cand),19) and np.isfinite(f).all(),'Bad observable features');scores=method_scores(f,source,w,mean,scale,penalty);pool=geo.pool(cand,keep) if source=='foreground' else []
   payload(d/'features.npz',features=f,keep=keep,voxels10=v10,voxels05=detector.scene_points,**scores)
   # Ground truth enters only the official metric's object assignment, never features/ranking/pool.
   state=geo.scene(r);methods={k:geo.select(cand,keep,s,state)[0] for k,s in scores.items()} if r['split']=='validation' else {};required=sorted(set(pool).union(*(set(v['indices']) for v in methods.values())));slots=geo.assignments(cand,required,state)
   sealed(p,dict(case=r,source=source,methods=methods,pool=pool,required_indices=required,required_slots=slots,feature_GT_used=False,method_scores_GT_used=False,pool_GT_used=False,official_metric_GT_assignment=True));done+=1;status('observable_features_and_selection',completed=done,total=225)
 b.save('complete');sealed(BASE/'selection_complete.json',dict(at=now(),units=225,labels_accessed=False))
def check_oracle(value,normalized,original,labels,selection):
 import numpy as np
 from scripts.verify_grasp_geometry_v10 import exact_indices
 checked=[]
 for gs,ss,cc in zip(value['groups'],value['scores'],value['collisions']):
  arr=np.asarray(gs,dtype=normalized.dtype).reshape(-1,17);ii=exact_indices(normalized,arr,np)
  for j,score,collision in zip(ii,ss,cc):
   ix=int(original[j]);r=labels[ix];need(r['score']==score and r['collision']==bool(collision),'Upstream physical result mismatch');checked.append(ix)
 need(sorted(checked)==sorted(selection['indices']),'Upstream selected support mismatch')
def score():
 import numpy as np
 from src.p8_math_v32 import ranked_candidates
 c=verify();b=Budget(c,'scoring');geo=Geometry(c);load(BASE/'selection_complete.json');done=0
 for r in cases(c):
  state=geo.scene(r)
  for source in sources(r):
   b.guard();d=folder(r)/source;sel=load(d/'selection.json');a=observations(r,source);cand=a['candidates'];records=[]
   old=load(oldfolder(r)/'labels.json') if source=='whole' else None;prior={v['index']:v for v in old['records']} if old else {}
   for ix,slot in zip(sel['required_indices'],sel['required_slots']):
    b.guard();p=d/'labels'/f'{ix:04d}.json'
    if p.exists():rec=load(p)
    elif ix in prior and prior[ix]['grasp'][1:]==cand[ix,1:].tolist() and prior[ix]['object_slot']==slot:
     oldrec=prior[ix];rec=dict(oldrec,grasp=cand[ix].tolist(),origin='P7_reuse',origin_path=(oldfolder(r)/'labels'/f'{ix:04d}.json').relative_to(ROOT).as_posix());sealed(p,rec)
    else:
     value,cid=invoke('official_score',lambda:geo.score(cand[ix],slot,state),case=r,source=source,index=ix);need(value['score'] in [-1.,.2,.4,.6,.8,1.,1.2],'Invalid friction label');rec=dict(index=ix,grasp=cand[ix].tolist(),object_slot=slot,object_id=int(state['ids'][slot]),**value,positive=bool(0<value['score']<=.4 and not value['collision']),call_id=cid,origin='P8');sealed(p,rec)
    need(rec['grasp']==cand[ix].tolist() and rec['object_slot']==slot,'Label geometry mismatch');records.append(rec)
   # Three required nonempty reference calls: first cal FG, first val whole D0, first val FG D4.
   oracle_case=r['scene']==c['groups'][r['split']][0] and r['condition']=='clean'
   if oracle_case:
    f=np.load(artifact(d/'features.npz'));sc=cand[:,0] if r['split']=='calibration' else f['D0' if source=='whole' else 'D4'];norm,orig=ranked_candidates(cand,f['keep'],sc)
    if r['split']=='calibration':
     # Restrict the oracle to the fixed outcome-blind pool; all are labeled.
     mask=np.isin(orig,sel['pool']);norm=norm[mask];orig=orig[mask];expected=geo.select(cand,np.isin(np.arange(len(cand)),sel['pool']),sc,state)[0]
    else:expected=sel['methods']['D0' if source=='whole' else 'D4']
    need(len(norm)>0,'Required baseline oracle has no candidates; finite gate cannot be accepted');op=d/'oracle.json'
    if not op.exists():value,cid=invoke('upstream_oracle',lambda:geo.oracle(norm,state),case=r,source=source);sealed(op,dict(value=value,call_id=cid,expected=expected))
    o=load(op);check_oracle(o['value'],norm,orig,{v['index']:v for v in records},expected);f.close()
   sealed(d/'labels.json',dict(indices=sel['required_indices'],records=records));done+=1;status('official_scoring',units_completed=done,total_units=225,current_case=r)
 b.save('complete');sealed(BASE/'scoring_complete.json',dict(at=now(),units=225))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):prepare() if sys.argv[1]=='prepare' else score()

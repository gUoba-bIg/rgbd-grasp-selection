"""Complete official-support scoring with independent upstream oracle checks."""
import os,sys,json,time,uuid,argparse,functools
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/build/graspnetAPI_eval_v7'))
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
from scripts.p3_train_v27 import CONFIG,LOG,event,sealed,load_sealed,resource_guard
OUT=ROOT/'results/P2P3/candidate_v27'

def run(stage):
 import numpy as np,open3d as o3d
 from graspnetAPI import GraspGroup
 from graspnetAPI.utils import eval_utils as official
 from graspnetAPI.utils.config import get_config
 from graspnetAPI.utils.xmlhandler import xmlReader
 from scripts.verify_grasp_geometry_v10 import select_groups,exact_indices,score_configs,LazyModels
 c=read(CONFIG);need(sha(CONFIG)==read(LOG/'freeze.json')['sha256'],'Protocol changed');root=ROOT/c['data_root'];api=(np,o3d,GraspGroup,official,get_config,xmlReader)
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Frozen source changed '+p)
 if (OUT/(stage+'_gate.json')).exists():print(json.dumps(load_sealed(OUT/(stage+'_gate.json'))),flush=True);return
 start=time.monotonic();usage=LOG/('scoring_usage_'+uuid.uuid4().hex+'.json');previous=sum(read(p)['seconds'] for p in LOG.glob('scoring_usage_*.json'))
 def guard():
  elapsed=time.monotonic()-start;resource_guard(c);atomic(usage,dict(at=now(),seconds=elapsed,state='running'));need(previous+elapsed<c['limits']['scoring_seconds'],'P3 scoring time budget reached')
 @functools.lru_cache(maxsize=40)
 def model_points(oid):
  pc=o3d.io.read_point_cloud(str(root/f'models/{oid:03d}/nontextured.ply'));p=np.asarray(pc.points);need(p.ndim==2 and p.shape[1]==3 and np.isfinite(p).all(),'Invalid model point cloud');return official.voxel_sample_points(p,.008)
 @functools.lru_cache(maxsize=16)
 def load_model(oid):return official.load_dexnet_model(str(root/f'models/{oid:03d}/textured'))
 reports=[]
 for number,item in enumerate(c[stage+'_evaluation']):
  guard();scene,frame=item['scene'],item['frame'];folder=OUT/stage/f'scene_{scene:04d}_{frame:04d}'
  generated=load_sealed(folder/'generation.json');need(stat(folder/'candidates.npy')==generated['candidates'],'Candidate cache changed');cand=np.load(folder/'candidates.npy');d=root/f'scenes/scene_{scene:04d}/realsense'
  if not len(cand):
   result=dict(scene=scene,frame=frame,candidate_count=0,selected_count=0,positive_mu04=0,prefix_mean=0.,prefix_matrix=np.zeros((50,6)).tolist(),records=[],oracle_executed=False);sealed(folder/'score.json',result);reports.append(result);continue
  poses_with_id=[official.parse_posevector(v) for v in xmlReader(str(d/f'annotations/{frame:04d}.xml')).getposevectorlist()];ids=[x[0] for x in poses_with_id];poses=[x[1] for x in poses_with_id];models=[model_points(i) for i in ids]
  clipped=cand.copy();clipped[:,1]=np.clip(clipped[:,1],0,.1);groups,transformed,world,selection=select_groups(clipped,models,poses,GraspGroup,official,np)
  transform=np.linalg.inv(np.load(d/'cam0_wrt_table.npy')@np.load(d/'camera_poses.npy')[frame]);table=official.transform_points(official.create_table_points(1.,1.,.05,dx=-.5,dy=-.5,dz=-.05,grid_size=.008),transform);world=np.concatenate([world,table])
  class Models:
   def __getitem__(self,slot):return load_model(ids[slot])
  dex=Models();friction,configs=score_configs(api);records=[];originals=[]
  for slot,group in enumerate(groups):
   for g,idx in zip(group,exact_indices(clipped,group,np)):originals.append((slot,g,idx))
  for slot,g,idx in originals:
   guard();path=folder/'scores'/f'{idx:04d}.json'
   if path.exists():r=load_sealed(path);need(r['grasp']==g.tolist(),'Sealed candidate identity differs');records.append(r);continue
   cid=uuid.uuid4().hex;event('score_intent',call_id=cid,stage=stage,scene=scene,frame=frame,index=idx);t=time.monotonic()
   collision,empty,grasps=official.collision_detection([g[None]],[transformed[slot]],[None],[poses[slot]],world,outlier=.05,empty_thresh=10,return_dexgrasps=True);co=bool(collision[0][0]);em=bool(empty[0][0]);grasp=grasps[0][0];score=-1.
   if not co and grasp is not None:score=float(official.get_grasp_score(grasp,dex[slot],friction,configs))
   need(score in [-1.,.2,.4,.6,.8,1.,1.2],'Unexpected official score');event('score_returned',call_id=cid,seconds=time.monotonic()-t,score=score)
   r=dict(call_id=cid,index=idx,object_slot=slot,object_id=ids[slot],grasp=g.tolist(),collision=co,empty=em,score=score);sealed(path,r);records.append(r)
  oracle_executed=number==0
  if oracle_executed:
   op=folder/'upstream_oracle.json'
   if op.exists():oracle=load_sealed(op)
   else:
    cid=uuid.uuid4().hex;event('oracle_intent',call_id=cid,stage=stage,scene=scene,frame=frame);t=time.monotonic();gg,ss,cc=official.eval_grasp(GraspGroup(clipped.copy()),models,dex,poses,get_config(),table=table,voxel_size=.008,TOP_K=50);oracle=dict(groups=[g.tolist() for g in gg],scores=[s.tolist() for s in ss],collision=[x.tolist() for x in cc]);event('oracle_returned',call_id=cid,seconds=time.monotonic()-t);sealed(op,oracle)
   expected=dict(groups=[g.tolist() for g in groups],scores=[[r['score'] for r in records if r['object_slot']==j] for j in range(len(ids))],collision=[[r['collision'] for r in records if r['object_slot']==j] for j in range(len(ids))]);need(oracle==expected,'Same-settings upstream oracle mismatch')
  order=np.argsort(-np.asarray([r['grasp'][0] for r in records],dtype=clipped.dtype));scores=np.asarray([records[i]['score'] for i in order]);matrix=np.array([[np.sum((scores[:k]>0)&(scores[:k]<=mu))/k for mu in [.2,.4,.6,.8,1.,1.2]] for k in range(1,51)])
  result=dict(scene=scene,frame=frame,candidate_count=len(cand),selected_count=len(records),positive_mu04=int(((scores>0)&(scores<=.4)).sum()),prefix_mean=float(matrix.mean()),prefix_matrix=matrix.tolist(),records=records,oracle_executed=oracle_executed,oracle_equal=True if oracle_executed else None,selection=selection)
  sealed(folder/'score.json',result);reports.append(result)
 positive_scenes=len({r['scene'] for r in reports if r['positive_mu04']>0});total_positive=sum(r['positive_mu04'] for r in reports)
 report=dict(at=now(),stage=stage,frames=len(reports),scenes=len(set(r['scene'] for r in reports)),positive_scenes_mu04=positive_scenes,positive_candidates_mu04=total_positive,mean_frame_prefix_matrix=float(np.mean([r['prefix_mean'] for r in reports])),full_official_test_AP=False,robot_success=False,frame_reports=[f'{stage}/scene_{r["scene"]:04d}_{r["frame"]:04d}/score.json' for r in reports])
 if stage=='early':report['continue_training']=positive_scenes>=c['early_gate']['positive_scenes_min'] and total_positive>=c['early_gate']['positive_candidates_min']
 else:report['P7_candidate_baseline_usable']=positive_scenes>=c['final_gate']['positive_scenes_min'] and total_positive>=c['final_gate']['positive_candidates_min']
 sealed(OUT/(stage+'_gate.json'),report);atomic(usage,dict(at=now(),seconds=time.monotonic()-start,state='complete'));print(json.dumps(report),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['early','final'],required=True);args=p.parse_args()
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run(args.stage)

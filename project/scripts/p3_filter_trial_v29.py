"""One bounded missing-postprocessing repair using the five existing raw pools."""
import os,sys,time,json,uuid,functools,argparse
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/build/graspnetAPI_eval_v7'))
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
from scripts.p3_train_v27 import sealed,load_sealed,resource_guard
LOG=ROOT/'logs/P3/repair_v29/filter_trial';OUT=ROOT/'results/P3/repair_v29/filter_trial';CONFIG=ROOT/'configs/p3_filter_trial_v29.json';OLD=ROOT/'results/P2P3/candidate_v27/early'

def event(kind,**kw):
 with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())

def verify_sources():
 c=read(CONFIG);need(sha(CONFIG)==read(LOG/'freeze.json')['sha256'],'Repair config changed')
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Prior/source artifact changed '+p)
 return c

def run():
 import numpy as np,open3d as o3d
 from PIL import Image
 from scipy.io import loadmat
 from graspnetAPI import GraspGroup
 from graspnetAPI.utils import eval_utils as official
 from graspnetAPI.utils.config import get_config
 from graspnetAPI.utils.xmlhandler import xmlReader
 from scripts.generate_multiview_grasps_v14r1 import reconstruct_points
 from scripts.verify_grasp_geometry_v10 import select_groups,exact_indices,score_configs
 from src.grasp_observed_filter_v29 import filter_candidates,independent_mask
 c=verify_sources();LOG.mkdir(parents=True,exist_ok=True);start=time.monotonic();usage=LOG/('usage_'+uuid.uuid4().hex+'.json');previous=sum(read(p)['seconds'] for p in LOG.glob('usage_*.json'));prior_calls=[json.loads(s) for s in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()] if (LOG/'calls.jsonl').exists() else [];intent_count=sum(r['kind']=='score_intent' for r in prior_calls)
 def guard():
  atomic(usage,dict(at=now(),seconds=time.monotonic()-start));need(previous+time.monotonic()-start<c['seconds'],'Trial time budget');resource_guard(c)
 root=ROOT/c['data_root'];api=(np,o3d,GraspGroup,official,get_config,xmlReader);friction,configs=score_configs(api)
 @functools.lru_cache(maxsize=40)
 def points(oid):return official.voxel_sample_points(np.asarray(o3d.io.read_point_cloud(str(root/f'models/{oid:03d}/nontextured.ply')).points),.008)
 @functools.lru_cache(maxsize=16)
 def dex(oid):return official.load_dexnet_model(str(root/f'models/{oid:03d}/textured'))
 reports=[]
 for scene in c['scenes']:
  guard();old=OLD/f'scene_{scene:04d}_0000';d=OUT/f'scene_{scene:04d}_0000';d.mkdir(parents=True,exist_ok=True);source=load_sealed(old/'generation.json');a=np.load(old/'candidates.npy');base=root/f'scenes/scene_{scene:04d}/realsense';m=loadmat(base/'meta/0000.mat',variable_names=['intrinsic_matrix','factor_depth']);depth=np.asarray(Image.open(base/'depth/0000.png'));cloud=reconstruct_points(depth,m['intrinsic_matrix'],float(m['factor_depth'].item()))['valid_points']
  if (d/'filter.json').exists():f=load_sealed(d/'filter.json');need(stat(d/'filter.npz')==f['artifact'],'Filter artifact changed');payload=np.load(d/'filter.npz');keep=payload['keep'];voxels=payload['voxels']
  else:
   keep,voxels=filter_candidates(a,cloud,GraspGroup);need(np.array_equal(keep,independent_mask(a,voxels)),'Independent observed collision algebra differs');np.savez_compressed(d/'filter.npz',keep=keep,voxels=voxels);sealed(d/'filter.json',dict(artifact=stat(d/'filter.npz'),raw_candidates=stat(old/'candidates.npy'),eligible=int(keep.sum()),removed=int((~keep).sum()),GT_used=False,independent_equal=True))
  parsed=[official.parse_posevector(v) for v in xmlReader(str(base/'annotations/0000.xml')).getposevectorlist()];ids=[p[0] for p in parsed];poses=[p[1] for p in parsed];models=[points(i) for i in ids];records=[];selection={}
  if keep.any():
   groups,transformed,world,selection=select_groups(a[keep],models,poses,GraspGroup,official,np);table=official.transform_points(official.create_table_points(1.,1.,.05,dx=-.5,dy=-.5,dz=-.05,grid_size=.008),np.linalg.inv(np.load(base/'cam0_wrt_table.npy')@np.load(base/'camera_poses.npy')[0]));world=np.concatenate([world,table]);old_scores={r['index']:r for r in load_sealed(old/'score.json')['records']}
   for slot,group in enumerate(groups):
    for g,index in zip(group,exact_indices(a,group,np)):
     guard();p=d/'scores'/f'{index:04d}.json'
     if p.exists():r=load_sealed(p)
     elif index in old_scores:
      r=dict(old_scores[index],reused_old=True);need(r['grasp']==g.tolist(),'Old score identity differs');sealed(p,r)
     else:
      need(intent_count<c['max_new_scores'],'New official score budget');cid=uuid.uuid4().hex;event('score_intent',call_id=cid,scene=scene,index=index);intent_count+=1;t=time.monotonic();co,em,gg=official.collision_detection([g[None]],[transformed[slot]],[None],[poses[slot]],world,outlier=.05,empty_thresh=10,return_dexgrasps=True);collision=bool(co[0][0]);empty=bool(em[0][0]);score=-1.
      if not collision and gg[0][0] is not None:score=float(official.get_grasp_score(gg[0][0],dex(ids[slot]),friction,configs))
      event('score_returned',call_id=cid,score=score,seconds=time.monotonic()-t);r=dict(call_id=cid,index=index,object_slot=slot,object_id=ids[slot],grasp=g.tolist(),collision=collision,empty=empty,score=score,reused_old=False);sealed(p,r)
     need(keep[index] and r['grasp']==g.tolist(),'Scored candidate not eligible');records.append(r)
   if scene==c['scenes'][0]:
    p=d/'oracle.json'
    if not p.exists():
     class Models:
      def __getitem__(self,i):return dex(ids[i])
     cid=uuid.uuid4().hex;event('oracle_intent',call_id=cid,scene=scene);t=time.monotonic();gg,ss,cc=official.eval_grasp(GraspGroup(a[keep].copy()),models,Models(),poses,get_config(),table=table,voxel_size=.008,TOP_K=50);oracle=dict(call_id=cid,groups=[g.tolist() for g in gg],scores=[s.tolist() for s in ss],collisions=[x.tolist() for x in cc]);event('oracle_returned',call_id=cid,seconds=time.monotonic()-t);sealed(p,oracle)
    oracle=load_sealed(p);need(oracle['groups']==[g.tolist() for g in groups] and oracle['scores']==[[r['score'] for r in records if r['object_slot']==i] for i in range(len(ids))] and oracle['collisions']==[[r['collision'] for r in records if r['object_slot']==i] for i in range(len(ids))],'Same-settings upstream scoring differs')
  scores=np.array([r['score'] for r in sorted(records,key=lambda r:-r['grasp'][0])]);matrix=np.array([[sum(0<s<=mu for s in scores[:k])/k for mu in [.2,.4,.6,.8,1,1.2]] for k in range(1,51)]);oldpositives={r['index'] for r in load_sealed(old/'score.json')['records'] if 0<r['score']<=.4};r=dict(scene=scene,raw_count=len(a),eligible_count=int(keep.sum()),selected_count=len(records),positive_candidates_mu04=int(((scores>0)&(scores<=.4)).sum()),old_positive_removed=sorted(i for i in oldpositives if not keep[i]),prefix_mean=float(matrix.mean()),prefix_matrix=matrix.tolist(),records=records,selection=selection)
  if (d/'result.json').exists():need(load_sealed(d/'result.json')==r,'Repeated frame result differs')
  else:sealed(d/'result.json',r)
  reports.append(r);atomic(ROOT/'runtime/pipeline/p3_repair_v29.json',dict(at=now(),phase='cached_collision_filter_trial',frames_completed=len(reports),frames=5,all_tasks_percent=43,experiment_percent=53.75,P7_started=False))
 positives=sum(r['positive_candidates_mu04'] for r in reports);scenes=sum(r['positive_candidates_mu04']>0 for r in reports);result=dict(at=now(),frames=5,positive_scenes_mu04=scenes,positive_candidates_mu04=positives,gate_passed=scenes>=3 and positives>=10,thresholds_unchanged=True,old_positive_removed=sum(len(r['old_positive_removed']) for r in reports),new_model_forwards=0,new_training_updates=0,reused_scores=sum(r['reused_old'] for f in reports for r in f['records']),new_score_results=sum(not r['reused_old'] for f in reports for r in f['records']),scope='five exposed training frames, not held-out validation')
 sealed(OUT/'result.json',result);guard();print(json.dumps(result),flush=True)

def review(cold=False):
 import numpy as np
 c=verify_sources();reports=[];oldcalls=[json.loads(s) for s in (ROOT/'logs/P2P3/candidate_v27/calls.jsonl').read_text(encoding='utf-8').splitlines()];calls=[json.loads(s) for s in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()] if (LOG/'calls.jsonl').exists() else [];returned={r['call_id']:r for r in oldcalls+calls if r['kind']=='score_returned'}
 from src.grasp_observed_filter_v29 import independent_mask
 for scene in c['scenes']:
  d=OUT/f'scene_{scene:04d}_0000';a=np.load(OLD/f'scene_{scene:04d}_0000/candidates.npy');f=load_sealed(d/'filter.json');need(stat(d/'filter.npz')==f['artifact'],'Filter payload differs');payload=np.load(d/'filter.npz');need(np.array_equal(payload['keep'],independent_mask(a,payload['voxels'])),'Independent filter failed');r=load_sealed(d/'result.json');records=r['records'];quality=[]
  for row in sorted(records,key=lambda x:-x['grasp'][0]):
   need(payload['keep'][row['index']] and row['grasp']==a[row['index']].tolist(),'Candidate/keep identity differs');need(row==load_sealed(d/'scores'/f'{row["index"]:04d}.json') and row['call_id'] in returned and returned[row['call_id']]['score']==row['score'],'Score lacks actual return');quality.append(row['score'])
  expected=np.zeros((50,6))
  for k in range(50):
   for j,mu in enumerate([.2,.4,.6,.8,1,1.2]):expected[k,j]=np.count_nonzero((np.array(quality[:k+1])>0)&(np.array(quality[:k+1])<=mu))/(k+1)
  need(np.array_equal(expected,np.asarray(r['prefix_matrix'])) and r['positive_candidates_mu04']==sum(0<s<=.4 for s in quality),'Independent matrix/positive counts differ');reports.append(r)
 result=load_sealed(OUT/'result.json');ps=sum(r['positive_candidates_mu04']>0 for r in reports);pc=sum(r['positive_candidates_mu04'] for r in reports);need((ps,pc)==(result['positive_scenes_mu04'],result['positive_candidates_mu04']) and result['gate_passed']==(ps>=3 and pc>=10),'Gate result differs');files={p.relative_to(ROOT).as_posix():stat(p) for p in OUT.rglob('*') if p.is_file()};proof=dict(at=now(),engineering_passed=True,scientific_gate_passed=result['gate_passed'],files=files,new_model_forwards=0,new_training_updates=0,new_official_scores=0)
 if cold:need(files==read(LOG/'readonly.json')['files'],'Cold reentry file mismatch')
 once(LOG/('cold.json' if cold else 'readonly.json'),proof);print('Full trial read-only review passed; scientific gate='+str(result['gate_passed']),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--review',action='store_true');p.add_argument('--cold',action='store_true');args=p.parse_args()
 from scripts.pipeline import project_lock
 with project_lock(ROOT):review(args.cold) if args.review or args.cold else run()

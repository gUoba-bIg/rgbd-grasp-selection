"""Validate only the changed filter integration; reuse sealed unchanged checks."""
import os,sys,ast,time,json
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/build/graspnetAPI_eval_v7'))
from scripts.p6_common_v24 import ROOT,read,stat,need,once,now
LOG=ROOT/'logs/P7/preflight_v31'

def run():
 import numpy as np
 from PIL import Image
 from scipy.io import loadmat
 from graspnetAPI import GraspGroup
 from scripts.p7_pool_v31 import select,commit_pool,review_pool
 from src.grasp_observed_filter_v29 import independent_mask
 from scripts.p3_filter_support_v30 import review_filter
 from scripts.p3_train_v30 import sealed
 old=read(ROOT/'logs/P7/preflight_v28/checks.json');need(old['passed'],'Old checks incomplete')
 for p,s in old['sources'].items():need(stat(ROOT/p)==s,'Old preparation source changed '+p)
 for key in ['vision','geometry','real_geometry']:need(stat(ROOT/f'logs/P7/preflight_v28/{key}.json')==old[key],'Old preparation result changed')
 # Unchanged mathematics and model calls inherit the exact previous validation.
 normalized=[]
 for name in ['generate','fit','model_identity','finish']:
  before=(ROOT/f'scripts/p7_{name}_v28.py').read_text(encoding='utf-8-sig');after=(ROOT/f'scripts/p7_{name}_v31.py').read_text(encoding='utf-8-sig')
  expected=before.replace('_v28','_v31').replace('results/P2P3/candidate_v27','results/P3/candidate_v30').replace('logs/P2P3/candidate_v27','logs/P3/candidate_v30').replace('experiments/candidate_v27','experiments/candidate_v30').replace('configs/candidate_v27.json','configs/candidate_v30.json')
  need(after==expected,'Unexpected change to previously checked '+name);normalized.append(name)
 for name in ['observations','metrics']:need((ROOT/f'src/risk_{name}_v28.py').read_bytes()==(ROOT/f'src/risk_{name}_v31.py').read_bytes(),'Changed math requires new checks')
 start=time.monotonic();rows=[];LOG.mkdir(parents=True,exist_ok=True)
 for scene in [3,4,5,8,9]:
  need(time.monotonic()-start<600,'Filter integration preparation time budget')
  source=ROOT/f'results/P2P3/candidate_v27/early/scene_{scene:04d}_0000';trial=ROOT/f'results/P3/repair_v29/filter_trial/scene_{scene:04d}_0000';cand=np.load(source/'candidates.npy');payload=np.load(trial/'filter.npz');keep=payload['keep'];voxels=payload['voxels'];need(np.array_equal(keep,independent_mask(cand,voxels)),'Changed filter algebra')
  chosen=select(cand,keep,GraspGroup);group=GraspGroup(cand[keep].copy()).nms(.03,np.pi/6);group.sort_by_score();expected=[int(np.flatnonzero(np.all(cand==g,axis=1))[0]) for g in group[:20].grasp_group_array];need(chosen==expected and all(keep[chosen]),'P7 pool differs from fixed filtered top20')
  # Verify the exact P3 final-review entry against independently reconstructed full observation.
  sd=ROOT/f'data/raw/prerequisites_v26/scenes/scene_{scene:04d}/realsense';m=loadmat(sd/'meta/0000.mat');depth=np.asarray(Image.open(sd/'depth/0000.png'));z=depth/float(m['factor_depth'].item());v,u=np.indices(z.shape);K=m['intrinsic_matrix'];cloud=np.stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z),-1);valid=np.isfinite(cloud).all(-1)&(z>0)&(z<=2)
  d=LOG/f'P3_review_scene_{scene:04d}';d.mkdir(exist_ok=False);np.savez_compressed(d/'observed_filter.npz',keep=keep,voxels=voxels);sealed(d/'observed_filter.json',dict(artifact=stat(d/'observed_filter.npz'),kept=int(keep.sum()),discarded=int((~keep).sum()),GT_used=False,voxel_size=.01,collision_thresh=.01,approach_dist=.05));need(np.array_equal(review_filter(d,cand,cloud[valid].astype(np.float32)),keep),'P3 final review integration mismatch')
  rows.append(dict(scene=scene,raw=len(cand),eligible=int(keep.sum()),pool=len(chosen)))
  if scene==3:
   im=Image.open(sd/'depth/0000.png');size=im.size;observed=np.asarray(im.resize((640,480),Image.Resampling.NEAREST),np.float32)/float(m['factor_depth'].item());im.close();mask=np.isfinite(observed)&(observed>0);observed=np.where(mask,np.minimum(observed,2.)/2,0).astype(np.float32)*2
   for empty in [False,True]:
    a=dict(candidates=cand[:0] if empty else cand,observed_depth=observed,observed_valid=mask,K=K,original_size=size);dest=LOG/('P7_empty' if empty else 'P7_resized');dest.mkdir(exist_ok=False);indices=commit_pool(dest,a,dict(scene=scene,condition='preparation_only'),GraspGroup);need(review_pool(dest,a,GraspGroup)==indices,'P7 actual producer/independent reader mismatch')
    if empty:need(indices==[],'Empty pool must remain empty')
  payload.close()
 report=dict(at=now(),passed=True,original_frames=rows,all_original_candidates=sum(r['raw'] for r in rows),P3_final_reader_integration=True,P7_resized_and_empty_pool_roundtrip=True,unchanged_code_parity=normalized,new_neural_forwards=0,new_training_updates=0,new_official_quality_scores=0,seconds=time.monotonic()-start,not_formal_experiment=True)
 once(LOG/'filter_pool.json',report)
 files=list((ROOT/'scripts').glob('p7_*_v31.py'))+list((ROOT/'src').glob('risk_*_v31.py'))+[ROOT/'src/grasp_observed_filter_v29.py',ROOT/'docs/P7固定候选风险与独立校准_v31_草案.txt']
 for p in files:
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
 once(LOG/'checks.json',dict(at=now(),passed=True,old_checks=stat(ROOT/'logs/P7/preflight_v28/checks.json'),filter_pool=stat(LOG/'filter_pool.json'),sources={p.relative_to(ROOT).as_posix():stat(p) for p in files},not_formal_experiment=True));print(json.dumps(report),flush=True)

if __name__=='__main__':run()

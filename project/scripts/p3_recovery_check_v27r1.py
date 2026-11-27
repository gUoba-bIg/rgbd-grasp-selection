"""Real diverse-scene batch parity after releasing mapped label cache."""
import os,sys,time,hashlib,json
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,need,now
from scripts.p3_train_v27 import LOG,CONFIG,resource_guard

def run():
 import numpy as np,torch
 from src.grasp_prerequisite_data_v27 import ExplicitSceneDataset,mapped_npz
 from src.grasp_prefetch_v27 import TrainingBatches
 torch.set_num_threads(2);c=read(CONFIG);log=LOG/'recovery_r1';source=ROOT/'scripts/p3_train_v27r1.py'
 once(log/'freeze.json',dict(at=now(),source=stat(source),checker=stat(Path(__file__)),original_protocol_sha=sha(CONFIG),change='serial CPU batches; clear mmap cache between batches; all samples/RNG/optimizer/limits unchanged',max_equality_batches=4,model_forwards=0,updates=0))
 def digest(value):
  if isinstance(value,dict):return {k:digest(v) for k,v in value.items()}
  if isinstance(value,(tuple,list)):return [digest(v) for v in value]
  if isinstance(value,torch.Tensor):return [str(value.dtype),list(value.shape),hashlib.sha256(value.numpy().tobytes()).hexdigest()]
  return value
 d=ExplicitSceneDataset(ROOT/c['data_root'],c['train_scenes'],c['frames'],augment=True);order=np.random.RandomState(c['seed']+1).permutation(len(d)).tolist();batches=TrainingBatches(d,order,c['seed'],1);rows=[];start=time.monotonic()
 for i in range(4):
  expected=digest(batches[i]);mapped_npz.cache_clear();actual=digest(batches[i]);mapped_npz.cache_clear();need(actual==expected,'Cache-release batch mismatch');resource_guard(c);rows.append(dict(batch=i,dataset_indices=order[2*i:2*i+2],bitwise_equal=True))
 once(log/'equality.json',dict(at=now(),passed=True,batches=rows,seconds=time.monotonic()-start,model_forwards=0,training_updates=0));print('4 diverse-scene serial batches exactly equal; memory guard passed',flush=True)

if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()

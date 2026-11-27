"""Actual serial/multiprocess batch equality, no model calls or optimization."""
import os,sys,time,torch,json
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,stat,once,need,now
from src.grasp_prerequisite_data_v27 import ExplicitSceneDataset
from src.grasp_prefetch_v27 import TrainingBatches,loader
LOG=ROOT/'logs/P2P3/prefetch_check_v27'

def compare(a,b):
 if isinstance(a,dict):need(set(a)==set(b),'Batch keys differ');[compare(a[k],b[k]) for k in a]
 elif isinstance(a,(tuple,list)):need(len(a)==len(b),'Batch lengths differ');[compare(x,y) for x,y in zip(a,b)]
 elif isinstance(a,torch.Tensor):need(a.dtype==b.dtype and torch.equal(a,b),'Prefetch values differ')
 else:need(a==b,'Batch index differs')

def run():
 torch.set_num_threads(2);paths=[Path(__file__),ROOT/'src/grasp_prefetch_v27.py',ROOT/'src/grasp_prerequisite_data_v27.py'];sources={p.relative_to(ROOT).as_posix():stat(p) for p in paths}
 once(LOG/'freeze.json',dict(at=now(),sources=sources,batches=4,workers=2,prefetch=1,training_updates=0))
 frames=ExplicitSceneDataset(ROOT/'data/raw/prerequisites_v26',[3],[0,16],augment=True,image_root=ROOT/'data/raw/graspnet_coverage_v8');ds=TrainingBatches(frames,[0,1]*4,20261021,1);start=time.monotonic()
 for i,actual in enumerate(loader(ds)):compare(ds[i],actual)
 once(LOG/'result.json',dict(at=now(),passed=True,batches=4,bitwise_identical_to_serial=True,seconds=time.monotonic()-start,model_forwards=0,training_updates=0,sources=sources));print('4 actual prefetched batches equal serial inputs',flush=True)

if __name__=='__main__':run()

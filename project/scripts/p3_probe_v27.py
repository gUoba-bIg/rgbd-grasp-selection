"""Finite training-resource and upstream-loader equivalence probe on train scene 3."""
import os,sys,json,time,random,uuid,traceback
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,once,stat,sha,need,now
LOG=ROOT/'logs/P2P3/training_probe_v27';DATA=ROOT/'data/raw/prerequisites_v26'

def run():
 import numpy as np,torch
 from src.grasp_prerequisite_data_v27 import ExplicitSceneDataset,GraspNetDataset,collate_fn,to_device
 from graspnet import GraspNet,get_loss
 LOG.mkdir(parents=True,exist_ok=True)
 need(not (LOG/'result.json').exists(),'Probe already returned; reuse result')
 # Probe only consumes fully committed labels and existing frozen image frames.
 objects=[2,21,22,41,43,44,52,60,62];paths=[DATA/'collision_label/scene_0003/collision_labels.npz',DATA/'scenes/scene_0003/object_id_list.txt']
 paths += [DATA/f'{kind}/{obj:03d}{suffix}' for obj in objects for kind,suffix in [('grasp_label','_labels.npz'),('tolerance','_tolerance.npy')]]
 receipts={r['path']:r for r in [json.loads(s) for s in (ROOT/'logs/P2P3/prerequisites_v26/member_receipts.jsonl').read_text(encoding='utf-8').splitlines()]}
 for p in paths:
  r=receipts.get(p.relative_to(DATA).as_posix());need(r and sha(p)==r['sha256'],'Missing/unverified probe input '+str(p))
 protected={p.relative_to(ROOT).as_posix():stat(p) for p in paths}
 sources=[Path(__file__),ROOT/'src/grasp_prerequisite_data_v27.py']
 protected.update({p.relative_to(ROOT).as_posix():stat(p) for p in sources})
 conf=dict(at=now(),seed=20261021,scenes=[3],frames=[0,16],steps=2,batch_size=2,num_points=20000,lr=.001,max_seconds=600,max_GPU_bytes=int(13.5*1024**3),max_process_rss_bytes=24*1024**3,train_only=True,formal_checkpoint_reuse=False,protected=protected)
 once(LOG/'freeze.json',conf);start=time.monotonic()
 def log(kind,**kw):
  with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
 def compare(a,b,p=''):
  if isinstance(a,dict):need(set(a)==set(b),'Output keys differ');[compare(a[k],b[k],p+'/'+k) for k in a]
  elif isinstance(a,list):need(len(a)==len(b),'List length differs '+p);[compare(x,y,p) for x,y in zip(a,b)]
  else:need(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),'Upstream loader mismatch '+p)
 d=ExplicitSceneDataset(DATA,[3],[0,16],augment=True,image_root=ROOT/'data/raw/graspnet_coverage_v8')
 # Build the same upstream object without invoking its constructor/global split.
 reference=GraspNetDataset.__new__(GraspNetDataset);reference.__dict__.update(d.__dict__)
 for index in range(2):
  np.random.seed(20261021+index);actual=d[index];np.random.seed(20261021+index);expected=GraspNetDataset.get_data_label(reference,index);compare(actual,expected)
 del actual,expected
 np.random.seed(20261021);random.seed(20261021);torch.manual_seed(20261021);torch.cuda.manual_seed_all(20261021);torch.set_num_threads(2)
 net=GraspNet(input_feature_dim=0,num_view=300,num_angle=12,num_depth=4,cylinder_radius=.05,hmin=-.02,hmax_list=[.01,.02,.03,.04]).cuda().train();opt=torch.optim.Adam(net.parameters(),lr=.001)
 losses=[];times=[];torch.cuda.reset_peak_memory_stats()
 for step in range(2):
  need(time.monotonic()-start<600,'Probe time budget');t=time.monotonic();batch=to_device(collate_fn([d[0],d[1]]),'cuda');cid=uuid.uuid4().hex;log('training_step_intent',call_id=cid,step=step)
  opt.zero_grad(set_to_none=True);out=net(batch);loss,_=get_loss(out);need(torch.isfinite(loss).item(),'Non-finite probe loss');loss.backward();opt.step();torch.cuda.synchronize();elapsed=time.monotonic()-t
  losses.append(float(loss.detach()));times.append(elapsed);log('training_step_returned',call_id=cid,loss=losses[-1],seconds=elapsed)
  need(torch.cuda.max_memory_reserved()<=conf['max_GPU_bytes'],'Probe GPU memory budget');del batch,out,loss
 checkpoint=LOG/'probe_only_do_not_train_from.pt';torch.save(dict(model_state_dict=net.state_dict(),optimizer_state_dict=opt.state_dict(),probe_only=True),checkpoint)
 import ctypes
 class Memory(ctypes.Structure):_fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ['peak_rss','rss','peakpaged','paged','peaknonpaged','nonpaged','pagefile','peakpagefile']]
 m=Memory();m.cb=ctypes.sizeof(m);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p;ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Memory),ctypes.c_ulong];need(ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(m),m.cb),'RSS query failed');need(m.peak_rss<=conf['max_process_rss_bytes'],'Probe RAM budget')
 for p,v in protected.items():need(stat(ROOT/p)==v,'Probe input/source changed')
 once(LOG/'result.json',dict(at=now(),loader_equal_to_upstream=True,known_training_updates=2,unknown_training_calls=0,losses=losses,step_seconds=times,projected_10080_updates_seconds=max(times)*10080,peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),peak_rss_bytes=m.peak_rss,total_seconds=time.monotonic()-start,checkpoint=stat(checkpoint),not_formal_training=True))
 print(json.dumps(read(LOG/'result.json')),flush=True)

if __name__=='__main__':
 try:run()
 except Exception as exc:
  once(LOG/('failure_'+uuid.uuid4().hex+'.json'),dict(at=now(),error=repr(exc),traceback=traceback.format_exc()));raise

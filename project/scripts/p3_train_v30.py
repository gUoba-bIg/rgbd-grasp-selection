"""Explicit 70-scene, fixed-view candidate model; no validation/calibration updates."""
import os,sys,time,json,random,uuid,argparse,traceback,shutil,ctypes,hashlib
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
LOG=ROOT/'logs/P3/candidate_v30';OUT=ROOT/'experiments/candidate_v30';CONFIG=ROOT/'configs/candidate_v30.json'
OLDLOG=ROOT/'logs/P2P3/candidate_v27';OLDOUT=ROOT/'experiments/candidate_v27'

def all_events():
 return [json.loads(s) for base in [OLDLOG,LOG] if (base/'calls.jsonl').exists() for s in (base/'calls.jsonl').read_text(encoding='utf-8').splitlines()]

def resource_guard(c,check_outputs=False,workers=()):
 class Memory(ctypes.Structure):_fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ['peak_rss','rss','peakpaged','paged','peaknonpaged','nonpaged','pagefile','peakpagefile']]
 m=Memory();m.cb=ctypes.sizeof(m);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p;ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Memory),ctypes.c_ulong]
 need(ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(m),m.cb),'Memory query failed');combined=m.peak_rss
 ctypes.windll.kernel32.OpenProcess.restype=ctypes.c_void_p;ctypes.windll.kernel32.CloseHandle.argtypes=[ctypes.c_void_p]
 for worker in workers:
  handle=ctypes.windll.kernel32.OpenProcess(0x410,False,worker)
  if handle:
   child=Memory();child.cb=ctypes.sizeof(child)
   try:need(ctypes.windll.psapi.GetProcessMemoryInfo(handle,ctypes.byref(child),child.cb),'Worker memory query failed');combined+=child.peak_rss
   finally:ctypes.windll.kernel32.CloseHandle(handle)
 need(combined<c['limits']['rss_bytes'],'Combined process RAM budget exceeded');need(shutil.disk_usage(ROOT).free>=c['limits']['disk_reserve_bytes'],'Disk reserve reached')
 if check_outputs:
  total=sum(p.stat().st_size for d in [OUT,ROOT/'results/P3/candidate_v30',LOG,OLDOUT,ROOT/'results/P2P3/candidate_v27',OLDLOG,ROOT/'results/P3/repair_v29',ROOT/'logs/P3/repair_v29'] for p in d.rglob('*') if p.is_file())
  need(total+200*1024**2<c['limits']['output_bytes'],'Stage output budget reached')

def sealed(path,value):
 once(path,value);once(path.with_suffix(path.suffix+'.receipt.json'),stat(path))

def load_sealed(path):
 need(stat(path)==read(path.with_suffix(path.suffix+'.receipt.json')),'Sealed result changed '+str(path));return read(path)

def event(kind,**kw):
 with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())

def snapshot(path,net,opt,epoch,step,order,torch,np):
 need(not path.exists(),'Checkpoint already committed')
 temp=path.with_suffix('.'+uuid.uuid4().hex+'.part');path.parent.mkdir(parents=True,exist_ok=True)
 torch.save(dict(model_state_dict=net.state_dict(),optimizer_state_dict=opt.state_dict(),epoch=epoch,next_step=step,order=order,numpy_rng=np.random.get_state(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config_sha256=sha(CONFIG)),temp);os.replace(temp,path)
 receipt=dict(at=now(),path=path.relative_to(ROOT).as_posix(),**stat(path));once(path.with_suffix('.receipt.json'),receipt);atomic(LOG/'latest_checkpoint.json',receipt)

def run(through):
 import numpy as np,torch
 from src.grasp_prerequisite_data_v27 import ExplicitSceneDataset,collate_fn,to_device
 from src.grasp_prefetch_v27 import TrainingBatches
 from src.grasp_prerequisite_data_v27 import mapped_npz
 from graspnet import GraspNet,get_loss
 need(read(ROOT/'results/P3/repair_v29/filter_trial/result.json')['gate_passed'],'Actual repaired continuation gate missing');need(read(ROOT/'logs/P3/repair_v29/filter_trial/cold.json')['engineering_passed'],'Repaired cold review missing')
 c=read(CONFIG);need(sha(CONFIG)==read(LOG/'freeze.json')['sha256'],'Training protocol changed')
 for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Training source changed '+p)
 need(read(ROOT/'results/P2P3/data_audit_v27r1/result.json')['status']=='passed','P2 audit missing')
 need(through==18,'Only the fixed remaining 4-to-18 segment is authorized here')
 need(read(ROOT/'logs/P2P3/candidate_v27/early_cold.json')['engineering_passed'],'Original first3 lineage incomplete')
 prior_events=all_events()
 intent_count=sum(r['kind']=='update_intent' for r in prior_events)
 previous=sum(read(p)['seconds'] for base in [OLDLOG,LOG] for p in base.glob('usage_*.json'));start=time.monotonic();usage=LOG/('usage_'+uuid.uuid4().hex+'.json')
 def usage_save(state):atomic(usage,dict(seconds=time.monotonic()-start,state=state,at=now()))
 d=ExplicitSceneDataset(ROOT/c['data_root'],c['train_scenes'],c['frames'],augment=True)
 need(len(d)==1120,'Training scope differs');need(set(d.sceneIds).isdisjoint({f'scene_{s:04d}' for s in c['excluded_scenes']}),'Training scene leak')
 np.random.seed(c['seed']);random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.set_num_threads(2)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(c['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
 net=GraspNet(input_feature_dim=0,num_view=300,num_angle=12,num_depth=4,cylinder_radius=.05,hmin=-.02,hmax_list=[.01,.02,.03,.04]).cuda();opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=0)
 initial={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in net.state_dict().items()};once(LOG/'initial_random_model.json',dict(seed=c['seed'],tensor_sha256=initial,pretrained_weight_loaded=False))
 epoch=1;step=0;order=None
 latest=LOG/'latest_checkpoint.json'
 need(latest.exists(),'Required epoch3 continuation receipt missing; fresh restart is forbidden')
 if latest.exists():
  receipt=read(latest);p=ROOT/receipt['path'];need(stat(p)=={k:receipt[k] for k in ['sha256','bytes','mtime_ns']},'Resume checkpoint differs');saved=torch.load(p,map_location='cpu',weights_only=False);need(saved['config_sha256']==sha(CONFIG) or (saved['config_sha256']==c['parent_protocol_sha256'] and receipt['path']==c['parent_checkpoint'] and stat(p)==c['protected'][c['parent_checkpoint']]),'Resume protocol/lineage differs')
  net.load_state_dict(saved['model_state_dict']);opt.load_state_dict(saved['optimizer_state_dict']);epoch=saved['epoch'];step=saved['next_step'];order=saved['order'];np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);del saved
  need(4<=epoch<=19 and 0<=step<=560,'Resume position outside fixed continuation')
  for state in opt.state.values():
   for key,value in state.items():
    if isinstance(value,torch.Tensor) and key!='step':state[key]=value.cuda()
 try:
  while epoch<=through:
   if order is None:order=np.random.RandomState(c['seed']+epoch).permutation(len(d)).tolist()
   net.train();lr=.001*(.1**sum(epoch-1>=n for n in [8,12,16]));momentum=max(.5*.5**((epoch-1)//2),.001)
   for g in opt.param_groups:g['lr']=lr
   for module in net.modules():
    if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.momentum=momentum
   batches=TrainingBatches(d,order,c['seed'],epoch,step)
   def serial_batches():
    for index in range(len(batches)):
     mapped_npz.cache_clear()
     value=batches[index]
     mapped_npz.cache_clear()
     yield value
   batch_iterator=serial_batches();worker_pids=[]
   for batch_idx,cpu_batch in batch_iterator:
    usage_save('running');resource_guard(c,workers=worker_pids);need(previous+time.monotonic()-start<c['limits']['seconds'],'Training cumulative time budget exhausted');need(intent_count<c['limits']['actual_update_intents'],'Training actual invocation budget exhausted')
    t=time.monotonic();batch=to_device(cpu_batch,'cuda');del cpu_batch;cid=uuid.uuid4().hex;event('update_intent',call_id=cid,epoch=epoch,batch=batch_idx,samples=[order[batch_idx*2],order[batch_idx*2+1]]);intent_count+=1
    opt.zero_grad(set_to_none=True);out=net(batch);loss,_=get_loss(out);need(torch.isfinite(loss).item(),'Nonfinite loss');loss.backward();opt.step();torch.cuda.synchronize();value=float(loss.detach());event('update_returned',call_id=cid,loss=value,seconds=time.monotonic()-t,GPU_peak_reserved=torch.cuda.max_memory_reserved());del batch,out,loss
    need(torch.cuda.max_memory_reserved()<=c['limits']['gpu_bytes'],'GPU budget')
    if (batch_idx+1)%100==0:
     resource_guard(c,True)
     snapshot(OUT/f'epoch_{epoch:02d}_batch_{batch_idx+1:04d}.pt',net,opt,epoch,batch_idx+1,order,torch,np)
     atomic(ROOT/'runtime/pipeline/candidate_v30.json',dict(at=now(),phase='training',epoch=epoch,batches_completed=batch_idx+1,batches_per_epoch=560,all_tasks_percent=43,experiment_percent=53.75,P7_started=False))
   resource_guard(c,True);snapshot(OUT/f'epoch_{epoch:02d}_final.pt',net,opt,epoch+1,0,None,torch,np);once(LOG/f'epoch_{epoch:02d}.json',dict(at=now(),epoch=epoch,batches=560,lr=lr,checkpoint=stat(OUT/f'epoch_{epoch:02d}_final.pt')))
   epoch+=1;step=0;order=None
  usage_save('segment_complete');print(json.dumps(dict(completed_through=through,unique_updates=through*560)),flush=True)
 except Exception as exc:usage_save('failed_preserved');event('failure',error=repr(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--through',required=True,type=int);args=a.parse_args()
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run(args.through)

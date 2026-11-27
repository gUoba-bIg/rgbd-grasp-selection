"""Three prespecified CPU seeds; epoch-committed deterministic recovery."""
import os,sys,random,uuid
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p9_common_v35 import *
def snapshot(p,net,opt,epoch,mean,scale,history,source,config_hash=None):
 import torch,numpy as np
 need(not p.exists(),'Uncommitted checkpoint exists; preserve before recovery');p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.'+uuid.uuid4().hex+'.part');state=dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,next_epoch=epoch+1,mean=mean,scale=scale,history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),config_sha256=config_hash or sha(CONF),source=source,scheduler=None,amp_scaler=None,device='cpu')
 with tmp.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);return stat(p)
def run():
 import numpy as np,torch
 from scripts.p7_fit_v31 import model
 c=verify();b=Budget(c,'training');torch.set_num_threads(2);torch.use_deterministic_algorithms(True);data={s:pool(s) for s in ['train','validation']};mean=data['train']['features'].mean(0,dtype=np.float64).astype(np.float32);scale=np.maximum(data['train']['features'].std(0,dtype=np.float64),1e-6).astype(np.float32);xt={s:torch.from_numpy((d['features']-mean)/scale) for s,d in data.items()};yt={s:torch.from_numpy(d['labels']) for s,d in data.items()};source={s:stat(P7/'predictions'/f'{s}.npz') for s in data}
 for seed in c['seeds']:
  out=seedpath(seed)
  if (out/'fit.json').exists():load(out/'fit.json');artifact(out/'weights.npz');continue
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);net=model();opt=torch.optim.Adam(net.parameters(),lr=.001);history=[];best=float('inf');stale=0;chosen=0;start_epoch=1
  saved=sorted((out/'epochs').glob('*.json')) if (out/'epochs').exists() else [];saved=[p for p in saved if not p.name.endswith('.receipt.json')]
  if saved:
   receipt=load(saved[-1]);p=EXP/str(seed)/f"epoch_{receipt['epoch']:02d}.pt";need(stat(p)==receipt['checkpoint'],'Checkpoint changed');v=torch.load(p,map_location='cpu',weights_only=False);need(v['config_sha256']==sha(CONF) and v['source']==source,'Checkpoint input/protocol mismatch');net.load_state_dict(v['model']);opt.load_state_dict(v['optimizer']);history=v['history'];start_epoch=v['next_epoch'];random.setstate(v['python_rng']);np.random.set_state(v['numpy_rng']);torch.set_rng_state(v['torch_rng'])
   for h in history:
    if h['validation_nll']<best:best=h['validation_nll'];chosen=h['epoch'];stale=0
    else:stale+=1
  for epoch in range(start_epoch,31):
   if stale>=5:break
   net.train();order=np.random.RandomState(seed+epoch).permutation(len(yt['train']));callids=[]
   for start in range(0,len(order),256):
    b.guard();ix=order[start:start+256]
    def update():
     opt.zero_grad(set_to_none=True);z=net(xt['train'][ix]).reshape(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,yt['train'][ix]);need(torch.isfinite(loss).item(),'Nonfinite P9 training loss');loss.backward();need(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()),'Nonfinite P9 gradients');opt.step();return float(loss.detach())
    value,cid=invoke('risk_update',update,seed=seed,epoch=epoch,batch=start//256,samples=len(ix));callids.append(cid)
   net.eval()
   def forward():
    with torch.inference_mode():return net(xt['validation']).reshape(-1).numpy().copy()
   z,cid=invoke('risk_forward',forward,seed=seed,phase='epoch_validation',epoch=epoch);nll=float(np.mean(np.logaddexp(0,z.astype(float))-data['validation']['labels']*z));history.append(dict(epoch=epoch,validation_nll=nll));p=EXP/str(seed)/f'epoch_{epoch:02d}.pt';receipt=snapshot(p,net,opt,epoch,mean,scale,history,source);sealed(out/'epochs'/f'{epoch:02d}.json',dict(epoch=epoch,checkpoint=receipt,training_calls=callids,validation_call=cid,nll=nll,samples=len(order),calibration_used=False))
   if nll<best:best=nll;chosen=epoch;stale=0
   else:stale+=1
   status('risk_training',seed=seed,epoch=epoch,best_epoch=chosen)
  need(chosen>0,'No valid checkpoint');p=EXP/str(seed)/f'epoch_{chosen:02d}.pt';net.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);net.eval();weights=out/'weights.npz'
  if weights.exists():artifact(weights)
  else:payload(weights,**{k:v.detach().numpy().copy() for k,v in net.state_dict().items()})
  for split in ['train','calibration','validation']:
   dest=out/'predictions'/f'{split}.npz'
   if dest.exists():artifact(dest);load(dest.with_suffix('.calls.json'));continue
   d=pool(split);X=torch.from_numpy((d['features']-mean)/scale)
   def ff():
    with torch.inference_mode():return net(X).reshape(-1).numpy().copy()
   logits,cid=invoke('risk_forward',ff,seed=seed,phase='selected_pool',split=split);payload(dest,logits=logits);sealed(dest.with_suffix('.calls.json'),dict(call_id=cid,input=stat(P7/'predictions'/f'{split}.npz'),selected_checkpoint=stat(p)))
  sealed(out/'fit.json',dict(at=now(),seed=seed,history=history,selected_epoch=chosen,checkpoint=stat(p),source=source,unique_updates=28*len(history),calibration_used_for_training=False,standardization='train_only',validation_used_for_epoch_choice=True));status('seed_training_complete',seed=seed)
 b.save('complete');sealed(BASE/'training_complete.json',dict(at=now(),seeds=c['seeds']))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()

"""Fair ablations using the same P7 training samples, init, budget and epoch selection."""
import os,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_common_v32 import *
def run():
 import numpy as np,torch
 from scripts.p7_fit_v31 import model
 from src.p8_math_v32 import REMOVED,logits
 c=verify();b=Budget(c,'fitting');torch.set_num_threads(2);torch.use_deterministic_algorithms(True);data={s:np.load(artifact(OLD/'predictions'/f'{s}.npz')) for s in ['train','validation']};mean=data['train']['mean'];scale=data['train']['scale']
 for name in ['no_quality','no_instance']:
  d=BASE/'models'/name
  if (d/'fit.json').exists():load(d/'fit.json');artifact(d/'weights.npz');continue
  torch.manual_seed(c['seed']);net=model();opt=torch.optim.Adam(net.parameters(),lr=.001);x={s:((a['features']-mean)/scale).copy() for s,a in data.items()}
  for s in x:x[s][:,REMOVED[name]]=0
  xt={s:torch.from_numpy(v) for s,v in x.items()};yt={s:torch.from_numpy(data[s]['labels']) for s in data};best=float('inf');chosen=0;stale=0;history=[]
  for epoch in range(1,31):
   net.train();order=np.random.RandomState(c['seed']+epoch).permutation(len(yt['train']))
   for start in range(0,len(order),256):
    b.guard();ix=order[start:start+256]
    def update():
     opt.zero_grad(set_to_none=True);z=net(xt['train'][ix]).reshape(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,yt['train'][ix]);need(torch.isfinite(loss).item(),'Nonfinite loss');loss.backward();opt.step();return float(loss.detach())
    invoke('risk_update',update,model=name,epoch=epoch,batch=start//256)
   net.eval()
   def forward():
    with torch.inference_mode():return net(xt['validation']).reshape(-1).numpy().copy()
   z,cid=invoke('risk_forward',forward,model=name,epoch=epoch);nll=float(np.mean(np.logaddexp(0,z.astype(float))-data['validation']['labels']*z));p=ROOT/'experiments/p8_ablation_v32'/name/f'epoch_{epoch:02d}.pt';p.parent.mkdir(parents=True,exist_ok=True);need(not p.exists(),'Unclosed fit checkpoint retained');torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,mean=mean,scale=scale),p);sealed(d/'epochs'/f'{epoch:02d}.json',dict(epoch=epoch,nll=nll,checkpoint=stat(p),validation_call=cid));history.append(dict(epoch=epoch,nll=nll));stale=0 if nll<best else stale+1
   if nll<best:best=nll;chosen=epoch
   status('ablation_training',model=name,epoch=epoch,best_epoch=chosen)
   if stale>=5:break
  p=ROOT/'experiments/p8_ablation_v32'/name/f'epoch_{chosen:02d}.pt';net.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);net.eval();weights={k:v.detach().numpy().copy() for k,v in net.state_dict().items()};payload(d/'weights.npz',**weights)
  for s in data:
   def audit():
    with torch.inference_mode():return net(xt[s]).reshape(-1).numpy().copy()
   actual,cid=invoke('risk_audit_forward',audit,model=name,split=s);expected,_=invoke('risk_numpy_audit',lambda:logits(data[s]['features'],weights,mean,scale,REMOVED[name]),model=name,split=s);need(np.allclose(actual,expected,rtol=2e-5,atol=2e-5),'Numpy model export differs')
  sealed(d/'fit.json',dict(selected_epoch=chosen,checkpoint=stat(p),history=history,removed=REMOVED[name],train_samples=len(yt['train']),calibration_used=False,validation_used_for_epoch_selection=True))
 b.save('complete');sealed(BASE/'fitting_complete.json',dict(at=now(),models=2,full_model_retrained=False))
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()

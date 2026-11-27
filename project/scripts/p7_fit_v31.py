"""Train-only risk fitting, separate calibration scenes, predeclared validation."""
import os,sys,json,copy,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p7_common_v31 import *

def dataset(c):
 import numpy as np
 result={}
 for split in ['train','calibration','validation']:
  xs=[];ys=[];spans=[];start=0
  for case in [r for r in cases(c) if r['split']==split]:
   d=folder(case);f=load_sealed(d/'features.json');need(stat(d/'features.npz')==f['artifact'],'Features changed');a=np.load(d/'features.npz');labels=load_sealed(d/'labels.json');need(a['indices'].tolist()==labels['indices'],'Input/target correspondence differs');x=a['features'];y=np.array([r['positive'] for r in labels['records']],np.float32);need(len(x)==len(y),'Missing real labels');xs.append(x);ys.append(y);spans.append(dict(**case,start=start,end=start+len(y)));start+=len(y)
  result[split]=dict(x=np.concatenate(xs),y=np.concatenate(ys),spans=spans)
 return result

def model():
 import torch
 return torch.nn.Sequential(torch.nn.Linear(19,128),torch.nn.ReLU(),torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Linear(64,1))

def run():
 import numpy as np,torch
 from scipy.optimize import minimize_scalar
 from src.risk_metrics_v31 import probability_metrics,ranking
 c=verify();b=Budget(c,'fitting');load_sealed(BASE/'scoring_complete.json');data=dataset(c);counts={s:dict(n=len(d['y']),positive=int(d['y'].sum()),negative=int((1-d['y']).sum())) for s,d in data.items()}
 adequate=all(d['positive']>=30 and d['negative']>=30 for d in counts.values());sealed(BASE/'class_support.json',dict(counts=counts,adequate=adequate))
 if not adequate:sealed(BASE/'fit_result.json',dict(at=now(),fitted=False,scientific_usable=False,reason='Insufficient positive/negative support; no invented calibration',counts=counts));b.save('insufficient_support');return
 if (BASE/'fit_result.json').exists():load_sealed(BASE/'fit_result.json');return
 torch.set_num_threads(2);torch.manual_seed(c['seed']);torch.use_deterministic_algorithms(True);net=model();opt=torch.optim.Adam(net.parameters(),lr=.001);mean=data['train']['x'].mean(0,dtype=np.float64).astype(np.float32);scale=np.maximum(data['train']['x'].std(0,dtype=np.float64),1e-6).astype(np.float32)
 inputs={s:torch.from_numpy((d['x']-mean)/scale) for s,d in data.items()};targets={s:torch.from_numpy(d['y']) for s,d in data.items()};best=float('inf');best_epoch=None;stale=0;history=[];checkpoints=ROOT/'experiments/risk_v31';checkpoints.mkdir(parents=True,exist_ok=True)
 for epoch in range(1,31):
  net.train();order=np.random.RandomState(c['seed']+epoch).permutation(len(targets['train']));losses=[]
  for start in range(0,len(order),256):
   b.guard();ix=order[start:start+256]
   def update():
    opt.zero_grad(set_to_none=True);z=net(inputs['train'][ix]).reshape(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,targets['train'][ix]);need(torch.isfinite(loss).item(),'Nonfinite risk loss');loss.backward();opt.step();return float(loss.detach())
   loss,_=invoke('risk_update',update,epoch=epoch,batch=start//256,samples=len(ix));losses.append(loss)
  net.eval()
  def vf():
   with torch.inference_mode():return net(inputs['validation']).reshape(-1).numpy().copy()
  val,_=invoke('risk_forward',vf,phase='epoch_validation',epoch=epoch);nll=probability_metrics(data['validation']['y'],val)['nll'];path=checkpoints/f'epoch_{epoch:02d}.pt';need(not path.exists(),'Existing unclosed fit checkpoint; preserve before recovery');torch.save(dict(model=net.state_dict(),mean=mean,scale=scale,epoch=epoch,optimizer=opt.state_dict()),path);sealed(BASE/'epochs'/f'{epoch:02d}.json',dict(epoch=epoch,training_loss=float(np.mean(losses)),validation_nll=nll,checkpoint=stat(path)))
  history.append(dict(epoch=epoch,validation_nll=nll));stale=0 if nll<best else stale+1
  if nll<best:best=nll;best_epoch=epoch
  status('risk_training',epoch=epoch,best_epoch=best_epoch)
  if stale>=5:break
 selected=checkpoints/f'epoch_{best_epoch:02d}.pt';net.load_state_dict(torch.load(selected,map_location='cpu',weights_only=False)['model']);net.eval();logits={};callids={}
 weightfile=BASE/'selected_weights.npz';np.savez_compressed(weightfile,**{k:v.detach().numpy() for k,v in net.state_dict().items()});sealed(BASE/'selected_weights.json',dict(artifact=stat(weightfile),selected_checkpoint=stat(selected)))
 for s in data:
  def ff():
   with torch.inference_mode():return net(inputs[s]).reshape(-1).numpy().copy()
  logits[s],callids[s]=invoke('risk_forward',ff,phase='selected_model',split=s)
 # The calibration split was excluded from every optimizer update and epoch choice.
 def objective(logT):
  b.guard();value,_=invoke('temperature_objective',lambda:probability_metrics(data['calibration']['y'],logits['calibration']/np.exp(logT))['nll'],log_temperature=float(logT));return value
 fitted=minimize_scalar(objective,bounds=(np.log(.05),np.log(20.)),method='bounded',options=dict(maxiter=100,xatol=1e-6));need(fitted.success,'Temperature optimizer did not converge');T=float(np.exp(fitted.x));penalty=float(np.ptp(data['train']['x'][:,0])+1.);reports={}
 for s,d in data.items():
  x,y=d['x'],d['y'];z=logits[s];scores=dict(raw=x[:,0].astype(np.float64),collision_penalty=x[:,0]-penalty*((x[:,12]>0)|(x[:,13]>0)),risk=z,calibrated=z/T)
  reports[s]=dict(ranking={name:ranking(y,v,d['spans']) for name,v in scores.items()},uncalibrated=probability_metrics(y,z),calibrated=probability_metrics(y,z/T))
  p=BASE/'predictions'/f'{s}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,features=x,labels=y,logits=z,mean=mean,scale=scale);sealed(p.with_suffix('.json'),dict(artifact=stat(p),spans=d['spans'],call_id=callids[s]))
 val=reports['validation'];simple=max(val['ranking'][n]['p5'] for n in ['raw','collision_penalty']);clean=max(val['ranking'][n]['clean_p5'] for n in ['raw','collision_penalty']);gates=dict(ranking_gain=val['ranking']['risk']['p5']-simple>=.01,clean_noninferiority=val['ranking']['risk']['clean_p5']>=clean-.01,calibration_nll=val['calibrated']['nll']<=val['uncalibrated']['nll']+.001,calibration_ece=val['calibrated']['ece']<=.1)
 sealed(BASE/'fit_result.json',dict(at=now(),fitted=True,selected_epoch=best_epoch,selected_checkpoint=stat(selected),temperature=T,temperature_iterations=int(fitted.nfev),simple_penalty_train_only=penalty,counts=counts,history=history,reports=reports,gates=gates,scientific_usable=all(gates.values()),probability_target='offline official mu<=0.4 and noncollision, not physical success'));b.save('complete')

if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()

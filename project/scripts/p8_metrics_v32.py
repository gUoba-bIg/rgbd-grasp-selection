"""Calibrate on calibration scenes only; explicit AP and scientific gates."""
import os,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p8_common_v32 import *
def datasets(c):
 import numpy as np
 result={}
 for split in ['calibration','validation']:
  yy=[];zz=[];spans=[];start=0
  for r in [v for v in cases(c) if v['split']==split]:
   d=folder(r)/'foreground';sel=load(d/'selection.json');labels={x['index']:x for x in load(d/'labels.json')['records']};f=np.load(artifact(d/'features.npz'));ix=sel['pool'];y=[int(labels[i]['positive']) for i in ix];yy.extend(y);zz.extend(f['D4'][ix].tolist());spans.append(dict(**r,start=start,end=start+len(ix)));start+=len(ix);f.close()
  result[split]=dict(y=np.asarray(yy,float),z=np.asarray(zz,float),spans=spans)
 return result
def calculate(c,T,adequate):
 import numpy as np
 from src.p8_math_v32 import probability,grasp_ap
 data=datasets(c);prob={sp:dict(uncalibrated=probability(d['y'],d['z']),calibrated=probability(d['y'],d['z']/T) if T is not None else None) for sp,d in data.items()};rows=[]
 for r in [x for x in cases(c) if x['split']=='validation']:
  for source in ['whole','foreground']:
   d=folder(r)/source;sel=load(d/'selection.json');labels={x['index']:x for x in load(d/'labels.json')['records']}
   for method,v in sel['methods'].items():
    records=[labels[ix] for ix in v['indices']];row=dict(**r,method=method,**grasp_ap([x['score'] for x in records],[x['collision'] for x in records]));rows.append(row)
 agg={m:{cond:float(np.mean([r['AP'] for r in rows if r['method']==m and r['condition']==cond])) for cond in c['conditions']} for m in c['groups_definition']};degraded={m:float(np.mean([v for k,v in d.items() if k!='clean'])) for m,d in agg.items()};best=max(['D0','D1','D2'],key=lambda m:degraded[m]);gain=degraded['D5']-degraded[best];clean_loss=max(agg[m]['clean'] for m in ['D0','D1','D2'])-agg['D5']['clean'];mask=load(BASE/'mask_summary.json');v=prob['validation'];ece_rel=1-v['calibrated']['ece15']/v['uncalibrated']['ece15'] if adequate and v['uncalibrated']['ece15']>0 else None;gates=dict(degraded_grasp_AP_gain=gain>=.01,clean_grasp_AP_preserved=clean_loss<=.01,clean_mask_AP_preserved=mask['clean_difference']==0,calibration_support=adequate,calibration_ECE_reduction=ece_rel is not None and ece_rel>=.10,calibration_NLL_noninferior=adequate and v['calibrated']['nll']<=v['uncalibrated']['nll']+.001)
 scene_gains={str(s):float(np.mean([r['AP'] for r in rows if r['scene']==s and r['condition']!='clean' and r['method']=='D5'])-np.mean([r['AP'] for r in rows if r['scene']==s and r['condition']!='clean' and r['method']==best])) for s in c['groups']['validation']}
 return dict(rows=rows,condition_AP=agg,degraded_mean_AP=degraded,best_preregistered_simple=best,mean_degraded_AP_gain=gain,clean_grasp_AP_loss=clean_loss,clean_mask_AP_difference=0.,probability=prob,temperature=T,relative_ECE15_reduction=ece_rel,gates=gates,scientific_usable=all(gates.values()),paired_scene_gains=scene_gains,scope=c['scope'],fixed_pool_risk_gain=degraded['D6']-degraded['D0'],foreground_chain_gain=degraded['D5']-degraded['D6'],ablation_gains={m:degraded['D4']-degraded[m] for m in ['D2','D3','D7']},temperature_ranking_effect=0.,segmentation_changed_between_groups=False)
def run():
 import numpy as np
 from scipy.optimize import minimize_scalar
 from src.p8_math_v32 import probability
 c=verify();b=Budget(c,'metrics');load(BASE/'scoring_complete.json');data=datasets(c);support={s:dict(n=len(d['y']),positive=int(d['y'].sum()),negative=int(len(d['y'])-d['y'].sum())) for s,d in data.items()};adequate=all(x['positive']>=30 and x['negative']>=30 for x in support.values());cp=BASE/'calibration.json'
 if not cp.exists():
  T=None;nfev=0
  if adequate:
   def objective(logT):
    b.guard();v,_=invoke('temperature_objective',lambda:probability(data['calibration']['y'],data['calibration']['z']/np.exp(logT))['nll'],logT=float(logT));return v
   result=minimize_scalar(objective,bounds=(np.log(.05),np.log(20.)),method='bounded',options=dict(maxiter=100,xatol=1e-6));need(result.success,'Temperature fitting failed');T=float(np.exp(result.x));nfev=int(result.nfev)
  sealed(cp,dict(temperature=T,adequate=adequate,support=support,nfev=nfev,validation_labels_used_to_fit=False))
 cal=load(cp);result=calculate(c,cal['temperature'],cal['adequate']);sealed(BASE/'metrics.json',result);b.save('complete');status('metrics_complete',scientific_usable=result['scientific_usable'])
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()

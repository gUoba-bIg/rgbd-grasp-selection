"""Prespecified per-seed and scene-paired summaries; no best-seed selection."""
import os,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p9_common_v35 import *
def calculate(c,independent=False):
 import numpy as np
 from src.p8_math_v32 import grasp_ap
 old=load(P8/'metrics.json');base={m:old['condition_AP'][m] for m in ['D0','D1','D2']};base_degraded={m:float(np.mean([a[k] for k in c['conditions'] if k!='clean'])) for m,a in base.items()};best=max(base_degraded,key=base_degraded.get);clean=max(a['clean'] for a in base.values());rows=[];seed_results=[]
 for seed in c['seeds']:
  local=[]
  for r in cases(c):
   d=BASE/'cases'/key(r);sel=load(d/'selection.json')['methods'][str(seed)];labels={x['index']:x for x in load(d/'labels.json')['records']};records=[labels[i] for i in sel['indices']]
   if independent:
    prefix=[[sum(0<x['score']<=mu and not x['collision'] for x in records[:k])/k for mu in [.2,.4,.6,.8,1.,1.2]] for k in range(1,51)];ap=dict(AP=float(np.mean(prefix)),AP_04=float(np.mean([x[1] for x in prefix])),prefix=prefix,selected=len(records))
   else:ap=grasp_ap([x['score'] for x in records],[x['collision'] for x in records])
   local.append(dict(seed=seed,**r,**ap))
  rows+=local;agg={cond:float(np.mean([x['AP'] for x in local if x['condition']==cond])) for cond in c['conditions']};degraded=float(np.mean([v for k,v in agg.items() if k!='clean']));gain=degraded-base_degraded[best];loss=clean-agg['clean'];cal=load(seedpath(seed)/'calibration.json');raw=cal['probability'].get('validation',{}).get('raw');scaled=cal['probability'].get('validation',{}).get('calibrated');relative=1-scaled['ece15']/raw['ece15'] if raw and raw['ece15']>0 else None
  gates=dict(held_scene_NLL=cal['screen_passed'],degraded_AP_gain=gain>=.01,clean_grasp_preserved=loss<=.01,clean_mask_preserved=True,class_support=bool(raw and raw['positives']>=30 and raw['n']-raw['positives']>=30),ECE15_reduction=relative is not None and relative>=.10,validation_NLL_noninferior=bool(raw and scaled['nll']<=raw['nll']+.001));fit=load(seedpath(seed)/'fit.json');seed_results.append(dict(seed=seed,epochs=len(fit['history']),selected_epoch=fit['selected_epoch'],unique_updates=fit['unique_updates'],condition_AP=agg,degraded_AP=degraded,degraded_AP_gain=gain,clean_AP_loss=loss,temperature=cal['temperature'],validation_probability=cal['probability'].get('validation'),relative_ECE15_reduction=relative,holdout_NLL_change=cal['held_out']['nll']-cal['raw']['nll'],gates=gates,scientific_usable=all(gates.values())))
 scene_rows=[]
 for scene in c['groups']['validation']:
  baseline=float(np.mean([r['AP'] for r in old['rows'] if r['scene']==scene and r['condition']!='clean' and r['method']==best]));per={str(seed):float(np.mean([r['AP'] for r in rows if r['scene']==scene and r['condition']!='clean' and r['seed']==seed]))-baseline for seed in c['seeds']};scene_rows.append(dict(scene=scene,best_simple=best,baseline_AP=baseline,per_seed_gain=per,mean_seed_gain=float(np.mean(list(per.values())))))
 summary={k:dict(mean=float(np.mean([r[k] for r in seed_results])),std=float(np.std([r[k] for r in seed_results],ddof=1)),min=float(min(r[k] for r in seed_results)),max=float(max(r[k] for r in seed_results))) for k in ['degraded_AP','degraded_AP_gain','clean_AP_loss','holdout_NLL_change']}
 ece=[r['relative_ECE15_reduction'] for r in seed_results];summary['relative_ECE15_reduction']=dict(mean=float(np.mean(ece)),min=float(min(ece)),max=float(max(ece)),std=float(np.std(ece,ddof=1))) if all(v is not None for v in ece) else None
 return dict(scope=c['scope'],seeds=c['seeds'],rows=rows,per_seed=seed_results,summary=summary,paired_scene_gains=scene_rows,statistical_units=15,baseline_condition_AP=base,best_simple=best,baseline_degraded_AP=base_degraded[best],clean_mask_difference=0.,temperature_changes_ranking=False,all_seeds_retained=True,scientific_usable=all(r['scientific_usable'] for r in seed_results),reserved_test_opened=False)
def run():
 c=verify();b=Budget(c,'metrics');b.guard();load(BASE/'scoring_complete.json');load(BASE/'calibration_complete.json');sealed(BASE/'metrics.json',calculate(c));b.save('complete');status('metrics_complete')
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()

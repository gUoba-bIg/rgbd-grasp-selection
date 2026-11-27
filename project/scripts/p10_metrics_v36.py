"""Locked D0 contrast, all seeds and held-out groups; calibration descriptive."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p10_common_v36 import *
def calculate(c,independent=False):
 import numpy as np
 from src.p8_math_v32 import grasp_ap,probability
 from scripts.p8_review_v32 import independent_probability
 rows=[];pools={};methods=['D0']+[str(s) for s in c['seeds']]
 for r in cases(c):
  d=folder(r);sel=load(d/'selection.json');labels={x['index']:x for x in load(d/'labels.json')['records']}
  for m in methods:
   records=[labels[i] for i in sel['methods'][m]['indices']]
   if independent:
    prefix=[[sum(0<t['score']<=mu and not t['collision'] for t in records[:k])/k for mu in [.2,.4,.6,.8,1.,1.2]] for k in range(1,51)];v=dict(AP=float(np.mean(prefix)),AP_04=float(np.mean([p[1] for p in prefix])),selected=len(records))
   else:v=grasp_ap([x['score'] for x in records],[x['collision'] for x in records]);v.pop('prefix')
   rows.append(dict(**r,method=m,**v))
  with np.load(artifact(d/'features.npz')) as f:
   for seed in c['seeds']:
    k=(r['split'],str(seed));pools.setdefault(k,dict(y=[],z=[]));pools[k]['y'].extend(int(labels[i]['positive']) for i in sel['pool']);pools[k]['z'].extend(float(f[str(seed)][i]) for i in sel['pool'])
 table={g:{m:{cond:float(np.mean([r['AP'] for r in rows if r['split']==g and r['method']==m and r['condition']==cond])) for cond in c['conditions']} for m in methods} for g in c['groups']}
 core=c['primary_degraded'];extra=c['extra_conditions'];per_seed=[];paired=[]
 for seed in c['seeds']:
  m=str(seed);group_results={}
  for g in c['groups']:
   baseline=table[g]['D0'];a=table[g][m];group_results[g]=dict(core_AP=float(np.mean([a[k] for k in core])),core_AP_gain=float(np.mean([a[k]-baseline[k] for k in core])),clean_AP=a['clean'],clean_AP_loss=baseline['clean']-a['clean'],extra_AP_gain=float(np.mean([a[k]-baseline[k] for k in extra])))
  core_gain=float(np.mean([v['core_AP_gain'] for v in group_results.values()]));loss=float(np.mean([v['clean_AP_loss'] for v in group_results.values()]));extra_gain=float(np.mean([v['extra_AP_gain'] for v in group_results.values()]));gates=dict(core_AP_gain=core_gain>=.01,clean_AP_preserved=loss<=.01,worst_group_core_noninferior=min(v['core_AP_gain'] for v in group_results.values())>=0,extra_conditions_mean_noninferior=extra_gain>=0)
  per_seed.append(dict(seed=seed,groups=group_results,core_AP_gain=core_gain,clean_AP_loss=loss,extra_AP_gain=extra_gain,gates=gates,ranking_usable=all(gates.values())))
 for g,scenes in c['groups'].items():
  for scene in scenes:
   local=[r for r in rows if r['scene']==scene];base=float(np.mean([r['AP'] for r in local if r['method']=='D0' and r['condition'] in core]));gains={str(s):float(np.mean([r['AP'] for r in local if r['method']==str(s) and r['condition'] in core]))-base for s in c['seeds']};paired.append(dict(group=g,scene=scene,baseline_core_AP=base,per_seed_core_gains=gains,mean_seed_gain=float(np.mean(list(gains.values())))))
 probability_results={}
 for (g,s),data in pools.items():
  y=np.asarray(data['y']);z=np.asarray(data['z']);T=c['temperatures'][s];fn=independent_probability if independent else probability;probability_results[g+'_'+s]=dict(raw=fn(y,z),frozen_temperature=fn(y,z/T),T=T,descriptive_only=True)
 return dict(scope=c['scope'],methods=methods,groups=c['groups'],conditions=c['conditions'],rows=rows,condition_AP=table,per_seed=per_seed,paired_scenes=paired,statistical_units=90,probability=probability_results,ranking_scientific_usable=all(v['ranking_usable'] for v in per_seed),calibration_improvement_claim=False,P9_scientific_failure_preserved=True,all_seeds_retained=True,training_or_tuning_on_test=False)
def run():
 c=verify();b=Budget(c,'metrics');b.guard();load(BASE/'scoring_complete.json');sealed(BASE/'metrics.json',calculate(c));b.save('complete');status('metrics_complete')
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()

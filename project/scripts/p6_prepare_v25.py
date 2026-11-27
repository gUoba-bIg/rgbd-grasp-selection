"""Freeze the one-shot repair before any repair data inference or fitting."""
import sys,ast
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v25 import *
from scripts.pipeline import project_lock
import numpy as np
from src import p6_error_mixture_v25 as model

def run():
 need(not CONFIG.exists(),'Already frozen')
 old=read(ROOT/'configs/p6_image_geometry_v24.json');proof=read(ROOT/'logs/P6/image_geometry_v24/final_readonly.json')
 for p,v in proof['files'].items():need(stat(ROOT/p)==v,'Old P6 changed '+p)
 scenes=sorted({r['scene'] for r in old['train_records']});fs=['0000','0064','0128','0192']
 c={k:old[k] for k in ['data_root','seed','quality_screen_epoch','runtime','decoder','input_size','split_path','quality_checkpoint','fixed_segmentation_checkpoint','development_records','core_conditions']}
 c.update(version=25,created_at=now(),fit_scenes=scenes[:16],holdout_scenes=scenes[16:],fit_records=[r for r in old['train_records'] if r['scene'] in scenes[:16] and r['frame'] in fs],holdout_records=[r for r in old['train_records'] if r['scene'] in scenes[16:] and r['frame'] in fs],limits=dict(quality_forward=404,segmentation_forward=82,corruption_replay=2500,error_model_fit=1,mixture_prediction=1760,seconds=1800,bytes=int(1.5*1024**3)),protected=dict(proof['files']))
 need(len(c['fit_records'])==64 and len(c['holdout_records'])==16,'Scope mismatch');need(not(set(scenes)&{r['scene'] for r in c['development_records']}),'Scene leakage')
 c['previous_P6_bytes']=sum(p.stat().st_size for folder in [ROOT/'results/P6',ROOT/'logs/P6'] for p in folder.rglob('*') if p.is_file() and not p.is_relative_to(BASE) and not p.is_relative_to(LOG))
 checks={};one=dict(weights=[.2,.3,.5],means=[-.01,0,.02],scales=[.001,.005,.01]);fake=dict(models={str(k):one for k in range(72)})
 pr=model.predict(fake,np.array([0,71]),np.array([-.4,.4]));checks['cdf_extremes']=bool(pr['pit'][0]<1e-12 and pr['pit'][1]>1-1e-12)
 expected_mean=.2*(-.01)+.5*.02;expected_var=sum(w*(s*s+(m-expected_mean)**2) for w,m,s in zip(one['weights'],one['means'],one['scales']))
 checks['mixture_total_variance']=bool(np.allclose(pr['mean'],expected_mean) and np.allclose(pr['variance'],expected_var))
 z=np.full((11,13),.5,np.float32);valid=np.ones_like(z,bool);sg=np.full_like(z,.004);cell,noise=model.observable_cells(z,valid,sg);checks['constant_observation']=bool(noise==0 and np.all(cell==4))
 y=np.tile(np.array([-.003,0,.004]),300);fit=model.fit_one(y);checks['fit_finite_normalized']=bool(abs(sum(fit['weights'])-1)<1e-12 and np.isfinite(fit['scales']).all() and min(fit['scales'])>=.0001)
 from scripts.p6_review_v25 import cells_reference,density_reference
 rng=np.random.RandomState(17);z=(.5+rng.normal(0,.01,(21,27))).astype(np.float32);z[:,14:]+=.1;valid=rng.uniform(size=z.shape)>.1;z[~valid]=0;sg=np.full_like(z,.003);ca,na=model.observable_cells(z,valid,sg);cb,nb=cells_reference(z,valid,sg);checks['independent_observable_cells']=bool(np.array_equal(ca,cb) and na==nb)
 a=dict(cell=np.array([0,71]),observed=np.array([.4,.6]),reference=np.array([.39,.61]),rays=np.array([[.2,.1,1.],[-.4,.2,1.]]),main=np.ones(2,bool),boundary=np.ones(2,bool));pp=model.predict(fake,a['cell'],a['observed']-a['reference']);rr,mm=density_reference(fake,a);checks['independent_density_moments']=bool(all(np.allclose(pp[k],rr[k],rtol=1e-10,atol=1e-12) for k in pp));checks['independent_scores']=all(np.allclose(v,mm[g][k]) for g,d in model.metrics(a,pp).items() for k,v in d.items())
 for p in list((ROOT/'scripts').glob('p6_*_v25.py'))+[ROOT/'src/p6_error_mixture_v25.py']:ast.parse(p.read_text(encoding='utf-8'));c['protected'][p.relative_to(ROOT).as_posix()]=stat(p)
 checks['no_reference_in_feature_signature']=list(__import__('inspect').signature(model.observable_cells).parameters)==['z','valid','sigma']
 need(all(checks.values()),'Component check failed '+str(checks));once(LOG/'component_tests.json',dict(passed=True,checks=checks,no_research_calls=True))
 for p in ['docs/P6误差模型单次修复协议_v25.txt','logs/P6/error_mixture_v25/component_tests.json','scripts/extract_mask_scores_v17.py','scripts/accept_mask_ap_v17.py','logs/P6/image_geometry_v24/root_final_review.json']:
  c['protected'][p]=stat(ROOT/p)
 once(CONFIG,c);once(LOG/'freeze.json',dict(sha256=sha(CONFIG),protected_files=len(c['protected']),created_at=now()));status('frozen',fit_cases=320,holdout_cases=80,full_regression_cases_if_gate_passes=800)
 hand=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(hand,LOG/'handoff_before.txt');hand.write_text('最新：'+now()+' 用户授权P6单次误差模型修复v25，已冻结。先320拟合项+80留出关口，全部通过才复用缓存做800项开发回归；不通过即真实汇总停止，不调参追加。旧v24不变；不进入P7，不恢复自动任务。全部41%、实验51.25%，修复不重复计分。入口scripts/p6_deliver_v25.py，状态runtime/pipeline/p6_part_v25.json。\n\n历史：\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8')
 ctrl=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(ctrl,LOG/'supervision_before.json');row=read(ctrl);row.update(recorded_at=now(),current_authorized_part='P6单次误差模型修复v25',current_execution_state='P6_v25_frozen',current_runner='scripts/p6_deliver_v25.py',current_execution_evidence=CONFIG.relative_to(ROOT).as_posix(),waiting_for_user_next_part_confirmation=False,active_internal_repair='v25',summary_report=None);atomic(ctrl,row)
 print('P6 v25 frozen',flush=True)
if __name__=='__main__':
 with project_lock(ROOT):run()

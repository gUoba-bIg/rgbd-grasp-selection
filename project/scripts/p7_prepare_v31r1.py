"""Actual P3 scientific gate and measured feasibility precede any P7 freeze."""
import os,sys,json,ast,shutil
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p7_common_v31 import *

def run():
 if CONFIG.exists():verify();print('Existing P7 frozen protocol verified');return
 import numpy as np
 p3=read(ROOT/'results/P3/candidate_v30/closeout.json');need(p3['engineering_passed'] and p3['candidate_usable_for_P7'],'P3 actual candidate gate failed')
 for name in ['final_readonly','final_cold']:
  r=read(ROOT/f'logs/P3/candidate_v30/{name}.json');need(r['engineering_passed'] and r['scientific_usable'],'P3 final review missing')
 for p,s in r['files'].items():need(stat(ROOT/p)==s,'P3 final artifact changed')
 need(read(ROOT/'logs/P3/candidate_v30/new_data_final_integrity.json')['all_passed'],'Final data integrity missing')
 checks=read(ROOT/'logs/P7/preflight_v31/checks.json');need(checks['passed'],'P7 preflight checks missing')
 for p,s in checks['sources'].items():need(stat(ROOT/p)==s,'P7 preflight source changed '+p)
 calls=[json.loads(s) for base in ['logs/P2P3/candidate_v27','logs/P3/candidate_v30','logs/P3/repair_v29/filter_trial'] for s in (ROOT/base/'calls.jsonl').read_text(encoding='utf-8').splitlines()];times=np.array([r['seconds'] for r in calls if r['kind']=='score_returned']);need(len(times)>0,'No actual scoring time observation')
 projected=float(max(times.mean()*2,np.quantile(times,.9))*10000+1800);LOG.mkdir(parents=True,exist_ok=True);feasibility=dict(at=now(),observed_calls=len(times),mean_seconds=float(times.mean()),p90_seconds=float(np.quantile(times,.9)),projection_seconds=projected,limit_seconds=21600,formula='max(2*mean,p90)*10000+1800; empirical feasibility estimate, not guarantee',within_budget=projected<=21600)
 if (LOG/'resource_feasibility.json').exists():need({k:v for k,v in read(LOG/'resource_feasibility.json').items() if k!='at'}=={k:v for k,v in feasibility.items() if k!='at'},'Prior measured feasibility changed')
 else:once(LOG/'resource_feasibility.json',feasibility)
 need(projected<=21600,'P7 measured resource feasibility outside frozen candidate budget')
 from src.p7_split_contract_v31r1 import fixed_development_groups
 groups=fixed_development_groups(read(ROOT/'data/manifests/scene_splits.json')['groups']);old=read(ROOT/'configs/p6_image_geometry_v24.json');c=dict(version=31,at=now(),seed=20261021,data_root='data/raw/prerequisites_v26',groups=groups,conditions=['clean','noise5','missing15','shift3','joint'],quality_screen_epoch=10000,decoder=old['decoder'],candidate_checkpoint='experiments/candidate_v30/epoch_18_final.pt',quality_checkpoint=old['quality_checkpoint'],segmentation_checkpoint=old['fixed_segmentation_checkpoint'],phase_seconds=dict(generation=7200,scoring=21600,fitting=1800,review=3600),limits=dict(gpu_bytes=int(13.5*1024**3),rss_bytes=24*1024**3,output_bytes=12*1024**3,disk_reserve_bytes=60*1024**3),call_limits=dict(grasp_forward=502,quality_forward=502,segmentation_forward=502,official_score=10000,upstream_oracle=3,risk_update=3000,risk_forward=33,temperature_objective=100,risk_audit_forward=3,corruption=500,corruption_audit=500),image_inputs={},protected={})
 need(len(groups['train'])==70 and len(groups['validation'])==len(groups['calibration'])==15 and len(set(sum(groups.values(),[])))==100,'P7 split invalid')
 receipts=[json.loads(s) for s in (ROOT/'logs/P2P3/prerequisites_v26/member_receipts.jsonl').read_text(encoding='utf-8').splitlines()]
 for r in receipts:
  if any(r['path'].endswith('/'+tail) for tail in ['rgb/0064.png','depth/0064.png','meta/0064.mat']):c['image_inputs'][r['path']]={k:r[k] for k in ['sha256','bytes','mtime_ns']}
 need(len(c['image_inputs'])==300,'P7 original frame coverage missing')
 files=list((ROOT/'scripts').glob('p7_*_v31.py'))+list((ROOT/'src').glob('risk_*_v31.py'))+[ROOT/'docs/P7固定候选风险与独立校准_v31_草案.txt',ROOT/'logs/P7/preflight_v31/checks.json',ROOT/'configs/candidate_v30.json',ROOT/'results/P3/candidate_v30/closeout.json',ROOT/'logs/P3/candidate_v30/final_readonly.json',ROOT/'logs/P3/candidate_v30/final_cold.json',ROOT/'logs/P3/candidate_v30/new_data_final_integrity.json',ROOT/'data/manifests/scene_splits.json',ROOT/'src/p5_quality_v23.py',ROOT/'src/graspnet_frames.py',ROOT/'src/segmentation.py',ROOT/'src/grasp_prerequisite_data_v27.py',ROOT/'scripts/p6_common_v24.py',ROOT/'scripts/p3_train_v27.py',ROOT/'scripts/p3_train_v27r1.py',ROOT/'scripts/verify_grasp_geometry_v10.py',ROOT/'scripts/extract_mask_scores_v17.py',ROOT/'scripts/pipeline.py']
 files += [ROOT/'scripts/p7_prepare_v31r1.py',ROOT/'src/p7_split_contract_v31r1.py',ROOT/'logs/P7/prepare_repair_v31r1/checks.json',ROOT/'docs/P7启动入口修复_v31r1.txt']
 files += [ROOT/'src/grasp_observed_filter_v29.py',ROOT/'logs/P7/preflight_v28/checks.json',ROOT/'logs/P7/preflight_v31/filter_pool.json',ROOT/'configs/p3_filter_trial_v29.json']
 files += [ROOT/c[k] for k in ['candidate_checkpoint','quality_checkpoint','segmentation_checkpoint']]
 for base in ['runtime/build/graspnet','runtime/build/graspnetAPI_eval_v7']:
  files += [p for p in (ROOT/base).rglob('*') if p.suffix in ['.py','.pyd','.cu','.cpp','.h']]
 for p in files:
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
  c['protected'][p.relative_to(ROOT).as_posix()]=stat(p)
 c['entrypoint_repair']='v31r1: select unchanged train/calibration/validation lists explicitly; reserved test groups excluded; preserve prior feasibility receipt'
 c['candidate_pool_filter']=dict(voxel_size=.01,collision_thresh=.01,approach_dist=.05,GT_used=False,source='Pinned official demo filter; same as accepted P3 repair')
 once(CONFIG,c);once(LOG/'freeze.json',dict(at=now(),sha256=sha(CONFIG)));status('frozen_not_yet_run',conditions_completed=0,total_conditions=500)
 hand=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(hand,LOG/'handoff_before_P7.txt');hand.write_text('最新：'+now()+' P2/P3实际验收、冷重入和数据不变性通过，全部47%、实验58.75%。P7已按固定70训练/15校准/15验证、500条件帧冻结；原始质量仅作描述特征，不采用失败P6概率。有限链路完成后停止，不启动P8/P9/周期任务。进度runtime/pipeline/risk_v31.json，先核真实进程和项目锁。\n\n历史：\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8');print('P7 frozen after actual P3 and measured resource gates',flush=True)

if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()

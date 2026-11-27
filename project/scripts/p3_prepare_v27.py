"""Freeze formal candidate model only after actual data and resource gates."""
import sys,ast,shutil
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
from scripts.p3_train_v27 import LOG,CONFIG

def run():
 need(not CONFIG.exists(),'Training already frozen; use existing entry')
 audit=read(ROOT/'results/P2P3/data_audit_v27r1/result.json');probe=read(ROOT/'logs/P2P3/training_probe_v27/result.json');need(read(ROOT/'results/P2P3/label_schema_v27a/result.json')['status']=='passed','Dense label audit missing')
 need(audit['status']=='passed' and audit['frames_verified']==25600,'P2 not accepted')
 need(probe['loader_equal_to_upstream'] and probe['known_training_updates']==2 and probe['unknown_training_calls']==0,'Probe incomplete')
 need(read(ROOT/'logs/P2P3/prefetch_check_v27/result.json')['bitwise_identical_to_serial'],'Prefetch equivalence not passed')
 need(probe['projected_10080_updates_seconds']<=42000,'12-hour training budget not supported by actual probe')
 groups=read(ROOT/'data/manifests/scene_splits.json')['groups']
 c=dict(version=27,at=now(),seed=20261021,data_root='data/raw/prerequisites_v26',train_scenes=groups['train'],excluded_scenes=groups['validation']+groups['calibration'],frames=list(range(0,256,16)),unique_training_updates=10080,pretrained_initialization=False,early_evaluation=[dict(scene=s,frame=0) for s in [3,4,5,8,9]],final_evaluation=[dict(scene=s,frame=f) for s in groups['validation'] for f in [0,128]],early_gate=dict(positive_scenes_min=3,positive_candidates_min=10),final_gate=dict(positive_scenes_min=10,positive_candidates_min=30),limits=dict(seconds=43200,scoring_seconds=14400,gpu_bytes=int(13.5*1024**3),rss_bytes=24*1024**3,actual_update_intents=11200,output_bytes=16*1024**3,disk_reserve_bytes=60*1024**3),protected={})
 files=list((ROOT/'scripts').glob('p3_*_v27.py'))+[ROOT/'src/grasp_prerequisite_data_v27.py',ROOT/'docs/P3独立候选基线训练与验收_v27.txt',ROOT/'configs/prerequisites_data_v26.json',ROOT/'logs/P2P3/prerequisites_v26/member_receipts.jsonl',ROOT/'results/P2P3/data_audit_v27r1/result.json',ROOT/'results/P2P3/label_schema_v27a/result.json',ROOT/'logs/P2P3/training_probe_v27/result.json',ROOT/'data/manifests/scene_splits.json']
 files += [ROOT/'src/grasp_prefetch_v27.py',ROOT/'logs/P2P3/prefetch_check_v27/result.json']
 files += [ROOT/'scripts/p6_common_v24.py',ROOT/'scripts/p23_common_v26.py',ROOT/'scripts/pipeline.py',ROOT/'scripts/generate_multiview_grasps_v14r1.py',ROOT/'scripts/verify_grasp_geometry_v10.py']
 files += [p for p in (ROOT/'runtime/build/graspnetAPI_eval_v7').rglob('*.py')]
 files += [p for p in (ROOT/'runtime/build/graspnet').rglob('*') if p.suffix in ['.py','.pyd','.cu','.cpp','.h']]
 for p in files:
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
  c['protected'][p.relative_to(ROOT).as_posix()]=stat(p)
 LOG.mkdir(parents=True,exist_ok=True);once(CONFIG,c);once(LOG/'freeze.json',dict(at=now(),sha256=sha(CONFIG)))
 once(LOG/'progress_after_P2.json',dict(before_all=41,before_experiment=51.25,earned_points=2,after_all=43,after_experiment=53.75,work_packages=['P2 final pre-test100scene data coverage closure2'],denominators=dict(all=100,experiment=80),not_scientific_success_percent=True,evidence='results/P2P3/data_audit_v27r1/result.json'))
 ctrl=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(ctrl,LOG/'supervision_before_candidate.json');row=read(ctrl);row.update(recorded_at=now(),current_execution_state='P3_candidate_training_frozen',all_tasks_percent_planning_estimate=43,experiment_percent_planning_estimate=53.75,current_progress_evidence='logs/P2P3/candidate_v27/progress_after_P2.json',remaining_research_updates_cap=11200,current_runner='scripts/p3_train_v27.py',current_execution_evidence=CONFIG.relative_to(ROOT).as_posix());atomic(ctrl,row)
 hand=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(hand,LOG/'handoff_before_candidate.txt');hand.write_text('最新：'+now()+' P2原100场景RealSense全256帧及标签前置审计通过，新增2点：全部43%、实验53.75%。P3独立70场景候选模型已冻结，先3轮和5场景有限继续关口，实际通过才完成18轮及30验证帧评价。P7须P2/P3实际准入后自动启动，结束后停止，不启P8/P9或周期任务；旧P6失败保留。\n\n历史：\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8')
 print('P3 protocol frozen after actual P2 and resource gates',flush=True)

if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()

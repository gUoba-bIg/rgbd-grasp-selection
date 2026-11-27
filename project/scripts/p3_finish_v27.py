"""Close the whole bounded P3 development baseline; preserve scientific failure."""
import sys,json,hashlib,shutil,concurrent.futures,time
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
from scripts.p3_train_v27 import CONFIG,LOG

def run():
 out=ROOT/'results/P2P3/candidate_v27/closeout.json'
 if out.exists():print(json.dumps(read(out)),flush=True);return
 c=read(CONFIG);review=read(LOG/'final_readonly.json');cold=read(LOG/'final_cold.json');need(review['engineering_passed'] and cold['engineering_passed'] and review['files']==cold['files'],'Final review/cold reentry incomplete')
 for p,s in {**c['protected'],**review['files']}.items():need(stat(ROOT/p)==s,'P3 final file changed '+p)
 from scripts.p23_common_v26 import verify
 verify()
 # Independently prove the newly acquired data, not merely its receipt file, unchanged.
 started=time.monotonic();receipts=[json.loads(s) for s in (ROOT/'logs/P2P3/prerequisites_v26/member_receipts.jsonl').read_text(encoding='utf-8').splitlines()]
 def check_data(r):
  need(time.monotonic()-started<7200,'Final data integrity time budget');p=ROOT/c['data_root']/r['path'];need(stat(p)=={k:r[k] for k in ['sha256','bytes','mtime_ns']},'P3 data changed '+r['path'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for _ in pool.map(check_data,receipts):pass
 once(LOG/'new_data_final_integrity.json',dict(at=now(),files=len(receipts),bytes=sum(r['bytes'] for r in receipts),seconds=time.monotonic()-started,all_passed=True))
 import torch
 from src.grasp_prerequisite_data_v27 import SOURCE
 from graspnet import GraspNet
 torch.manual_seed(c['seed']);model=GraspNet(input_feature_dim=0,num_view=300,num_angle=12,num_depth=4,cylinder_radius=.05,hmin=-.02,hmax_list=[.01,.02,.03,.04]);initial={k:hashlib.sha256(v.detach().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()};need(initial==read(LOG/'initial_random_model.json')['tensor_sha256'],'Independent from-scratch initialization check failed')
 calls=[json.loads(s) for s in (LOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()];counts={}
 for kind in ['update','generation','score','oracle']:
  a={r['call_id'] for r in calls if r['kind']==kind+'_intent'};b={r['call_id'] for r in calls if r['kind']==kind+'_returned'};need(b<=a,'Unexpected return');counts[kind]=dict(intents=len(a),known_returns=len(b),unknown=sorted(a-b))
 result=dict(at=now(),engineering_passed=True,candidate_usable_for_P7=review['scientific_usable'],formal_unique_training_updates=10080,compute=counts,scope='70train scenes x16views;30validation frames;15calibration scenes unused for model fitting/selection',P6_probability_adopted=False,official_test_AP=False,robot_success=False,all_tasks_percent=47,experiment_percent=58.75)
 once(out,result);once(LOG/'progress_after_P3.json',dict(before_all=43,before_experiment=53.75,earned_points=4,after_all=47,after_experiment=58.75,work_packages=['P3 bounded main baseline evaluation and closure4'],denominators=dict(all=100,experiment=80),not_scientific_success_percent=True,scientific_usable=review['scientific_usable']))
 ctrl=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(ctrl,LOG/'supervision_before_closeout.json');row=read(ctrl);row.update(recorded_at=now(),current_execution_state='P3_completed_P7_ready' if review['scientific_usable'] else 'P3_completed_candidate_not_usable',all_tasks_percent_planning_estimate=47,experiment_percent_planning_estimate=58.75,current_progress_evidence='logs/P2P3/candidate_v27/progress_after_P3.json',current_execution_evidence=out.relative_to(ROOT).as_posix());atomic(ctrl,row)
 report=ROOT/'docs/P2P3补齐与独立候选基线结果_v27.txt';report.write_text('P2/P3前置交付结果\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\nP2真实检查100场景25600帧及密集标签；相机表示精度合同修订保留旧失败。P3仅在70训练场景16固定视角训练，验证30帧，完整只读和实际冷重入通过。上述数值不是官方90测试场景AP，也不是机器人成功率。P6概率失败仍保留。\n',encoding='utf-8')
 print(json.dumps(result),flush=True)

if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()

"""Freeze corrected P3 remaining scope, within the original cumulative budget."""
import sys,json,ast,shutil
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,atomic,need,now
from scripts.p3_train_v30 import CONFIG,LOG,OLDLOG,OLDOUT

def run():
 need(not CONFIG.exists(),'v30 already frozen; use existing continuation')
 c=read(ROOT/'configs/candidate_v27.json');trial=read(ROOT/'results/P3/repair_v29/filter_trial/result.json');proof=read(ROOT/'logs/P3/repair_v29/filter_trial/readonly.json');cold=read(ROOT/'logs/P3/repair_v29/filter_trial/cold.json')
 need(trial['gate_passed'] and trial['positive_scenes_mu04']>=3 and trial['positive_candidates_mu04']>=10,'Original numerical gate not met by repair');need(proof['engineering_passed'] and cold['engineering_passed'] and proof['files']==cold['files'],'Full repaired review/cold missing')
 for p,s in {**c['protected'],**proof['files']}.items():need(stat(ROOT/p)==s,'Frozen prior evidence changed '+p)
 LOG.mkdir(parents=True,exist_ok=True);parent='experiments/candidate_v27/epoch_03_final.pt';need(stat(ROOT/parent)==read(OLDLOG/'epoch_03.json')['checkpoint'],'Parent model changed');need(read(OLDLOG/'latest_checkpoint.json')['path']==parent,'Latest old checkpoint is not the fixed epoch3 parent');once(LOG/'latest_checkpoint.json',read(OLDLOG/'latest_checkpoint.json'));once(LOG/'initial_random_model.json',read(OLDLOG/'initial_random_model.json'))
 elapsed=sum(read(p)['seconds'] for p in OLDLOG.glob('usage_*.json'));oldcalls=[json.loads(s) for s in (OLDLOG/'calls.jsonl').read_text(encoding='utf-8').splitlines()];attempts=sum(r['kind']=='update_intent' for r in oldcalls);need(elapsed<43200 and attempts+8400<=11200,'Original cumulative budget unavailable')
 c.update(version=30,at=now(),parent_protocol_sha256=sha(ROOT/'configs/candidate_v27.json'),parent_checkpoint=parent,repair='Add pinned observed collision postprocess; do not change training/data/scientific thresholds',old_gate_still_failed=True,continuation_epoch=4,final_epoch=18,remaining_unique_training_updates=8400,prior_actual_update_intents=attempts,prior_training_seconds=elapsed,gate_evidence='results/P3/repair_v29/filter_trial/result.json')
 files=list((ROOT/'scripts').glob('p3_*_v30.py'))+[ROOT/'src/grasp_observed_filter_v29.py',ROOT/'configs/p3_filter_trial_v29.json',ROOT/'results/P3/repair_v29/filter_trial/result.json',ROOT/'logs/P3/repair_v29/filter_trial/readonly.json',ROOT/'logs/P3/repair_v29/filter_trial/cold.json',ROOT/'logs/P3/repair_v29/filter_trial/calls.jsonl',ROOT/'docs/P3修复诊断与固定碰撞过滤_v29.txt',ROOT/'configs/candidate_v27.json',ROOT/parent,OLDLOG/'initial_random_model.json',OLDLOG/'calls.jsonl',OLDLOG/'early_readonly.json',OLDLOG/'early_cold.json',ROOT/'scripts/p3_train_v27r1.py',ROOT/'src/grasp_prefetch_v27.py']
 files += [p for p in OLDLOG.glob('usage_*.json')]+[p for p in (ROOT/'logs/P3/repair_v29/filter_trial').glob('usage_*.json')]
 for p in files:
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
  c['protected'][p.relative_to(ROOT).as_posix()]=stat(p)
 c['protected'].update(proof['files']);once(CONFIG,c);once(LOG/'freeze.json',dict(at=now(),sha256=sha(CONFIG)));atomic(ROOT/'runtime/pipeline/candidate_v30.json',dict(at=now(),phase='remaining_training_frozen',starting_epoch=4,final_epoch=18,remaining_updates=8400,all_tasks_percent=43,experiment_percent=53.75,P7_started=False))
 hand=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(hand,LOG/'handoff_before_v30.txt');hand.write_text('最新：'+now()+' P3缓存推理修复v29已完整通过：原5训练场景mu<=0.4阳性4→14，阳性场景3→5，旧4阳性全部保留；0新训练/神经前向，138新候选评分+68复用，完整只读/冷重入通过。原v27失败不改写。v30已冻结，从原第3轮完整状态继续第4至18轮（余8400更新），合并原1695更新及4758.406秒，原11200调用/10080唯一更新/12小时/4小时评分预算均不增加。最终原15验证场景30帧应用固定观测碰撞过滤，仍须10场景/30阳性及全复验、冷重入和数据不变性通过，才自动进入适配P7；P7完成停止，不进P8/P9或周期任务。全部43%、实验53.75%，入口scripts/p3_train_v30.py --through 18，状态runtime/pipeline/candidate_v30.json。先核真实进程及锁禁止重复。\n\n历史：\n'+hand.read_text(encoding='utf-8-sig'),encoding='utf-8')
 ctrl=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(ctrl,LOG/'supervision_before_v30.json');row=read(ctrl);row.update(recorded_at=now(),current_execution_state='P3_postprocess_repaired_remaining_training_frozen',current_runner='scripts/p3_train_v30.py',active_internal_repair='v30 pinned observed-collision filter; same data, thresholds and cumulative training budget',current_execution_evidence='runtime/pipeline/candidate_v30.json');atomic(ctrl,row);print('v30 frozen: resume original epoch3, remaining8400, cumulative budget unchanged',flush=True)

if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):run()

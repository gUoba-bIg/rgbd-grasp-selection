"""Publish whole-P4 development delivery only after every frozen gate passes."""
import os,sys,json,hashlib,copy,shutil,argparse,uuid
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.pipeline import project_lock
BASE=ROOT/'results/P4/full_baseline_v22r1'
LOG=ROOT/'logs/P4/full_baseline_v22r1/closeout'
CONFIG=ROOT/'configs/p4_closeout_v22r1.json'
def need(v,m):
    if not v:raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now().astimezone().isoformat()
def once(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():need(read(p)==x,'Immutable closeout differs: '+str(p));return
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def prepare():
    need(not CONFIG.exists(),'Closeout protocol exists')
    reference='logs/P4/baseline_grasp_link_v21/progress_after_trial.json';ledger=read(ROOT/reference)
    need(ledger['earned_points_current']==26 and ledger['total_weight']==100 and ledger['experiment_weight']==80,'Progress baseline differs')
    required=['training_and_export.json','rgb_full_development.json','rgbd_full_development.json','selection.json','evaluation/metrics.json','evaluation/readonly.json','evaluation/figures/index.json','representative_link/result.json','representative_link/readonly.json']
    c=dict(created_at=now(),source_sha256=sha(Path(__file__)),required=required,prior_progress=reference,prior_progress_sha256=sha(ROOT/reference),
        scope='P4 stable development baseline part: paired RGB/RGBD at640x480, fixed320train/160development, stable/valid segmentation, complete official visible-mask AP and error visualization, and representative20view candidate interface. Official full-data grasp performance is P3/P8-P10 and is not completed here.',
        gates='All frozen v22r1 scientific gates and complete independent segmentation/AP and representative-linkage acceptance. Source unchanged, exact matched-arm model selection and budget accounting. No threshold relaxation.',
        credit=dict(stable_baseline_closure=2,representative_linkage=1,maximum_all_tasks=29,maximum_experiments=36.25,no_new_total_weight=True),
        boundaries=['No independent held-out generalization because development scenes repeatedly inspected','Single paired seed and matched model-only warm start; not P9 formal multi-seed training','R18 dual-stream zero-instance negative result unchanged; ordinary-fusion replacement does not prove quality gating','Official grasp checkpoint exposure unknown; candidate linkage is engineering-only, all new geometry outcomes unmeasured','P4 part may finish without RGBD outperforming RGB; no automatic P5 or P9 run'],
        stop_policy='After whole-P4 delivery, stop and await user confirmation for next main part; automation remains absent; original pipeline/state.json is not manually edited.')
    once(CONFIG,c);once(LOG/'freeze.json',dict(protocol_sha256=sha(CONFIG),source_sha256=sha(Path(__file__))))
    print('Whole-P4 final acceptance and existing 3-point credit rule fixed before final results.')
def calls(path):
    events=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    intents={e.get('call_id',e.get('id')) for e in events if e['kind']=='forward_intent'};returns={e.get('call_id',e.get('id')) for e in events if e['kind']=='forward_returned'}
    need(returns<=intents,'Unbound return in model-call ledger')
    return dict(forward_intents=len(intents),known_forward_returns=len(returns),unknown_forward_outcomes=len(intents-returns),actual_forward_calls_exact=len(intents) if not intents-returns else None,fit_optimizer_returns=sum(e['kind']=='optimizer_returned' and e.get('phase')=='fit' for e in events),research_optimizer_returns=sum(e['kind']=='optimizer_returned' and e.get('phase')=='research' for e in events))
def inspect_training(c,report):
    config=read(ROOT/'configs/p4_full_baseline_v22r1.json');n=report['completed_epochs'];need(5<=n<=35,'Training epoch cap/first gate differs');pairs=[]
    for epoch in range(1,n+1):
        arms=[read(BASE/f'epochs/{epoch:02d}_{a}.json') for a in ['rgb','rgbd']];pairs.append(sum(x['scene_macro_f1'] for x in arms)/2)
    best=max(pairs);selected=pairs.index(best)+1;sel=read(BASE/'selection.json')
    need(selected==report['selected_epoch']==sel['selected_epoch'] and best==sel['best_paired_monitor_f1'] and pairs==sel['paired_monitor_history'],'Independent epoch selection differs')
    need(n==35 or n-selected>=8,'Early stop before patience threshold')
    need(read(BASE/'epochs/05_rgbd.json')['scene_macro_f1']>=.2,'First-five gate not passed')
    need(report['new_research_updates_per_arm']==n*80,'Research steps and completed epochs differ')
    for arm in ['rgb','rgbd']:
        fit=read(ROOT/f'experiments/p4_full_baseline_v22r1/fit/{arm}/complete.json');check=read(ROOT/f"experiments/p4_full_baseline_v22r1/fit/{arm}/check_{fit['steps']:04d}.json")
        need(fit['steps']<=150 and fit['status']=='passed' and check['passed'] and all(r['metrics']['instances']['f1_iou50']['value']>=.9 and r['metrics']['foreground_pixel_iou']['value']>=.9 for r in check['result']['rows']),'Small-fit gate differs')
        updates=[json.loads(x) for x in (ROOT/f'experiments/p4_full_baseline_v22r1/research/{arm}/updates.jsonl').read_text().splitlines()]
        need([r['step'] for r in updates]==list(range(1,n*80+1)),'Research update sequence missing/duplicate')
        need(all(r['overflow_retries']<=16 for r in updates),'Overflow retry cap differs')
    rgbd=read(BASE/'rgbd_full_development.json')
    need(rgbd['frames']==160 and rgbd['scene_macro_f1']>=.5 and rgbd['scene_macro_foreground_iou']>=.7 and max(pairs[-3:])-min(pairs[-3:])<=.05,'Full development adoption/stability gate failed')
    for p,v in config['protected'].items():
        s=(ROOT/p).stat();need(sha(ROOT/p)==v['sha256'] and s.st_size==v['bytes'] and s.st_mtime_ns==v['mtime_ns'],'Frozen source/input/failure evidence changed: '+p)
    return dict(selected_epoch=selected,completed_epochs=n,selected_model_lineage_updates_per_arm=1600+selected*80,research_updates_per_arm=n*80,monitor_last_three_span=max(pairs[-3:])-min(pairs[-3:]))
def close():
    c=read(CONFIG);need(sha(CONFIG)==read(LOG/'freeze.json')['protocol_sha256'] and sha(Path(__file__))==c['source_sha256'],'Closeout implementation/protocol changed')
    need(not (BASE/'closeout.json').exists(),'P4 already delivered; do not rerun or enter next part')
    need(all((BASE/p).exists() for p in c['required']),'P4 required acceptance evidence incomplete')
    result=read(BASE/'training_and_export.json');need(result['preliminary_baseline_usable'] and all(result['scientific_gates'].values()),'P4 baseline scientifically unusable')
    selection=inspect_training(c,result)
    ap=read(BASE/'evaluation/metrics.json');ro=read(BASE/'evaluation/readonly.json');link=read(BASE/'representative_link/result.json');lr=read(BASE/'representative_link/readonly.json')
    need(ro['status']=='passed' and ro['decoded_frames']==320 and ro['metrics_sha256']==sha(BASE/'evaluation/metrics.json'),'Segmentation full readonly invalid')
    need(link['status']=='accepted_representative_engineering_linkage' and link['frames']==20 and link['scenes']==10 and lr['status']=='passed' and lr['result_sha256']==sha(BASE/'representative_link/result.json'),'Representative interface acceptance invalid')
    for p,v in read(ROOT/'logs/P4/full_baseline_v22r1/evaluation/input_binding.json').items():
        need(sha(ROOT/p)==v['sha256'] and (ROOT/p).stat().st_size==v['bytes'] and (ROOT/p).stat().st_mtime_ns==v['mtime_ns'],'Full readonly input evidence changed')
    for p,v in read(ROOT/'logs/P4/full_baseline_v22r1/evaluation/production_outputs.json').items():need(sha(ROOT/p)==v['sha256'],'AP/figure output changed')
    account={v:calls(ROOT/f'logs/P4/{v}/calls.jsonl') for v in ['full_baseline_v22','full_baseline_v22r1']};account['representative_link']=calls(ROOT/'logs/P4/full_baseline_v22r1/representative_link/calls.jsonl')
    total={k:sum(a[k] for a in account.values()) for k in ['forward_intents','known_forward_returns','unknown_forward_outcomes','fit_optimizer_returns','research_optimizer_returns']};total['actual_forward_calls_exact']=total['forward_intents'] if total['unknown_forward_outcomes']==0 else None
    need(total['forward_intents']<=35000 and total['research_optimizer_returns']<=6400 and total['fit_optimizer_returns']+total['research_optimizer_returns']<=7000,'Cumulative P4 numerical budget exceeded')
    need(sha(ROOT/c['prior_progress'])==c['prior_progress_sha256'],'Previous progress ledger changed')
    ledger=copy.deepcopy(read(ROOT/c['prior_progress']));p4=next(p for p in ledger['packages'] if p['id']=='P4')
    for sub in p4['subdeliverables']:
        if sub['name'] in ['最终稳定基线覆盖与协议闭环','代表性范围的正式抓取链路验收']:need(sub['status']=='pending','P4 credit already used');sub['status']='done'
    p4['earned']=sum(x['weight'] for x in p4['subdeliverables'] if x['status']=='done');p4['evidence'].extend([(BASE/p).relative_to(ROOT).as_posix() for p in c['required']])
    earned=sum(p['earned'] for p in ledger['packages']);need(earned==29,'Unexpected progress increment')
    ledger.update(updated_at=now(),earned_points_current=earned,all_percent_current=earned,experiment_percent_current=earned/80*100,change_reason='Completed existing P4 stable640x480 development baseline2points and fixed10scene20view formal engineering-interface acceptance1point. No new scope/weights; no method-advantage or whole-project completion claim.')
    once(LOG/'progress_after_P4.json',ledger)
    arms={a:dict(f1=read(BASE/(a+'_full_development.json'))['scene_macro_f1'],foreground_iou=read(BASE/(a+'_full_development.json'))['scene_macro_foreground_iou'],**ap['arms'][a]) for a in ['rgb','rgbd']}
    final=dict(status='P4_development_part_accepted_stopped_for_user_review',at=now(),scope=c['scope'],usable=True,selection=selection,arms=arms,linkage={k:link[k] for k in ['frames','scenes','candidates','associated','background','new_official_scores']},accounting_by_version=account,total_new_P4_accounting=total,all_tasks_percent=29,experiment_percent=36.25,progress_definition='fixed planned work-package weighted estimate, not exact elapsed-work measurement',boundaries=c['boundaries'],next_part_started=False,automation_recreated=False,original_state_json_manually_modified=False,evidence={p:sha(BASE/p) for p in c['required']})
    once(BASE/'closeout.json',final)
    text='P4整体开发基线交付与停止记录\n'+final['at']+'\n全部任务总进度29%，实验总进度36.25%，固定工作包加权估算。P4开发部分整体验收通过；未启动P5或任何下一部分，等用户检查。\n'
    text+='RGB/RGBD完整640x480开发评价：\n'+json.dumps(arms,ensure_ascii=False,indent=2)+'\n'
    text+='本次原v22第5轮失败已保留；修复版v22r1仅放大中心监督高斯宽度恢复归一化尺度，未改解码或科学门槛。训练选择与累计实际工作见closeout.json。\n'
    text+='代表性抓取接口：'+json.dumps(final['linkage'],ensure_ascii=False)+'。背景保留；没有新官方几何成绩，所有新候选的官方成功/碰撞结果未测量。\n'
    text+='可用范围：本项目20训练场景、10反复查看的开发场景的单种子普通融合基线及工程抓取接口。不代表70/15/15完整正式实验、样本外候选、方法优势或实体抓取成功。R18双流负结果不改。\n'
    text+='剩余主工作：P2/P3全量数据与完整抓取基线；P5有效质量门控；P6真实几何传播；P7风险与校准；P8全链路消融科学门槛；通过后P9三种子、P10保留测试、P11统计图表和P12论文复现。\n'
    (ROOT/'docs/P4整体交付与停止记录_v22r1.txt').write_text(text,encoding='utf-8')
    handoff=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(handoff,LOG/'handoff_before_closeout.txt');handoff.write_text('最新：'+now()+' P4开发部分整体验收完成，已停止等待用户检查并确认下一主部分。全部任务29%，实验36.25%（固定工作包加权估算）。结果、适用边界、负结果及实际资源账本见docs/P4整体交付与停止记录_v22r1.txt及results/P4/full_baseline_v22r1/closeout.json。没有启动P5，自动任务没有恢复。原state.json未修改。\n\n历史：\n'+handoff.read_text(encoding='utf-8-sig'),encoding='utf-8')
    control=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(control,LOG/'part_control_before_closeout.json');value=read(control);value.update(current_execution_state='P4_development_part_accepted_stopped_for_user_review',recorded_at=now(),all_tasks_percent_planning_estimate=29,experiment_percent_planning_estimate=36.25,current_progress_evidence='logs/P4/full_baseline_v22r1/closeout/progress_after_P4.json');control.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(final,ensure_ascii=False),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','close']);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:
        with project_lock(ROOT):close()
if __name__=='__main__':main()

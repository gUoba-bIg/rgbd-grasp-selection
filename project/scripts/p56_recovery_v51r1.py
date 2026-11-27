"""Single bounded recovery launch; no change to scientific inputs or decisions."""
import os,sys,time,uuid,subprocess,traceback,argparse,json,shutil
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p56_recovery_common_v51r1 import *
from scripts import p56_common_v51 as old
from scripts.pipeline import project_lock,pid_alive
from scripts.p56_official_v51 import event_ledger,load_sealed


def freeze():
    with project_lock(ROOT),old.own_lock(),own_lock():
        need(not CONFIG.exists(),'Recovery already frozen')
        checks=read(LOG/'internal_checks.json');review=read(LOG/'independent_review.json')
        for receipt in (checks,review):
            need(receipt['passed'],'Recovery preparation failed')
            for name in SOURCES:need(receipt['source_sha256'][name]==identity(ROOT/name)['sha256'],'Unreviewed recovery source '+name)
        need(review['protocol']==identity(PROTOCOL),'Protocol differs from review')
        original=old.verify(full=True);state=read(old.STATE)
        need(state['phase']=='failure_preserved_stopped' and state['error']=="RuntimeError('Process tree RSS cap reached')",'Original recovery state differs')
        need(state['elapsed_seconds']==919.5470000000205,'Historical resource use differs')
        original_feature_proof()
        termination=events(old.LOG/'termination_evidence.jsonl')[-1]
        need(termination['confirmed'] and not termination['survivors'],'Old child termination unconfirmed')
        old_calls=old.LOG/'official/calls.jsonl';calls=event_ledger(old_calls)
        need(calls['started_by_kind']==dict(eval_grasp=246,get_grasp_score=2254) and calls['returned_by_kind']==dict(eval_grasp=245,get_grasp_score=2254),'Original actual invocation counts differ')
        need(len(calls['unknown'])==1,'Historical unknown identity count differs')
        unknown=next(r for r in events(old_calls) if r['event']=='intent' and r['id']==calls['unknown'][0])
        need(unknown['kind']=='eval_grasp' and unknown['key']=='score/scene_0022_0064_clean/A','Unexpected historical unknown')
        old_returns={r['id']:r for r in events(old_calls) if r['event']=='returned'}
        source_bindings={rel(p):identity(p) for p in [old.CONFIG,old.STATE,old_calls,old.LOG/'freeze.json',old.LOG/'feature_review.json',old.LOG/'features_complete.json',old.LOG/'feature_calls.jsonl',old.LOG/'neural_calls.jsonl',old.LOG/'children.jsonl',old.LOG/'termination_evidence.jsonl']}
        results=[];keys=set();used_calls=set()
        for p in (old.OUT/'official/results').rglob('*.json'):
            if p.name.endswith('.receipt.json'):continue
            v=load_sealed(p);key=(v['scene'],v['condition'],v['group']);need(key not in keys,'Repeated existing result')
            keys.add(key);need(v['config']==identity(old.CONFIG),'Old result config differs')
            event=old_returns.get(v['call_id']);need(event and event['kind']=='eval_grasp' and event['key']==f"score/scene_{v['scene']:04d}_0064_{v['condition']}/{v['group']}",'Old result lacks returned actual call')
            need(v['call_id'] not in used_calls,'Repeated old result call');used_calls.add(v['call_id'])
            source_bindings[rel(p)]=identity(p);source_bindings[rel(p.with_suffix('.json.receipt.json'))]=identity(p.with_suffix('.json.receipt.json'))
            results.append(rel(p))
        expected={(s,c,g) for s in SCENES[:7] for c in CONDITIONS for g in GROUPS}
        need(keys==expected and len(results)==245,'Old245 scope incomplete or changed')
        protected=dict(original['protected'])
        for directory in (old.LOG,old.OUT):
            for p in directory.rglob('*'):
                if p.is_file():protected[rel(p)]=identity(p)
        for p in [old.CONFIG,old.STATE,CONFIG.parent/'p56_joint_v51.json',PROTOCOL,LOG/'internal_checks.json',LOG/'independent_review.json',LOG/'preparation_usage.json']:
            protected[rel(p)]=identity(p)
        for name in SOURCES:protected[name]=identity(ROOT/name)
        cfg=dict(version='51r1',at=now(),authorization='User explicitly instructed 那就运行呀 after one extra engineering retry was disclosed',
            base_config_path=rel(old.CONFIG),base_config_identity=identity(old.CONFIG),shards=[22,33,39],
            combined_caps=dict(eval_grasp=358,get_grasp_score=7140),formal_wall_seconds=7200,
            old_calls_path=rel(old_calls),original_unknown={k:unknown[k] for k in ('id','kind','key')},allowed_retry_key=unknown['key'],
            original_formal_seconds=state['elapsed_seconds'],source_bindings=source_bindings,protected=protected,
            controller_state=rel(STATE),source_sha256={name:identity(ROOT/name)['sha256'] for name in SOURCES},protocol=identity(PROTOCOL),
            limits=dict(tree_rss_bytes=6*1024**3,cpu_threads=2,output_bytes=2*1024**3,free_disk_min=20*1024**3),
            original_existing_official_results=results,scientific_config_unchanged=True,scientific_thresholds_unchanged=True,
            actual_new_training=0,actual_new_head_forward=0,automatic_expansion=False,
            execution_plan=['score22','score33','score39','review350','cold7','metrics','full_final_integrity_and_stop'])
        need(shutil.disk_usage(ROOT).free>=cfg['limits']['free_disk_min'],'Insufficient disk reserve')
        once(CONFIG,cfg);once(LOG/'freeze.json',dict(at=now(),config=identity(CONFIG),protected_count=len(protected),old_verified_results=245))
        verify();status('frozen_ready',remaining_official_units=105)
        handoff('P5-P6联合验证v51r1分段恢复已冻结：仅补剩余3场景105项，旧245结果与100次特征/后验返回复用；1次历史unknown保留，累计启动帽明确358。科学门槛不变，累计7200秒扣旧919.547秒、6GiB不变。全部任务总进度45%、实验总进度56.25%。入口scripts/p56_recovery_v51r1.py deliver。')
        print(json.dumps(dict(frozen=True,protected=len(protected),old_results=245,missing=105)))


def run():
    from scripts.research_cache_recovery_v44r1 import process_tree_rss
    with project_lock(ROOT),old.own_lock(),own_lock():
        start=time.monotonic();epoch=time.time();unconfirmed=False;peak=0;last_check=-1e30
        c=read(CONFIG)
        need(not events(LOG/'children.jsonl') and not (OUT/'closeout.json').exists(),'Prior recovery launch exists; no implicit retry or reset')
        token=uuid.uuid4().hex;deadline=epoch+remaining_seconds(c,0)
        def guard(force=False):
            nonlocal peak,last_check
            elapsed=time.monotonic()-start
            need(remaining_seconds(c,elapsed)>0 and time.time()<deadline,'Original plus recovery formal wall budget exhausted')
            need(not (LOG/'STOP').exists() and not (old.LOG/'STOP').exists(),'STOP requested')
            if not force and time.monotonic()-last_check<2:return
            last_check=time.monotonic();rss,_=process_tree_rss(os.getpid());peak=max(peak,rss)
            need(rss<c['limits']['tree_rss_bytes'],'Process tree RSS cap reached')
            need(output_bytes()<c['limits']['output_bytes'],'Combined output cap reached')
            need(shutil.disk_usage(ROOT).free>=c['limits']['free_disk_min'],'Disk reserve reached')
            atomic(LOG/'usage.json',dict(at=now(),recovery_seconds=elapsed,original_formal_seconds=c['original_formal_seconds'],cumulative_formal_seconds=c['original_formal_seconds']+elapsed,formal_cap_seconds=7200,peak_tree_rss_bytes=peak,deadline_epoch=deadline))
        def execute(name,mode,scene=None):
            nonlocal unconfirmed
            guard();verify();cid=uuid.uuid4().hex
            cmd=[str(ROOT/'runtime/envs/grasp_eval/Scripts/python.exe'),'-B','scripts/p56_official_recovery_v51r1.py','--config',rel(CONFIG),'--mode',mode]
            if scene is not None:cmd+=['--scene',str(scene)]
            append(LOG/'children.jsonl',dict(at=now(),event='intent',id=cid,kind=name,command=cmd))
            env=os.environ.copy();env['P56_V51R1_CHILD_TOKEN']=token
            details=dict(run_token=token,token=token,deadline_epoch=deadline,controller_pid=os.getpid(),child_pid=None,call_id=cid)
            status(name,**details)
            with (LOG/(name+'.stdout.log')).open('xb') as out,(LOG/(name+'.stderr.log')).open('xb') as err:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                details['child_pid']=child.pid;status(name,**details)
                try:
                    while child.poll() is None:guard();time.sleep(2)
                except BaseException:
                    try:
                        _,pids=process_tree_rss(child.pid)
                        killed=subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False)
                        child.wait(timeout=30);survivors=[p for p in pids if pid_alive(p)]
                        unconfirmed=bool(survivors) or killed.returncode!=0
                        append(LOG/'termination_evidence.jsonl',dict(at=now(),confirmed=not unconfirmed,child_pid=child.pid,returncode=killed.returncode,survivors=survivors))
                    except BaseException as exc:
                        unconfirmed=True;append(LOG/'termination_evidence.jsonl',dict(at=now(),confirmed=False,child_pid=child.pid,error=repr(exc)))
                    raise
                code=child.returncode
            append(LOG/'children.jsonl',dict(at=now(),event='returned',id=cid,kind=name,exit_code=code))
            need(code==0,'Recovery stage failed; preserve all evidence '+name)
        try:
            status('initial_full_integrity',run_token=token,deadline_epoch=deadline,controller_pid=os.getpid())
            verify(full=True);guard(True)
            handoff('P5-P6联合验证v51r1已单次后台启动，剩余105项按场景分进程，随后全350只读/7cold/汇总自动停止。1历史未知保留，原结果和科学门槛不变。全部任务总进度45%、实验总进度56.25%。实际状态runtime/pipeline/p56_joint_v51r1.json；无周期任务。')
            for scene in c['shards']:execute('score_'+str(scene),'score',scene)
            for mode in ('review','cold','metrics'):execute(mode,mode)
            guard();verify(full=True);guard(True);finish(c,guard)
        except BaseException as exc:
            append(LOG/'failures.jsonl',dict(at=now(),error=repr(exc),traceback=traceback.format_exc()))
            status('orphan_stop_unconfirmed' if unconfirmed else 'failure_preserved_stopped',error=repr(exc),recovery_seconds=time.monotonic()-start,cumulative_formal_seconds=c['original_formal_seconds']+time.monotonic()-start,peak_tree_rss_bytes=peak,child_calls=ledger(LOG/'children.jsonl'))
            handoff('P5-P6联合验证v51r1恢复发生失败，证据保留；'+('子树退出未确认，须核进程。' if unconfirmed else '本次已停止。')+'不自动第二次重试或扩预算。全部任务总进度45%、实验总进度56.25%。实际状态runtime/pipeline/p56_joint_v51r1.json。')
            raise


def finish(c,guard):
    import numpy as np
    from scripts.p56_official_recovery_v51r1 import audit_combined,verify_eligible_view
    guard();actual=audit_combined(c)
    need(actual['started_by_kind']['eval_grasp']==358 and actual['returned_by_kind']['eval_grasp']==357,'Actual recovery official scope incomplete')
    need(actual['unknown']==[c['original_unknown']['id']] and not actual['new_unknown_ids'] and len(actual['engineering_retry_ids'])==1,'Historical/new unknown accounting differs')
    need(actual['started_by_kind']['get_grasp_score']==actual['returned_by_kind']['get_grasp_score']<=7140,'Getter calls incomplete or budget exceeded')
    for mode in ('review','metrics'):verify_eligible_view(LOG/'official'/('eligible_returned_'+mode+'.jsonl'))
    wrapper=load_sealed(LOG/'official/recovery_metrics.json')
    metrics=load_sealed(old.OUT/'official/metrics.json');review=load_sealed(old.OUT/'official/review.json')
    need(wrapper['complete_combined_ledger']==actual,'Final complete ledger not bound')
    need(wrapper['scientific_metrics']==identity(old.OUT/'official/metrics.json'),'Scientific metrics binding differs')
    need(metrics['config']==identity(old.CONFIG) and metrics['full_review']==identity(old.OUT/'official/review.json'),'Final scientific config/review binding differs')
    need(metrics['gate_thresholds']==read(old.CONFIG)['gates'],'Scientific gates changed')
    need(review['passed'] and review['official_units']==350 and review['label_sets']==50 and len(review['checked'])==400,'Full350 read-only review missing')
    for name,expected in review['checked'].items():guard();need(identity(ROOT/name)==expected,'Reviewed artifact changed '+name)
    for group in GROUPS:
        cold=load_sealed(old.OUT/'official/cold/scene_0002_0064_clean'/(group+'.json'))
        production=load_sealed(old.OUT/'official/results/scene_0002_0064_clean'/(group+'.json'))
        need(all(cold[k]==production[k] for k in ('official','selected','metrics','normalized','local_order','bindings')),'Cold output differs')
    proofs=original_feature_proof();children=ledger(LOG/'children.jsonl')
    need(children['intents']==6 and children['known_returns']==6 and not children['unknown'],'Recovery lifecycle incomplete')
    lines=['P5-P6下游抓取候选选择联合验证：v51r1恢复结果',now(),'',
        '全部任务总进度45%；实验总进度56.25%。本轮有限开发验证完成后停止，不自动扩大。',
        '本次所需结果均已补齐并通过完整复核；原v51内存失败及1条结果未知的官方调用永久保留。',
        '完整实际eval_grasp账本：358次启动、357次已知返回、1次历史未知；不是全程0未知，也不是原357启动帽内完成。',
        '已返回调用证明视图只支持逐结果复核，不代表完整实际账本。原P5/P6科学采用未通过结论保持。',
        '另有原v51控制器official_score子进程中断未返回记录，单独保留；官方调用未知与子进程未知是两个层级，不相互替代。',
        '固定50开发案例、每例20候选、7组；非全量GraspNet测试、非独立未见确认、非机械臂成功率。','',
        '冻结扩大条件全部通过：'+str(metrics['proposal_expand'])+'。'+('仅支持提出扩大建议，仍须用户确认。' if metrics['proposal_expand'] else '未达到扩大门槛，不能作为两模块有效提升的完整采用证据，不扩大。'),'',
        '组别 | clean AP(%) | 四退化AP(%) | 四退化Top5(mu=.4,%)']
    for g in GROUPS:
        top5=np.mean([r['metrics']['topk']['5']['mu04'] for r in metrics['rows'] if r['group']==g and r['condition']!='clean'])
        lines.append(f"{g} | {metrics['condition_AP'][g]['clean']*100:.6f} | {metrics['mean_degraded_AP'][g]*100:.6f} | {top5*100:.6f}")
    lines+=['','各模块贡献及交互（百分点）']
    for key,value in metrics['contrasts'].items():lines.append(f'{key}: {value*100:+.6f}')
    lines+=['','Top-K：四退化等权平均，依次Top1/5/10/20/50；每项为mu=.4 / 六摩擦平均(%)，分母不因候选不足而缩小。']
    for g in GROUPS:
        parts=[]
        for k in (1,5,10,20,50):
            values=[r['metrics']['topk'][str(k)] for r in metrics['rows'] if r['group']==g and r['condition']!='clean']
            parts.append(f"{np.mean([v['mu04'] for v in values])*100:.4f}/{np.mean([v['six_average'] for v in values])*100:.4f}")
        lines.append(g+': '+', '.join(parts))
    lines+=['','冻结条件逐项结果：'+json.dumps(metrics['gates'],ensure_ascii=False),
        '条件几何关联：'+json.dumps(metrics['conditional_geometry_association'],ensure_ascii=False),
        '候选支持/未知/空抓取/碰撞/失败计数：'+json.dumps(metrics['candidate_support'],ensure_ascii=False),
        '本次新增模型训练、分割前向、P6头前向及抓取生成均0；原100次P6实际返回及全50复验逐字复用。',
        'get_grasp_score实际已知返回：'+str(actual['returned_by_kind']['get_grasp_score'])+'；更深forceclosure未计数，不推断。',
        '完整实际事件及逐行来源：logs/joint_validation_v51r1/official/recovery_metrics.json；逐例结果/AP/TopK：results/joint_validation_v51/official。',
        '资源及最终保护证据：results/joint_validation_v51r1/closeout.json。']
    need(not REPORT.exists(),'Existing recovery report retained');REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    guard(True)
    artifacts={}
    for p in (old.OUT/'official').rglob('*'):
        if p.is_file():guard();artifacts[rel(p)]=identity(p)
    guard(True)
    once(OUT/'closeout.json',dict(at=now(),engineering_recovery_complete=True,original_v51_status='failure_preserved_stopped',
        complete_actual_ledger=actual,original_controller_children=ledger(old.LOG/'children.jsonl'),eligible_returned_view_ledger=metrics['ledger'],historical_unknown_preserved=True,
        required_scientific_outputs_complete=True,bounded_joint_validation=metrics,feature_proof=proofs,
        artifacts=artifacts,report=identity(REPORT),recovery_wrapper=identity(LOG/'official/recovery_metrics.json'),
        usage=read(LOG/'usage.json'),children=children,all_tasks_percent=45,experiment_percent=56.25,
        P5_original_scientific_passed=False,P6_original_scientific_passed=False,expansion_started=False,expansion_requires_user_confirmation=True))
    status('completed_stopped',report=rel(REPORT),closeout=rel(OUT/'closeout.json'),joint_screen_passed=metrics['proposal_expand'])
    handoff('P5-P6联合验证v51r1已完成剩余105项、全350只读、7组真实cold及结果汇总后停止。完整账本358启动/357返回/1历史未知，原失败保留。扩大条件全部通过='+str(metrics['proposal_expand'])+'，不自动扩大。全部任务总进度45%、实验总进度56.25%。报告docs/P5-P6联合候选选择验证结果_v51r1.txt。')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','verify','deliver']);args=parser.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='verify':verify(full=True);print('Recovery full frozen reentry passed')
    else:run()

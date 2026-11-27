"""Finite local P6 correspondence job. No scheduler, no chat polling, automatic report/stop."""
import sys,os,json,time,uuid,hashlib,shutil,subprocess,traceback,argparse
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.p5_fusion_delivery_v38 import read,sha,stat,atomic,need
LOG=ROOT/'logs/P6/selective_v42';OUT=ROOT/'results/P6/selective_v42'
CONF=ROOT/'configs/p6_selective_v42.json';STATE=ROOT/'runtime/pipeline/p6_selective_v42.json'
REPORT=ROOT/'docs/P6匹配歧义有限修复结果_v42.txt'
ORPHAN_UNCONFIRMED=False
def now():return datetime.now().astimezone().isoformat()
def event(kind,**v):
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'controller.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(at=now(),kind=kind,**v),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def records():
    p=LOG/'controller.jsonl'
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
def child_ledger():
    rows=records();aa=[x for x in rows if x['kind']=='child_intent'];bb=[x for x in rows if x['kind']=='child_returned']
    a={x['call_id']:x for x in aa};b={x['call_id']:x for x in bb}
    need(len(aa)==len(a) and len(bb)==len(b) and set(b)<=set(a),'Child ledger inconsistent')
    return dict(intents=len(a),known_returns=len(b),unknown_ids=sorted(set(a)-set(b)))
def status(phase,**v):
    atomic(STATE,dict(at=now(),phase=phase,current_part='P6_selective_correspondence_v42',
        all_tasks_percent=45,experiment_percent=56.25,progress_basis='Fixed deliverables; correspondence alone does not close remaining P6 coverage/sharpness',
        P5_scientific_adoption=False,P7_started=False,P10_still_stopped=True,recurring_automation=False,**v))
def verify():
    c=read(CONF);need(sha(CONF)==read(LOG/'controller_freeze.json')['config_sha256'],'Frozen controller changed')
    for name,value in c['protected'].items():need(stat(ROOT/name)==value,'Frozen file changed '+name)
    need(c['controller_wall_seconds']==1800,'Frozen cumulative budget differs')
    return c
def handoff(text_line,tag):
    p=ROOT/'docs/实验自动续接说明.txt';backup=LOG/('handoff_before_'+tag+'.txt')
    if not backup.exists():shutil.copy2(p,backup)
    p.write_text('最新：'+now()+' '+text_line+'\n\n历史：\n'+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
def supervision(phase,**values):
    p=ROOT/'runtime/pipeline/part_level_supervision.json';backup=LOG/('supervision_before_'+phase+'.json')
    if not backup.exists():shutil.copy2(p,backup)
    v=read(p);v.update(recorded_at=now(),current_authorized_part='P6_selective_correspondence_v42',
        current_execution_state=phase,all_tasks_percent_planning_estimate=45,experiment_percent_planning_estimate=56.25,
        automatic_resume_authorized=False,automatic_scheduler_recreated=False,remaining_research_updates_cap=0,
        stop_after='P6_selective_correspondence_v42',stop_after_part='P6_selective_correspondence_v42',**values)
    atomic(p,v)
def completed_verify(c):
    need(not child_ledger()['unknown_ids'],'Missing child return')
    for stage in c['stages']:need((ROOT/stage['done']).exists(),'Missing completed '+stage['name'])
    r=read(LOG/'review.json');cold=read(LOG/'cold.json');result=read(OUT/'result.json')
    need(r['passed'] and cold['passed'],'Independent review/cold must actually pass')
    need(r.get('engineering_passed') is True and cold.get('engineering_passed') is True,'Engineering acceptance not complete')
    need(r['identifiability_supported']==cold['identifiability_supported']==result['identifiability_supported'],'Review/cold/result scientific decision differs')
    need(result['P6_full_passed'] is False and result['uvz_calibrated'] is False,'Correspondence gate cannot declare full P6/calibration')
    need(bool(result['identifiability_supported'])==all(result['gates'].values()),'Scientific decision differs from fixed gates')
    for name,value in r['reviewed_outputs'].items():need(stat(ROOT/name)==value,'Reviewed result changed '+name)
    inventory={p.relative_to(ROOT).as_posix():stat(p) for p in OUT.rglob('*') if p.is_file() and p!=OUT/'closeout.json'}
    expected={k:v for k,v in r['reviewed_outputs'].items() if k.startswith('results/P6/selective_v42/')}
    need(inventory==expected,'Reviewed output inventory differs')
    from scripts.p6_selective_v42 import audit_ledger
    actual=audit_ledger()
    need(not actual['unknown_calls'],'Unknown correspondence/fit call remains unresolved')
    for kind,count in [('local_alignment_generate',720),('local_alignment_review',720),('ridge_fit',0)]:
        need(actual['counts'][kind]==count,'Incomplete actual call count '+kind)
    need(actual['counts']['neural_forward']==0 and actual['counts']['ap']==0,'Unexpected neural/AP research')
    need(r['counts']==cold['counts']==actual['counts'],'Review/cold/actual research counts differ')
    return result
def finish(c,guard):
    result=completed_verify(c);verify();guard()
    passed=bool(result['identifiability_supported'])
    elapsed=c['preparation_seconds']+sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'))
    lines=['P6匹配歧义有限修复实验 v42',now(),'',
        '全部任务总进度45%；实验总进度56.25%。固定交付权重；本次可识别性关口不替代P6剩余覆盖/锐度与完整误差验收，不重复计分。',
        '后台有限链路已完成：720个预定案例、复用冻结模型且0新增拟合、固定开发评价、完整独立复验及真实新进程冷重入，现已停止。',
        '工程验收通过；对应可识别性关口通过='+str(passed)+'；完整P6通过=False；full uvz校准=False。',
        '范围仅限既有20暴露场景、80帧、9种全图整数平移。不是局部非刚性配准、独立测试、真实传感器校准或抓取收益。已知warp仅为不可部署正对照，方法预测不读取GT位移/标签/原参考深度。',
        '本次0神经网络训练/前向、0官方AP、0抓取调用。真实CPU局部估计调用与重放计数见组件账本和验收；原全局结果缓存复用，0重复全局估计，不将720名义案例改称神经前向次数。',
        '局部选择性规则：六格中至少四格支持，且支持覆盖两行及至少两列。精确零票和非零票互斥；低支持保持原位置是unknown_abstain，不是成功识别零位移。所有拒绝保留分母并触发原no_rejected门槛。',
        '科学判定明细：'+json.dumps(result['gates'],ensure_ascii=False),
        '结果摘要：'+json.dumps(result['summary'],ensure_ascii=False,indent=2),
        '评价集预测状态计数：'+json.dumps(result['summary']['decision_counts'],ensure_ascii=False),
        '冻结原全局方法对照：'+json.dumps(read(ROOT/'results/P6/correspondence_v40/result.json')['summary'],ensure_ascii=False,indent=2),
        '所有逐案例结果及可复查计数见results/P6/selective_v42/result.json和generated.json；本报告的工程验收状态以review/cold为准。',
        '下一步解释：'+('该唯一选择性候选通过限定开发关口；还需未参与选择的数据确认和残余u/v/z覆盖、锐度与边界多表面检查，不自动宣布P6整体成功或接入抓取。' if passed else '该唯一选择性候选未通过完整原门槛。若动作误差改善但仍有拒绝，也只能报告保守动作改善，不能宣称对应已可识别。停止继续改网格/投票数/近零阈值追求通过；下一项建议是明确评估多个对应假设及保留原观测的不确定性表示，须另立可审查目标，不自动执行。'),
        '无论本次局部支持强弱，都没有独立真值保证原生偏差与外加warp已被分开；复用全局ridge到局部特征也不等于局部概率已校准。',
        'P5质量融合尚未科学采用，结论不变。未启动P7/P10或周期任务，未自动换模型、追加拟合或扩大数据。',
        f'本控制链加准备预算记账约{elapsed:.3f}秒（上限1800秒），含根预检保守扣款10秒，并非精确端到端耗时；实现和代码人工审查时间另计。',
        '后台自动写入本报告与状态文件；没有使用LLM实时轮询，也不承诺没有后续对话唤醒时自动发送聊天通知。']
    need(not REPORT.exists(),'Preserve any previous report before recovery')
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    proof=dict(at=now(),engineering_passed=True,identifiability_supported=passed,P6_full_passed=False,uvz_calibrated=False,
        result=result,report=stat(REPORT),reviews={p.relative_to(ROOT).as_posix():stat(p) for p in (LOG/'review.json',LOG/'cold.json')},
        all_tasks_percent=45,experiment_percent=56.25,child_calls=child_ledger(),stopped=True,P7_started=False,P10_still_stopped=True)
    atomic(OUT/'closeout.json',proof)
    status('completed_stopped',engineering_passed=True,identifiability_supported=passed,P6_full_passed=False,report=REPORT.relative_to(ROOT).as_posix())
    supervision('P6_v42_completed_stopped',current_runner=None,waiting_for_user_next_part_confirmation=True,
        active_internal_repair=None,current_execution_evidence='results/P6/selective_v42/closeout.json',
        summary_report=REPORT.relative_to(ROOT).as_posix(),final_readonly='logs/P6/selective_v42/review.json',
        final_cold='logs/P6/selective_v42/cold.json',final_delivery_index='results/P6/selective_v42/closeout.json')
    handoff(f'P6 v42匹配歧义修复后台实验、独立复验、真实冷重入及报告已完成并停止。可识别性通过={passed}，完整P6/uvz校准未完成。报告docs/P6匹配歧义有限修复结果_v42.txt。全部任务总进度45%、实验总进度56.25%，本子关口不加满P6剩余分。不自动启动新模型、P7/P10或周期任务。','complete')
    event('completed_stopped',identifiability_supported=passed)
def fail(error):
    event('failure_preserved',error=repr(error),traceback=traceback.format_exc())
    phase='orphan_stop_unconfirmed' if ORPHAN_UNCONFIRMED else 'failed_stopped'
    status(phase,error=repr(error),unknown_child_ids=child_ledger()['unknown_ids'],termination_confirmed=not ORPHAN_UNCONFIRMED)
    failure_report=ROOT/'docs/P6匹配歧义有限修复运行异常_v42.txt'
    if failure_report.exists():shutil.copy2(failure_report,LOG/('failure_report_'+uuid.uuid4().hex+'.txt'))
    failure_report.write_text(('P6 v42控制器因异常退出；子进程停止尚未确认，请先核查日志中的PID和锁。' if ORPHAN_UNCONFIRMED else 'P6 v42后台任务已因异常停止，证据保留。')+'\n'+now()+'\n'+repr(error)+'\n全部任务总进度45%；实验总进度56.25%。未通过完整验收，不报告科学成功。\n调用/未知记录：'+json.dumps(child_ledger(),ensure_ascii=False)+'\n不自动重试、扩预算或启动P7/P10。\n',encoding='utf-8')
    supervision('P6_v42_'+phase,current_runner=None,waiting_for_user_next_part_confirmation=True,
        summary_report=failure_report.relative_to(ROOT).as_posix(),current_execution_evidence='runtime/pipeline/p6_selective_v42.json')
    handoff(('P6 v42控制器异常退出、子进程停止未确认；' if ORPHAN_UNCONFIRMED else 'P6 v42后台任务异常停止；')+'日志、未知调用及中间结果保留。见docs/P6匹配歧义有限修复运行异常_v42.txt。全部任务总进度45%、实验总进度56.25%；先核进程与锁，禁止重复启动/扩预算，不进P7/P10。','failure')
def deliver():
    global ORPHAN_UNCONFIRMED
    start=time.monotonic();c=verify();need(not child_ledger()['unknown_ids'],'Unknown child requires process/evidence audit before recovery')
    if (OUT/'closeout.json').exists():
        result=completed_verify(c);p=read(OUT/'closeout.json')
        need(read(STATE)['phase']=='completed_stopped' and p['result']==result,'Completed state/result differs')
        need(stat(REPORT)==p['report'],'Completed report changed')
        for name,value in p['reviews'].items():need(stat(ROOT/name)==value,'Completed review receipt changed')
        print(json.dumps(dict(already_complete=True,no_new_research=True)));return
    usage=LOG/('controller_usage_'+uuid.uuid4().hex+'.json')
    previous=c['preparation_seconds']+sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'))
    def guard():
        seconds=time.monotonic()-start;atomic(usage,dict(at=now(),seconds=seconds,state='running'))
        need(previous+seconds<1800,'Cumulative1800s control budget exhausted')
        need(not (LOG/'STOP').exists(),'User stop marker found')
    try:
        from scripts.pipeline import project_lock
        for stage in c['stages']:
            verify();guard()
            if (ROOT/stage['done']).exists():event('stage_reused',stage=stage['name']);continue
            with project_lock(ROOT):pass
            cid=uuid.uuid4().hex;cmd=[str(ROOT/c['python']),'-X','utf8','-B','scripts/p6_selective_v42.py',stage['name']]
            event('child_intent',call_id=cid,stage=stage['name'],command=cmd)
            with (LOG/(stage['name']+'_'+cid+'.stdout.log')).open('xb') as out,(LOG/(stage['name']+'_'+cid+'.stderr.log')).open('xb') as err:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                event('child_started',call_id=cid,child_pid=child.pid,controller_pid=os.getpid(),stage=stage['name'])
                status(stage['name'],controller_pid=os.getpid(),child_pid=child.pid)
                try:
                    while child.poll() is None:guard();time.sleep(2)
                except BaseException:
                    killed=subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
                    ORPHAN_UNCONFIRMED=killed.returncode!=0 or child.poll() is None
                    atomic(LOG/('termination_'+cid+'.json'),dict(at=now(),call_id=cid,child_pid=child.pid,
                        taskkill_returncode=killed.returncode,child_exit_code=child.poll(),tree_stop_confirmed=not ORPHAN_UNCONFIRMED))
                    raise
            event('child_returned',call_id=cid,stage=stage['name'],exit_code=child.returncode)
            need(child.returncode==0 and (ROOT/stage['done']).exists(),'Stage failed/missing receipt '+stage['name'])
        finish(c,guard);atomic(usage,dict(at=now(),seconds=time.monotonic()-start,state='completed_stopped'))
    except BaseException:
        atomic(usage,dict(at=now(),seconds=time.monotonic()-start,state='failed_preserved'));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('deliver','verify'));args=p.parse_args()
    if args.mode=='verify':print(json.dumps(dict(passed=True,files=len(verify()['protected']))));sys.exit(0)
    import msvcrt
    with (ROOT/'runtime/pipeline/p6_selective_v42.lock').open('a+b') as f:
        f.seek(0,2)
        if not f.tell():f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:deliver()
        except BaseException as e:fail(e);raise
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

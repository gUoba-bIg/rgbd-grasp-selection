"""One finite, frozen restoration development part; no automatic next experiment."""
import os, sys, json, time, uuid, hashlib, shutil, subprocess, traceback
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
LOG = ROOT / 'logs/Restore/v37'
OUT = ROOT / 'results/Restore/v37'
CONF = ROOT / 'configs/restore_direction_v37.json'
STATE = ROOT / 'runtime/pipeline/restoration_v37.json'

def now(): return datetime.now().astimezone().isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def need(v, m):
    if not v: raise RuntimeError(m)
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def stat(p):
    s=Path(p).stat()
    return dict(sha256=sha(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)
def atomic(p, v):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with t.open('x', encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    start=time.monotonic()
    while True:
        try: os.replace(t,p); return
        except PermissionError as e:
            if getattr(e,'winerror',None) not in (5,32) or time.monotonic()-start>2: raise
            time.sleep(.05)
def once(p,v):
    if Path(p).exists(): need(read(p)==v,'Immutable record differs '+str(p))
    else: atomic(p,v)
def event(kind,**v):
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'controller.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(at=now(),kind=kind,**v),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def status(phase,**v):
    atomic(STATE,dict(at=now(),phase=phase,all_tasks_percent=None,experiment_percent=None,
        historical_revised_scope_progress=dict(all_tasks_percent=72,experiment_percent=90),
        progress_reason='Restored original scope is being rebaselined; no use of local completion as total percent',
        P10_still_stopped=True,P11_started=False,recurring_automation=False,stop_after='restoration_v37',**v))
def verify():
    c=read(CONF);need(sha(CONF)==read(LOG/'freeze.json')['config_sha256'],'Frozen restoration protocol changed')
    for p,s in c['protected'].items():need(stat(ROOT/p)==s,'Frozen source or parent changed '+p)
    return c
def records():
    p=LOG/'controller.jsonl'
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
def ledger():
    rows=records();aa=[r for r in rows if r['kind']=='child_intent'];bb=[r for r in rows if r['kind']=='child_returned']
    a={r['call_id']:r for r in aa};b={r['call_id']:r for r in bb}
    need(len(a)==len(aa) and len(b)==len(bb),'Repeated controller call identity')
    need(set(b)<=set(a),'Controller return lacks intent')
    return a,b,sorted(set(a)-set(b))

def deliver():
    from scripts.pipeline import project_lock
    c=verify();need(not ledger()[2],'Unknown controller child; first audit actual processes and preserve unknown')
    old=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'))
    start=time.monotonic();usage=LOG/('controller_usage_'+uuid.uuid4().hex+'.json')
    def guard():
        elapsed=time.monotonic()-start;atomic(usage,dict(at=now(),seconds=elapsed))
        need(old+elapsed<c['controller_wall_seconds'],'Cumulative restoration controller budget exhausted')
        need(not (LOG/'STOP').exists(),'User stop marker present')
    for s in c['stages']:
        marker=ROOT/s['done']
        if marker.exists():
            read(marker);event('stage_reused',stage=s['name'],marker=s['done']);continue
        verify();guard()
        with project_lock(ROOT): pass
        cid=uuid.uuid4().hex
        cmd=[str(ROOT/s['python']),'-X','utf8','-B',s['script']]+s['args']
        event('child_intent',call_id=cid,stage=s['name'],command=cmd)
        with (LOG/(s['name']+'_'+cid+'.stdout.log')).open('xb') as out,(LOG/(s['name']+'_'+cid+'.stderr.log')).open('xb') as err:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            event('child_started',call_id=cid,stage=s['name'],controller_pid=os.getpid(),child_pid=p.pid)
            status(s['name'],controller_pid=os.getpid(),child_pid=p.pid)
            try:
                while p.poll() is None: guard();time.sleep(2)
            except BaseException:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False)
                raise
        event('child_returned',call_id=cid,stage=s['name'],exit_code=p.returncode)
        need(p.returncode==0,'Stage failed with logs retained: '+s['name'])
        need(marker.exists(),'Stage returned without completion artifact '+s['name']);read(marker)
    verify();guard();finish(c)
    atomic(usage,dict(at=now(),seconds=time.monotonic()-start,state='completed_stopped'))

def finish(c):
    need(not ledger()[2],'Controller child missing return')
    for s in c['stages']:need((ROOT/s['done']).exists(),'Missing completed stage '+s['name'])
    quality=read(ROOT/c['reports']['quality_decision'])
    uncertainty=read(ROOT/c['reports']['uncertainty_decision'])
    reviews={k:read(ROOT/p) for k,p in c['reports']['reviews'].items()}
    need(all(r.get('passed') is True for r in reviews.values()),'A complete independent review or cold entry is missing')
    qpass=bool(quality['development_repair_supported']);upass=bool(uncertainty[c['reports']['uncertainty_pass_field']])
    report=ROOT/'docs/原方向恢复有限验证结果_v37.txt'
    need(not report.exists(),'Existing final report must be preserved')
    lines=['原方向恢复有限开发验证 v37',now(),'',
      '本部分已完成并停止；没有恢复P10，也没有进入P11或建立周期任务。',
      '全部任务总进度：恢复范围待重算。实验总进度：恢复范围待重算。旧72%/90%为收缩路线历史记账，不作为本路线当前比例。',
      '工程复验与真实新进程冷重入：全部通过。开发质量准入='+str(qpass)+'；条件深度误差准入='+str(upass)+'。',
      '本次采用固定已暴露开发数据。通过只说明值得扩大验证，不是原P5/P6全范围科学完成，也不是抓取已改善或论文可发表的保证。',
      '质量特征残差结果：'+json.dumps(quality,ensure_ascii=False),
      '条件Student-t误差结果：'+json.dumps({k:v for k,v in uncertainty.items() if k!='artifact_sha256'},ensure_ascii=False),
      '质量真实调用账本：'+json.dumps(reviews['quality_cold'].get('accounting',{}),ensure_ascii=False),
      '后续边界：完整(u,v,z)误差辨识、实例软关联与几何风险抓取对照、完整方法多种子及结果盲独立确认尚需完成。点级覆盖不能直接视为抓取碰撞概率。',
      '路线取舍：'+('两个开发关口均通过；建议在原方向内准备后续完整验证，当前停止供用户检查。' if qpass and upass else '至少一个开发关口未通过；保留负结果，当前不扩大训练或全量抓取评价。结合具体失败指标提出下一次有界修复，用户检查后再继续。'),
      '旧P10停机未知调用91176b8f3d544cf7b98ff793e364d497和旧P7未知调用2990905cbfa24838bb5e7b932827f9d7保持未知；不计成本为零或补造返回。',
      '冻结协议：configs/restore_direction_v37.json；子调用及失败证据：logs/Restore/v37。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    proof=dict(at=now(),engineering_passed=True,quality_development_supported=qpass,
        uncertainty_development_supported=upass,full_original_method_demonstrated=False,
        report=stat(report),review_files={p:stat(ROOT/p) for p in c['reports']['reviews'].values()},
        all_tasks_percent=None,experiment_percent=None,stopped=True,P10_still_stopped=True,P11_started=False)
    once(OUT/'closeout.json',proof)
    h=ROOT/'docs/实验自动续接说明.txt';shutil.copy2(h,LOG/'handoff_before_close.txt')
    h.write_text(f'最新：{now()} 原方向恢复v37有限开发验证已完成并停止。完整工程复验和真实新进程冷重入通过；质量开发准入={qpass}，条件深度误差准入={upass}。本次不是原P5/P6完整科学验收，不是抓取收益。报告docs/原方向恢复有限验证结果_v37.txt。P10保持停止，不进P11或周期任务。全部任务总进度与实验总进度均待按恢复范围重算，旧72%/90%仅历史记账。等待用户检查下一部分。\n\n历史：\n'+h.read_text(encoding='utf-8-sig'),encoding='utf-8')
    p=ROOT/'runtime/pipeline/part_level_supervision.json';shutil.copy2(p,LOG/'supervision_before_close.json');v=read(p)
    v.update(recorded_at=now(),current_execution_state='restoration_v37_completed_stopped',current_runner=None,
        waiting_for_user_next_part_confirmation=True,current_execution_evidence='results/Restore/v37/closeout.json',
        summary_report=report.relative_to(ROOT).as_posix(),all_tasks_percent_planning_estimate=None,
        experiment_percent_planning_estimate=None,automatic_resume_authorized=False)
    atomic(p,v);status('completed_stopped',engineering_passed=True,quality_development_supported=qpass,uncertainty_development_supported=upass)
    event('completed_stopped',quality_development_supported=qpass,uncertainty_development_supported=upass)

if __name__=='__main__':
    import msvcrt
    mode=sys.argv[1]
    if mode=='verify': print(json.dumps(dict(passed=True,protected=len(verify()['protected']))));sys.exit(0)
    need(mode=='deliver','Unknown command')
    lock=ROOT/'runtime/pipeline/restoration_v37.lock'
    with lock.open('a+b') as f:
        f.seek(0,2)
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try: deliver()
        except Exception as e:
            event('failure_preserved',error=repr(e),traceback=traceback.format_exc());status('engineering_failure_preserved',error=repr(e));raise
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

"""Finite cache-evidence delivery. No scheduler, new fit, network inference or AP."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='2'
import sys,json,time,hashlib,uuid,shutil,subprocess,traceback,ctypes,argparse
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
LOG=ROOT/'logs/research_v44r1';OUT=ROOT/'results/research_v44r1'
CONFIG=ROOT/'configs/research_cache_v44r1.json'
STATE=ROOT/'runtime/pipeline/research_cache_v44r1.json'
REPORT=ROOT/'docs/双创新点缓存证据审查恢复结果_v44r1.txt'
PROTOCOL=ROOT/'docs/双创新点缓存证据审查恢复协议_v44r1.txt'
ORIGINAL_STATE_SHA='b48eadde3c1f7ea400c48dd206f707ecf804d54f6f018a013e9d662dcc4b2b76'
PARTS={
 'P5':dict(module='scripts.p5_complementarity_review_v44r1',script='scripts/p5_complementarity_review_v44r1.py',log='logs/P5/complementarity_v44r1',out='results/P5/complementarity_v44r1',budget=1800),
 'P6':dict(module='scripts.p6_cache_audit_v44',script='scripts/p6_cache_audit_v44.py',log='logs/P6/cache_audit_v44',out='results/P6/cache_audit_v44',result='result.json',budget=600)}
OLD_LOG=ROOT/'logs/research_v44'
OLD_P5_LOG=ROOT/'logs/P5/complementarity_v44'
OLD_P5_OUT=ROOT/'results/P5/complementarity_v44'

def now():return datetime.now().astimezone().isoformat()
def need(v,m):
    if not bool(v):raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def identity(p):
    p=Path(p);s=p.stat()
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(sha256=h,bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def write_once(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def append(value):
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'controller.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(at=now(),**value),ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def ledger():
    p=LOG/'controller.jsonl';events=[json.loads(v) for v in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
    starts={e['id']:e for e in events if e['kind']=='child_intent'}
    ends={e['id']:e for e in events if e['kind']=='child_returned'}
    need(set(ends)<=set(starts),'Unpaired return')
    return dict(intents=len(starts),known_returns=len(ends),unknown=sorted(set(starts)-set(ends)))
def backup(p,tag):
    p=Path(p)
    if p.exists():
        dest=LOG/'history';dest.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest/(tag+'_'+uuid.uuid4().hex+'_'+p.name))
def status(phase,**fields):
    atomic(STATE,dict(at=now(),phase=phase,all_tasks_percent=45,experiment_percent=56.25,
        scope='Existing P5 development predictions and P6 training/calibration cache evidence only',
        new_neural_inference=0,new_AP=0,new_fit=0,P5_scientific_adoption=False,P6_scientific_adoption=False,
        P7_started=False,P10_started=False,recurring_automation=False,**fields))
def supervise(phase,**fields):
    p=ROOT/'runtime/pipeline/part_level_supervision.json';backup(p,phase);x=read(p)
    x.update(recorded_at=now(),current_authorized_part='P5_P6_evidence_and_bounded_repair',
        current_execution_state=phase,all_tasks_percent_planning_estimate=45,experiment_percent_planning_estimate=56.25,
        user_authorization='2026-11-02用户同意按两创新点路线先缓存证据、成立后有限修复，可后台；整部分后停止',
        current_execution_evidence=rel(STATE),current_runner='scripts/research_cache_recovery_v44r1.py',
        current_progress_evidence='logs/P5/completion_v39/progress_after_part.json',
        unchanged_progress_reason='Cache decision evidence adds no completed main-delivery points',
        waiting_for_user_next_part_confirmation=False,internal_checks_trials_and_repairs_require_new_confirmation=False,
        automatic_scheduler_recreated=False,automatic_resume_authorized=False,active_internal_repair=None,
        stop_after='current_authorized_P5_P6_evidence_and_bounded_repair',stop_after_part='current_authorized_P5_P6_evidence_and_bounded_repair',
        summary_report=None,final_readonly=None,final_delivery_index=None,final_cold=None,next_repair_started=False)
    x.update(fields);atomic(p,x)
def handoff(message,tag):
    p=ROOT/'docs/实验自动续接说明.txt';backup(p,tag)
    p.write_text('最新：'+now()+' '+message+'\n\n历史：\n'+p.read_text(encoding='utf-8-sig'),encoding='utf-8')

@contextmanager
def own_lock():
    import msvcrt
    p=ROOT/'runtime/pipeline/research_cache_v44r1.lock';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+b') as f:
        f.seek(0,2)
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def protected_check(c):
    need(Path.cwd().resolve()==ROOT,'Explicit project cwd required')
    for name,value in c['protected'].items():need(identity(ROOT/name)==value,'Changed protected file '+name)
    need(identity(ROOT/'runtime/pipeline/state.json')['sha256']==ORIGINAL_STATE_SHA,'Original state differs')
def verify():
    c=read(CONFIG);need(identity(CONFIG)==read(LOG/'freeze.json')['config'],'Frozen config differs');protected_check(c);return c

def freeze():
    import importlib
    from scripts.pipeline import project_lock
    from scripts import research_cache_delivery_v44 as parent
    started=time.monotonic()
    with project_lock(ROOT),own_lock():
        need(not CONFIG.exists(),'Recovery already frozen')
        old=parent.verify()
        checks=read(LOG/'internal_checks.json');review=read(LOG/'independent_review.json')
        for receipt in (checks,review):
            need(receipt['passed'],'Preparation or independent review failed')
            for path,h in receipt['source_sha256'].items():need(identity(ROOT/path)['sha256']==h,'Reviewed source changed '+path)
        need(not parent.ledger()['unknown'],'Unknown parent children require audit')
        five=importlib.import_module(PARTS['P5']['module'])
        protected=dict(old['protected'])
        bound={rel(ROOT/p if not Path(p).is_absolute() else p):identity(ROOT/p if not Path(p).is_absolute() else p) for p in five.dependency_paths()}
        for name in ('scripts/research_cache_recovery_v44r1.py',rel(PROTOCOL),'logs/research_v44r1/independent_review.json','logs/research_v44r1/internal_checks.json'):
            bound[name]=identity(ROOT/name)
        auth=ROOT/PARTS['P5']['log']/'authorization.json'
        write_once(auth,dict(at=now(),authorized=True,cache_analysis_only=True,new_model_calls=0,new_AP_calls=0,new_fit_calls=0,
            old_expected_unknown=1,extra_failed_case_engineering_verification=1,old_review_attempts=26,new_review_attempts_cap=775,
            total_unique_reviewed_cases=800,total_review_attempts_expected=801,cumulative_seconds=1800,protected=bound))
        protected.update(bound);protected[rel(auth)]=identity(auth)
        for p in [ROOT/'configs/research_cache_v44.json',OLD_LOG/'freeze.json',OLD_LOG/'controller.jsonl',ROOT/'runtime/pipeline/research_cache_v44.json']+list(OLD_LOG.glob('controller_usage_*.json')):
            protected[rel(p)]=identity(p)
        old_usage=[read(p) for p in OLD_LOG.glob('controller_usage_*.json')]
        measured_prep=sum(read(p)['seconds'] for p in (ROOT/PARTS['P5']['log']).glob('usage_*.json'))
        recovery_prep=float(checks['measured_preparation_seconds'])+measured_prep+30.
        inherited_total=old['preparation_seconds']+sum(x['seconds'] for x in old_usage)
        inherited_parts={k:old['part_preparation_seconds'][k]+sum(x.get('part_seconds',{}).get(k,0) for x in old_usage) for k in PARTS}
        inherited_parts['P5']+=measured_prep
        stages=[dict(part='P5',mode=mode,script=PARTS['P5']['script'],receipt=PARTS['P5']['log']+'/'+mode+'.json') for mode in ('review','cold')]
        stages += [x for x in old['stages'] if x['part']=='P6']
        c=dict(at=now(),protected=protected,stages=stages,wall_limit_seconds=old['wall_limit_seconds'],
            preparation_seconds=inherited_total+recovery_prep,part_preparation_seconds=inherited_parts,
            inherited_failed_seconds=sum(x['seconds'] for x in old_usage),recovery_preparation_seconds=recovery_prep,
            part_wall_limits=old['part_wall_limits'],total_output_limit=old['total_output_limit'],peak_rss_limit=old['peak_rss_limit'],
            original_v44_status='failed_preserved',old_unknown_preserved=1,extra_engineering_verification=1,
            scope='Only exact-nearest verification correction and remaining original P6 evidence analysis')
        need(c['preparation_seconds']<c['wall_limit_seconds'],'No remaining original time budget')
        protected_check(c);write_once(CONFIG,c);write_once(LOG/'freeze.json',dict(at=now(),config=identity(CONFIG)))
        need(time.monotonic()-started<=30,'Freeze overhead exceeded reserved30s')
        status('frozen_ready',old_unknown_preserved=1)
        supervise('v44r1_frozen_ready')
        handoff('v44独立复验器的Pillow缩放边界误差已用精确有理NN约定定位，v44r1补充验收已冻结。主800结果/25成功审查及旧失败、1未知不改；只补775审查，其中1次失败项替代工程验证单列。实际总尝试801、唯一覆盖800不能混写。总2700秒/P5原1800秒与raw解码/RSS硬帽沿用累计不重置；之后原P6缓存分析/复验/冷重入/报告后停止。全部任务总进度45%、实验总进度56.25%。入口scripts/research_cache_recovery_v44r1.py deliver，禁止重复。','freeze')
        print(json.dumps(dict(frozen=True,stages=len(stages),remaining_seconds=c['wall_limit_seconds']-c['preparation_seconds'],old_unknown_preserved=1)))

def process_tree_rss(pid):
    """Inspect the actual venv Python descendants as well as its redirector."""
    from ctypes import wintypes as W
    class Entry(ctypes.Structure):
        _fields_=[('dwSize',W.DWORD),('cntUsage',W.DWORD),('pid',W.DWORD),('heap',ctypes.c_size_t),('module',W.DWORD),('threads',W.DWORD),('ppid',W.DWORD),('priority',W.LONG),('flags',W.DWORD),('exe',W.WCHAR*260)]
    class Counters(ctypes.Structure):
        _fields_=[('cb',W.DWORD),('faults',W.DWORD)]+[(n,ctypes.c_size_t) for n in ('peak','working','a','b','c','d','e','f')]
    k=ctypes.WinDLL('kernel32',use_last_error=True);p=ctypes.WinDLL('psapi',use_last_error=True)
    k.CreateToolhelp32Snapshot.argtypes=[W.DWORD,W.DWORD];k.CreateToolhelp32Snapshot.restype=W.HANDLE
    k.Process32FirstW.argtypes=[W.HANDLE,ctypes.POINTER(Entry)];k.Process32NextW.argtypes=[W.HANDLE,ctypes.POINTER(Entry)]
    k.CloseHandle.argtypes=[W.HANDLE];k.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];k.OpenProcess.restype=W.HANDLE
    p.GetProcessMemoryInfo.argtypes=[W.HANDLE,ctypes.POINTER(Counters),W.DWORD]
    k.GetExitCodeProcess.argtypes=[W.HANDLE,ctypes.POINTER(W.DWORD)]
    def exited(member):
        handle=k.OpenProcess(0x1000,False,member)
        if not handle:
            need(ctypes.get_last_error()==87,'Cannot establish process liveness after memory failure')
            return True
        try:
            code=W.DWORD();need(k.GetExitCodeProcess(handle,ctypes.byref(code)),'Cannot read process exit code')
            return code.value!=259
        finally:k.CloseHandle(handle)
    snapshot=k.CreateToolhelp32Snapshot(2,0);need(snapshot!=ctypes.c_void_p(-1).value,'Process snapshot failed')
    entry=Entry();entry.dwSize=ctypes.sizeof(entry);parents={}
    try:
        ok=k.Process32FirstW(snapshot,ctypes.byref(entry))
        while ok:
            parents[entry.pid]=entry.ppid;ok=k.Process32NextW(snapshot,ctypes.byref(entry))
    finally:k.CloseHandle(snapshot)
    found={pid}
    while True:
        extra={n for n,parent in parents.items() if parent in found}
        if extra<=found:break
        found|=extra
    total=0;seen=[]
    for member in found:
        handle=k.OpenProcess(0x410,False,member)
        if not handle:
            need(exited(member),'Cannot inspect live process memory');continue
        try:
            exit_code=W.DWORD()
            need(k.GetExitCodeProcess(handle,ctypes.byref(exit_code)),'Cannot determine queried process state')
            if exit_code.value!=259:continue
            value=Counters();value.cb=ctypes.sizeof(value)
            if not p.GetProcessMemoryInfo(handle,ctypes.byref(value),value.cb):
                need(exited(member),'Process memory read failed for live process');continue
            total+=int(value.peak);seen.append(member)
        finally:k.CloseHandle(handle)
    return total,seen

def output_bytes():
    folders=[LOG,OUT,OLD_LOG,ROOT/'results/research_v44',OLD_P5_LOG,OLD_P5_OUT]+[ROOT/p[k] for p in PARTS.values() for k in ('log','out')]
    return sum(x.stat().st_size for folder in folders if folder.exists() for x in folder.rglob('*') if x.is_file())
def verified_outputs():
    import importlib
    five=importlib.import_module(PARTS['P5']['module']);five_checked=five.verify_complete()
    five_review=read(ROOT/PARTS['P5']['log']/'review.json');five_cold=read(ROOT/PARTS['P5']['log']/'cold.json')
    need(five_review['passed'] and five_cold['passed'],'P5 recovery review/cold incomplete')
    for key,value in five_checked.items():
        need(five_review[key]==value and five_cold[key]==value,'P5 completed/cold field differs '+key)
    need(five_cold['review_sha256']==identity(ROOT/PARTS['P5']['log']/'review.json')['sha256'],'P5 cold binding differs')
    summary=read(OLD_P5_OUT/'summary.json')
    six=importlib.import_module(PARTS['P6']['module']);c,b=six.verified_config(ROOT,six.CONFIG)
    six.verify_calls(ROOT,b,('run','review'));result=six.completed_report(ROOT,b)
    sr=six.load_commit(ROOT/PARTS['P6']['log']/'review.json',b);sc=six.load_commit(ROOT/PARTS['P6']['log']/'cold.json',b)
    need(sr['passed'] and sc['passed'],'P6 review/cold incomplete')
    need(sc['review_sha256']==identity(ROOT/PARTS['P6']['log']/'review.json')['sha256'],'P6 cold binding differs')
    need(sr['result_sha256']==sc['result_sha256']==identity(ROOT/PARTS['P6']['out']/'result.json')['sha256'],'P6 result differs')
    need(sr['completion_sha256']==sc['completion_sha256']==identity(ROOT/PARTS['P6']['out']/'completion.json')['sha256'],'P6 inventory differs')
    folders=[OLD_P5_OUT]+[ROOT/part['out'] for part in PARTS.values()]
    outputs={rel(p):identity(p) for folder in folders for p in folder.rglob('*') if p.is_file()}
    return outputs,summary,result

def finish(c,guard):
    output,five,six=verified_outputs();protected_check(c);need(not ledger()['unknown'],'Unreturned recovery child')
    receipts={s['receipt']:identity(ROOT/s['receipt']) for s in c['stages']}
    text=['双创新点缓存证据审查补充验收 v44r1',now(),'',
        '全部任务总进度45%；实验总进度56.25%。证据审计不新增主交付权重。',
        '当前有限后台任务已完成补充全量验收及新进程冷重入并停止，不等于方法科学可用。',
        '工程故障：原独立复验采用Pillow nearest，在480到720的精确边界浮点累加漂移；主计算的精确整数中心映射符合已规定口径。',
        '恢复只换审查端为独立整数重复映射；原800主结果、原25成功审查、原失败及1条unknown全部保持。',
        '补充775审查含774未开始+1失败替代，实际累计801次审查尝试，800唯一案例完整验证。原v44失败不回写为通过，额外1次属单列工程修复例外；没有新模型/科学样本。',
        'P5可启动新路由训练候选='+str(five['training_candidate_worthwhile'])+'；筛查='+json.dumps(five['screening_gates'],ensure_ascii=False),
        'P5四退化平均补回GT比例='+str(five['degraded_scene_macro_rescue_recall_gain'])+'；匹配对质量一致率='+str(five['quality']['degraded_scene_macro_concordance']),
        'P6机制判断='+json.dumps(six['decision'],ensure_ascii=False),
        'P5预测/质量复用缓存并有限重建known-valid；P6仅训练/校准缓存，不以evaluation行调修复。',
        '0新增神经前向/官方AP/训练。P5原图实际解码/扰动次数及775新审查返回见logs/P5/complementarity_v44r1/review.json。',
        '原2700秒整链、P5累计1800秒、P6累计600秒、2GiB内存与raw帽均保留，旧失败消耗和新准备全部计账。',
        'P5/P6旧科学负结果不变；本次只确定是否值得提出一个有证据且固定预算的修复，不能凭诊断自动采用。P7/P10及周期任务未启动。']
    guard();need(not REPORT.exists(),'Do not overwrite report');REPORT.write_text('\n'.join(text)+'\n',encoding='utf-8')
    proof=dict(at=now(),supplemental_engineering_passed=True,original_v44_failed_preserved=True,
        old_unknown_preserved=1,review_attempts=801,unique_P5_cases_reviewed=800,scientific_adoption=False,
        all_tasks_percent=45,experiment_percent=56.25,report=identity(REPORT),reviewed_outputs=output,receipts=receipts,child_ledger=ledger(),stopped=True)
    guard();write_once(OUT/'closeout.json',proof)
    status('evidence_complete_stopped',old_unknown_preserved=1,report=rel(REPORT))
    supervise('v44r1_evidence_complete_stopped',current_runner=None,current_execution_evidence=rel(OUT/'closeout.json'),summary_report=rel(REPORT),
        final_delivery_index=rel(OUT/'closeout.json'),final_readonly={k:v['log']+'/review.json' for k,v in PARTS.items()},final_cold={k:v['log']+'/cold.json' for k,v in PARTS.items()})
    handoff('v44r1补充完整复验、原P6缓存分析、冷重入及真实报告完成并停止。原v44缩放复验失败和1unknown保留；实际801审查尝试/800唯一验证已分列。全部任务总进度45%、实验总进度56.25%。见docs/双创新点缓存证据审查恢复结果_v44r1.txt；本次未运行修复训练，不代表P5/P6科学通过，不进P7/P10。','complete')
    append(dict(kind='completed_stopped'))

def deliver():
    from scripts.pipeline import project_lock
    started=time.monotonic();c=verify()
    if (OUT/'closeout.json').exists():
        proof=read(OUT/'closeout.json');need(identity(REPORT)==proof['report'],'Report changed')
        for p,v in proof['receipts'].items():need(identity(ROOT/p)==v,'Completed receipt changed '+p)
        all_outputs,_,_=verified_outputs();need(all_outputs==proof['reviewed_outputs'],'Completed artifact identity differs')
        print(json.dumps(dict(already_complete=True,no_new_research=True)));return
    need(not ledger()['unknown'],'Unknown prior calls: inspect processes and preserve evidence before resuming')
    prior=c['preparation_seconds']+sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'))
    part_seconds=dict(c['part_preparation_seconds'])
    for p in LOG.glob('controller_usage_*.json'):
        for part,t in read(p).get('part_seconds',{}).items():part_seconds[part]+=t
    initial_parts=dict(part_seconds);usage=LOG/('controller_usage_'+uuid.uuid4().hex+'.json');active=None
    def guard():
        elapsed=time.monotonic()-started
        active_parts=dict(part_seconds)
        if active:active_parts[active[0]]+=time.monotonic()-active[1]
        atomic(usage,dict(at=now(),seconds=elapsed,prior_seconds=prior,part_seconds={k:active_parts[k]-initial_parts[k] for k in PARTS},status='running'))
        need(prior+elapsed<c['wall_limit_seconds'],'Cumulative2700s limit')
        for part,total in active_parts.items():need(total<c['part_wall_limits'][part],part+' cumulative limit')
        need(output_bytes()<c['total_output_limit'],'Output cap')
        need(not (LOG/'STOP').exists(),'User STOP marker')
    child=None;cid=None
    try:
        status('starting',controller_pid=os.getpid());supervise('v44r1_running')
        for stage in c['stages']:
            guard()
            if (ROOT/stage['receipt']).exists():
                need(read(ROOT/stage['receipt']).get('passed',True),'Existing failed receipt');continue
            cid=uuid.uuid4().hex;active=(stage['part'],time.monotonic());command=[str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-X','utf8','-B',stage['script'],stage['mode']]
            append(dict(kind='child_intent',id=cid,stage=stage,command=command))
            with (LOG/(cid+'.stdout.log')).open('xb') as stdout,(LOG/(cid+'.stderr.log')).open('xb') as stderr:
                child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                append(dict(kind='child_started',id=cid,pid=child.pid));status(stage['part']+'_'+stage['mode'],controller_pid=os.getpid(),child_pid=child.pid)
                while child.poll() is None:
                    guard();rss,pids=process_tree_rss(child.pid)
                    need(rss<c['peak_rss_limit'],'2GiB actual process-tree peak RSS cap')
                    time.sleep(2)
                append(dict(kind='child_returned',id=cid,exit_code=child.returncode,stage=stage))
            part_seconds[active[0]]+=time.monotonic()-active[1];active=None
            need(child.returncode==0 and (ROOT/stage['receipt']).exists(),'Child failed '+stage['part']+'/'+stage['mode'])
            child=None;cid=None
        guard();finish(c,guard)
        atomic(usage,dict(at=now(),seconds=time.monotonic()-started,prior_seconds=prior,part_seconds={k:part_seconds[k]-initial_parts[k] for k in PARTS},status='completed_stopped'))
    except BaseException as exc:
        orphan=False
        if child is not None and child.poll() is None:
            # This is the exact child created by this controller; retain unmatched subcalls.
            killed=subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],cwd=ROOT,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:orphan=True
            append(dict(kind='terminated',id=cid,pid=child.pid,returncode=killed.returncode,orphan_unconfirmed=orphan))
            if child.poll() is not None:append(dict(kind='child_returned',id=cid,exit_code=child.returncode,terminated=True))
        if child is not None and child.poll() is not None and cid in ledger()['unknown']:
            append(dict(kind='child_returned',id=cid,exit_code=child.returncode,observed_during_failure=True))
        if active:part_seconds[active[0]]+=time.monotonic()-active[1]
        atomic(usage,dict(at=now(),seconds=time.monotonic()-started,prior_seconds=prior,part_seconds={k:part_seconds[k]-initial_parts[k] for k in PARTS},status='failed_preserved'))
        append(dict(kind='failure_preserved',error=repr(exc),traceback=traceback.format_exc(),orphan_unconfirmed=orphan))
        status('failed_stopped' if not orphan else 'orphan_unconfirmed',error=repr(exc),unknown_children=ledger()['unknown'])
        supervise('v44r1_failed_stopped',current_runner=None,current_execution_evidence=rel(STATE))
        handoff('v44r1补充验收出现异常，失败日志/未知调用/使用量保留；先核实际进程与锁再处理，禁止重复/扩预算。全部任务总进度45%、实验总进度56.25%。本次未证明方法科学可用，不进P7/P10。','failure');raise

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('freeze','verify','deliver'));args=p.parse_args()
    need(Path.cwd().resolve()==ROOT,'Explicit project cwd required')
    if args.mode=='freeze':freeze()
    elif args.mode=='verify':print(json.dumps(dict(passed=True,files=len(verify()['protected']))))
    else:
        from scripts.pipeline import project_lock
        from scripts.research_cache_delivery_v44 import own_lock as parent_lock
        with parent_lock(),own_lock(),project_lock(ROOT):deliver()
if __name__=='__main__':main()

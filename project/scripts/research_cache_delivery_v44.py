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
LOG=ROOT/'logs/research_v44';OUT=ROOT/'results/research_v44'
CONFIG=ROOT/'configs/research_cache_v44.json'
STATE=ROOT/'runtime/pipeline/research_cache_v44.json'
REPORT=ROOT/'docs/双创新点缓存证据审查结果_v44.txt'
PROTOCOL=ROOT/'docs/双创新点缓存证据审查协议_v44.txt'
ORIGINAL_STATE_SHA='b48eadde3c1f7ea400c48dd206f707ecf804d54f6f018a013e9d662dcc4b2b76'
PARTS={
 'P5':dict(module='scripts.p5_complementarity_v44',script='scripts/p5_complementarity_v44.py',log='logs/P5/complementarity_v44',out='results/P5/complementarity_v44',result='run.json',budget=1800),
 'P6':dict(module='scripts.p6_cache_audit_v44',script='scripts/p6_cache_audit_v44.py',log='logs/P6/cache_audit_v44',out='results/P6/cache_audit_v44',result='result.json',budget=600)}

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
        current_execution_evidence=rel(STATE),current_runner='scripts/research_cache_delivery_v44.py',
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
    p=ROOT/'runtime/pipeline/research_cache_v44.lock';p.parent.mkdir(parents=True,exist_ok=True)
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
    started=time.monotonic()
    with project_lock(ROOT),own_lock():
        need(not CONFIG.exists(),'Already frozen; do not replace')
        checks=read(LOG/'internal_checks.json');review=read(LOG/'independent_review.json')
        for receipt in (checks,review):
            need(receipt['passed'],'Internal/independent review not passed')
            for path,h in receipt['source_sha256'].items():need(identity(ROOT/path)['sha256']==h,'Reviewed code changed '+path)
        protected={}
        for part,info in PARTS.items():
            mod=importlib.import_module(info['module']);paths=mod.dependency_paths()
            bound={rel(ROOT/p if not Path(p).is_absolute() else p):identity(ROOT/p if not Path(p).is_absolute() else p) for p in paths}
            for name in (info['script'],'scripts/research_cache_delivery_v44.py',rel(PROTOCOL),'logs/research_v44/independent_review.json','logs/research_v44/internal_checks.json','runtime/pipeline/state.json'):
                bound[name]=identity(ROOT/name)
            auth=ROOT/info['log']/'authorization.json'
            write_once(auth,dict(at=now(),authorized=True,cache_analysis_only=True,new_model_calls=0,new_AP_calls=0,new_fit_calls=0,
                cumulative_seconds=info['budget'],progress=[45,56.25],protected=bound,
                candidate_use_only=True))
            protected.update(bound);protected[rel(auth)]=identity(auth)
        for name in ('configs/p6_mixture_v43.json','logs/P6/mixture_v43/freeze.json','results/P6/mixture_v43/closeout.json','results/P5/completion_v39/closeout.json'):
            protected[name]=identity(ROOT/name)
        stages=[]
        for part,info in PARTS.items():
            for mode in ('prepare','run','review','cold'):
                marker=ROOT/info['out']/info['result'] if mode=='run' else ROOT/info['log']/(mode+'.json')
                stages.append(dict(part=part,mode=mode,script=info['script'],receipt=rel(marker)))
        component_prep={'P5':sum(read(p)['seconds'] for p in (ROOT/PARTS['P5']['log']).glob('usage_*.json')),
                        'P6':sum(read(p)['seconds'] for p in (ROOT/PARTS['P6']['log']).glob('preparation_usage_*.json'))}
        prep=checks.get('measured_preparation_seconds',0)+sum(component_prep.values())+120.
        c=dict(at=now(),protected=protected,stages=stages,wall_limit_seconds=2700,preparation_seconds=prep,
            part_preparation_seconds=component_prep,part_wall_limits={k:v['budget'] for k,v in PARTS.items()},total_output_limit=192*1024**2,peak_rss_limit=2*1024**3,
            training_or_inference_authorized_by_this_config=False,scientific_progress=[45,56.25])
        protected_check(c);write_once(CONFIG,c);write_once(LOG/'freeze.json',dict(at=now(),config=identity(CONFIG)))
        need(time.monotonic()-started<=120,'Freeze overhead exceeded120s reserve')
        status('frozen_ready');supervise('v44_frozen_ready')
        handoff('用户授权的两创新点路线已进入v44单次缓存证据分析，协议/内部检查/独立审查通过冻结。只复用P5已有实例及质量地图、P6训练与校准缓存；不新推理/AP/训练，不读取独立测试。后台限2700秒，完成复验/冷重入和决策证据报告后停止计算；根只在机制支持时接一次已授权且事前预算明确的修复，不能自生诊断轮。全部任务总进度45%、实验总进度56.25%。入口scripts/research_cache_delivery_v44.py deliver；核实际进程及锁，禁止重复。','freeze')
        print(json.dumps(dict(frozen=True,stages=len(stages),limit_seconds=2700)))

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
    folders=[LOG,OUT]+[ROOT/p[k] for p in PARTS.values() for k in ('log','out')]
    return sum(x.stat().st_size for folder in folders if folder.exists() for x in folder.rglob('*') if x.is_file())
def verified_outputs():
    import importlib
    five=importlib.import_module(PARTS['P5']['module'])
    five_review=read(ROOT/PARTS['P5']['log']/'review.json');five_cold=read(ROOT/PARTS['P5']['log']/'cold.json')
    need(five_review['passed'] and five_cold['passed'],'P5 review/cold not passed')
    need(five_review['reviewed_outputs']==five.output_inventory(),'P5 output identity/inventory differs')
    need(five_cold['review_sha256']==identity(ROOT/PARTS['P5']['log']/'review.json')['sha256'],'P5 cold-review binding differs')
    need(five_review['accounting']==five.accounting()==five_cold['accounting'],'P5 call accounting differs')
    five_result=read(ROOT/PARTS['P5']['out']/'run.json');five_summary=read(ROOT/PARTS['P5']['out']/'summary.json')
    need(five_result['summary_sha256']==identity(ROOT/PARTS['P5']['out']/'summary.json')['sha256'],'P5 summary differs')
    need(five_result['training_candidate_worthwhile']==five_summary['training_candidate_worthwhile']==five_review['training_candidate_worthwhile']==five_cold['training_candidate_worthwhile'],'P5 decision differs')
    six=importlib.import_module(PARTS['P6']['module']);config,binding=six.verified_config(ROOT,six.CONFIG)
    six.verify_calls(ROOT,binding,('run','review'))
    six_result=six.completed_report(ROOT,binding);six_review=six.load_commit(ROOT/PARTS['P6']['log']/'review.json',binding);six_cold=six.load_commit(ROOT/PARTS['P6']['log']/'cold.json',binding)
    need(six_review['passed'] and six_cold['passed'],'P6 review/cold not passed')
    need(six_cold['review_sha256']==identity(ROOT/PARTS['P6']['log']/'review.json')['sha256'],'P6 cold-review binding differs')
    need(six_review['result_sha256']==six_cold['result_sha256']==identity(ROOT/PARTS['P6']['out']/'result.json')['sha256'],'P6 result binding differs')
    need(six_review['completion_sha256']==six_cold['completion_sha256']==identity(ROOT/PARTS['P6']['out']/'completion.json')['sha256'],'P6 completion binding differs')
    all_outputs={rel(p):identity(p) for info in PARTS.values() for p in (ROOT/info['out']).rglob('*') if p.is_file()}
    return all_outputs,five_summary,six_result

def finish(c,guard):
    receipts={}
    for stage in c['stages']:
        p=ROOT/stage['receipt'];record=read(p)
        need(record.get('passed',True),'Failed receipt '+stage['receipt']);receipts[stage['receipt']]=identity(p)
    need(not ledger()['unknown'],'Unknown child calls remain');protected_check(c)
    all_outputs,five_summary,six_result=verified_outputs()
    # Summaries are evidence, not automatic permission to alter models or old gates.
    evidence={}
    for part,info in PARTS.items():
        evidence[part]=dict(result=rel(ROOT/info['out']/info['result']),review=info['log']+'/review.json',cold=info['log']+'/cold.json')
    text=['双创新点缓存证据审查 v44',now(),'',
      '全部任务总进度45%；实验总进度56.25%。本证据任务不增加主交付权重点数。',
      '本次有限后台缓存分析、完整只读复算及独立进程冷重入已完成，计算停止。P6为同实现复算，不声称独立统计实现。',
      'P5：160旧开发帧×5条件已有输出的实例互补及可观测质量区分性。有限读取旧GT/depth/meta并重建已冻结known-valid，不是0原图读取；不是新AP或新模型通过。',
      'P6：只用旧训练/校准缓存识别尺度/混合支持机制；无原始数据重放，不用评价结果选择修复。',
      'P5与P6原科学失败保留；本次候选筛查通过也不等于方法科学采用或独立泛化。',
      'P5值得训练候选='+str(five_summary['training_candidate_worthwhile'])+'；筛查='+json.dumps(five_summary['screening_gates'],ensure_ascii=False),
      'P5四退化宏平均被补回GT比例='+str(five_summary['degraded_scene_macro_rescue_recall_gain'])+'；质量匹配一致率='+str(five_summary['quality']['degraded_scene_macro_concordance']),
      'P6机制判断='+json.dumps(six_result['decision'],ensure_ascii=False),
      '证据路径：'+json.dumps(evidence,ensure_ascii=False,indent=2),
      '根需按预登记条件选定最多一个有机制证据且可给出固定预算的修复；不足则明确停止该候选。禁止盲目重训或扩候选、删候选、改科学阈值追通过。',
      '0新增神经前向，0官方AP调用，0新拟合。缓存分析条目与重算次数单列于组件账本，不能称实际神经前向次数。',
      '未启动P7/P10或周期任务。']
    guard();need(not REPORT.exists(),'Existing report must not be overwritten');REPORT.write_text('\n'.join(text)+'\n',encoding='utf-8')
    proof=dict(at=now(),engineering_passed=True,evidence_analysis_complete=True,scientific_adoption=False,root_repair_decision_pending=True,
      all_tasks_percent=45,experiment_percent=56.25,receipts=receipts,reviewed_outputs=all_outputs,report=identity(REPORT),child_ledger=ledger(),stopped=True)
    guard();write_once(OUT/'closeout.json',proof)
    status('evidence_complete_stopped',report=rel(REPORT),root_repair_decision_pending=True)
    supervise('v44_evidence_complete_stopped',current_runner=None,current_execution_evidence=rel(OUT/'closeout.json'),summary_report=rel(REPORT),
        final_readonly={k:v['log']+'/review.json' for k,v in PARTS.items()},final_cold={k:v['log']+'/cold.json' for k,v in PARTS.items()},final_delivery_index=rel(OUT/'closeout.json'))
    handoff('v44单次缓存证据及完整复验/冷重入完成，后台计算已停止；修复选择需由根读取实际结果后按原授权作有限决定，不能把缓存筛查称为科学采用。全部任务总进度45%、实验总进度56.25%。结果docs/双创新点缓存证据审查结果_v44.txt；P5/P6旧失败保留，未启动P7/P10或周期任务。','complete')
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
        status('starting',controller_pid=os.getpid());supervise('v44_running')
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
        supervise('v44_failed_stopped',current_runner=None,current_execution_evidence=rel(STATE))
        handoff('v44缓存分析出现异常，失败日志/未知调用/使用量保留；先核实际进程与锁再处理，禁止重复/扩预算。全部任务总进度45%、实验总进度56.25%。本次未证明方法科学可用，不进P7/P10。','failure');raise

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('freeze','verify','deliver'));args=p.parse_args()
    need(Path.cwd().resolve()==ROOT,'Explicit project cwd required')
    if args.mode=='freeze':freeze()
    elif args.mode=='verify':print(json.dumps(dict(passed=True,files=len(verify()['protected']))))
    else:
        from scripts.pipeline import project_lock
        with own_lock(),project_lock(ROOT):deliver()
if __name__=='__main__':main()

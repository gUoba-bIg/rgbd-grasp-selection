"""Finite P5 modal-fusion delivery. Frozen inputs, conditional full evaluation, stop."""
import os, sys, json, time, uuid, hashlib, shutil, subprocess, traceback, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
LOG = ROOT / 'logs/P5/fusion_v38'
OUT = ROOT / 'results/P5/fusion_v38'
CONF = ROOT / 'configs/p5_fusion_v38.json'
STATE = ROOT / 'runtime/pipeline/p5_fusion_delivery_v38.json'

def now(): return datetime.now().astimezone().isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def need(v, message):
    if not v: raise RuntimeError(message)
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def stat(p):
    s=Path(p).stat()
    return dict(sha256=sha(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)
def atomic(p, v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with t.open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    start=time.monotonic()
    while True:
        try:os.replace(t,p);return
        except PermissionError as e:
            if getattr(e,'winerror',None) not in (5,32) or time.monotonic()-start>2:raise
            time.sleep(.05)
def event(kind,**values):
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'controller.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(at=now(),kind=kind,**values),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def records():
    p=LOG/'controller.jsonl'
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
def ledger():
    rows=records();aa=[r for r in rows if r['kind']=='child_intent'];bb=[r for r in rows if r['kind']=='child_returned']
    a={r['call_id']:r for r in aa};b={r['call_id']:r for r in bb}
    need(len(a)==len(aa) and len(b)==len(bb),'Duplicate child identity')
    need(set(b)<=set(a),'Child return without intent')
    return a,b,sorted(set(a)-set(b))
def status(phase,**values):
    atomic(STATE,dict(at=now(),phase=phase,current_part='P5_original_modal_fusion_v38',
        all_tasks_percent=41,experiment_percent=51.25,progress_basis='Fixed restored-scope delivery weights; not scientific success or measured hours',
        P10_still_stopped=True,P6_started=False,P11_started=False,recurring_automation=False,stop_after='P5_v38',**values))
def verify():
    c=read(CONF);need(sha(CONF)==read(LOG/'freeze.json')['config_sha256'],'Frozen controller configuration changed')
    for path,s in c['protected'].items():need(stat(ROOT/path)==s,'Frozen source or parent changed: '+path)
    return c
def condition(stage):
    cond=stage.get('condition')
    if not cond:return True
    value=read(ROOT/cond['path'])
    for key in cond['keys']:value=value[key]
    return value==cond['equals']

def run_child(stage, guard):
    from scripts.pipeline import project_lock
    with project_lock(ROOT):pass
    cid=uuid.uuid4().hex
    cmd=[str(ROOT/stage['python']),'-X','utf8','-B',stage['script']]+stage['args']
    event('child_intent',call_id=cid,stage=stage['name'],command=cmd)
    with (LOG/(stage['name']+'_'+cid+'.stdout.log')).open('xb') as out,(LOG/(stage['name']+'_'+cid+'.stderr.log')).open('xb') as err:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        event('child_started',call_id=cid,stage=stage['name'],controller_pid=os.getpid(),child_pid=child.pid)
        status(stage['name'],controller_pid=os.getpid(),child_pid=child.pid)
        try:
            while child.poll() is None:guard();time.sleep(2)
        except BaseException:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False)
            # No returned event: termination is not a scientific return.
            raise
    event('child_returned',call_id=cid,stage=stage['name'],exit_code=child.returncode)
    need(child.returncode==0,'Engineering failure; logs preserved: '+stage['name'])
    need((ROOT/stage['done']).exists(),'Child returned without expected receipt '+stage['name'])
    read(ROOT/stage['done'])

def finish(c):
    need(not ledger()[2],'Unknown child returns require audit')
    for stage in c['stages']:
        if condition(stage):need((ROOT/stage['done']).exists(),'Missing completed stage '+stage['name'])
    reviews={name:read(ROOT/path) for name,path in c['reports']['reviews'].items()}
    need(all(x.get('passed') is True for x in reviews.values()),'Independent review and actual cold entry must pass')
    screen=read(ROOT/c['reports']['screen'])
    full_path=ROOT/c['reports']['full']
    full=read(full_path) if full_path.exists() else None
    success=bool(full and full.get(c['reports']['pass_field']) is True)
    proof=dict(at=now(),engineering_passed=True,screen=screen,full=full,
        scientific_adoption_supported=success,complete_original_method_demonstrated=False,
        independent_confirmation=False,stopped=True,P10_still_stopped=True,P6_started=False,
        all_tasks_percent=41,experiment_percent=51.25,
        progress_note='No additional credit before root maps actual completed work to the frozen delivery criteria',
        reviews={p:stat(ROOT/p) for p in c['reports']['reviews'].values()},
        returned_children=len(ledger()[1]),unknown_children=ledger()[2])
    path=OUT/'closeout.json'
    if path.exists():need(read(path)['reviews']==proof['reviews'],'Existing closeout differs')
    else:atomic(path,proof)
    handoff=ROOT/'docs/实验自动续接说明.txt'
    backup=LOG/'handoff_before_controller_close.txt'
    if not backup.exists():shutil.copy2(handoff,backup)
    handoff.write_text(f'最新：{now()} P5 v38有限链路已完成并停止，完整开发评价完成={full is not None}，科学采用={success}，独立复验与真实冷重入通过。根最终核对与结果说明待收尾；不得继续P6/P10或周期任务。全部任务总进度41%、实验总进度51.25%暂未增加，待根按事前1+2+1交付口径核实。证据results/P5/fusion_v38/closeout.json。\n\n历史：\n'+handoff.read_text(encoding='utf-8-sig'),encoding='utf-8')
    supervision=ROOT/'runtime/pipeline/part_level_supervision.json'
    backup=LOG/'supervision_before_controller_close.json'
    if not backup.exists():shutil.copy2(supervision,backup)
    metadata=read(supervision)
    metadata.update(recorded_at=now(),current_execution_state='P5_v38_completed_stopped_pending_root_summary',
        waiting_for_user_next_part_confirmation=True,current_runner=None,automatic_resume_authorized=False,
        current_execution_evidence='results/P5/fusion_v38/closeout.json',remaining_research_updates_cap=0)
    atomic(supervision,metadata)
    status('completed_stopped',engineering_passed=True,scientific_adoption_supported=success,
        full160_evaluation_completed=full is not None,report='results/P5/fusion_v38/closeout.json')
    event('completed_stopped',scientific_adoption_supported=success,full160_evaluation_completed=full is not None)

def deliver():
    c=verify();need(not ledger()[2],'Unknown child: inspect process and preserve evidence before recovery')
    if (OUT/'closeout.json').exists():
        need(read(STATE)['phase']=='completed_stopped','Closeout and controller state inconsistent')
        proof=read(OUT/'closeout.json')
        for path,identity in proof['reviews'].items():
            need(stat(ROOT/path)==identity,'Completed review receipt changed '+path)
            receipt=read(ROOT/path);need(receipt['passed'],'Completed review not passed')
            for artifact,expected in receipt.get('reviewed_outputs',{}).items():
                need(stat(ROOT/artifact)==expected,'Completed output changed '+artifact)
        need(read(ROOT/c['reports']['screen'])==proof['screen'],'Completed screen decision changed')
        full=ROOT/c['reports']['full']
        need((read(full) if full.exists() else None)==proof['full'],'Completed full decision changed')
        expected=bool(proof['full'] and proof['full'].get(c['reports']['pass_field']) is True)
        need(expected==proof['scientific_adoption_supported'],'Completed scientific decision inconsistent')
        need(read(ROOT/c['reports']['reviews']['cold'])['scientific_usable']==expected,'Cold decision inconsistent')
        print(json.dumps(dict(already_complete=True,no_new_research=True)));return
    start=time.monotonic();usage=LOG/('controller_usage_'+uuid.uuid4().hex+'.json')
    old=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'))+float(c['preflight_compute_seconds'])
    def guard():
        elapsed=time.monotonic()-start;atomic(usage,dict(at=now(),seconds=elapsed,state='running'))
        need(old+elapsed<c['controller_wall_seconds'],'Cumulative controller budget exhausted')
        need(not (LOG/'STOP').exists(),'Explicit user stop marker exists')
    try:
        for stage in c['stages']:
            verify();guard()
            if not condition(stage):
                event('stage_not_applicable',stage=stage['name'],reason=stage['condition']);continue
            marker=ROOT/stage['done']
            if marker.exists():read(marker);event('stage_reused',stage=stage['name'],marker=stage['done']);continue
            run_child(stage,guard)
        verify();guard();finish(c)
        atomic(usage,dict(at=now(),seconds=time.monotonic()-start,state='completed_stopped'))
    except BaseException:
        atomic(usage,dict(at=now(),seconds=time.monotonic()-start,state='failed_preserved'))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('verify','deliver'));args=parser.parse_args()
    if args.mode=='verify':print(json.dumps(dict(passed=True,protected=len(verify()['protected']))));sys.exit(0)
    import msvcrt
    with (ROOT/'runtime/pipeline/p5_fusion_v38.lock').open('a+b') as f:
        f.seek(0,2)
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:deliver()
        except BaseException as e:
            event('failure_preserved',error=repr(e),traceback=traceback.format_exc());status('engineering_failure_preserved',error=repr(e));raise
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

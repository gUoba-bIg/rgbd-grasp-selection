"""One explicitly authorized full P5 evaluation of immutable v38 weights, then stop."""
import sys, os, json, time, uuid, shutil, subprocess, traceback, argparse
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.p5_fusion_delivery_v38 import read,sha,stat,atomic,need
LOG=ROOT/'logs/P5/completion_v39'
OUT=ROOT/'results/P5/completion_v39'
CONF=ROOT/'configs/p5_completion_v39.json'
STATE=ROOT/'runtime/pipeline/p5_completion_v39.json'
def now():return datetime.now().astimezone().isoformat()
def event(kind,**values):
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'controller.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(at=now(),kind=kind,**values),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def rows():
    p=LOG/'controller.jsonl'
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
def ledger():
    r=rows();aa=[x for x in r if x['kind']=='child_intent'];bb=[x for x in r if x['kind']=='child_returned']
    a={x['call_id']:x for x in aa};b={x['call_id']:x for x in bb}
    need(len(a)==len(aa) and len(b)==len(bb) and set(b)<=set(a),'Child identity ledger inconsistent')
    return a,b,sorted(set(a)-set(b))
def status(phase,**values):
    atomic(STATE,dict(at=now(),phase=phase,current_part='P5_complete_existing_fusion_v39',
        all_tasks_percent=42,experiment_percent=52.5,P6_started=False,P10_still_stopped=True,
        recurring_automation=False,stop_after='P5_v39',**values))
def verify():
    c=read(CONF);need(sha(CONF)==read(LOG/'freeze.json')['config_sha256'],'Frozen v39 config changed')
    for name,value in c['protected'].items():need(stat(ROOT/name)==value,'Frozen input/source/output changed '+name)
    need(c['max_new_optimizer_intents']==0 and c['max_new_forward_intents']==3920 and c['max_new_AP_intents']==140,'Budget differs')
    need(c['controller_wall_seconds']==3600,'Budget scope differs')
    return c
def validate_completion(c):
    need(not ledger()[2],'Missing controller return requires audit')
    for stage in c['stages']:need((ROOT/stage['done']).exists(),'Missing '+stage['name'])
    review=read(LOG/'review.json');cold=read(LOG/'cold.json')
    need(review['passed'] and cold['passed'],'Review or cold entry not accepted')
    for path,identity in review['reviewed_outputs'].items():need(stat(ROOT/path)==identity,'Reviewed output changed '+path)
    current={p.relative_to(ROOT).as_posix():stat(p) for p in OUT.rglob('*') if p.is_file() and p!=OUT/'closeout.json'}
    # Component review may additionally include immutable inherited artifacts.
    reviewed={p:v for p,v in review['reviewed_outputs'].items() if p.startswith('results/P5/completion_v39/')}
    need(current==reviewed,'New output inventory changed after review')
    d=read(OUT/'full/decision.json')
    need(cold['scientific_usable']==d['scientific_usable'],'Cold decision differs')
    from scripts import complete_p5_v39 as component
    calls=component.accounting()
    for family,count in [('forward',3920),('optimizer',0),('official_AP',140)]:
        need(calls[family]['intents']==count and calls[family]['known_returns']==count and not calls[family]['unknown_ids'],'Incomplete actual '+family)
    return d,calls
def finish(c,guard):
    decision,calls=validate_completion(c)
    guard()
    proof=dict(at=now(),engineering_passed=True,full160_evaluation_completed=True,
        scientific_adoption_supported=decision['scientific_usable'],decision=decision,
        complete_original_method_demonstrated=False,independent_confirmation=False,
        all_tasks_percent=42,experiment_percent=52.5,progress_credit_pending_root_review=True,
        new_calls=calls,parent_recorded_compute_seconds=c['parent_recorded_compute_seconds'],
        reviews={str(p.relative_to(ROOT)):stat(p) for p in [LOG/'review.json',LOG/'cold.json']},
        stopped=True,P6_started=False,P10_still_stopped=True,recurring_automation=False)
    atomic(OUT/'closeout.json',proof)
    status('completed_stopped',engineering_passed=True,scientific_adoption_supported=decision['scientific_usable'],full160_evaluation_completed=True)
    metadata=ROOT/'runtime/pipeline/part_level_supervision.json'
    shutil.copy2(metadata,LOG/'supervision_before_close.json')
    v=read(metadata);v.update(recorded_at=now(),current_execution_state='P5_v39_completed_stopped_pending_root_summary',
        current_runner=None,automatic_resume_authorized=False,waiting_for_user_next_part_confirmation=True,
        remaining_research_updates_cap=0,current_execution_evidence='results/P5/completion_v39/closeout.json')
    atomic(metadata,v)
    event('completed_stopped',scientific_adoption_supported=decision['scientific_usable'])
def deliver():
    start=time.monotonic()
    c=verify();need(not ledger()[2],'Unknown child: audit processes/logs, never erase unknown')
    if (OUT/'closeout.json').exists():
        decision,calls=validate_completion(c);proof=read(OUT/'closeout.json')
        need(read(STATE)['phase']=='completed_stopped','Completed state differs')
        need(proof['decision']==decision and proof['new_calls']==calls,'Completed proof differs')
        for path,identity in proof['reviews'].items():need(stat(ROOT/path)==identity,'Completed receipt changed')
        print(json.dumps(dict(already_complete=True,no_new_research=True)));return
    usage=LOG/('controller_usage_'+uuid.uuid4().hex+'.json')
    old=sum(read(p)['seconds'] for p in LOG.glob('controller_usage_*.json'))+c['preflight_compute_seconds']
    def guard():
        elapsed=time.monotonic()-start;atomic(usage,dict(at=now(),seconds=elapsed,state='running'))
        need(old+elapsed<c['controller_wall_seconds'],'Cumulative v39 controller budget exhausted')
        need(not (LOG/'STOP').exists(),'User stop marker')
    try:
        from scripts.pipeline import project_lock
        for stage in c['stages']:
            verify();guard()
            if (ROOT/stage['done']).exists():event('stage_reused',stage=stage['name']);continue
            with project_lock(ROOT):pass
            cid=uuid.uuid4().hex
            cmd=[str(ROOT/stage['python']),'-X','utf8','-B','scripts/complete_p5_v39.py',stage['name']]
            event('child_intent',call_id=cid,stage=stage['name'],command=cmd)
            with (LOG/(stage['name']+'_'+cid+'.stdout.log')).open('xb') as out,(LOG/(stage['name']+'_'+cid+'.stderr.log')).open('xb') as err:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                event('child_started',call_id=cid,stage=stage['name'],controller_pid=os.getpid(),child_pid=child.pid)
                status(stage['name'],controller_pid=os.getpid(),child_pid=child.pid)
                try:
                    while child.poll() is None:guard();time.sleep(2)
                except BaseException:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False)
                    raise
            event('child_returned',call_id=cid,stage=stage['name'],exit_code=child.returncode)
            need(child.returncode==0 and (ROOT/stage['done']).exists(),'Stage failed/missing receipt; preserve logs '+stage['name'])
        verify();guard();finish(c,guard)
        atomic(usage,dict(at=now(),seconds=time.monotonic()-start,state='completed_stopped'))
    except BaseException:
        atomic(usage,dict(at=now(),seconds=time.monotonic()-start,state='failed_preserved'));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('verify','deliver'));args=p.parse_args()
    if args.mode=='verify':print(json.dumps(dict(passed=True,protected=len(verify()['protected']))));sys.exit(0)
    import msvcrt
    with (ROOT/'runtime/pipeline/p5_completion_v39.lock').open('a+b') as f:
        f.seek(0,2)
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:deliver()
        except BaseException as e:event('failure_preserved',error=repr(e),traceback=traceback.format_exc());status('engineering_failure_preserved',error=repr(e));raise
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

"""Isolated recovery metadata. All frozen v51 code and existing results stay unchanged."""
from scripts import p56_common_v51 as old
from scripts.p56_common_v51 import ROOT,read,identity,atomic,once,append,events,ledger,now,need,rel,GROUPS,SCENES,CONDITIONS
import os,time,uuid,shutil
from contextlib import contextmanager
LOG=ROOT/'logs/joint_validation_v51r1'
OUT=ROOT/'results/joint_validation_v51r1'
CONFIG=ROOT/'configs/p56_joint_v51r1.json'
STATE=ROOT/'runtime/pipeline/p56_joint_v51r1.json'
PROTOCOL=ROOT/'docs/P5-P6联合验证分段恢复协议_v51r1.txt'
REPORT=ROOT/'docs/P5-P6联合候选选择验证结果_v51r1.txt'
SOURCES=['scripts/p56_recovery_common_v51r1.py','scripts/p56_recovery_v51r1.py',
         'scripts/p56_official_recovery_v51r1.py','tests/test_p56_official_recovery_v51r1.py',
         'tests/test_p56_recovery_controller_v51r1.py']

def status(phase,**kw):
    atomic(STATE,dict(at=now(),phase=phase,current_part='P5_P6_joint_use_v51r1',
        all_tasks_percent=45,experiment_percent=56.25,P5_original_scientific_passed=False,
        P6_original_scientific_passed=False,original_v51_status='failure_preserved_stopped',
        historical_unknown_preserved=True,expansion_started=False,recurring_automation=False,**kw))

@contextmanager
def own_lock():
    import msvcrt
    p=ROOT/'runtime/pipeline/p56_joint_v51r1.lock'
    with p.open('a+b') as f:
        f.seek(0,2)
        if not f.tell():f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def verify(full=False):
    need(Path_cwd()==ROOT,'Explicit project directory required')
    c=read(CONFIG);need(identity(CONFIG)==read(LOG/'freeze.json')['config'],'Recovery freeze changed')
    for name,expected in c['protected'].items():
        p=ROOT/name;st=p.stat()
        need(st.st_size==expected['bytes'] and st.st_mtime_ns==expected['mtime_ns'],'Protected metadata differs '+name)
        if full:need(identity(p)==expected,'Protected original content differs '+name)
    need(identity(old.CONFIG)==c['base_config_identity'],'Original scientific config changed')
    return c

def Path_cwd():
    from pathlib import Path
    return Path.cwd().resolve()

def handoff(message):
    p=ROOT/'docs/实验自动续接说明.txt';folder=LOG/'handoff_history';folder.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,folder/(uuid.uuid4().hex+'_'+p.name))
    previous=p.read_text('utf-8-sig')
    p.write_text('最新：'+now()+' '+message+'\n\n历史：\n'+previous,encoding='utf-8')

def output_bytes():
    return sum(p.stat().st_size for d in (old.LOG,old.OUT,LOG,OUT) if d.exists() for p in d.rglob('*') if p.is_file())

def remaining_seconds(c,elapsed):
    return c['formal_wall_seconds']-c['original_formal_seconds']-elapsed

def original_feature_proof():
    proof=read(old.LOG/'feature_review.json')
    need(proof['passed'] and proof['complete_numeric_replay'] and proof['cases']==50,'Original full feature replay missing')
    need(len(proof['records'])==50 and {r['key'] for r in proof['records']}=={old.case_key(s,c) for s in SCENES for c in CONDITIONS},'Original replay case coverage differs')
    for r in proof['records']:
        folder=old.OUT/'cases'/r['key']
        need(identity(folder/'ranking.json')==r['ranking'],'Original ranking changed')
        need(identity(folder/'uncertainty_inputs.npz')==r['uncertainty'],'Original feature arrays changed')
    for name in ('feature_calls.jsonl','neural_calls.jsonl'):
        value=ledger(old.LOG/name)
        need(value['intents']==100 and value['known_returns']==100 and not value['unknown'],'Original feature/NN evidence incomplete')
    return dict(proof=identity(old.LOG/'feature_review.json'),feature=ledger(old.LOG/'feature_calls.jsonl'),neural=ledger(old.LOG/'neural_calls.jsonl'))

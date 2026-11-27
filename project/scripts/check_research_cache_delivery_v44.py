"""Small operational checks; no data scoring or research calls."""
import sys,time,json,uuid,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts import research_cache_delivery_v44 as r
from scripts.pipeline import project_lock

def main():
    started=time.monotonic();checks=[]
    # Preserve unknown return semantics under interrupted work.
    actual_log=r.LOG;r.LOG=ROOT/'runtime/checks'/('cache_v44_'+uuid.uuid4().hex)
    r.append(dict(kind='child_intent',id='unknown-demo'))
    assert r.ledger()==dict(intents=1,known_returns=0,unknown=['unknown-demo']);checks.append('unreturned intent remains unknown')
    r.append(dict(kind='child_returned',id='unknown-demo',exit_code=0))
    assert r.ledger()==dict(intents=1,known_returns=1,unknown=[]);checks.append('known return paired once')
    r.LOG=actual_log
    with r.own_lock():
        blocked=False
        try:
            with r.own_lock():pass
        except OSError:blocked=True
        assert blocked
    checks.append('second controller lock rejected')
    with project_lock(ROOT):pass
    checks.append('project lock actually available')
    child=subprocess.Popen([str(ROOT/'runtime/envs/vision/Scripts/python.exe'),'-X','utf8','-B','-c','import time;time.sleep(4)'],cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        time.sleep(.75);rss,members=r.process_tree_rss(child.pid)
        assert child.pid in members and len(members)>=2 and rss>0
        checks.append('actual venv redirector plus Python descendant memory observed')
    finally:
        child.wait(timeout=10)
    gone_rss,gone_members=r.process_tree_rss(child.pid)
    assert gone_rss==0 and not gone_members
    checks.append('exited short child is not reported as unreadable live process')
    assert r.identity(ROOT/'runtime/pipeline/state.json')['sha256']==r.ORIGINAL_STATE_SHA
    checks.append('original pipeline state hash unchanged')
    # Syntax only, no imports of experiment workers and no cached numeric calculations.
    paths=['scripts/research_cache_delivery_v44.py','scripts/check_research_cache_delivery_v44.py']
    for name in paths:compile((ROOT/name).read_text(encoding='utf-8-sig'),name,'exec')
    checks.append('standalone entry source compiles')
    result=dict(passed=True,checks=checks,measured_preparation_seconds=time.monotonic()-started,
        new_model_calls=0,new_AP_calls=0,new_fit_calls=0,source_sha256={name:r.identity(ROOT/name)['sha256'] for name in paths})
    target=r.LOG/'internal_checks.json'
    if target.exists():
        old=r.read(target);result['measured_preparation_seconds']+=old['measured_preparation_seconds']
        r.backup(target,'pre_freeze_code_correction');r.atomic(target,result)
    else:r.write_once(target,result)
    print(json.dumps(result))
if __name__=='__main__':main()

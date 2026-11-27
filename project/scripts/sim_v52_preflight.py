"""Negative safety tests using synthetic bookkeeping only; no physics or process launch."""
import ast
import copy
import json
import sys
import uuid
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import sim_v52_control as c
from scripts.sim_v52_common import ROOT, LOG, write, identity, now

def run():
    passed=[]
    for path in (ROOT/'scripts').glob('sim_v52_*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'))
    passed.append('syntax_all_v52')
    for count in (-1,0,49):
        try:c.run(count)
        except ValueError:pass
        else:raise AssertionError('Bad count accepted')
    passed.append('negative_zero_excess_count_rejected_before_launch')
    saved=(c.ROOT,c.LOG,c.CONFIG,c.LEDGER)
    fixture=LOG/'safety_fixtures'/uuid.uuid4().hex[:8]
    fixture.mkdir(parents=True)
    try:
        c.ROOT=fixture;c.LOG=fixture/'logs';c.CONFIG=fixture/'config.json';c.LEDGER=fixture/'episodes.jsonl'
        cfg={'bindings':{},'slots':[{'slot_id':0,'kind':'nominal'}]}
        write(c.CONFIG,cfg)
        frozen=identity(c.CONFIG);write(c.LOG/'freeze.json',{'config':frozen})
        c.verify_frozen(cfg)
        changed=copy.deepcopy(cfg);changed['attempts']=999
        write(c.CONFIG,changed)
        try:c.verify_frozen(changed)
        except RuntimeError:pass
        else:raise AssertionError('Config tampering not rejected')
        passed.append('frozen_config_tampering_rejected')
        # Recreate a new synthetic freeze receipt, not a production protocol.
        write(c.CONFIG,cfg);frozen=identity(c.CONFIG);write(c.LOG/'freeze.json',{'config':frozen})
        intent=dict(event='intent',attempt_id='fixture',slot_id=0,directory='episode',config=frozen)
        def ledger(rows):
            c.LEDGER.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        ledger([intent])
        assert c.verified_ledger(cfg)[3]==['fixture']
        passed.append('missing_return_blocks_resume')
        ret=dict(event='returned',attempt_id='fixture',result_status='unknown_episode_result',result=None,exit_code=0,termination=None)
        ledger([intent,ret]);assert c.verified_ledger(cfg)[3]==['fixture']
        passed.append('returned_without_result_still_unknown')
        result={'slot':cfg['slots'][0],'config_identity':frozen,'outcome':'success','success':True}
        write(fixture/'episode/result.json',result)
        ret.update(result_status='known_result',result=identity(fixture/'episode/result.json'))
        ledger([intent,ret]);assert not c.verified_ledger(cfg)[3]
        passed.append('fully_bound_result_recognized')
        ret['exit_code']=1;ledger([intent,ret]);assert c.verified_ledger(cfg)[3]==['fixture']
        passed.append('nonzero_exit_not_admitted')
        ret['exit_code']=0;ret['termination']={'survivors':[123]};ledger([intent,ret]);assert c.verified_ledger(cfg)[3]==['fixture']
        passed.append('terminated_or_orphan_result_not_admitted')
        ret['termination']=None;ledger([intent,ret]);result['success']=False;write(fixture/'episode/result.json',result)
        assert c.verified_ledger(cfg)[3]==['fixture']
        passed.append('modified_result_hash_rejected')
    finally:
        c.ROOT,c.LOG,c.CONFIG,c.LEDGER=saved
    target=LOG/'safety_preflight.json'
    write(target,dict(at=now(),passed=True,checks=passed,physics_steps=0,physical_episodes=0,process_launches=0,
                     scope='Synthetic negative bookkeeping tests only',sources={str(p.relative_to(ROOT)):identity(p) for p in [Path(__file__),ROOT/'scripts/sim_v52_control.py']}),once=True)
    print(json.dumps({'passed':len(passed),'physics_steps':0}))

if __name__=='__main__':run()

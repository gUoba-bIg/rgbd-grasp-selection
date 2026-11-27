"""Bounded R3 controller. No R4 command, scheduling, training, or silent retries."""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sim_v52_common import *

ORIGINAL_CONFIG=CONFIG
CONFIG=ROOT/'configs/simulation_v52r1.json'
FREEZE=LOG/'recovery_v52r1/freeze.json'

PYTHON=ROOT/'runtime/simulation_v52/venv/Scripts/python.exe'
LEDGER=LOG/'episodes.jsonl'

def source_bindings():
    paths=list((ROOT/'scripts').glob('sim_v52_*.py'))+list((ROOT/'scripts').glob('sim_v52r1_*.py'))
    paths += [ORIGINAL_CONFIG,ROOT/'docs/机器人仿真工程适配修复_v52r1.txt']
    paths += [ROOT/'runtime/simulation_v52/attempt001_static_geometry.json']
    paths += [ROOT/'docs/机器人仿真工程准入协议_v52.txt']
    for directory in ('data/simulation_v52/robot','data/simulation_v52/objects'):
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and 'preparation_history' not in p.parts]
    return {relative(p):identity(p) for p in sorted(set(paths))}

def freeze():
    with lock(ROOT/'runtime/pipeline/execution.lock'),lock(ROOT/'runtime/pipeline/simulation_v52.lock'):
        if CONFIG.exists(): raise FileExistsError('Recovery already frozen')
        from scripts import sim_v52_control as old
        base=read(ORIGINAL_CONFIG)
        old.verify_frozen(base)
        intents,returns,rows,unknown=old.verified_ledger(base)
        assert len(intents)==1 and not unknown and rows[0]['result']['outcome']=='planned_table_collision'
        verify_history()
        cfg=json.loads(json.dumps(base))
        cfg.update(version='52r1',at=now(),parent_config=identity(ORIGINAL_CONFIG),
                   repair='Known robot downward collision envelope in engineering proposals only; no P3 pose modification',
                   bindings=source_bindings(),historical_attempts=1,remaining_attempts_max=47)
        cfg['engineering_repair']=dict(downward_collision_envelope_m=.2356573495,
            evidence='runtime/simulation_v52/attempt001_static_geometry.json',
            applies_to='Observation-derived engineering proposals only, before negative-control construction',
            retained_initial_failure_slot=0,repeat_failed_slot=False)
        for key in ('slots','criteria','physics','limits','motion','negative_controls','objects','home_joints'):
            assert cfg[key]==base[key],key
        write(CONFIG,cfg,once=True)
        write(FREEZE,dict(at=now(),config=identity(CONFIG),source_count=len(cfg['bindings']),
                         original_config=identity(ORIGINAL_CONFIG),historical_attempts=1,
                         historical_unchanged=verify_history()),once=True)
        status('R3_engineering_recovery_frozen',engineering_revision='52r1',engineering_attempts_started=1,engineering_attempts_cap=48)
        print(json.dumps({'recovery_frozen':True,'historical_attempts':1,'remaining':47}))

def verify_frozen(cfg):
    if identity(CONFIG)!=read(FREEZE)['config']:
        raise RuntimeError('Frozen config identity changed')
    changed=[name for name,value in cfg['bindings'].items() if not (ROOT/name).is_file() or identity(ROOT/name)!=value]
    if changed:
        raise RuntimeError('Unreviewed v52 change: '+str(changed[:10]))

def verified_ledger(cfg):
    ledger=events(LEDGER)
    intents=[e for e in ledger if e['event']=='intent']
    returned=[e for e in ledger if e['event']=='returned']
    if len({e['attempt_id'] for e in intents})!=len(intents) or len({e['attempt_id'] for e in returned})!=len(returned):
        raise RuntimeError('Duplicate intent or return')
    returns={e['attempt_id']:e for e in returned}
    values=[];unknown=[]
    for intent in intents:
        ret=returns.get(intent['attempt_id'])
        path=ROOT/intent['directory']/'result.json'
        valid=(ret is not None and ret.get('result_status')=='known_result' and ret.get('result') is not None
               and ret.get('exit_code')==0 and ret.get('termination') is None and path.is_file()
               and identity(path)==ret['result'] and intent['config'] in (read(FREEZE)['config'],identity(ORIGINAL_CONFIG)))
        result=read(path) if valid else None
        if result is not None and (result.get('slot')!=cfg['slots'][intent['slot_id']] or result.get('config_identity')!=intent['config']):
            valid=False
        if not valid:
            unknown.append(intent['attempt_id'])
            result=dict(success=False,outcome='unknown_episode_result',slot=cfg['slots'][intent['slot_id']])
        values.append(dict(attempt_id=intent['attempt_id'],slot_id=intent['slot_id'],directory=intent['directory'],result=result))
    if set(returns)-{e['attempt_id'] for e in intents}:
        raise RuntimeError('Return lacks intent')
    return intents,returns,values,unknown

def stop_child(child):
    import psutil
    children=psutil.Process(child.pid).children(recursive=True) if child.poll() is None else []
    for proc in reversed(children):
        try:proc.kill()
        except psutil.Error:pass
    if child.poll() is None:
        child.kill()
    child.wait(timeout=20)
    survivors=[proc.pid for proc in children if proc.is_running()]
    return survivors

def run(count):
    import psutil
    if not 1<=count<=48:
        raise ValueError('count must be an integer in 1..48')
    with lock(ROOT/'runtime/pipeline/execution.lock'),lock(ROOT/'runtime/pipeline/simulation_v52.lock'):
        cfg=read(CONFIG);verify_frozen(cfg);verify_history()
        intents,returns,_,unknown=verified_ledger(cfg)
        if unknown:
            raise RuntimeError('Unreturned attempts preserved; inspect orphans before any resume')
        existing_slots={e['slot_id'] for e in intents}
        cumulative=sum(e['wall_seconds'] for e in returns.values())
        pending=[s for s in cfg['slots'] if s['slot_id'] not in existing_slots]
        if len(intents)>=cfg['limits']['attempts']:
            return report()
        for slot in pending[:min(count,cfg['limits']['attempts']-len(intents))]:
            if (LOG/'STOP').exists():
                status('user_stop_preserved',engineering_attempts_started=len(intents))
                break
            assert cumulative<cfg['limits']['cumulative_wall_seconds']
            assert shutil.disk_usage(ROOT).free>cfg['limits']['free_disk_bytes']
            attempt_id=uuid.uuid4().hex
            directory=OUT/'episodes'/f"{len(intents)+1:03d}_slot_{slot['slot_id']:02d}_{attempt_id[:8]}"
            directory.mkdir(parents=True,exist_ok=False)
            command=[str(PYTHON),'-B','scripts/sim_v52r1_episode.py','--slot',str(slot['slot_id']),'--attempt-dir',relative(directory)]
            intent=dict(event='intent',at=now(),attempt_id=attempt_id,slot_id=slot['slot_id'],
                        directory=relative(directory),command=command,config=identity(CONFIG))
            append(LEDGER,intent);intents.append(intent)
            write(directory/'intent.json',intent,once=True)
            env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['SIM_V52_CONTROLLER_TOKEN']=attempt_id
            env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
            started=time.monotonic();peak=0;cpu=0.;termination=None
            with (directory/'stdout.log').open('xb') as stdout,(directory/'stderr.log').open('xb') as stderr:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                status('R3_running',controller_pid=os.getpid(),child_pid=child.pid,slot_id=slot['slot_id'],
                       engineering_attempts_started=len(intents),engineering_attempts_cap=48)
                try:
                    while child.poll() is None:
                        elapsed=time.monotonic()-started
                        try:
                            process=psutil.Process(child.pid);tree=[process]+process.children(recursive=True)
                            rss=sum(p.memory_info().rss for p in tree)
                            cpu=max(cpu,sum(p.cpu_times().user+p.cpu_times().system for p in tree))
                            peak=max(peak,rss)
                        except psutil.Error:
                            rss=0
                        reason=None
                        if elapsed>cfg['limits']['episode_wall_seconds']+5:reason='child_wall_limit'
                        if cumulative+elapsed>cfg['limits']['cumulative_wall_seconds']:reason='cumulative_wall_limit'
                        if rss>cfg['limits']['tree_rss_bytes']:reason='tree_rss_limit'
                        if (LOG/'STOP').exists():reason='stop_requested'
                        if reason:
                            termination=dict(reason=reason,survivors=stop_child(child))
                            break
                        time.sleep(.25)
                except BaseException:
                    termination=dict(reason='controller_exception',survivors=stop_child(child),traceback=traceback.format_exc())
            elapsed=time.monotonic()-started;cumulative+=elapsed
            result_path=directory/'result.json'
            event=dict(event='returned',at=now(),attempt_id=attempt_id,slot_id=slot['slot_id'],exit_code=child.returncode,
                       wall_seconds=elapsed,peak_tree_rss_bytes=peak,cpu_seconds=cpu,
                       cpu_utilization_one_core_percent=cpu/max(elapsed,1e-9)*100,
                       renderer='PyBullet TinyRenderer CPU',new_neural_forwards=0,gpu_allocation_by_design=False,
                       termination=termination,result=identity(result_path) if result_path.exists() else None,
                       result_status='known_result' if result_path.exists() else 'unknown_episode_result')
            append(LEDGER,event);write(directory/'resources.json',event,once=True)
            if termination or not result_path.exists() or read(result_path)['outcome']=='engineering_exception':
                status('engineering_failure_preserved_stopped',engineering_attempts_started=len(intents),directory=relative(directory))
                break
        report()

def report():
    cfg=read(CONFIG)
    verify_frozen(cfg)
    intents,returns,rows,unknown=verified_ledger(cfg)
    nominal=[r for r in rows if r['result']['slot']['kind']=='nominal']
    miss=[r for r in rows if r['result']['slot']['kind']=='miss']
    collision=[r for r in rows if r['result']['slot']['kind']=='table_collision']
    successes=sum(r['result']['success'] for r in nominal)
    per_object={obj:sum(r['result']['success'] for r in nominal if r['result']['slot']['object_id']==obj) for obj in cfg['objects']}
    miss_ok=sum(r['result']['outcome']=='grasp_or_lift_failed' and r['result'].get('dynamic_coupling_passed',False) and not r['result'].get('forbidden_collision',False) for r in miss)
    collision_ok=sum(r['result']['outcome'] in ('planned_table_collision','target_contact_below_known_table') for r in collision)
    complete=len(rows)==48 and len({r['slot_id'] for r in rows})==48 and not unknown
    integrity=not unknown and not any(r['result']['outcome'] in ('engineering_exception','dynamic_coupling_failed') for r in rows)
    passed=complete and integrity and successes>=cfg['criteria']['positive_min'] and min(per_object.values())>=cfg['criteria']['per_object_positive_min'] and miss_ok==12 and collision_ok==12
    formed=sum(r['result'].get('grasp_formed',False) for r in rows)
    slipped=sum(r['result'].get('slipped',False) for r in rows)
    data=dict(at=now(),scope='R3 engineering admission; not Raw/A/C0 simulation comparison',attempts_started=len(intents),
        attempts_with_results=len(rows)-len(unknown),attempts_cap=48,unknown_episode_results=unknown,
        planned_slots=48,complete=complete,execution_integrity_passed=integrity,engineering_passed=passed,R4_started=False,review_required=True,
        nominal_successes=successes,nominal_attempts=len(nominal),positive_successes_per_object=per_object,
        miss_controls_correct=miss_ok,miss_controls_attempted=len(miss),table_controls_correct=collision_ok,table_controls_attempted=len(collision),
        all_attempt_success_rate=sum(r['result']['success'] for r in rows)/max(len(rows),1),
        forbidden_collision_rate=sum(r['result'].get('forbidden_collision',False) for r in rows)/max(len(rows),1),
        infeasible_rate=sum(r['result'].get('infeasible',False) for r in rows)/max(len(rows),1),
        slip_count=slipped,grasp_formed_count=formed,slip_rate_all=slipped/max(len(rows),1),slip_rate_formed=slipped/formed if formed else None,
        total_worker_wall_seconds=sum(e['wall_seconds'] for e in returns.values()),
        total_worker_cpu_seconds=sum(e['cpu_seconds'] for e in returns.values()),
        peak_tree_rss_bytes=max((e['peak_tree_rss_bytes'] for e in returns.values()),default=0),
        actual_new_neural_forwards=0,actual_new_training=0,old_P5_P6_scientific_passed=False,
        historical_unchanged=verify_history(),rows=rows)
    name=OUT/'admission'/f'report_{len(rows):03d}.json'
    if name.exists():
        name=OUT/'admission'/f'report_{len(rows):03d}_{uuid.uuid4().hex[:8]}.json'
    write(name,data,once=True)
    status('completed_stopped_for_review' if complete else 'R3_partial_stopped',engineering_attempts_started=len(intents),
           engineering_attempts_cap=48,engineering_passed=passed,report=relative(name),R3_complete=complete)
    print(json.dumps({k:data[k] for k in ('attempts_started','complete','engineering_passed','nominal_successes','miss_controls_correct','table_controls_correct','total_worker_wall_seconds')}))
    return data

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run','report']);parser.add_argument('--count',type=int,default=48)
    args=parser.parse_args()
    {'freeze':freeze,'run':lambda:run(args.count),'report':report}[args.command]()

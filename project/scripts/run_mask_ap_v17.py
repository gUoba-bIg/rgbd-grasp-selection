"""Prove bounded restart and immutable read-only reuse of the fixed mask evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from scripts import mask_ap_common_v17 as common
from scripts.pipeline import project_environment


def last_record(file,offset):
    with Path(file).open('rb') as stream:
        stream.seek(offset)
        for line in reversed(stream.read().decode('utf-8').splitlines()):
            try:
                row=json.loads(line)
            except ValueError:
                continue
            if isinstance(row,dict) and 'status' in row:
                return row
    raise RuntimeError('Child has no current completion record')


def invoke(root,protocol,command,name,guard):
    guard.check(65536)
    file=root/'logs/P4/mask_ap_v17'/name
    file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('ab',buffering=0) as stream:
        stream.write(('\nCOMMAND '+json.dumps([str(x) for x in command],ensure_ascii=False)+'\n').encode('utf-8'))
        offset=stream.tell()
        result=subprocess.run(command,cwd=root,env=project_environment(root),stdout=stream,
            stderr=subprocess.STDOUT,timeout=protocol['resources']['invocation_soft_seconds']+60)
        os.fsync(stream.fileno())
    common.require(result.returncode==0,'Evaluation stopped; saved work retained. Inspect '+str(file))
    guard.check()
    return last_record(file,offset)


def validate_record(row,phase,key,operation):
    unit='frames' if phase=='scores' else 'blocks'
    expected=160 if phase=='scores' else 22
    common.require(row.get('schema')==17 and row.get('expected_'+unit)==expected,'Child identity differs')
    if phase=='scores':
        common.require(row.get('arm')==key,'Child arm differs')
    for name in ('new_'+unit,'reused_'+unit,'committed_'+unit,'new_model_forward_calls','new_training_updates'):
        common.require(type(row.get(name)) is int and row[name]>=0,'Invalid work accounting')
    common.require(row['new_'+unit]+row['reused_'+unit]==row['committed_'+unit]
        and row['committed_'+unit]<=expected and row['new_training_updates']==0,'Inconsistent work accounting')
    if phase=='scores':
        common.require(row['new_model_forward_calls']<=row['new_frames'],'Extra forward calls in one invocation')
    else:
        common.require(row['new_model_forward_calls']==0 and type(row.get('new_official_evaluations')) is int
            and 0<=row['new_official_evaluations']<=row['new_blocks'],'Metrics performed unexpected work')
    if operation=='probe':
        common.require(row['status']=='partial' and row['committed_'+unit]==2,'Probe must stop at two units')
    else:
        common.require(row['status'] in ('completed','verified_completed_read_only')
            and row['committed_'+unit]==expected,'Evaluation is incomplete')
    if operation=='read_only' or row['status']=='verified_completed_read_only':
        common.require(row['status']=='verified_completed_read_only' and row['new_'+unit]==0
            and row['new_model_forward_calls']==0,'Read-only performed new work')
        if phase=='metrics':
            common.require(row['new_official_evaluations']==0,'Read-only reran official evaluation')


def checked_prefix(root,directory,prefix):
    observed=common.snapshot(directory,root)
    common.require(bool(prefix) and all(observed.get(name)==row for name,row in prefix.items()),
        'Saved prefix SHA, size or mtime changed')
    return {name:observed[name] for name in prefix}


def phase_layout(root,p,phase,key):
    if phase=='scores':
        return (common.safe(root,p['outputs']['scores_root'])/key,
            common.safe(root,p['outputs']['scores_root'])/key/'report.json')
    return common.safe(root,p['outputs']['metrics_root']),common.safe(root,p['outputs']['metrics_report'])


def completed_snapshot(root,directory,report):
    result=common.snapshot(directory,root)
    if not report.is_relative_to(directory):
        for file in (report,common.receipt_path(report)):
            common.require(file.is_file(),'Missing aggregate report or receipt')
            result[file.relative_to(root).as_posix()]={'sha256':common.digest(file),'bytes':file.stat().st_size,
                                                      'mtime_ns':file.stat().st_mtime_ns}
    return result


def unrecorded_probe_count(directory,p,phase):
    if phase=='scores':
        permitted=[directory/'frames'/('scene_%04d_%04d'%(scene,frame))
                   for scene in p['validation_scenes'] for frame in p['frames']][:2]
        candidates=list((directory/'frames').iterdir()) if (directory/'frames').exists() else []
        marker='record.json'
    else:
        permitted=[directory/arm/scope for arm in p['arms']
                   for scope in ['scene_%04d'%scene for scene in p['validation_scenes']]+['pooled']][:2]
        candidates=[file for arm in p['arms'] if (directory/arm).exists() for file in (directory/arm).iterdir()]
        marker='completion.json'
    common.require(all(file in permitted and file.is_dir() for file in candidates),
        'Unrecorded payload or attempt extends beyond the two-unit probe')
    return sum((file/marker).is_file() for file in permitted)


def execute(root,config,p,phase):
    common.require(phase in ('scores','metrics'),'Unknown evaluation phase')
    common.verify_source_maps(root,p)
    common.require(p['schema']==17 and p['arms']==['rgb','rgbd'],'Wrong evaluation protocol')
    interpreter=common.safe(root,p['interpreter'] if phase=='scores' else p['evaluation_interpreter'])
    child=root/('scripts/extract_mask_scores_v17.py' if phase=='scores' else 'scripts/evaluate_mask_ap_v17.py')
    proof_path=common.safe(root,p['outputs']['score_resume_report' if phase=='scores' else 'metrics_resume_report'])
    binding={'protocol_sha256':common.digest(config),'wrapper_sha256':common.digest(Path(__file__)),
        'child_sha256':common.digest(child),'interpreter_sha256':common.digest(interpreter),
        'pipeline_sha256':common.digest(root/'scripts/pipeline.py')}
    keys=['rgb','rgbd'] if phase=='scores' else ['all']
    proof=common.read(proof_path) if proof_path.exists() else {'schema':17,'phase':phase,'status':'in_progress',
        'binding':binding,'entries':{},'scope':'Actual process boundaries; no claim of physical power removal'}
    common.require(proof['schema']==17 and proof['phase']==phase and proof['binding']==binding
        and set(proof['entries']).issubset(keys) and proof['status'] in ('in_progress','passed'),
        'Recovery proof binding differs')
    def save():
        common.Guard(root,p,phase).check(len(common.json_bytes(proof))+1024)
        common.atomic(proof_path,proof)
    if not proof_path.exists():
        save()
    for key in keys:
        directory,report=phase_layout(root,p,phase,key)
        def call(operation,flags):
            command=[str(interpreter),'-B',str(child),'--config',str(config)]
            if phase=='scores':
                command+=['--arm',key]
            command+=flags
            print('Mask evaluation %s %s: %s'%(phase,key,operation),flush=True)
            row=invoke(root,p,command,phase+'_'+key+'_'+operation+'.log',common.Guard(root,p,phase))
            validate_record(row,phase,key,operation)
            return row
        entry=proof['entries'].get(key)
        if entry is None:
            common.require(proof['status']=='in_progress' and not (directory/'completion.json').exists(),
                'Completed output lacks actual restart history')
            entry={'status':'in_progress'}
            proof['entries'][key]=entry
            save()
        if entry['status']=='passed':
            common.require(completed_snapshot(root,directory,report)==entry['completed_snapshot'],'Sealed outputs changed')
            call('read_only',['--check-only'])
            common.require(completed_snapshot(root,directory,report)==entry['completed_snapshot'],'Read-only changed outputs')
            continue
        common.require(entry['status']=='in_progress' and proof['status']=='in_progress','Unknown recovery entry state')
        if 'prefix_before' not in entry:
            count=unrecorded_probe_count(directory,p,phase)
            common.require(count<=2,'Unrecorded prefix exceeds the frozen probe size')
            flag='--max-new-frames' if phase=='scores' else '--max-new-blocks'
            record=call('probe',[flag,str(2-count)])
            entry.update(probe_record=record,prefix_before=common.snapshot(directory,root))
            save()
        checked_prefix(root,directory,entry['prefix_before'])
        complete=call('completion',[])
        prefix_after=checked_prefix(root,directory,entry['prefix_before'])
        before=completed_snapshot(root,directory,report)
        readonly=call('read_only',['--check-only'])
        common.require(before==completed_snapshot(root,directory,report),'Read-only changed SHA, size or mtime')
        if phase=='metrics':
            common.require(complete['report_sha256']==readonly['report_sha256']==common.digest(report),
                'Aggregate report SHA differs')
        entry.update(status='passed',prefix_after=prefix_after,completion_record=complete,read_only_record=readonly,
            byte_and_mtime_identical=True,read_only_check_passed=True,completed_snapshot=before,
            completion_sha256=common.digest(directory/'completion.json'),report_sha256=common.digest(report))
        save()
    if proof['status']!='passed':
        proof['status']='passed'
        save()
    print(json.dumps({'schema':17,'status':'passed','phase':phase,'proof_sha256':common.digest(proof_path)}),flush=True)
    return proof


def run(root=ROOT,config=None,phase=None):
    root=Path(root).resolve()
    config=common.safe(root,config or 'configs/mask_ap_v17.json')
    common.require(phase in ('scores','metrics'),'Unknown evaluation phase')
    p=common.read(config)
    with common.lock(root/'runtime/locks/mask_ap_v17_wrapper'/phase):
        return execute(root,config,p,phase)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mask_ap_v17.json')
    parser.add_argument('--phase',choices=('scores','metrics'),required=True)
    args=parser.parse_args()
    run(config=args.config,phase=args.phase)

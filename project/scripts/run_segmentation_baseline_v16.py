"""Run the fixed paired baseline and actual two-frame evaluation recovery."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts import multiview_common_v14r1 as storage
from scripts.pipeline import project_environment

ARMS = ('rgb', 'rgbd')
CONFIG = ROOT/'configs/segmentation_baseline_v16.json'


def snapshot(root, directory):
    return {path.relative_to(root).as_posix(): {'sha256': storage.digest(path), 'bytes': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns}
            for path in sorted(Path(directory).rglob('*')) if path.is_file()}


def budget(root, protocol, extra=0):
    total = 0
    for name in protocol['budget_paths']:
        target = storage.safe_path(root, name)
        if target.is_file():
            total += target.stat().st_size
        elif target.exists():
            total += sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
    limits = protocol['resources']
    storage.require(total+extra <= limits['output_limit_bytes'], 'Baseline output budget exhausted')
    storage.require(shutil.disk_usage(root).free-extra >= limits['disk_reserve_bytes'], 'Baseline disk reserve exhausted')
    return total


def verify_contract(root, config, protocol):
    storage.require(protocol['schema'] == 16 and tuple(protocol['arms']) == ARMS
                    and protocol['planned_steps'] == 1600 and protocol['segment_boundaries'] == [200,800,1600],
                    'Wrong paired training budget or arms')
    storage.require(config.is_relative_to(root) and bool(protocol['source_hashes']), 'Source freeze required')
    for group in ('source_hashes', 'input_hashes'):
        for name, sha in protocol[group].items():
            storage.require(storage.digest(storage.safe_path(root,name)) == sha, 'Frozen source/input changed: '+name)
    budget(root,protocol)


def last_record(path, offset):
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        for line in reversed(stream.read().decode('utf-8').splitlines()):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value,dict) and 'status' in value:
                return value
    raise RuntimeError('Child did not emit a current completion record')


def invoke(root, protocol, command, name, timeout):
    budget(root,protocol,65536)
    logfile = root/'logs/P4/segmentation_baseline_v16'/name
    logfile.parent.mkdir(parents=True,exist_ok=True)
    with logfile.open('ab',buffering=0) as stream:
        stream.write(('\nCOMMAND '+json.dumps(command,ensure_ascii=False)+'\n').encode('utf-8'))
        offset = stream.tell()
        result = subprocess.run(command,cwd=root,env=project_environment(root),stdout=stream,
                                stderr=subprocess.STDOUT,timeout=timeout)
        os.fsync(stream.fileno())
    storage.require(result.returncode == 0,'Baseline child stopped; saved state retained. Inspect '+str(logfile))
    budget(root,protocol)
    return last_record(logfile,offset)


def current_training_step(directory):
    if not (directory/'config.json').exists():
        storage.require(not directory.exists() or not any(directory.iterdir()), 'Unbound training directory')
        return 0
    from src.checkpoints import CheckpointStore, fingerprint
    config, manifest = (storage.read(directory/name) for name in ('config.json','data_manifest.json'))
    store = CheckpointStore(directory/'checkpoints',fingerprint(config),fingerprint(manifest),keep_last=3)
    artifacts = [path for path in store.directory.iterdir() if path.name != '.checkpoint.lock']
    if not artifacts:
        from scripts.train_segmentation_pilot import assert_no_training_evidence
        assert_no_training_evidence(directory)
        return 0  # The child rechecks identical config/data before completing step zero.
    for path in store._manifest_paths():
        try:
            metadata, _file = store._read_manifest(path)
        except (OSError,ValueError,KeyError,RuntimeError):
            continue
        step = metadata['global_step']
        storage.require(type(step) is int and 0 <= step <= 1600,'Invalid committed training cursor')
        return step
    raise RuntimeError('Existing run has no verified checkpoint; refuse a new initialization')


def remaining_boundaries(step):
    storage.require(type(step) is int and 0 <= step <= 1600,'Invalid research cursor')
    return [limit for limit in (200,800,1600) if limit > step] or [1600]


def run_training(root, config, protocol):
    from scripts import train_segmentation_baseline_v16 as trainer
    from scripts import verify_segmentation_baseline_v16_resume as recovery
    proof_path = storage.safe_path(root,protocol['outputs']['training_resume_report'])
    storage.require(proof_path.exists(),'Actual GPU resume proof must already exist')
    recovery.verify(config,check_only=True,root=root)
    proof_sha = storage.digest(proof_path)
    interpreter = str(storage.safe_path(root,protocol['interpreter']))
    arms = {}
    for arm in ARMS:
        directory = storage.safe_path(root,protocol['arms'][arm]['run_directory'])
        existing = storage.read(directory/'result.json') if (directory/'result.json').exists() else {}
        if existing.get('status') != 'trained':
            current = current_training_step(directory)
            for limit in remaining_boundaries(current):
                command = [interpreter,'-B',str(root/'scripts/train_segmentation_baseline_v16.py'),
                    '--config',str(config),'--arm',arm,'--run-dir',str(directory),'--max-steps',str(limit)]
                if (directory/'config.json').exists():
                    command.append('--resume')
                print('Paired baseline %s: through %d of 1600' % (arm,limit),flush=True)
                record = invoke(root,protocol,command,'training_'+arm+'_through_'+str(limit)+'.log',
                                protocol['resources']['training_seconds_per_arm']+60)
                storage.require(record['arm'] == arm and record['optimizer_steps'] == limit
                    and record['status'] == ('trained' if limit == 1600 else 'step_limit_reached')
                    and record['validation_evaluated'] is False,'Unexpected training boundary or evaluation')
        before = snapshot(root,directory)
        model, run_config, manifest, result, evidence = trainer.load_final_model(directory,protocol)
        del model
        storage.require(before == snapshot(root,directory),'Final training verification changed saved artifacts')
        storage.require(result['status'] == 'trained' and result['optimizer_steps'] == 1600
                        and result['validation_evaluated'] is False,'Incomplete paired arm')
        arms[arm] = {'run_directory':directory.relative_to(root).as_posix(),'result':result,
            'result_sha256':storage.digest(directory/'result.json'),'config_sha256':storage.digest(directory/'config.json'),
            'history_sha256':storage.digest(directory/'history.json'),
            'initialization_sha256':storage.digest(directory/'initialization.json'),
            'checkpoint_sha256':evidence['checkpoint_sha256']}
    report = {'schema':16,'status':'completed','protocol_sha256':storage.digest(config),
        'resume_proof_sha256':proof_sha,'new_research_updates':3200,'arms':arms,'automatic_extension':False,
        'validation_evaluated':False,'efficacy_gate':False,'scope':'Fixed matched-budget paired development baseline'}
    path = storage.safe_path(root,protocol['outputs']['training_report'])
    payload = storage.json_bytes(report)
    budget(root,protocol,0 if path.exists() else len(payload))
    storage.publish_once(path,payload)
    print(json.dumps({'status':'completed','phase':'training','new_research_updates':3200,
                      'report_sha256':storage.digest(path)}),flush=True)
    return report


def validate_evaluation_record(record, arm, operation):
    storage.require(record.get('schema') == 16 and record.get('arm') == arm and record.get('expected_frames') == 480,
                    'Evaluation child identity differs')
    for key in ('new_frames','reused_frames','committed_frames','new_model_forward_calls'):
        storage.require(type(record.get(key)) is int and record[key] >= 0,'Invalid frame accounting')
    storage.require(record['new_frames']+record['reused_frames'] == record['committed_frames']
                    and record['new_model_forward_calls'] <= record['new_frames']
                    and record.get('new_training_updates') == 0,'Evaluation work accounting differs')
    if operation == 'probe':
        storage.require(record['status'] == 'partial' and record['committed_frames'] == 2,'Probe must stop at two frames')
    else:
        storage.require(record['status'] in ('completed','verified_completed_read_only') and record['committed_frames'] == 480,
                        'Incomplete evaluation')
    if operation == 'read_only' or record['status'] == 'verified_completed_read_only':
        storage.require(record['status'] == 'verified_completed_read_only'
            and record['new_frames'] == record['new_model_forward_calls'] == 0,'Read-only recomputed a model frame')


def verify_prefix(root, directory, prefix):
    observed = snapshot(root,directory)
    storage.require(bool(prefix) and all(observed.get(key) == value for key,value in prefix.items()),
                    'Two-frame prefix SHA, size or mtime changed')
    return {name:observed[name] for name in prefix}


def run_evaluation(root, config, protocol):
    interpreter = str(storage.safe_path(root,protocol['interpreter']))
    script = root/'scripts/evaluate_segmentation_baseline_v16.py'
    proof_path = storage.safe_path(root,protocol['outputs']['evaluation_resume_report'])
    binding = {'protocol_sha256':storage.digest(config),'wrapper_sha256':storage.digest(Path(__file__)),
        'child_sha256':storage.digest(script),'interpreter_sha256':storage.digest(Path(interpreter)),
        'pipeline_sha256':storage.digest(root/'scripts/pipeline.py')}
    def save(proof):
        budget(root,protocol,len(storage.json_bytes(proof))+1024)
        storage.atomic(proof_path,proof)
    proof = storage.read(proof_path) if proof_path.exists() else {'schema':16,'phase':'evaluation',
        'status':'in_progress','binding':binding,'arms':{},'scope':'Actual process boundaries; not physical power removal'}
    storage.require(proof['schema'] == 16 and proof['phase'] == 'evaluation' and proof['binding'] == binding
                    and set(proof['arms']).issubset(ARMS),'Evaluation recovery binding differs')
    if not proof_path.exists():
        save(proof)
    def call(arm,operation,flags):
        command = [interpreter,'-B',str(script),'--config',str(config),'--arm',arm]+flags
        print('Paired evaluation %s: %s' % (arm,operation),flush=True)
        record = invoke(root,protocol,command,'evaluation_'+arm+'_'+operation+'.log',
                        protocol['resources']['evaluation_seconds_per_invocation']+60)
        validate_evaluation_record(record,arm,operation)
        return record
    for arm in ARMS:
        directory = storage.safe_path(root,protocol['outputs']['evaluation_root'])/arm
        entry = proof['arms'].get(arm)
        if entry is None:
            storage.require(not (directory/'completion.json').exists(),'Completed arm lacks actual recovery history')
            entry = {'status':'in_progress'}
            proof['arms'][arm] = entry
            save(proof)
        if entry['status'] == 'passed':
            storage.require(snapshot(root,directory) == entry['completed_snapshot'],'Sealed evaluation changed')
            call(arm,'read_only',['--check-only'])
            storage.require(snapshot(root,directory) == entry['completed_snapshot'],'Read-only changed evaluation')
            continue
        storage.require(entry['status'] == 'in_progress','Unknown recovery entry status')
        if 'prefix_before' not in entry:
            count = sum(1 for path in (directory/'frames').rglob('record.json') if path.is_file())
            storage.require(count <= 2,'Unrecorded evaluation prefix exceeds frozen probe')
            record = call(arm,'probe',['--max-new-frames',str(2-count)])
            entry.update(probe_record=record,prefix_before=snapshot(root,directory))
            save(proof)
        verify_prefix(root,directory,entry['prefix_before'])
        complete = call(arm,'completion',[])
        prefix_after = verify_prefix(root,directory,entry['prefix_before'])
        before = snapshot(root,directory)
        readonly = call(arm,'read_only',['--check-only'])
        storage.require(before == snapshot(root,directory),'Read-only evaluation changed byte/size/mtime')
        entry.update(status='passed',prefix_after=prefix_after,completion_record=complete,read_only_record=readonly,
            byte_and_mtime_identical=True,read_only_check_passed=True,completed_snapshot=before,
            completion_sha256=storage.digest(directory/'completion.json'),report_sha256=storage.digest(directory/'report.json'))
        save(proof)
    command = [interpreter,'-B',str(script),'--config',str(config),'--aggregate']
    completed = invoke(root,protocol,command,'evaluation_aggregate.log',protocol['resources']['evaluation_seconds_per_invocation']+60)
    output = storage.safe_path(root,protocol['outputs']['evaluation_report'])
    before = {'sha256':storage.digest(output),'bytes':output.stat().st_size,'mtime_ns':output.stat().st_mtime_ns}
    readonly = invoke(root,protocol,command+['--check-only'],'evaluation_aggregate_read_only.log',
                      protocol['resources']['evaluation_seconds_per_invocation']+60)
    storage.require(before == {'sha256':storage.digest(output),'bytes':output.stat().st_size,'mtime_ns':output.stat().st_mtime_ns},
                    'Aggregate read-only changed bytes or mtime')
    storage.require(completed.get('schema') == readonly.get('schema') == 16
        and completed['status'] == 'completed' and readonly['status'] == 'verified_completed_read_only'
        and completed['aggregate_report_sha256'] == readonly['aggregate_report_sha256'] == before['sha256']
        and completed['new_model_forward_calls'] == readonly['new_model_forward_calls'] == 0
        and completed['new_training_updates'] == readonly['new_training_updates'] == 0,'Aggregate performed model work')
    if proof['status'] != 'passed':
        proof.update(status='passed',aggregate_record=completed,aggregate_read_only_record=readonly,
                     aggregate_report_sha256=before['sha256'],aggregate_read_only_unchanged=True)
        save(proof)
    else:
        storage.require(proof['aggregate_report_sha256'] == before['sha256'],'Completed aggregate changed')
    print(json.dumps({'status':'passed','phase':'evaluation','arms':list(ARMS)}),flush=True)
    return proof


def run(root=ROOT, config=CONFIG, phase='training'):
    root,config = Path(root).resolve(),Path(config).resolve()
    protocol = storage.read(config)
    verify_contract(root,config,protocol)
    storage.require(phase in ('training','evaluation'),'Unknown wrapper phase')
    with storage.lock(root/'runtime/locks/segmentation_baseline_wrapper_v16'/phase):
        return run_training(root,config,protocol) if phase == 'training' else run_evaluation(root,config,protocol)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--phase',choices=('training','evaluation'),required=True)
    args=parser.parse_args()
    run(config=args.config,phase=args.phase)

"""Independent cached-mask arithmetic audit of the paired development baseline."""
import argparse
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Shared functions below are storage, identity, and resource checks, never metric arithmetic.
from scripts import evaluate_segmentation_baseline_v16 as storage
import numpy as np
from PIL import Image, ImageDraw, ImageFont

require = storage.require
ARMS = ('rgb','rgbd')
PARTITIONS = ('train','development')
FIGURE = 'paired_development_baseline.png'


def exact(numerator,denominator,empty=0):
    number = Fraction(numerator,denominator) if denominator else Fraction(empty,1)
    return {'numerator':number.numerator,'denominator':number.denominator,'value':float(number)}


def independently_match(prediction,reference):
    """Integer IoU threshold and augmenting-path maximum bipartite matching."""
    require(prediction.shape == reference.shape and prediction.ndim == 2
        and np.issubdtype(prediction.dtype,np.integer) and np.issubdtype(reference.dtype,np.integer)
        and np.min(prediction,initial=0) >= 0 and np.min(reference,initial=0) >= 0, 'Invalid matching masks')
    pids,pinverse,pcounts = np.unique(prediction,return_inverse=True,return_counts=True)
    gids,ginverse,gcounts = np.unique(reference,return_inverse=True,return_counts=True)
    intersections = np.bincount(pinverse*len(gids)+ginverse,minlength=len(pids)*len(gids)).reshape(len(pids),len(gids))
    unions = pcounts[:,None]+gcounts[None,:]-intersections
    # >= .5 is evaluated as exact integers, including equality at the boundary.
    adjacency = {int(i):[int(j) for j in np.flatnonzero((2*intersections[i] >= unions[i]) & (gids > 0))]
        for i in np.flatnonzero(pids > 0)}
    owner = {}
    def augment(pred,seen):
        for actual in adjacency[pred]:
            if actual in seen:
                continue
            seen.add(actual)
            if actual not in owner or augment(owner[actual],seen):
                owner[actual] = pred
                return True
        return False
    count = sum(int(augment(pred,set())) for pred in adjacency)
    return {'tp':count,'fp':int(np.count_nonzero(pids > 0))-count,'fn':int(np.count_nonzero(gids > 0))-count}


def count_metrics(tp,fp,fn,intersection,union):
    require(all(type(v) is int and v >= 0 for v in (tp,fp,fn,intersection,union)) and intersection <= union,
        'Invalid integer counts')
    return {'instances':{'tp':tp,'fp':fp,'fn':fn,
        'precision_iou50':exact(tp,tp+fp),'recall_iou50':exact(tp,tp+fn),'f1_iou50':exact(2*tp,2*tp+fp+fn)},
        'foreground_intersection':intersection,'foreground_union':union,
        'foreground_pixel_iou':exact(intersection,union,empty=1)}


def independent_frame(prediction,foreground,labels):
    require(foreground.dtype == np.bool_ and foreground.shape == prediction.shape, 'Invalid foreground cache')
    counts = independently_match(prediction,labels)
    reference = labels != 0
    intersection = int(np.count_nonzero(foreground[reference]))
    union = int(np.count_nonzero(foreground))+int(np.count_nonzero(reference))-intersection
    result = count_metrics(counts['tp'],counts['fp'],counts['fn'],intersection,union)
    result.update(predicted_instances=counts['tp']+counts['fp'],actual_instances=counts['tp']+counts['fn'])
    return result


def independent_summary(rows):
    require(rows, 'Cannot pool an empty partition')
    groups = {}
    for row in rows:
        scene = row['identity']['scene']
        groups.setdefault(scene,[]).append(row['metrics'])
    def summarize(items):
        counts = [sum(item['instances'][name] for item in items) for name in ('tp','fp','fn')]
        return count_metrics(*counts,sum(item['foreground_intersection'] for item in items),sum(item['foreground_union'] for item in items))
    scenes = [dict(scene=scene,frame_count=len(groups[scene]),**summarize(groups[scene])) for scene in sorted(groups)]
    fs = [Fraction(s['instances']['f1_iou50']['numerator'],s['instances']['f1_iou50']['denominator']) for s in scenes]
    us = [Fraction(s['foreground_pixel_iou']['numerator'],s['foreground_pixel_iou']['denominator']) for s in scenes]
    fmean,umean = sum(fs,Fraction())/len(fs),sum(us,Fraction())/len(us)
    return {'frame_count':len(rows),'scene_count':len(scenes),'scenes':scenes,
        'micro':summarize([row['metrics'] for row in rows]),
        'scene_macro':{'instance_f1_iou50':exact(fmean.numerator,fmean.denominator),
            'foreground_pixel_iou':exact(umean.numerator,umean.denominator)}}


def independent_paired(reports):
    output = {}
    def difference(left,right):
        value = Fraction(right['numerator'],right['denominator'])-Fraction(left['numerator'],left['denominator'])
        return exact(value.numerator,value.denominator)
    for split in PARTITIONS:
        first,second = (reports[arm]['partitions'][split] for arm in ARMS)
        require([s['scene'] for s in first['scenes']] == [s['scene'] for s in second['scenes']], 'Unpaired scene identities')
        output[split] = {'scene_macro_delta_rgbd_minus_rgb':{
            name:difference(first['scene_macro'][name],second['scene_macro'][name]) for name in ('instance_f1_iou50','foreground_pixel_iou')},
            'scenes':[{'scene':a['scene'],'instance_f1_iou50_delta':difference(a['instances']['f1_iou50'],b['instances']['f1_iou50']),
                'foreground_pixel_iou_delta':difference(a['foreground_pixel_iou'],b['foreground_pixel_iou'])}
                for a,b in zip(first['scenes'],second['scenes'])]}
    return output


def reference_labels(root,protocol,record):
    file = storage.path(root,protocol['data_root'])/record['files']['label']
    require(storage.digest(file) == record['sha256']['label'], 'Reference label source changed')
    with Image.open(file) as source:
        raw = np.asarray(source)
        require(raw.ndim == 2 and np.all((raw >= 0) & (raw <= 88)), 'Reference label range differs')
        # Same declared nearest-neighbor resize, independently opening the original labels.
        resized = source.resize((protocol['input_size'][1],protocol['input_size'][0]),resample=Image.Resampling.NEAREST)
        return np.array(resized,dtype=np.int64,copy=True)


def tensor_fingerprint(state):
    import torch
    content = {}
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        content[name] = {'dtype':str(value.dtype),'shape':list(value.shape),
            'sha256':hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    return storage.fingerprint(content)


def reconstructed_initial_weights(seed):
    """Independent tensor construction using the pinned architecture; no model forward."""
    import torch
    from src.segmentation import PilotInstanceNet
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rgb = PilotInstanceNet(mode='rgb').state_dict()
    extended = {name:value.clone() for name,value in rgb.items()}
    first = rgb['stem.0.weight']
    extended['stem.0.weight'] = torch.cat([first.clone(),torch.zeros((first.shape[0],2,*first.shape[2:]),dtype=first.dtype)],dim=1)
    return {'shared_model_tensor_sha256':tensor_fingerprint({name:value for name,value in rgb.items() if name!='stem.0.weight'}),
        'rgb_initial_model_tensor_sha256':tensor_fingerprint(rgb),'rgbd_initial_model_tensor_sha256':tensor_fingerprint(extended),
        'rgb_conv_tensor_sha256':tensor_fingerprint({'stem.0.weight':first})}


def expected_lr(step,protocol):
    index = min(step,protocol['planned_steps'])
    if index <= protocol['warmup_steps']:
        return protocol['learning_rate']*index/protocol['warmup_steps']
    position = (index-protocol['warmup_steps'])/(protocol['planned_steps']-protocol['warmup_steps'])
    return protocol['min_learning_rate']+(protocol['learning_rate']-protocol['min_learning_rate'])*(1+math.cos(math.pi*position))/2


def independent_history(history,records,protocol,expected_steps=1600):
    require(len(history) == expected_steps and len(records) == 320, 'Incomplete fixed training budget')
    orders = [np.random.RandomState(protocol['seed']+epoch).permutation(len(records)).tolist() for epoch in range(10)]
    exposures = {(r['scene'],r['frame']):0 for r in records}
    for step,row in enumerate(history,1):
        epoch = (step-1)//160
        offset = ((step-1)%160)*2
        expected = [{key:records[index][key] for key in ('scene','frame')} for index in orders[epoch][offset:offset+2]]
        require(row['step'] == step and row['epoch'] == epoch and row['samples'] == expected, 'Training exposure/order differs')
        require(row['next_update_index'] == (step+1 if step<1600 else None) and row['next_update_planned'] is (step<1600),
            'Training next-update ledger differs')
        for key,index in (('learning_rate_used',step),('learning_rate_next_update',step+1)):
            require(type(row[key]) in (int,float) and math.isfinite(row[key])
                and math.isclose(row[key],expected_lr(index,protocol),rel_tol=1e-12,abs_tol=1e-12), 'Independent LR schedule differs')
        require(type(row['amp_retries']) is int and 0 <= row['amp_retries'] < 16
            and type(row['amp_scale']) in (float,int) and math.isfinite(row['amp_scale']) and row['amp_scale'] > 0,
            'Invalid AMP update accounting')
        require(all(type(value) in (float,int) and math.isfinite(value) for value in [row['loss'],row['gradient_norm_before_clip'],*row['components'].values()]),
            'Nonfinite training history')
        parts = row['components']
        require(set(parts) == {'foreground_bce','foreground_dice_loss','center_mse','center_weighted','offset_smooth_l1'}, 'Loss component schema changed')
        # FP32 operations inside each of two separately averaged microbatches incur rounding.
        require(math.isclose(parts['center_weighted'],200*parts['center_mse'],rel_tol=2e-6,abs_tol=2e-6)
            and math.isclose(row['loss'],parts['foreground_bce']+parts['foreground_dice_loss']+parts['center_weighted']+5*parts['offset_smooth_l1'],
                rel_tol=2e-6,abs_tol=2e-6), 'Frozen loss weighting differs')
        for item in expected:
            exposures[(item['scene'],item['frame'])] += 1
    if expected_steps == 1600:
        require(set(exposures.values()) == {10}, 'Not every training frame received ten exposures')
    return {'optimizer_updates':expected_steps,'unique_training_frames':320,
        'exposures_per_frame':10 if expected_steps == 1600 else None,
        'sample_sequence_sha256':storage.fingerprint([r['samples'] for r in history]),
        'learning_rate_sequence_sha256':storage.fingerprint([r['learning_rate_used'] for r in history])}


def inspect_training(root,protocol,sources,records,guard):
    import torch
    expected_initial = reconstructed_initial_weights(protocol['seed'])
    summaries,initializations = {},{}
    for arm in ARMS:
        source = sources[arm]
        config = source['values']['config.json']
        result = source['values']['result.json']
        initialization = source['values']['initialization.json']
        history = source['values']['history.json']
        require(config['device'] == 'cuda' and config['effective_amp'] is True and config['engineering_checkpoint_override'] is None
            and config['save_every_optimizer_steps'] == 10 and config['deterministic_algorithms'] is True
            and config['data_fingerprint'] == source['identity']['data_fingerprint'], 'Training execution policy differs')
        require(all(protocol['source_hashes'].get(name) == sha for name,sha in config['source_sha256'].items()), 'Unfrozen training helper source')
        paired = config['paired_initialization']
        require(all(paired.get(key) == value for key,value in expected_initial.items()) and initialization['paired_initialization'] == paired,
            'Initial weight tensors differ from independent scratch construction')
        require(paired['seed'] == protocol['seed'] and paired['pretrained_weights_used'] is False and paired['parent_checkpoint_used'] is False
            and paired['extra_input_channels_zero'] is True and paired['shared_rgb_channels_equal'] is True
            and paired['common_tensors_equal'] is True and paired['initial_output_bitwise_equality_claimed'] is False,
            'Paired initial-state scope differs')
        require(initialization['schema'] == 16 and initialization['arm'] == arm and initialization['cursor'] == {'epoch':0,'next_batch':0,'global_step':0}
            and initialization['history_length'] == initialization['optimizer_state_entries'] == initialization['scheduler_last_epoch'] == 0
            and initialization['optimizer_history_empty'] is True and initialization['initial_used_lr'] == expected_lr(1,protocol)
            and initialization['rng_seed'] == protocol['seed'] and initialization['run_config_fingerprint'] == source['identity']['config_fingerprint']
            and initialization['initial_model_tensor_sha256'] == expected_initial[arm+'_initial_model_tensor_sha256'], 'Initialization ledger differs')
        require(storage.digest(source['directory']/'history.json') == result['history_sha256']
            and storage.digest(source['directory']/'initialization.json') == result['initialization_sha256']
            and result['arm_training_frames'] == 320 and result['equivalent_epochs'] == 10
            and result['automatic_extension'] is False, 'Training result/history seal differs')
        resources=result['resources']; limits=protocol['resources']
        for key,limit in (('elapsed_training_seconds','training_seconds_per_arm'),('peak_rss_bytes','rss_soft_bytes'),
                ('peak_gpu_allocated_bytes','gpu_allocated_soft_bytes'),('peak_gpu_reserved_bytes','gpu_reserved_soft_bytes'),
                ('current_output_bytes','output_limit_bytes')):
            require(type(resources[key]) in (int,float) and math.isfinite(resources[key]) and 0 <= resources[key] <= limits[limit],
                'Training resource ledger exceeds the frozen limit')
        require(resources['schema'] == 16 and resources['config_fingerprint'] == source['identity']['config_fingerprint'],
            'Training resource ledger identity differs')
        payload = torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
        require(payload['cursor'] == result['cursor'] and payload['config_fingerprint'] == source['identity']['config_fingerprint']
            and payload['data_fingerprint'] == source['identity']['data_fingerprint']
            and payload['extra'] == {'history':history,'config':config,'data_fingerprint':source['identity']['data_fingerprint'],
                'initialization':initialization}, 'Final checkpoint-embedded history/config differs')
        require(payload['model']['stem.0.weight'].shape[1] == (3 if arm=='rgb' else 5)
            and all(bool(torch.isfinite(value).all()) for value in payload['model'].values()), 'Final weights are invalid')
        require(payload['scheduler']['last_epoch'] == 1600 and payload['optimizer']['state']
            and all(int(value['step']) == 1600 for value in payload['optimizer']['state'].values()), 'Final optimizer/scheduler update count differs')
        summaries[arm] = dict(independent_history(history,records['train'],protocol),
            checkpoint_sha256=source['identity']['checkpoint_sha256'],final_model_tensor_sha256=tensor_fingerprint(payload['model']))
        initializations[arm] = initialization
        del payload
        guard.check()
    for key in ('paired_initialization','initial_rng_fingerprint','scheduler_state','scaler_state','initial_used_lr'):
        require(initializations['rgb'][key] == initializations['rgbd'][key], 'Paired optimizer/scheduler/scaler/RNG initialization differs')
    require(summaries['rgb']['sample_sequence_sha256'] == summaries['rgbd']['sample_sequence_sha256']
        and summaries['rgb']['learning_rate_sequence_sha256'] == summaries['rgbd']['learning_rate_sequence_sha256'], 'Arms did not share data order and LR schedule')
    return {'arms':summaries,'independently_reconstructed_initial_tensor_hashes':expected_initial,
        'research_optimizer_updates':3200,'initialization_model_forward_calls':0,'no_pretrained_or_learned_parent':True}


def inspect_evaluation(root,protocol,config_path,sources,records,guard):
    reports,seals = {},{}
    for arm in ARMS:
        directory = storage.path(root,protocol['outputs']['evaluation_root'])/arm
        binding = storage.make_binding(root,protocol,config_path,arm,sources)
        require(storage.read(directory/'binding.json') == binding, 'Evaluation source binding differs')
        storage.verify_completion(directory,binding,records)
        rows = []
        for split in PARTITIONS:
            for source_record in records[split]:
                identity = {'partition':split,'scene':source_record['scene'],'frame':source_record['frame'],'camera':'realsense'}
                frame = storage.frame_directory(directory,identity)
                instances,foreground = storage.load_prediction(frame/'prediction.npz',binding,protocol['input_size'])
                metrics = independent_frame(instances,foreground,reference_labels(root,protocol,source_record))
                expected = {'schema':16,'binding_fingerprint':binding['fingerprint'],'arm':arm,'identity':identity,
                    'source_sha256':source_record['sha256'],'prediction_sha256':storage.digest(frame/'prediction.npz'),
                    'checkpoint_sha256':sources[arm]['identity']['checkpoint_sha256'],
                    'prediction_committed_before_metric_label_decoding':True,'model_forward_calls_committed':1,'metrics':metrics}
                require(storage.load_commit(frame/'record.json',binding) == expected, 'Independent frame matching/pixel counts differ: '+str(identity))
                rows.append(expected)
                if len(rows)%32 == 0:
                    guard.check()
        report = {'schema':16,'status':'completed_final_checkpoint_development_evaluation','arm':arm,
            'binding_fingerprint':binding['fingerprint'],'checkpoint':binding['final_checkpoints'][arm],
            'frame_count':len(rows),'partitions':{split:independent_summary([r for r in rows if r['identity']['partition']==split]) for split in PARTITIONS},
            'model_forward_calls_committed':len(rows),'new_training_updates':0,'checkpoint_selection':'final_step_1600_only',
            'efficacy_gate':False,'limitations':storage.LIMITATIONS}
        require(storage.load_commit(directory/'report.json',binding) == report, 'Independent scene/micro summary differs')
        reports[arm] = report
        seals[arm] = {'binding_fingerprint':binding['fingerprint'],'completion_sha256':storage.digest(directory/'completion.json'),
            'report_sha256':storage.digest(directory/'report.json')}
    aggregate = {'schema':16,'status':'completed_paired_development_baseline','protocol_sha256':storage.digest(config_path),
        'arms':reports,'evaluation_seals':seals,'paired':independent_paired(reports),
        'model_frames_committed':960,'new_training_updates':0,'efficacy_gate':False,'limitations':storage.LIMITATIONS}
    require(storage.read(storage.path(root,protocol['outputs']['evaluation_report'])) == aggregate,
        'Paired aggregate differs from independent exact arithmetic')
    return aggregate


def verify_snapshot(root,record,exact_directory=None):
    files = [storage.path(root,name) for name in record]
    require(record == storage.snapshot(root,files), 'Recovery file checksum, size or mtime changed')
    if exact_directory is not None:
        require(set(files) == {p.resolve() for p in Path(exact_directory).rglob('*') if p.is_file()}, 'Recovery directory inventory changed')


def verify_file_evidence(root,record):
    file = storage.path(root,record['path'])
    require(record == dict(path=record['path'],**storage.snapshot(root,[file])[record['path']]), 'Retained file evidence changed')
    return file


def same_state(first,second):
    """Independent exact recursive comparison of cached checkpoint components."""
    import torch
    if type(first) is not type(second):
        return False
    if isinstance(first,torch.Tensor):
        return first.dtype == second.dtype and first.shape == second.shape and torch.equal(
            first.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            second.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    if isinstance(first,np.ndarray):
        return first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes()
    if isinstance(first,dict):
        return first.keys() == second.keys() and all(same_state(first[k],second[k]) for k in first)
    if isinstance(first,(tuple,list)):
        return len(first) == len(second) and all(same_state(a,b) for a,b in zip(first,second))
    if isinstance(first,float):
        return first.hex() == second.hex()
    return first == second


def engineering_attempt_accounting(root,binding,record):
    base = root/'runtime/verification/baseline_v16/attempts'
    attempts,known,exact_count = [],0,True
    for file in sorted(base.glob('*.intent.json')):
        intent = storage.load_commit(file,binding)
        result_file = file.with_name(file.name.replace('.intent.json','.result.json'))
        result = storage.read(result_file) if result_file.exists() else None
        log = storage.path(root,intent['log'])
        rows,malformed = [],0
        if log.exists():
            for line in log.read_text(encoding='utf-8').splitlines():
                if not line.startswith('{'):
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(value,dict) and 'step' in value:
                    require(type(value['step']) is int and 1 <= value['step'] <= 162, 'Invalid engineering log update')
                    rows.append(value)
        require(all(b['step'] == a['step']+1 for a,b in zip(rows,rows[1:])), 'Noncontinuous engineering update log')
        if result is not None:
            require(all(result.get(key) == value for key,value in intent.items()), 'Engineering return differs from intent')
            verify_file_evidence(root,result['log_evidence'])
        certain = result is not None and result['exit_code'] in (0,37) and not result['timed_out'] and not malformed
        known += len(rows); exact_count = exact_count and certain
        attempts.append({'arm':intent['arm'],'phase':intent['phase'],'attempt_index':intent['attempt_index'],
            'logged_accepted_updates':len(rows),'exact_count_known':certain,'observed_exit_code':result['exit_code'] if result else None,
            'log':intent['log'],'log_sha256':storage.digest(log) if log.exists() else None,
            'intent_sha256':storage.digest(file),'result_sha256':storage.digest(result_file) if result else None})
    expected = {'nominal_updates_per_arm':325,'nominal_total_updates':650,'logged_accepted_updates':known,
        'exact_count_known':exact_count,'interpretation':'Exact when all attempt returns and flushed step logs survive; otherwise a lower bound',
        'excluded_from_research_update_budget':True,'research_updates_used':0,'attempts':attempts}
    require(attempts and record == expected, 'Engineering update ledger differs')
    return expected


def inspect_gpu_recovery(root,protocol,config_path,sources,records,guard):
    import torch
    proof_file = storage.path(root,protocol['outputs']['training_resume_report'])
    proof = storage.read(proof_file)
    body = {'schema':16,'phase':'gpu_resume_verification','protocol_sha256':storage.digest(config_path),
        'input_hashes':protocol['input_hashes'],'source_hashes':protocol['source_hashes'],
        'interpreter_sha256':storage.digest(storage.path(root,protocol['interpreter'])),'runtime':protocol['runtime'],
        'data_fingerprint':sources['rgb']['identity']['data_fingerprint'],'recovery_contract':protocol['recovery_contract']}
    binding = dict(body,fingerprint=storage.fingerprint(body))
    require(proof['schema'] == 16 and proof['status'] == 'passed' and proof['binding'] == binding
        and proof['binding_fingerprint'] == binding['fingerprint'], 'GPU recovery binding differs')
    storage.verify_bytes(proof_file,binding)
    require(proof['planned_research_updates'] == 3200 and proof['excluded_from_research_results'] is True
        and proof['validation_evaluated'] is False and proof['real_power_cut_tested'] is False
        and proof['engineering_checkpoints_retained'] is True and proof['checkpoint_interval'] == 2, 'Recovery scope changed')
    base = root/'runtime/verification/baseline_v16'
    require(storage.read(base/'binding.json') == binding, 'Retained recovery binding differs')
    initial = storage.load_commit(base/'initialization_probe.json',binding)
    require(initial == proof['paired_initialization'] and initial['status'] == 'passed' and initial['new_optimizer_updates'] == 0
        and initial['model_forward_calls'] == 2 and initial['validation_evaluated'] is False
        and initial['mode'] == 'float32_without_autocast' and initial['device'] == 'cuda', 'Initial function probe scope differs')
    require(initial['paired_model_evidence'] == sources['rgb']['values']['config.json']['paired_initialization']
        and all(initial['tensor_checks'].values()), 'Recovery initial tensors differ from research start')
    first_index = int(np.random.RandomState(protocol['seed']).permutation(320)[0])
    expected_sample = records['train'][first_index]
    require({key:initial['sample'][key] for key in ('scene','frame')} == {key:expected_sample[key] for key in ('scene','frame')}
        and initial['sample']['record_index'] == first_index and initial['sample']['epoch'] == 0,
        'Initial-output probe selected another training sample')
    require(initial['atol'] == protocol['recovery_contract']['initial_output_atol']
        and initial['rtol'] == protocol['recovery_contract']['initial_output_rtol']
        and set(initial['output_comparisons']) == {'foreground','centers','offsets'}, 'Initial-output tolerance/head schema changed')
    for name,value in initial['output_comparisons'].items():
        require(value['shape'] == [1,2 if name=='offsets' else 1,240,320] and value['within_frozen_tolerance'] is True
            and type(value['bitwise_equal']) is bool
            and all(math.isfinite(value[key]) and value[key] >= 0 for key in ('maximum_absolute_difference','maximum_symmetric_relative_difference')),
            'Invalid initial function-comparison evidence')
        if value['bitwise_equal']:
            require(value['maximum_absolute_difference'] == value['maximum_symmetric_relative_difference'] == 0
                and value['rgb_tensor_sha256'] == value['rgbd_tensor_sha256'], 'Bitwise initial head claim is inconsistent')
    checked = {}
    components = ('model','optimizer','scheduler','scaler','cursor','rng','extra')
    for arm in ARMS:
        item = proof['arms'][arm]
        require(storage.load_commit(base/'proofs'/(arm+'_complete.json'),binding) == item
            and item['schema'] == 16 and item['status'] == 'passed' and item['arm'] == arm
            and item['binding_fingerprint'] == binding['fingerprint'] and set(item['bitwise_equal']) == set(components)
            and all(item['bitwise_equal'].values()), 'Incomplete arm recovery evidence')
        for trajectory in ('continuous','interrupted'):
            directory = base/arm/trajectory
            verify_snapshot(directory,item['completed_snapshots'][trajectory],directory)
        payloads = []
        for key in ('continuous_checkpoint','recovered_checkpoint'):
            evidence = item[key]
            meta_file = verify_file_evidence(root,evidence['manifest'])
            payload_file = verify_file_evidence(root,evidence['payload'])
            metadata = storage.read(meta_file)
            require(metadata['sha256'] == storage.digest(payload_file) and metadata['global_step'] == evidence['global_step'] == 162,
                'Recovery checkpoint seal differs')
            payloads.append(torch.load(payload_file,map_location='cpu',weights_only=False))
        first,second = payloads
        require(all(same_state(first[key],second[key]) for key in components), 'Independent full-state comparison failed')
        require(first['cursor'] == item['final_cursor'] == {'epoch':1,'next_batch':4,'global_step':162}
            and first['config_fingerprint'] == item['config_fingerprint'] and first['data_fingerprint'] == item['data_fingerprint']
            and first['extra']['initialization'] == item['initialization']
            and item['durable_step_before_forced_exit'] == 160 and item['logical_updates_after_durable_boundary'] == 2
            and item['validation_evaluated'] is False, 'Recovered epoch boundary differs')
        history = first['extra']['history']
        independent_history(history,records['train'],protocol,162)
        require(storage.fingerprint(history) == item['history_sha256']
            and storage.fingerprint([r['samples'] for r in history]) == item['samples_sha256']
            and [r['learning_rate_used'] for r in history] == item['learning_rates'], 'Recovery history trajectory differs')
        for phase in ('continuous','interrupted','resumed'):
            phase_proof = item['process_phases'][phase]
            require(storage.load_commit(base/'proofs'/(arm+'_'+phase+'.json'),binding) == phase_proof
                and phase_proof['status'] == 'passed' and phase_proof['phase'] == phase, 'Process boundary proof differs')
            for field in ('manifest','payload'):
                verify_file_evidence(root,phase_proof['checkpoint'][field])
            if phase_proof['process'] is not None:
                verify_file_evidence(root,phase_proof['process']['log_evidence'])
            if phase == 'interrupted':
                require(phase_proof['checkpoint']['global_step'] == 160 and phase_proof['process']['exit_code'] == 37,
                    'Forced process exit evidence missing')
                interrupted = torch.load(storage.path(root,phase_proof['checkpoint']['payload']['path']),map_location='cpu',weights_only=False)
                require(interrupted['cursor'] == {'epoch':1,'next_batch':0,'global_step':160}, 'Durable pre-exit checkpoint differs')
                del interrupted
            else:
                verify_snapshot(base/arm/('continuous' if phase=='continuous' else 'interrupted'),phase_proof['completed_snapshot'])
                require(phase_proof['checkpoint']['global_step'] == 162, 'Wrong completed engineering step')
        checked[arm] = {'full_state_bitwise_equal':True,'history_sha256':item['history_sha256'],
            'durable_step':160,'completed_step':162}
        del first,second,payloads
        guard.check()
    checks = proof['cross_arm_checks']
    require(set(checks) == {'same_data_fingerprint','same_training_sample_trajectory','same_learning_rate_trajectory',
        'same_reset_optimization_and_rng_state','both_full_state_bitwise_resume'} and all(checks.values()), 'Cross-arm recovery gate failed')
    for key in ('data_fingerprint','samples_sha256','learning_rates'):
        require(proof['arms']['rgb'][key] == proof['arms']['rgbd'][key], 'Paired engineering trajectory differs')
    accounting = engineering_attempt_accounting(root,binding,proof['engineering_update_accounting'])
    return {'report_sha256':storage.digest(proof_file),'arms':checked,'engineering_update_accounting':accounting,
        'initial_head_forward_repeated_by_acceptance':False,'initial_head_comparison_is_retained_native_proof':True,
        'real_power_cut_tested':False}


def validate_evaluation_work(record,arm,phase):
    require(record['schema'] == 16 and record['arm'] == arm and record['expected_frames'] == 480
        and record['new_training_updates'] == 0, 'Evaluation invocation identity differs')
    names = ('new_frames','reused_frames','committed_frames','new_model_forward_calls')
    require(all(type(record[name]) is int and 0 <= record[name] <= 480 for name in names), 'Invalid evaluation work counts')
    require(record['new_frames']+record['reused_frames'] == record['committed_frames']
        and record['new_model_forward_calls'] <= record['new_frames'], 'Evaluation reuse/forward arithmetic differs')
    if phase == 'probe':
        require(record['status'] == 'partial' and record['committed_frames'] == 2
            and record['completion_sha256'] is None and record['report_sha256'] is None, 'Probe exceeded the frozen two-frame boundary')
    else:
        require(record['committed_frames'] == 480 and record['status'] in ('completed','verified_completed_read_only'), 'Evaluation did not complete all frames')
    if phase == 'read_only' or record['status'] == 'verified_completed_read_only':
        require(record['status'] == 'verified_completed_read_only' and record['new_frames'] == record['new_model_forward_calls'] == 0
            and record['reused_frames'] == 480, 'Read-only evaluation recomputed a model frame')


def inspect_evaluation_recovery(root,protocol,config_path,records):
    file = storage.path(root,protocol['outputs']['evaluation_resume_report'])
    proof = storage.read(file)
    binding = {'protocol_sha256':storage.digest(config_path),
        'wrapper_sha256':storage.digest(root/'scripts/run_segmentation_baseline_v16.py'),
        'child_sha256':storage.digest(root/'scripts/evaluate_segmentation_baseline_v16.py'),
        'interpreter_sha256':storage.digest(storage.path(root,protocol['interpreter'])),
        'pipeline_sha256':storage.digest(root/'scripts/pipeline.py')}
    require(proof['schema'] == 16 and proof['phase'] == 'evaluation' and proof['status'] == 'passed'
        and proof['binding'] == binding and set(proof['arms']) == set(ARMS), 'Evaluation recovery proof identity differs')
    summaries = {}
    for arm in ARMS:
        item = proof['arms'][arm]
        require(item['status'] == 'passed' and item['byte_and_mtime_identical'] is True
            and item['read_only_check_passed'] is True and item['prefix_before'] == item['prefix_after'], 'Prefix/read-only proof failed')
        directory = storage.path(root,protocol['outputs']['evaluation_root'])/arm
        prefix = {directory/'binding.json'}
        for record in records['train'][:2]:
            parent = storage.frame_directory(directory,{'partition':'train','scene':record['scene'],'frame':record['frame']})
            prefix.update(parent/name for name in ('prediction.npz','prediction.npz.sha256.json','record.json','record.json.sha256.json'))
        require(set(item['prefix_before']) == {p.relative_to(root).as_posix() for p in prefix}, 'Probe file membership differs from exactly two complete frames')
        verify_snapshot(root,item['prefix_before'])
        verify_snapshot(root,item['completed_snapshot'],directory)
        for phase,key in (('probe','probe_record'),('completion','completion_record'),('read_only','read_only_record')):
            validate_evaluation_work(item[key],arm,phase)
            resources = item[key]['resources']
            limits = protocol['resources']
            require(all(type(resources[name]) in (int,float) and math.isfinite(resources[name]) and resources[name] >= 0 for name in (
                'elapsed_seconds','peak_rss_bytes','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','output_bytes'))
                and resources['elapsed_seconds'] <= limits['evaluation_seconds_per_invocation']
                and resources['peak_rss_bytes'] <= limits['rss_soft_bytes']
                and resources['peak_gpu_allocated_bytes'] <= limits['gpu_allocated_soft_bytes']
                and resources['peak_gpu_reserved_bytes'] <= limits['gpu_reserved_soft_bytes']
                and resources['output_bytes'] <= limits['output_limit_bytes'], 'Recorded evaluation resource limit failed')
        for key in ('completion_sha256','report_sha256'):
            expected = storage.digest(directory/('completion.json' if key=='completion_sha256' else 'report.json'))
            require(item[key] == item['completion_record'][key] == item['read_only_record'][key] == expected, 'Recovery completion/report seal differs')
        require(item['completion_record']['reused_frames'] >= 2, 'Evaluation lost its frozen prefix')
        summaries[arm] = {'prefix_frame_count':2,'complete_frame_count':480,'prefix_bytes_and_mtime_preserved':True,
            'read_only_model_forward_calls':0,'completion_sha256':item['completion_sha256'],'report_sha256':item['report_sha256'],
            'recorded_new_model_forward_calls':item['probe_record']['new_model_forward_calls']+item['completion_record']['new_model_forward_calls'],
            'count_scope':'Recorded wrapper invocations; committed prediction count is separate from any interrupted uncommitted call'}
    total_sha=storage.digest(storage.path(root,protocol['outputs']['evaluation_report']))
    require(proof['aggregate_report_sha256'] == total_sha and proof['aggregate_read_only_unchanged'] is True,
        'Aggregate report recovery seal differs')
    for key,status in (('aggregate_record','completed'),('aggregate_read_only_record','verified_completed_read_only')):
        record=proof[key]
        require(record['schema'] == 16 and record['status'] == status and record['aggregate_report_sha256'] == total_sha
            and record['new_model_forward_calls'] == record['new_training_updates'] == 0, 'Aggregate recovery record differs')
    return {'report_sha256':storage.digest(file),'arms':summaries,'physical_power_cut_tested':False}


def inspect_training_summary(root,protocol,config_path,sources):
    file=storage.path(root,protocol['outputs']['training_report'])
    arms={}
    for arm in ARMS:
        source=sources[arm]; directory=source['directory']
        arms[arm]={'run_directory':directory.relative_to(root).as_posix(),'result':source['values']['result.json'],
            'result_sha256':storage.digest(directory/'result.json'),'config_sha256':storage.digest(directory/'config.json'),
            'history_sha256':storage.digest(directory/'history.json'),'initialization_sha256':storage.digest(directory/'initialization.json'),
            'checkpoint_sha256':source['identity']['checkpoint_sha256']}
    expected={'schema':16,'status':'completed','protocol_sha256':storage.digest(config_path),
        'resume_proof_sha256':storage.digest(storage.path(root,protocol['outputs']['training_resume_report'])),
        'new_research_updates':3200,'arms':arms,'automatic_extension':False,'validation_evaluated':False,'efficacy_gate':False,
        'scope':'Fixed matched-budget paired development baseline'}
    require(storage.read(file) == expected,'Training aggregate is inconsistent with both final arms')


def scientific_png(report):
    image = Image.new('RGB',(1640,1100),'white')
    draw = ImageDraw.Draw(image)
    fonts = {size:ImageFont.load_default(size=size) for size in (17,19,22,26,34)}
    ink,muted,grid = '#202d3b','#526476','#dce4ec'
    colors = {'rgb':'#376b9b','rgbd':'#cb8033'}
    draw.text((50,28),'Paired RGB / RGB-D development baseline',font=fonts[34],fill=ink)
    draw.text((50,76),'Final step 1600 per arm  |  one seed  |  320 training + 160 development frames per arm',font=fonts[19],fill=muted)
    for index,metric in enumerate(('instance_f1_iou50','foreground_pixel_iou')):
        left = 70+index*800
        draw.text((left,123),'Instance F1 at mask IoU >= 0.50' if index==0 else 'Foreground pixel IoU',font=fonts[26],fill=ink)
        x0,y0,width,height = left+48,440,640,240
        for tick in range(6):
            y = y0-height*tick/5
            draw.line((x0,y,x0+width,y),fill=grid,width=1)
            draw.text((left,y-9),'%.1f'%(tick/5),font=fonts[17],fill=muted)
        for split_index,split in enumerate(PARTITIONS):
            center = x0+170+split_index*310
            for arm_index,arm in enumerate(ARMS):
                value = report['arms'][arm]['partitions'][split]['scene_macro'][metric]['value']
                x = center-78+arm_index*80
                draw.rectangle((x,y0-height*value,x+55,y0),fill=colors[arm])
                draw.text((x-4,y0-height*value-27),'%.4f'%value,font=fonts[19],fill=ink)
            draw.text((center-75,y0+16),'Train: 20 scenes' if split=='train' else 'Develop: 10 scenes',font=fonts[19],fill=ink)
        delta = report['paired']['development']['scene_macro_delta_rgbd_minus_rgb'][metric]['value']
        draw.text((left,492),'Development difference (RGB-D minus RGB): %+.4f'%delta,font=fonts[22],fill=ink)
    for index,arm in enumerate(ARMS):
        x=650+index*160
        draw.rectangle((x,546,x+23,563),fill=colors[arm]); draw.text((x+34,540),'RGB' if arm=='rgb' else 'RGB-D',font=fonts[22],fill=ink)
    draw.text((70,594),'Paired development-scene differences',font=fonts[26],fill=ink)
    draw.text((70,633),'Scene',font=fonts[19],fill=muted)
    draw.text((430,633),'Instance F1 difference',font=fonts[19],fill=muted)
    draw.text((1100,633),'Foreground IoU difference',font=fonts[19],fill=muted)
    for index,item in enumerate(report['paired']['development']['scenes']):
        y=675+index*29
        draw.text((75,y-9),'%04d'%item['scene'],font=fonts[19],fill=ink)
        for center,name in ((600,'instance_f1_iou50_delta'),(1280,'foreground_pixel_iou_delta')):
            value=item[name]['value']
            draw.line((center-240,y,center+240,y),fill=grid,width=1)
            draw.line((center,y-9,center,y+9),fill=muted,width=1)
            endpoint=center+value*240
            draw.line((center,y,endpoint,y),fill=colors['rgbd'] if value>=0 else colors['rgb'],width=7)
            draw.text((center+255,y-10),'%+.4f'%value,font=fonts[19],fill=ink)
    draw.text((70,985),'Scene macro averages pooled per-scene counts equally. All ten development scenes were previously inspected.',font=fonts[19],fill=muted)
    draw.text((70,1019),'Negative differences are valid results. No efficacy gate, AP, independent test, or full P4 completion claim.',font=fonts[19],fill=muted)
    draw.text((70,1053),'Cached masks independently recounted; neural forward passes are source-bound and are not independently repeated.',font=fonts[17],fill=muted)
    stream=io.BytesIO(); image.save(stream,format='PNG',optimize=False)
    return stream.getvalue()


def publication_inventory(output,figure,read_only):
    expected={output,figure}
    for directory in (output.parent,figure.parent):
        if not directory.exists():
            continue
        for file in directory.iterdir():
            if not file.is_file():
                continue
            if file in expected:
                continue
            match=re.fullmatch(r'(.+)\.([0-9a-f]{32})\.tmp',file.name)
            if match and file.with_name(match.group(1)) in expected:
                require(not read_only and not output.exists(),'Temporary file beside completed acceptance')
                file.unlink()
            elif directory==figure.parent:
                raise ValueError('Unexpected acceptance visual artifact: '+str(file))


def locked_acceptance(root,protocol,config_path,check_only,guard):
    output=storage.path(root,protocol['outputs']['acceptance_report'])
    figure=storage.path(root,protocol['outputs']['visuals_dir'])/FIGURE
    read_only=check_only or output.exists()
    publication_inventory(output,figure,read_only)
    require(not check_only or output.exists() and figure.exists(),'Read-only acceptance requires complete output')
    manifest,records=storage.verified_manifest(root,protocol)
    sources={arm:storage.training_sources(root,protocol,config_path,arm,manifest) for arm in ARMS}
    watched={storage.path(root,name) for group in ('input_hashes','source_hashes') for name in protocol[group]}
    watched.update(storage.path(root,name) for source in sources.values() for name in source['files'])
    for name in ('evaluation_root','evaluation_report','evaluation_resume_report','training_resume_report','training_report'):
        target=storage.path(root,protocol['outputs'][name])
        watched.update(p for p in target.rglob('*') if p.is_file()) if target.is_dir() else watched.add(target)
    watched.add(storage.receipt_path(storage.path(root,protocol['outputs']['training_resume_report'])))
    watched.update(p for p in (root/'runtime/verification/baseline_v16').rglob('*') if p.is_file())
    before=storage.snapshot(root,watched)
    training=inspect_training(root,protocol,sources,records,guard)
    inspect_training_summary(root,protocol,config_path,sources)
    gpu_recovery=inspect_gpu_recovery(root,protocol,config_path,sources,records,guard)
    evaluation=inspect_evaluation(root,protocol,config_path,sources,records,guard)
    evaluation_recovery=inspect_evaluation_recovery(root,protocol,config_path,records)
    binding_body={'schema':16,'phase':'independent_acceptance','protocol_sha256':storage.digest(config_path),
        'inputs':{name:item['sha256'] for name,item in before.items()},'runtime':protocol['runtime'],
        'interpreter_sha256':storage.digest(storage.path(root,protocol['interpreter']))}
    binding=dict(binding_body,fingerprint=storage.fingerprint(binding_body))
    image=scientific_png(evaluation)
    report={'schema':16,'status':'accepted_complete_paired_development_baseline','binding':binding,
        'binding_fingerprint':binding['fingerprint'],'training':training,'gpu_resume':gpu_recovery,
        'evaluation_resume':evaluation_recovery,'arms':evaluation['arms'],'paired':evaluation['paired'],
        'evaluation_report_sha256':storage.digest(storage.path(root,protocol['outputs']['evaluation_report'])),
        'independently_recounted_model_frames':960,'recounted_training_frames_per_arm':320,'recounted_development_frames_per_arm':160,
        'matching':'Independent integer IoU >= 1/2 adjacency and augmenting-path maximum cardinality',
        'metrics':'Exact fractions; counts pooled within scenes, then equal scene mean; micro retained',
        'new_model_forward_calls':0,'new_training_updates':0,'independent_neural_forward_reproduction':False,
        'source_bound_cached_predictions':True,'all_development_scenes_included':True,'efficacy_gate':False,
        'method_superiority_established':False,'independent_confirmation':False,'mask_ap_computed':False,
        'full_P4_research_stage_complete':False,'physical_power_cut_tested':False,
        'figure':{'path':figure.relative_to(root).as_posix(),'sha256':hashlib.sha256(image).hexdigest()},
        'limitations':protocol['limitations']}
    guard.check(len(image)+len(storage.json_bytes(report)))
    require(before==storage.snapshot(root,watched),'Frozen sources or training/evaluation outputs changed during acceptance')
    storage.publish(figure,image,read_only)
    storage.publish(output,storage.json_bytes(report),read_only)
    return {'schema':16,'status':'verified_completed_read_only' if read_only else 'passed',
        'report_sha256':storage.digest(output),'figure_sha256':storage.digest(figure),
        'independently_recounted_model_frames':960,'new_model_forward_calls':0,'new_training_updates':0,
        'resources':guard.check()}


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=storage.CONFIG)
    parser.add_argument('--check-only',action='store_true')
    return parser.parse_args(argv)


def run(args):
    root=ROOT
    config_path=storage.path(root,args.config)
    protocol=storage.read(config_path)
    guard=storage.Guard(root,protocol,'acceptance'); guard.check()
    storage.verify_protocol(root,protocol,config_path)
    with storage.lock(root/'runtime/locks/segmentation_baseline_acceptance_v16'):
        result=locked_acceptance(root,protocol,config_path,args.check_only,guard)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    return result


if __name__=='__main__':
    try:
        run(arguments())
    except (ValueError,RuntimeError,OSError,KeyError) as error:
        print('Baseline acceptance stopped: '+str(error),file=sys.stderr,flush=True)
        sys.exit(2)

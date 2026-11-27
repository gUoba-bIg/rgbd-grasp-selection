"""Independent decoder, mask provenance and primary COCO-style AP audit."""
import argparse
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
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TOLERANCE = 1e-12
IOU_THRESHOLDS = np.linspace(.5,.95,10)
RECALL_THRESHOLDS = np.linspace(0.,1.,101)
ARMS = ('rgb','rgbd')
FIGURE = 'paired_development_mask_ap.png'


def require(condition,message):
    if not condition:
        raise ValueError(message)


def close_arrays(actual,expected,message):
    actual,expected=np.asarray(actual),np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
        and np.allclose(actual,expected,rtol=0,atol=TOLERANCE),message)


def independent_decode(foreground_probability,center_probability,offsets,decoder=None):
    """Rebuild decoder decisions with NumPy, without calling the source decoder."""
    policy={'foreground_threshold':.5,'center_threshold':.35,'min_pixels':20,'max_centers':64}
    if decoder is not None:
        require(decoder == policy,'Unexpected frozen decoder')
    foreground_probability=np.asarray(foreground_probability)
    center_probability=np.asarray(center_probability)
    offsets=np.asarray(offsets)
    require(foreground_probability.ndim == 2 and foreground_probability.dtype == center_probability.dtype == offsets.dtype == np.float32
        and center_probability.shape == foreground_probability.shape and offsets.shape == (2,*foreground_probability.shape),
        'Invalid probability/offset cache shape or precision')
    require(all(np.isfinite(value).all() for value in (foreground_probability,center_probability,offsets))
        and np.all((foreground_probability >= 0)&(foreground_probability <= 1))
        and np.all((center_probability >= 0)&(center_probability <= 1)), 'Nonfinite or out-of-range probability cache')
    height,width=foreground_probability.shape
    foreground=foreground_probability >= np.float32(.5)
    padded=np.pad(center_probability,4,constant_values=-np.inf)
    local_maximum=np.lib.stride_tricks.sliding_window_view(padded,(9,9)).max(axis=(-2,-1))
    peaks=np.argwhere((center_probability == local_maximum)&(center_probability >= np.float32(.35)))
    if len(peaks):
        confidence=center_probability[peaks[:,0],peaks[:,1]]
        order=np.argsort(-confidence,kind='stable')
    else:
        order=[]
    selected=[]
    for index in order:
        row,col=map(int,peaks[index])
        if all((row-y)**2+(col-x)**2 >= 25 for y,x in selected):
            selected.append((row,col))
            if len(selected)==64:
                break
    instances=np.zeros((height,width),np.int32)
    retained=[]
    yy,xx=np.nonzero(foreground)
    if selected and len(yy):
        target_y=yy.astype(np.float64)+(offsets[1,yy,xx]*np.float32(height)).astype(np.float64)
        target_x=xx.astype(np.float64)+(offsets[0,yy,xx]*np.float32(width)).astype(np.float64)
        centers=np.asarray(selected,np.int64)
        nearest=np.empty(len(yy),np.int32)
        for start in range(0,len(yy),4096):
            stop=start+4096
            distances=(target_y[start:stop,None]-centers[None,:,0])**2+(target_x[start:stop,None]-centers[None,:,1])**2
            nearest[start:stop]=np.argmin(distances,axis=1)
        for selected_index,(row,col) in enumerate(selected):
            member=nearest == selected_index
            if np.count_nonzero(member) >= 20:
                instance_id=len(retained)+1
                instances[yy[member],xx[member]]=instance_id
                retained.append({'instance_id':instance_id,'selected_center_index':selected_index,
                    'row':row,'col':col,'score':float(center_probability[row,col])})
    return {'instances':instances,'foreground':foreground,'selected_centers':[[row,col] for row,col in selected],
        'retained_instances':retained}


def original_resolution_instances(low):
    require(low.dtype == np.int32 and low.shape == (240,320),'Unexpected prediction grid')
    repeated=np.repeat(np.repeat(low,3,axis=0),4,axis=1)
    nearest=np.asarray(Image.fromarray(low).resize((1280,720),Image.Resampling.NEAREST),np.int32)
    require(np.array_equal(repeated,nearest),'Nearest interpolation and integer replication differ')
    return repeated


def decode_rle(rle):
    """Decode signed delta/5-bit compressed COCO counts and Fortran-order runs."""
    require(set(rle)=={'size','counts'} and isinstance(rle['size'],list) and len(rle['size'])==2
        and all(type(v) is int and v > 0 for v in rle['size']), 'Invalid RLE dimensions')
    height,width=rle['size']
    require(height*width <= 720*1280,'Unexpected RLE image budget')
    encoded=rle['counts']
    require(isinstance(encoded,str) and encoded,'Expected compressed ASCII RLE')
    runs=[]; position=0
    while position<len(encoded):
        value=0; bits=0
        while True:
            require(position<len(encoded) and bits<=60,'Truncated or overlong RLE integer')
            code=ord(encoded[position])-48; position+=1
            require(0<=code<=63,'Noncanonical RLE character')
            value|=(code&31)<<bits; bits+=5
            if not code&32:
                if code&16:
                    value-=1<<bits
                break
        if len(runs)>=3:
            value+=runs[-2]
        require(0<=value<=height*width,'Invalid RLE run length')
        runs.append(value)
        require(len(runs)<=height*width+1,'Excess RLE runs')
    require(sum(runs)==height*width,'RLE run sum differs from image area')
    flat=np.repeat(np.arange(len(runs),dtype=np.int64)%2,np.asarray(runs,np.int64)).astype(bool)
    return flat.reshape((height,width),order='F')


def dense_overlap(prediction,reference,predicted_ids,reference_ids):
    """Integer contingency counts from dense labels, independent of RLE overlap code."""
    require(prediction.shape == reference.shape and prediction.ndim == 2,'Dense overlap image shape differs')
    pids,pinverse,pc=np.unique(prediction,return_inverse=True,return_counts=True)
    gids,ginverse,gc=np.unique(reference,return_inverse=True,return_counts=True)
    intersection=np.bincount(pinverse*len(gids)+ginverse,minlength=len(pids)*len(gids)).reshape(len(pids),len(gids))
    pslots={int(value):index for index,value in enumerate(pids)}
    gslots={int(value):index for index,value in enumerate(gids)}
    require(all(value>0 and value in pslots for value in predicted_ids)
        and all(value>0 and value in gslots for value in reference_ids),'Missing positive instance in dense masks')
    selected=intersection[np.ix_([pslots[value] for value in predicted_ids],[gslots[value] for value in reference_ids])]
    pa=np.asarray([pc[pslots[value]] for value in predicted_ids],np.int64)
    ga=np.asarray([gc[gslots[value]] for value in reference_ids],np.int64)
    union=pa[:,None]+ga[None,:]-selected
    overlap=np.divide(selected,union,out=np.zeros(selected.shape,np.float64),where=union>0)
    return {'intersection':selected,'union':union,'iou':overlap,'predicted_areas':pa,'reference_areas':ga}


def independent_image_matching(image_id,gt_ids,dt_ids,scores,ious,max_detections=100):
    """COCO noncrowd all-area greedy assignment; equal IoU selects later GT."""
    require(type(image_id) is int and image_id>0 and len(dt_ids)==len(scores)
        and len(set(gt_ids))==len(gt_ids) and len(set(dt_ids))==len(dt_ids)
        and all(type(value) is int and value>0 for value in gt_ids+dt_ids),'Invalid annotation identities')
    scores=np.asarray(scores,np.float64); ious=np.asarray(ious,np.float64)
    require(ious.shape==(len(dt_ids),len(gt_ids)) and np.isfinite(ious).all() and np.isfinite(scores).all()
        and np.all((ious>=0)&(ious<=1)),'Invalid IoU or score matrix')
    if not len(gt_ids) and not len(dt_ids):
        return None
    order=np.argsort(-scores,kind='stable')[:max_detections]
    ids=[dt_ids[int(i)] for i in order]
    overlap=ious[order]
    matches=np.zeros((10,len(ids)),np.int64)
    truth_matches=np.zeros((10,len(gt_ids)),np.int64)
    for ti,threshold in enumerate(IOU_THRESHOLDS):
        for di,det_id in enumerate(ids):
            best=-1; best_overlap=float(threshold)
            for gi,gt_id in enumerate(gt_ids):
                if truth_matches[ti,gi]:
                    continue
                candidate=float(overlap[di,gi])
                if candidate>=best_overlap:
                    best=gi; best_overlap=candidate
            if best>=0:
                matches[ti,di]=gt_ids[best]
                truth_matches[ti,best]=det_id
    return {'image_id':image_id,'category_id':1,'aRng':[0,10000000000.0],'maxDet':max_detections,
        'dtIds':ids,'gtIds':list(gt_ids),'dtMatches':matches.tolist(),'gtMatches':truth_matches.tolist(),
        'dtScores':scores[order].tolist(),'gtIgnore':[0]*len(gt_ids),'dtIgnore':np.zeros(matches.shape,bool).tolist()}


def independent_ap(image_results):
    """Stable pooled ranking, monotone precision envelope, 101 recall samples."""
    items=[item for item in image_results if item is not None]
    require([item['image_id'] for item in items] == sorted(item['image_id'] for item in items),'AP image order differs')
    ntruth=sum(len(item['gtIds']) for item in items)
    precision=np.full((10,101),-1.,np.float64)
    recalls=np.full(10,-1.,np.float64)
    score_grid=np.full((10,101),-1.,np.float64)
    if ntruth:
        confidence=np.asarray([score for item in items for score in item['dtScores']],np.float64)
        ordered=np.argsort(-confidence,kind='stable')
        match=np.concatenate([np.asarray(item['dtMatches'],np.int64).reshape(10,len(item['dtIds'])) for item in items],axis=1)[:,ordered]
        ordered_scores=confidence[ordered]
        for index in range(10):
            truth=np.cumsum(match[index]>0).astype(np.float64)
            false=np.cumsum(match[index]==0).astype(np.float64)
            recall=truth/ntruth
            observed=truth/(truth+false+np.spacing(1))
            envelope=np.maximum.accumulate(observed[::-1])[::-1]
            precision[index].fill(0.); score_grid[index].fill(0.)
            recalls[index]=recall[-1] if len(recall) else 0.
            positions=np.searchsorted(recall,RECALL_THRESHOLDS,side='left')
            covered=positions<len(recall)
            precision[index,covered]=envelope[positions[covered]]
            score_grid[index,covered]=ordered_scores[positions[covered]]
    summary={'AP':float(precision.mean()) if ntruth else None,
        'AP50':float(precision[0].mean()) if ntruth else None,'AP75':float(precision[5].mean()) if ntruth else None,
        'reason':None if ntruth else 'no_ground_truth'}
    return {'precision':precision,'recall':recalls,'scores':score_grid,'summary':summary,
        'ground_truth_count':ntruth,'detection_count_after_maxdet':sum(len(item['dtIds']) for item in items)}


def common():
    from scripts import mask_ap_common_v17
    return mask_ap_common_v17


def compact_fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def read_parent(root,protocol):
    """Reuse frozen R16 storage validation only; no model or metric call."""
    from scripts import evaluate_segmentation_baseline_v16 as parent_storage
    store=common()
    parent_file=store.safe(root,protocol['parent_protocol'])
    acceptance_file=store.safe(root,protocol['parent_acceptance_report'])
    require(store.digest(parent_file)==protocol['parent_protocol_sha256']
        and store.digest(acceptance_file)==protocol['parent_acceptance_sha256'],'Frozen parent identity changed')
    parent,accepted=store.read(parent_file),store.read(acceptance_file)
    require(parent['schema']==accepted['schema']==16 and accepted['status']=='accepted_complete_paired_development_baseline',
        'Parent paired baseline has not passed independent acceptance')
    require(accepted['binding']['protocol_sha256']==store.digest(parent_file),'Parent acceptance binds another protocol')
    for group in ('source_hashes','input_hashes'):
        for name,checksum in parent[group].items():
            require(store.digest(store.safe(root,name))==checksum,'Parent source or data changed')
    require(all(parent[key]==protocol[key] for key in ('validation_scenes','frames','input_size','decoder','camera')),
        'Child changed the parent frame selection or decoder')
    manifest,partitions=parent_storage.verified_manifest(root,parent)
    sources={arm:parent_storage.training_sources(root,parent,parent_file,arm,manifest) for arm in ARMS}
    bindings,seals={},{}
    for arm in ARMS:
        directory=store.safe(root,parent['outputs']['evaluation_root'])/arm
        binding=parent_storage.make_binding(root,parent,parent_file,arm,sources)
        require(store.read(directory/'binding.json')==binding,'Parent cache binding differs')
        parent_storage.verify_completion(directory,binding,partitions)
        bindings[arm]=binding
        seals[arm]={'binding_fingerprint':binding['fingerprint'],'completion_sha256':store.digest(directory/'completion.json'),
            'report_sha256':store.digest(directory/'report.json')}
    require(len(partitions['development'])==160,'Incomplete development support')
    return {'protocol':parent,'protocol_file':parent_file,'records':partitions['development'],
        'sources':sources,'evaluation_bindings':bindings,'evaluation_seals':seals}


def score_binding(root,protocol,config,arm,context):
    store=common()
    body={'schema':17,'phase':'scores','arm':arm,'protocol_sha256':store.digest(config),
        'source_hashes':protocol['source_hashes'],'input_hashes':protocol['input_hashes'],
        'parent_protocol_sha256':protocol['parent_protocol_sha256'],'parent_acceptance_sha256':protocol['parent_acceptance_sha256'],
        'parent_training_files':{name:sha for source in context['sources'].values() for name,sha in source['files'].items()},
        'parent_evaluation_seals':context['evaluation_seals'],
        'final_checkpoints':{name:source['identity'] for name,source in context['sources'].items()},
        'runtime':protocol['runtime'],'interpreter_sha256':store.digest(store.safe(root,protocol['interpreter'])),
        'decoder':protocol['decoder'],'input_size':protocol['input_size'],'precision':'float32_without_autocast',
        'expected_frames':160,'score_definition':'Float32 sigmoid at the selected center, retained through small-mask ID reassignment'}
    return dict(body,fingerprint=store.fingerprint(body))


def verify_manifest(directory,binding,expected_files=None):
    store=common(); directory=Path(directory)
    files={p for p in directory.rglob('*') if p.is_file()}
    require(all(not p.is_symlink() for p in directory.rglob('*')),'Link in sealed output inventory')
    if expected_files is not None:
        require(files==set(expected_files),'Output inventory has missing or extra files')
    completion=store.load_commit(directory/'completion.json',binding)
    excluded={directory/'completion.json',store.receipt_path(directory/'completion.json')}
    require(completion['schema']==17 and completion['status']=='completed' and completion['binding_fingerprint']==binding['fingerprint']
        and completion['outputs']=={p.relative_to(directory).as_posix():store.digest(p) for p in files-excluded}
        and completion['report_sha256']==store.digest(directory/'report.json'),'Completion manifest differs')
    return completion


def score_attempts(frame,binding,identity,checkpoint_sha256,heads_file):
    store=common(); ledger=frame/'attempts'
    files=set(); returned=exceptional=unknown=0; input_hashes=set(); corresponding=False
    intents=sorted(ledger.glob('attempt_*.intent.json'))
    require(intents,'Prediction lacks a saved forward intent')
    for index,file in enumerate(intents):
        require(file.name=='attempt_%04d.intent.json'%index,'Forward attempt sequence has a gap')
        intent=store.load_commit(file,binding)
        require(intent['schema']==17 and intent['phase']=='model_forward' and intent['arm']==binding['arm']
            and intent['identity']==identity and intent['attempt_index']==index and intent['binding_fingerprint']==binding['fingerprint']
            and intent['checkpoint_sha256']==checkpoint_sha256 and re.fullmatch('[0-9a-f]{64}',intent['input_tensor_sha256']),
            'Forward intent source identity differs')
        input_hashes.add(intent['input_tensor_sha256'])
        files.update((file,store.receipt_path(file)))
        outcome_file=file.with_name(file.name.replace('.intent.json','.outcome.json'))
        if outcome_file.exists():
            outcome=store.load_commit(outcome_file,binding)
            require(all(outcome.get(k)==v for k,v in intent.items()),'Forward outcome changed its source intent')
            files.update((outcome_file,store.receipt_path(outcome_file)))
            if outcome['status']=='model_returned':
                returned+=1
                require(re.fullmatch('[0-9a-f]{64}',outcome['heads_sha256']) and type(outcome['heads_bytes']) is int
                    and outcome['heads_bytes']>0,'Invalid returned-head evidence')
                corresponding=corresponding or (outcome['heads_sha256']==store.digest(heads_file) and outcome['heads_bytes']==heads_file.stat().st_size)
            else:
                require(outcome['status']=='model_exception','Unknown model outcome state')
                exceptional+=1
        else:
            unknown+=1
            pending=store.receipt_path(outcome_file)
            if pending.exists():
                receipt=store.read(pending)
                require(receipt['schema']==17 and receipt['binding_fingerprint']==binding['fingerprint']
                    and re.fullmatch('[0-9a-f]{64}',receipt['sha256']) and type(receipt['bytes']) is int and receipt['bytes']>0,
                    'Invalid interrupted outcome publication intent')
                files.add(pending)
    require(len(input_hashes)==1 and corresponding,'Sealed heads have no matching model-return evidence or inputs changed')
    require(files=={p for p in ledger.iterdir() if p.is_file()},'Foreign or unbound attempt artifact')
    return {'intents_committed':len(intents),'model_calls_returned':returned,'model_exceptions_observed':exceptional,
        'unknown_outcome_intents':unknown,'known_model_call_lower_bound':returned+exceptional,'exact_call_count_known':unknown==0},files


def inspect_scores(root,protocol,config,context,guard):
    from scripts import evaluate_segmentation_baseline_v16 as parent_storage
    store=common(); outputs={}; seals={}
    for arm in ARMS:
        directory=store.safe(root,protocol['outputs']['scores_root'])/arm
        binding=score_binding(root,protocol,config,arm,context)
        require(store.read(directory/'binding.json')==binding,'Score extraction binding differs')
        expected={directory/'binding.json'}
        for name in ('report.json','completion.json'):
            expected.update((directory/name,store.receipt_path(directory/name)))
        rows=[]; frames={}
        for source in context['records']:
            identity={'partition':'development','scene':source['scene'],'frame':source['frame'],'camera':'realsense'}
            image_id=source['scene']*1000+int(source['frame'])
            frame=directory/'frames'/('scene_%04d_%s'%(source['scene'],source['frame']))
            for name in ('heads.npz','prediction.npz','record.json'):
                file=frame/name; store.verify_bytes(file,binding)
                expected.update((file,store.receipt_path(file)))
            with np.load(frame/'heads.npz',allow_pickle=False) as cache:
                require(set(cache.files)=={'foreground_probability','center_probability','offsets'},'Unexpected cached probability heads')
                rebuilt=independent_decode(cache['foreground_probability'],cache['center_probability'],cache['offsets'],protocol['decoder'])
                centers=[{'selected_center_index':index,'row':point[0],'col':point[1],
                    'score':float(cache['center_probability'][point[0],point[1]])} for index,point in enumerate(rebuilt['selected_centers'])]
                require(cache['foreground_probability'].shape==(240,320),'Wrong cached neural output resolution')
            with np.load(frame/'prediction.npz',allow_pickle=False) as prediction:
                require(set(prediction.files)=={'instances','foreground'} and prediction['instances'].dtype==np.int32
                    and prediction['foreground'].dtype==np.bool_
                    and np.array_equal(prediction['instances'],rebuilt['instances'])
                    and np.array_equal(prediction['foreground'],rebuilt['foreground']),'Independent decoder differs from sealed prediction')
            parent_dir=store.safe(root,context['protocol']['outputs']['evaluation_root'])/arm
            parent_file=parent_storage.frame_directory(parent_dir,identity)/'prediction.npz'
            parent_instances,parent_foreground=parent_storage.load_prediction(parent_file,context['evaluation_bindings'][arm])
            require(np.array_equal(parent_instances,rebuilt['instances']) and np.array_equal(parent_foreground,rebuilt['foreground']),
                'Score-extraction prediction differs from immutable R16 prediction')
            accounting,attempt_files=score_attempts(frame,binding,identity,context['sources'][arm]['identity']['checkpoint_sha256'],frame/'heads.npz')
            expected.update(attempt_files)
            row={'schema':17,'arm':arm,'identity':identity,'binding_fingerprint':binding['fingerprint'],
                'heads_sha256':store.digest(frame/'heads.npz'),'parent_prediction_sha256':store.digest(parent_file),
                'checkpoint_sha256':context['sources'][arm]['identity']['checkpoint_sha256'],
                'instances':rebuilt['retained_instances'],'selected_centers':centers,'predicted_instance_count':len(rebuilt['retained_instances']),
                'parent_instances_and_foreground_bitwise_equal':True,'model_inputs_and_scores_use_ground_truth':False,
                'training_updates':0,'forward_attempt_accounting':accounting,'prediction_sha256':store.digest(frame/'prediction.npz')}
            require(store.load_commit(frame/'record.json',binding)==row,'Independent center score/ID map or attempt accounting differs')
            rows.append(row)
            frames[image_id]={'source':source,'row':row,'prediction_file':frame/'prediction.npz'}
            if len(rows)%16==0:
                guard.check()
        completion=verify_manifest(directory,binding,expected)
        require(completion['arm']==arm and completion['expected_frames']==160,'Score completion support differs')
        accounting={key:sum(row['forward_attempt_accounting'][key] for row in rows) for key in
            ('intents_committed','model_calls_returned','model_exceptions_observed','unknown_outcome_intents','known_model_call_lower_bound')}
        accounting['exact_call_count_known']=accounting['unknown_outcome_intents']==0
        report={'schema':17,'status':'completed_development_mask_scores','arm':arm,'binding_fingerprint':binding['fingerprint'],
            'expected_frames':160,'frame_count':160,'predicted_instance_count':sum(row['predicted_instance_count'] for row in rows),
            'rows':rows,'forward_attempt_accounting':accounting,'nominal_model_forward_calls':160,
            'parent_predictions_bitwise_equal':True,'ground_truth_metric_evaluation_performed':False,
            'new_training_updates':0,'scores_are_calibrated_probabilities':False}
        require(store.load_commit(directory/'report.json',binding)==report,'Score summary differs')
        outputs[arm]={'frames':frames,'report':report,'binding':binding}
        seals[arm]={'binding_fingerprint':binding['fingerprint'],'completion_sha256':store.digest(directory/'completion.json'),
            'report_sha256':store.digest(directory/'report.json')}
    return outputs,seals


def verify_base_executable(specification):
    base=Path(specification['path']).resolve()
    require(base.is_file() and common().digest(base)==specification['sha256'],'Base interpreter changed')


def verify_environment(root,protocol):
    import importlib.metadata
    store=common()
    report_file=store.safe(root,protocol['environment_report'])
    require(store.digest(report_file)==protocol['environment_report_sha256'],'Official evaluation environment report changed')
    report=store.read(report_file)
    env_config=store.safe(root,protocol['environment_protocol']); env_protocol=store.read(env_config)
    body={'schema':17,'phase':'coco_environment','protocol_sha256':store.digest(env_config),
        'base_python':env_protocol['base_python'],'source_hashes':env_protocol['source_hashes'],'wheels':env_protocol['wheels']}
    binding=dict(body,fingerprint=compact_fingerprint(body))
    require(report['schema']==17 and report['status']=='passed' and report['binding']==binding
        and report['old_environments_modified'] is False and report['source_compilation_performed'] is False,
        'Environment completion identity differs')
    verify_base_executable(env_protocol['base_python'])
    probe=report['probe']
    require(probe['schema']==17 and probe['status']=='passed' and probe['synthetic_only'] is True
        and probe['research_frames_read']==report['research_frames_read']==0 and probe['gpu_used'] is report['gpu_used'] is False
        and probe['new_model_forward_calls']==report['new_model_forward_calls']==0
        and probe['rle_non_square_round_trip'] is True,'Environment was not verified with the bounded synthetic probe')
    store.verify_bytes(report_file,binding)
    directory=store.safe(root,report['environment'])
    actual={file.relative_to(directory).as_posix():{'sha256':store.digest(file),'bytes':file.stat().st_size}
        for file in directory.rglob('*') if file.is_file()}
    require(actual==report['environment_inventory'],'Official evaluation installation changed')
    require(Path(sys.executable).resolve()==store.safe(root,protocol['evaluation_interpreter']), 'Wrong isolated audit interpreter')
    installed={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()}
    require(installed==report['package_inventory']=={key.lower():value for key,value in env_protocol['expected_packages'].items()},
        'Official evaluation package identity differs')
    require(probe['packages']==report['package_inventory'] and all(probe['runtime'][name]==value
        for name,value in protocol['expected_evaluation_runtime'].items()),'Environment probe runtime differs')
    for name,checksum in env_protocol['source_hashes'].items():
        require(store.digest(store.safe(root,name))==checksum,'Environment setup source changed')
    return report


def expected_parameters():
    return {'iouType':'segm','iouThrs':IOU_THRESHOLDS.tolist(),'recThrs':RECALL_THRESHOLDS.tolist(),
        'maxDets':[1,10,100],'areaRng':[[0,10000000000],[0,1024],[1024,9216],[9216,10000000000]],
        'areaRngLbl':['all','small','medium','large'],'useCats':1,'catIds':[1]}


def metrics_binding(root,protocol,config,score_seals):
    store=common()
    directory=store.safe(root,protocol['outputs']['scores_root'])
    files={file.relative_to(root).as_posix():store.digest(file) for file in directory.rglob('*') if file.is_file()}
    body={'schema':17,'phase':'metrics','protocol_sha256':store.digest(config),'source_hashes':protocol['source_hashes'],
        'input_hashes':protocol['input_hashes'],'score_files':files,
        'score_completion_sha256':{arm:score_seals[arm]['completion_sha256'] for arm in ARMS},
        'environment_report_sha256':protocol['environment_report_sha256'],'runtime':protocol['expected_evaluation_runtime'],
        'interpreter_sha256':store.digest(store.safe(root,protocol['evaluation_interpreter'])),
        'parameters':expected_parameters(),'new_training_updates':0,'new_model_forward_calls':0}
    return dict(body,fingerprint=store.fingerprint(body))


def source_labels(root,protocol,context,source):
    store=common()
    file=store.safe(root,context['protocol']['data_root'])/source['files']['label']
    relative=file.relative_to(root).as_posix()
    require(protocol['input_hashes'].get(relative)==store.digest(file)==source['sha256']['label'],
        'Original development label is not directly frozen')
    with Image.open(file) as opened:
        values=np.array(opened,copy=True)
    require(values.shape==(720,1280) and np.issubdtype(values.dtype,np.integer)
        and np.all((values>=0)&(values<=88)),'Original visible-object labels are invalid')
    return values,relative


def reference_identity(root,protocol,context,guard):
    """Derive lightweight global IDs, areas, and provenance; retain no mask stack."""
    result={}; next_id=1
    for source in context['records']:
        labels,relative=source_labels(root,protocol,context,source)
        image_id=source['scene']*1000+int(source['frame'])
        annotations=[]
        for label in sorted(int(value) for value in np.unique(labels) if value>0):
            yy,xx=np.nonzero(labels==label)
            annotations.append({'id':next_id,'image_id':image_id,'category_id':1,'area':int(len(yy)),'iscrowd':0,
                'bbox':[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)],
                'source':{'scene':source['scene'],'frame':source['frame'],'image_id':image_id,'original_label_id':label,
                    'label_path':relative,'label_sha256':source['sha256']['label']}})
            next_id+=1
        result[image_id]={'image':{'id':image_id,'height':720,'width':1280},'source':source,'annotations':annotations}
        if len(result)%16==0:
            guard.check()
    return result


def compare_metrics(actual,expected,message):
    """Only derived floating metrics use the predeclared tolerance."""
    error=0.
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),message)
        for key in expected:
            error=max(error,compare_metrics(actual[key],expected[key],message+' / '+key))
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),message)
        for first,second in zip(actual,expected):
            error=max(error,compare_metrics(first,second,message))
    elif type(expected) is float:
        require(type(actual) in (int,float) and math.isfinite(actual) and abs(actual-expected)<=TOLERANCE,message)
        error=abs(actual-expected)
    else:
        require(type(actual) is type(expected) and actual==expected,message)
    return error


def inspect_block(root,protocol,arm,scope,directory,binding,score_frames,references,context,guard):
    store=common()
    expected_files={directory/name for name in ('inputs.json','official.json','report.json','completion.json')}
    expected_files.update(store.receipt_path(file) for file in list(expected_files))
    require({p for p in directory.rglob('*') if p.is_file()}==expected_files,'Metric block inventory differs')
    for file in expected_files:
        require(not file.is_symlink(),'Link in metric block')
    completion=store.load_commit(directory/'completion.json',binding)
    payload_files=expected_files-{directory/'completion.json',store.receipt_path(directory/'completion.json')}
    require(completion=={'schema':17,'status':'completed','arm':arm,'scope':scope,'binding_fingerprint':binding['fingerprint'],
        'outputs':{file.name:store.digest(file) for file in payload_files}},'Metric block completion differs')
    inputs,official,reported=(store.load_commit(directory/name,binding) for name in ('inputs.json','official.json','report.json'))
    selected=[image_id for image_id in references if scope=='pooled' or image_id//1000==int(scope[6:])]
    require(len(selected)==(160 if scope=='pooled' else 16),'Wrong fixed AP block support')
    require(set(inputs)=={'schema','arm','scope','binding_fingerprint','gt','detections','source_map'}
        and inputs['schema']==17 and inputs['arm']==arm and inputs['scope']==scope
        and inputs['binding_fingerprint']==binding['fingerprint'],'Metric input identity differs')
    truth=inputs['gt']
    require(set(truth)=={'info','images','categories','annotations'} and truth['info']=={}
        and truth['categories']==[{'id':1,'name':'object'}]
        and truth['images']==[references[index]['image'] for index in selected],'Image cohort or category interpretation differs')
    expected_gt=[annotation for image_id in selected for annotation in references[image_id]['annotations']]
    require(len(truth['annotations'])==len(expected_gt)
        and [ann['id'] for ann in truth['annotations']]==[ann['id'] for ann in expected_gt],'Original visible instances were lost or reordered')
    require(set(inputs['source_map'])=={'gt_annotation_id','dt_list_index'}
        and inputs['source_map']['gt_annotation_id']=={str(ann['id']):ann['source'] for ann in expected_gt},'GT annotation provenance differs')
    gt_by_image={image_id:[] for image_id in selected}
    for actual,expected in zip(truth['annotations'],expected_gt):
        require(set(actual)=={'id','image_id','category_id','segmentation','area','iscrowd','bbox'}
            and {key:value for key,value in actual.items() if key!='segmentation'}=={key:value for key,value in expected.items() if key!='source'},
            'GT area, bbox, ID, or crowd policy differs')
        gt_by_image[expected['image_id']].append(actual)
    detections=inputs['detections']
    expected_dt=[(image_id,item) for image_id in selected for item in score_frames[image_id]['row']['instances']]
    require(len(detections)==len(expected_dt) and set(inputs['source_map']['dt_list_index'])=={str(i) for i in range(len(detections))},
        'Detection-score support changed')
    dt_by_image={image_id:[] for image_id in selected}
    for index,(actual,(image_id,mapping)) in enumerate(zip(detections,expected_dt)):
        source=score_frames[image_id]; record=source['row']; item=source['source']
        require(set(actual)=={'image_id','category_id','segmentation','score'} and actual['image_id']==image_id
            and actual['category_id']==1 and type(actual['score']) is float and actual['score']==mapping['score'],
            'Detection score/order differs or a bbox entered official loadRes')
        expected_source={'scene':item['scene'],'frame':item['frame'],'image_id':image_id,'arm':arm,**mapping,
            'score_record_sha256':store.digest(source['prediction_file'].parent/'record.json'),
            'parent_prediction_sha256':record['parent_prediction_sha256'],'scope_detection_id':index+1}
        require(inputs['source_map']['dt_list_index'][str(index)]==expected_source,'Detection selected-center provenance differs')
        dt_by_image[image_id].append((index+1,actual,mapping['instance_id']))
    results=[]; checked_masks=0
    for image_id in selected:
        reference=references[image_id]
        labels,_=source_labels(root,protocol,context,reference['source'])
        # One dense predicted map and one reference map are resident for this image only.
        with np.load(score_frames[image_id]['prediction_file'],allow_pickle=False) as cache:
            high=original_resolution_instances(cache['instances'])
        gt_ids=[]; labels_ids=[]
        for annotation,original in zip(gt_by_image[image_id],reference['annotations']):
            binary=decode_rle(annotation['segmentation'])
            require(binary.shape==(720,1280) and np.array_equal(binary,labels==original['source']['original_label_id'])
                and int(np.count_nonzero(binary))==annotation['area'],'GT compressed RLE differs from original visible mask')
            gt_ids.append(annotation['id']); labels_ids.append(original['source']['original_label_id']); checked_masks+=1
        dt_ids=[]; instance_ids=[]; confidence=[]
        for detection_id,annotation,instance_id in dt_by_image[image_id]:
            binary=decode_rle(annotation['segmentation'])
            require(binary.shape==(720,1280) and np.array_equal(binary,high==instance_id),'Detection RLE differs from nearest-resized immutable prediction')
            dt_ids.append(detection_id); instance_ids.append(instance_id); confidence.append(annotation['score']); checked_masks+=1
        overlap=dense_overlap(high,labels,instance_ids,labels_ids)
        results.append(independent_image_matching(image_id,gt_ids,dt_ids,confidence,overlap['iou']))
        if len(results)%16==0:
            guard.check()
    require(official['schema']==17 and official['parameters']==dict(expected_parameters(),imgIds=selected)
        and official['counts']==[10,101,1,4,3] and len(official['evalImgs'])==4*len(selected),'Official parameter/array support changed')
    require(official['evalImgs'][:len(selected)]==results,'Independent noncrowd all-area greedy matching differs')
    precision=np.asarray(official['precision'],np.float64)
    recall=np.asarray(official['recall'],np.float64)
    scores=np.asarray(official['scores'],np.float64)
    require(precision.shape==scores.shape==(10,101,1,4,3) and recall.shape==(10,1,4,3)
        and all(np.isfinite(value).all() for value in (precision,recall,scores)),'Official accumulation cache has invalid shape or values')
    expected=independent_ap(results)
    close_arrays(precision[:,:,0,0,2],expected['precision'],'Independent all-area primary precision differs')
    close_arrays(recall[:,0,0,2],expected['recall'],'Independent all-area maxDet100 recall differs')
    close_arrays(scores[:,:,0,0,2],expected['scores'],'Independent all-area sampled confidence differs')
    metric={key:{'value':expected['summary'][key],'reason':None if expected['ground_truth_count'] else 'no_valid_ground_truth'}
        for key in ('AP','AP50','AP75')}
    require(len(official['stats'])==12,'Official summary size differs')
    close_arrays(official['stats'][:3],[metric[key]['value'] if metric[key]['value'] is not None else -1 for key in ('AP','AP50','AP75')],
        'Independent primary AP summary differs')
    report={'schema':17,'status':'completed','arm':arm,'scope':scope,'binding_fingerprint':binding['fingerprint'],
        'image_count':len(selected),'ground_truth_count':len(expected_gt),'detection_count':len(expected_dt),
        'metrics':metric,'metric_scale':'fraction_0_1','all_images_preserved':True,'official_evaluator_unmodified':True}
    difference=compare_metrics(reported,report,'Block primary metrics differ')
    return report,{'arm':arm,'scope':scope,'image_count':len(selected),'rle_masks_checked':checked_masks,
        'maximum_primary_precision_difference':float(np.max(np.abs(precision[:,:,0,0,2]-expected['precision']))),
        'maximum_summary_difference':difference,'greedy_matching_exact':True,'all_area_maxdet100_audited':True,
        'auxiliary_area_and_lower_maxdets_independently_recomputed':False,'report_sha256':store.digest(directory/'report.json')}


def independent_aggregate(protocol,reports,binding):
    metrics=('AP','AP50','AP75')
    arms={}
    for arm in ARMS:
        scenes={str(scene):reports[(arm,'scene_%04d'%scene)]['metrics'] for scene in protocol['validation_scenes']}
        macro={}
        for name in metrics:
            available={scene:row[name]['value'] for scene,row in scenes.items() if row[name]['value'] is not None}
            macro[name]={'value':math.fsum(available.values())/len(available) if available else None,
                'reason':None if available else 'no_eligible_scenes','eligible_scenes':[int(scene) for scene in available],
                'excluded_scenes':{scene:row[name]['reason'] for scene,row in scenes.items() if row[name]['value'] is None}}
        arms[arm]={'scenes':scenes,'scene_macro':macro,'pooled':reports[(arm,'pooled')]['metrics']}
    def delta(first,second):
        present=first['value'] is not None and second['value'] is not None
        return {'value':second['value']-first['value'] if present else None,'reason':None if present else 'one_or_both_arms_undefined'}
    paired={'by_scene':{str(scene):{name:delta(arms['rgb']['scenes'][str(scene)][name],arms['rgbd']['scenes'][str(scene)][name])
        for name in metrics} for scene in protocol['validation_scenes']}}
    for scope in ('scene_macro','pooled'):
        paired[scope]={name:delta(arms['rgb'][scope][name],arms['rgbd'][scope][name]) for name in metrics}
    return {'schema':17,'status':'completed_development_mask_ap','binding':binding,'expected_blocks':22,'completed_blocks':len(reports),
        'arms':arms,'paired_rgbd_minus_rgb':paired,'metric_scale':'fraction_0_1','new_model_forward_calls':0,'new_training_updates':0,
        'held_out_data_used':False,'thresholds_or_checkpoints_selected':False,'limitations':protocol['limitations']}


def inspect_metrics(root,protocol,config,score_outputs,score_seals,context,guard):
    store=common(); binding=metrics_binding(root,protocol,config,score_seals)
    directory=store.safe(root,protocol['outputs']['metrics_root'])
    require(store.read(directory/'binding.json')==binding,'Official metric binding differs')
    expected={directory/'binding.json',directory/'completion.json',store.receipt_path(directory/'completion.json')}
    references=reference_identity(root,protocol,context,guard)
    reports={}; diagnostics=[]
    for arm in ARMS:
        for scope in ['scene_%04d'%scene for scene in protocol['validation_scenes']]+['pooled']:
            block=directory/arm/scope
            for name in ('inputs.json','official.json','report.json','completion.json'):
                expected.update((block/name,store.receipt_path(block/name)))
            report,audit=inspect_block(root,protocol,arm,scope,block,binding,score_outputs[arm]['frames'],references,context,guard)
            reports[(arm,scope)]=report; diagnostics.append(audit)
    require({p for p in directory.rglob('*') if p.is_file()}==expected,'Official metric block membership differs from fixed 22 blocks')
    completion=store.load_commit(directory/'completion.json',binding)
    aggregate_file=store.safe(root,protocol['outputs']['metrics_report'])
    excluded={directory/'completion.json',store.receipt_path(directory/'completion.json')}
    require(completion=={'schema':17,'status':'completed','binding_fingerprint':binding['fingerprint'],'expected_blocks':22,
        'outputs':{file.relative_to(directory).as_posix():store.digest(file) for file in expected-excluded},
        'report_sha256':store.digest(aggregate_file)},'Official metric aggregate seal differs')
    aggregate=independent_aggregate(protocol,reports,binding)
    difference=compare_metrics(store.load_commit(aggregate_file,binding),aggregate,'Independent scene-macro/pooled/pair summary differs')
    return aggregate,diagnostics,difference


def validate_protocol(protocol):
    require(protocol['schema']==17 and protocol['arms']==list(ARMS)
        and protocol['validation_scenes']==[2,6,7,11,13,15,18,22,33,39]
        and protocol['frames']==list(range(0,256,16)) and protocol['camera']=='realsense'
        and protocol['input_size']==[240,320] and protocol['original_size']==[720,1280], 'Fixed evaluation support changed')
    require(protocol['decoder']=={'foreground_threshold':.5,'center_threshold':.35,'min_pixels':20,'max_centers':64}
        and protocol['precision']=='float32_without_autocast','Fixed numerical decoding changed')
    require(protocol['score_policy']=={'name':'retained_selected_center_sigmoid_float32',
        'ties':'stable_preserve_original_order_no_perturbation',
        'maps_required':['instance_id','selected_center_index','row','col','score'],
        'raw_prediction_cache_must_match_parent':True,'maximum_instances':64},'Score definition changed')
    require(protocol['mask_policy']=={'resize':'integer_repeat_rows_3_columns_4_equivalent_to_nearest',
        'gt':'all_original_visible_label_ids_greater_than_zero_one_instance_per_id','category_id':1,'useCats':1,
        'iscrowd':0,'gt_minimum_area_filter':False,'gt_area':'original_visible_mask_pixel_count',
        'dt_bbox_in_loadRes':False},'Original mask interpretation changed')
    policy=protocol['coco_policy']; expected=expected_parameters()
    require(policy['iou_type']=='segm' and policy['iou_thresholds']==expected['iouThrs']
        and policy['recall_threshold_count']==101 and policy['max_dets']==expected['maxDets']
        and policy['area_ranges']==expected['areaRng'] and policy['area_names']==expected['areaRngLbl']
        and policy['metrics']==['AP','AP50','AP75'] and policy['summaries']==['equal_scene_macro','pooled']
        and policy['pair_direction']=='rgbd_minus_rgb' and policy['no_valid_ground_truth']=='null_with_reason'
        and policy['empty_detections']=='retain_all_images_with_empty_annotation_list','Official metric policy changed')
    require(protocol['independent_audit']=={'scope':'all_area_max_dets_100_matching_precision_AP_AP50_AP75',
        'absolute_tolerance':TOLERANCE,'relative_tolerance':0,
        'auxiliary_area_and_smaller_max_dets':'official_outputs_only_not_independently_reimplemented'},'Independent audit scope changed')
    limits=protocol['resources']
    require(limits['cpu_threads']==4 and limits['invocation_soft_seconds']==1800
        and limits['rss_soft_bytes']==4*1024**3 and limits['output_limit_bytes']==3*1024**3
        and limits['disk_reserve_bytes']==60*1024**3,'Frozen acceptance resource budget changed')


def file_state(file):
    require(Path(file).is_file() and not Path(file).is_symlink(),'Missing or linked evidence file')
    return {'sha256':common().digest(file),'bytes':file.stat().st_size,'mtime_ns':file.stat().st_mtime_ns}


def resource_evidence(record,protocol):
    value=record['resources']; limits=protocol['resources']
    for name,maximum in (('elapsed_seconds',limits['invocation_soft_seconds']),('peak_rss_bytes',limits['rss_soft_bytes']),
            ('peak_gpu_allocated_bytes',limits['gpu_allocated_soft_bytes']),('peak_gpu_reserved_bytes',limits['gpu_reserved_soft_bytes']),
            ('current_output_bytes',limits['output_limit_bytes'])):
        require(type(value[name]) in (int,float) and math.isfinite(value[name]) and 0<=value[name]<=maximum,
            'Recorded child exceeded a resource budget')


def work_record(record,phase,key,operation,protocol):
    unit='frames' if phase=='scores' else 'blocks'; total=160 if phase=='scores' else 22
    require(record['schema']==17 and record['expected_'+unit]==total,'Wrong recovery work identity')
    if phase=='scores':
        require(record['arm']==key,'Wrong recovery arm')
    for name in ('new_'+unit,'reused_'+unit,'committed_'+unit,'new_model_forward_calls','new_training_updates'):
        require(type(record[name]) is int and record[name]>=0,'Invalid child work accounting')
    require(record['new_'+unit]+record['reused_'+unit]==record['committed_'+unit]
        and record['new_training_updates']==0,'Inconsistent child work accounting')
    if operation=='probe':
        require(record['status']=='partial' and record['committed_'+unit]==2,'Recovery probe was not two units')
    else:
        require(record['status'] in ('completed','verified_completed_read_only') and record['committed_'+unit]==total,
            'Recovery did not cover the complete cohort')
    readonly=operation=='read_only' or record['status']=='verified_completed_read_only'
    if readonly:
        require(record['status']=='verified_completed_read_only' and record['new_'+unit]==record['new_model_forward_calls']==0,
            'Read-only recovery performed new work')
    if phase=='scores':
        require(record['new_model_forward_calls']<=record['new_frames'],'Extra forward calls within a completed invocation')
        pending=record['pending_payload_accounting']
        require(set(pending)=={'heads_without_frame_record','heads_receipt_without_payload','unknown_outcome_intents'}
            and all(type(value) is int and value>=0 for value in pending.values()),'Invalid unresolved attempt accounting')
        require(pending['heads_without_frame_record']==pending['heads_receipt_without_payload']==0,
            'Completed recovery boundary has unresolved frame outputs')
    else:
        require(record['new_model_forward_calls']==0 and type(record['new_official_evaluations']) is int
            and 0<=record['new_official_evaluations']<=record['new_blocks'],'Unexpected work in official metric recovery')
        if readonly:
            require(record['new_official_evaluations']==0,'Read-only recovery reran official evaluation')
    resource_evidence(record,protocol)


def inspect_recovery(root,protocol,config):
    store=common(); result={}
    for phase in ('scores','metrics'):
        path=store.safe(root,protocol['outputs']['score_resume_report' if phase=='scores' else 'metrics_resume_report'])
        proof=store.read(path)
        child='scripts/extract_mask_scores_v17.py' if phase=='scores' else 'scripts/evaluate_mask_ap_v17.py'
        interpreter=protocol['interpreter'] if phase=='scores' else protocol['evaluation_interpreter']
        expected_binding={'protocol_sha256':store.digest(config),'wrapper_sha256':store.digest(root/'scripts/run_mask_ap_v17.py'),
            'child_sha256':store.digest(root/child),'interpreter_sha256':store.digest(store.safe(root,interpreter)),
            'pipeline_sha256':store.digest(root/'scripts/pipeline.py')}
        keys=list(ARMS) if phase=='scores' else ['all']
        require(proof['schema']==17 and proof['phase']==phase and proof['status']=='passed' and proof['binding']==expected_binding
            and set(proof['entries'])==set(keys)
            and proof['scope']=='Actual process boundaries; no claim of physical power removal','Recovery proof identity differs')
        entries={}
        for key in keys:
            directory=store.safe(root,protocol['outputs']['scores_root'])/key if phase=='scores' else store.safe(root,protocol['outputs']['metrics_root'])
            report=directory/'report.json' if phase=='scores' else store.safe(root,protocol['outputs']['metrics_report'])
            entry=proof['entries'][key]; current=store.snapshot(directory,root)
            if phase=='metrics':
                current.update({file.relative_to(root).as_posix():file_state(file) for file in (report,store.receipt_path(report))})
            require(entry['status']=='passed' and entry['byte_and_mtime_identical'] is True and entry['read_only_check_passed'] is True
                and entry['completed_snapshot']==current and entry['completion_sha256']==store.digest(directory/'completion.json')
                and entry['report_sha256']==store.digest(report),'Completed recovery cache SHA, size or mtime changed')
            prefix=entry['prefix_before']; require(prefix==entry['prefix_after'],'Prefix changed across restart')
            relative=directory.relative_to(root).as_posix()
            if phase=='scores':
                first=[relative+'/frames/scene_%04d_%04d/'%(scene,frame)
                    for scene in protocol['validation_scenes'] for frame in protocol['frames']][:2]
            else:
                first=[relative+'/'+arm+'/'+scope+'/' for arm in ARMS
                    for scope in ['scene_%04d'%scene for scene in protocol['validation_scenes']]+['pooled']][:2]
            expected_prefix={name:row for name,row in current.items()
                if name==relative+'/binding.json' or any(name.startswith(start) for start in first)}
            require(prefix==expected_prefix and bool(prefix),'Probe prefix omits a sealed artifact or exceeds two units')
            for operation in ('probe','completion','read_only'):
                work_record(entry[operation+'_record'],phase,key,operation,protocol)
            if phase=='metrics':
                require(entry['completion_record']['report_sha256']==entry['read_only_record']['report_sha256']==store.digest(report),
                    'Official summary receipt differs across recovery')
            entries[key]={'prefix_files_checked':len(prefix),'completed_files_checked':len(current),
                'byte_and_mtime_identical':True,'new_units_at_probe':entry['probe_record']['new_frames' if phase=='scores' else 'new_blocks'],
                'new_units_at_completion':entry['completion_record']['new_frames' if phase=='scores' else 'new_blocks']}
        result[phase]={'proof_sha256':store.digest(path),'entries':entries,'physical_power_loss_tested':False}
    return result


def figure_bytes(summary):
    image=Image.new('RGB',(1440,990),'white'); draw=ImageDraw.Draw(image)
    title=ImageFont.load_default(size=28); heading=ImageFont.load_default(size=22); font=ImageFont.load_default(size=18)
    small=ImageFont.load_default(size=16); colors={'rgb':'#2775a5','rgbd':'#bd6236'}
    draw.text((40,25),'Visible-instance mask AP: paired development baseline',fill='#152b3b',font=title)
    draw.text((40,66),'10 previously used development scenes | 160 frames per arm | final checkpoints | one seed',fill='#44525c',font=font)
    for panel,scope in enumerate(('scene_macro','pooled')):
        x0=75+panel*700; y0=410; width=560; height=255
        draw.text((x0,112),'Equal-scene mean' if scope=='scene_macro' else 'Pooled images',fill='#152b3b',font=heading)
        for tick in range(6):
            y=y0-height*tick/5
            draw.line((x0,y,x0+width,y),fill='#dce3e7',width=1)
            draw.text((x0-40,y-8),'%.1f'%(tick/5),fill='#53616b',font=small)
        for index,metric in enumerate(('AP','AP50','AP75')):
            cx=x0+100+index*190
            for offset,arm in ((-58,'rgb'),(8,'rgbd')):
                value=summary['arms'][arm][scope][metric]['value']
                if value is not None:
                    draw.rectangle((cx+offset,y0-height*value,cx+offset+48,y0),fill=colors[arm])
                    draw.text((cx+offset-4,y0-height*value-23),'%.3f'%value,fill=colors[arm],font=small)
                else:
                    draw.text((cx+offset,y0-28),'NA',fill=colors[arm],font=small)
            draw.text((cx-30,y0+13),metric,fill='#152b3b',font=font)
        draw.text((x0,460),'RGB',fill=colors['rgb'],font=font); draw.text((x0+90,460),'RGB-D',fill=colors['rgbd'],font=font)
        delta=summary['paired_rgbd_minus_rgb'][scope]
        labels=['%s %s'%(name,'NA' if delta[name]['value'] is None else '%+.4f'%delta[name]['value']) for name in ('AP','AP50','AP75')]
        draw.text((x0,494),'RGB-D minus RGB: '+ '   '.join(labels),fill='#394b58',font=small)
    draw.text((60,555),'Scene AP (IoU .50:.95; 101 recall points; all areas; maximum 100 detections/image)',fill='#152b3b',font=heading)
    for index,scene in enumerate(summary['arms']['rgb']['scenes']):
        column=index//5; row=index%5; x=70+column*700; y=600+row*48
        rgb=summary['arms']['rgb']['scenes'][scene]['AP']['value']; rgbd=summary['arms']['rgbd']['scenes'][scene]['AP']['value']
        delta=summary['paired_rgbd_minus_rgb']['by_scene'][scene]['AP']['value']
        fmt=lambda value:'NA' if value is None else '%.4f'%value
        draw.text((x,y),'Scene %04d'%int(scene),fill='#152b3b',font=font)
        draw.text((x+145,y),'RGB '+fmt(rgb),fill=colors['rgb'],font=font)
        draw.text((x+295,y),'RGB-D '+fmt(rgbd),fill=colors['rgbd'],font=font)
        draw.text((x+475,y),'Delta '+('NA' if delta is None else '%+.4f'%delta),fill='#394b58',font=font)
    draw.text((60,865),'Original visible masks: 720 x 1280. Predictions: 240 x 320, nearest upsampling 3 x 4.',fill='#44525c',font=font)
    draw.text((60,898),'Independent audit: decoding, RLE masks, all-area matching and primary AP. No repeated neural inference.',fill='#44525c',font=small)
    draw.text((60,928),'Development evidence only; no held-out generalization or full-method completion claim. Undefined values remain NA.',fill='#44525c',font=small)
    stream=io.BytesIO(); image.save(stream,format='PNG'); return stream.getvalue()


def watched_snapshot(root,protocol,config,context):
    store=common(); files={config,store.safe(root,protocol['parent_protocol']),store.safe(root,protocol['parent_acceptance_report']),
        store.safe(root,protocol['environment_protocol']),store.safe(root,protocol['environment_report']),
        store.safe(root,protocol['outputs']['score_resume_report']),store.safe(root,protocol['outputs']['metrics_resume_report']),
        store.safe(root,protocol['outputs']['metrics_report'])}
    files.add(store.receipt_path(store.safe(root,protocol['outputs']['metrics_report'])))
    for p in (protocol,context['protocol']):
        for group in ('source_hashes','input_hashes'):
            files.update(store.safe(root,name) for name in p[group])
    for source in context['sources'].values():
        files.update(store.safe(root,name) for name in source['files'])
    for directory in (store.safe(root,protocol['outputs']['scores_root']),store.safe(root,protocol['outputs']['metrics_root']),
            store.safe(root,context['protocol']['outputs']['evaluation_root'])):
        files.update(file for file in directory.rglob('*') if file.is_file())
    return {file.relative_to(root).as_posix():file_state(file) for file in sorted(files)}


def prepare_own_outputs(root,report_file,visuals,check_only):
    store=common(); figure=visuals/FIGURE
    for file in (report_file,figure):
        require(file.is_relative_to(root) and not file.is_symlink(),'Acceptance output escapes the project')
    temporary=[]
    for destination in (report_file,figure):
        for file in destination.parent.glob(destination.name+'.*.tmp'):
            require(re.fullmatch(re.escape(destination.name)+r'\.[0-9a-f]{32}\.tmp',file.name) and not file.is_symlink(),
                'Unknown acceptance temporary artifact')
            temporary.append(file)
    require(not temporary or (not check_only and not report_file.exists()),'Completed acceptance contains temporary artifacts')
    for file in temporary:
        file.unlink()
    require(not visuals.exists() or {file for file in visuals.iterdir()}<={figure},'Unknown scientific figure artifact')
    return figure


def execute(root,protocol,config,check_only=False):
    store=common(); validate_protocol(protocol)
    report_file=store.safe(root,protocol['outputs']['acceptance_report']); visuals=store.safe(root,protocol['outputs']['visuals_dir'])
    figure=prepare_own_outputs(root,report_file,visuals,check_only)
    guard=store.Guard(root,protocol,'acceptance'); guard.check()
    store.verify_source_maps(root,protocol)
    verify_environment(root,protocol)
    context=read_parent(root,protocol)
    before=watched_snapshot(root,protocol,config,context)
    outputs,seals=inspect_scores(root,protocol,config,context,guard)
    aggregate,diagnostics,summary_error=inspect_metrics(root,protocol,config,outputs,seals,context,guard)
    recovery=inspect_recovery(root,protocol,config)
    require(watched_snapshot(root,protocol,config,context)==before,'Sealed sources or results changed during independent audit')
    store.verify_source_maps(root,protocol); guard.check()
    body={'schema':17,'phase':'acceptance','protocol_sha256':store.digest(config),
        'source_hashes':protocol['source_hashes'],'input_hashes':protocol['input_hashes'],
        'parent_acceptance_sha256':protocol['parent_acceptance_sha256'],
        'environment_report_sha256':protocol['environment_report_sha256'],
        'score_seals':seals,'metrics_completion_sha256':store.digest(store.safe(root,protocol['outputs']['metrics_root'])/'completion.json'),
        'metrics_report_sha256':store.digest(store.safe(root,protocol['outputs']['metrics_report'])),
        'recovery_proof_sha256':{phase:row['proof_sha256'] for phase,row in recovery.items()},
        'interpreter_sha256':store.digest(store.safe(root,protocol['evaluation_interpreter']))}
    binding=dict(body,fingerprint=store.fingerprint(body))
    png=figure_bytes(aggregate)
    report={'schema':17,'status':'accepted_complete_development_mask_ap','binding':binding,
        'decision':'engineering_and_metric_consistency_accepted_without_rgbd_improvement_gate',
        'arms':aggregate['arms'],'paired_rgbd_minus_rgb':aggregate['paired_rgbd_minus_rgb'],
        'fixed_support':{'scenes':protocol['validation_scenes'],'frames_per_scene':16,'frames_per_arm':160,'arms':list(ARMS),'metric_blocks':22},
        'score_forward_attempt_accounting':{arm:outputs[arm]['report']['forward_attempt_accounting'] for arm in ARMS},
        'independent_checks':{'decoded_frames':320,'parent_predictions_bitwise_equal':True,'blocks':diagnostics,
            'maximum_aggregate_difference':summary_error,'absolute_tolerance':TOLERANCE,'relative_tolerance':0,
            'source_files_unchanged':True,'source_file_count_checked':len(before),
            'neural_forward_independently_repeated':False,'auxiliary_area_and_lower_maxdets_independently_recomputed':False},
        'recovery':recovery,'figure':{'file':protocol['outputs']['visuals_dir']+'/'+FIGURE,'sha256':hashlib.sha256(png).hexdigest()},
        'new_model_forward_calls':0,'new_official_evaluations':0,'new_training_updates':0,
        'held_out_data_used':False,'thresholds_or_checkpoints_selected':False,'is_full_p4_method_completion':False,
        'limitations':protocol['limitations']}
    completed=report_file.exists(); readonly=check_only or completed
    old={file: file_state(file) for file in (report_file,figure)} if readonly else None
    encoded=store.json_bytes(report); guard.check(0 if readonly else len(png)+len(encoded)+8192)
    store.publish_once(figure,png,readonly); store.publish_once(report_file,encoded,readonly)
    if readonly:
        require(old=={file:file_state(file) for file in (report_file,figure)},'Read-only acceptance changed bytes or mtime')
    resources=guard.check()
    return {'schema':17,'status':'verified_completed_read_only' if readonly else report['status'],
        'report_sha256':store.digest(report_file),'figure_sha256':store.digest(figure),
        'new_model_forward_calls':0,'new_official_evaluations':0,'new_training_updates':0,'resources':resources}


def run(root=ROOT,config=None,check_only=False):
    root=Path(root).resolve(); store=common(); config=store.safe(root,config or 'configs/mask_ap_v17.json')
    with store.lock(root/'runtime/locks/mask_ap_v17_acceptance'):
        return execute(root,store.read(config),config,check_only)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mask_ap_v17.json')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(config=args.config,check_only=args.check_only)),flush=True)

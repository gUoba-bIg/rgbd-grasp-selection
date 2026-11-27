"""Post-return classification only. No engine import, motion, controller or geometry change."""
from collections import Counter
from pathlib import Path
import gzip,hashlib,json,math,re
import codec
from common import read,require,sha
from contact_semantics import validate_entry_certificate

DISTANCE=1e-9
BRACKET=1e-10
DETAIL='static_distance_uncertified'
NO_BILATERAL="ValueError('both_registered_entries_required_before_lift')"
ROOT_REJECTIONS={'entry_surface_not_registered','root_residual_exceeds_tolerance','root_bracket_not_converged','requires_d_clear_positive_d_overlap_nonpositive','normal_not_unit','full_witness_residual'}
ENTRY_REJECTIONS=ROOT_REJECTIONS|{'entry_solver_not_passed','entry_witness_face_audit_missing_or_failed','entry_signed_distance_residual','entry_not_on_registered_surface','entry_root_distance_signs','entry_root_bracket_range_width','entry_root_bracket_missing','entry_full_witness_residual_m','entry_normal_unit_error','entry_normal_wrong_inner_face_direction','entry_positive_to_contact_transition_unproven','pad_contact_without_legal_entry','missing_per_side_entry_certificate','no_clean_positive_to_nonpositive_entry','reference_negative_or_manifold_before_legal_entry','pad_witness_position_outside_registered_surfaces','normal_projection_residual','fresh_full_3d_residual'}
KNOWN_COLLISION={'proxy_pad_target_penetration','nonpad_target_collision','simultaneous_nonpad_target_collision_at_entry','finger_independent_robot_target_collision','unregistered_self_collision','robot_table_collision','pad_target_contact_before_close','entry_nonpad_target_clearance_unproven'}
def hard(message):raise ValueError('shared_native_or_evidence_integrity: '+message)
def check(value,message):
    if not value:hard(message)
def numbers(value):
    if isinstance(value,float):check(math.isfinite(value),'nonfinite number')
    elif isinstance(value,dict):
        for x in value.values():numbers(x)
    elif isinstance(value,(list,tuple)):
        for x in value:numbers(x)
def validsummary(summary,steps):
    c=summary['calls'];r=summary['returns']
    check(c==r and not summary['exceptions'] and c.get('connect')==c.get('disconnect')==1,'missing return, exception or connection owner')
    check(c.get('stepSimulation',0)==summary['actual_native_step_calls']==steps and c.get('setCollisionFilterPair')==14 and sum(c.values())==summary['total_calls'],'native count, filter or step')
def rowowner(row,domains,args,kwargs,method):
    check(isinstance(row,list) and len(row)==14,'malformed contact')
    a,b,la,lb=row[1:5]
    check(all(type(x) is int for x in (a,b,la,lb)) and a in domains and b in domains and la in domains[a] and lb in domains[b],'unregistered body or link')
    check(all(isinstance(row[i],list) and len(row[i])==3 for i in (5,6,7,11,13)) and all(type(row[i]) in (int,float) for i in (8,9,10,12)),'contact value structure')
    # A scope with one filtered endpoint can legitimately be reported in native reverse order.
    body_a=args[0] if args else kwargs.get('bodyA',-1)
    body_b=args[1] if len(args)>1 else kwargs.get('bodyB',-1)
    link_a=kwargs.get('linkIndexA',-2);link_b=kwargs.get('linkIndexB',-2)
    def fits(x,y,i,j):
        return (body_a<0 or x==body_a) and (body_b<0 or y==body_b) and (link_a==-2 or i==link_a) and (link_b==-2 or j==link_b)
    check(fits(a,b,la,lb) or fits(b,a,lb,la),'native contact does not match query pair')
def native_rows(directory,body,links,*,steps=0,watch=None):
    summary=read(directory/'summary.json');validsummary(summary,steps)
    robot,target,table,client=body;domains={robot:set(links.values())|{-1},target:{-1},table:{-1}};check(len(domains)==3,'body alias')
    c=Counter();r=Counter();seq=0;pending=None;captured={};contacts=0
    file=directory/'calls.jsonl.zst'
    if not file.is_file():file=directory/'calls.jsonl.gz'
    wanted=watch or {};for_ids={'getContactPoints','getClosestPoints'};identity_loads=[];archive_blobs={}
    def no_constant(x):hard('native '+x)
    with codec.open(file,'rb') as fp:
        for raw in fp:
            # All values are from Python's compact JSON encoder. A non-finite binary64
            # produces NaN/Infinity. Check exceptionally large exponent tokens as well.
            obj=json.loads(raw,parse_constant=no_constant)
            if (b'e+' in raw or b'E+' in raw) and re.search(br'[eE]\+(?:30[89]|3[1-9]\d|[4-9]\d\d|\d{4,})',raw):numbers(obj)
            idx,which=obj[:2]
            if which=='call':
                check(pending is None and type(idx) is int and idx==seq+1 and len(obj)==6,'native sequence or overlapping call')
                seq=idx;method,args,kwargs=obj[3:]
                if 'physicsClientId' in kwargs:check(kwargs['physicsClientId']==client,'different native client')
                pending=(method,args,kwargs);c[method]+=1;continue
            check(which=='return' and pending is not None and seq==idx and len(obj)==4,'native call exception, missing or unexpected return')
            method,args,kwargs=pending;pending=None;r[method]+=1;v=obj[3]
            if method in for_ids:
                check(isinstance(v,list),'missing contact collection')
                for row in v:rowowner(row,domains,args,kwargs,method)
                contacts+=len(v)
                if idx in wanted:
                    wi=wanted[idx];side=wi['pad'];values=[(j,x) for j,x in enumerate(v) if (x[1:5]==[robot,target,side,-1] or x[1:5]==[target,robot,-1,side])]
                    check(method=='getClosestPoints' and len(v)==wi['total'] and bool(values),'missing native root query')
                    j,x=min(values,key=lambda pair:pair[1][8])
                    check(v[wi['index']]==wi['row'] and x[8]==wi['distance'] and len(values)==wi['side_count'],'native root distance, identity or row disagrees')
                    captured[idx]=dict(side=side,count=len(values),distance_m=x[8])
            elif method in ('loadURDF','createMultiBody'):
                identity_loads.append([method,v]);check(type(v) is int and v in domains,'native body allocation')
            elif method=='connect':check(v==client,'connection id mismatch')
            elif method=='getBodyUniqueId':check(v in domains,'body id mismatch')
            elif method=='getJointInfo':check(isinstance(v,list) and v[0]==args[1] and args[0]==robot,'joint response identity')
            elif method=='getCollisionShapeData' and v:
                check(v[0][0:2]==args[:2],'collision geometry first-row owner')
            elif method in ('getCameraImage','getMeshData'):
                # ndarray payloads may be externalized in the unmodified native logger.
                def collect(x):
                    if isinstance(x,dict) and 'data_sha256' in x:archive_blobs[x['data_sha256']]=x
                    elif isinstance(x,(list,tuple)):
                        for item in x:collect(item)
                collect(v)
    check(pending is None and dict(c)==summary['calls'] and dict(r)==summary['returns'] and seq==summary['total_calls'] and set(captured)==set(wanted),'trace/summary or root-range reconciliation')
    check(sorted([v for _,v in identity_loads])==sorted(domains),'native loaded body set')
    if archive_blobs:
        import numpy as np
        for digest,meta in archive_blobs.items():
            with gzip.open(directory/(digest+'.bin.gz'),'rb') as f:data=f.read()
            check(hashlib.sha256(data).hexdigest()==digest,'native array blob hash')
            arr=np.frombuffer(data,dtype=meta['dtype']).reshape(meta['shape'])
            if arr.dtype.kind in 'fc':check(np.isfinite(arr).all(),'nonfinite native array')
    return dict(native_calls=seq,native_returns=sum(r.values()),contacts_reconciled=contacts,root_queries_checked=len(wanted),actual_physics_steps=steps,exceptions=0,finite=True,body_link_ownership=True,api_summary_sha256=sha(directory/'summary.json'),api_stream_sha256=sha(file))
def roots_and_queries(base,sr,links):
    roots=read(base/'left_right_contact_entry_roots.json');numbers(roots);robot,target=sr['body_robot'],sr['body_object']
    check(roots['physics_steps']==0 and not roots['object_pose_modified'] and not roots['proxy_used_by_controller'],'unknown preflight state')
    wanted={};checked={};sides=roots['sides']
    for side,root in sides.items():
        check(side in ('left','right'),'incorrect certificate side')
        pad=links['ip140_'+side+'_pad'];sub=base/'contact_entries'/side;source=sub/'queries.jsonl.gz'
        check(read(sub/'result.json')==root and root['side']==side and (root['robot_body_id'],root['target_body_id'],root['pad_link_index'])==(robot,target,pad),'root body/link/side')
        check(root['root_distance_tolerance_m']==DISTANCE and root['root_bracket_tolerance_m']==BRACKET and sha(source)==root['query_log_sha256'],'frozen root limits or query hash')
        rs=[];n_queries=0
        with gzip.open(source,'rt',encoding='utf8') as f:
            for line in f:
                q=json.loads(line,parse_constant=lambda x:hard('root '+x));n_queries+=1
                check(q['side']==side and q['query_index']==n_queries and q['physics_step_calls']==0 and q['object_pose']==root['object_initial_pose'],'root query identity')
                values=q['raw_robot_target_closest'];idx=q['selected_raw_index'];check(type(idx) is int and 0<=idx<len(values),'no returned closest-point row')
                row=values[idx];check(row[1:5] in ([robot,target,pad,-1],[target,robot,-1,pad]) and row[8]==q['signed_distance_m'] and math.isfinite(row[8]),'root witness body/link or distance')
                side_count=sum(x[1:5] in ([robot,target,pad,-1],[target,robot,-1,pad]) for x in values)
                # getClosestPoints is followed by getJointStates and getBasePositionAndOrientation;
                # its exact sequence is bound below from the recorded native command, not inferred by time.
                rs.append((q['api_sequence_first'],q['api_sequence_last'],dict(index=idx,row=row,distance=row[8],side_count=side_count,pad=pad,total=len(values))))
        check(n_queries==root['queries_completed'] and n_queries>0,'missing root query')
        for lo,hi,definition in rs:
            # In frozen contact_entry.query the first eight reset calls are followed
            # by getLinkStates, then the single getClosestPoints at lo+9.
            sid=lo+9
            check(lo<=sid<=hi and sid not in wanted,'overlapping root API ranges')
            wanted[sid]=definition
        frozen_fail=validate_entry_certificate(root,robot=robot,target=target,pad=pad)
        checked[side]=dict(passed=bool(root['passed'] and not frozen_fail),solver_passed=root['passed'],frozen_validation=frozen_fail,violations=root['violations'],status=root['status'],source_query_sha256=sha(source),root_distance_tolerance_m=DISTANCE,bracket=root.get('bracket'))
    check(roots['both_found']==(set(sides)=={'left','right'}) and roots['both_passed']==(set(sides)=={'left','right'} and all(s['passed'] for s in sides.values())),'incorrect bilateral root receipt')
    return roots,checked,wanted
def post_static(trial,*,intent,links=None):
    base=trial/'static_preflight';sr=read(base/'path_result.json');numbers(sr)
    if links is None:links={k:v['index'] for k,v in read(trial/'static_setup/robot_link_readback.json').items()}
    check(sr['trial_id']==intent['trial_id'] and sr['candidate_id']==intent['selected_candidate_id'] and sr['method']==intent['method'] and sr['physical_steps']==sr['physics_step_calls']==0 and sr['filter_call_count']==14,'static receipt owner, filters or physics')
    check(sr['target_pose_modified'] is False and sr['proxy_used_by_controller'] is False,'static world scope')
    roots,checked,wanted=roots_and_queries(base,sr,links)
    native=native_rows(trial/'static_api',(sr['body_robot'],sr['body_object'],sr['body_table'],sr['client']),links,watch=wanted)
    result=classify_summary(sr,roots,checked,native)
    return dict(detail_class=result,path_passed=sr['passed'],certificate_sides=checked,native=native,static_result_sha256=sha(base/'path_result.json'),roots_sha256=sha(base/'left_right_contact_entry_roots.json'),unchanged_distance_tolerance_m=DISTANCE)
def classify_summary(sr,roots,checked,native):
    check(native['finite'] and native['body_link_ownership'] and native['exceptions']==0 and native['native_calls']==native['native_returns'] and native['actual_physics_steps']==0,'integrity has priority over classification')
    if sr['passed']:
        check(roots['both_passed'] and len(checked)==2 and all(x['passed'] for x in checked.values()) and sr['snapshots_completed']==4728,'success without two certificates')
        return 'static_admitted'
    first=sr.get('first_failure') or {};exc=sr.get('exception')
    if exc is not None:
        if exc==NO_BILATERAL and not sr['violations'] and first.get('phase')=='lift' and first.get('phase_tick')==1 and not roots['both_passed']:
            return DETAIL
        return None
    if first.get('phase')!='close' or not sr['violations']:return None
    leaves={text.split(':')[-1] for text in sr['violations']}
    if any(text.split(':')[0] not in ('reference','proxy') for text in sr['violations']):return None
    if not leaves<=ENTRY_REJECTIONS|KNOWN_COLLISION:return None
    failed=[v for v in checked.values() if not v['passed']]
    if failed and all(set(v['violations'])<=ROOT_REJECTIONS|{'simultaneous_nonpad_target_collision_at_entry'} and v['status'] in ('entry_not_admitted','not_bracketed','geometric_entry_localized') for v in failed) and leaves & ENTRY_REJECTIONS:return DETAIL
    if not failed and leaves & {'missing_per_side_entry_certificate','no_clean_positive_to_nonpositive_entry','pad_contact_without_legal_entry'} and not leaves & KNOWN_COLLISION:return DETAIL
    return None

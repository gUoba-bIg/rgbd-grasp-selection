"""Pure offline semantic cases and immutable smoke13 observations. No engine import."""
from copy import deepcopy
import gzip,hashlib,json,math
from pathlib import Path
import numpy as np
from contact_semantics import classify_contact,validate_entry_certificate,SCHEMA
from contact_entry import validate_contact_row,model_face_spec
from object_evaluator import ObjectEvaluator

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/'IP140_R1_smoke13_slot14_legacy_height_bridge021_aggregate_support01'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text('utf-8-sig'))


def safe(value):
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite_literal':repr(value)}
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.ndarray):return safe(value.tolist())
    return value


def contact(a,b,la,lb,d=.001,*,normal=(0.,-1.,0.),force=0.):
    pa=[.03,0.,0.];pb=[pa[k]-d*normal[k] for k in range(3)]
    return [0,a,b,la,lb,pa,pb,list(normal),d,force,0.,[1.,0.,0.],0.,[0.,0.,1.]]


def reverse(row):
    c=deepcopy(row);c[1],c[2]=c[2],c[1];c[3],c[4]=c[4],c[3];c[5],c[6]=c[6],c[5]
    c[7]=[-v for v in c[7]];return c


def build():
    links={n:r['index'] for n,r in read(OLD/'path_precheck/zero_step/all_links_readback.json').items()}
    robot,target,table=0,2,1;lp,rp=links['ip140_left_pad'],links['ip140_right_pad']
    lb,rb=links['ip140_left_bone'],links['ip140_right_bone']
    cert=dict(passed=True,robot_body_id=robot,target_body_id=target,pad_link_index=lp,phase='close',
        previous_snapshot=2751,current_snapshot=2752,previous_signed_distance_m=.00001,current_signed_distance_m=-.00015,
        precontact_clean=True,q_entry_m=.032,entry_audit=dict(passed=True,synthetic=True,
            signed_distance_m=2e-11,face_outside_m=0.,full_witness_residual_m=0.,normal_unit_error=0.,
            on_effective_inner_face=True,inward_normal_alignment=1.),
        bracket=dict(q_clear_m=.032,q_overlap_m=.032-5e-11,d_clear_m=2e-11,d_overlap_m=-3e-11,width_m=.032-(.032-5e-11)),
        nonpad_target_collisions_at_entry=[],synthetic_test_fixture=True)
    cases=[]
    def run(name,row,*,expected=True,channel='reference',phase='close',source='robot_target',entries=None,classification=None):
        entries={} if entries is None else entries
        out=classify_contact(row,source=source,phase=phase,channel=channel,robot=robot,target=target,table=table,links=links,entry_by_pad=entries)
        ok=(not out['violations'])==expected and (classification is None or out['classification']==classification)
        cases.append(dict(name=name,passed=ok,expected_accepted=expected,input=safe(row),
            context=safe(dict(channel=channel,phase=phase,source=source,entry_by_pad=entries)),actual=out,synthetic=True))
    c=contact(robot,target,lp,-1,-.001125493581548879)
    run('precontact_reference_positive_clearance',contact(robot,target,lp,-1,.001),phase='approach')
    run('precontact_proxy_positive_clearance',contact(robot,target,lp,-1,.001),phase='approach',channel='proxy')
    for phase in ('close','lift','hold'):
        run('certified_reference_overtravel_'+phase,c,entries={lp:cert},phase=phase,classification='post_contact_motor_reference_overtravel')
    run('certified_native_order_reversal',reverse(c),entries={lp:cert},classification='post_contact_motor_reference_overtravel')
    run('certified_proxy_contact_zero',contact(robot,target,lp,-1,0.),channel='proxy',entries={lp:cert})
    run('proxy_preserves_exact_minus1mm_boundary',contact(robot,target,lp,-1,-.001),channel='proxy',entries={lp:cert})
    run('proxy_nextafter_minus1mm_rejected',contact(robot,target,lp,-1,math.nextafter(-.001,-math.inf)),channel='proxy',entries={lp:cert},expected=False)
    run('certified_reference_bone_overclosure',contact(robot,target,lb,-1,-.002),entries={lp:cert},classification='post_contact_bone_motor_reference_overclosure')
    run('proxy_bone_overlap_remains_hard',contact(robot,target,lb,-1,-.002),entries={lp:cert},channel='proxy',expected=False)
    run('opposite_bone_without_own_entry_stays_hard',contact(robot,target,rb,-1,-.002),entries={lp:cert},expected=False)
    for phase in ('settle','opening','transit','pregrasp','approach'):
        run('early_pad_negative_'+phase,contact(robot,target,lp,-1,-1e-15),phase=phase,entries={lp:cert},expected=False)
        run('early_pad_manifold_'+phase,contact(robot,target,lp,-1,0.),phase=phase,source='manifold',expected=False)
    run('unbracketed_reference_negative_rejected',contact(robot,target,lp,-1,-1e-15),expected=False)
    run('unbracketed_proxy_negative_rejected',contact(robot,target,lp,-1,-1e-15),channel='proxy',expected=False)
    run('proxy_penetration_even_after_entry_rejected',c,channel='proxy',entries={lp:cert},expected=False)
    for link in ('ip140_palm','wrist_3_link','upper_arm_link'):
        for channel in ('reference','proxy'):
            run('finger_independent_'+link+'_'+channel,contact(robot,target,links[link],-1,-1e-15),entries={lp:cert},channel=channel,expected=False)
    run('proxy_nonpad_zero_manifold_rejected',contact(robot,target,lb,-1,0.),entries={lp:cert},channel='proxy',source='manifold',expected=False)
    invalid_cert=[('bare_true',True),('empty',{}),('not_passed',{**cert,'passed':False}),
        ('wrong_robot',{**cert,'robot_body_id':77}),('wrong_target',{**cert,'target_body_id':77}),
        ('wrong_side',{**cert,'pad_link_index':rp}),('wrong_phase',{**cert,'phase':'approach'}),
        ('unclean_precontact',{**cert,'precontact_clean':False}),('nonconsecutive_snapshot',{**cert,'current_snapshot':2753}),
        ('zero_previous_distance',{**cert,'previous_signed_distance_m':0.}),
        ('already_negative_previous',{**cert,'previous_signed_distance_m':-.00001}),
        ('current_still_positive',{**cert,'current_signed_distance_m':.00001}),
        ('nan_entry_q',{**cert,'q_entry_m':float('nan')}),('out_of_travel_entry_q',{**cert,'q_entry_m':.071}),
        ('failed_entry_face',{**cert,'entry_audit':{'passed':False}}),
        ('entry_nonpad_collision',{**cert,'nonpad_target_collisions_at_entry':[{'collision':True}]})]
    for field in ('face_outside_m','full_witness_residual_m','normal_unit_error'):
        invalid_cert.append(('forged_true_'+field,{**cert,'entry_audit':{**cert['entry_audit'],field:2e-9}}))
    for name,patch in [('root_d_above_tol',{'signed_distance_m':2e-9}),('nan_root_d',{'signed_distance_m':float('nan')}),
        ('wrong_face_boolean',{'on_effective_inner_face':False}),('zero_inward_alignment',{'inward_normal_alignment':0.})]:
        invalid_cert.append(('forged_true_'+name,{**cert,'entry_audit':{**cert['entry_audit'],**patch}}))
    for name,patch in [('too_wide',{'width_m':2e-10}),('bad_clear_sign',{'d_clear_m':0.}),
        ('bad_overlap_sign',{'d_overlap_m':1e-11}),('outside_limits',{'q_clear_m':.071}),
        ('width_inconsistent',{'width_m':1e-11}),('nonfinite_bracket',{'d_clear_m':float('nan')})]:
        invalid_cert.append(('forged_true_'+name,{**cert,'bracket':{**cert['bracket'],**patch}}))
    invalid_cert.append(('root_q_not_clear',{**cert,'q_entry_m':cert['bracket']['q_overlap_m']}))
    invalid_cert.append(('historical_proof_not_current_root',{**cert,'not_engine_entry_certificate':True}))
    for name,certificate in invalid_cert:
        run('invalid_entry_'+name,c,entries={lp:certificate},expected=False)
    run('wrong_target_body',contact(robot,77,lp,-1,-.002),entries={lp:cert},expected=False)
    run('wrong_target_link',contact(robot,target,lp,0,-.002),entries={lp:cert},expected=False)
    run('wrong_robot_link',contact(robot,target,99,-1,.001),entries={lp:cert},expected=False)
    run('wrong_robot_body',contact(77,target,lp,-1,.001),entries={lp:cert},expected=False)
    run('wrong_scene_environment',contact(88,99,-1,-1,.001),expected=False)
    for index,name in ((8,'distance'),(9,'normal_force'),(10,'tangent_force')):
        bad=deepcopy(c);bad[index]=float('nan');run('nonfinite_'+name,bad,entries={lp:cert},expected=False)
    bad=deepcopy(c);bad[5][0]=float('inf');run('nonfinite_witness',bad,entries={lp:cert},expected=False)
    run('unsupported_dynamic_channel',c,channel='dynamic',entries={lp:cert},expected=False)
    run('unknown_phase',c,phase='posthoc',entries={lp:cert},expected=False)
    run('unregistered_self_collision',contact(robot,robot,lb,rb,-2e-9),channel='proxy',expected=False)
    run('filtered_pair_manifold_returns',contact(robot,robot,links['wrist_3_link'],links['ip140_palm'],-.001036),source='manifold',expected=False)
    run('active_pads_negative_gap',contact(robot,robot,lp,rp,-2e-9),channel='proxy',expected=False)
    run('robot_table_penetration',contact(robot,table,lp,-1,-2e-9),channel='proxy',expected=False)
    run('robot_table_zero_force_manifold',contact(robot,table,lp,-1,.0001),source='manifold',expected=False)
    run('filtered_base_table_manifold_returns',contact(robot,table,links['base_link_inertia'],-1,-.0010000957),source='manifold',expected=False)
    run('registered_object_table_support_geometry',contact(target,table,-1,-1,0.),source='manifold')
    run('support_below_old_minus1mm',contact(target,table,-1,-1,math.nextafter(-.001,-math.inf)),source='manifold',expected=False)
    # Effective-face tests use the exact frozen URDF coordinates, never a relaxed surrogate.
    I=np.eye(4);valid_face=contact(robot,target,lp,-1,0.)
    face_variants=[('effective_face_positive',deepcopy(valid_face),True),
        ('effective_face_native_reversed',reverse(valid_face),True)]
    for name,edit in (
        ('outside_face_y',lambda r:(r[5].__setitem__(1,2e-9),r[6].__setitem__(1,2e-9))),
        ('outside_face_x',lambda r:(r[5].__setitem__(0,.060+2e-9),r[6].__setitem__(0,.060+2e-9))),
        ('outside_face_z',lambda r:(r[5].__setitem__(2,.010+2e-9),r[6].__setitem__(2,.010+2e-9))),
        ('nonunit_normal',lambda r:r.__setitem__(7,[0.,-2.,0.])),
        ('wrong_inward_direction',lambda r:r.__setitem__(7,[0.,1.,0.])),
        ('witness_residual',lambda r:r[6].__setitem__(0,r[6][0]+2e-9)),
        ('nonfinite_face_witness',lambda r:r[5].__setitem__(0,float('nan')))):
        r=deepcopy(valid_face);edit(r);face_variants.append((name,r,False))
    for name,r,expected in face_variants:
        out=validate_contact_row(r,robot=robot,target=target,pad_index=lp,W_T_pad=I,W_T_G=I,side='left',require_face=True)
        cases.append(dict(name=name,passed=out['passed']==expected,expected_accepted=expected,input=safe(r),actual=out,synthetic=True))
    evaluator_hash=sha(HERE/'object_evaluator.py');old_evaluator_hash=sha(OLD/'object_evaluator.py')
    cases.append(dict(name='dynamic_evaluator_byte_identical_to_smoke13',passed=evaluator_hash==old_evaluator_hash,
        new_sha256=evaluator_hash,old_sha256=old_evaluator_hash))
    ev=ObjectEvaluator(robot,target,table,links)
    for name,d,expected in [('dynamic_exact_minus1mm_retains_boundary',-.001,True),
                            ('dynamic_nextafter_below_minus1mm_rejected',math.nextafter(-.001,-math.inf),False),
                            ('dynamic_reference_overtravel_number_is_still_physical_failure',-.001125493581548879,False)]:
        row=contact(robot,target,lp,-1,d,force=1.)
        out,violations,_,_=ev._contact(row,0)
        accepted='external_penetration_below_minus_1mm' not in violations
        cases.append(dict(name=name,passed=accepted==expected,expected_accepted=expected,raw=row,
            actual=dict(contact=out,violations=violations),scope='unchanged dynamic evaluator on synthetic returned-contact input; no physics'))
    # Read every existing record, retaining complete raw targeted observations.
    trace=OLD/'path_precheck/path_snapshots.jsonl.gz';selected={};prior_clean=True
    with gzip.open(trace,'rt',encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            row=json.loads(line)
            if row['snapshot']<=2751 and row.get('passed') is not True:prior_clean=False
            if row['snapshot'] in(2751,2752,2757,2758):selected[row['snapshot']]=dict(line=number,data=row)
    source_sha=sha(trace)
    observations=[];face_checks={}
    for index in (2751,2752,2757,2758):
        r=selected[index]['data'];G=np.asarray(r['actual_W_T_G']);T=np.eye(4);T[:3,3]=[-.060,-r['finger_target_q_m'],0.]
        pad=G@T;row=next(c for c in r['raw_closest']['robot_target'] if c[3]==lp)
        face_checks[index]=validate_contact_row(row,robot=robot,target=target,pad_index=lp,W_T_pad=pad,W_T_G=G,side='left',require_face=True)
        observations.append(dict(snapshot=index,jsonl_line_1based=selected[index]['line'],q_ref_m=r['finger_target_q_m'],
            source_relative_path=trace.relative_to(ROOT).as_posix(),source_sha256=source_sha,raw_contact=row,
            frame_derivation='actual_W_T_G from original record times frozen URDF prismatic+fixed pad transform',
            W_T_G=G.tolist(),W_T_pad=pad.tolist(),effective_face_audit=face_checks[index]))
    dp=observations[0]['raw_contact'][8];dc=observations[1]['raw_contact'][8]
    qp=observations[0]['q_ref_m'];qc=observations[1]['q_ref_m'];q_linear=qp-dp*(qc-qp)/(dc-dp)
    historical_proof=dict(previous_snapshot=2751,current_snapshot=2752,
        previous_signed_distance_m=dp,current_signed_distance_m=dc,precontact_clean=prior_clean,
        q_zero_linear_interpolation_m=q_linear,recorded_face_audit=face_checks[2752],
        evidence_kind='historical bracket and face observations; not current root certificate',
        not_engine_entry_certificate=True,admission_passed=False)
    comparisons=[]
    for index in (2757,2758):
        r=selected[index]['data'];records=[]
        for source,rows in [('manifold',r['raw_world_manifolds']),('robot_target',r['raw_closest']['robot_target'])]:
            for row_index,c in enumerate(rows):
                if {c[1],c[2]}!={robot,target}:continue
                new=classify_contact(c,source=source,phase=r['phase'],channel='reference',robot=robot,target=target,table=table,links=links,entry_by_pad={})
                is_recorded_pad=(c[3]==lp or c[4]==lp)
                label=('post_contact_motor_reference_overtravel' if is_recorded_pad and c[8]<=0 else
                       'unchanged_clearance_or_other_geometry')
                records.append(dict(source=source,row_index=row_index,raw_contact=c,
                    strict_gate_without_new_entry_certificate=new,
                    conditional_semantic_label=label,
                    condition='Only if fresh same-side root and proxy independently pass; historical proof cannot admit',
                    current_admission_passed=False))
                if (c[3]==lp or c[4]==lp) and c[8]<-.001:
                    _,v,_,_=ev._contact(c,0)
                    cases.append(dict(name='historical_'+str(index)+'_'+source+'_still_rejected_by_dynamic_minus1mm',
                        passed='external_penetration_below_minus_1mm' in v,raw_contact=c,actual_violations=v,
                        scope='actual immutable static raw contact passed to unchanged dynamic classifier; not a physical result'))
        comparisons.append(dict(snapshot=index,jsonl_line_1based=selected[index]['line'],original_passed=r['passed'],
            original_violations=r['violations'],records=records,
            actual_new_engine_entry_or_proxy_validated=False,
            conclusion='Only raw-reference semantics changes conditionally; original smoke13 stop remains. New entry localization and proxy must independently pass.'))
    cases.append(dict(name='historical_actual_positive_to_negative_face_bracket',
        passed=prior_clean and dp>0 and dc<0 and all(a['passed'] for a in face_checks.values()),
        previous_signed_distance_m=dp,current_signed_distance_m=dc,face_checks=face_checks,
        new_entry_or_proxy_proven=False))
    cases.append(dict(name='historical_2757_2758_conditional_reference_reclassification',
        passed=all(y['strict_gate_without_new_entry_certificate']['violations'] and y['conditional_semantic_label']=='post_contact_motor_reference_overtravel'
            for x in comparisons for y in x['records'] if y['raw_contact'][3]==lp or y['raw_contact'][4]==lp),
        scope='Reference interpretation recognized; actual gate correctly rejects absent fresh root/proxy. Not path admission or grasp evidence'))
    return dict(version=HERE.name,schema=SCHEMA,passed=all(c['passed'] for c in cases),cases=cases,
        source_sha256={name:sha(HERE/name) for name in ('contact_semantics.py','contact_entry.py','positive_negative_replay.py','object_evaluator.py')},
        acquisition_policy=dict(path='acquisition_policy.json',sha256=sha(HERE/'acquisition_policy.json'),
            scope='future approved static-query acquisition policy; this replay performs no acquisition or connection'),
        total_cases=len(cases),physics_steps=0,connections=0,charged=False,
        dynamic_evaluator_sha256=evaluator_hash,old_dynamic_evaluator_sha256=old_evaluator_hash,
        frozen_effective_face_spec=model_face_spec(),
        historical_reclassification=dict(source_path=trace.relative_to(ROOT).as_posix(),source_sha256=source_sha,
            original_stop_result_sha256=sha(OLD/'path_precheck/result.json'),
            raw_source_records_preserved=True,observations=observations,comparisons=comparisons,
            q_zero_linear_interpolation_m=q_linear,historical_only_proof=historical_proof,
            limitation='Linear interpolation is offline calculation, not new bisection. No fresh proxy/entry result is claimed.'),
        scope='Pure rule replay only; no PyBullet imported or connected; no physical outcome inferred.')


def persist_report(*,replace_unfrozen_proofs=False):
    mode='x'
    if replace_unfrozen_proofs:
        if (HERE/'input_identity.json').exists():raise RuntimeError('Frozen proofs cannot be replaced')
        for name in ('positive_negative_replay.json','static_gate_semantics.json'):
            archived=HERE/'development_before_cache_isolation'/name
            if not archived.is_file() or sha(archived)!=sha(HERE/name):
                raise RuntimeError('Prior proof must be copied and hash-preserved before metadata refresh')
        mode='w'
    report=build()
    with (HERE/'positive_negative_replay.json').open(mode,encoding='utf-8',newline='\n') as f:
        json.dump(safe(report),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    semantics=dict(schema=SCHEMA,version=HERE.name,acquisition_policy=report['acquisition_policy'],
        source_sha256={name:sha(HERE/name) for name in ('contact_semantics.py','contact_entry.py','positive_negative_replay.py','object_evaluator.py')},
        entry_certificate_numeric_requirements=dict(q_range_m=[0.,.070],bracket_width_max_m=1e-10,
            clear_distance_strict_positive=True,overlap_distance_nonpositive=True,q_entry_equals_clear_endpoint=True,
            max_absolute_entry_distance_m=1e-9,max_face_outside_m=1e-9,max_full_witness_residual_m=1e-9,
            max_normal_unit_error=1e-9,on_effective_inner_face_required=True,inward_alignment_strict_positive=True),
        reference={'meaning':'frozen commanded q_ref; not simulated loaded state','postcontact_exception_phases':['close','lift','hold'],
            'allowed_reference_only_labels':['post_contact_motor_reference_overtravel','post_contact_bone_motor_reference_overclosure'],
            'required':'same-side valid entry certificate; no palm/arm exemption'},
        proxy={'meaning':'static contact-limited geometry; never controller target or loaded prediction',
            'nonpad_target':'hard failure in every phase','pad_penetration_below_m':-.001,
            'entry_face':'exact frozen local y=0, x[0,.060], z[-.010,.010], full residual and face tolerance1e-9'},
        unchanged={'dynamic_external_penetration_lower_m':-.001,'dynamic_evaluator_sha256':report['dynamic_evaluator_sha256'],
            'precontact':'any early pad contact/negative gap rejected','nonpad_palm_arm':'hard in both channels',
            'robot_table_self_unknown_nonfinite':'strict original gates remain','filters':14},
        no_new_dynamic_approval=True,offline_replay_sha256=sha(HERE/'positive_negative_replay.json'),
        mandatory_supplementary_gates=['registered internal assembly invariants','active pad seam witnesses','fixed base-table mount invariant','entry effective face and no nonpad collision','unchanged trajectory and asset hashes'])
    with (HERE/'static_gate_semantics.json').open(mode,encoding='utf-8',newline='\n') as f:
        json.dump(semantics,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return report


if __name__=='__main__':
    import sys
    if Path.cwd().resolve()!=ROOT or sys.argv[1:]!=['--offline-only']:
        raise SystemExit('Explicit project root and --offline-only required')
    r=persist_report();print(json.dumps(dict(passed=r['passed'],cases=r['total_cases'],connections=0,physics_steps=0)))

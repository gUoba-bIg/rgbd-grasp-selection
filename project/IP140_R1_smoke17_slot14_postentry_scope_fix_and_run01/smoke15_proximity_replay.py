"""smoke15 pure positive-gap manifold checks. No PyBullet import or scene connection."""
from copy import deepcopy
import gzip,hashlib,json,math
from pathlib import Path
import numpy as np
from contact_semantics import classify_contact,validate_entry_certificate
from contact_entry import validate_contact_row

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SMOKE14=ROOT/'IP140_R1_smoke14_slot14_contact_aware_static_admission01'
LABEL='positive_gap_zero_force_proximity_manifold'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text('utf-8-sig'))


def safe(value):
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite_literal':repr(value)}
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    return value


def row(a,b,la,lb,d=.0005):
    n=[0.,-1.,0.];pa=[.03,0.,0.];pb=[.03,d,0.]
    return [0,a,b,la,lb,pa,pb,n,d,0.,0.,[1.,0.,0.],0.,[0.,0.,1.]]


def reverse(c):
    r=deepcopy(c);r[1],r[2]=r[2],r[1];r[3],r[4]=r[4],r[3]
    r[5],r[6]=r[6],r[5];r[7]=[-v for v in r[7]];return r


def build():
    links={name:r['index'] for name,r in read(SMOKE14/'zero_step/all_links_readback.json').items()}
    robot,target,table=0,2,1;pad=links['ip140_left_pad'];right=links['ip140_right_pad']
    canonical=row(robot,target,pad,-1)
    cert=dict(passed=True,robot_body_id=robot,target_body_id=target,pad_link_index=pad,phase='close',
        previous_snapshot=2751,current_snapshot=2752,previous_signed_distance_m=.00001,current_signed_distance_m=-.00015,
        precontact_clean=True,q_entry_m=.032,entry_audit=dict(passed=True,synthetic=True,signed_distance_m=2e-11,
            face_outside_m=0.,full_witness_residual_m=0.,normal_unit_error=0.,on_effective_inner_face=True,inward_normal_alignment=1.),
        bracket=dict(q_clear_m=.032,q_overlap_m=.032-5e-11,d_clear_m=2e-11,d_overlap_m=-3e-11,width_m=.032-(.032-5e-11)),
        nonpad_target_collisions_at_entry=[],synthetic=True)
    cases=[]
    def run(name,raw,*,fresh=None,expected=True,phase='close',channel='reference',source='manifold',entries=None,label=None,face=None):
        out=classify_contact(raw,source=source,phase=phase,channel=channel,robot=robot,target=target,table=table,
            links=links,entry_by_pad=entries,fresh_closest=fresh)
        accepted=not out['violations'];audit=None
        if face is not None:
            audit=validate_contact_row(raw,robot=robot,target=target,pad_index=pad,
                W_T_pad=face[0],W_T_G=face[1],side='left',require_face=True)
            accepted=accepted and audit['passed']
        cases.append(dict(name='smoke15_'+name,passed=accepted==expected and (label is None or out['classification']==label),
            expected_accepted=expected,raw=safe(raw),fresh_closest=safe(fresh),context=safe(dict(phase=phase,channel=channel,source=source,entry_by_pad=entries)),
            actual=out,effective_face_audit=audit,synthetic=True))
    for channel in ('reference','proxy'):
        for phase in ('approach','close'):
            run('positive_gap_zero_force_'+channel+'_'+phase,canonical,fresh=[canonical],channel=channel,phase=phase,label=LABEL)
    run('reverse_manifold_order',reverse(canonical),fresh=[canonical],label=LABEL)
    run('reverse_fresh_order',canonical,fresh=[reverse(canonical)],label=LABEL)
    run('both_native_orders_reversed',reverse(canonical),fresh=[reverse(canonical)],label=LABEL)
    run('with_irrelevant_nonpad_clearance',canonical,fresh=[row(robot,target,links['ip140_left_bone'],-1,.0025),canonical],label=LABEL)
    for value in (None,[]):run('missing_fresh_'+str(value),canonical,fresh=value,expected=False)
    for name,changed in [('wrong_body',row(robot,77,pad,-1)),('wrong_robot_body',row(77,target,pad,-1)),
        ('wrong_pad',row(robot,target,right,-1)),('wrong_object_link',row(robot,target,pad,0))]:
        run('fresh_identity_'+name,canonical,fresh=[changed],expected=False)
    for name,d in [('zero',0.),('negative',-1e-15),('distance_mismatch',canonical[8]+2e-9)]:
        run('fresh_distance_'+name,canonical,fresh=[row(robot,target,pad,-1,d)],expected=False)
    small=row(robot,target,pad,-1,2e-9)
    run('fresh_difference_exact_1e9_boundary',small,fresh=[row(robot,target,pad,-1,1e-9)],label=LABEL)
    run('fresh_difference_nextafter_rejected',row(robot,target,pad,-1,math.nextafter(2e-9,math.inf)),fresh=[row(robot,target,pad,-1,1e-9)],expected=False)
    for index,name in ((9,'Fn'),(10,'Ft1'),(12,'Ft2')):
        for sign in (-1.,1.):
            bound=deepcopy(canonical);bound[index]=sign*1e-9
            run('force_exact_band_'+name+str(sign),bound,fresh=[canonical],label=LABEL)
            over=deepcopy(canonical);over[index]=sign*math.nextafter(1e-9,math.inf)
            run('force_nextafter_'+name+str(sign),over,fresh=[canonical],expected=False,label='positive_gap_nonzero_force_inconsistency')
            run('ordinary_closest_force_anomaly_'+name+str(sign),over,source='robot_target',expected=False,label='positive_gap_nonzero_force_inconsistency')
            run('valid_entry_cannot_hide_positive_force_'+name+str(sign),over,fresh=[canonical],entries={pad:cert},expected=False,label='positive_gap_nonzero_force_inconsistency')
        fresh=deepcopy(canonical);fresh[index]=2e-9
        run('fresh_force_anomaly_'+name,canonical,fresh=[fresh],expected=False)
    run('multiple_branches_cannot_select_favorable_positive',canonical,
        fresh=[canonical,row(robot,target,pad,-1,.0004)],expected=False)
    run('multiple_branches_negative_minimum',canonical,
        fresh=[canonical,row(robot,target,pad,-1,-.0001)],expected=False)
    run('multiple_branches_minimum_exact_match',canonical,
        fresh=[row(robot,target,pad,-1,.0006),canonical],label=LABEL)
    for name,edit in [('distance',lambda c:c.__setitem__(8,float('nan'))),
        ('normal_force',lambda c:c.__setitem__(9,float('inf'))),('tangent_force',lambda c:c.__setitem__(10,float('nan'))),
        ('witness',lambda c:c[5].__setitem__(0,float('nan'))),('normal',lambda c:c[7].__setitem__(0,float('inf')))]:
        c=deepcopy(canonical);edit(c);run('manifold_nonfinite_'+name,c,fresh=[canonical],expected=False)
        run('fresh_nonfinite_'+name,canonical,fresh=[c],expected=False)
    for name,link in [('bone',links['ip140_left_bone']),('palm',links['ip140_palm']),('wrist',links['wrist_3_link']),('arm',links['upper_arm_link'])]:
        for channel in ('reference','proxy'):
            c=row(robot,target,link,-1)
            run('nonpad_no_proximity_exception_'+name+'_'+channel,c,fresh=[c],channel=channel,entries={pad:cert},expected=False)
    for d,name in [(0.,'zero'),(-1e-15,'negative')]:
        for phase in ('approach','close'):
            c=row(robot,target,pad,-1,d)
            run('uncertified_'+name+'_'+phase,c,fresh=[canonical],phase=phase,expected=False)
        run('certified_'+name+'_still_not_proximity',row(robot,target,pad,-1,d),fresh=[canonical],entries={pad:cert},
            label='post_contact_motor_reference_overtravel')
    run('ordinary_positive_closest_no_manifest_needed',canonical,source='robot_target')
    I=np.eye(4)
    run('positive_proximity_effective_face',canonical,fresh=[canonical],face=(I,I),label=LABEL)
    for name,edit in [('face_y',lambda c:(c[5].__setitem__(1,2e-9),c[6].__setitem__(1,c[6][1]+2e-9))),
        ('normal',lambda c:c.__setitem__(7,[0.,-2.,0.])),('witness_residual',lambda c:c[6].__setitem__(0,c[6][0]+2e-9))]:
        c=deepcopy(canonical);edit(c)
        run('proximity_cannot_bypass_'+name,c,fresh=[canonical],face=(I,I),expected=False)
    trace=SMOKE14/'dual_channel_snapshots.jsonl.gz';raw_record=None;line_number=None
    with gzip.open(trace,'rt',encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            r=json.loads(line)
            if r['snapshot']==2748:raw_record=r;line_number=i;break
    if raw_record is None:raise ValueError('Original smoke14 snapshot2748 missing')
    copied=read(SMOKE14/'postrun_evidence/snapshot_2748.json')
    if copied!=raw_record:raise ValueError('Existing extracted snapshot differs from raw gzip')
    source=dict(version=SMOKE14.name,path=trace.relative_to(ROOT).as_posix(),sha256=sha(trace),jsonl_line_1based=line_number,
        extracted_path=(SMOKE14/'postrun_evidence/snapshot_2748.json').relative_to(ROOT).as_posix(),
        extracted_sha256=sha(SMOKE14/'postrun_evidence/snapshot_2748.json'),original_passed=raw_record['passed'],
        original_violations=raw_record['violations'],original_unchanged=True)
    facts=[]
    for channel in ('reference','proxy'):
        record=raw_record[channel+'_channel'];manifold=record['raw_manifolds'][0];fresh=record['raw_closest']['robot_target']
        for reversed_native in (False,True):
            c=reverse(manifold) if reversed_native else manifold
            out=classify_contact(c,source='manifold',phase=raw_record['phase'],channel=channel,
                robot=robot,target=target,table=table,links=links,entry_by_pad={},fresh_closest=fresh)
            face=validate_contact_row(c,robot=robot,target=target,pad_index=pad,W_T_pad=record['W_T_links']['ip140_left_pad'],
                W_T_G=record['W_T_G'],side='left',require_face=True)
            name='smoke15_actual_smoke14_2748_'+channel+('_reversed' if reversed_native else '_native')
            cases.append(dict(name=name,passed=not out['violations'] and out['classification']==LABEL and face['passed']
                and out['geometric_contact'] is False and out['force_active'] is False,
                actual=out,effective_face_audit=face,raw=c,synthetic=reversed_native,
                source=source,semantic_reclassification_only=True,original_result_rewritten=False))
            facts.append(dict(channel=channel,reversed_native=reversed_native,raw=c,fresh_closest=fresh,
                interpretation=out,effective_face_audit=face))
    return dict(version=HERE.name,passed=all(c['passed'] for c in cases),cases=cases,total_cases=len(cases),
        source_sha256={name:sha(HERE/name) for name in ('contact_semantics.py','contact_entry.py','smoke15_proximity_replay.py')},
        actual_smoke14_2748=dict(source=source,records=facts),connections=0,physics_steps=0,charged=False,
        scope='Pure semantic replay. Old failure retained; this does not prove fresh static or physical success.')

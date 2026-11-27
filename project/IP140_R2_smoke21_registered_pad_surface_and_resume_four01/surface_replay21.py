"""Offline R2 surface admission using archived crossing queries and explicit mutations."""
from collections import Counter
from contact_aware_path import classify_channel
from pathlib import Path
import gzip
import json
import hashlib
import math
import sys
from tempfile import TemporaryDirectory
from copy import deepcopy
import numpy as np
from contact_entry import find_contact_entry, validate_contact_row, _safe
from contact_semantics import validate_entry_certificate
from registered_pad_surface21 import registered_surface
from scope_replay20 import RecordedQueries

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
R17=ROOT/'IP140_R1_smoke17_slot14_postentry_scope_fix_and_run01'
R19=ROOT/'IP140_R1_smoke19_asset_typed_zero_schema_and_resume_five01'
R20=ROOT/'IP140_R1_smoke20_positive_clearance_face_scope_and_resume_four01'

def read(path):return json.loads(path.read_bytes())
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def replay_entry(base, side, *, simultaneous_bone=False):
    original_base=base
    branch=next(key for root,key in [(R17,'smoke17'),(R19,'smoke19'),(R20,'smoke20')] if base.is_relative_to(root))
    if not (base/'contact_entries'/side/'queries.jsonl.gz').is_file():
        base=HERE/'replay_inputs21'/branch
    parent=base/'contact_entries'/side
    source_parent=original_base/'contact_entries'/side
    cert=read(parent/'result.json')
    with gzip.open(parent/'queries.jsonl.gz','rb') as f:
        records=[json.loads(line) for line in f]
    indices=read(base/'path_result.json')['joint_indices']
    links={name:r['index'] for name,r in read(base/'zero_step/all_links_readback.json').items()}
    last=records[-1]
    if simultaneous_bone:
        overlap=deepcopy(last['raw_robot_target_closest'][last['selected_raw_index']])
        a=overlap[1]==cert['robot_body_id']
        overlap[3 if a else 4]=links['ip140_'+side+'_bone']
        overlap[8]=-.0002
        overlap[5]=(np.array(overlap[6])+overlap[8]*np.array(overlap[7])).tolist()
        last['raw_robot_target_closest'].append(overlap)
    mock=RecordedQueries(records,indices)
    with TemporaryDirectory(dir=HERE,prefix='archived_query_replay_') as folder:
        result=find_contact_entry(mock,0,cert['robot_body_id'],cert['target_body_id'],links,side=side,arm_joint_indices=indices[:6],
            arm_q=records[0]['robot_arm_q_rad'],q_clear_m=cert['q_clear_input_m'],q_overlap_m=cert['q_overlap_input_m'],
            other_finger_q_m=cert['other_finger_q_m'],source_snapshot=cert['source_snapshot'],output_dir=Path(folder)/'outputs')
    errors=validate_entry_certificate(result,robot=cert['robot_body_id'],target=cert['target_body_id'],pad=cert['pad_link_index'])
    audit=result['entry_audit']
    geom=registered_surface(side,audit['pad_witness_pad_link_m'],audit['normal_pad_to_object_pad'])
    ok=(mock.i==len(records) and result['queries_completed']==len(records) and result['q_entry_m']==cert['q_entry_m'] and result['previous_signed_distance_m']>0>=result['current_signed_distance_m']
        and (('simultaneous_nonpad_target_collision_at_entry' in result['violations'] and errors and not result['passed']) if simultaneous_bone else (result['passed'] and not errors)))
    data=dict(historical_result_path=(source_parent/'result.json').relative_to(ROOT).as_posix(),historical_result_sha256=sha(parent/'result.json'),original_query_sha256=sha(parent/'queries.jsonl.gz'),
        stored_R1_pass=cert['passed'],queries_consumed=mock.i,total_queries=len(records),bracket=result['bracket'],
        original_q_entry_m=cert['q_entry_m'],q_entry_m=result['q_entry_m'],previous_snapshot=result['previous_snapshot'],current_snapshot=result['current_snapshot'],
        previous_d=result['previous_signed_distance_m'],current_d=result['current_signed_distance_m'],new_entry_audit=audit,
        surface=geom,root_violations=result['violations'],certificate_errors=errors,simultaneous_nonpad_overlap=result['nonpad_target_collisions_at_entry'])
    return bool(ok),data,result

def replay_handoff(cert, add):
    base=R20/'slots/09/static_preflight'
    src=base/'dual_channel_snapshots.jsonl.gz'
    archive=HERE/'replay_inputs21/smoke20'
    provenance=read(archive/'state2761_source.json')
    if src.is_file():
        with gzip.open(src,'rb') as stream:
            for line in stream:
                pass
        if sha(src)!=provenance['source_sha256']:
            raise ValueError('historical snapshot stream SHA changed')
    else:
        with gzip.open(archive/'state2761.jsonl.gz','rb') as stream:
            line=stream.read()
    if hashlib.sha256(line).hexdigest()!=provenance['raw_record_sha256']:
        raise ValueError('state2761 uncompressed record SHA changed')
    state=json.loads(line)
    assert state['snapshot']==2761 and state['phase']=='close'
    link_file=base/'zero_step/all_links_readback.json'
    if not link_file.is_file():
        link_file=archive/'zero_step/all_links_readback.json'
    links={name:r['index'] for name,r in read(link_file).items()}
    replaced={'entry_witness_outside_registered_effective_face','entry_solver_not_passed','entry_witness_face_audit_missing_or_failed',
        'entry_face_outside_m','entry_not_on_effective_inner_face','pad_contact_without_legal_entry','missing_per_side_entry_certificate',
        'reference_negative_or_manifold_before_legal_entry'}
    input_digest=provenance['source_sha256']
    for channel_name in ('reference_channel','proxy_channel'):
        c=deepcopy(state[channel_name])
        old=list(c['violations'])
        c['violations']=[v for v in old if v.split(':')[-1] not in replaced]
        certified={cert['pad_link_index']:cert}
        counts=Counter()
        passed=classify_channel(c,'close',robot=cert['robot_body_id'],target=cert['target_body_id'],table=1,links=links,certs=certified,classes=counts)
        rows=[]
        for audit in c['pad_witness_audits']:
            src_name=audit['query'];i=audit['row_index']
            raw=(c['raw_manifolds'] if src_name=='manifold' else c['raw_closest'][src_name])[i]
            if raw[8]<=0 or audit['audit']['violations']:
                classification=next(item for item in c['classifications'] if item['query']==src_name and item['row_index']==i)
                rows.append(dict(query=src_name,index=i,native_row=raw,classification=classification,witness_audit=audit))
        actual_fresh={r['index']:r for r in rows if r['query']=='robot_target'}
        cache_indices={r['index'] for r in rows if r['query']=='manifold'}
        expected_fresh=({28,46}==set(actual_fresh) and all(not r['witness_audit']['audit']['violations'] for r in actual_fresh.values()))
        original_reference_overtravel=(channel_name!='reference_channel' or all(r['classification']['classification']=='post_contact_motor_reference_overtravel' for r in actual_fresh.values()))
        add('archived_017_2761_handoff_'+channel_name,passed and expected_fresh and cache_indices=={0,1,2,3} and original_reference_overtravel,
            'stored unmodified native static returns; prospective R2 classification with freshly re-solved archived root',
            dict(source=src.relative_to(ROOT).as_posix(),source_sha256=input_digest,snapshot=2761,phase=state['phase'],q_ref_m=state['q_ref_m'],
                archived_q_proxy_m=state['q_proxy_m'],archive_warning='Old proxy never applied the accepted R2 root; both archived channels replayed as captured, no load or new path.',
                new_root_m=cert['q_entry_m'],historical_violations=old,retained_independent_violations=[v for v in old if v.split(':')[-1] not in replaced],
                production_channel_passed=passed,production_violations=c['violations'],production_class_counts=dict(counts),rows=rows))

def run():
    cases=[]
    def add(name,passed,kind,evidence):
        cases.append(dict(name=name,passed=bool(passed),data_origin=kind,evidence=_safe(evidence)))
    casesets=[('archived021',R17/'static_preflight',['left','right'],'flat_inner_face'),('archived014',R19/'slots/08/static_preflight',['left','right'],'flat_inner_face'),('archived017',R20/'slots/09/static_preflight',['left'],'inner_tip_quarter_cylinder')]
    for alias,base,sides,expected in casesets:
        for side in sides:
            ok,data,raw=replay_entry(base,side)
            add(alias+'_'+side+'_complete_native_bisection',ok and data['surface']['surface_type']==expected,
                'historical native returns supplied to current root solver, no new native calls',data)
    summary=raw
    replay_handoff(summary,add)
    ok,d,_=replay_entry(R20/'slots/09/static_preflight','left',simultaneous_bone=True)
    add('simultaneous_nonpad_overlap_rejects_valid_arc',ok,'mutation of one native root return',d)
    for title,changes in [('missing_positive_origin',{'previous_signed_distance_m':0.}),('wrong_crossing_sign',{'current_signed_distance_m':1e-5}),('nonconsecutive_states',{'previous_snapshot':summary['previous_snapshot']-1})]:
        cert=deepcopy(summary);cert.update(changes)
        errors=validate_entry_certificate(cert,robot=cert['robot_body_id'],target=cert['target_body_id'],pad=cert['pad_link_index'])
        add(title,bool(errors),'mutated certificate',{'changes':changes,'errors':errors})
    # Independent metric construction in local SI units. The production geometry reads its own frozen registration.
    length,thickness,height,radius=.060,.002,.020,.0005
    angle=math.pi/4
    for side,sign,pad in [('left',1.,13),('right',-1.,15)]:
        center=np.array([length-radius,-sign*radius,0.])
        n=np.array([math.sin(angle),sign*math.cos(angle),0.])
        tangent=np.array([math.cos(angle),-sign*math.sin(angle),0.])
        arc=center+radius*n
        def check(label,p,normal,expected_type,*,wrong_id=False,bad_residual=False,reverse=False):
            pos=np.array(p,float);direction=np.array(normal,float);sep=3e-11
            # Native nB points from B to A. Construct one ordered pair with a known residual.
            pa=pos.copy();pb=pa+sep*direction;nb=-direction
            if bad_residual:pb[2]+=2e-9
            row=[0,0,2,pad-1 if wrong_id else pad,-1,pa.tolist(),pb.tolist(),nb.tolist(),sep,0.,0.,[0.,1.,0.],0.,[1.,0.,0.]]
            if reverse:
                row[1:5]=[2,0,-1,pad];row[5:8]=[pb.tolist(),pa.tolist(),direction.tolist()]
            evidence=validate_contact_row(row,robot=0,target=2,pad_index=pad,W_T_pad=np.eye(4),W_T_G=np.eye(4),side=side,require_face=True)
            geom=registered_surface(side,p,normal)
            expect=expected_type is not None
            passed=bool(evidence['passed'])==expect and (not expect or geom['surface_type']==expected_type)
            add(side+'_'+label,passed,'constructed pad-local geometry',dict(point_m=pos,expected_allowed_type=expected_type,normal=direction,geometry=geom,actual_violations=evidence['violations']))
        check('flat', [.024,0.,0.], [0,sign,0], 'flat_inner_face')
        check('arc', arc,n,'inner_tip_quarter_cylinder')
        check('arc_reversed_endpoints', arc,n,'inner_tip_quarter_cylinder',reverse=True)
        for delta,allow in [(0.75e-9,True),(1.25e-9,False),(-1.25e-9,False),(1.3e-5,False)]:
            check('radius_delta_'+str(delta),center+(radius+delta)*n,n,'inner_tip_quarter_cylinder' if allow else None)
        for theta,allow in [(0.75e-9,True),(1.25e-9,False),(.01,False)]:
            check('normal_direction_error_'+str(theta),arc,n*math.cos(theta)+tangent*math.sin(theta),'inner_tip_quarter_cylinder' if allow else None)
        check('reversed_normal',arc,-n,None)
        check('witness_residual',arc,n,None,bad_residual=True)
        check('wrong_pad_link',arc,n,None,wrong_id=True)
        check('wrong_radius',center+radius*1.1*n,n,None)
        check('arc_outside_quadrant',center+radius*np.array([-n[0],n[1],0.]), [-n[0],n[1],0],None)
        outer=np.array([length-radius,-sign*(thickness-radius),0.]);nout=np.array([n[0],-n[1],0.])
        check('outer_tip_round',outer+radius*nout,nout,None)
        check('outer_flat',[.02,-sign*thickness,0.],[0,-sign,0],None)
        check('back_face',[0.,-sign*thickness/2,0.],[-1.,0,0],None)
        check('tip_front_flat',[length,-sign*thickness/2,0.],[1.,0,0],None)
        check('tip_front_endpoint',center+np.array([radius,0,0]),[1.,0,0],None)
        for zsign in (-1,1):
            nz=np.array([.5,sign*.5,zsign*math.sqrt(.5)])
            dz=np.array([0.,0.,zsign*(height/2-radius)])
            check('upper_lower_spherical_round_'+str(zsign),center+dz+radius*nz,nz,None)
            check('outside_straight_cylinder_extent_'+str(zsign),arc+dz+[0,0,zsign*2e-9],n,None)
    return dict(version=HERE.name,scope='R2 registered pad static surface; offline only',passed=all(c['passed'] for c in cases),cases=cases,case_count=len(cases),
        physical_steps=0,engine_connections=0,charged=0,sources={f.name:sha(f) for f in [HERE/'surface_replay21.py',HERE/'registered_pad_surface21.py',HERE/'contact_entry.py',HERE/'contact_semantics.py',HERE/'scope_replay20.py',HERE/'model/manifest.json',HERE/'model/ur5e_ip140_r1.urdf']})
if __name__=='__main__':
    result=run()
    print([(c['name'],c['evidence'].get('actual_violations'),c['evidence'].get('certificate_errors')) for c in result['cases'] if not c['passed']])
    if sys.argv[1:]==['--seal']:
        with (HERE/'surface_replay21.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(result['passed'],result['case_count'])
    raise SystemExit(0 if result['passed'] else 2)

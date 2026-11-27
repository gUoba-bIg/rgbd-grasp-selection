"""Smoke07 offline validation on sealed smoke05/06 inputs with shared runtime checks."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from mountpair import check_mount_invariant
from seam_geometry import PADS, check_manifold, check_row
from structural_filter import audit_scene_manifolds

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TELEMETRY = ROOT/'IP140_R1_smoke05_fixed_pad_mount_filter01/episodes/03/telemetry.jsonl.gz'
FROZEN_TELEMETRY = 'd1eb49fe401aacf150d20b4a0d4d7ae226c385eb57922475ccc42393e55224a7'
DT = 1/240
SCHEMA = 'IP140_smoke07_offline_gate_replay_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(obj):
    # Nonfinite mutations must survive in standard JSON with an explicit representation.
    if isinstance(obj, float) and not math.isfinite(obj):
        return {'nonfinite_float':repr(obj)}
    if isinstance(obj, np.ndarray):
        return json_value(obj.tolist())
    if isinstance(obj, dict):
        return {k:json_value(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):
        return [json_value(v) for v in obj]
    if isinstance(obj,np.generic):
        return json_value(obj.item())
    return obj


def preserved_gates(initial,state):
    arm = state['actual_arm_and_finger_states'][:6]
    fingers = state['actual_arm_and_finger_states'][6:]
    geo = state['geometry']
    drift = state['tcp_G_drift']
    cmd = state['commanded_q_m']
    width = state['station_width_mm']/1000.
    outside = any(not 0.<=x[0]<=.070 for x in fingers)
    gap_ok = all(v>=-1.e-9 for v in geo['gaps_m'])
    finite = all(math.isfinite(v) for row in arm+fingers for v in (row[0],row[1],*row[2],row[3]))
    stationary = (gap_ok and abs(geo['gap_error_m'])<=.001 and abs(geo['midline_m'])<=.0005
                  and abs(geo['left_error_m'])<=.0005 and abs(geo['right_error_m'])<=.0005
                  and geo['abs_symmetry_error_m']<=.001 and geo['spread_m']<=.0005)
    tracking = (abs(geo['gap_m']-2*cmd)<=.001 and abs(geo['left_error_m']-width/2+cmd)<=.0005
                and abs(geo['right_error_m']+width/2-cmd)<=.0005)
    checks = dict(finite=finite,arm_q=all(abs(x[0]-h)<=.01 for x,h in zip(arm,initial['arm_home'])),
        arm_dq=all(abs(x[1])<=.01 for x in arm),
        arm_effort=all(abs(x[3])<f for x,f in zip(arm,state['arm_cap_Nm'])),
        finger_command=0.<=cmd<=.070,
        finger_q=all(-1.e-5<=x[0]<=.07001 for x in fingers),
        finger_dq=all(abs(x[1])<=.040 for x in fingers),
        finger_effort=all(abs(x[3])<=20. for x in fingers),
        tcp_translation=drift['translation_m']<=.005,tcp_rotation=drift['rotation_deg']<=3.,
        corner_gaps=gap_ok,midline=abs(geo['midline_m'])<=.0005,
        symmetry=geo['abs_symmetry_error_m']<=.001,parallel=geo['spread_m']<=.0005,
        measured_rail_band_tracking=not outside or tracking,
        settled=not state['in_settled_window'] or stationary,
        matches_recorded_geometry=(stationary==state['within_frozen_tolerance'] and
            outside==state['measured_outside_nominal'] and tracking==state['dynamic_face_tracking_in_rail_band']))
    keys=('index','station_tick','station_width_mm','joints','actual_arm_and_finger_states','commanded_q_m',
          'arm_cap_Nm','tcp_G_drift','geometry','in_settled_window','within_frozen_tolerance','violations')
    return {'passed':all(checks.values()),'checks':checks,'unchanged_inputs':{k:state[k] for k in keys}}


def replay_state(initial,state,links,line_number):
    step=state['index']
    poses={k:np.asarray(v,dtype=float) for k,v in state['W_T_link_post'].items()}
    raw_rows=[];checks=[];points=[]
    for name,info in state['seam_audit']['pairs'].items():
        pair=tuple(name.split('|'))
        ids=tuple(links[n] for n in pair)
        if tuple(info['link_ids'])!=ids:
            raise ValueError('historical link id mismatch '+name)
        for i,row in enumerate(info['raw_points']):
            result=check_row(pair,ids,row,poses,initial['body'],DT)
            points.append({'pair':name,'row':i,'raw':row,'result':result})
        for i,row in enumerate(info['manifolds']):
            result=check_manifold(pair,ids,row,initial['body'],DT)
            checks.append({'historical_pair':name,'row':i,'raw':row,'result':result})
            raw_rows.append(json.dumps(row))
    intact = Counter(raw_rows)==Counter(json.dumps(row) for row in state['raw_manifolds'])
    unchanged=preserved_gates(initial,state)
    ok=(intact and len(checks)==(12 if step>=5 else 0) and all(c['result']['passed'] for c in checks)
        and len(points)==5 and all(c['result']['passed'] for c in points) and unchanged['passed'])
    return {'step':step,'telemetry_jsonl_line':line_number,'passed':ok,
            'status':'all_named_replay_gates_pass' if ok else 'failed_named_replay_gate',
            'unchanged_motion_and_geometry':unchanged,'source_raw_manifolds_exhausted':intact,
            'closest_results':points,'persistent_normal_results':checks,'raw_contacts_count':len(checks)}


def negative_cases(initial,state,links,line):
    body=initial['body']
    world={k:np.asarray(v,dtype=float) for k,v in state['W_T_link_post'].items()}
    slide=('ip140_palm','ip140_left_bone')
    sid=tuple(links[n] for n in slide);pid=tuple(links[n] for n in PADS)
    orig={};loc={}
    for tag,pair,stream in (('m',slide,'manifolds'),('c',PADS,'raw_points')):
        name='|'.join(pair)
        orig[tag]=state['seam_audit']['pairs'][name][stream][0]
        loc[tag]=f"seam_audit.pairs[{name!r}].{stream}[0]"
    tests=[]
    def add(number,label,tag,mode,changes,prefix,gate_pair=None):
        row=deepcopy(orig[tag])
        for k,new in changes.items():
            row[k]=new
        if mode=='persistent':
            verdict=check_manifold(slide,sid,row,body,DT)
        elif mode=='closest':
            verdict=check_row(PADS,pid,row,world,body,DT)
        else:
            verdict=audit_scene_manifolds(body,links,[row])
        rejected=verdict.get('passed') is False
        detected=any(v.startswith(prefix) for v in verdict['violations'])
        tests.append({'number':number,'case':label,'source_step':state['index'],'telemetry_jsonl_line':line,
            'source_field':loc[tag],'mode':mode,'target_pair':gate_pair or (PADS if tag=='c' else slide),
            'raw_source':orig[tag],'mutated_input':row,'mutation':[{'tuple_index':k,'before':orig[tag][k],
                'after':new} for k,new in changes.items()],'expected':'reject','actual':'reject' if rejected else 'accept',
            'required_violation_prefix':prefix,'targeted_violation_observed':detected,'passed':rejected and detected,
            'result':verdict})
    x=orig['m'];pa=np.asarray(x[5]);n=np.asarray(x[7]);d=x[8]
    add(1,'negative_signed_distance','m','persistent',
        {5:(pa+(-2e-9-d)*n).tolist(),8:-2e-9},'penetration_below_numeric_band')
    add(2,'normal_inconsistency','m','persistent',{5:(pa+2e-9*n).tolist()},'normal_residual')
    for k,s in ((9,'Fn'),(10,'Ft1'),(12,'Ft2')):
        add(3,s+'_over_force_limit','m','persistent',{k:2e-9},'reported_force')
    add(4,'impulse_only_via_finite_subthreshold_normal_force','m','persistent',
        {9:2e-12/DT,10:0.,12:0.},'vector_impulse')
    add(5,'nonunit_normal','m','persistent',{7:(2*n).tolist()},'nonunit_normal')
    add(6,'wrong_body','m','persistent',{2:body+1},'pair_identity')
    add(6,'wrong_link','m','persistent',{4:links['shoulder_link']},'pair_identity')
    add(7,'NaN_in_point_coordinate','m','persistent',{5:[float('nan'),*pa[1:].tolist()]},'nonfinite_field')
    c=orig['c'];direction=world[PADS[0]][:3,0]
    add(8,'closest_full_residual','c','closest',{5:(np.asarray(c[5])+5e-9*direction).tolist()},
        'witness_residual')
    shift=.2*direction
    add(9,'both_closest_witnesses_beyond_shared_face','c','closest',
        {5:(np.asarray(c[5])+shift).tolist(),6:(np.asarray(c[6])+shift).tolist()},'outside_registered_face:')
    # Run exact copied historical slide contact through the prospective empty-scene classifier.
    # It was a legitimate record of an enabled smoke05 pair; in smoke06 it is prohibited.
    add(10,'filtered_contact_reappears','m','scene',{},'filtered_manifold:')
    add(11,'unregistered_external_zero_force_contact','m','scene',{2:body+1,4:-1},
        'unregistered_empty_scene_manifold:',gate_pair=['robot','unknown external body'])
    return tests


SMOKE06 = ROOT/'IP140_R1_smoke06_sliding_seam_acm01'
MOUNT_INPUT_HASHES = {
    'zero_step/filter_post.json':'73c74ef5a527c93d8f4e4d74692b929cd3fe90c892b1a7fdb66d3d355c4cce90',
    'episodes/04/telemetry.jsonl.gz':'1da7202c1b926d54c659f7f820683d7d11a6f3ea49743a7165e08a35320b9a6c',
    'zero_step/pre_filter_wrist_fixed_chain.json':'6e101e7771d8561c3779fe0a048bc8450d7c0ab4d1af4ccffe68b69a44237238',
}
D_REF_M = -0.0010361016226507268
BASELINE_TOL_M = 1e-9
WRIST = 'wrist_3_link'
PALM = 'ip140_palm'
MOUNT_KEY = WRIST+'|'+PALM


def mount_sources():
    hashes = {rel:sha(SMOKE06/rel) for rel in MOUNT_INPUT_HASHES}
    if hashes!=MOUNT_INPUT_HASHES:
        raise ValueError('archived smoke06 input identity mismatch')
    zero=json.loads((SMOKE06/'zero_step/filter_post.json').read_text('utf-8'))
    topology=json.loads((SMOKE06/'zero_step/pre_filter_wrist_fixed_chain.json').read_text('utf-8'))
    with gzip.open(SMOKE06/'episodes/04/telemetry.jsonl.gz','rt',encoding='utf-8') as stream:
        steps=[(i,r) for i,line in enumerate(stream,1)
               if (r:=json.loads(line)).get('event')=='state']
    if len(steps)!=1 or steps[0][1]['index']!=1 or zero['physics_steps']!=0:
        raise ValueError('expected archived post-filter zero and one returned physical step')
    line,step=steps[0]
    post=zero['pairs'][MOUNT_KEY]
    dyn=step['filtered_pairs_audit']['pairs'][MOUNT_KEY]
    specs=[
        (0,'post_filter_zero_step','zero_step/filter_post.json',f'pairs[{MOUNT_KEY!r}]',
         post['link_ids'],post['closest'],post['manifolds'],zero['W_T_link']),
        (1,'dynamic','episodes/04/telemetry.jsonl.gz',
         f'line {line}: filtered_pairs_audit.pairs[{MOUNT_KEY!r}]',dyn['link_ids'],
         dyn['current_closest'],dyn['manifolds'],step['W_T_link_post'])
    ]
    out=[]
    for index,stage,path,locator,ids,rows,manifold,world in specs:
        if ids!=[7,11] or len(rows)!=1 or manifold!=[]:
            raise ValueError(f'unexpected archived mount witness identity or multiplicity: step {index}')
        args={'body':topology['body_id'],'ids':ids,'closest_rows':rows,'manifold_rows':manifold,
              'W_T_wrist':world[WRIST],'W_T_G':world[PALM],'topology':topology,'stage':stage}
        out.append(dict(step=index,stage=stage,file=path,locator=locator,inputs=args))
    return hashes,out


def replay_mount():
    hashes,sources=mount_sources()
    positives=[]
    for source in sources:
        observation=check_mount_invariant(**source['inputs'])
        accepted=observation.get('passed') is True
        positives.append(dict(**source,result=observation,passed=accepted,expected='accept',
                              actual='accept' if accepted else 'reject'))
    dyn=sources[1]
    base=dyn['inputs']
    row=base['closest_rows'][0]
    negatives=[]

    def add(case,prefix,changes,description):
        inputs=deepcopy(base)
        inputs.update(deepcopy(changes))
        obs=check_mount_invariant(**inputs)
        rejected=obs.get('passed') is False
        targeted=any(reason.startswith(prefix) for reason in obs['violations'])
        negatives.append(dict(case=case,step=dyn['step'],source=dyn['file'],locator=dyn['locator'],
            raw_source='mount_positives[1].inputs',mutation_description=description,
            mutation={field:{'before':base[field],'after':new} for field,new in changes.items()},
            expected='reject',actual='reject' if rejected else 'accept',required_violation_prefix=prefix,
            targeted_violation_observed=targeted,passed=rejected and targeted,result=obs))

    for sign,label in [(-1,'minus'),(1,'plus')]:
        modified=deepcopy(row)
        modified[8]=D_REF_M+sign*2e-9
        # Shift wrist witness by n*delta_d to preserve the instantaneous witness equation.
        displacement=(modified[8]-row[8])*np.asarray(row[7])
        modified[5]=(np.asarray(row[5])+displacement).tolist()
        add(f'baseline_{label}_2nm','mount_distance_baseline',{'closest_rows':[modified]},
            f'set d to d_ref {label} 2 nm; move pA by n*(d_new-d_old)')
    wrist=np.asarray(base['W_T_wrist'])
    shifted=wrist.copy()
    shifted[:3,3]+=wrist[:3,0]*4e-9
    add('translation','fixed_transform_translation',{'W_T_wrist':shifted.tolist()},
        'translate W_T_wrist 4 nm along local X; G and witness world coordinates copied')
    rotated=wrist.copy()
    # Rotate about local Z at the wrist origin. Z is the mount translation axis.
    theta=1e-5
    c,s=math.cos(theta),math.sin(theta)
    rotated[:3,:3]=wrist[:3,:3]@np.array([[c,-s,0],[s,c,0],[0,0,1]])
    add('rotation','fixed_transform_rotation',{'W_T_wrist':rotated.tolist()},
        'rotate W_T_wrist by 1e-5 rad about wrist-local Z at the same origin')
    moved=deepcopy(row)
    shift=np.asarray(base['W_T_G'])[:3,1]*.2
    for i in (5,6):
        moved[i]=(np.asarray(row[i])+shift).tolist()
    add('witness_outside','mount_face',{'closest_rows':[moved]},
        'translate pA and pB equally 0.2 m along G lateral Y, preserving vector residual')
    add('manifold','mount_solver_manifold',{'manifold_rows':[deepcopy(row)]},
        'insert copied zero-force wrist/palm record in the dynamic solver stream')
    wrong=deepcopy(row)
    wrong[4]=12
    add('identity','pair_identity',{'closest_rows':[wrong]},'replace reported palm id 11 with bone id 12')
    invalid=deepcopy(row)
    invalid[9]=float('nan')
    add('nonfinite','nonfinite_field',{'closest_rows':[invalid]},
        'replace reported normal force by NaN, serialized with explicit nonfinite tag')
    return hashes,positives,negatives




def build():
    digest=sha(TELEMETRY)
    if digest!=FROZEN_TELEMETRY:raise ValueError('historical telemetry digest differs')
    with gzip.open(TELEMETRY,'rt',encoding='utf8') as stream:
        records=[(i,json.loads(line)) for i,line in enumerate(stream,1)]
    initial,=[r for _,r in records if r['event']=='initial_configuration']
    links={name:int(i) for i,name in initial['links'].items()}
    states=[(i,r) for i,r in records if r['event']=='state']
    if [r['index'] for _,r in states]!=list(range(1,7)) or initial['step_seconds']!=DT:
        raise ValueError('unexpected historical trajectory or step duration')
    positives=[replay_state(initial,r,links,i) for i,r in states]
    negatives=negative_cases(initial,states[-1][1],links,states[-1][0])
    passed=all(p['passed'] for p in positives+negatives) and {r['number'] for r in negatives}==set(range(1,12))
    mount_hashes,mount_positives,mount_negatives=replay_mount()
    passed=passed and all(r['passed'] for r in mount_positives+mount_negatives)
    return {'mount_input_sha256':mount_hashes,'mount_positives':mount_positives,'mount_negatives':mount_negatives,
        'mount_distance_reference_m':D_REF_M,'mount_distance_tolerance_m':BASELINE_TOL_M,
        'mount_registration_sha256':sha(HERE/'mountpair_registration.json'),
        'schema':SCHEMA,'passed':passed,'input_telemetry_path':TELEMETRY.relative_to(ROOT).as_posix(),
        'input_telemetry_sha256':digest,'replay_source_sha256':sha(__file__),
        'gate_source_hashes':{k:sha(HERE/k) for k in ('seam_geometry.py','structural_filter.py','mountpair.py')},
        'initial_body_id':initial['body'],'link_map':links,'step_duration_s':DT,
        'scope':'Archived smoke05 seam and motion checks plus smoke06 post-filter zero and first-step wrist '
                'invariant replay. Existing recorded run verdicts are preserved. These are offline results.',
        'source_reference':'JSONL line numbers are one-based in decompressed original.',
        'positives':positives,'negatives':negatives}


def main():
    if Path.cwd().resolve()!=ROOT:raise RuntimeError('project root working directory required')
    product=HERE/'offline_gate_replay.json'
    if product.exists():raise FileExistsError(product)
    payload=build()
    with product.open('x',encoding='utf-8',newline='\n') as out:
        json.dump(json_value(payload),out,ensure_ascii=False,indent=2,allow_nan=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    return 0 if payload['passed'] else 2


if __name__=='__main__':raise SystemExit(main())


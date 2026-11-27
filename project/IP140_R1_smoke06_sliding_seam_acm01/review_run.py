"""Read-only reconstruction from the audit package. No simulator or project imports."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(name):
    return json.loads((HERE/name).read_bytes().decode('utf-8-sig'))

def rows(name):
    with gzip.open(HERE/name,'rt',encoding='utf-8') as stream:
        for line in stream: yield json.loads(line)

def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def review():
    reg=read('preregistration.json')
    assert reg['eligible_slots_after_zero_gate']==[4] and reg['prior_physical_starts']==3
    hist=read('history/sources.json')
    for f in hist:
        assert digest(HERE/f['package'])==f['sha256']
    for e in read('ledger_initial.json')['slots'][:3]:
        assert e['charged'] and e['state']=='returned_failed' and read(e['result'])['slot_passed'] is False
        assert digest(HERE/e['telemetry'])==read(e['result'])['telemetry_sha256']
    replay=read('offline_gate_replay.json');gate=read('zero_readback.json')
    assert replay['passed'] and all(x['passed'] for x in replay['positives']+replay['negatives'])
    assert {t['number'] for t in replay['negatives']}==set(range(1,12))
    assert gate['passed'] and not gate['required_mismatches'] and gate['physics_step_calls']==0
    pre=read('zero_step/filter_pre.json');post=read('zero_step/filter_post.json')
    names=reg['approved_filter_order']
    assert len(names)==len(pre['pairs'])==len(post['pairs'])==13
    assert all(pre['pairs'][k]['closest'] and post['pairs'][k]['closest'] and not post['pairs'][k]['manifolds'] for k in names)
    counters=collections.Counter()
    filters=[]; steps=[]; unknown=[]; last=None; seq=0
    wrist_id=pre['pairs']['wrist_3_link|ip140_palm']['link_ids']
    active_step=0; wrist_samples={};other_closest={}
    for row in rows('full_api/calls.jsonl.gz'):
        if row[1]=='call':
            assert last is None and row[0]==seq+1
            seq=row[0];last=row;counters[row[3]]+=1
        elif row[1]=='return':
            assert last is not None and last[0]==row[0]
            method=last[3];args=last[4];kw=last[5];val=row[3]
            if method.startswith('setCollisionFilter'):
                assert method=='setCollisionFilterPair' and val is None
                filters.append({'seq':row[0],'args':args,'kwargs':kw})
            elif method=='stepSimulation':
                active_step+=1;steps.append(row[0])
            elif method=='getClosestPoints' and active_step and 'linkIndexA' in kw:
                ids=[kw['linkIndexA'],kw['linkIndexB']]
                if ids==wrist_id:
                    assert val
                    nearest=min(val,key=lambda p:p[8])
                    residual=math.dist([nearest[5][i]-nearest[6][i] for i in range(3)],[nearest[8]*v for v in nearest[7]])
                    wrist_samples[active_step]=dict(api_seq=row[0],raw_min_distance_m=nearest[8],full_vector_residual_m=residual)
                other_closest.setdefault(active_step,set()).add(tuple(ids))
            last=None
        else:unknown.append(row[:2]);last=None
    api=read('full_api/summary.json')
    assert not unknown and last is None and dict(counters)==api['calls']==api['returns']
    assert counters['connect']==counters['loadURDF']==counters['disconnect']==1
    assert len(filters)==13 and all(r['args']==[gate['body_id'],gate['body_id'],*pre['pairs'][k]['link_ids'],0] for r,k in zip(filters,names))
    assert not steps or filters[-1]['seq']<steps[0]
    for h,size in api['lossless_raw_blobs_bytes'].items():
        with gzip.open(HERE/'full_api'/(h+'.bin.gz'),'rb') as f:
            content=f.read()
        assert len(content)==size and hashlib.sha256(content).hexdigest()==h
    res=read('episodes/04/result.json')
    assert res['telemetry_sha256']==digest(HERE/'episodes/04/telemetry.jsonl.gz')
    stations=[dict(order=i+1,opening_mm=w,returned_samples=0,passed_samples=0,stable_window_samples=0,
                   complete=False,worst_observed=None) for i,w in enumerate(reg['slots'][3]['opening_sequence_mm'])]
    failure=None;ns=0;intents=0;returns=0;violation_steps=0
    for r in rows('episodes/04/telemetry.jsonl.gz'):
        event=r['event']
        if event=='step_intent':intents+=1
        if event=='step_return':returns+=1
        if event!='state':continue
        ns+=1; assert r['index']==ns and failure is None
        st=stations[(ns-1)//480]
        assert r['station_width_mm']==st['opening_mm'] and r['station_tick']==(ns-1)%480+1
        st['returned_samples']+=1
        st['stable_window_samples']+=int(r['in_settled_window'])
        st['passed_samples']+=int(not r['violations'])
        assert len(r['q_arm_rad'])==len(r['dq_arm_rad_s'])==len(r['arm_torque_Nm'])==6
        assert len(r['actual_arm_and_finger_states'])==8
        required={tuple(pre['pairs'][k]['link_ids']) for k in names[6:]}
        assert required.issubset(other_closest.get(ns,set()))
        a=r['geometry']
        metrics={'max_abs_arm_delta_q_rad':max(map(abs,r['arm_error_from_home_rad'])),
            'max_abs_arm_speed_rad_s':max(map(abs,r['dq_arm_rad_s'])),
            'max_arm_effort_fraction':max(abs(t)/c for t,c in zip(r['arm_torque_Nm'],r['arm_cap_Nm'])),
            'max_tcp_translation_m':r['tcp_G_drift']['translation_m'],'max_tcp_rotation_deg':r['tcp_G_drift']['rotation_deg'],
            'max_abs_finger_speed_m_s':max(abs(x[1]) for x in r['actual_arm_and_finger_states'][-2:]),
            'max_abs_finger_force_N':max(map(abs,r['sim_motor_force_N'])),
            'max_abs_gap_error_to_station_m':abs(a['gap_error_m']),'max_abs_midline_m':abs(a['midline_m']),
            'max_abs_single_face_error_to_station_m':max(abs(a['left_error_m']),abs(a['right_error_m'])),
            'max_symmetry_error_m':a['abs_symmetry_error_m'],'max_corner_spread_m':a['spread_m'],
            'max_wrist_penetration_m':max(0,-wrist_samples[ns]['raw_min_distance_m'])}
        if st['worst_observed'] is None:st['worst_observed']=metrics
        else:st['worst_observed']={k:max(st['worst_observed'][k],v) for k,v in metrics.items()}
        if r['violations']:
            violation_steps+=1
            failure=dict(step=ns,station_mm=r['station_width_mm'],tick=r['station_tick'],reasons=r['violations'],
                         world_manifold_points=len(r['raw_manifolds']),wrist_closest=wrist_samples[ns])
    assert ns==intents==returns==len(steps)==res['step_returns']==res['step_intents'] and res['unknown_steps']==0
    for s in stations:s['complete']=s['passed_samples']==480 and s['stable_window_samples']==24
    valid=ns==4320 and violation_steps==0 and all(s['complete'] for s in stations)
    assert res['slot_passed']==valid and [s['opening_mm'] for s in stations if s['complete']]==res['tested_stations_mm']
    assert read('session_result.json')['slots']==[4] and read('zero_step/result.json')['disconnected']
    return dict(version=HERE.name,attempt=reg['slots'][3]['candidate_id'],zero_readable_passes=len(gate['fields']),
                offline_positive_steps=len(replay['positives']),offline_negative_variants=len(replay['negatives']),
                offline_negative_classes=len({t['number'] for t in replay['negatives']}),filter_calls=filters,total_api_calls=seq,
                pre_filter_world_contact_points=len(pre['raw_manifolds']),post_filter_world_contact_points=len(post['raw_manifolds']),
                physical_steps=ns,unknown_steps=0,slot_passed=valid,first_violation=failure,stations=stations,
                not_observed='Null worst values denote stations without a physics sample.')
if __name__=='__main__':print(json.dumps(review(),indent=2,ensure_ascii=False))

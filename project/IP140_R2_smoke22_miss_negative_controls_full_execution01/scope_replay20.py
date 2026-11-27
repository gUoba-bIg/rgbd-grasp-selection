"""No engine import. Verify the shared static code using archived native geometry and explicit mutations."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from contact_aware_path import classify_channel
import gzip
import json
import hashlib
import tempfile
import math
import sys
import numpy as np
from contact_semantics import audit_pad_witness, classify_contact, validate_entry_certificate, pad_pair_state
from contact_entry import find_contact_entry, _safe

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PARENT=ROOT/'IP140_R1_smoke19_asset_typed_zero_schema_and_resume_five01'
LEGACY=ROOT/'IP140_R1_smoke17_slot14_postentry_scope_fix_and_run01'
S9=PARENT/'slots/09/static_preflight'
S8=PARENT/'slots/08/static_preflight'
read=lambda p:json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

class RecordedQueries:
    """Serve one actual archived response for each requested pose. No simulator."""
    step_authorized=False
    counts={'stepSimulation':0,'setCollisionFilterPair':14}
    sequence=0
    def __init__(self, queries, indices):
        self.q=queries
        self.i=0
        self.resets=[]
        self.indices=indices
    def getBasePositionAndOrientation(self,body,**kw):
        return self.q[min(self.i,len(self.q)-1)]['object_pose']
    def resetJointState(self,body,index,*,targetValue,**kw):
        self.resets.append((index,targetValue))
    def getLinkStates(self,*args,**kw):
        expected=list(zip(self.indices,self.q[self.i]['all_reset_targets']))
        assert self.resets==expected,'solver requested a state unavailable in the archive'
        self.resets=[]
        return self.q[self.i]['raw_link_states']
    def getClosestPoints(self,*args,**kw):
        return self.q[self.i]['raw_robot_target_closest']
    def getJointStates(self,*args,**kw):
        value=self.q[self.i]['joint_states']
        self.i+=1
        return value

def historic_root(links, outside=False):
    directory=S8/'contact_entries/left'
    cert=read(directory/'result.json')
    with gzip.open(directory/'queries.jsonl.gz','rb') as stream:queries=[json.loads(line) for line in stream]
    if outside:
        final=queries[-1]
        row=final['raw_robot_target_closest'][final['selected_raw_index']]
        wp=np.array(final['world_link_frames']['ip140_left_pad'])
        point=np.array(row[5 if row[1]==0 else 6])
        local=np.linalg.solve(wp,np.r_[point,1.])
        # Pure translation of both witnesses preserves the signed distance and full residual.
        delta=wp[:3,1]*(2e-9-local[1])
        for j in (5,6):row[j]=(np.asarray(row[j])+delta).tolist()
    indices=read(S8/'path_result.json')['joint_indices']
    adapter=RecordedQueries(queries,indices)
    with tempfile.TemporaryDirectory(dir=HERE,prefix='offline_root_only_') as tmp:
        outcome=find_contact_entry(adapter,0,0,2,links,side='left',arm_joint_indices=indices[:6],
            arm_q=queries[0]['robot_arm_q_rad'],q_clear_m=cert['q_clear_input_m'],q_overlap_m=cert['q_overlap_input_m'],
            other_finger_q_m=cert['other_finger_q_m'],source_snapshot=cert['source_snapshot'],output_dir=Path(tmp)/'root')
    errors=validate_entry_certificate(outcome,robot=0,target=2,pad=links['ip140_left_pad'])
    evidence={k:outcome.get(k) for k in ('status','passed','queries_completed','q_entry_m','previous_snapshot','current_snapshot','previous_signed_distance_m','current_signed_distance_m','bracket','violations','entry_audit')}
    evidence['certificate_errors']=errors
    return (adapter.i==len(queries) and ((not errors and outcome['q_entry_m']==cert['q_entry_m'] and outcome['passed']) if not outside else
        ('entry_surface_not_registered' in outcome['violations'] and 'entry_not_on_registered_surface' in errors))),evidence

def historic_positive(dirpath):
    prior=read(dirpath/'result.json')
    tailpath=dirpath/prior['terminal_observations_file']; tail=read(tailpath)['observed_rows']
    bilateral=unsupported=high=0
    for row in tail:
        effective=set(); table=0.
        for c in row['raw_manifolds']:
            bodies={c[1],c[2]}
            if bodies=={1,2}:table+=max(0,c[9])
            if bodies=={0,2} and c[9]>.2: effective.add(c[3] if c[1]==0 else c[4])
        bilateral+=int({13,15}<=effective)
        unsupported+=int(table<=.2)
        high+=int(row['object_state']['position'][2]-prior['settle_baseline']['position'][2]>=.1)
    details=dict(historical_result_sha256=sha(dirpath/'result.json'),tail_sha256=sha(tailpath),bilateral=bilateral,unsupported=unsupported,above_height=high,steps=prior['step_returns'])
    return (bilateral==unsupported==high==480 and [x['step'] for x in tail]==list(range(4249,4729)) and prior['grasp_success'] is True and prior['engineering_gate_passed'] is True
            and prior['slot_passed'] is True and prior['step_returns']==4728 and prior['terminal_observations_sha256']==details['tail_sha256']),details

def run():
    out=[]
    def check(name,passed,evidence):
        out.append(dict(name=name,passed=bool(passed),evidence=_safe(evidence)))
    source=S9/'dual_channel_snapshots.jsonl.gz'
    wanted={}
    with gzip.open(source,'rb') as f:
        for i,line in enumerate(f,1):
            if 2757<=i<=2759:wanted[i]=json.loads(line)
            if i==2759:break
    links={k:w['index'] for k,w in read(S9/'zero_step/all_links_readback.json').items()}
    pad=links['ip140_left_pad'];ident=dict(robot=0,target=2,pad_index=pad)
    def same(x):return tuple(x[1:5]) in ((0,2,pad,-1),(2,0,-1,pad))
    def classify(x,source,channel='proxy',**kw):
        return classify_contact(x,source=source,channel=channel,phase='close',robot=0,target=2,table=1,links=links,**kw)
    for i,s in wanted.items():
        for channel in ('reference','proxy'):
            ch=s[channel+'_channel'];raw=ch['raw_closest'];fresh=[r for r in raw['robot_target']+raw.get('robot_target_supplemental',[]) if same(r)]
            cached=[r for r in ch['raw_manifolds'] or [] if same(r)]
            state=pad_pair_state(fresh,cached,**ident); selected=fresh[state['fresh_selected_index']]
            geometry=dict(**ident,W_T_pad=ch['W_T_links']['ip140_left_pad'], W_T_G=ch['W_T_G'],side='left')
            audit=audit_pad_witness(selected,source='closest',phase='close',require_face=True,pair_state=state,**geometry)
            items=[]
            for mode,rr in [('closest',fresh),('manifold',cached)]:
                for row in rr:
                    flag=classify(row,mode,channel,pair_state=state,fresh_closest=selected)
                    w=audit_pad_witness(row,source=mode,phase='close',require_face=True,pair_state=state,**geometry)
                    items.append((flag,w))
            full=deepcopy(ch)
            retained=full["violations"]
            full["violations"]=[e for e in retained if e!="robot_target:28:witness:entry_witness_outside_registered_effective_face"]
            classify_channel(full,'close',robot=0,target=2,table=1,links=links,certs={},classes=Counter())
            authoritative=full['registered_pad_pair_geometry']['ip140_left_pad']
            dist=selected[8]*1000
            expected={2757:.507512,2758:.367244,2759:.230869}[i]
            passed=(s['snapshot']==i and state['positive_clearance'] and not state['violations'] and not state['geometric_contact']
                and full['passed'] and not full['violations'] and authoritative['classification']==state['classification'] and abs(dist-expected)<=.00000051 and all(not a['violations'] and not w['violations'] and not a.get('entry_trigger') for a,w in items))
            if i==2759:
                passed=passed and len(cached)==4 and all(abs(r[8]*1000+.0714322)<1e-6 and all(r[k]==0 for k in (9,10,12)) for r in cached)
                passed=passed and abs(audit['face_outside_m']*1e6-87.3265)<.0001 and audit['surface']['label']=='rounded_tip_edge_contact_risk' and not audit['on_effective_inner_face']
                passed=passed and state['classification']=='positive_clearance_with_zero_force_cached_manifold' and all(a['classification']==state['classification'] for a,w in items if a['source']=='manifold')
            check(f'history_{i}_{channel}',passed,dict(snapshot=i,phase_tick=s['phase_tick'],fresh_signed_distance_m=selected[8],fresh_force_N=selected[9:11]+[selected[12]],
                pair_class=state['classification'],cached_original=cached,selected_original=selected,current_geometric_contact=state['geometric_contact'],
                entry_triggered=any(a.get('entry_trigger',False) for a,w in items),new_certificate=False,full_channel_violations=full['violations'],source_channel_violations=retained,face_witness=audit,violations=state['violations']+[e for a,w in items for e in a['violations']+w['violations']]))
    one=[deepcopy(selected)]
    def state(fs,cs):return pad_pair_state(fs,cs,**ident)
    for source_kind in ['fresh','cached']:
        for k in [9,10,12]:
            fs,cs=deepcopy(one),deepcopy(cached)
            (fs if source_kind=='fresh' else cs)[-1][k]=math.nextafter(1e-9,math.inf)*(-1 if k==12 else 1)
            r=state(fs,cs)
            check(f'{source_kind}_component_{k}_nextfloat_above_force_limit','positive_gap_nonzero_force_inconsistency' in r['violations'],r)
    fs,cs=deepcopy(one),deepcopy(cached);cs[0][10]=-1e-9
    r=state(fs,cs);check('force_numeric_zero_boundary',not r['violations'],r)
    fs,cs=deepcopy(one),deepcopy(cached);fs[0][3]=15 if fs[0][1]==0 else -1
    if fs[0][1]!=0:fs[0][4]=15
    r=state(fs,cs);check('other_pad_fresh_cannot_prove_this_cached_pair',bool(r['violations']),r)
    r=state([],cached);check('cached_has_no_matching_fresh',bool(r['violations']),r)
    fs=deepcopy(one);fs[0][8]=float('nan')
    r=state(fs,cached);check('fresh_nonfinite_cannot_prove_pair',bool(r['violations']),r)
    f=deepcopy(selected);f[5][0]+=2e-8
    w=audit_pad_witness(f,source='closest',pair_state=state([f],cached),**geometry)
    check('fresh_full_witness_residual_is_hard','full_witness_residual' in w['violations'],w['violations'])
    c=deepcopy(cached[0]);c[5]=(np.asarray(c[5])+2e-8*np.asarray(c[7])).tolist()
    w=audit_pad_witness(c,source='manifold',pair_state=state(one,[c]),**geometry)
    check('cached_projection_remains_hard','normal_projection_residual' in w['violations'],w['violations'])
    f=deepcopy(selected);f[8]=-.001001;f[5]=(np.asarray(f[6])+f[8]*np.asarray(f[7])).tolist()
    r=classify(f,'closest',pair_state=state([f],[]))
    check('unbracketed_nonpositive_has_no_certificate','pad_contact_without_legal_entry' in r['violations'],r['violations'])
    cert=read(S8/'contact_entries/left/result.json')
    r=classify(f,'closest',entry_by_pad={pad:cert},pair_state=state([f],[]))
    check('proxy_actual_distance_below_minus_one_mm','proxy_pad_target_penetration' in r['violations'],r['violations'])
    for name in ['ip140_palm','ip140_left_bone','wrist_3_link','shoulder_link']:
        v=deepcopy(f);v[3 if v[1]==0 else 4]=links[name]
        r=classify(v,'closest')
        check('reject_nonpad_target_'+name,bool(r['violations']),r['violations'])
    v=deepcopy(f);v[1]=911
    r=classify(v,'manifold');check('reject_unknown_external_body','unknown_contact_identity' in r['violations'],r['violations'])
    cert2=deepcopy(cert);cert2['previous_snapshot']-=1
    issues=validate_entry_certificate(cert2,**dict(robot=0,target=2,pad=pad))
    check('refuse_nonconsecutive_bracket','entry_snapshot_bracket_invalid' in issues,issues)
    for outside in [False,True]:
        ok,e=historic_root(links,outside)
        check('recorded_solver_and_face_2nm_mutation' if outside else 'recorded_bisection_inside_flat_face',ok,e)
    for nick,folder in [('021',LEGACY/'episodes/14'),('014',PARENT/'slots/08/episodes/08')]:
        ok,e=historic_positive(folder);check('preserve_historic_nominal_'+nick,ok,e)
    schema=read(HERE/'schema_replay19.json')
    check('asset_typed_replay_54',schema['passed'] and len(schema['cases'])==54 and all(r['passed'] for r in schema['cases']),dict(sha256=sha(HERE/'schema_replay19.json')))
    files=[source,HERE/'contact_aware_path.py',HERE/'contact_semantics.py',HERE/'contact_entry.py',HERE/'scope_replay20.py',S8/'contact_entries/left/queries.jsonl.gz']
    return dict(version=HERE.name,scope='pure replay of archived geometry/terminal data, labelled force and witness mutations, archived query responses for bisection',passed=all(r['passed'] for r in out),
        cases=out,case_count=len(out),connections=0,physical_steps=0,source_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files})
if __name__=='__main__':
    report=run()
    print([(r['name'],r['evidence']) for r in report['cases'] if not r['passed']])
    if sys.argv[1:]==['--seal']:
        with (HERE/'scope_replay20.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(report['passed'],report['case_count'])
    raise SystemExit(0 if report['passed'] else 2)

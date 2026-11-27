"""Offline candidate provenance and deterministic frozen IK reference preparation; no simulator imports."""
from pathlib import Path
import ast
import copy
import gzip
import hashlib
import json
import math
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
from batch_context import VERSION, ROOT, SLOT_DIR, SLOT, OBJECT_ID, KIND
from speed_policy import phase_speed
import trajectory

PARENT = ROOT / "IP140_R1_smoke17_slot14_postentry_scope_fix_and_run01"
DELTA = {"021": .012778562556716472, "014": .012087886894313670, "017": .014903531131342462}

def load(p):
    return json.loads(p.read_bytes())

def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,d):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def freeze():
    if Path.cwd().resolve()!=ROOT or 'pybullet' in sys.modules:
        raise RuntimeError('offline, explicit project root required')
    registry_path=PARENT/'preregistration.json'
    old=load(registry_path)
    parent=old['slots'][SLOT-1]
    assert (parent['slot'],parent['kind'],parent['object_id'],parent['not_p3'])==(SLOT,KIND,OBJECT_ID,True)
    assert load(PARENT/'slot_ledger.json')['slots'][SLOT-1]['charged'] is False
    sources={}
    def bind(path,expected=None):
        if not path.resolve().is_relative_to(ROOT):raise ValueError('unbound asset path: '+str(path))
        h=digest(path)
        if expected is not None and h!=expected:raise ValueError('source changed: '+str(path))
        sources[path.relative_to(ROOT).as_posix()]={'sha256':h,'bytes':path.stat().st_size}
        return path
    cfg=load(bind(ROOT/parent['source_config']['project_relative'],parent['source_config']['sha256']))
    for row in parent['source_files']:
        bind(ROOT/row['project_relative'],row['sha256'])
    intent_path=ROOT/next(row['project_relative'] for row in parent['source_files'] if row['project_relative'].endswith('/intent.json'))
    folder=intent_path.parent
    intent=load(intent_path)
    policy=load(folder/'policy_input.json')
    legacy=load(folder/'result.json')
    assert intent['attempt_id']==parent['original_attempt_id'] and intent['slot_id']==parent['original_slot']['slot_id']
    assert intent['config']['sha256']==parent['source_config']['sha256']
    assert policy['candidate_world']==legacy['candidate_world']==parent['candidate_world']
    h=legacy['engineering_proposal_height_adjustment_m']
    assert h==DELTA[OBJECT_ID]==parent['height_adjustment_m_as_recorded']
    mathfile=bind(ROOT/'scripts/sim_v52_math.py',cfg['bindings']['scripts/sim_v52_math.py']['sha256'])
    defs=[x for x in ast.parse(mathfile.read_text('utf-8-sig')).body if isinstance(x,ast.FunctionDef) and x.name in ('observed_proposal','unproject')]
    assert len(defs)==2 and not any(isinstance(x,(ast.Import,ast.ImportFrom)) for fn in defs for x in ast.walk(fn))
    env={'np':np}
    exec(compile(ast.Module(body=defs,type_ignores=[]),str(mathfile),'exec'),env)
    with np.load(folder/'observed_rgbd.npz',allow_pickle=False) as z:
        points=env['unproject'](z['depth_buffer'],z['view'],z['projection'])
    raw,obs=env['observed_proposal'](points,parent['original_slot']['yaw'],cfg['table_z'],cfg['workspace'])
    assert obs['point_count']==policy['observation']['point_count'] and obs['estimated_object_width_m']==policy['observation']['estimated_object_width_m']
    a=raw.copy()
    if KIND=='miss': a[13]+=cfg['negative_controls']['miss_x_m']
    b=np.array(parent['candidate_world'],dtype=float)
    b[15]-=h
    diff=float(np.max(np.abs(a-b)))
    changed=[i for i in range(17) if struct.pack('>d',b[i])!=struct.pack('>d',parent['candidate_world'][i])]
    assert np.isfinite(a).all() and np.isfinite(b).all() and diff<=1e-12 and changed==[15] and np.array_equal(np.delete(a,15),np.delete(b,15))
    R=b[4:13].reshape(3,3); G=np.eye(4);G[:3,:3]=R;G[:3,3]=b[13:16]+R[:,0]*b[3]
    G_old=np.array(parent['T_W_G_m'])
    assert (np.argwhere(G!=G_old)==[[2,3]]).all() and abs((G_old[2,3]-G[2,3])-h)<1e-12
    obj=parent['scene_setup_only']['urdf'];urdf=bind(ROOT/obj['project_relative'],obj['sha256'])
    for x in ET.parse(urdf).getroot().iter('mesh'):bind(urdf.parent/x.attrib['filename'])
    robot=load(VERSION/'model/manifest.json');bind(VERSION/'model/ur5e_ip140_r1.urdf',robot['model_sha256'])
    proof=dict(slot=SLOT,new_candidate_id=trajectory.ID,object_id=OBJECT_ID,kind=KIND,not_p3=True,
      parent_id=parent['candidate_id'],parent_registration_path=registry_path.relative_to(ROOT).as_posix(),parent_registration_sha256=digest(registry_path),
      parent_frozen_record=parent,parent_record_canonical_sha256=hashlib.sha256(json.dumps(parent,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
      old_candidate=parent['candidate_world'],new_candidate=b.tolist(),npz_observed_before_registered_miss_offset=raw.tolist(),
      source_branch_world_x_shift_m=cfg['negative_controls']['miss_x_m'] if KIND=='miss' else 0.,
      npz_reconstructed_candidate=a.tolist(),legacy_height_removed_m=h,two_path_max_abs_error=diff,
      changed_candidate_indices=changed,changed_pose_indices=[[2,3]],old_G=G_old.tolist(),new_G=G.tolist(),
      parent_sources=sources,scene_setup_unchanged=True,checks_passed=True)
    s=copy.deepcopy(parent);s.update(candidate_id=trajectory.ID,parent_id=parent['candidate_id'],parent_slot=SLOT,candidate_world=b.tolist(),T_W_G_m=G.tolist(),
        legacy_height_bridge=dict(delta_h_legacy_m=h,old_t_z_m=parent['candidate_world'][15],new_t_z_m=float(b[15]),two_path_max_abs_error=diff),
        kind=KIND,accept='Full normal execution, engineering gates passed, grasp_success must be '+str(KIND=='nominal'))
    reg=copy.deepcopy(old);reg['slots'][SLOT-1]=s;reg['damping_evidence']['object_id']=OBJECT_ID
    reg.update(version=VERSION.name,prior_revision=PARENT.name,prior_physical_starts=9,max_starts=14,eligible_slots_after_zero_gate=[SLOT],
        authorized_slot14_only=False,slots8_to12_frozen=False,authorized_batch_order=[8,9,10,11,12],retry=False,
        stop='Stop the batch on the first shared engineering fault; retain scientific outcomes and continue in order.',
        required_completion='Full execution of each authorized slot unless a shared hard gate fails.',
        ids='Original IDs, poses and source hashes are retained in each candidate_difference_proof.json.',
        correction='Only per-object registered legacy world-z adjustment removed.',
        start_requires='Successful offline, independent 4728 state static admission and new dynamic zero-step in frozen batch order.',
        revision_scope='Five IP140 bridge candidates without retraining, reranking or object-dependent control.',
        changed_slots=[SLOT],budget_change={'total':14,'starts_before_batch':9,'reserved':5,'new_budget':0,'repeat':False})
    SLOT_DIR.mkdir(parents=True,exist_ok=True)
    write(SLOT_DIR/'preregistration.json',reg)
    plan=trajectory.build();write(SLOT_DIR/'trajectory_plan.json',plan)
    assert plan['slot']==SLOT and plan['candidate_world']==b.tolist()
    limits=np.asarray(plan['limits_rad_rad_s'])
    ik=[]
    for wp in [*plan['waypoints'],dict(phase='lift',**plan['lift'])]:
        F=trajectory.fk(wp['target_q_rad']);Gt=np.asarray(wp['target_W_T_G']);pos=float(np.linalg.norm(F[:3,3]-Gt[:3,3]));rot=float(np.linalg.norm(trajectory.rotation_vector(F[:3,:3]@Gt[:3,:3].T)))
        assert pos<=1e-9 and rot<=1e-9
        ik.append(dict(phase=wp['phase'],FK_translation_error_m=pos,FK_rotation_error_rad=rot))
    i=0;path=SLOT_DIR/'frozen_trajectory_commands.jsonl.gz'
    with gzip.open(path,'xt',encoding='utf-8',newline='\n',compresslevel=3) as stream:
        for phase,count in trajectory.PHASES:
            for tick in range(1,count+1):
                q,dq,finger=trajectory.reference(plan,phase,tick);i+=1
                assert np.isfinite(q).all() and np.isfinite(dq).all() and 0<=finger<=.07
                assert np.all((q>=limits[:,0]) & (q<=limits[:,1]) & (np.abs(dq)<=limits[:,2]))
                rec=dict(planned_index=i,phase=phase,phase_tick=tick,dt_s=trajectory.DT,arm_target_q_rad=q.tolist(),
                    arm_reference_dq_rad_s=dq.tolist(),finger_reference_q_m=finger,single_finger_command_speed_m_s=phase_speed(phase),
                    scope='offline frozen command; not a returned physics step')
                stream.write(json.dumps(rec,allow_nan=False,separators=(',',':'))+'\n')
    assert i==4728 and trajectory.validate_plan()==plan
    proof.update(artifact_sha256={k:digest(SLOT_DIR/k) for k in ['preregistration.json','trajectory_plan.json','frozen_trajectory_commands.jsonl.gz']},
      regenerated_reference_count=i,ik_roundtrips=ik,passed=True,physics_steps=0)
    write(SLOT_DIR/'candidate_difference_proof.json',proof)
    print(SLOT,OBJECT_ID,KIND,trajectory.ID,diff,changed,i)

if __name__=='__main__':freeze()

"""GT-free finite IK and known robot/table qualification. Each native check is logged."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from platform_imports import VERSION,ROOT,PLATFORM,bind_frozen_libraries
bind_frozen_libraries()
from platform_plan import plan_for
from structural_filter import audit_scene_manifolds
from trajectory import DT,reference
from platform_trace import R4Trace

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def robot_table_path(ctx, plan):
    p=ctx.p;client=ctx.client
    if getattr(ctx,'target',None) is not None or p.counts['stepSimulation'] or p.getNumBodies(physicsClientId=client)!=2:
        raise RuntimeError('eligibility connection must contain exactly robot/table and zero physical steps')
    kw=dict(physicsClientId=client)
    allowed_bodies={ctx.robot,ctx.table}
    if allowed_bodies!={p.getBodyUniqueId(i,**kw) for i in (0,1)}:raise RuntimeError('presence of unknown scene body')
    for phase in ('opening','transit','pregrasp','approach','close','lift'):
        count=dict(plan['phase_steps'])[phase]
        # Frozen v52 planning density, full motion still checked independently after selection.
        ticks=sorted(set(np.rint(np.linspace(1,count,41)).astype(int).tolist()))
        for tick in ticks:
            q,dq,w=reference(plan,phase,tick)
            for j,v in zip([*ctx.arm,*ctx.fingers],[*q,w,w],strict=True):
                p.resetJointState(ctx.robot,j,float(v),targetVelocity=0.,**kw)
            p.performCollisionDetection(**kw)
            contacts=p.getContactPoints(**kw)
            for row in contacts:
                if not all(np.isfinite(np.asarray(row[k],dtype=float)).all() for k in range(1,len(row))):
                    raise RuntimeError('nonfinite closest/manifold in eligibility')
                if {row[1],row[2]}==allowed_bodies:
                    pair=(row[3],row[4]) if row[1]==ctx.robot else (row[4],row[3])
                    if pair==(ctx.links['base_link_inertia'],-1):raise RuntimeError('filtered exact installation pair returned')
                    return dict(passed=False,reason='planned_table_collision',phase=phase,tick=tick)
            self_rows=[row for row in contacts if row[1]==ctx.robot and row[2]==ctx.robot]
            check=audit_scene_manifolds(ctx.robot,ctx.links,self_rows)
            if any(v.startswith('filtered_manifold:') for v in check['violations']):
                raise RuntimeError('registered internal filter pair returned')
            if not check['passed']:return dict(passed=False,reason='planned_self_collision',phase=phase,tick=tick)
    return dict(passed=True,reason=None)

def evaluate(data,ctx):
    if set(data)!= {'unit_id','T_W_C','candidate_rows','rankings'}:raise ValueError('unregistered selection inputs')
    rows=data['candidate_rows'];ranks=data['rankings']
    if set(ranks)!={'Raw','A','C0'} or any(sorted(order)!=list(range(len(rows))) for order in ranks.values()):
        raise RuntimeError('shared pool/ranking mismatch')
    eligibility=[]
    for idx,row in enumerate(rows):
        start=time.perf_counter()
        cid=f"{data['unit_id']}:p3:{idx}"
        try:
            plan=plan_for(row,data['T_W_C'],cid)
        except ValueError as error:
            if str(error) not in ('frozen opening out of bounds','static IK failed to converge','ambiguous IK rotation branch','no feasible IK winding','frozen joint/velocity path limit','frozen lift joint/velocity path limit','target_contact_below_known_table'):
                raise
            c=dict(passed=False,reason=str(error));plan=None
        else:c=robot_table_path(ctx,plan)
        eligibility.append(dict(index=idx,candidate_id=cid,eligible=c['passed'],reason=c['reason'],path_check=c,plan=plan,planning_seconds=time.perf_counter()-start))
    selected={name:next((i for i in ranks[name] if eligibility[i]['eligible']),None) for name in ranks}
    return dict(unit_id=data['unit_id'],eligibility=eligibility,selected=selected,used_ground_truth=False,
                never_fallback_after_postlock_safety=True,scope='41 equally spaced frozen reference ticks per moving/open/close phase in target-absent robot and table scene; post-lock complete 4728-state review')

def main():
    import pybullet as backend
    from scene_world import load_eligibility_world
    arg=argparse.ArgumentParser();arg.add_argument('unit_dir',type=Path);a=arg.parse_args()
    dest=a.unit_dir.resolve()
    if dest.parent!=VERSION/'units' or Path.cwd().resolve()!=ROOT:raise ValueError('unregistered output scope')
    src=dest/'platform_input.json'
    p=R4Trace(backend,dest/'eligibility_api','eligibility')
    cid=p.connect(p.DIRECT)
    try:
        ctx=load_eligibility_world(p,cid,dest/'eligibility_setup')
        out=evaluate(json.loads(src.read_bytes()),ctx)
    finally:
        p.disconnect(physicsClientId=cid);p.close()
    out['selection_input_sha256']=digest(src);out['native_eligibility_api_sha256']=digest(dest/'eligibility_api/calls.jsonl.gz')
    with (dest/'selection.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,allow_nan=False)
if __name__=='__main__':main()

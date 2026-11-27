"""Post-lock zero-physics handoff. Replay the archived 021 stable pose in a new static-only client."""
from pathlib import Path
import gzip,json
from platform_imports import ROOT,VERSION,bind_frozen_libraries
bind_frozen_libraries()
from scene_world import write,read,setup_world
from platform_select import digest
from platform_static import scan
from platform_trace import R4Trace
import pybullet as backend
out=VERSION/'audit/platform_zero_challenge'
def main():
    historical=ROOT/'IP140_R1_smoke17_slot14_postentry_scope_fix_and_run01/episodes/14/telemetry.jsonl.gz'
    with gzip.open(historical,'rt',encoding='utf8') as f:
        rec=next(r for line in f if (r:=json.loads(line)).get('event')=='state' and r['index']==240)
    obj=rec['object_state']
    state=dict(scene_key='scene-021-p2',object_id='021',returned_settle_steps=240,
        object_pose=[obj['position'],obj['orientation']],evidence_role='archived R2 settled pose, not a new R4 stabilized observation',
        archival_source=str(historical.relative_to(ROOT)),archival_sha256=digest(historical))
    select=read(out/'selection.json');idx=select['selected']['Raw']
    row=select['eligibility'][idx]
    intent=dict(trial_id='archived-p3-raw-interface',probe_only=True,method='Raw',selected_pool_index=idx,candidate_id=row['candidate_id'],selection_sha256=digest(out/'selection.json'))
    write(out/'handoff_intent.json',intent)
    write(out/'archived_settle_pose.json',state)
    p=R4Trace(backend,out/'static_api','contact_aware_static');cid=p.connect(p.DIRECT)
    try:
        ctx=setup_world(p,cid,'scene-021-p2',out/'static_setup',settled_state=state)
        result=scan(p,cid,ctx.robot,ctx.links,row['plan'],dict(passed=True,dynamic_execution_authorized=False,eligible_slots_after_zero_gate=[],object_body_id=ctx.target,table_body_id=ctx.table),intent=intent,output_root=out/'static_preflight')
    finally:
        p.disconnect(physicsClientId=cid);p.close()
    print(result['status'],result['snapshots_completed'],result['first_failure'],flush=True)
if __name__=='__main__':main()


"""Freeze slot14 candidate from two pure arithmetic paths and generate a fresh IK trajectory.
This module never imports an engine and never loads a simulation scene.
"""
from pathlib import Path
import ast,copy,datetime,hashlib,itertools,json,struct,sys,traceback,xml.etree.ElementTree as ET
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/"IP140_R1_smoke12_slot07_base_table_mount_filter01"
AUDIT=ROOT/"IP140_R1_smoke12_postrun_candidate_interface_audit01"
ID="ip140bridge:slot14:parent-slot07:undo-legacy-z021"
DELTA_H=.012778562556716472
def read(p):return json.loads(p.read_text("utf-8-sig"))
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def dump(p,d):
    with p.open("x",encoding="utf-8",newline="\n") as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    if Path.cwd().resolve()!=ROOT:raise ValueError("explicit root cwd required")
    for n in ("candidate_difference_proof.json","trajectory_plan.json"):
        if (HERE/n).exists():raise ValueError("refuse replacing a prior candidate preparation")
    checks=[];sources=[]
    def ck(name,passed,**d):
        row=dict(name=name,passed=bool(passed),**d);checks.append(row)
        if not row["passed"]:raise ValueError(name)
    def bind(p,expected=None):
        digest=sha(p)
        if expected is not None:ck("sha:"+p.relative_to(ROOT).as_posix(),digest==expected,actual=digest,expected=expected)
        sources.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=digest,bytes=p.stat().st_size))
        return p
    package_checks=[]
    for base in (OLD,AUDIT):
        archive=base/(base.name+"_审查包.zip")
        sidecar=Path(str(archive)+".sha256")
        expected=sidecar.read_text("utf-8-sig").split()[0]
        actual=sha(archive);ck("package:"+base.name,actual==expected,actual=actual,expected=expected)
        package_checks.append(dict(path=archive.relative_to(ROOT).as_posix(),sha256=actual,sidecar_sha256=sha(sidecar)))
        bind(base/"SHA256SUMS.txt")
    try:
        old=read(bind(OLD/"preregistration.json"))
        current=read(HERE/"preregistration.json")
        ck("new_input_is_exact_smoke12_copy",current==old)
        parent=old["slots"][6]
        ck("parent_identity",parent["slot"]==7 and parent["candidate_id"]=="v52eng:slot04:4f097762f0894ddda2454852daf8da59" and parent["object_id"]=="021" and parent["not_p3"] is True)
        cfg=read(bind(ROOT/parent["source_config"]["project_relative"],parent["source_config"]["sha256"]))
        for s in parent["source_files"]:bind(ROOT/s["project_relative"],s["sha256"])
        intent_path=ROOT/next(s["project_relative"] for s in parent["source_files"] if s["project_relative"].endswith("/intent.json"))
        intent=read(intent_path);policy=read(intent_path.with_name("policy_input.json"));result=read(intent_path.with_name("result.json"))
        mathpath=bind(ROOT/"scripts/sim_v52_math.py",cfg["bindings"]["scripts/sim_v52_math.py"]["sha256"])
        names={"unproject","observed_proposal"}
        nodes=[n for n in ast.parse(mathpath.read_text("utf-8-sig")).body if isinstance(n,ast.FunctionDef) and n.name in names]
        ck("pure_math_definition_names",{n.name for n in nodes}==names)
        ck("no_import_in_selected_functions",not any(isinstance(n,(ast.Import,ast.ImportFrom)) for node in nodes for n in ast.walk(node)))
        env={"np":np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(mathpath),"exec"),env)
        with np.load(intent_path.with_name("observed_rgbd.npz"),allow_pickle=False) as z:
            points=env["unproject"](z["depth_buffer"],z["view"],z["projection"])
        observed,obs=env["observed_proposal"](points,parent["original_slot"]["yaw"],cfg["table_z"],cfg["workspace"])
        frozen=np.array(parent["candidate_world"],dtype=np.float64)
        correction=result["engineering_proposal_height_adjustment_m"]
        ck("source_delta_exact",correction==DELTA_H,actual=correction)
        ck("source_policy_equals_parent",policy["candidate_world"]==parent["candidate_world"] and result["candidate_world"]==parent["candidate_world"])
        bridge=frozen.copy();bridge[15]-=DELTA_H
        error=float(np.max(np.abs(observed-bridge)))
        ck("two_path_candidate_max_error",error<=1e-12,max_abs_error=error,tolerance=1e-12)
        ck("observation_identity",obs["point_count"]==policy["observation"]["point_count"] and obs["estimated_object_width_m"]==policy["observation"]["estimated_object_width_m"])
        element_checks=[{"index":i,"old":float(frozen[i]),"new":float(bridge[i]),
            "same_binary_float64":struct.pack(">d",frozen[i])==struct.pack(">d",bridge[i])} for i in range(17)]
        changed=[r["index"] for r in element_checks if not r["same_binary_float64"]]
        ck("only_candidate_index15_changes",changed==[15],changed_indices=changed)
        ck("all_other_npz_elements_binary_equal",all(struct.pack(">d",observed[i])==struct.pack(">d",frozen[i]) for i in range(17) if i!=15))
        ck("frozen_tz_and_new_tz",frozen[15]==.06381280043712453 and bridge[15]==.05103423788040806)
        ck("finite_candidate",np.isfinite(bridge).all())
        R=bridge[4:13].reshape(3,3);G=np.eye(4);G[:3,:3]=R;G[:3,3]=bridge[13:16]+R[:,0]*bridge[3]
        ck("new_G_z",abs(float(G[2,3])-.031034237880408058)<=1e-12,actual=float(G[2,3]))
        model=read(bind(HERE/"model/manifest.json"))
        bind(HERE/"model/ur5e_ip140_r1.urdf",model["model_sha256"])
        F_T_G=np.array(model["F_T_G"],dtype=float)
        ck("mount_070",F_T_G[0,3]==.070)
        F=G@np.linalg.inv(F_T_G);Gback=F@F_T_G;tback=G[:3,3]-R[:,0]*bridge[3]
        translation_error=float(np.linalg.norm(G[:3,3]-Gback[:3,3]))
        rv=Gback[:3,:3]@G[:3,:3].T;skew=np.array([rv[2,1]-rv[1,2],rv[0,2]-rv[2,0],rv[1,0]-rv[0,1]])
        rot_error=float(np.arctan2(np.linalg.norm(skew),np.trace(rv)-1))
        ck("flange_G_roundtrip",translation_error<=1e-9 and rot_error<=1e-9,translation_error_m=translation_error,rotation_error_rad=rot_error)
        ck("candidate_G_roundtrip",float(np.max(abs(tback-bridge[13:16])))<=1e-9,max_abs_error_m=float(np.max(abs(tback-bridge[13:16]))))
        initial_z=parent["scene_setup_only"]["position_world_m"][2]
        offset=float(G[2,3]-initial_z)
        ck("tip_initial_relative_height",abs(offset-.000243236977755322)<=1e-12,actual_m=offset)
        # Pure corner projection of all five frozen component boxes; no margin claim.
        mins={}
        for name,c in model["components"].items():
            center=np.array(c["com_at_zero_in_G_m"])
            if name!="ip140_palm":center[1]+=(-1 if "left" in name else 1)*bridge[1]/2
            half=np.array(c["box_size_m"])/2
            corners=np.array([center+half*np.array(sign) for sign in itertools.product((-1,1),repeat=3)])
            world=corners@R.T+G[:3,3]
            mins[name]=float(world[:,2].min()-cfg["table_z"])
        clearance=min(mins.values())
        ck("nominal_component_table_clearance",abs(clearance-.031034237880408058)<=1e-12,actual_m=clearance)
        child=copy.deepcopy(parent)
        child.update(slot=14,candidate_id=ID,parent_id=parent["candidate_id"],parent_slot=7,
            kind="nominal",bridge_kind="legacy_height_bridge",development_corrective_bridge=True,
            development_corrective_repeat=True,repeat_of_slot=7,independent_confirmation=False,
            execution_order=9,candidate_world=bridge.tolist(),T_W_G_m=G.tolist(),source_parent_version=OLD.name,
            legacy_height_bridge=dict(rule="subtract only source-bound legacy height correction from world t_z",
                delta_h_legacy_m=DELTA_H,old_t_z_m=float(frozen[15]),new_t_z_m=float(bridge[15]),
                old_candidate_id=parent["candidate_id"],new_candidate_id=ID,
                source=intent_path.with_name("result.json").relative_to(ROOT).as_posix(),
                source_sha256=sha(intent_path.with_name("result.json")),two_path_max_abs_error=error),
            accept="development bridge only: full engineering gates and final480 bilateral target contact with height>=0.10m, aggregate table support false, no forbidden collision/slip")
        updated=copy.deepcopy(old);updated["slots"].append(child)
        updated.update(version=HERE.name,frozen_at_utc=now(),max_starts=14,prior_physical_starts=8,
            eligible_slots_after_zero_gate=[14],prior_revision=OLD.name,
            revision_scope="slot14 parent7: undo exactly the source-recorded legacy t_z correction; evaluate parallel aggregate table support; all other physical parameters unchanged",
            start_requires="all offline and independent full planned-path zero-step preflight checks pass; formal dynamic connection freshly loaded; charge immediately before first physics call",
            stop="first gate failure or completion of slot14, no retry and no slots8-12",
            correction="smoke12 slot7 charged failure remains immutable; only authorized physical variable is removing legacy height correction",
            budget_change=dict(old_total=13,new_total=14,old_starts=8,new_slot=14,remaining_before_start=6,
                cumulative_after_first_step=9,remaining_after_first_step=5,no_retries=True,preserved_reserved_slots=[8,9,10,11,12]),
            aggregate_support=dict(version="aggregate_support01",legacy_field_retained=True,
                source_field="table_support",normal_force_rule="sum(max(Fn_i,0)) over identity-valid finite legal object-table contacts, strictly >0.2N",
                contact_force_threshold_N=.2,success_requires_aggregate_table_support=False,invalid_contacts_are_independent_hard_failures=True),
            authorized_slot14_only=True,slots8_to12_frozen=True)
        ck("historical_slots_verbatim_preserved",updated["slots"][:13]==old["slots"])
        # This is the first construction of this independent registration; history remains in OLD.
        (HERE/"preregistration.json").write_text(json.dumps(updated,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        import trajectory
        plan=trajectory.build()
        ck("trajectory_new_identity",plan["candidate_id"]==ID and plan["slot"]==14)
        oldplan=read(OLD/"trajectory_plan.json")
        ck("fresh_plan_joint_knots_changed",plan["waypoints"]!=oldplan["waypoints"] and plan["W_T_G_goal"]!=oldplan["W_T_G_goal"])
        ck("trajectory_structure_preserved",plan["phase_steps"]==oldplan["phase_steps"] and plan["home_joints"]==oldplan["home_joints"] and plan["pre_open_m"]==oldplan["pre_open_m"] and plan["limits_rad_rad_s"]==oldplan["limits_rad_rad_s"])
        limits=np.array(plan["limits_rad_rad_s"]);states=[];max_qdot=np.zeros(6);max_step=np.zeros(6);prior_q=None
        for phase,count in trajectory.PHASES:
            for tick in range(1,count+1):
                q,dq,finger=trajectory.reference(plan,phase,tick)
                if not (np.isfinite(q).all() and np.isfinite(dq).all() and np.isfinite(finger)):raise ValueError("nonfinite plan reference")
                if np.any(q<limits[:,0]) or np.any(q>limits[:,1]) or np.any(abs(dq)>limits[:,2]):raise ValueError("joint/velocity plan limit")
                if not 0<=finger<=.07:raise ValueError("finger command outside 0..70mm")
                max_qdot=np.maximum(max_qdot,np.abs(dq))
                if prior_q is not None:max_step=np.maximum(max_step,abs(q-prior_q))
                prior_q=q;states.append([phase,tick,*q.tolist(),*dq.tolist(),float(finger)])
        ck("complete_4728_plan_reference_count",len(states)==4728,actual=len(states))
        fk_errors=[]
        for row in plan["waypoints"]+[dict(target_q_rad=plan["lift"]["target_q_rad"],target_W_T_G=plan["lift"]["target_W_T_G"],phase="lift")]:
            actual=trajectory.fk(row["target_q_rad"]);target=np.array(row["target_W_T_G"])
            pos=float(np.linalg.norm(actual[:3,3]-target[:3,3]));rot=float(np.linalg.norm(trajectory.rotation_vector(actual[:3,:3]@target[:3,:3].T)))
            fk_errors.append(dict(phase=row["phase"],translation_error_m=pos,rotation_error_rad=rot))
            ck("IK_target_roundtrip:"+row["phase"],pos<=1e-9 and rot<=1e-9,translation_error_m=pos,rotation_error_rad=rot)
        dump(HERE/"trajectory_plan.json",plan)
        proof=dict(version=HERE.name,created_utc=now(),passed=True,all_checks_passed=True,
           mode="pure offline preparation; no engine connection or physics",
           candidate_id=ID,parent_id=parent["candidate_id"],parent_slot=7,slot=14,
           old_candidate=frozen.tolist(),new_candidate=bridge.tolist(),npz_reconstructed_candidate=observed.tolist(),
           two_path_max_abs_error=error,changed_indices=changed,element_comparison=element_checks,
           delta_h_legacy_m=DELTA_H,sources=sources,package_checks=package_checks,checks=checks,
           pure_function_definition_lines=[dict(name=n.name,first_line=n.lineno,last_line=n.end_lineno) for n in nodes],
           observation=obs,old_G=parent["T_W_G_m"],new_G=G.tolist(),new_flange_pose=F.tolist(),
           roundtrip=dict(translation_error_m=translation_error,rotation_error_rad=rot_error),
           tip_relative_initial_object_center_m=offset,
           static_component_clearance_m=mins,minimum_static_component_clearance_m=clearance,
           static_clearance_scope="nominal IP140 boxes only at fixed target, without collision-margin or entire-path claim",
           plan_validation=dict(reference_count=len(states),all_references_finite=True,all_joint_position_velocity_limits=True,
              finger_command_range_m=[0.,float(bridge[1]/2)],max_abs_qdot_rad_s=max_qdot.tolist(),max_adjacent_command_delta_rad=max_step.tolist(),
              IK_target_errors=fk_errors,plan_sha256=sha(HERE/"trajectory_plan.json"),plan_regenerated_not_reused=True,
              references_json_sha256=hashlib.sha256(json.dumps(states,separators=(",",":"),allow_nan=False).encode()).hexdigest()),
           historical_parent_unchanged_sha256=sha(OLD/"preregistration.json"),new_preregistration_sha256=sha(HERE/"preregistration.json"),
           physical_steps=0,charges=0)
        dump(HERE/"candidate_difference_proof.json",proof)
        print(json.dumps(dict(passed=True,checks=len(checks),candidate_id=ID,new_G_z=float(G[2,3]),max_diff=error,plan_steps=len(states)),ensure_ascii=False))
    except BaseException as e:
        failure=dict(version=HERE.name,passed=False,error=repr(e),traceback=traceback.format_exc(),checks=checks,created_utc=now(),physics_steps=0,charges=0)
        if not (HERE/"candidate_difference_proof.json").exists():dump(HERE/"candidate_difference_proof.json",failure)
        raise
if __name__=="__main__":main()

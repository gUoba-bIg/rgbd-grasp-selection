"""Zero-step dual-channel static review. No physical execution or target motion."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import gzip,json,math,traceback
import numpy as np
from batch_context import SLOT, SLOT_DIR, OBJECT_ID
from trajectory import HERE,ROOT,PLAN,PHASES,DT,read,reference,fk,sha
from seam_geometry import PALM,PADS,BONES,current_frames,rigid,inverse,local_xyz
from structural_filter import audit_filtered,audit_seams
from base_table_mount import capture as capture_mount
from contact_entry import find_contact_entry,model_face_spec,_safe
from contact_semantics import audit_pad_witness,classify_contact,validate_entry_certificate,raw_contact_finite

QUERY_DISTANCE_M=.005
def registered_pad_index(row, robot, target, pads):
    if raw_contact_finite(row):
        if row[1] == robot and row[2] == target and row[4] == -1 and row[3] in pads:
            return row[3]
        if row[2] == robot and row[1] == target and row[3] == -1 and row[4] in pads:
            return row[4]
    return None


def fresh_target_rows(queries):
    return [*queries["robot_target"], *queries.get("robot_target_supplemental", ())]


def query_missing_cached_pairs(p, robot, target, pads, manifolds, queries, kw):
    """The 5 mm bulk survey need not cover every cached point's current pair."""
    cached = {registered_pad_index(r, robot, target, pads) for r in manifolds}
    present = {registered_pad_index(r, robot, target, pads) for r in queries["robot_target"]}
    missing = sorted(cached - present - {None})
    supplements, details, violations = [], [], []
    for pad in missing:
        seq_first = p.sequence + 1
        pad_box = p.getAABB(robot, pad, **kw)
        target_box = p.getAABB(target, -1, **kw)
        lo = np.minimum(pad_box[0], target_box[0])
        hi = np.maximum(pad_box[1], target_box[1])
        if not np.isfinite([pad_box, target_box]).all() or np.any(hi < lo):
            raise ValueError("invalid_aabb_for_same_state_fresh_pair")
        # The diagonal of the enclosing world AABB bounds distance between any
        # points of the two shapes. This only chooses a query radius.
        radius = math.nextafter(float(np.linalg.norm(hi - lo)), math.inf)
        rows = p.getClosestPoints(robot, target, radius, linkIndexA=pad, linkIndexB=-1, **kw)
        if any(registered_pad_index(row, robot, target, pads) != pad for row in rows):
            violations.append("specific_fresh_query_returned_wrong_pair_or_nonfinite")
        details.append(dict(pad=pad, robot=robot, target=target, target_link=-1,
            radius_m=radius, radius_basis="diagonal of the union AABB rounded upward",
            pad_aabb=pad_box, target_aabb=target_box, result_start=len(supplements),
            returned_count=len(rows), api_sequence_first=seq_first, api_sequence_last=p.sequence))
        supplements.extend(rows)
    if details:
        queries["robot_target_supplemental"] = supplements
    return details, violations


def write_once(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(_safe(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def now():return datetime.now(timezone.utc).isoformat()
def geometry_only(record):
    """Discard no observation: solver manifolds were deliberately not requested."""
    def strip(value):
        if isinstance(value,dict):
            return {k:(None if "manifold" in k.lower() else strip(v)) for k,v in value.items()}
        if isinstance(value,list):return [strip(x) for x in value]
        return value
    return dict(geometry_passed=not record["violations"],solver_manifolds_observed=False,
        solver_manifold_passed=None,violations=record["violations"],geometry_evidence=strip(record),
        scope="Explicit closest/kinematics only; native manifold response is checked in mandatory proxy channel")


def scan(p,client,robot,links,manifest,registration,gate,*,output_root=None):
    output_root=SLOT_DIR/"static_preflight" if output_root is None else Path(output_root).resolve()
    if output_root!=SLOT_DIR/"static_preflight":
        raise ValueError("unregistered_static_output_directory")
    output_root.mkdir(parents=True,exist_ok=True)
    for name in ("dual_channel_snapshots.jsonl.gz","left_right_contact_entry_roots.json","path_result.json","contact_entries"):
        if (output_root/name).exists():raise FileExistsError("Static output exists; no retry or overwrite: "+name)
    ledger=read(HERE/"slot_ledger.json")
    result=dict(version=HERE.name,slot=SLOT,object_id=OBJECT_ID,candidate_id=registration["slots"][SLOT-1]["candidate_id"],started_at=now(),passed=False,status="not_started",
        physical_steps=0,charged=False,new_charges=0,current_slot_started=False,
        cumulative_starts=ledger["starts"],total_budget=14,remaining=ledger["remaining_total_budget"],
        planned_snapshots=4728,snapshots_completed=0,first_failure=None,
        episode_completed=None,engineering_gate_passed=None,grasp_success=None,slot_passed=None,
        target_pose_modified=False,proxy_used_by_controller=False,
        reference_state_is_dynamic_prediction=False,proxy_is_dynamic_prediction=False,
        reference_source_sha256={"trajectory.py":sha(HERE/"trajectory.py"),"trajectory_plan.json":sha(PLAN)},
        witness_policy={
            "fresh":"all rows need finite identities, unit normal and full 3-D residual <= 1e-9 m",
            "cached":"registered face, unit normal, signed normal projection residual <= 1e-9 m; tangential/full errors diagnostic",
            "face_scope":"cache, same-state matching fresh minimum, entry and proxy near contact require the registered inner face; clearance survey and certified q_ref overtravel retain measured face diagnostics",
            "basis":"smoke15 contact_aware_path.classify recorded clearance/overtravel face geometry without a physical contact claim; root and near-contact proxy are effective-face witnesses",
            "decision":"fresh closest alone controls geometric contact and consecutive sign bracket; cached distance never triggers entry"},
        checked_phase_counts={},violations=[],last_unchecked_snapshots=4728)
    result["witness_policy"].update(
        static_postentry_scope="During lift and hold only, registered robot/target geometry in a frozen-target scene is diagnostic. All returned observations retain identity, finiteness and source-specific witness checks.",
        face_scope="Before lift the smoke16 face scope and 1 nm inner plane remain in force. The opening, approach, close-entry and proxy gates are unchanged. From lift onward record local points and physical pad-surface tags.",
        cached="Same-state pair query, unit normal, signed normal projection residual <= 1e-9 m. The registered face is mandatory through close; lift/hold surface tags are diagnostic.")
    roots={};certs={};prior_pad={};previous_proxy=None;previous_ref=None
    record_path=output_root/"dual_channel_snapshots.jsonl.gz"
    per_phase=Counter();classes=Counter();n=0;phase="setup";tick=0;stream=None
    try:
        if not gate["passed"] or gate["dynamic_execution_authorized"] or gate["eligible_slots_after_zero_gate"]!=[]:
            raise ValueError("static_only_zero_gate_required")
        if p.step_authorized or p.counts["stepSimulation"] or p.counts["setCollisionFilterPair"]!=14:
            raise ValueError("zero_physics_exact_filters_required")
        if p.connection_scope!="contact_aware_static":raise ValueError("wrong_connection_scope")
        plan=read(PLAN); target=gate["object_body_id"]; table=gate["table_body_id"];kw={"physicsClientId":client}
        infos=[p.getJointInfo(robot,j,**kw) for j in range(p.getNumJoints(robot,**kw))]
        byjoint={v[1].decode():v[0] for v in infos}
        names=read(ROOT/"data/simulation_v52/robot/robot_assets.json")["arm_joint_names"]
        arm=[byjoint[k] for k in names];fingers=[links[k] for k in BONES];indices=arm+fingers
        limits=np.asarray(plan["limits_rad_rad_s"])
        original_object=p.getBasePositionAndOrientation(target,**kw);original_table=p.getBasePositionAndOrientation(table,**kw)
        result.update(body_robot=robot,body_object=target,body_table=table,client=client,
            joint_indices=indices,initial_target_pose=original_object,initial_table_pose=original_table,
            face_spec=model_face_spec())
        def capture(qarm,qf,channel,phase):
            seq0=p.sequence+1
            values=[*qarm,qf["left"],qf["right"]]
            if not all(math.isfinite(float(x)) for x in values):raise ValueError("nonfinite_reference")
            if np.any(qarm<limits[:,0]) or np.any(qarm>limits[:,1]) or any(not 0<=qf[s]<=.070 for s in ("left","right")):
                raise ValueError("reference_or_proxy_joint_limit")
            for j,v in zip(indices,values):p.resetJointState(robot,j,targetValue=float(v),targetVelocity=0.,**kw)
            fresh_reference_only=channel=="reference" and bool(certs)
            if not fresh_reference_only:p.performCollisionDetection(**kw)
            states=p.getJointStates(robot,indices,**kw);frames=current_frames(p,robot,links,client)
            raw=None if fresh_reference_only else p.getContactPoints(**kw)
            geometry_input=[] if raw is None else raw
            queries={"robot_self":p.getClosestPoints(robot,robot,QUERY_DISTANCE_M,**kw),
                "robot_target":p.getClosestPoints(robot,target,QUERY_DISTANCE_M,**kw),
                "robot_table":p.getClosestPoints(robot,table,QUERY_DISTANCE_M,**kw),
                "target_table":p.getClosestPoints(target,table,QUERY_DISTANCE_M,**kw)}
            pad_indices = tuple(links[name] for name in PADS)
            supplements, supplemental_errors = query_missing_cached_pairs(
                p, robot, target, pad_indices, geometry_input, queries, kw)
            if phase in ("lift", "hold"):
                # Acquire every required link frame at the same reference/proxy
                # pose as its witness, before any bisection or further resets.
                by_index = {value:name for name,value in links.items()}
                rows = [*geometry_input,*fresh_target_rows(queries)]
                for row in rows:
                    if not raw_contact_finite(row) or set(row[1:3]) != {robot,target}:
                        continue
                    idx = row[3] if row[1] == robot else row[4]
                    if idx not in by_index and idx != -1:
                        continue  # classify_contact rejects unregistered indices.
                    name = by_index.get(idx, "__base__")
                    if name in frames:
                        continue
                    if idx == -1:
                        w_com = p.getBasePositionAndOrientation(robot, **kw)
                        dynamics = p.getDynamicsInfo(robot, -1, **kw)
                        frames[name] = rigid(*w_com) @ inverse(rigid(dynamics[3], dynamics[4]))
                    else:
                        st = p.getLinkState(robot,idx,computeForwardKinematics=1,**kw)
                        frames[name] = rigid(st[0],st[1]) @ inverse(rigid(st[2],st[3]))
            structural=audit_filtered(p,robot,links,client,geometry_input,current_frames_w=frames,
                q_by_side={s:v[0] for s,v in zip(("left","right"),states[-2:])})
            active=audit_seams(p,robot,links,client,geometry_input,
                require_pad_witness=qf["left"]+qf["right"]<=.002,manifold_frames=frames,current_frames_w=frames,dt=DT)
            mount=capture_mount(p,client,robot,table,links,"dynamic",raw_manifolds=geometry_input)
            problems=[*supplemental_errors,*structural["violations"],*active["violations"],*mount["violations"]]
            expected=fk(qarm)
            if not np.isfinite(np.asarray(states,dtype=object)[...,0].astype(float)).all():
                problems.append("nonfinite_reset_q")
            for st,v in zip(states,values):
                if not all(math.isfinite(x) for x in (st[0],st[1],*st[2],st[3])):problems.append("nonfinite_joint_readback")
                elif abs(st[0]-v)>1e-12 or st[1]!=0:problems.append("reset_readback_mismatch")
            if not all(np.isfinite(t).all() for t in frames.values()):problems.append("nonfinite_link_frames")
            terr=float(np.linalg.norm(frames[PALM][:3,3]-expected[:3,3]))
            rerr=float(np.max(np.abs(frames[PALM][:3,:3]-expected[:3,:3])))
            if terr>1e-9 or rerr>1e-9:problems.append("FK_mismatch")
            obj=p.getBasePositionAndOrientation(target,**kw);tbl=p.getBasePositionAndOrientation(table,**kw)
            if obj!=original_object or tbl!=original_table:problems.append("static_world_pose_changed")
            if p.counts["stepSimulation"] or p.step_authorized:problems.append("physical_steps_forbidden")
            if p.counts["setCollisionFilterPair"]!=14:problems.append("filter_count_changed")
            return dict(channel=channel,q_finger_m=qf,actual_joint_states=states,
                W_T_G=frames[PALM].tolist(),W_T_links={k:v.tolist() for k,v in frames.items()},
                FK_translation_error_m=terr,FK_rotation_matrix_error=rerr,
                raw_manifolds=raw,raw_manifolds_requested=not fresh_reference_only,
                acquisition_mode="fresh_closest_without_persistent_detection" if fresh_reference_only else "full_collision_detection_and_manifolds",
                raw_closest=queries,fresh_pair_supplements=supplements,structural_audit=geometry_only(structural) if fresh_reference_only else structural,
                pad_pair_audit=geometry_only(active) if fresh_reference_only else active,
                base_table_mount_audit=geometry_only(mount) if fresh_reference_only else mount,
                object_pose=obj,table_pose=tbl,api_sequence_first=seq0,api_sequence_last=p.sequence,
                violations=problems,static_reset_state_not_a_dynamic_return=True)
        def classify(channel_data,phase):
            problems=channel_data["violations"];items=[];faces=[]
            pads = tuple(links[name] for name in PADS)
            cached_pads = {registered_pad_index(r, robot, target, pads) for r in channel_data["raw_manifolds"] or ()}
            fresh_rows = fresh_target_rows(channel_data["raw_closest"])
            # Each cached pair designates the current minimum as its same-state fresh witness.
            # Other fresh clearance/command rows still get the full 3-D audit.
            selected = {}
            for fresh in fresh_rows:
                pad_idx = registered_pad_index(fresh, robot, target, pads)
                if pad_idx is not None and pad_idx in cached_pads:
                    if pad_idx not in selected or fresh[8] < selected[pad_idx][8]:
                        selected[pad_idx] = fresh
            streams=list(channel_data["raw_closest"].items())
            if channel_data["raw_manifolds_requested"]: streams.insert(0,("manifold",channel_data["raw_manifolds"]))
            for source,contacts in streams:
                for i,row in enumerate(contacts):
                    c=classify_contact(row,source=source,phase=phase,channel=channel_data["channel"],
                        robot=robot,target=target,table=table,links=links,entry_by_pad=certs,
                        fresh_closest=fresh_rows)
                    if c.get("frozen_target_geometry_diagnostic"):
                        rlink = c["robot_link"]
                        name = next((name for name, index in links.items() if index == rlink), "__base__")
                        pad_first = row[1] == robot
                        point = row[5] if pad_first else row[6]
                        transform = np.asarray(channel_data["W_T_links"][name])
                        c.update(robot_link_name=name, robot_witness_link_local_m=local_xyz(transform,point).tolist(),
                                 target_witness_local_COM_m=local_xyz(rigid(*channel_data["object_pose"]),
                                     row[6 if pad_first else 5]).tolist())
                        if c["contact_part"] == "nonpad":
                            c.update(surface_label="nonpad_target_geometry", tip_depth_m=None)
                    c.update(query=source,row_index=i);items.append(c);classes[c["classification"]]+=1
                    problems.extend(f"{source}:{i}:{reason}" for reason in c["violations"])
                    if source.startswith("robot_target") and (not raw_contact_finite(row) or
                            {row[1],row[2]} != {robot, target}):
                        problems.append(f"{source}:{i}:wrong_body_identity_or_invalid_fresh_target_query")
                    idx = registered_pad_index(row, robot, target, pads)
                    if idx is None:
                        continue
                    si = pads.index(idx)
                    side = ("left", "right")[si]
                    if phase in ("lift", "hold"):
                        face_reason = "postentry_frozen_target_surface_diagnostic"
                    elif source == "manifold":
                        face_reason = "persistent_pad_witness"
                    elif selected.get(idx) is row:
                        face_reason = "same_state_fresh_minimum_for_cached_pair"
                    elif channel_data["channel"] == "proxy" and row[8] <= 1e-9:
                        face_reason = "proxy_near_or_in_contact"
                    elif c["classification"] == "post_contact_motor_reference_overtravel":
                        face_reason = "certified_motor_reference_overtravel_diagnostic"
                    elif row[8] <= 0:
                        face_reason = "geometric_contact_witness"
                    else:
                        face_reason = "positive_clearance_survey"
                    require_face = face_reason not in {"positive_clearance_survey",
                        "certified_motor_reference_overtravel_diagnostic", "postentry_frozen_target_surface_diagnostic"}
                    audit = audit_pad_witness(row, source=source, robot=robot, target=target, pad_index=idx,
                        W_T_pad=channel_data["W_T_links"][PADS[si]], W_T_G=channel_data["W_T_G"],
                        side=side, require_face=require_face, phase=phase)
                    audit["face_gate_reason"] = face_reason
                    if "surface" in audit:
                        c.update(surface_label=audit["surface"]["label"],tip_depth_m=audit["tip_depth_m"])
                    faces.append(dict(query=source,row_index=i,side=side,audit=audit))
                    problems.extend(f"{source}:{i}:witness:{x}" for x in audit["violations"])
            channel_data["classifications"]=items;channel_data["pad_witness_audits"]=faces
            channel_data["violations"]=list(dict.fromkeys(problems));channel_data["passed"]=not channel_data["violations"]
            return channel_data["passed"]
        stream=gzip.open(record_path,"xt",encoding="utf-8",newline="\n")
        for phase,count in PHASES:
            for tick in range(1,count+1):
                n+=1;qarm,dq,qref=reference(plan,phase,tick)
                if phase == "lift" and tick == 1 and (set(roots) != {"left", "right"}
                        or not all(root["passed"] for root in roots.values())):
                    raise ValueError("both_registered_entries_required_before_lift")
                if not np.isfinite(dq).all() or np.any(np.abs(dq)>limits[:,2]):raise ValueError("arm_reference_velocity")
                reference_data=capture(qarm,dict(left=qref,right=qref),"reference",phase)
                current={}
                for side,pad in zip(("left","right"),PADS):
                    subset=[x for x in fresh_target_rows(reference_data["raw_closest"])
                        if registered_pad_index(x, robot, target, (links[pad],)) == links[pad]]
                    current[side]=min(subset,key=lambda c:c[8]) if subset else None
                root_failure=[]
                for side,pad in zip(("left","right"),PADS):
                    c=current[side]
                    if side in roots or c is None or c[8]>0 or phase!="close":continue
                    previous=prior_pad.get(side)
                    if (previous is None or previous["snapshot"]!=n-1 or previous["distance_m"]<=0
                        or reference_data["violations"]):
                        root_failure.append(side+":no_clean_positive_to_nonpositive_entry")
                        continue
                    other="right" if side=="left" else "left"
                    other_q=max(qref,roots[other]["q_entry_m"]) if other in roots and roots[other]["passed"] else qref
                    context=dict(phase="close",previous_snapshot=n-1,current_snapshot=n,
                        previous_signed_distance_m=previous["distance_m"],current_signed_distance_m=c[8],
                        precontact_clean=True)
                    root=find_contact_entry(p,client,robot,target,links,side=side,
                        arm_joint_indices=arm,arm_q=qarm.tolist(),q_clear_m=previous["q_ref_m"],
                        q_overlap_m=qref,other_finger_q_m=other_q,
                        output_dir=output_root/"contact_entries"/side,source_snapshot=context)
                    roots[side]=root
                    errs=validate_entry_certificate(root,robot=robot,target=target,pad=links[pad])
                    if not root["passed"] or errs:root_failure.extend(side+":"+v for v in (root["violations"]+errs))
                    else:certs[links[pad]]=root
                reference_data["violations"].extend(root_failure)
                classify(reference_data,phase)
                qproxy={s:max(qref,roots[s]["q_entry_m"]) if s in roots and roots[s]["passed"] else qref for s in ("left","right")}
                proxy_data=capture(qarm,qproxy,"proxy",phase)
                proxy_speed={s:0. if previous_proxy is None else (qproxy[s]-previous_proxy[s])/DT for s in ("left","right")}
                if previous_proxy is not None and any(abs(qproxy[side]-previous_proxy[side])>abs(qref-previous_ref) for side in ("left","right")):
                    proxy_data["violations"].append("proxy_step_displacement_exceeds_reference_step")
                # Derived differences are preserved raw. Speed admissibility follows
                # the unchanged analytic reference and max-with-constant's 1-Lipschitz bound,
                # not a new tolerance on measured dynamic velocity.
                classify(proxy_data,phase)
                issues=[*["reference:"+x for x in reference_data["violations"]],
                        *["proxy:"+x for x in proxy_data["violations"]]]
                record=dict(snapshot=n,phase=phase,phase_tick=tick,q_ref_m=qref,q_proxy_m=qproxy,
                    arm_reference_q_rad=qarm.tolist(),arm_reference_velocity_rad_s=dq.tolist(),
                    proxy_derived_position_slope_m_s=proxy_speed,
                    reference_channel=reference_data,proxy_channel=proxy_data,
                    entry_side_available={s:s in roots and roots[s]["passed"] for s in ("left","right")},
                    physical_steps=0,charged=False,passed=not issues,violations=issues)
                stream.write(json.dumps(_safe(record),ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n");stream.flush()
                result["snapshots_completed"]=n;per_phase[phase]+=1
                if issues:
                    result.update(status="static_contact_aware_admission_blocked",violations=issues,
                        first_failure=dict(snapshot=n,phase=phase,phase_tick=tick,violations=issues))
                    return result
                previous_proxy=qproxy.copy();previous_ref=qref
                prior_pad={s:dict(snapshot=n,q_ref_m=qref,distance_m=row[8]) for s,row in current.items() if row is not None}
        if set(roots)!={"left","right"} or not all(v["passed"] for v in roots.values()):
            raise ValueError("both_contact_entries_not_found_within_registered_range")
        result.update(passed=True,status="target-independent path passed; loaded target contact dynamics deferred")
    except BaseException as exc:
        result.update(status="static_contact_aware_admission_exception",exception=repr(exc),traceback=traceback.format_exc(),
            first_failure=dict(snapshot=n,phase=phase,phase_tick=tick,error=repr(exc)))
    finally:
        if stream is not None:stream.close()
        result.update(ended_at=now(),checked_phase_counts=dict(per_phase),last_unchecked_snapshots=4728-result["snapshots_completed"],
            classification_counts=dict(classes),physics_step_calls=p.counts["stepSimulation"],filter_call_count=p.counts["setCollisionFilterPair"])
        write_once(output_root/"left_right_contact_entry_roots.json",dict(version=HERE.name,sides=roots,
            both_found=set(roots)=={"left","right"},both_passed=set(roots)=={"left","right"} and all(v["passed"] for v in roots.values()),
            physics_steps=0,object_pose_modified=False,proxy_used_by_controller=False))
        if record_path.exists():result["dual_channel_sha256"]=sha(record_path)
        write_once(output_root/"path_result.json",result)
    return result

"""Pure static reference/proxy semantics; never a loaded-state or dynamic gate.

The frozen dynamic evaluator remains separate and unchanged. A motor reference
inside the object after a certified contact entry is not a physical state.
"""
from math import isfinite
import numpy as np
from pad_surface import classify_pad_surface
from contact_entry import validate_contact_row
from structural_filter import STRUCTURAL_PAIRS

SCHEMA="IP140-smoke20-static-positive-clearance-face-scope-v1"
FORCE_NUMERIC_ZERO_N=1e-9
STATIC_NUMERIC_ZERO_M=1e-9
DYNAMIC_PENETRATION_LIMIT_M=.001
REFERENCE_OVERTRAVEL_PHASES=("close","lift","hold")


def finite(values):
    return all(isinstance(v,(int,float)) and not isinstance(v,bool) and isfinite(v) for v in values)


def raw_contact_finite(row):
    if not isinstance(row,(list,tuple)) or len(row)!=14: return False
    if any(not isinstance(row[k],(list,tuple)) or len(row[k])!=3 for k in (5,6,7,11,13)): return False
    return finite((*row[:5],*row[5],*row[6],*row[7],*row[8:11],*row[11],row[12],*row[13]))


def validate_entry_certificate(certificate, *, robot,target,pad):
    """Require the caller's actual clean approach and per-side positive-to-contact bracket.

    A bare bool is never evidence. Detailed entry-face and root-solve checks are
    part of entry_audit. The caller freezes its proxy independently of control.
    """
    problems=[]
    if not isinstance(certificate,dict): return ["missing_per_side_entry_certificate"]
    c=certificate
    if c.get("passed") is not True: problems.append("entry_solver_not_passed")
    if (c.get("robot_body_id"),c.get("target_body_id"),c.get("pad_link_index"))!=(robot,target,pad):
        problems.append("entry_identity_mismatch")
    if c.get("phase")!="close" or c.get("precontact_clean") is not True:
        problems.append("entry_phase_or_clean_precontact_unproven")
    previous,current=c.get("previous_snapshot"),c.get("current_snapshot")
    if not isinstance(previous,int) or not isinstance(current,int) or previous<1 or current!=previous+1:
        problems.append("entry_snapshot_bracket_invalid")
    dp,dc=c.get("previous_signed_distance_m"),c.get("current_signed_distance_m")
    if not finite((dp,dc)) or not dp>0 or not dc<=0:
        problems.append("entry_positive_to_contact_transition_unproven")
    q=c.get("q_entry_m")
    if not finite((q,)) or not 0<=q<=.070: problems.append("entry_q_nonfinite_or_outside_travel")
    entry=c.get("entry_audit")
    if not isinstance(entry,dict) or entry.get("passed") is not True:
        problems.append("entry_witness_face_audit_missing_or_failed")
    if isinstance(entry,dict):
        distance=entry.get('signed_distance_m')
        if not finite((distance,)) or abs(distance)>1e-9:
            problems.append('entry_signed_distance_residual')
        for field in ('face_outside_m','full_witness_residual_m','normal_unit_error'):
            value=entry.get(field)
            if not finite((value,)) or not 0<=value<=1e-9:problems.append('entry_'+field)
        if entry.get('on_effective_inner_face') is not True:
            problems.append('entry_not_on_effective_inner_face')
        alignment=entry.get('inward_normal_alignment')
        if not finite((alignment,)) or not alignment>0:problems.append('entry_inward_alignment')
    bracket=c.get('bracket')
    if not isinstance(bracket,dict):problems.append('entry_root_bracket_missing')
    else:
        qc,qo,dc,do,width=(bracket.get(k) for k in ('q_clear_m','q_overlap_m','d_clear_m','d_overlap_m','width_m'))
        if not finite((qc,qo,dc,do,width)):
            problems.append('entry_root_bracket_nonfinite')
        else:
            if not 0<=qo<qc<=.070 or not 0<width<=1e-10 or width!=qc-qo:
                problems.append('entry_root_bracket_range_width')
            if not dc>0 or not do<=0:problems.append('entry_root_distance_signs')
            if q!=qc:problems.append('entry_root_q_differs_from_clear_endpoint')
            if isinstance(entry,dict) and finite((entry.get('signed_distance_m'),)):
                if abs(entry['signed_distance_m']-dc)>1e-9:problems.append('entry_final_query_distance_inconsistent')
    if c.get("nonpad_target_collisions_at_entry")!=[]:
        problems.append("entry_nonpad_target_clearance_unproven")
    if c.get('not_engine_entry_certificate') is True:
        problems.append('historical_offline_proof_is_not_current_entry_certificate')
    return problems


def audit_pad_witness(row, *, source, robot, target, pad_index, W_T_pad, W_T_G, side,
                      require_face=True, phase="close", pair_state=None):
    """A solver manifold retains a normal projection, a fresh closest pair a 3D witness."""
    cached = source == "manifold"
    diagnostic = phase in ("lift", "hold")
    require_face = require_face and not diagnostic and not (pair_state and pair_state["positive_clearance"])
    audit = validate_contact_row(row, robot=robot, target=target, pad_index=pad_index,
        W_T_pad=W_T_pad, W_T_G=W_T_G, side=side, require_face=require_face and not cached)
    errors = list(audit["violations"])
    audit.update(role="persistent_manifold" if cached else "fresh_closest",
        require_face=require_face, full_residual_is_hard_gate=not cached)
    if cached:
        errors = [error for error in errors if error != "full_witness_residual"]
    if raw_contact_finite(row) and "full_witness_residual_m" in audit:
        delta = np.asarray(row[5]) - np.asarray(row[6])
        normal = np.asarray(row[7])
        projection = float(delta @ normal)
        signed_error = projection - row[8]
        tangent = delta - projection * normal
        full = delta - row[8] * normal
        audit.update(normal_projection_residual_m=abs(signed_error),
            signed_normal_projection_error_m=signed_error, tangential_residual_m=float(np.linalg.norm(tangent)),
            tangential_residual_vector_m=tangent.tolist(), full_witness_residual_vector_m=full.tolist())
        if not finite((audit["full_witness_residual_m"], audit["normal_projection_residual_m"],
                       audit["tangential_residual_m"], audit["normal_unit_error"],audit["face_outside_m"])):
            errors.append("nonfinite_derived_witness")
        if cached and abs(signed_error) > STATIC_NUMERIC_ZERO_M:
            errors.append("normal_projection_residual")
        audit["surface"] = classify_pad_surface(side, audit["pad_witness_pad_link_m"],
                                               audit["normal_pad_to_object_pad"])
        audit["frozen_target_surface_diagnostic"] = diagnostic
        if cached and require_face and audit["on_effective_inner_face"] is not True:
            errors.append("entry_witness_outside_registered_effective_face")
        if pair_state is not None:
            errors.extend(pair_state["violations"])
    elif not raw_contact_finite(row):
        errors.append("nonfinite_or_malformed_contact")
    audit["violations"] = list(dict.fromkeys(errors))
    audit["passed"] = not audit["violations"]
    return audit


def pad_pair_state(fresh_rows, cached_rows, *, robot, target, pad_index):
    """Reduce only one captured channel at one snapshot, before any further resets.

    Indices refer to the caller's unmodified query streams. A bounded fresh survey
    can return no point; only a cached pair *without* a matching query is invalid.
    """
    pair = {(robot, pad_index), (target, -1)}
    errors = []
    fresh_indices = []
    cached_indices = []
    force_active = False
    for stream, indices, rows in (("fresh", fresh_indices, fresh_rows), ("cached", cached_indices, cached_rows)):
        for i, row in enumerate(rows):
            if not isinstance(row, (tuple, list)) or len(row) != 14:
                errors.append(stream + "_malformed_contact")
                continue
            endpoints = (row[1], row[3], row[2], row[4])
            if not finite(endpoints) or any(int(k) != k for k in endpoints):
                errors.append(stream + "_invalid_body_link_identity")
                continue
            if {(row[1], row[3]), (row[2], row[4])} != pair:
                continue
            if not raw_contact_finite(row):
                errors.append(stream + "_nonfinite_or_incomplete")
                continue
            indices.append(i)
            if any(abs(row[k]) > FORCE_NUMERIC_ZERO_N for k in (9, 10, 12)):
                force_active = True

    selected = min(fresh_indices, key=lambda i: fresh_rows[i][8]) if fresh_indices else None
    d = fresh_rows[selected][8] if selected is not None else None
    clear = d is not None and d > 0.0
    if cached_indices and selected is None:
        errors.append("positive_gap_manifold_no_exact_fresh_pair")
    if clear and force_active:
        errors.append("positive_gap_nonzero_force_inconsistency")
        label = "positive_gap_nonzero_force_inconsistency"
    elif clear:
        label = "positive_clearance_with_zero_force_cached_manifold" if cached_indices else "positive_clearance_survey"
    elif d is None:
        label = "no_matching_fresh_witness"
    else:
        label = "nonpositive_fresh_pad_target_geometry"
    return dict(passed=not errors, violations=list(dict.fromkeys(errors)), robot=robot, target=target,
                pad_index=pad_index, classification=label, matching_indices=fresh_indices,
                fresh_selected_index=selected, cached_indices=cached_indices, fresh_signed_distance_m=d,
                positive_clearance=clear, geometric_contact=None if d is None else d<=0.0, force_active=force_active)


def classify_contact(row, *, source,phase,channel,robot,target,table,links,entry_by_pad=None,fresh_closest=None,pair_state=None):
    """Classify a *static* query. No state, force, target, TCP or trajectory is changed.

    channel: reference = full commanded q_ref, proxy = admitted contact-limited
    geometry. Reference-only labels never admit the proxy and never predict load.
    Registered assembly gates remain separately mandatory on both channels.
    """
    out=dict(schema=SCHEMA,channel=channel,source=source,phase=phase,classification=None,
             violations=[],reference_only=False,dynamic_gate_applied=False,
             loaded_state_claim=False,geometric_contact=None,force_active=None,required_supplementary_gates=[])
    errors=out["violations"]
    if channel not in ("reference","proxy"):
        out["classification"]="invalid_static_channel";errors.append("unsupported_static_channel");return out
    if not raw_contact_finite(row):
        out["classification"]="malformed_or_nonfinite";errors.append("nonfinite_or_malformed_contact");return out
    if any(int(row[k])!=row[k] for k in (1,2,3,4)):
        out["classification"]="invalid_identity";errors.append("nonintegral_body_or_link_identity");return out
    a,b,la,lb=map(int,(row[1],row[2],row[3],row[4]));d=row[8]
    out.update(body_a=a,body_b=b,link_a=la,link_b=lb,signed_distance_m=d,
        geometric_contact=d<=0,force_active=any(abs(row[k])>FORCE_NUMERIC_ZERO_N for k in (9,10,12)))
    if len({robot,target,table})!=3 or min(robot,target,table)<0:
        out["classification"]="invalid_scene_identity";errors.append("scene_body_identity_conflict");return out
    if phase not in ("settle","opening","transit","pregrasp","approach","close","lift","hold"):
        out['classification']='invalid_phase';errors.append('unsupported_static_phase');return out
    if any(body==robot and link not in set(links.values())|{-1} for body,link in ((a,la),(b,lb))):
        out['classification']='invalid_robot_link';errors.append('unknown_robot_link_identity');return out
    if any(body in (target,table) and link!=-1 for body,link in ((a,la),(b,lb))):
        out['classification']='invalid_object_or_table_link';errors.append('wrong_target_or_table_link');return out
    manifold=source=="manifold"
    pads=(links['ip140_left_pad'],links['ip140_right_pad'])
    bones=(links['ip140_left_bone'],links['ip140_right_bone'])
    bone_pad=dict(zip(bones,pads))
    registered={frozenset((links[x],links[y])) for x,y in STRUCTURAL_PAIRS}
    if a==robot and b==robot:
        pair=frozenset((la,lb))
        if la==lb:
            out['classification']='same_link_closest_query'
            if manifold:errors.append('same_link_manifold')
        elif pair in registered:
            out['classification']='registered_internal_assembly'
            out['required_supplementary_gates']=['unchanged_registered_assembly_audit']
            if manifold:errors.append('filtered_internal_pair_reappeared')
        elif pair==frozenset(pads):
            out['classification']='active_opposing_pads'
            out['required_supplementary_gates']=['unchanged_pad_seam_witness_audit']
            if d < -STATIC_NUMERIC_ZERO_M: errors.append('active_pad_pair_negative_gap')
            if manifold and any(abs(row[k])>1e-9 for k in (9,10,12)):
                errors.append('active_pad_pair_nonzero_force')
        else:
            out['classification']='unregistered_self_pair'
            if manifold or d < -STATIC_NUMERIC_ZERO_M:errors.append('unregistered_self_collision')
    elif {a,b}=={robot,table}:
        rlink=la if a==robot else lb;tlink=lb if b==table else la
        if rlink==links['base_link_inertia'] and tlink==-1:
            out['classification']='registered_fixed_world_mount'
            out['required_supplementary_gates']=['unchanged_base_table_mount_invariant']
            if manifold:errors.append('filtered_base_table_pair_reappeared')
        else:
            out['classification']='forbidden_robot_table'
            if manifold or d < -STATIC_NUMERIC_ZERO_M:errors.append('robot_table_collision')
    elif {a,b}=={robot,target}:
        rlink=la if a==robot else lb;tlink=lb if b==target else la
        if tlink!=-1:
            out['classification']='invalid_target_link';errors.append('wrong_target_link');return out
        is_pad=rlink in pads; is_bone=rlink in bones
        geometric_contact=d<=0
        force_active=out['force_active']
        if is_pad:
            if pair_state is None:
                # Pure row-level replay supplies the fresh and cached sources separately.
                fresh = fresh_closest if fresh_closest is not None else ([] if manifold else [row])
                pair_state = pad_pair_state(fresh, [row] if manifold else [],
                                            robot=robot, target=target, pad_index=rlink)
            if (pair_state["robot"], pair_state["target"], pair_state["pad_index"]) != (robot, target, rlink):
                errors.append("pad_state_body_link_mismatch")
            errors.extend(pair_state["violations"])
            if pair_state["fresh_selected_index"] is None and "positive_gap_manifold_no_exact_fresh_pair" not in errors:
                errors.append("pad_state_missing_fresh_witness")
            out.update(geometric_contact=pair_state["geometric_contact"],force_active=pair_state["force_active"],
                       fresh_signed_distance_m=pair_state["fresh_signed_distance_m"],
                       fresh_selected_index=pair_state["fresh_selected_index"],entry_trigger=False)
            if errors:
                out["classification"]=pair_state["classification"]
                return out
            if pair_state["positive_clearance"]:
                out.update(classification=pair_state["classification"], proximity=True,
                           required_supplementary_gates=["same_state_fresh_witness_audit"])
                if phase in ("lift","hold"):
                    out.update(robot_link=rlink, contact_part="pad", frozen_target_geometry_diagnostic=True)
                return out
        elif d>0 and force_active:
            out['classification']='positive_gap_nonzero_force_inconsistency'
            errors.append('positive_gap_nonzero_force_inconsistency');return out
        if phase in ("lift", "hold"):
            # Only an identified robot/021 pair can reach this branch. A target
            # frozen in its initial pose cannot predict contacts after lifting.
            out.update(classification=("fixed_target_postentry_pad_geometry" if is_pad else
                                       "fixed_target_postentry_nonpad_geometry"),
                       robot_link=rlink, contact_part="pad" if is_pad else "nonpad",
                       frozen_target_geometry_diagnostic=True, entry_trigger=False)
            # Apply source integrity checks to nonpad observations as well.
            normal = np.asarray(row[7])
            delta = np.asarray(row[5]) - np.asarray(row[6])
            proxy_values = dict(normal_unit_error=abs(float(np.linalg.norm(normal))-1.),
                full_witness_residual_m=float(np.linalg.norm(delta-d*normal)),
                normal_projection_residual_m=abs(float(delta @ normal)-d))
            out["witness_arithmetic"] = proxy_values
            if not finite(proxy_values.values()):
                errors.append("nonfinite_postentry_witness_arithmetic")
            if proxy_values["normal_unit_error"] > STATIC_NUMERIC_ZERO_M:
                errors.append("normal_not_unit")
            metric = "normal_projection_residual_m" if manifold else "full_witness_residual_m"
            if proxy_values[metric] > STATIC_NUMERIC_ZERO_M:
                errors.append(metric)
            return out
        if is_pad and manifold:
            out.update(classification="persistent_manifold_with_nonpositive_fresh_distance",
                       proximity=False, entry_trigger=False, required_supplementary_gates=[
                           "persistent_pad_face_projection_audit", "same_state_fresh_witness_audit"])
            return out
        # For pads, a manifold by itself is not contact. Other links retain strict handling.
        hit=(geometric_contact or force_active) if is_pad else (manifold or geometric_contact)
        if not (is_pad or is_bone):
            # Palm/arm pose is independent of q_finger: a reference exemption is invalid.
            out['classification']='forbidden_finger_independent_robot_target'
            if hit:errors.append('finger_independent_robot_target_collision')
            return out
        own_pad=rlink if is_pad else bone_pad[rlink]
        c=(entry_by_pad or {}).get(own_pad)
        certificate_errors=validate_entry_certificate(c,robot=robot,target=target,pad=own_pad)
        admitted=not certificate_errors
        if channel=='reference' and geometric_contact and admitted and phase in REFERENCE_OVERTRAVEL_PHASES:
            out['classification']=('post_contact_motor_reference_overtravel' if is_pad else
                                   'post_contact_bone_motor_reference_overclosure')
            out.update(reference_only=True,entry_pad_link=own_pad,
                       dynamic_penetration_threshold_not_applicable_to_motor_reference=True)
        elif is_pad:
            out['classification']='pad_target_static_geometry'
            if hit and phase not in REFERENCE_OVERTRAVEL_PHASES: errors.append('pad_target_contact_before_close')
            elif hit and not admitted: errors.extend(['pad_contact_without_legal_entry',*certificate_errors])
            if channel=='proxy' and d < -DYNAMIC_PENETRATION_LIMIT_M:
                errors.append('proxy_pad_target_penetration')
            if channel=='reference' and hit and not admitted:
                errors.append('reference_negative_or_manifold_before_legal_entry')
        else:
            out['classification']='forbidden_nonpad_target'
            if hit: errors.append('nonpad_target_collision')
        # The proxy retains the existing external -1 mm gate; it is not dynamics.
    elif {a,b}=={target,table}:
        out['classification']='registered_target_table_support_geometry'
        if la!=-1 or lb!=-1:errors.append('wrong_support_link')
        if d < -DYNAMIC_PENETRATION_LIMIT_M:errors.append('support_penetration_below_minus_1mm')
    else:
        out['classification']='unknown_body_or_environment';errors.append('unknown_contact_identity')
    out['violations']=list(dict.fromkeys(errors))
    return out

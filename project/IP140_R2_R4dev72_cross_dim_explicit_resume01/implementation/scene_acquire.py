"""Settle twelve registered base scenes once and acquire RGB-D with no segmentation labels.

Call only after the parent pipeline has admitted the end-to-end, zero-physics interface.
Preparation has no grasp outcomes, method intents or trial count.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from platform_imports import ROOT, VERSION, bind_frozen_libraries
from scene_world import (canonical, read, registered_scene, render_observation, settle_state,
                         setup_world, sha, write)
from scene_camera import camera_geometry


def settle_observation(ctx,destination):
    from platform_episode import (arm_checks, corner_template, get_object_state, log_motor_commands,
                                  pose_drift, surfaces, PAD, SIDES, PALM)
    from platform_evaluator import ObjectEvaluator
    from platform_support import bind_asset, GRAVITY_WORLD_M_S2
    from base_table_mount import capture as capture_mount
    from structural_filter import current_frames, audit_filtered, audit_seams
    from trajectory import chain, DT
    from speed_policy import phase_speed
    p,c,b,links=ctx.p,ctx.client,ctx.robot,ctx.links
    kw=dict(physicsClientId=c)
    bind_asset(ctx.object_urdf_path,ctx.object_mass_kg)
    evaluator=ObjectEvaluator(b,ctx.target,ctx.table,links,object_mass_kg=ctx.object_mass_kg,
                              gravity_world_m_s2=GRAVITY_WORLD_M_S2)
    limits=np.asarray([lim for _,_,_,lim in chain() if lim is not None])
    corners=[corner_template(p,b,links[n],s,c) for n,s in zip(PAD,SIDES)]
    previous=current_frames(p,b,links,c)
    initial=previous[PALM].copy()
    speed=phase_speed("settle")
    p.authorize_observation(c)
    with gzip.open(destination/"settle_telemetry.jsonl.gz","xt",encoding="utf8",compresslevel=1) as log:
        for tick in range(1,241):
            log_motor_commands(p,c,b,ctx.arm,ctx.fingers,ctx.home,0.,ctx.arm_efforts,phase="settle",reference_speed=speed,motor_max_velocity=speed)
            p.stepSimulation(**kw)
            states=p.getJointStates(b,ctx.arm+ctx.fingers,**kw)
            frames=current_frames(p,b,links,c)
            contacts=p.getContactPoints(**kw)
            truth=get_object_state(p,c,ctx.target)
            fingers=states[-2:]
            widths,_,nonneg=surfaces(frames,corners,0.)
            structural=audit_filtered(p,b,links,c,contacts,current_frames_w=frames,q_by_side=dict(zip(SIDES,[r[0] for r in fingers])))
            seams=audit_seams(p,b,links,c,contacts,require_pad_witness=min(widths["gaps_m"])<=.002,
                              manifold_frames=previous,current_frames_w=frames,dt=DT)
            mount=capture_mount(p,c,b,ctx.table,links,stage="dynamic",raw_manifolds=contacts)
            issues=structural["violations"]+seams["violations"]+mount["violations"]+arm_checks(states[:6],ctx.home,limits,ctx.arm_efforts,ctx.home)
            hold=pose_drift(initial,frames[PALM])
            if hold["translation_m"]>.005 or hold["rotation_deg"]>3.:issues.append("hold_tcp_drift")
            if any(not -1e-5<=st[0]<=.07001 for st in fingers):issues.append("measured_slider_limit")
            if not nonneg or any(not np.isfinite(v).all() for v in frames.values()):issues.append("nonfinite_or_negative_geometry")
            if widths["spread_m"]>.0005 or abs(widths["midline_m"])>.0005 or widths["abs_symmetry_error_m"]>.001:issues.append("pad_geometry")
            decision=evaluator.observe(tick,"settle",contacts,fingers,truth,frames[PALM][:3,3].tolist(),engineering_violations=issues)
            row=dict(tick=tick,phase="observation_scene_settle",q_dq_wrench_motor=states,world_manifolds=contacts,
                     tool=frames[PALM],target=truth,table_normal_force_sum_N=decision["table_normal_force_sum_N"],
                     validation_issues=decision["violations"],widths=widths)
            log.write(canonical(row)+"\n")
            if decision["violations"]:raise ValueError(f"scene settle admission at tick {tick}: {decision['violations']}")
            previous=frames
    return settle_state(ctx)


def observation(scene, admission):
    from platform_trace import R4Trace
    import pybullet
    if not admission.is_file():raise ValueError("missing end-to-end gate")
    gate=read(admission)
    if gate.get("passed") is not True or gate.get("method_trials",0)!=0 or gate.get("stepSimulation",0)!=0:
        raise ValueError("zero-physics end-to-end gate did not pass")
    reg=registered_scene(scene)
    dst=VERSION/"scenes"/reg["base_scene"]
    dst.mkdir(parents=True,exist_ok=False)
    trace=R4Trace(pybullet,dst/"api","observation")
    cid=None
    started=time.perf_counter()
    try:
        cid=trace.connect(trace.DIRECT)
        if cid<0:raise RuntimeError("DIRECT connection failed")
        ctx=setup_world(trace,cid,scene,dst/"admission")
        stable=settle_observation(ctx,dst)
        write(dst/"settled_state.json",stable)
        rgbd=render_observation(ctx)
        K,W_C,z,valid=camera_geometry(rgbd["depth_buffer"],rgbd["view"],rgbd["projection"])
        with (dst/"observed_rgbd.npz").open("xb") as f:np.savez_compressed(f,**rgbd)
        if not valid.any() or not np.isfinite(z[valid]).all():raise ValueError("no valid finite RGB-D observation")
        trace.disconnect(physicsClientId=cid)
        cid=None
        trace.close()
        write(dst/"receipt.json",dict(scene_key=scene,base=reg["base_scene"],observation_rgbd_sha256=sha(dst/"observed_rgbd.npz"),
            settled_state_sha256=sha(dst/"settled_state.json"),admission_sha256=sha(admission),camera_intrinsics=K,
            T_world_camera=W_C,native_width=320,native_height=240,valid_native_depth_pixels=int(valid.sum()),camera_mode="TinyRenderer RGB-D, segmentation disabled",
            actual_steps=trace.actual_step_calls,step_returns=trace.returns["stepSimulation"],method_trials=0,grasp_outcomes=0,
            observed_methods=[],wall_seconds=time.perf_counter()-started,full_native_api_summary_sha256=sha(dst/"api/summary.json")))
    except BaseException as exc:
        try:
            if cid is not None:trace.disconnect(physicsClientId=cid)
        finally:
            if not trace.stream.closed:trace.close()
            write(dst/"STOP.json",dict(error=repr(exc),traceback=traceback.format_exc(),actual_step_calls=trace.actual_step_calls,
                                     returned_steps=trace.returns["stepSimulation"],method_trials=0))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--interface-gate",type=Path,required=True)
    args=parser.parse_args()
    if ROOT.resolve()!=Path.cwd().resolve():raise ValueError("run in registered project root")
    bind_frozen_libraries()
    for scene in range(1,13):
        observation(scene,args.interface_gate.resolve())

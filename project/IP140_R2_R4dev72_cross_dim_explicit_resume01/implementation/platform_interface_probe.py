"""No Bullet import or native connection. Check frozen transforms on archived P3 rows."""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
ENGINE=ROOT/"IP140_R2_smoke22_miss_negative_controls_full_execution01"
sys.dont_write_bytecode=True
os.environ["IP140_SLOT"]="10"
sys.path.insert(0,str(ENGINE))
from miss_scope22 import pre_lift_gate,registered_miss

def read(path):return json.loads(path.read_bytes())
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def relative(path):return path.relative_to(ROOT).as_posix()
def H(r,t):
    x=np.eye(4);x[:3,:3]=r;x[:3,3]=t
    return x

def ip140(row,WC,FG):
    """R4 calibrations have to be supplied from their own paired rendering."""
    row=np.asarray(row,dtype=np.float64)
    if row.shape!=(17,) or not np.isfinite(row).all() or not np.isfinite(WC).all():
        raise ValueError("invalid input")
    R=row[4:13].reshape(3,3); depth=float(row[3]); width=float(row[1]);t=row[13:16]
    if not 0<=width<=.14:raise ValueError("frozen opening out of bounds")
    CG=H(R,t+R[:,0]*depth)
    WG=WC@CG
    return WG,WG@np.linalg.inv(FG),width

def main():
    if Path.cwd().resolve()!=ROOT:raise ValueError("explicit root required")
    mf=read(ENGINE/"model/manifest.json")
    FG=np.asarray(mf["F_T_G"])
    fingerprints={};sample=None;n=0;maxima={};wid=[];depths=[];source_cases=0
    def metric(key,v): maxima[key]=max(maxima.get(key,0.),float(np.max(np.abs(v))))
    for p in sorted((ROOT/"results/simulation_v52/audit/policy_inputs").glob("scene*.json")):
        if p.name.endswith(".receipt.json"):continue
        source_cases+=1
        d=read(p);fingerprints[relative(p)]=sha(p)
        folder=ROOT/"data/raw/prerequisites_v26/scenes"/("scene_%04d"%d["case"]["scene"])/"realsense"
        cam=folder/"cam0_wrt_table.npy";pose=folder/"camera_poses.npy"
        for path in (cam,pose):fingerprints[relative(path)]=sha(path)
        WC=np.asarray(np.load(cam,allow_pickle=False),dtype=float)@np.asarray(np.load(pose,allow_pickle=False)[d["case"]["frame"]],dtype=float)
        for native in d["candidate_rows"]:
            row=np.asarray(native); R=row[4:13].reshape(3,3)
            WG,WF,width=ip140(row,WC,FG)
            # Independent path: transform t to world first, then add the world approach times d.
            world_t=(WC@np.r_[row[13:16],1.])[:3]
            world_R=WC[:3,:3]@R
            world_tip=world_t+world_R[:,0]*row[3]
            metric("two_mapping_paths_m",world_tip-WG[:3,3])
            metric("flange_mount_roundtrip_m", (WF@FG-WG)[:3,3])
            metric("flange_mount_roundtrip_rotation_components",(WF@FG-WG)[:3,:3])
            metric("floating_P3_native_SO3_orthogonality_defect",R.T@R-np.eye(3))
            metric("floating_P3_native_SO3_determinant_error",np.linalg.det(R)-1.)
            if sample is None:
                sample=dict(policy_input=relative(p),candidate_index=int(d["pool_indices"][0]),row_camera_17=native,
                            T_W_C=WC.tolist(),T_W_G=WG.tolist(),T_W_F=WF.tolist(),pre_open_m=width)
            wid.append(width);depths.append(float(row[3]));n+=1
    if not n or maxima["two_mapping_paths_m"]>1e-9 or maxima["flange_mount_roundtrip_m"]>1e-9:
        raise ValueError("tip/flange transform regression")
    for path in [ENGINE/"model/manifest.json",ENGINE/"model/ur5e_ip140_r1.urdf",ENGINE/"trajectory.py",
                 ENGINE/"object_episode.py",ENGINE/"object_evaluator.py",ENGINE/"zero_scene.py",ENGINE/"static_admission_check.py",
                 ENGINE/"contact_aware_path.py",ENGINE/"miss_scope22.py",ROOT/"scripts/sim_v52r2_episode.py",
                 ROOT/"scripts/sim_v52_math.py",ROOT/"IP140-R1_执行端替换两页协议_小修02.pdf",ROOT/"IP140-R1_参数与证据附表_小修02.pdf",
                 ROOT/"docs/机器人仿真工程准入协议_v52.txt",ROOT/"configs/simulation_v52r2.json"]:
        fingerprints[relative(path)]=sha(path)
    p21=ROOT/"IP140_R2_smoke21_registered_pad_surface_and_resume_four01"/"slots/10/static_rejection.json"
    proof=read(p21);fingerprints[relative(p21)]=sha(p21)
    rfail=None
    try:pre_lift_gate({},False)
    except ValueError as e:rfail=str(e)
    assert rfail=="both_registered_entries_required_before_lift"
    assert pre_lift_gate({},True)["pre_lift_bilateral_entry_absent"] is True
    assert not registered_miss(10,dict(slot=10,not_p3=False,kind="Raw",object_id="021",candidate_world=sample["row_camera_17"]),"")
    obs=dict(n_archival_cases=source_cases,n_rows=n,example=sample,maximum_absolute_residuals=maxima,
             pre_open_range_m=[min(wid),max(wid)],depth_range_m=[min(depths),max(depths)],F_T_G=FG.tolist(),
             analytical_transform="W_T_G = W_T_C @ [R | t + d*R[:,0]], W_T_F = W_T_G @ inverse(F_T_G)",
             pre_open="frozen P3 width directly; do not add the engineering +0.015 m",
             legacy_height_term="absent",test_uses_no_shape_truth=True,scope="historical P3 coordinate arithmetic only; not a 24-unit R4 pool, ranking, planned trajectory, admission, or trial")
    out=dict(version=HERE.name,zero_physical_steps=True,pybullet_connections=0,method_trials=0,
       source_sha256=fingerprints,coordinates=obs,
       exact_engineering_worker=dict(not_p3_required_by="object_episode.run_slot, trajectory.build",
          zero_receipt="smoke22/zero_scene.py validates the bridge preregistration and parent slot width.",
          actual_missing_cert_exception_for_generic=rfail,
          actual_archival_counterexample=dict(source=relative(p21),charged=proof["charged"],reason=proof["rejection"]["reason"],missing=proof["rejection"]["missing_sides"])),
       selection_roles=dict(observable="R4 explicit immutable selection receipt must declare eligibility inputs (RGB-D, calibrated workspace/robot/table, reachability, robot-only path). The frozen score ranking and this first-eligible policy must not query the simulated target.",
          safety_only_after_lock="Only after irrevocably binding a candidate, run the original geometric safety/entry checks. Reject an inadmissible candidate as trial-level path/admission failure, never select the next candidate because of true target, and never waive bilateral entry for ordinary R4 candidates.",
          no_executable_candidate="Reserved for an exhausted observable eligibility list; distinct from an exact target-aware failure after irrevocable selection.",
          history="v52r2 planner actually loaded only robot/table; reviewed appendix C prohibits swapping the top selected candidate after outcome. The smoke22 production entry itself is typed for its registered misses only. A separate method intent adapter is required; passing not_p3=true would mis-state origin.",
          proof_limit="The observed-only/postlock split is a possible protocol-compliant composition, not an existing sealed R4 end-to-end receipt and not a reason to expand the smoke22 exemption.",
          hard_stop="If decision registration still uses true body/contact for fallback, or the actual live P3/A/C0 pipeline fails the pretrial contract, stop before first trial. Historical mapping alone cannot authorize it."),
       scoring_geometry=dict(frozen_source="IP140-R1_参数与证据附表_小修02.pdf p.4, appendix C",
          finger_boxes_G_m="x=[-.060,0], z=[-.010,+.010], each 10mm thick outside +w/2 and -w/2",
          swept_palm_box_G_m="x=[-.120,-.060], y=[-.080,+.080], z=[-.010,+.010]",
          partitioned_union_reference_m3=0.000216,normalization_voxels=216,
          old_v52_geometry_equivalent=False,nongeometry_parameters_bind_to_v52=True))
    dest=HERE/"audit/platform_interface_probe.json"
    with dest.open("x",encoding="utf8") as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(source_cases,n,maxima,rfail)
if __name__=="__main__":main()


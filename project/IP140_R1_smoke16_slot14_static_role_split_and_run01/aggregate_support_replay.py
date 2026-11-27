"""Pure aggregation replay, including real historical telemetry. No engine import."""
import copy,gzip,hashlib,json,math
from pathlib import Path
from aggregate_support import aggregate_table_contacts, verify_physics_context
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/"IP140_R1_smoke12_slot07_base_table_mount_filter01"
FIELDS=("table_contact_present","table_normal_force_sum_N","table_load_fraction","aggregate_table_support")
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(path):return json.loads(path.read_text("utf-8-sig"))
def c(fn=.3,d=0.,a=2,b=1,la=-1,lb=-1):
    return [0,a,b,la,lb,[0.,0.,d],[0.,0.,0.],[0.,0.,1.],d,fn,0.,[1.,0.,0.],0.,[0.,1.,0.]]
def safe(v):
    if isinstance(v,float) and not math.isfinite(v):return str(v)
    if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    return v
def build():
    cases=[]
    def test(name,raw,support,present=None,invalid=None):
        result=aggregate_table_contacts(raw,2,1)
        ok=result["aggregate_table_support"]==support
        if present is not None:ok=ok and result["table_contact_present"]==present
        if invalid is not None:ok=ok and invalid in result["aggregation_violations"]
        cases.append(dict(name=name,passed=ok,input=safe(raw),expected_aggregate_support=support,
                          expected_violation=invalid,actual=result))
    test("aggregate_no_manifold",[],False,False)
    test("aggregate_zero_force_manifold",[c(0.)],False,True)
    test("aggregate_single_legacy_positive",[c(.3)],True,True)
    test("aggregate_four_subthreshold_positive_points",[c(.1) for _ in range(4)],True,True)
    test("aggregate_exact_total_0p2",[c(.1),c(.1)],False,True)
    test("aggregate_nextafter_total_above_0p2",[c(math.nextafter(.2,math.inf))],True,True)
    test("aggregate_nonpositive_forces_excluded",[c(-.5),c(.19)],False,True)
    test("aggregate_unrelated_body_never_counted",[c(2.,a=99)],False,False)
    test("aggregate_mount_is_not_object_support",[c(2.,a=0,la=1)],False,False)
    test("aggregate_wrong_object_link_fails",[c(.3,la=2)],None,True,"wrong_support_link")
    test("aggregate_wrong_table_link_fails",[c(.3,lb=2)],None,True,"wrong_support_link")
    test("aggregate_exact_minus1mm",[c(.3,d=-.001)],True,True)
    test("aggregate_penetration_below_limit_fails",[c(.3,d=math.nextafter(-.001,-math.inf))],None,True,"external_penetration_below_minus_1mm")
    bad=c();bad[9]=float("nan")
    test("aggregate_nonfinite_force_fails",[bad],None,True,"nonfinite_or_incomplete_contact")
    bad=c();bad[11][1]=float("inf")
    test("aggregate_nonfinite_tangent_fails",[bad],None,True,"nonfinite_or_incomplete_contact")
    test("aggregate_sum_overflow_fails",[c(1.e308),c(1.e308)],None,True,"nonfinite_aggregate_table_force")
    rev=c(.3);rev[1],rev[2]=rev[2],rev[1];rev[3],rev[4]=rev[4],rev[3];rev[5],rev[6]=rev[6],rev[5];rev[7]=[-v for v in rev[7]]
    test("aggregate_native_endpoint_reversed",[rev],True,True)
    for name,mass,gravity in [("wrong_actual_mass",.041,(0,0,-9.81)),("wrong_actual_gravity",.04,(0,0,-9.8)),("nonfinite_actual_mass",float("nan"),(0,0,-9.81))]:
        rejected=False
        try:verify_physics_context(mass,gravity)
        except ValueError:rejected=True
        cases.append(dict(name="aggregate_"+name,passed=rejected,expected_rejection=True,input=safe([mass,gravity])))
    previous=read(OLD/"offline_gate_replay.json")
    historic=[]
    for row in previous["cases"]:
        sample=row.get("selected_raw_input_and_result")
        if not sample:continue
        out=aggregate_table_contacts(sample["raw_contacts"],previous["synthetic_object_body"],previous["synthetic_table_body"])
        legacy_recomputed=any(sample["raw_contacts"][i][9]>.2 for i in out["legal_table_contact_indices"])
        historic.append(dict(name=row["name"],step=sample["step"],original_table_support=sample["result"]["table_support"],
                         legacy_recomputed=legacy_recomputed,aggregation=out,passed=legacy_recomputed==sample["result"]["table_support"]))
    cases.append(dict(name="aggregate_replays_33_saved_historical_samples",passed=len(historic)==33 and all(r["passed"] for r in historic),
                      samples=historic,scope="Stored support inputs only; no rewriting full historical results."))
    binding=read(OLD/"api_and_episode_audit.json")["bindings"]
    mass=read(OLD/"zero_step/object021_readback.json")["dynamics"][0]
    phy=read(OLD/"zero_readback.json")["physics_readback"]
    gravity=[phy["gravityAcceleration"+axis] for axis in "XYZ"]
    actual=[]
    with gzip.open(OLD/"episodes/07/telemetry.jsonl.gz","rt",encoding="utf-8") as stream:
        for line_no,line in enumerate(stream,1):
            row=json.loads(line)
            if "base_table_mount_audit" not in row:continue
            out=aggregate_table_contacts(row["raw_manifolds"],binding["object"],binding["table"],object_mass_kg=mass,gravity_world_m_s2=gravity)
            old=any(row["raw_manifolds"][i][9]>.2 for i in out["legal_table_contact_indices"])
            actual.append(dict(step=row["index"],phase=row["phase"],telemetry_line=line_no,
                               original_table_support=row["object_evaluation"]["table_support"],
                               legacy_recomputed=old,**out))
    end=actual[-480:]
    valid=(len(actual)==4728 and all(r["aggregation_valid"] for r in actual)
       and all(r["legacy_recomputed"]==r["original_table_support"] for r in actual)
       and [r["step"] for r in end]==list(range(4249,4729))
       and all(not r["original_table_support"] and r["aggregate_table_support"] is True for r in end)
       and all(.392355<=r["table_normal_force_sum_N"]<=.392445 for r in end))
    end_summary=dict(count=len(end),original_true=sum(r["original_table_support"] for r in end),
            aggregate_true=sum(r["aggregate_table_support"] is True for r in end),
            force_sum_min_N=min(r["table_normal_force_sum_N"] for r in end),
            force_sum_max_N=max(r["table_normal_force_sum_N"] for r in end))
    cases.append(dict(name="aggregate_smoke12_full_4728_and_terminal480",passed=valid,actual_final480=end_summary,
              actual_mass_kg=mass,actual_gravity_world_m_s2=gravity))
    sources=[OLD/"episodes/07/telemetry.jsonl.gz",OLD/"offline_gate_replay.json",OLD/"zero_readback.json",OLD/"zero_step/object021_readback.json",
             HERE/"aggregate_support.py",Path(__file__)]
    return safe(dict(version=HERE.name,passed=all(x["passed"] for x in cases),cases=cases,case_count=len(cases),
         final480=end_summary,actual_4728_rows=actual,source_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in sources},
         engine_connections=0,physical_steps=0,new_charges=0,
         note="New version evaluation only. Old table_support and charged failure are unchanged."))
if __name__=="__main__":
    raise SystemExit("Use pure build() for development or formal offline_replay.persist_report integration; no standalone session")

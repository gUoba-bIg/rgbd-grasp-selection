"""Verify frozen inputs directly, including the deployed V1 runtime and its physical engine."""
from paths import *
import subprocess
import zipfile

def tag(repo):
    return subprocess.check_output(["git","--no-optional-locks","-C",str(repo),"rev-parse",TAG+"^{commit}"],text=True).strip()

def post_cold_bindings(m, members):
    # These four outputs were produced by the accepted cold run in its disposable copy.
    # Reconstruct every byte from the original R6 evidence and its independently sealed receipts.
    seal_name=R6NAME+"/sealed_intents.json"
    seal=read(ROOT/seal_name)
    need(sha(ROOT/seal_name)==members[seal_name][1],"source formal registration changed")
    def encoded(value):
        return (json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\r\n").encode("utf8")
    outputs={};proof={}
    for tid in ("R6FORMAL-007","R6FORMAL-004","R6FORMAL-001"):
        name=R6NAME+"/trials/"+tid+"/intent.json"
        rel="c/cold_receipts/"+tid+"/"
        for suffix in ("before_intent.json","before_sealed_intents.json","receipt_rebinding.json","rank_return.json","method_return.json"):
            verify_old(rel+suffix,m)
        before=read(PRIOR/rel/"before_intent.json"); corrections=read(PRIOR/rel/"receipt_rebinding.json")
        need(read(PRIOR/rel/"before_sealed_intents.json")==seal and
             sha(PRIOR/rel/"before_intent.json")==sha(ROOT/name)==members[name][1]==corrections["original"],"cold before-image mismatch")
        for key,change in corrections["differences"].items():
            need(key in ("ranking","selection","scoring_components") and before[key+"_sha256"]==change["old"] and
                 sha(C/"p"/R6NAME/before[key+"_path"])==change["new"],"unregistered cold binding")
            before[key+"_sha256"]=change["new"]
        content=encoded(before)
        need(content==(C/"p"/name).read_bytes(),"cold output does not reproduce byte for byte")
        newhash=hashlib.sha256(content).hexdigest()
        outputs[name]=(len(content),newhash)
        seal["files"]["trials/"+tid+"/intent.json"]=newhash
        proof[name]=dict(original=members[name][1],accepted_post_cold=newhash,registered_changes=corrections["differences"],receipt=m[rel+"receipt_rebinding.json"])
    content=encoded(seal)
    need(content==(C/"p"/seal_name).read_bytes(),"cold seal reconstruction differs")
    outputs[seal_name]=(len(content),hashlib.sha256(content).hexdigest())
    proof[seal_name]=dict(original=members[seal_name][1],accepted_post_cold=outputs[seal_name][1],trial_bindings=list(outputs)[:-1])
    return outputs,proof

def check():
    m=original_manifest()
    identity=read(PRIOR/"admission/v1_identity.json")
    old_hashes=read(PRIOR/"admission/code_identity.json")
    original_repo=v1.ARCHIVES.parent/"v1_baseline"
    need(tag(original_repo)==tag(SOURCE)==identity["commit"]==COMMIT,"V1 tag or commit drift")
    need(verify_old("admission/v1_identity.json",m) and verify_old("admission/code_identity.json",m),"previous identity")
    need(read(PRIOR/"admission/cold_3of3.json")["complete_cases"]==3,"previous V1 cold identity")
    verify_old("admission/cold_3of3.json",m)
    for rel,h in identity["control"].items():
        need(sha(SOURCE/rel)==h,"V1 baseline control "+rel)
    with zipfile.ZipFile(PRIOR/"delivery/V1_supplement_robustness_admission01_review.zip") as z:
        for name,h in old_hashes["files"].items():
            need(sha(PREVIOUS_CODE/name)==h==hashlib.sha256(z.read("source/"+name)).hexdigest(),"prior wrapper "+name)
    members=v1.index("execution_project")
    cold_outputs,cold_receipts=post_cold_bindings(m,members)
    for rel,(size,h) in members.items():
        v1.validate_file(C/"p"/rel,*cold_outputs.get(rel,(size,h)))
    extras=read(ROOT/"V1_repro_final04/audit/runtime_supplement.json")["files"]
    for row in extras:
        v1.validate_file(C/"p"/row["registered_path"],row["bytes"],row["sha256"])
    registered=read(ROOT/R6FIRST/"preregistration/frozen_inputs.json")
    for rel,h in registered.items():
        h=h if isinstance(h,str) else h["sha256"]
        need(sha(ROOT/rel)==sha(C/"p"/rel)==h,"registered V1 scientific input "+rel)
    for rel,h in identity["weights"].items():
        need(sha(C/"p"/rel)==sha(ROOT/rel)==h,"weight "+rel)
    engine=C/"p/runtime/simulation_v52e/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd"
    need(sha(engine)==identity["engine_sha256"]==ENGINE,"engine changed")
    return dict(utc=utc(),tag=TAG,commit=COMMIT,repository=str(original_repo),runtime=str(C/"p"),
           execution_project_members=len(members),execution_project_manifest=sha(SOURCE/"locks/execution_project.jsonl.gz"),
           four_source_supplements=extras,registered_inputs=len(registered),weights=identity["weights"],prior_cold_output_derivations=cold_receipts,
           engine_path=str(engine),engine_sha256=ENGINE,previous_code=old_hashes["files"],
           cold_3of3_sha256=sha(PRIOR/"admission/cold_3of3.json"),new_model_calls=0,
           new_environment=False,read_only_runtime=True)

if __name__=="__main__":
    write("admission/v1_identity.json",check())
    print("Frozen V1 verified",flush=True)

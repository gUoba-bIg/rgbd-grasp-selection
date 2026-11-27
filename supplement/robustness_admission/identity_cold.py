"""Verify the V1 source contracts and start its frozen three-case procedure in a fresh destination."""
from common_v1 import *
import sys
import zipfile
import traceback

def verify():
    record={"tag":TAG,"commit":COMMIT,"disk":disk(),"control":{},"archives":{},"raw_evidence":{}}
    git=lambda *args:subprocess.check_output(["git","--no-optional-locks","-C",str(WORK),*args],text=True).strip()
    need(git("rev-parse",TAG+"^{commit}")==COMMIT and git("merge-base","--is-ancestor",COMMIT,"HEAD")=="","V1 tag identity")
    for line in (WORK/"metadata/baseline_files.sha256").read_text("utf-8-sig").splitlines():
        h,n=line.split("  ",1);need(sha(WORK/n)==h,"baseline "+n);record["control"][n]=h
    final=ROOT/"V1_repro_final04"; old_budget=read(final/"audit/final_cold_budget.json")
    for path,h in old_budget["freeze_sources"].items():need(sha(final/path)==h,"V1 accepted wrapper "+path)
    record["final04_budget_sha256"]=sha(final/"audit/final_cold_budget.json")
    for group in ("python312","offline_wheels","execution_project","source_R6_initial","source_R6_final","source_R7_final"):
        rec=read(WORK/"locks"/(group+".receipt.json")); idx=index(group)
        arc=ARCHIVES/rec["name"];validate_file(arc,rec["size"],rec["sha256"])
        with zipfile.ZipFile(arc) as z:
            names={x.filename:x.file_size for x in z.infolist()}
            need(len(names)==len(z.infolist())==len(idx) and all(names.get(k)==n for k,(n,h) in idx.items()),"archive membership "+group)
        record["archives"][group]=dict(path=str(arc),sha256=rec["sha256"],members=len(idx))
        if group.startswith("source_R"):
            count=0;byte_count=0
            for n,(size,h) in idx.items():
                validate_file(ROOT/n,size,h);count+=1;byte_count+=size
            record["raw_evidence"][group]=dict(local_files_verified=count,bytes=byte_count)
        print("V1 identity returned",group,flush=True)
    stores=read(ROOT/R6FIRST/"preregistration/frozen_inputs.json")
    record["frozen_execution_inputs"]=len(stores)
    for path,binding in stores.items():
        h=binding if isinstance(binding,str) else binding["sha256"]
        need(sha(ROOT/path)==h,"original registered input "+path)
    record["weights"]={p:sha(ROOT/p) for p in ("experiments/candidate_v30/epoch_18_final.pt","experiments/p5_fusion_v38/fixed/final.pt")}
    record["engine_sha256"]=sha(ROOT/"runtime/simulation_v52e/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd")
    need(record["engine_sha256"]==ENGINE,"native engine")
    record["utc"]=utc();write("admission/v1_identity.json",record)

def launch(label,cmd):
    disk()
    folder=OUT/("cold_logs_packaging03" if "--installed" in sys.argv else "cold_logs_packaging02" if "--resume-shell" in sys.argv else "cold_logs");folder.mkdir(exist_ok=True)
    shell_modules=Path(os.environ["SystemRoot"])/"System32/WindowsPowerShell/v1.0/Modules"
    with (folder/(label+"_stdout.log")).open("xb") as so,(folder/(label+"_stderr.log")).open("xb") as se:
        task=subprocess.Popen([str(x) for x in cmd],cwd=WORK,stdin=subprocess.DEVNULL,stdout=so,stderr=se,
             env=dict(os.environ,PSModulePath=str(shell_modules),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",PYTHONUTF8="1"))
        while task.poll() is None:
            try:task.wait(timeout=45)
            except subprocess.TimeoutExpired:
                try:disk()
                except Exception:
                    subprocess.run(["taskkill","/T","/F","/PID",str(task.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    raise
        need(task.returncode==0,label+" stopped, exit "+str(task.returncode)+"; see "+str(se.name))
    print("V1 cold stage returned",label,flush=True)

def main():
    reg=read(KIT/"metadata/cold_preregistration.json")
    need([r["expected_physical_returns"] for r in reg["cases"]]==[4728,0,0],"frozen cold budget")
    if "--installed" in sys.argv:
        need(sha(C/"setup_complete.json")=="a750f296ac91d893ae31d24e0a44b21b220d6c184223516a50ff7dc70ea34eff","installed environment receipt")
        need(sha(OUT/"cold_logs_packaging02/four_runtime_originals_stderr.log")=="ca0bad16e294104dc3b46851da52e13b83b5fc39aa1dd7616bd61a2e872e6a85","argument parse receipt")
        need(not (C/"cold_receipts").exists(),"cold run already dispatched")
        write("admission/positional_arguments.json",dict(utc=utc(),disk=disk(),target=str(C),
             correction="Pass target, archive and manifest positionally as required by the sealed restoration script.",
             model_calls=0,physics=0,installation_reused_sha256=sha(C/"setup_complete.json")))
    elif "--resume-shell" in sys.argv:
        need(C.is_dir() and not any(C.iterdir()),"not an unused installation target")
        expected={"admission/v1_identity.json":"e998ed03d8a70120d0f05e3961e5f9c461340e3e27f62e5ca63bfaf866f648fb","admission/cold_preregistration.json":"8d837e584b55c8d171490f8412221675848bd554b2826236e0ac9947be1f586d","cold_logs/setup_stderr.log":"cbd4604552094ce9df8400fb944ea89dd646469dbbbe785ce905e781d2010365"}
        for path,h in expected.items():need(sha(OUT/path)==h,"failed setup evidence changed "+path)
        write("admission/shell_packaging02.json",dict(utc=utc(),disk=disk(),prior_evidence=expected,
            powershell="Windows PowerShell 5.1; native Modules directory passed explicitly to child",
            pre_fix="inherited PSModulePath did not resolve Get-FileHash",post_fix_test="SHA256 of baseline_files.sha256: 1a14bd94e17e0e6ce8faf28eafd0ba1eb8fd59d06cca2428604efb742071fb1f",
            previous_model_calls=0,previous_physical_calls=0,scientific_budget_change=0,
            event_query="UTC Event Log XML; last registered prior-boot event was Ntfs 140 at 2026-11-14T12:27:45.995497Z; boot at 12:28:16.500Z"))
    else:
        verify()
        need(not C.exists(),"setup target already exists")
        write("admission/cold_preregistration.json",dict(utc=utc(),source_sha256=sha(KIT/"metadata/cold_preregistration.json"),
            cases=reg["cases"],physics_budget=4728,P3=3,P5=3,retries=0))
    shell=Path(os.environ["SystemRoot"])/"System32/WindowsPowerShell/v1.0/powershell.exe"
    powershell=[shell,"-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File"]
    if "--installed" not in sys.argv:launch("setup",[*powershell,KIT/"setup_v1.ps1","-Target",C,"-Archives",ARCHIVES])
    launch("four_runtime_originals",[C/"bin/python.exe","-I","-B",ROOT/"V1_repro_final04/ops/restore_runtime_supplement.py",C,
        ROOT/"V1_repro_dependency_closure03/supply/v1_smoke11_registered_sources.zip",ROOT/"V1_repro_final04/audit/runtime_supplement.json"])
    launch("preflight",[*powershell,KIT/"preflight_v1.ps1","-Target",C])
    launch("three_cases",[*powershell,KIT/"run_smoke_v1.ps1","-Target",C])
    sys.path.insert(0,str(KIT/"adapters"));from summarize import main as summarize
    result=summarize(C)
    need(result["complete_cases"]==3 and result["gate_passed"] and not result["stop"],"cold gate")
    for case, expected in zip(result["cases"],reg["cases"],strict=True):
        need(case["classification"]==expected["expected_detail"] and case["native_returns"]==case["native_calls"]==expected["expected_physical_returns"],"cold case")
        need(all(v=={"entered":1,"returned":1} for v in case["model_calls"].values()),"fresh inference")
    write("admission/cold_3of3.json",dict(utc=utc(),fresh=True,**result));print("V1 cold 3/3",flush=True)
if __name__=="__main__":
    try:main()
    except BaseException as exc:
        print(traceback.format_exc(),flush=True)
        if volume()["dirty"]==0:
            disk();write("admission/HALT_installed.json" if "--installed" in sys.argv else "admission/HALT_shell02.json" if "--resume-shell" in sys.argv else "admission/HALT.json",dict(utc=utc(),error=repr(exc),traceback=traceback.format_exc(),cold_complete=(OUT/"admission/cold_3of3.json").exists()))
        raise

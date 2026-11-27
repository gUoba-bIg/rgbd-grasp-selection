"""One hidden finite local job. No scheduler, no automatic scientific retries."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.p6_mixture_common_v43 import *
import argparse,subprocess,traceback,ctypes
ORPHAN_UNCONFIRMED=False

def event(kind,**values):append(LOG/"controller.jsonl",dict(at=now(),kind=kind,**values))
def child_ledger():
    p=LOG/"controller.jsonl"
    rows=[json.loads(v) for v in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
    starts={r["id"]:r for r in rows if r["kind"]=="child_intent"}
    returned={r["id"]:r for r in rows if r["kind"]=="child_returned"}
    need(set(returned)<=set(starts),"Unpaired child return")
    return dict(intents=len(starts),known_returns=len(returned),unknown=sorted(set(starts)-set(returned)))
def process_peak(child):
    class C(ctypes.Structure):
        _fields_=[("cb",ctypes.c_ulong),("faults",ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ("peak","working","a","b","c","d","e","f")]
    x=C();x.cb=ctypes.sizeof(x)
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(int(child._handle)),ctypes.byref(x),x.cb)
    if not ok:
        need(child.poll() is not None,"Cannot monitor live child memory")
        return 0
    return int(x.peak)
def freeze():
    from scripts.pipeline import project_lock
    from src.p6_geometry_reference_v43 import inventory
    start=time.monotonic()
    with project_lock(ROOT):
        need(not CONF.exists() and not (LOG/"authorization.json").exists(),"Do not replace existing freeze")
        tests=read(LOG/"component_tests.json");review=read(LOG/"independent_review.json")
        geo=read(LOG/"geometry/independent_review.json")
        for rec in (tests,review,geo):
            need(rec["passed"],"Tests or independent review failed")
            for name,h in rec["source_sha256"].items():need(sha(ROOT/name)==h,"Reviewed source changed: "+name)
        parent=read(ROOT/"configs/p6_correspondence_v40.json")
        protected=dict(parent["protected"])
        for name,value in protected.items():need(identity(ROOT/name)==value,"Historical v40 dependency changed: "+name)
        protected.update(inventory()["protected"])
        extras=["scripts/p6_mixture_common_v43.py","scripts/p6_mixture_v43.py","scripts/p6_mixture_delivery_v43.py",
            "src/uncertainty_mixture_v43.py","tests/test_uncertainty_mixture_v43.py","scripts/check_p6_mixture_v43.py",
            "src/p6_geometry_reference_v43.py","scripts/p6_geometry_v43.py",
            "docs/P6共享对应与混合不确定性协议_v43.txt","docs/P6研究路线纠偏与下一方案_20261101.txt",
            "logs/P6/mixture_v43/component_tests.json","logs/P6/mixture_v43/independent_review.json",
            "logs/P6/mixture_v43/geometry/component_tests.json","logs/P6/mixture_v43/geometry/independent_review.json","logs/P6/mixture_v43/geometry/independent_array_checks.json",
            "logs/P6/correspondence_v40/data_audit.json","runtime/pipeline/state.json",
            "runtime/envs/vision/Scripts/python.exe","runtime/envs/grasp_eval/Scripts/python.exe",
            "results/P6/selective_v42/result.json","docs/P6匹配歧义有限修复结果_v42.txt"]
        for name in extras:protected[name]=identity(ROOT/name)
        # Bound prep computation conservatively; code-authoring wall time is not research runtime.
        measured=read(LOG/"preparation_usage.json")
        prep=float(measured["seconds"])+30.
        need(prep<120,"Preparation exceeds reserved120 seconds; do not launch")
        once(LOG/"authorization.json",dict(at=now(),frozen=True,protected=protected,
            user_authorization="Complete method design/internal checks and run corresponding P6 experiment in background; stop after this part",
            current_title="基于质量感知RGB-D实例分割与几何不确定性引导的六自由度机器人抓取位姿估计方法",
            all_tasks_percent=45,experiment_percent=56.25,P10_still_stopped=True))
        protected["logs/P6/mixture_v43/authorization.json"]=identity(LOG/"authorization.json")
        stage_data=[
            ("geometry","runtime/envs/grasp_eval/Scripts/python.exe","scripts/p6_geometry_v43.py","run","results/P6/mixture_v43/geometry/result.json",600),
            ("geometry_review","runtime/envs/grasp_eval/Scripts/python.exe","scripts/p6_geometry_v43.py","review","logs/P6/mixture_v43/geometry/review.json",600),
            ("generate","runtime/envs/vision/Scripts/python.exe","scripts/p6_mixture_v43.py","generate","results/P6/mixture_v43/generated.json",1800),
            ("fit","runtime/envs/vision/Scripts/python.exe","scripts/p6_mixture_v43.py","fit","results/P6/mixture_v43/model.json",600),
            ("evaluate","runtime/envs/vision/Scripts/python.exe","scripts/p6_mixture_v43.py","evaluate","results/P6/mixture_v43/result.json",1800),
            ("review","runtime/envs/vision/Scripts/python.exe","scripts/p6_mixture_v43.py","review","logs/P6/mixture_v43/review.json",2400),
            ("cold","runtime/envs/vision/Scripts/python.exe","scripts/p6_mixture_v43.py","cold","logs/P6/mixture_v43/cold.json",120)]
        c=dict(at=now(),version="v43",protected=protected,controller_wall_seconds=3600,preparation_seconds=prep,
            stages=[dict(name=n,python=p,script=s,arg=a,done=d,stage_wall_seconds=t) for n,p,s,a,d,t in stage_data],
            progress_before=[45,56.25],progress_after_if_P6_development_passes=[47,58.75],call_limits=CALL_LIMITS)
        once(CONF,c);once(LOG/"freeze.json",dict(at=now(),config_sha256=sha(CONF),protected_files=len(protected)))
        verify()
        need(time.monotonic()-start<=30,"Freeze exceeded reserved preflight30s; do not launch")
        status("frozen_ready",prepared_calls=tests.get("checks",[]),protected_files=len(protected))
        supervise("P6_v43_frozen_ready",waiting_for_user_next_part_confirmation=False,current_execution_evidence="logs/P6/mixture_v43/freeze.json")
        handoff("用户已授权P6 v43整体概率模型后台实验，设计/内部检查/独立审查通过并冻结。80旧开发帧×9位移×3观测模式=2160案例，两概率头各一次拟合；四帧CAD仅辅助，绝不输入预测。完整复验/冷重入/真实汇总后停止；不启动P7/P10/周期任务。全部任务总进度45%、实验总进度56.25%，仅P6本开发完整验收通过后按原剩余2点更新。入口scripts/p6_mixture_delivery_v43.py deliver，先核实际进程/锁，禁止重复。","freeze")
        print(json.dumps(dict(frozen=True,files=len(protected),preparation_seconds=prep)))
def finish(c,elapsed,guard):
    review=read(LOG/"review.json");cold=read(LOG/"cold.json");geo=read(LOG/"geometry/review.json")
    need(review["passed"] and cold["passed"] and geo["passed"],"Read-only/cold must pass")
    need(sha(LOG/"review.json")==cold["review_sha256"],"Cold receipt mismatch")
    for name,value in review["reviewed_outputs"].items():need(identity(ROOT/name)==value,"Final output changed: "+name)
    need(not ledger()["unknown"] and not child_ledger()["unknown"],"Unknown calls")
    need(ledger()["counts"]==CALL_LIMITS,"Complete scientific call counts differ")
    need(geo["real_render_calls_total"]==4 and not geo["unknown_calls"],"Geometry call accounting incomplete")
    actual={relative(p) for p in OUT.rglob("*") if p.is_file()}
    need(actual==set(review["reviewed_outputs"]),"Unreviewed or missing final artifact")
    result=read(OUT/"result.json");passed=result["development_scientific_passed"]
    need(passed==all(result["gates"].values())==cold["development_scientific_passed"],"Gate/cold/result mismatch")
    verify()
    progress=[47,58.75] if passed else [45,56.25]
    report=[
        "P6共享对应与混合不确定性实验v43",now(),"",
        f"全部任务总进度{progress[0]}%；实验总进度{progress[1]}%。固定主交付权重；科学有效性单列。",
        "本次已完成一次有界后台部分、完整只读复验及真实新进程冷重入，现已停止。",
        "P6已知合成扰动开发验收通过="+str(passed)+"。工程验收通过=True。",
        "范围：20旧暴露场景80帧，9种已知整数位移与3种噪声/缺测模式；不是独立最终测试或真实传感器绝对校准，也未验证新的抓取收益。",
        "A为观测物理样本的源uvz；B为固定RGB查询的表面/缺测分布。缺测与可见数值覆盖分开，全部分母保留。原始XYZ未覆盖。",
        "CAD四帧仅辅助参考，三项geometry_trusted标志不认证全部表面/ID，未用于模型拟合或预测。信任标志="+str(result["geometry_auxiliary"]["geometry_trusted"]),
        "模型：共享81候选条件概率与零位置t3同源深度尺度，各仅拟合一次；强基线为训练经验先验，非仅均匀先验。",
        "校准集仅诊断，未调参/重新校准；所有前次科学失败、旧源码/输入/阈值/预算及原state.json不变。",
        "科学门槛："+json.dumps(result["gates"],ensure_ascii=False),
        "完整摘要："+json.dumps(result["summary"],ensure_ascii=False,indent=2),
        "真实调用账本："+json.dumps(ledger(),ensure_ascii=False),
        "0神经前向、0官方AP、0抓取调用；案例数不是神经前向。两头实际objective/迭代次数见model.json，未收敛状态如实保留。",
        "进入最终收尾时累计记账秒数（含准备及保守预检扣款）："+str(elapsed)+"；最终实际累计见controller_usage，硬上限3600。",
        ("可用于原P6已知合成开发范围；关闭原剩余2点。后续P7/P8验证抓取用途，需用户检查后再进入下一主部分。" if passed else
         "当前候选未达到完整采用门槛。失败项已明列；保留原概率输出/原观测与负结果。下一步仅针对失败类型提出结构性建议，不自动改阈值或追加实验。"),
        "失败时的修复判断：若warp proper分数/可靠性不佳，应核可观测对应证据是否足够；若同源噪声覆盖/宽度失败，应修噪声分布而非扩大所有分量；若B边界失败，应核候选支持与missing事件，不回到强制唯一配准。任何新训练另立有限协议。",
        "P5科学优势未获证实，未进入P7/P10，未恢复周期任务。"]
    guard()
    need(not REPORT.exists(),"Refuse to overwrite previous report")
    REPORT.write_text("\n".join(report)+"\n",encoding="utf-8")
    guard()
    proof=dict(at=now(),engineering_passed=True,P6_development_scope_passed=passed,P6_full_passed_in_declared_development_scope=passed,
        scope=result["scope"],all_tasks_percent=progress[0],experiment_percent=progress[1],report=identity(REPORT),
        result=identity(OUT/"result.json"),review=identity(LOG/"review.json"),cold=identity(LOG/"cold.json"),stopped=True,
        ledger=ledger(),child_ledger=child_ledger(),P5_scientific_adoption=False,P7_started=False,P10_still_stopped=True)
    once(OUT/"closeout.json",proof)
    status("completed_stopped",P6_full_passed=passed,P6_scope=result["scope"],all_tasks_percent=progress[0],experiment_percent=progress[1],report=relative(REPORT))
    supervise("P6_v43_completed_stopped",current_runner=None,waiting_for_user_next_part_confirmation=True,
        all_tasks_percent_planning_estimate=progress[0],experiment_percent_planning_estimate=progress[1],
        current_execution_evidence="results/P6/mixture_v43/closeout.json",summary_report=relative(REPORT),
        unchanged_progress_reason="P6 known-synthetic development all gates and full review/cold passed; original remaining2points closed" if passed else "P6 development gates failed; no added main-stage points")
    handoff(f"P6 v43后台部分及完整复验/冷重入已完成并停止。已知合成开发验收通过={passed}；范围不等于真实传感器、独立测试或抓取收益。全部任务总进度{progress[0]}%、实验总进度{progress[1]}%。报告{relative(REPORT)}。P5科学采用结论未变，不自动启动下一实验、P7/P10或周期任务。","complete")
    event("completed_stopped",P6_development_scope_passed=passed)
def deliver():
    global ORPHAN_UNCONFIRMED
    started=time.monotonic()
    c=verify()
    need(not child_ledger()["unknown"] and not ledger()["unknown"],"Unknown prior calls require process/evidence audit")
    if (OUT/"closeout.json").exists():
        p=read(OUT/"closeout.json");need(identity(REPORT)==p["report"],"Completed report differs")
        for field,path in (("result",OUT/"result.json"),("review",LOG/"review.json"),("cold",LOG/"cold.json")):need(identity(path)==p[field],"Completed receipt differs")
        reviewed=read(LOG/"review.json")["reviewed_outputs"]
        for name,value in reviewed.items():need(identity(ROOT/name)==value,"Completed artifact differs "+name)
        need({relative(x) for x in OUT.rglob("*") if x.is_file()}==set(reviewed)|{relative(OUT/"closeout.json")},"Completed output inventory differs")
        print(json.dumps(dict(already_complete=True,no_new_research=True)));return
    prior=c["preparation_seconds"]+sum(read(p)["seconds"] for p in LOG.glob("controller_usage_*.json"))
    usage=LOG/("controller_usage_"+uuid.uuid4().hex+".json")
    status("starting",controller_pid=os.getpid())
    def guard():
        seconds=time.monotonic()-started
        atomic(usage,dict(at=now(),seconds=seconds,status="running",prior_seconds=prior))
        need(prior+seconds<=c["controller_wall_seconds"],"Cumulative3600s budget")
        need(not (LOG/"STOP").exists(),"User STOP marker")
        need(storage()<=512*1024**2,"Total output cap")
    try:
        from scripts.pipeline import project_lock
        for stage in c["stages"]:
            guard()
            if (ROOT/stage["done"]).exists():
                event("existing_stage_receipt",stage=stage["name"]);continue
            with project_lock(ROOT):pass
            cid=uuid.uuid4().hex;command=[str(ROOT/stage["python"]),"-X","utf8","-B",stage["script"],stage["arg"]]
            event("child_intent",id=cid,stage=stage["name"],command=command)
            phase_start=time.monotonic()
            with (LOG/(stage["name"]+"_"+cid+".stdout.log")).open("xb") as out,(LOG/(stage["name"]+"_"+cid+".stderr.log")).open("xb") as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                event("child_started",id=cid,pid=child.pid,controller_pid=os.getpid(),stage=stage["name"])
                status(stage["name"],controller_pid=os.getpid(),child_pid=child.pid)
                try:
                    while child.poll() is None:
                        guard();need(time.monotonic()-phase_start<=stage["stage_wall_seconds"],"Stage time cap "+stage["name"])
                        need(process_peak(child)<=3*1024**3,"Child3GiB peak RSS cap");time.sleep(2)
                except BaseException:
                    stopped=subprocess.run(["taskkill","/PID",str(child.pid),"/T","/F"],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,check=False)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
                    ORPHAN_UNCONFIRMED=child.poll() is None
                    event("termination",id=cid,pid=child.pid,taskkill_returncode=stopped.returncode,exit_code=child.poll(),orphan_unconfirmed=ORPHAN_UNCONFIRMED)
                    raise
            event("child_returned",id=cid,stage=stage["name"],exit_code=child.returncode)
            need(child.returncode==0 and (ROOT/stage["done"]).exists(),"Child failed: "+stage["name"])
        guard();finish(c,prior+time.monotonic()-started,guard)
        atomic(usage,dict(at=now(),seconds=time.monotonic()-started,status="completed_stopped",prior_seconds=prior))
    except BaseException:
        atomic(usage,dict(at=now(),seconds=time.monotonic()-started,status="failed_preserved",prior_seconds=prior));raise
def failure(e):
    event("failure_preserved",error=repr(e),traceback=traceback.format_exc(),orphan_unconfirmed=ORPHAN_UNCONFIRMED)
    status("failed_stopped" if not ORPHAN_UNCONFIRMED else "orphan_unconfirmed",error=repr(e),unknown=ledger()["unknown"],unknown_children=child_ledger()["unknown"])
    supervise("P6_v43_failed_stopped",current_runner=None,waiting_for_user_next_part_confirmation=True,current_execution_evidence=relative(STATE))
    handoff("P6 v43后台链路出现工程/资源异常，实际状态及日志已保留。先核进程、锁和未知返回，禁止重复或扩预算启动。全部任务总进度45%、实验总进度56.25%；没有宣称P6通过，不进P7/P10。","failure")
def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("freeze","verify","deliver"));args=p.parse_args()
    if args.mode=="freeze":freeze();return
    if args.mode=="verify":print(json.dumps(dict(passed=True,files=len(verify()["protected"]))));return
    import msvcrt
    lock=ROOT/"runtime/pipeline/p6_mixture_v43.lock";lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("a+b") as f:
        f.seek(0,2)
        if f.tell()==0:f.write(b"0");f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:deliver()
        except BaseException as e:failure(e);raise
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
if __name__=="__main__":main()


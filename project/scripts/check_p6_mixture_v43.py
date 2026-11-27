"""Pre-freeze synthetic integration and source-bound test receipt, no real data."""
import sys,time,json,hashlib,copy,unittest,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.p6_mixture_common_v43 import LOG,sha,once,atomic,archive,now
import numpy as np
from scripts import p6_mixture_v43 as r
from src import uncertainty_mixture_v43 as c
SOURCES=["src/uncertainty_mixture_v43.py","tests/test_uncertainty_mixture_v43.py","scripts/p6_mixture_v43.py",
         "scripts/p6_mixture_common_v43.py","scripts/p6_mixture_delivery_v43.py","scripts/check_p6_mixture_v43.py"]
def main():
    start=time.perf_counter();LOG.mkdir(parents=True,exist_ok=True)
    prior_root_seconds=0.
    previous=LOG/"preparation_usage.json"
    if previous.exists():
        old=json.loads(previous.read_text(encoding="utf-8"))
        prior_root_seconds=float(old.get("root_tests_cumulative_seconds",old.get("root_current_tests_seconds",0)))
    for script in ("scripts/p6_mixture_v43.py","scripts/p6_mixture_delivery_v43.py"):
        help_run=subprocess.run([sys.executable,"-X","utf8","-B",script,"--help"],cwd=ROOT,capture_output=True,text=True)
        assert help_run.returncode==0,help_run.stderr
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_uncertainty_mixture_v43.py")
    with (LOG/("numeric_tests_"+str(time.time_ns())+".log")).open("x",encoding="utf-8") as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    assert result.wasSuccessful()
    h,w=40,56;y,x=np.indices((h,w));z=.7+.002*x+.001*y;z[:,w//2:]+=.2
    f=dict(depth=z.astype(np.float32),valid=np.ones((h,w),bool),eligible=np.ones((h,w),bool),
        rgb=np.repeat((x%7/7.)[:,:,None],3,axis=2),labels=(x>=w//2).astype(int)+1,
        K=np.array([[80.,0,50.],[0,80.,30.],[0,0,1.]]),original_size=(112,60))
    m=dict(family="same_source_zero_location_student_t",features=list(c.DEPTH_FEATURES),df=3.,
        center=[0.]*6,scale=[1.]*6,log_scale_coefficients=[float(np.log(.0035))]+[0.]*6,
        scale_bounds_m=list(c.SCALE_BOUNDS_M),calibration_multiplier=1.)
    prior=c.empirical_prior(np.array([int(np.flatnonzero(np.all(c.CANDIDATES==d,axis=1))[0]) for d in r.WARPS]))
    record=dict(scene=999,frame="toy");out=[];toys={}
    for mode in r.MODES:
        d=(3,-3);j=int(np.flatnonzero(np.all(c.CANDIDATES==d,axis=1))[0]);p=np.ones(81)*.1/80;p[j]=.9
        a=r.evaluate_case(f,record,d,mode,p,m,prior);json.dumps(a,allow_nan=False)
        z1,o1,s1=r.observation(f,record,d,mode);z2,o2,s2=r.independent_source(f,record,d,mode)
        assert np.array_equal(z1,z2) and np.array_equal(o1,o2) and np.array_equal(s1,s2)
        assert a["original_xyz_unchanged"] and a["geometry_covariance_finite"] and a["candidate_support_checked"]
        out.append(dict(mode=mode,source_n=a["oracle_visible"],missing_n=a["oracle_missing"],finite_json=True))
        toys[mode]=a
    rows=[dict(split="evaluation",scene=s,frame=str(frame),warp=list(d),mode=mode,**copy.deepcopy(toys[mode]))
        for s in (42,43,57,93) for frame in range(4) for d in r.WARPS for mode in r.MODES]
    summary,gates=r.aggregate(rows);json.dumps(dict(summary=summary,gates=gates),allow_nan=False)
    broken=copy.deepcopy(rows);broken[0]["B"]["method"]["main"]["zero_probability_event_count"]=1
    broken[0]["B"]["method"]["main"]["joint_nll"]=None;broken[0]["B"]["method"]["main"]["uncapped_joint_nll_is_infinite"]=True
    bs,bg=r.aggregate(broken)
    assert not bg["target_joint_nll_noninferior"] and bs["target_joint_nll_method"] is None
    query=r.fixed_query(f);am=r.strata(f,query,(3,0),"A");bm=r.strata(f,query,(3,0),"B")
    assert not np.array_equal(am["cross_instance"],bm["cross_instance"])
    checks=["A/B full synthetic scoring finite","independent warp/noise/missing source reconstruction",
        "exact original XYZ preservation","candidate support/missing denominators","aggregate432 synthetic duplicate fixtures",
        "zero probability event cannot silently drop from gate","A/B opposite unregistered source-stratum directions"]
    elapsed=time.perf_counter()-start
    archive(LOG/"component_tests.json","pre_freeze_test_revision")
    atomic(LOG/"component_tests.json",dict(at=now(),passed=True,numerical_tests=result.testsRun,integration_checks=len(checks),checks=checks,
        synthetic_scored_cases=3,synthetic_aggregate_fixtures=432,real_cases=0,neural_calls=0,official_AP=0,
        seconds=elapsed,source_sha256={p:sha(ROOT/p) for p in SOURCES},examples=out))
    # Prior executions are explicitly separate; process startup/preflight conservatively added by controller.
    geo=json.loads((LOG/"geometry/preparation_usage.json").read_text(encoding="utf-8")) if (LOG/"geometry/preparation_usage.json").exists() else {"seconds":2.340809}
    totals=prior_root_seconds+elapsed+5.3117019+float(geo.get("seconds",2.340809))+.385559+2.+3.
    archive(LOG/"preparation_usage.json","pre_freeze_test_revision")
    atomic(LOG/"preparation_usage.json",dict(at=now(),seconds=totals,root_current_tests_seconds=elapsed,root_tests_cumulative_seconds=prior_root_seconds+elapsed,
        earlier_probability_test_process_seconds=5.3117019,geometry_preparation_seconds=geo.get("seconds",2.340809),
        geometry_independent_arrays_seconds=.385559,root_failed_toy_conservative_seconds=2.,root_receipt_io_failure_conservative_seconds=3.,
        original_neural_calls=0,formal_research_calls=0,coding_and_readonly_review_wall_time_excluded=True))
    print(json.dumps(dict(passed=True,numeric_tests=result.testsRun,integration_checks=len(checks),seconds=elapsed,prep_total_seconds=totals)))
if __name__=="__main__":main()


"""Boundary checks on derivative tables; no model calls or physics steps."""
from paths import *
from statistics_expand import classify, paircalc, sumrow
from cluster_statistics import estimate

def check():
    data=[]
    for c in FACTORS[:2]:
        for s in ("a","b"):
            for m in METHODS:
                data.append(dict(unit=s+"-"+c,scene=s,condition=c,method=m,origin="new",trial_id=s+c+m,candidate="p0",
                                 pool_sha256="pool-"+s+c,state_sha256="state-"+s,
                                 success=(s=="a" and m=="A"),selected=True,static_admitted=True,physics=True,physical_steps=4728,
                                 failure_class="success" if s=="a" and m=="A" else "closure_failure",
                                 detail="success" if s=="a" and m=="A" else "closure_failure"))
    units={}
    for x in data:units.setdefault(x["unit"],{})[x["method"]]=x
    contrast=paircalc(units,"all","A","Raw")
    need(contrast["n"]==4 and contrast["scene_clusters"]==2 and contrast["first_only_success"]==2
         and contrast["second_only_success"]==0 and contrast["difference"]["value"]==.5
         and contrast["difference"]["ci95"]==[0.,1.],"paired units or base-scene clustering")
    undefined=estimate([{"scene":"a"},{"scene":"b"}],lambda r:0,lambda r:0)
    need(undefined["value"] is None and undefined["ci95"] is None,"zero denominator")
    single=estimate([{"scene":"a"},{"scene":"b"}],lambda r:int(r["scene"]=="a"),lambda r:int(r["scene"]=="a"))
    need(single["value"]==1 and single["ci95"] is None,"single informative cluster")
    unselected=dict(detail="empty_old_E",selected=False,physical_returns=0)
    need(classify(unselected,{"no_candidate":True})=="empty_old_E","empty E")
    stop=dict(detail="static_distance_uncertified",selected=True,full_static_path=False,grasp_success=False,physical_returns=0,physical_calls=0)
    need(classify(stop,{"no_candidate":False})=="static_distance_uncertified","uncertified mislabel")
    for wrong in ("unknown_return",None,"unregistered"):
        try:classify(dict(stop,detail=wrong),{"no_candidate":False})
        except RuntimeError:pass
        else:raise AssertionError("unknown static return accepted")
    dyn=dict(selected=True,full_static_path=True,dynamic_outcome_observed=True,grasp_success=False,physical_returns=3000)
    for d,category in (("finite_opposing_pad_clearance_hard_stop","dynamic_hard_gate"),("forbidden_collision","forbidden_collision"),
        ("timeout","timeout"),("slip","slip")):
        need(classify(dict(dyn,detail=d),{"no_candidate":False})==category,"dynamic category "+d)
    write("admission/reporting_boundary_tests.json",dict(passed=True,fixtures=[
        "two conditions share each base scene","exclusive paired wins and losses","zero denominator","one informative base scene",
        "empty E","static uncertified","three unknown static details rejected","finite finger clearance vs forbidden collision",
        "timeout","slip"],model_calls=0,physical_steps=0))
if __name__=="__main__":check()

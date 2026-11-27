"""Witness lines for the preregistered controls. Original V1 data are opened read-only."""
from common_v1 import *
import sys
import math
import re

def main():
    sys.path.insert(0,str(C/"p"/R6NAME/"code"));import codec
    reg=read(OUT/"preregistration/diagnostic_sample.json")
    orig=ROOT/R6NAME;out=[]
    for case in reg["cases"]:
        if case["group"]=="uncertified":continue
        td=orig/"trials"/case["trial"];m=read(td/"method_result.json")
        r=dict(supplement_id=case["supplement_trial_id"],original=case["trial"],kind=case["group"],
            candidate=case["candidate"],result_sha256=sha(td/"method_result.json"))
        if case["group"]=="success_control":
            ep=read(td/"episode/result.json");need(m["grasp_success"] and ep["grasp_success"] and ep["final_hold_streak"]>=480,"frozen positive")
            r.update(source="episode/result.json",source_sha256=sha(td/"episode/result.json"),final_hold_streak=ep["final_hold_streak"],max_lift_m=ep["max_lift_m"],physical_returns=m["physical_returns"])
        elif m["physical_returns"]:
            e=read(td/"episode/result.json");path=td/"episode/telemetry.jsonl.gz";last=None
            with gzip.open(path,"rb") as f:
                for line in f:
                    if b'"external_penetration_below_minus_1mm"' in line:
                        item=json.loads(line)
                        if item.get("event")=="state" and "external_penetration_below_minus_1mm" in item.get("violations",[]):last=item
            need(last is not None and m["forbidden_collision"],"original physical negative witness")
            pts=[p for p in last["raw_manifolds"] if p[1]!=p[2] and p[8]<-.001]
            need(pts,"no physical penetration witness")
            r.update(source=path.name,source_sha256=sha(path),step=last["index"],violations=last["violations"],native_records=pts)
        else:
            sr=read(td/"static_preflight/path_result.json");ff=sr["first_failure"];path=td/"static_preflight/dual_channel_snapshots.jsonl.zst"
            needle=b'{"snapshot":'+str(ff["snapshot"]).encode()+b','
            found=None
            with codec.open(path,"rb") as f:
                for line in f:
                    if line.startswith(needle):found=json.loads(line);break
            need(found is not None and found["violations"]==ff["violations"],"first failure snapshot")
            pts=[]
            for label in found["violations"]:
                v=label.split(":")
                if v[-1] not in case["source_witness_labels"] or len(v)!=4:continue
                tag,source,i,reason=v
                ch=found[tag+"_channel"];seq=ch["raw_manifolds"] if source=="manifold" else ch["raw_closest"][source]
                pt=seq[int(i)];need(len(pt)==14 and all(math.isfinite(float(x)) for x in [pt[8],pt[9],pt[10],pt[12],*pt[5],*pt[6],*pt[7]]),"native negative contact")
                if pt[8]<=0:pts.append(dict(channel=tag,source=source,index=int(i),label=reason,raw=pt))
            need(pts,"no genuine geometric contact in control witness")
            r.update(source=path.name,source_sha256=sha(path),snapshot=ff["snapshot"],phase=ff["phase"],native_records=pts)
        out.append(r)
    need(len(out)==20,"control witnesses")
    write("preregistration/original_control_witnesses.json",dict(utc=utc(),physics=0,controls=out))
if __name__=="__main__":main()

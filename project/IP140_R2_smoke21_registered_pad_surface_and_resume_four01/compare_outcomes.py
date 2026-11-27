"""Read-only old/new observed telemetry comparison. Never imports a simulator."""
import gzip,json,math,hashlib
from pathlib import Path
import numpy as np
from input_identity import HERE,ROOT,digest,write_once
OLD=ROOT/"IP140_R1_smoke12_slot07_base_table_mount_filter01"
def series(path,label):
    if not path.exists():return dict(physical_observed=False,metrics=None,reason="No new physical telemetry; admission did not authorize steps")
    rows=[];baseline=None;tipdepth=[];normalz=[]; effective=0;tcp={};last={}
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for raw in f:
            row=json.loads(raw)
            if row.get("event")!="state":continue
            n=row["index"];G=np.asarray(row["W_T_link_post"]["ip140_palm"]); inv=np.linalg.inv(G)
            e=row["object_evaluation"];obj=row["object_state"];cp=e.get("contacts",[])
            # Bind bodies from the initial configuration independently in outer pass.
            if n==240:baseline=float(obj["position"][2])
            pads=["ip140_left_pad","ip140_right_pad"]
            metrics=dict(step=n,phase=row["phase"],object_z_m=obj["position"][2],lift_m=None if baseline is None else obj["position"][2]-baseline,
                tcp_z_m=float(G[2,3]),object_relative_G_m=(inv@np.r_[obj["position"],1])[:3].tolist(),
                raw_table_support=e["table_support"],aggregate_table_support=e.get("aggregate_table_support"),table_normal_force_sum_N=0.,
                pad_force_on_object_world_N=[0.,0.,0.],pad_contacts=[])
            robot,objid,table,links=identity(path)
            net=np.zeros(3)
            for c in row["raw_manifolds"]:
                a,b,la,lb=c[1:5]
                roles={(a,la),(b,lb)}
                if {(objid,-1),(table,-1)}==roles:
                    metrics["table_normal_force_sum_N"]+=max(c[9],0.)
                if objid not in (a,b) or robot not in (a,b):continue
                rlink=la if a==robot else lb
                if rlink not in [links[s] for s in pads]:continue
                s=pads[[links[x] for x in pads].index(rlink)]
                sign=1. if a==objid else -1.
                normal=sign*np.asarray(c[7]);force=sign*(c[9]*np.asarray(c[7])+c[10]*np.asarray(c[11])+c[12]*np.asarray(c[13]))
                net+=force
                pt=np.asarray(c[5] if a==objid else c[6]);g=(inv@np.r_[pt,1])[:3]
                padlocal=(np.linalg.inv(np.asarray(row["W_T_link_post"][s]))@np.r_[pt,1])[:3]
                is_eff=c[9]>.2 and c[8]>=-.001
                if is_eff:
                    effective+=1;tipdepth.append(float(-g[0]));normalz.append(float(normal[2]))
                metrics["pad_contacts"].append(dict(link=s,raw=c,effective=is_eff,object_witness_G_m=g.tolist(),
                    object_witness_pad_link_m=padlocal.tolist(),depth_from_tip_m=float(-g[0]),
                    force_on_object_world_N=force.tolist(),normal_on_object_world=normal.tolist(),
                    force_times_dt_impulse_Ns=(force/240.).tolist(),
                    impulse_scope="force times frozen solver substep; not a hardware measurement"))
            metrics["pad_force_on_object_world_N"]=net.tolist()
            metrics["aggregate_table_support_recomputed"]=metrics["table_normal_force_sum_N"]>.2
            rows.append(metrics);tcp[n]=G;last[n]=metrics
    output=HERE/(label+"_contact_lift_timeseries.jsonl.gz")
    with gzip.open(output,"xt",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,separators=(",",":"),allow_nan=False)+"\n")
    z=[r["lift_m"] for r in rows if r["lift_m"] is not None];tail=[r for r in rows if 4249<=r["step"]<=4728]
    return dict(physical_observed=True,source=path.relative_to(ROOT).as_posix(),source_sha256=digest(path),returned_states=len(rows),
        metrics=dict(max_lift_m=max(z) if z else None,final_lift_m=z[-1] if z else None,
            effective_pad_contact_records=effective,mean_tip_depth_m=float(np.mean(tipdepth)) if tipdepth else None,
            normal_on_object_z_range=[min(normalz),max(normalz)] if normalz else None,
            tcp_lift_close_to_lift_end_m=float(tcp[4008][2,3]-tcp[3216][2,3]) if 4008 in tcp and 3216 in tcp else None,
            tail480_count=len(tail),tail480_legacy_support_steps=sum(r["raw_table_support"] for r in tail),
            tail480_aggregate_support_steps=sum(r["aggregate_table_support_recomputed"] for r in tail),
            tail480_table_sum_range_N=[min(r["table_normal_force_sum_N"] for r in tail),max(r["table_normal_force_sum_N"] for r in tail)] if tail else None),
        selected_steps={str(n):last[n] for n in (2788,2807,2832,3216,3217,3325,3326,3327,3328,3329,3330) if n in last},
        timeseries=output.name,timeseries_sha256=digest(output))
_IDENT={}
def identity(path):
    key=str(path)
    if key not in _IDENT:
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                r=json.loads(line)
                if r.get("event")=="initial_configuration":
                    c=r["evaluator_context"];_IDENT[key]=(r["body"],c["object_body_id"],c["table_body_id"],r["links"]);break
        if key not in _IDENT:raise ValueError("missing logged identity")
    return _IDENT[key]
def main():
    old=series(OLD/"episodes/07/telemetry.jsonl.gz","smoke12")
    new=series(HERE/"episodes/14/telemetry.jsonl.gz","smoke13")
    write_once(HERE/"smoke12_smoke13_comparison.json",dict(version=HERE.name,prior=old,new=new,
        calculation_scope="Logged engine contact forces, transformations and returned states only; no simulator; no absent values filled",
        causal_limit="Only one development bridge; no independent confirmation and no algorithm comparison",
        interpretation=None if not new["physical_observed"] else "See result and report; force directions are simulated observations"))
if __name__=="__main__":main()

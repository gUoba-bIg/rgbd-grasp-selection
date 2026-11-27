"""Read-only paired ranking analysis, using the frozen V1 two-pose NMS matching algorithm."""
from paths import *
from cluster_statistics import estimate
from rank_stability import load_unit, compare
from collections import Counter
import io

def csv_bytes(rows):
    output=io.StringIO(newline="")
    wr=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator="\n")
    wr.writeheader();wr.writerows(rows)
    return output.getvalue().encode("utf-8-sig")

def numeric_estimate(rows,key):
    return estimate(rows,lambda r: r[key] if r[key] is not None else 0.0,lambda r: int(r[key] is not None))

def run():
    idx=original_manifest()
    rel="robustness/rank_comparisons.csv"
    verify_old(rel,idx)
    with (PRIOR/rel).open(encoding="utf-8-sig",newline="") as f:
        archived={(r["scene"],r["condition"],r["method"]):r for r in csv.DictReader(f)}
    sample=read(OUT/"preregistration/sample80.json")
    ranks={r["unit_id"]:r["rank"] for rows in sample["selections"].values() for r in rows}
    plan=read(PRIOR/"preregistration/observation_plan.json")["jobs"]
    clean={};per_unit=[]
    for j in plan:
        c=j["condition"]
        if c not in FACTORS:continue
        scene=j["scene"];uid=j["unit_id"]
        if scene not in clean:clean[scene]=load_unit(STUDY/"units"/(scene+"-clean"))
        newer=load_unit(STUDY/"units"/uid)
        obs,_=compare(clean[scene],newer,scene,c)
        for r in obs:
            old=archived[(scene,c,r["method"])]
            for key,value in r.items():
                need(("" if value is None else str(value))==old[key],"historical rank-comparison mismatch "+uid+"/"+key)
            per_unit.append(dict(unit=uid,sample_rank=ranks.get(uid),**r))
    need(len(clean)==120 and len(per_unit)==1440 and sum(row["sample_rank"] is not None for row in per_unit)==240,"ranking denominator")
    exclusive_bytes(OUT/"statistics/unit_rank_stability.csv",csv_bytes(per_unit))
    summaries=[]
    for population in ("execution_sample","full_120_per_factor"):
        rows=[r for r in per_unit if population=="full_120_per_factor" or r["sample_rank"] is not None]
        for c in ("all",*FACTORS):
            for m in METHODS:
                group=[r for r in rows if r["method"]==m and (c=="all" or r["condition"]==c)]
                summaries.append(dict(population=population,condition=c,method=m,units=len(group),
                    distinct_scenes=len({r["scene"] for r in group}),
                    top1_counts=dict(Counter(r["top1"] for r in group)),
                    pre_E_top1_change=estimate(group,lambda r:int(r["top1"] in ("pose_changed","presence_changed")),lambda r:1),
                    post_E_top1_change=estimate(group,lambda r:int(r["locked"] in ("pose_changed","presence_changed")),lambda r:1),
                    top3_jaccard=numeric_estimate(group,"top3_jaccard"),kendall_tau_b=numeric_estimate(group,"score_tau_b")))
    write("statistics/rank_stability.json",dict(utc=utc(),source_sha256=idx[rel],sample_sha256=sha(OUT/"preregistration/sample80.json"),
        source_reconciled_rows=len(per_unit),full_scene_clustering=True,summaries=summaries))
    print("Verified 1440 V1 pre-E rank rows; 240 in the execution panel",flush=True)
if __name__=="__main__":run()

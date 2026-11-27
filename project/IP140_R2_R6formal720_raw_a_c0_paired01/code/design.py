from itertools import permutations
from math import sin,cos,pi
from pathlib import Path
import hashlib,random,json,difflib,inspect
from common import *

def code_copies():
    """Only names, registry/budget ownership, I/O codec and workspace guards differ."""
    imp=HERE/"implementation"; old=R4/"implementation";changes={};diff=[]
    for name in ("platform_imports.py","scene_world.py","platform_trace.py","platform_static.py","platform_trial.py"):
        base=(old/name).read_bytes();s=base.decode("utf8");t=s
        if name=="scene_world.py":
            fr=t.index("def registered_scene(");to=t.index("def bound_damping(",fr)
            t=t[:fr]+"from common import scene_binding as registered_scene, workspace_ok\n\n"+t[to:]
            t=t.replace("if ROOT.resolve()!=Path.cwd().resolve() or p.counts","if not workspace_ok() or p.counts")
        if name=="platform_trace.py":
            t=t.replace("import base64","import codec\nimport base64",1).replace("gzip.open(self.directory/'calls.jsonl.gz'","codec.open(self.directory/'calls.jsonl.zst'")
        if name=="platform_static.py":
            t=t.replace("import gzip,json","import codec\nimport gzip,json",1).replace("dual_channel_snapshots.jsonl.gz","dual_channel_snapshots.jsonl.zst").replace("gzip.open(record_path","codec.open(record_path")
        if name=="platform_trial.py":
            t=t.replace("import argparse","import codec\nfrom common import workspace_ok\nimport argparse",1)
            t=t.replace("if Path.cwd().resolve()!=ROOT or path.name","if not workspace_ok() or path.name").replace("sealed['method_budget']!=72 or len(sealed['files'])!=72","sealed['method_budget']!=720 or len(sealed['files'])!=720").replace("dual_channel_snapshots.jsonl.gz","dual_channel_snapshots.jsonl.zst").replace("with gzip.open(path,'rt'","with codec.open(path,'rt'")
        with (imp/name).open("xb") as f:f.write(t.encode("utf8"))
        changes[name]={"old_sha256":sha(old/name),"new_sha256":sha(imp/name),"production_tree_modified":False}
        diff.extend(difflib.unified_diff(s.splitlines(True),t.splitlines(True),fromfile="R4/"+name,tofile="R6/"+name))
    with (HERE/"audit/adapter.diff").open("x",encoding="utf8",newline="") as f:f.writelines(diff)
    return changes

def check_inputs():
    doc=read(R4/"audit/input_files.json");h={k:v["sha256"] for k,v in doc["files"].items()}
    baseline={line.split("  ",1)[1]:line.split("  ",1)[0] for line in (R4/"SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines() if line}
    for p in (R4/"implementation").glob("*.py"):
        expected=baseline[p.relative_to(R4).as_posix()];h[p.relative_to(ROOT).as_posix()]=expected
    for p in (PLATFORM).glob("*.py"):
        key=p.relative_to(ROOT).as_posix()
        if key not in h:
            # Engineering source manifest is authoritative for auxiliary direct imports.
            mf=PLATFORM/"SHA256SUMS.txt"
            if mf.exists():
                mani={ln.split("  ",1)[1]:ln.split("  ",1)[0] for ln in mf.read_text(encoding="utf-8-sig").splitlines() if "  " in ln}
                if p.name in mani:h[key]=mani[p.name]
    bad=[p for p,z in h.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=z]
    require(not bad,"frozen source/asset/input drift: "+repr(bad))
    require(h["runtime/build/graspnet/utils/loss_utils.py"]=="0a8dea7a09020f869c329781a6b2a713f8ba62d385d07dab43deaab55e007eb7","cross decoder digest")
    require(read(R5/"analysis/stratified_summary.json")["conclusion"]=="接口准入未通过","R5 seal")
    for other in [R4,R5]:
        sums=other/"SHA256SUMS.txt";expected={l.split("  ",1)[1]:l.split("  ",1)[0] for l in sums.read_text(encoding="utf-8-sig").splitlines() if l}
        for key in (["development_ledger.json","analysis/paired_statistics.json"] if other==R4 else ["analysis/stratified_summary.json","admission/seed_registry.json","admission/shadow_design.json"]):
            require(sha(other/key)==expected[key],"historical provenance")
    return h

def main():
    require(read(HERE/"audit/resource_preflight.json")["admitted"],"storage preflight")
    h=check_inputs()
    protected=put(HERE/"preregistration/frozen_inputs.json",h)
    r4=read(R4/"preregistration/paired_design.json")["units"]
    r5=read(R5/"admission/seed_registry.json")["scene_identity"]
    old_seeds=set(range(1,13))|set(v for arr in r5.values() for v in arr)
    oldpose={(u["object_id"],*u["xy_m"],*u["object_placement_quaternion_xyzw"]) for u in r4}
    oldpose.update((s["object_id"],*s["position_world"][:2],*s["xyzw"]) for s in read(R5/"admission/shadow_design.json")["scenes"])
    seed_root=hashlib.sha256((HERE.name+"|authorized-formal720|balanced-v1").encode()).hexdigest()
    def seed(kind,val):
        return int.from_bytes(hashlib.sha256((seed_root+"|"+kind+"|"+str(val)).encode()).digest()[:4],"big")
    rng=random.Random(seed("planner",0))
    cfg=read(ROOT/"configs/simulation_v52r2.json");assets=read(PLATFORM/"asset_typed_schema.json")["assets"]
    risk=read(ROOT/"configs/risk_calibration_v31.json")
    def noise(key,m):
        return int(hashlib.sha256(f"P5v23|20261023|{risk['quality_screen_epoch']}|{key}|0000|{m}".encode()).hexdigest()[:8],16)
    forbidden=set(old_seeds)
    for k in old_seeds:
        forbidden.update([risk["seed"]+k*1000,noise(k,"clean"),noise(k,"joint")])
    used=set(forbidden);scenes=[];units=[]
    for oid in ["014","017","021"]:
        for pos in range(1,5):
            desc=next(u for u in r4 if u["object_id"]==oid and u["position_index"]==pos)
            xy=desc["xy_m"];asset=cfg["objects"][oid];b=assets[asset["urdf"]]
            sectors=rng.sample(range(10),10)
            for i,bin_id in enumerate(sectors,1):
                sid=f"formal-{oid}-p{pos}-r{i:02}"
                while True:
                    key=rng.randrange(800000,1800000)
                    numbers=[key,risk["seed"]+key*1000,noise(key,"clean"),noise(key,"joint")]
                    if len(set(numbers))==4 and not used.intersection(numbers):break
                used.update(numbers);gen=random.Random(key)
                yaw=(-45+(bin_id+gen.random())*9)*pi/180.
                quat=[0.,0.,sin(yaw/2),cos(yaw/2)]
                pose=(oid,*xy,*quat);require(pose not in oldpose,"exposed placement");oldpose.add(pose)
                sc=dict(base_scene=sid,scene_index=key,object_id=oid,position_index=pos,repeat=i,
                    registry_row=desc["first_nominal_registry_slot"],asset=b,spawn_position=[*xy,cfg["table_z"]+asset["height_m"]/2+.003],
                    spawn_quaternion=quat,object_yaw_rad=yaw,yaw_stratum=bin_id,sampling_seed=numbers[1],
                    condition_seed=dict(clean=numbers[2],joint=numbers[3]))
                scenes.append(sc)
                for mode in ("clean","joint"):units.append(dict(unit_id=sid+"-"+mode,scene=sid,scene_key=key,condition=mode,object_id=oid,position=pos,repeat=i))
    rng.shuffle(units)
    perms=list(permutations(("Raw","A","C0")));methods=perms*40;rng.shuffle(methods);trials=[]
    for i,(u,perm) in enumerate(zip(units,methods,strict=True),1):
        u["unit_order"]=i;u["method_order"]=list(perm)
        for j,m in enumerate(perm,1):
            tid=f"R6FORMAL-{len(trials)+1:03}";k=seed("worker",tid)
            require(k not in used,"worker seed conflict");used.add(k)
            trials.append(dict(trial_id=tid,order=len(trials)+1,method=m,unit_id=u["unit_id"],scene=u["scene"],condition=u["condition"],worker_seed=k,within_unit=j))
    identity=dict(R4_full_archive_sha256="7279b99f3561091406b4e1dac44c5ab52bb7a8bb04c5672952ea145e42bc7bef",R4_seal=sha(R4/"SHA256SUMS.txt"),R4_ledger=sha(R4/"development_ledger.json"),
        R5_seal=sha(R5/"SHA256SUMS.txt"),R5_conclusion="接口准入未通过",R5_archive_sidecar=(ROOT/(R5.name+"_轻量审查包.zip.sha256")).read_text(encoding="utf8").split()[0])
    d=dict(version=HERE.name,seed_root=seed_root,created=now(),scope="formal only; R4 and R5 excluded from outcome aggregation",
        prior_protocol_deviation="R4 C0-A=0/24; original development expansion gate failed. Formal expansion separately authorized by user for complete experimental evaluation. No change to production scoring or eligibility.",
        conditions="Exact four registered XY anchors. For each object-position, ten seeded disjoint 9-degree azimuth strata in [-45,45) deg with fixed spawn z, zero roll/pitch. Sampled physical angle separates repeated scenes from R4 identity pose. No unseen-object claim. Fixed network seed and frozen algorithm constants are shared; repeated-scene, noise, point-sampling and worker seeds are separate.",
        n_scenes=120,n_units=240,n_trials=720,ablation=0,scenes=scenes,units=units,trials=trials,frozen_inputs_sha256=protected,history=identity,
        selection="P3+frozen collision/NMS/nonnegative domain, then original Raw/A/C0 over the same target-absent E. No R5 contactability, no fallback.",
        statistics="All 240 assigned units per method; primary C0-A. Use observed Bernoulli rates, descriptive 95% Wilson intervals, and paired stratified cluster bootstrap CIs: jointly resample ten repeated physical scenes within each of 12 object-position blocks, retain clean/joint and all three methods. 20000 replicates, percentile 2.5/97.5, fixed seed. Objects and board locations fixed, not a population generalization.",
        bootstrap_seed=seed("statistics",1),balanced_order="Each of six method permutations occurs exactly 40 times. Distinct method workspaces/clients, one shot per intent; no recycling method outcomes.",
        bootstrap_reps=20000)
    require(len(scenes)==120 and len(units)==240 and len(trials)==720,"design cardinality")
    code=code_copies();put(HERE/"audit/adapter_files.json",code)
    h=put(HERE/"preregistration/design.json",d)
    put(HERE/"preregistration/design_seal.json",dict(design_sha256=h,frozen_inputs_sha256=protected,created=now(),method_results_seen=0,method_trials_started=0))
    put(HERE/"run/progress.json",dict(planned=720,completed=0,started=0,stage="input_preparation",utc=now()))
    print("design",h,"120/240/720")
if __name__=="__main__":main()

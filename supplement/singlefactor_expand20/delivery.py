"""Close the evidence chain and build one shareable package; leave full native streams in place."""
from paths import *
from baseline import check
from ranking_analysis import csv_bytes
from collections import Counter
import difflib
import gzip
import stat
import zipfile

def regular_tree(path):
    if not path.is_dir():return
    for root,dirs,files in os.walk(path,followlinks=False):
        root=Path(root)
        dirs[:]=[d for d in dirs if d not in ("node_modules","__pycache__","tmp","visual_qa") and
                 not ((root/d).lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)]
        for name in sorted(files):
            f=root/name
            if f.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:continue
            yield f

def keep(name,path):
    if name.startswith(("src/","preregistration/","statistics/","figures/","admission/")):return True
    parts=Path(name).parts
    if name.startswith("study/run/jobs/"):return path.name in ("dispatch.json","return.json")
    if name.startswith("study/trials/"):
        return (len(parts)==4 and path.suffix==".json") or path.name in ("path_result.json","left_right_contact_entry_roots.json","final_480_steps.json") or (
            path.name=="result.json" and path.parent.name=="episode")
    return True

def main():
    book=read(OUT/"ledger240.json")
    checks=read(OUT/"admission/terminal_window_verification.json")
    health=disk();need(health["above_floor"] and book["trial_count"]==240 and checks["methods"]==240,"incomplete close")
    baseline=check();prior=read(OUT/"admission/v1_identity.json")
    need({k:v for k,v in prior.items() if k!="utc"}=={k:v for k,v in baseline.items() if k!="utc"},"post-run V1 source or model mismatch")
    idx=original_manifest()
    inherited=read(OUT/"admission/carried72_reconciliation.json")
    count=0
    for rec in inherited.values():
        for rel,h in rec["checked_original_file_hashes"].items():
            need(verify_old(rel,idx)==h,"original result evidence drift after dispatch")
            count+=1
    units=read(OUT/"preregistration/unit_input_hashes.json")
    actual=0
    for uid,u in units.items():
        for name,h in u["files"].items():
            rel="study/units/"+uid+"/"+name
            need(verify_old(rel,idx)==h,"unit data drift after dispatch "+uid);actual+=1
    new=read(OUT/"study/preregistration/physical_release.json")
    for f,h in new["code"].items():
        p=PREVIOUS_CODE/f.split("/",1)[1] if f.startswith("prior_wrapper/") else OUT/"src"/f
        need(sha(p)==h,"execution code drift after launch")
    write("admission/post_integrity.json",dict(utc=utc(),disk=health,V1=baseline,original_trial_files_reverified=count,selected_unit_files_reverified=actual,
         original_delivery_sha256=sha(PRIOR/"delivery/V1_supplement_robustness_admission01_review.zip"),
         code_same_since_release=True,shared_faults=[],trials=240))
    need(read(OUT/"admission/post_integrity.json")["original_delivery_sha256"]==OLD_ZIP_SHA,"previous delivery changed")
    sample=read(OUT/"preregistration/sample80.json")
    units_flat=[r for c in FACTORS for r in sample["selections"][c]]
    units_flat=[dict(**{k:v for k,v in r.items() if k!="method_order"},method_order=" / ".join(r["method_order"])) for r in units_flat]
    exclusive_bytes(OUT/"preregistration/sample80.csv",csv_bytes(units_flat))
    # Include the exact prior method-wrapper hunk, and all new orchestration and reporting sources.
    diff=(OUT/"admission/method_wrapper.diff").read_text("utf8")
    for f in sorted((OUT/"src").iterdir()):
        if f.is_file() and f.suffix in (".py",".mjs") and f.name!="method_trial_expand.py":
            diff+="".join(difflib.unified_diff([],f.read_text("utf8").splitlines(True),fromfile="/dev/null",tofile="src/"+f.name))
    exclusive_bytes(OUT/"delivery/code.diff",diff.encode())
    readme=("\n".join(["# V1 单因素扩充审查包","","主报告列出 80 个观测单元上的配对结果。工作簿提供可编辑论文表格，figures 中的 PDF 为矢量图。"
        ,"","preregistration/sample80.csv 给出 4 组的完整哈希顺序前 20 项与方法顺序。intents240.json 将这 80 个单元绑定到 240 份意图。"
        ,"","ledger240.json 对应 72 个沿用结果和 168 个新增结果。首次分派和返回回执逐一保存。statistics/paired_effects.json 保留每对方法逐单元的独有成败。"
        ,"","preregistration/unit_input_hashes.json 和 admission/carried72_reconciliation.json 分别登记共享输入和历史证据。新增试次的原始 API、路径快照与动力学遥测保存在同名本地版本的 study/trials 下。随包的 evidence_index.csv.gz 标有位置、大小、SHA256 和包内收录状态。"
        ,"","读取已有结果、复算聚类统计及生成表格和图均使用 src 中的独立包装。源码差异见 delivery/code.diff，V1 运行基线及原生引擎身份见 admission/v1_identity.json。","","SHA256SUMS.txt 校验包内各成员，ZIP 文件哈希另附。"]))+"\n"
    exclusive_bytes(OUT/"README.md",readme.encode())
    files=[]
    for name in ("src","preregistration","statistics","figures","admission","study/preregistration","study/run","study/trials"):
        files.extend(regular_tree(OUT/name))
    files.extend([OUT/"V1单因素扩充实验报告.md",OUT/"ledger240.json",OUT/"README.md",OUT/"delivery/code.diff"])
    unique={f.relative_to(OUT).as_posix():f for f in files}
    need(len(unique)==len(files),"duplicate local evidence")
    records=[]; entries={}
    for rel,f in sorted(unique.items()):
        h=sha(f);include=keep(rel,f);records.append((rel,f.stat().st_size,h,int(include),"this_version"))
        if include:entries[rel]=(f,h)
    # Carried raw evidence stays at its original location. Include individual native final windows for review.
    for tid,rec in sorted(inherited.items()):
        for rel,h in sorted(rec["checked_original_file_hashes"].items()):
            f=PRIOR/rel
            include=rel.endswith("/final_480_steps.json") or rel.endswith("/path_result.json") or rel.endswith("/episode/result.json")
            key="prior_evidence/"+rel
            records.append((rel,f.stat().st_size,h,int(include),"V1_supplement_robustness_admission01"))
            if include:entries[key]=(f,h)
    d=OUT/"delivery";ind=d/"evidence_index.csv.gz"
    with gzip.open(ind,"xt",encoding="utf8",newline="") as fp:
        w=csv.writer(fp);w.writerow(["relative_path","bytes","sha256","included_in_zip","root"]);w.writerows(records)
    entries["delivery/evidence_index.csv.gz"]=(ind,sha(ind))
    sums=d/"SHA256SUMS.txt"
    exclusive_bytes(sums,"".join(h+"  "+name+"\n" for name,(f,h) in sorted(entries.items())).encode())
    dest=d/(OUT.name+"_review.zip")
    with zipfile.ZipFile(dest,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
        for name,(f,h) in sorted(entries.items()):
            need(".." not in Path(name).parts and not Path(name).is_absolute() and ":" not in name,"unsafe name")
            z.write(f,name,compress_type=zipfile.ZIP_STORED if f.suffix in (".gz",".png",".xlsx",".pdf") else zipfile.ZIP_DEFLATED)
        z.write(sums,"SHA256SUMS.txt")
    with zipfile.ZipFile(dest) as z:
        need(len(z.namelist())==len(set(z.namelist()))==len(entries)+1,"bad package membership")
        for name,(f,h) in entries.items():
            with z.open(name) as fp:need(hashlib.file_digest(fp,"sha256").hexdigest()==h,"zip hash "+name)
        need(z.read("SHA256SUMS.txt")==sums.read_bytes(),"included manifest")
    digest=sha(dest);exclusive_bytes(d/(dest.name+".sha256"),(digest+"  "+dest.name+"\n").encode())
    final=dict(utc=utc(),disk=disk(),name=dest.name,bytes=dest.stat().st_size,sha256=digest,verified_zip_members=len(entries)+1,
               indexed_records=len(records),unique_new_version_file_bytes=sum(size for _,size,_,_,scope in records if scope=="this_version"),
               trial_count=book["trial_count"],inherited=72,new=168,additional_experiments=0)
    write("delivery/verification.json",final);print(json.dumps(final),flush=True)
if __name__=="__main__":main()

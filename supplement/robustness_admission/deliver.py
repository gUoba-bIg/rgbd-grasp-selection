"""Create the portable review volume and a full independent hash index for local scientific evidence."""
import csv
import gzip
import io
import stat
import sys
import zipfile
from common_v1 import *

def protected():
    hashes={}
    rows={}
    for group in ("source_R6_initial","source_R6_final","source_R7_final"):
        indexed=index(group);n=b=0
        for path,(size,h) in indexed.items():
            validate_file(ROOT/path,size,h);n+=1;b+=size
        rows[group]=dict(members=n,bytes=b,member_manifest_sha256=sha(WORK/"locks"/(group+".jsonl.gz")))
    for f,h in read(ROOT/R6FIRST/"preregistration/frozen_inputs.json").items():
        h=h if isinstance(h,str) else h["sha256"]
        need(sha(ROOT/f)==h,"protected input drift "+f)
    tag=subprocess.check_output(["git","--no-optional-locks","-C",str(WORK),"rev-parse",TAG+"^{commit}"],text=True).strip()
    need(tag==COMMIT,"protected base changed")
    source_repo=ARCHIVES.parent/"v1_baseline"
    original_tag=subprocess.check_output(["git","--no-optional-locks","-C",str(source_repo),"rev-parse",TAG+"^{commit}"],text=True).strip()
    baseline=read(OUT/"admission/v1_identity.json")
    weight_hashes={name:sha(ROOT/name) for name in baseline["weights"]}
    engine=sha(ROOT/"runtime/simulation_v52e/venv/Lib/site-packages/pybullet.cp312-win_amd64.pyd")
    need(tag==original_tag==baseline["commit"] and weight_hashes==baseline["weights"] and engine==ENGINE==baseline["engine_sha256"],"protected V1 source or runtime drift")
    return dict(utc=utc(),source_identity=sha(OUT/"admission/v1_identity.json"),members=rows,timestamp_scope="all originally sealed contents, checked again",
                tag_commit=tag,original_repository_tag_commit=original_tag,weights=weight_hashes,engine_sha256=engine,disk=disk())
def walk(base):
    # Do not enter model wheels, role environments, caches, or junctions.
    for root,dirs,names in os.walk(base,followlinks=False):
        r=Path(root)
        dirs[:]=sorted([n for n in dirs if not (r/n).is_junction() and not (r/n).is_symlink() and n not in
            ("__pycache__",".cache","plot_runtime","mpl","numba","cuda_cache","torch","tmp","delivery","user","mpl_rank","mpl_vision","mpl_physics")
            and not n.startswith("user_") and not n.endswith("_temp")])
        for name in sorted(names):
            f=r/name
            if f.is_file() and not f.is_symlink():yield f
def scientific():
    for root in ("admission","analysis","preregistration","figures","diagnostic","robustness","study","execution_logs","cold_logs","cold_logs_packaging02","cold_logs_packaging03"):
        if (OUT/root).is_dir():yield from walk(OUT/root)
    for leaf in ("c/cold_receipts","c/setup_logs","c/p/"+R6NAME+"/trials","c/p/"+R6NAME+"/units"):
        if (OUT/leaf).is_dir():yield from walk(OUT/leaf)
    for f in (OUT/"c/setup_complete.json",*OUT.glob("*.md")):
        if f.is_file():yield f
def include(rel,size):
    if rel.startswith(("analysis/","admission/","preregistration/","figures/","diagnostic/","robustness/","execution_logs/","cold_logs")):return True
    if "/" not in rel:return True
    if rel.startswith(("study/preregistration/","study/method_reporting_resume01/")) or rel in ("study/run/method_ledger.json","study/run/method_ledger_continued.json","study/STOP.json","study/method_STOP.json","study/run/forward_completed.json","study/run/rank_completed.json"):return True
    parts=rel.split("/")
    if parts[:3]==["study","run","jobs"] and len(parts)==5 and parts[-1] in ("dispatch.json","return.json"):return True
    if parts[:4]==["study","run","jobs","ADM-001"] and parts[-1] in ("stdout.log","stderr.log"):return True
    if len(parts)==4 and parts[:2]==["study","units"]:
        return parts[-1] in ("orders.json","candidate_domain_ledger.json","pool.npz","raw_decoder.npy","network.json","forward_intent.json","forward_done.json","rank_done.json","ranking_Raw.json","ranking_A.json","ranking_C0.json","selection.json")
    if rel.startswith("study/trials/"):
        return (len(parts)==4 and parts[-1] in ("result.json","intent.json","static_integrity.json","start.json","safety_receipt.json","dynamic_integrity.json","shared_or_unknown_stop.json","decoded_stop_fields.json","first_reporting_exception.json")) or parts[-1] in ("path_result.json","left_right_contact_entry_roots.json","final_480_steps.json")
    return rel.startswith("c/cold_receipts/") and (parts[-1].endswith("_return.json") or parts[-1] in ("complete.json","ledger.jsonl","model_invocations.jsonl"))
def main():
    disk()
    need(not (OUT/"delivery").exists(),"delivery exists")
    delivery=OUT/"delivery";delivery.mkdir()
    write("admission/protected_final.json",protected())
    code=WORK/"supplement"
    git=lambda *args:subprocess.run(["git","--no-optional-locks","-C",str(WORK),*args],capture_output=True,check=True)
    need(git("rev-parse","HEAD").stdout.decode().strip()==COMMIT,"unexpected branch tip")
    need(git("branch","--show-current").stdout.decode().strip()=="research/"+NAME,"wrong branch")
    git("add","--","supplement")
    diff=git("diff","--cached","--binary","--full-index",COMMIT,"--","supplement").stdout
    (delivery/"code.diff").write_bytes(diff)
    git("-c","user.name=SCI3 supplementary packaging","-c","user.email=supplement@localhost","-c","commit.gpgsign=false","commit","-m","Add V1-only supplemental robustness and admission wrappers")
    write("admission/code_identity.json",dict(utc=utc(),branch="research/"+NAME,base=COMMIT,commit=git("rev-parse","HEAD").stdout.decode().strip(),
          diff_sha256=sha(delivery/"code.diff"),files={p.name:sha(p) for p in sorted(code.glob("*")) if p.is_file()}))
    files=sorted(set(scientific()),key=lambda p:p.relative_to(OUT).as_posix())
    records=[]
    for f in files:
        rel=f.relative_to(OUT).as_posix();sz=f.stat().st_size;records.append((rel,sz,sha(f),include(rel,sz),f))
    localsum=delivery/"SHA256SUMS_local.txt"
    with localsum.open("x",encoding="utf8",newline="\n") as f:
        for name,_,h,_,_ in records:f.write(h+"  "+name+"\n")
    manifest=delivery/"evidence_index.csv.gz"
    with gzip.open(manifest,"xt",encoding="utf8",newline="") as out:
        w=csv.writer(out);w.writerow(["relative_to_output","bytes","sha256","included_in_core_zip"])
        w.writerows((n,s,h,inc) for n,s,h,inc,_ in records)
    entries={n:(f,h) for n,s,h,inc,f in records if inc}
    for f in sorted(code.glob("*")):
        if f.is_file():entries["source/"+f.name]=(f,sha(f))
    for f in (localsum,manifest,delivery/"code.diff"):entries["manifest/"+f.name]=(f,sha(f))
    internal="".join(h+"  "+n+"\n" for n,(f,h) in sorted(entries.items()))
    sums=delivery/"SHA256SUMS.txt";sums.write_text(internal,encoding="utf8",newline="\n")
    out=delivery/(NAME+"_review.zip")
    with zipfile.ZipFile(out,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for n,(f,h) in sorted(entries.items()):
            need(not n.startswith("/") and ".." not in Path(n).parts and ":" not in n,"unsafe archive path")
            z.write(f,n,compress_type=zipfile.ZIP_STORED if f.suffix in (".gz",".npz",".png",".pdf") else zipfile.ZIP_DEFLATED)
        z.write(sums,"SHA256SUMS.txt")
    with zipfile.ZipFile(out) as z:
        need(len(z.namelist())==len(set(z.namelist()))==len(entries)+1,"zip member count")
        for n,(f,h) in entries.items():
            with z.open(n) as fp:need(hashlib.file_digest(fp,"sha256").hexdigest()==h,"zipped evidence changed "+n)
        need(z.read("SHA256SUMS.txt")==sums.read_bytes(),"zip manifest changed")
    (delivery/(out.name+".sha256")).write_text(sha(out)+"  "+out.name+"\n",encoding="ascii")
    with (delivery/"verification.json").open("x",encoding="utf8") as f:
        json.dump(dict(utc=utc(),package_bytes=out.stat().st_size,package_sha256=sha(out),members_verified=len(entries)+1,local_index_records=len(records),local_index_sha256=sha(localsum),bytes_indexed=sum(r[1] for r in records),disk=disk()),f,indent=2,ensure_ascii=False)
if __name__=="__main__":main()

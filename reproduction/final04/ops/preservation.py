"""Check every historical R6/R7/paper/archive-source member without writing to the original trees."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,gzip,json,os,shutil,subprocess,sys
V=Path(__file__).resolve().parents[1]; Root=V.parent;Old=Root/'IP140_R3_V2_original_title_recovery01';base=Old/'v1_baseline'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(f):return json.loads(f.read_bytes())
def main(phase):
    initial=read(V/'audit/start.json');seen={};bad=[]
    for m in sorted((base/'locks').glob('*.receipt.json')):
        meta=read(m);k=m.name.split('.')[0]
        if not(k.startswith('source_') or k in ('execution_project','native_build_inputs')):continue
        lock=base/meta['member_manifest'];assert sha(lock)==meta['member_manifest_sha256']
        for rel,size,h in map(json.loads,gzip.open(lock,'rt',encoding='utf8')):
            target=(Path(os.environ['ProgramFiles'])/rel.removeprefix('vendor/ProgramFiles/')) if rel.startswith('vendor/ProgramFiles/') else Root/rel
            if target in seen:
                if seen[target]!=(size,h):bad.append([str(target),'conflicting prior locks'])
                continue
            seen[target]=(size,h)
            if not target.is_file() or target.stat().st_size!=size or sha(target)!=h:bad.append([str(target),'history mismatch'])
    gate=[]
    for item in initial['previous']:
        f=Root/item['path'];actual=sha(f);gate.append(dict(path=item['path'],actual=actual,match=actual==item['sha256']))
    control=0
    for l in (base/'metadata/baseline_files.sha256').read_text('utf-8-sig').splitlines():
        want,rel=l.split('  ',1);control+=1
        if sha(base/rel)!=want:bad.append([rel,'control identity'])
    def git(repo,*args):return subprocess.check_output([shutil.which('git'),'--no-optional-locks','-C',str(repo),*args],encoding='utf8').strip()
    tag=git(base,'rev-parse','paper_v1_r6r7_frozen')
    commit=git(base,'rev-parse','paper_v1_r6r7_frozen^{commit}')
    clean=not git(base,'status','--porcelain')
    volumes=[]
    for l in (Old/'external_checksums/V1_SHA256SUMS.txt').read_text('utf-8-sig').splitlines():
        want,rel=l.split('  ',1);f=Old/'v1_archive'/rel
        volumes.append(dict(name=rel,bytes=f.stat().st_size,sha256=sha(f),expected=want))
    out=dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),unique_prior_sources_checked=len(seen),frozen_control_files=control,historical_files_with_issues=bad,
        baseline=commit,tag=tag,clean_git=clean,old_stop_and_call_ledgers=gate,primary_archive_volumes=volumes)
    out['passed']=not bad and clean and initial['tag']==tag and initial['base']==commit and all(x['match'] for x in gate) and all(x['sha256']==x['expected'] for x in volumes)
    with (V/'audit'/('preservation_'+phase+'.json')).open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False)
    if not out['passed']:raise RuntimeError('protected sources changed')
if __name__=='__main__':main(sys.argv[1])

"""Verify every unique historical input against the before-copy inventory, without running any workflow."""
from seal_io import *
def measure():
    from collections import defaultdict
    counts=defaultdict(int);conflicts=[];external=[];unique={}
    for q in sorted((REPO/'locks').glob('*.receipt.json')):
        meta=read(q);name=q.name.split('.')[0]
        if not name.startswith('source_') and name not in ('execution_project','native_build_inputs','signed_display_driver'):continue
        with gzip.open(REPO/meta['member_manifest'],'rt',encoding='utf8') as f:
            for line in f:
                rel,length,digest=json.loads(line);counts[name]+=1
                if name=='signed_display_driver':
                    p=Path(os.environ['SystemRoot'])/'System32/DriverStore/FileRepository'/read(REPO/'metadata/display_driver.json')['folder']/rel
                elif rel.startswith('vendor/ProgramFiles/'):
                    p=Path(os.environ['ProgramFiles'])/rel[len('vendor/ProgramFiles/'):]
                else:p=ROOT/rel
                if p in unique:check(unique[p]==(length,digest),('inconsistent archived fingerprints',rel))
                unique[p]=(length,digest)
    for p,(n,h) in unique.items():
        try:
            actual=sha(p)
            if p.stat().st_size!=n or actual!=h:conflicts.append([str(p),n,h,p.stat().st_size,actual])
        except OSError as e:conflicts.append([str(p),str(e)])
    out=dict(verified_after_utc=utc(),sources_by_part=dict(counts),unique_sources=len(unique),mismatches=conflicts,passed=not conflicts,policy='read-only SHA256 of real source files including both language manuscripts, result manifests and locked runtime dependencies')
    put(V/'external_checksums/protected_before_after.json',out);check(not conflicts,'original file changed')
if __name__=='__main__':measure()

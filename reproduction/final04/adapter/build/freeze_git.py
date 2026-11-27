"""Create an actual new baseline exactly once, without backdating or force-updating refs."""
from seal_io import *
from protect import copy_checked
import subprocess,shutil
def run(*args):
    cmd=[shutil.which('git'),'-C',str(REPO),*args]
    return subprocess.run(cmd,check=True,capture_output=True,encoding='utf8').stdout.strip()
def main():
    check(not (REPO/'.git').exists(),'existing repository; inspect; never rewrite a tag')
    check(all((REPO/'locks'/(k+'.receipt.json')).exists() for k in ['source_supplement','signed_display_driver']),'new parts missing')
    files=sorted(walk(REPO),key=lambda p:p.relative_to(REPO).as_posix())
    with (REPO/'metadata/baseline_files.sha256').open('x',encoding='utf8',newline='\n') as f:
        for item in files:f.write(sha(item)+'  '+item.relative_to(REPO).as_posix()+'\n')
    print(run('init','-b','v1-archive'))
    for k,v in [('user.name','Local research archive'),('user.email','v1-freeze@localhost'),('core.autocrlf','false'),('core.hooksPath','hooks')]:run('config','--local',k,v)
    run('add','--all');run('update-index','--chmod=+x','hooks/reference-transaction')
    run('commit','-m','Freeze V1 R6, R7, bilingual manuscripts and offline reproduction materials')
    tag='paper_v1_r6r7_frozen';check(run('tag','--list',tag)=='','existing tag')
    run('tag','-a',tag,'-m','First local V1 archive; attached independent SHA-256 payloads')
    commit=run('rev-parse',tag+'^{commit}');tag_id=run('rev-parse',tag)
    bundle=ARCH/'paper_v1_r6r7_frozen.bundle';control=ARCH/'v1_control.zip'
    run('bundle','create',str(bundle),'--all');run('bundle','verify',str(bundle))
    run('archive','--format=zip','--output',str(control),tag)
    check(not run('status','--porcelain'),'baseline dirty after commit')
    new=[]
    for x in [bundle,control]+[ARCH/(n+'.zip') for n in ['source_supplement','signed_display_driver']]:
        new.append(copy_checked(x,V/'v1_archive_readonly'/x.name,sha(x)))
    put(V/'external_checksums/control_copy_verification.json',dict(files=new,checked_utc=utc()))
    put(V/'external_checksums/tag_identity.json',dict(tag=tag,tag_object=tag_id,commit=commit,created=run('show','-s','--format=%cI',commit),tagger=run('for-each-ref','--format=%(taggerdate:iso-strict)',f'refs/tags/{tag}'),upstream=read(REPO/'metadata/git_before.json'),bundle_sha256=sha(bundle),control_zip_sha256=sha(control),status=run('status','--porcelain=v1'),history_claim='new baseline, no reconstructed historical project commits'))
    files={read(p)['name']:read(p)['sha256'] for p in (REPO/'locks').glob('*.receipt.json')}
    files.update({x.name:sha(x) for x in [bundle,control]})
    with (V/'external_checksums/V1_SHA256SUMS.txt').open('x',encoding='utf8',newline='\n') as f:
        for k,h in sorted(files.items()):f.write(h+'  '+k+'\n')
    print(commit,tag_id,len(files),flush=True)
if __name__=='__main__':main()

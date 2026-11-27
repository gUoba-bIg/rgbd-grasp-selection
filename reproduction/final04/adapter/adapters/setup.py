"""Install only from sealed local files. The bootstrap interpreter itself is part of the archive."""
from pathlib import Path
import csv, gzip, hashlib, importlib.metadata, json, os, re, subprocess, sys, zipfile
BASE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def demand(v,reason):
    if not v:raise RuntimeError(str(reason))
def save(p,obj):
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')
def inventory(name):
    meta=read(BASE/'locks'/(name+'.receipt.json')); data=BASE/meta['member_manifest']
    demand(sha(data)==meta['member_manifest_sha256'],name+' index')
    with gzip.open(data,'rt',encoding='utf8') as f:rows=[json.loads(l) for l in f]
    return meta,rows
def verify_tree(root,rows):
    for rel,n,d in rows:
        p=(root/rel).resolve();demand(p.is_relative_to(root) and p.stat().st_size==n and sha(p)==d,rel)
def install(target,archives):
    demand(Path(sys.executable).resolve()==target/'bin/python.exe','unregistered bootstrap')
    _,rows=inventory('python312');verify_tree(target/'bin',rows)
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('PYTHON','PIP_','CONDA','VIRTUAL_ENV')):del env[k]
    temp=target/'tmp';temp.mkdir()
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_CONFIG_FILE=os.devnull,TEMP=str(temp),TMP=str(temp))
    env['PATH']=os.pathsep.join([str(target/'bin'),str(target/'bin/DLLs'),str(Path(env['SYSTEMROOT'])/'System32'),env['SYSTEMROOT']])
    logs=target/'setup_logs';logs.mkdir()
    with (logs/'install.log').open('xb') as stdout:
        def run(args):
            stdout.write((repr([str(x) for x in args])+'\n').encode());stdout.flush()
            r=subprocess.run([str(x) for x in args],env=env,cwd=target,stdout=stdout,stderr=subprocess.STDOUT)
            demand(r.returncode==0,('offline install',args,r.returncode))
        for key,folder in [('execution_project','p'),('offline_wheels','wh')]:
            meta,rows=inventory(key);arc=archives/meta['name'];demand(sha(arc)==meta['sha256'],arc)
            output=target/folder;output.mkdir()
            with zipfile.ZipFile(arc) as z:
                demand(set(z.namelist())=={l[0] for l in rows},key+' membership');z.extractall(output)
            verify_tree(output,rows)
        roles=read(BASE/'locks/wheel_identities.json')
        for role,items in roles.items():
            dest=target/'env'/role
            run([sys.executable,'-I','-B','-m','venv','--without-pip',dest])
            exe=dest/'Scripts/python.exe'
            run([exe,'-I','-B','-m','ensurepip','--default-pip'])
            run([exe,'-I','-B','-m','pip','--isolated','--disable-pip-version-check','install','--no-index','--no-cache-dir','--no-compile','--no-deps','--require-hashes','--find-links',target/'wh','-r',BASE/'locks'/('requirements_'+role+'.txt')])
            code="import sys,json,importlib.metadata as m;print(json.dumps({'exe':sys.executable,'prefix':sys.prefix,'base':sys.base_prefix,'packages':{d.metadata['Name']:d.version for d in m.distributions()},'paths':sys.path}))"
            data=json.loads(subprocess.check_output([str(exe),'-I','-B','-c',code],env=env,cwd=target))
            demand(Path(data['base']).resolve()==target/'bin' and Path(data['prefix']).resolve()==dest,'environment isolation '+role)
            norm=lambda k:re.sub(r'[-_.]+','-',k).lower()
            installed={norm(n):v for n,v in data['packages'].items()}
            demand(installed=={norm(e['name']):e['version'] for e in items},('version lock',role,installed))
            save(logs/(role+'.json'),data)
    save(target/'setup_complete.json',dict(isolated=True,environment=str(target),archive_directory=str(archives),python=sys.version,role_packages={k:len(v) for k,v in roles.items()}))
if __name__=='__main__':install(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())

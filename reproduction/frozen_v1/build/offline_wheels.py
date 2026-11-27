"""Use existing installer archives first; rebuild an explicitly labelled wheel from verified RECORD when needed."""
from seal_io import *
import importlib.metadata as md, email, re, csv, io, base64, sys, subprocess
from collections import defaultdict
norm=lambda n:re.sub(r'[-_.]+','-',n).lower()
envs={'vision':'runtime/envs/vision','rank':'runtime/envs/grasp_eval','physics':'runtime/simulation_v52e/venv'}
dists={e:sorted(md.distributions(path=[str(ROOT/p/'Lib/site-packages')]),key=lambda d:norm(d.metadata['Name'])) for e,p in envs.items()}
wanted={(norm(d.metadata['Name']),d.version) for ds in dists.values() for d in ds}
known={};sources={}
def tags_from(payload):
    tags=[line[4:].strip() for line in payload.splitlines() if line.startswith('Tag:')]
    if not tags:raise IndexError('no platform tag in installer')
    return next((x for x in tags if x.startswith(('cp312-','py3-'))),tags[0])
def wheel_meta(z):
    n=next(s for s in z.namelist() if s.endswith('.dist-info/METADATA'));m=email.message_from_bytes(z.read(n))
    tag=tags_from(z.read(n.rsplit('/',1)[0]+'/WHEEL').decode())
    return m['Name'],m['Version'],tag
directories=[ROOT/'runtime/wheels',ROOT/'runtime/simulation_v52e/wheels',ROOT/'runtime/downloads/grasp_eval_v7/archives',
    Path(sys.base_prefix)/'Lib/ensurepip/_bundled']
directories += [ROOT/base/'Lib/site-packages' for base in envs.values()]
for d in directories:
    for f in d.glob('*.whl'):
        if f.stat().st_size==0 and f.parent.name=='site-packages':
            continue  # Historical empty placeholders are not installer payloads.
        with zipfile.ZipFile(f) as z:n,v,tag=wheel_meta(z)
        k=norm(n),v
        if k in wanted and k not in known:known[k]=(f,f.name,'original_local_install_archive',str(f))
for base in [ROOT/'runtime/cache/pip',ROOT/'runtime/simulation_v52/cache',ROOT/'runtime/simulation_v52e/cache',Path(os.environ['LOCALAPPDATA'])/'pip/Cache']:
    if not base.is_dir():continue
    for f in sorted(base.rglob('*.body')):
        try:
            if not zipfile.is_zipfile(f):continue
            with zipfile.ZipFile(f) as z:n,v,t=wheel_meta(z)
        except (StopIteration,zipfile.BadZipFile,KeyError,IndexError):continue
        k=norm(n),v
        if k in wanted and k not in known:
            name=re.sub(r'[-.]+','_',n)+'-'+v+'-'+t+'.whl'
            known[k]=(f,name,'verified_local_wheel_cache',str(f))
scratch=V/'work'/('wheel_materialization_'+str(os.getpid()));scratch.mkdir(exist_ok=False)
def normrel(s):
    return s.replace('\\','/')
def from_install(dist,role):
    record=dist.read_text('RECORD');check(record,'missing RECORD '+str(dist.metadata['Name']))
    info=Path(dist._path).name;site=Path(dist._path).parent;name=dist.metadata['Name'];version=dist.version
    tag=tags_from(dist.read_text('WHEEL'))
    whlname=re.sub(r'[-.]+','_',name)+'-'+version+'-'+tag+'.whl';target=scratch/whlname
    output=[];checked=0
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for rel,h,size in csv.reader(io.StringIO(record)):
            rel=normrel(rel);f=Path(dist.locate_file(rel)).resolve()
            if not f.is_relative_to(site.resolve()) or rel.endswith(('.pyc','/RECORD','/REQUESTED','/INSTALLER','/direct_url.json')) or '__pycache__' in Path(rel).parts:continue
            check(f.is_file(),(name,rel))
            data=f.read_bytes();digest=hashlib.sha256(data).digest();signature='sha256='+base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
            check(h==signature and (not size or int(size)==len(data)),(name,rel,'installed RECORD mismatch'))
            z.writestr(rel,data);output.append([rel,signature,len(data)]);checked+=1
        output.append([info+'/RECORD','','']);s=io.StringIO();csv.writer(s,lineterminator='\n').writerows(output);z.writestr(info+'/RECORD',s.getvalue().encode())
    return (target,whlname,'installed_RECORD_verified_reconstructed_wheel',role+':'+info+'; verified '+str(checked)+' payload files; generated entry point launchers excluded')
files={};byrole={};cachedig={}
for e,ds in dists.items():
    entries=[]
    for d in ds:
        k=norm(d.metadata['Name']),d.version
        if k not in known:known[k]=from_install(d,e)
        path,name,origin,source=known[k];dig=cachedig.setdefault(name,None)
        if not dig:dig=sha(path);cachedig[name]=dig
        files[name]=(path,dig)
        # Compare module payload to original installation, independent of wheel name and metadata.
        installed_record=d.read_text('RECORD');scope=hashlib.sha256(installed_record.encode()).hexdigest()
        matches=0
        with zipfile.ZipFile(path) as zz:
            wheelrecord=next(s for s in zz.namelist() if s.endswith('.dist-info/RECORD'))
            wheelrows={normrel(s):sig for s,sig,_ in csv.reader(io.StringIO(zz.read(wheelrecord).decode()))}
            for rel,expect,_ in csv.reader(io.StringIO(installed_record)):
                rel=normrel(rel)
                if not expect or rel not in wheelrows or '.dist-info/' in rel:continue
                check(expect==wheelrows[rel],(e,d.metadata['Name'],rel,'wheel differs from installed RECORD'))
                matches+=1
        entries.append(dict(name=d.metadata['Name'],version=d.version,wheel=name,sha256=dig,origin=origin,
            source_receipt=source,installed_RECORD_text_sha256=scope,matching_installed_payload_entries=matches))
    byrole[e]=entries
    contents=''.join(d['name']+'=='+d['version']+' --hash=sha256:'+d['sha256']+'\n' for d in entries)
    (REPO/'locks'/('requirements_'+e+'.txt')).write_text(contents,encoding='utf8')
    # Snapshot only, in addition to a complete shipped and hashed installer lock.
    exe=ROOT/envs[e]/'Scripts/python.exe'
    fr=subprocess.run([str(exe),'-B','-m','pip','--disable-pip-version-check','freeze','--all'],capture_output=True,env={**os.environ,'PIP_NO_INDEX':'1','PYTHONDONTWRITEBYTECODE':'1'})
    check(fr.returncode==0,str(fr.stderr));(REPO/'metadata'/('pip_freeze_'+e+'.txt')).write_bytes(fr.stdout)
put(REPO/'locks/wheel_identities.json',byrole)
pack('offline_wheels',files)
print('envs', {k:len(v) for k,v in byrole.items()},'RECORD-reconstructed',[(k[0],k[1]) for k,v in known.items() if v[2].startswith('installed_')],flush=True)

"""Only ZIP directory metadata before freeze; exact members after freeze."""
import sys,io,re,zipfile,urllib.request,uuid,time,threading,json,shutil
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p10_common_v36 import *
REV='9073d1d4cb6c92272b6aa072545def5101e4fecb'
NLOG=LOG/'assets';NETLOCK=threading.RLock()
def netevent(kind,**kw):
 NLOG.mkdir(parents=True,exist_ok=True)
 with NETLOCK,(NLOG/'acquisition.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),kind=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())
def index(name):
 from src.remote_zip import RangeReader
 target=NLOG/'indexes'/(name+'.json')
 if target.exists():return load(target)
 class Reader(RangeReader):
  def range(self,start,length,cache=True):
   if not length:return b''
   need(0<length<32*1024**2,'Metadata range too large')
   for attempt in range(3):
    rows=[json.loads(x) for x in (NLOG/'acquisition.jsonl').read_text().splitlines()] if (NLOG/'acquisition.jsonl').exists() else [];need(sum(x['length'] for x in rows if x['kind']=='metadata_intent')+length<=64*1024**2,'Metadata 64MiB reservation limit')
    cid=uuid.uuid4().hex;netevent('metadata_intent',call_id=cid,archive=name,start=start,length=length,attempt=attempt)
    try:
     req=urllib.request.Request(self.resolved_url,headers={'Range':f'bytes={start}-{start+length-1}','Accept-Encoding':'identity','User-Agent':'P10FixedHeldout/36'})
     with urllib.request.urlopen(req,timeout=30) as r:
      m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''));need(r.status==206 and m and tuple(map(int,m.groups()[:2]))==(start,start+length-1),'Full or wrong HTTP response refused');total=int(m[3]);etag=r.headers.get('ETag');need(self.size is None or (self.size,self.etag)==(total,etag),'Archive changed');data=r.read(length+1);need(len(data)==length,'Truncated metadata');self.size,self.etag=total,etag;self.resolved_url=r.url;self.network_bytes+=len(data)
     netevent('metadata_returned',call_id=cid,bytes=len(data));return data
    except Exception as e:
     self.resolved_url=self.url;netevent('metadata_failure',call_id=cid,error=str(e).split('?')[0],unknown_transfer=True)
     if attempt==2:raise
     time.sleep(attempt+1)
 reader=Reader(f'https://hf-mirror.com/saic3d/graspnet/resolve/{REV}/{name}',timeout=30)
 with zipfile.ZipFile(reader) as z:entries={x.filename:dict(bytes=x.file_size,compressed=x.compress_size,crc32=x.CRC,offset=x.header_offset,method=x.compress_type) for x in z.infolist()}
 r=dict(at=now(),archive=name,url=reader.url,size=reader.size,etag=reader.etag,entries=entries,only_central_directory=True);target.parent.mkdir(parents=True,exist_ok=True);sealed(target,r);print('metadata',name,len(entries),flush=True);return r
def prepare_indexes():
 for name in ['test_seen.zip','test_similar.zip','test_novel.zip']:index(name)
def acquisition_config():
 need(not ACONF.exists(),'Assets already frozen');groups=read(ROOT/'data/manifests/scene_splits.json')['groups'];selected={};archives={}
 for group in ['test_seen','test_similar','test_novel']:
  ix=load(NLOG/'indexes'/(group+'.zip.json'));archives[group+'.zip']={k:ix[k] for k in ['url','size','etag']}
  for s in groups[group]:
   for tail in ['rgb/0064.png','depth/0064.png','meta/0064.mat','annotations/0064.xml','cam0_wrt_table.npy','camera_poses.npy']:
    source=f'scene_{s:04d}/realsense/'+tail;need(source in ix['entries'],'Missing exact fixed member '+source);selected['scenes/'+source]=dict(ix['entries'][source],archive=group+'.zip',source=source)
 ix=read(ROOT/'logs/P2P3/prerequisites_v26/indexes/models.zip.json');archives['models.zip']={k:ix[k] for k in ['url','size','etag']}
 for name,e in ix['entries'].items():
  if name.endswith(('/nontextured.ply','/textured.obj','/textured.sdf')) and int(name.split('/')[1])>=71:selected[name]=dict(e,archive='models.zip',source=name)
 c=dict(version=36,at=now(),selected=selected,archives=archives,limits=dict(bytes=6*1024**3,network_bytes=6*1024**3,seconds=7200,workers=4,chunk_bytes=16*1024**2));need(sum(x['bytes'] for x in selected.values())<c['limits']['bytes'],'Acquisition capacity');once(ACONF,c);return c
def fetch():
 c=verify();ac=read(ACONF);need(stat(ACONF)==c['asset_config'],'Asset scope changed')
 import scripts.p23_fetch_v26 as impl
 impl.CONFIG=ACONF;impl.LOG=NLOG;impl.DATA=DATA;impl.atomic=atomic;impl.events=netevent;impl.status=lambda phase,**kw:status('asset_acquisition',asset_state=phase,**kw)
 a=impl.Acquisition(ac)
 try:a.parallel(a.groups(list(ac['selected'].items())));a.save('finished')
 except Exception as e:a.save('failed_preserved');netevent('acquisition_failure',error=repr(e));raise
 files={}
 for name,e in ac['selected'].items():
  p=DATA/name;st=stat(p);need(st=={k:a.receipts[name][k] for k in ['sha256','bytes','mtime_ns']},'Member receipt differs');files[p.relative_to(ROOT).as_posix()]=st
 for oid in range(71):
  for tail in ['nontextured.ply','textured.obj','textured.sdf']:
   p=ROOT/f'data/raw/prerequisites_v26/models/{oid:03d}/{tail}';files[p.relative_to(ROOT).as_posix()]=stat(p)
 # Parse only after the scientific protocol is frozen. Test outcomes never select cases.
 import numpy as np
 from scipy.io import loadmat
 from PIL import Image
 for group,scenes in c['groups'].items():
  for s in scenes:
   sd=DATA/f'scenes/scene_{s:04d}/realsense';meta=loadmat(sd/'meta/0064.mat');K=meta['intrinsic_matrix'];need(K.shape==(3,3) and np.isfinite(K).all() and K[0,0]>0 and float(meta['factor_depth'].item())>0,'Invalid intrinsic data')
   with Image.open(sd/'rgb/0064.png') as im:need(im.size==(1280,720),'RGB size differs')
   with Image.open(sd/'depth/0064.png') as im:need(im.size==(1280,720),'Depth size differs')
 sealed(BASE/'assets.json',dict(at=now(),files=files,scene_count=90,frame=64,known_network_bytes=a.known_network,network_reserved_upper=a.reserved,unknown_network_calls=sorted(a.intents-a.returned),mirror_revision=REV,integrity='ZIP CRC32 plus local SHA256; third-party mirror, not an official archive hash proof'))
 print('All fixed test assets acquired and checked',len(files),flush=True)
if __name__=='__main__':
 from scripts.pipeline import project_lock
 with project_lock(ROOT):prepare_indexes() if sys.argv[1]=='index' else fetch()

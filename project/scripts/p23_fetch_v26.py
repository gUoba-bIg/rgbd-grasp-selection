"""Finite, resumable acquisition. Every committed member is CRC/SHA verified.

No whole archive fallback, no official test payload, no overwrite of committed
data, and no inference/training in this entry. Source archives remain untouched.
"""
import sys,struct,zlib,hashlib,re,urllib.request,concurrent.futures,traceback
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p23_common_v26 import *
from scripts.pipeline import project_lock
from src.remote_zip import safe_destination

class Acquisition:
 def __init__(self,c):
  self.c=c;self.lock=threading.RLock();self.start=time.monotonic();self.session=uuid.uuid4().hex;self.protocol_sha=sha(CONFIG);self.local=threading.local()
  self.usage_path=LOG/('fetch_usage_'+self.session+'.json');self.previous=sum(read(p)['seconds'] for p in LOG.glob('fetch_usage_*.json'))
  self.output=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file()) if DATA.exists() else 0
  self.reserved=0;self.known_network=0;self.intents=set();self.returned=set();self.receipts={};self.stop=False
  for p in [LOG/'acquisition.jsonl']:
   if p.exists():
    for line in p.read_text(encoding='utf-8').splitlines():
     r=json.loads(line)
     if r['kind']=='range_intent':self.reserved+=r['length'];self.intents.add(r['call_id'])
     elif r['kind']=='range_returned':self.known_network+=r['bytes'];self.returned.add(r['call_id'])
  p=LOG/'member_receipts.jsonl'
  if p.exists():
   for line in p.read_text(encoding='utf-8').splitlines():
    r=json.loads(line);need(r['path'] not in self.receipts,'Duplicate committed member receipt');self.receipts[r['path']]=r
  self.save('running')

 def save(self,state):
  atomic(self.usage_path,dict(at=now(),seconds=time.monotonic()-self.start,state=state))
  status(state,committed_files=len(self.receipts),planned_zip_files=len(self.c['selected']),committed_bytes=sum(r['bytes'] for r in self.receipts.values()),known_network_bytes=self.known_network,network_reserved_upper_bytes=self.reserved,unknown_network_calls=len(self.intents-self.returned),P7_started=False)

 def guard(self):
  need(not self.stop,'Another acquisition worker failed; stop and preserve partial files')
  need(self.previous+time.monotonic()-self.start<self.c['limits']['seconds'],'Cumulative acquisition time limit reached')
  need(self.output<self.c['limits']['bytes'],'Acquisition output budget reached')
  need(shutil.disk_usage(ROOT).free>=60*1024**3,'Disk reserve reached')

 def request(self,url,start,length,total,etag=None):
  need(0<length<=self.c['limits']['chunk_bytes'],'Unbounded HTTP range')
  # Reservation includes failed/unknown transfers; this is an upper bound, not a measured byte total.
  for attempt in range(3):
   with self.lock:
    self.guard();need(self.reserved+length<=self.c['limits']['network_bytes'],'Cumulative network reservation exhausted')
    cid=uuid.uuid4().hex;self.reserved+=length;self.intents.add(cid)
    events('range_intent',call_id=cid,url=url,start=start,length=length,attempt=attempt)
   try:
    if not hasattr(self.local,'resolved'):self.local.resolved={}
    req=urllib.request.Request(self.local.resolved.get(url,url),headers={'Range':f'bytes={start}-{start+length-1}','Accept-Encoding':'identity','User-Agent':'GraspNetPrerequisiteAudit/26'})
    with urllib.request.urlopen(req,timeout=60) as r:
     need(r.status==206 and r.headers.get('Content-Range')==f'bytes {start}-{start+length-1}/{total}','HTTP Range mismatch; whole download refused')
     if etag is not None:need(r.headers.get('ETag')==etag,'Pinned remote archive ETag changed')
     data=r.read(length+1);need(len(data)==length,'Incomplete range return');self.local.resolved[url]=r.url
    with self.lock:
     events('range_returned',call_id=cid,bytes=len(data));self.known_network+=len(data);self.returned.add(cid);self.save('P2_acquiring')
    return data
   except Exception as exc:
    self.local.resolved.pop(url,None)
    events('range_failure',call_id=cid,error_type=type(exc).__name__,error=str(exc).split('?')[0],partial_return_bytes_unknown=True)
    if attempt==2:raise
    time.sleep(1+attempt)

 def done(self,dest):
  p=safe_destination(DATA,dest);r=self.receipts.get(dest)
  if r:
   need(p.exists() and p.stat().st_size==r['bytes'] and p.stat().st_mtime_ns==r['mtime_ns'],'Committed member changed '+dest);return True
  need(not p.exists(),'Unreceipted final member; preserve and reconcile '+dest)
  return False

 def begin(self,dest):
  p=safe_destination(DATA,dest);p.parent.mkdir(parents=True,exist_ok=True)
  part=p.with_name(p.name+'.'+self.session+'.part');need(not part.exists(),'Duplicate pending member '+dest)
  return p,part

 def write(self,f,b):
  if not b:return
  with self.lock:
   self.guard();need(self.output+len(b)<=self.c['limits']['bytes'],'Output budget exceeded');self.output+=len(b)
  f.write(b)

 def commit(self,dest,p,part,digest,n,crc,provenance):
  need(not p.exists(),'Refuse overwrite '+dest);os.replace(part,p)
  r=dict(at=now(),path=dest,sha256=digest,bytes=n,crc32=crc,mtime_ns=p.stat().st_mtime_ns,provenance=provenance)
  with self.lock:
   with (LOG/'member_receipts.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
   self.receipts[dest]=r

 def zip_member(self,dest,e,head,getpayload):
  h=struct.unpack_from('<4s5H3I2H',head)
  need(h[0]==b'PK\x03\x04' and not h[2]&1 and h[3]==e['method'],'Invalid ZIP member header')
  need(head[30:30+h[-2]].decode('utf-8' if h[2]&2048 else 'cp437')==e['source'],'ZIP local/central name mismatch')
  skip=30+h[-2]+h[-1];need(skip<=len(head),'ZIP header exceeds acquired prefix');p,part=self.begin(dest);digest=hashlib.sha256();crc=n=0;dec=zlib.decompressobj(-15) if e['method']==8 else None
  need(e['method'] in [0,8],'Unsupported ZIP codec')
  def emit(b,f):
   nonlocal n,crc
   n+=len(b);need(n<=e['bytes'],'ZIP decompressed size exceeds manifest');digest.update(b);crc=zlib.crc32(b,crc);self.write(f,b)
  with part.open('xb') as f:
   for chunk in getpayload(skip):
    if dec:
     pending=chunk
     while pending:
      out=dec.decompress(pending,4*1024**2);pending=dec.unconsumed_tail;emit(out,f)
    else:emit(chunk,f)
   if dec:
    emit(dec.flush(),f);need(dec.eof and not dec.unused_data,'Incomplete/extra ZIP stream')
   need(n==e['bytes'] and crc&0xffffffff==e['crc32'],'ZIP size/CRC failed '+dest);f.flush();os.fsync(f.fileno())
  self.commit(dest,p,part,digest.hexdigest(),n,crc&0xffffffff,dict(archive=e['archive'],source=e['source'],frozen_config_sha256=self.protocol_sha))

 def fetch_group(self,group):
  name=group[0][1]['archive'];a=self.c['archives'][name];chunk=self.c['limits']['chunk_bytes']
  start=group[0][1]['offset'];end=max(e['offset']+e['compressed']+4096 for _,e in group);end=min(end,a['size'])
  if name.startswith('train_') and end-start<=chunk:
   raw=self.request(a['url'],start,end-start,a['size'],a['etag'])
   for dest,e in group:
    o=e['offset']-start
    self.zip_member(dest,e,raw[o:o+4096],lambda skip,e=e,o=o:[raw[o+skip:o+skip+e['compressed']]])
  else:
   need(len(group)==1,'Invalid oversized grouped range');dest,e=group[0]
   first=self.request(a['url'],start,30,a['size'],a['etag']);h=struct.unpack('<4s5H3I2H',first)
   tail=h[-2]+h[-1];header=first+self.request(a['url'],start+30,tail,a['size'],a['etag']) if tail else first
   def payload(skip):
    for off in range(0,e['compressed'],chunk):yield self.request(a['url'],start+skip+off,min(chunk,e['compressed']-off),a['size'],a['etag'])
   self.zip_member(dest,e,header,payload)

 def groups(self,items):
  grouped=[];batch=[];end=0
  for dest,e in sorted(items,key=lambda x:(x[1]['archive'],x[1]['offset'])):
   if self.done(dest):continue
   extent=e['offset']+e['compressed']+4096
   if batch and (not e['archive'].startswith('train_') or e['archive']!=batch[0][1]['archive'] or e['offset']-end>16384 or extent-batch[0][1]['offset']>self.c['limits']['chunk_bytes']):grouped.append(batch);batch=[]
   batch.append((dest,e));end=extent
  if batch:grouped.append(batch)
  return grouped

 def parallel(self,groups):
  with concurrent.futures.ThreadPoolExecutor(max_workers=self.c['limits']['workers']) as pool:
   pending={};it=iter(groups)
   for _ in range(self.c['limits']['workers']):
    g=next(it,None)
    if g:pending[pool.submit(self.fetch_group,g)]=g
   while pending:
    done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
    for future in done:
     try:future.result()
     except Exception:
      self.stop=True
      for q in pending:q.cancel()
      raise
     del pending[future];g=next(it,None)
     if g:pending[pool.submit(self.fetch_group,g)]=g

 def tolerance(self):
  ip=LOG/'tolerance_tar_index.json';url=self.c['tolerance_url'];total=self.c['tolerance_archive_bytes'];chunk=self.c['limits']['chunk_bytes']
  if ip.exists():entries=read(ip)['entries']
  else:
   entries=[];off=0
   while off+512<=total:
    b=self.request(url,off,512,total)
    if b==bytes(512):break
    check=int(b[148:156].strip(b'\0 ') or b'0',8);need(sum(b[:148])+8*32+sum(b[156:])==check,'TAR header checksum failed')
    name=b[:100].split(b'\0')[0].decode();size=int(b[124:136].strip(b'\0 ') or b'0',8);typ=b[156:157]
    need(typ in [b'0',b'\0',b'5'],'Unsupported TAR entry type; preserve evidence')
    if typ!=b'5':entries.append(dict(name=name,offset=off+512,bytes=size,header_sha256=hashlib.sha256(b).hexdigest()))
    off+=512+((size+511)//512)*512
   once(ip,dict(at=now(),source_url=url,archive_bytes=total,entries=entries))
  selected=[e for e in entries if re.fullmatch(r'tolerance/\d{3}_tolerance.npy',e['name']) and int(e['name'].split('/')[1][:3])<=70]
  need(len(selected)==71,'Tolerance object coverage differs')
  for e in selected:
   dest=e['name']
   if self.done(dest):continue
   p,part=self.begin(dest);digest=hashlib.sha256();crc=n=0
   with part.open('xb') as f:
    for off in range(0,e['bytes'],chunk):
     b=self.request(url,e['offset']+off,min(chunk,e['bytes']-off),total);digest.update(b);crc=zlib.crc32(b,crc);n+=len(b);self.write(f,b)
    f.flush();os.fsync(f.fileno())
   need(n==e['bytes'],'Tolerance size mismatch');self.commit(dest,p,part,digest.hexdigest(),n,crc&0xffffffff,dict(url=url,header_sha256=e['header_sha256'],official_algorithm_sample_verification_pending=True))

 def run(self):
  selected=self.c['selected'];inventories=[(p,e) for p,e in selected.items() if p.endswith('/object_id_list.txt')]
  self.parallel(self.groups(inventories))
  ids=set()
  for p,_ in inventories:ids.update(map(int,(DATA/p).read_text().split()))
  need(ids<=set(range(71)),'Training object IDs outside frozen asset scope')
  if not (LOG/'scene_object_inventory.json').exists():once(LOG/'scene_object_inventory.json',dict(at=now(),scenes=100,object_ids=sorted(ids),official_test_scenes_opened=0))
  # Make label and training-probe assets available first; no model training occurs here.
  labels=[(p,e) for p,e in selected.items() if e['archive'] in ['grasp_label.zip','collision_label.zip']]
  self.parallel(self.groups(labels));self.tolerance()
  remaining=[(p,e) for p,e in selected.items() if p not in self.receipts]
  self.parallel(self.groups(remaining));self.save('P2_acquisition_finished_audit_pending')
  once(LOG/'acquisition_complete.json',dict(at=now(),zip_members=len(selected),total_committed=len(self.receipts),known_network_bytes=self.known_network,unknown_network_calls=sorted(self.intents-self.returned),P2_accepted=False,P3_accepted=False,P7_started=False))

def main():
 c=verify();a=Acquisition(c)
 try:a.run()
 except Exception as exc:
  a.save('P2_acquisition_failed_preserved');events('acquisition_failure',error=repr(exc),traceback=traceback.format_exc());raise
 finally:a.save(read(ROOT/'runtime/pipeline/prerequisites_v26.json')['phase'])
if __name__=='__main__':
 with project_lock(ROOT):main()

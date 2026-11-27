"""Exercise CRC rejection, streamed expansion, range scope, and path containment."""
import sys,io,zipfile,unittest,hashlib,zlib,struct,uuid
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p23_fetch_v26 import Acquisition,safe_destination,ROOT,LOG,once,now

class Checks(unittest.TestCase):
 def test_stream_and_crc(self):
  payload=b'bounded-stream\0'*(1024*1024);b=io.BytesIO()
  with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('probe.bin',payload)
  raw=b.getvalue()
  with zipfile.ZipFile(io.BytesIO(raw)) as z:e=z.getinfo('probe.bin')
  h=struct.unpack_from('<4s5H3I2H',raw);skip=30+h[-2]+h[-1]
  record=dict(source='probe.bin',bytes=len(payload),compressed=e.compress_size,crc32=e.CRC,method=8,archive='local-test.zip')
  a=Acquisition.__new__(Acquisition);a.protocol_sha='unit';out=LOG/'implementation_checks'/uuid.uuid4().hex;out.mkdir(parents=True)
  a.begin=lambda d:(out/d,out/(d+'.part'));sizes=[]
  def write(f,b):sizes.append(len(b));f.write(b)
  a.write=write;got={};a.commit=lambda dest,p,part,digest,n,crc,prov:got.update(sha=digest,n=n,crc=crc)
  chunks=lambda s:(raw[i:min(i+257,skip+e.compress_size)] for i in range(skip,skip+e.compress_size,257))
  a.zip_member('ok.bin',record,raw[:skip],chunks)
  self.assertEqual(got['sha'],hashlib.sha256(payload).hexdigest());self.assertLessEqual(max(sizes),4*1024**2)
  with self.assertRaisesRegex(RuntimeError,'CRC'):a.zip_member('bad.bin',dict(record,crc32=0),raw[:skip],chunks)
 def test_no_cross_member_label_range(self):
  a=Acquisition.__new__(Acquisition);a.c={'limits':{'chunk_bytes':16*1024**2}};a.done=lambda d:False
  r=dict(archive='collision_label.zip',compressed=20,offset=0)
  self.assertEqual(len(a.groups([('a',r),('b',dict(r,offset=500))])),2)
 def test_path_scope(self):
  for name in ['../out','C:/outside','/outside','bad\\path']:
   with self.assertRaises(ValueError):safe_destination(ROOT/'data/raw/prerequisites_v26',name)

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 once(LOG/('implementation_check_'+uuid.uuid4().hex+'.json'),dict(at=now(),tests=result.testsRun,successful=result.wasSuccessful(),not_scientific_acceptance=True))
 sys.exit(0 if result.wasSuccessful() else 1)

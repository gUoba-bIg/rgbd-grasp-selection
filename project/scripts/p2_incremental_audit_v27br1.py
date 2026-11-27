"""Audit complete scene transactions while the remaining assets are downloading."""
import sys,json,time,concurrent.futures,uuid,traceback
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p2_audit_v27r1 import scene_audit,freeze,DATA,OUT
from scripts.p6_common_v24 import ROOT,read,once,atomic,stat,need,now
from scripts.multiview_common_v14r1 import lock
LOG=ROOT/'logs/P2P3/incremental_audit_v27br1'

def run():
 c=freeze();sources={p.relative_to(ROOT).as_posix():stat(p) for p in [Path(__file__),ROOT/'scripts/p2_audit_v27r1.py',ROOT/'src/grasp_prerequisite_data_v27.py']}
 fp=LOG/'freeze.json'
 if fp.exists():need(read(fp)['sources']==sources,'Incremental audit source changed')
 else:once(fp,dict(at=now(),sources=sources,max_seconds=7200,workers=4,only_committed_scene_members=True))
 start=time.monotonic();prior=sum(read(p)['seconds'] for p in LOG.glob('usage_*.json'));usage=LOG/('usage_'+uuid.uuid4().hex+'.json');pending={}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   while True:
    seconds=time.monotonic()-start;need(prior+seconds<7200,'Incremental audit time budget')
    text=(ROOT/'logs/P2P3/prerequisites_v26/member_receipts.jsonl').read_text(encoding='utf-8');lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    counts={i:[] for i in range(100)}
    for line in lines:
     r=json.loads(line)
     if r['path'].startswith('scenes/'):counts[int(r['path'].split('/')[1][6:])].append(r)
    for f in list(pending):
     if f.done():f.result();del pending[f]
    completed={i for i in range(100) if (OUT/'scenes'/f'{i:04d}.json').exists()}
    queued=set(pending.values())
    for i,rows in counts.items():
     if len(rows)==1284 and i not in completed|queued:
      for r in rows:
       p=DATA/r['path'];s=p.stat();need(s.st_size==r['bytes'] and s.st_mtime_ns==r['mtime_ns'],'Committed input changed')
      pending[pool.submit(scene_audit,i,c)]=i
    atomic(usage,dict(at=now(),seconds=seconds,state='running',scenes_completed=len(completed),queued=len(pending)))
    atomic(LOG/'status.json',dict(at=now(),scenes_completed=len(completed),frames_completed=len(completed)*256,scenes_queued=len(pending),P2_complete=False))
    if len(completed)==100:break
    state=read(ROOT/'runtime/pipeline/prerequisites_v26.json')['phase']
    if state=='P2_acquisition_failed_preserved' and not pending:raise RuntimeError('Acquisition stopped before all scene assets arrived')
    time.sleep(10)
  atomic(usage,dict(at=now(),seconds=time.monotonic()-start,state='complete',scenes_completed=100));once(LOG/'complete.json',dict(at=now(),scenes=100,frames=25600,global_hash_audit_pending=True))
 except Exception as exc:
  atomic(usage,dict(at=now(),seconds=time.monotonic()-start,state='failed_preserved'));once(LOG/('failure_'+uuid.uuid4().hex+'.json'),dict(at=now(),error=repr(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':
 with lock(ROOT/'runtime/pipeline/p2_incremental_v27br1.lock'):run()

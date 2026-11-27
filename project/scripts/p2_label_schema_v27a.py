"""Independent dense-label schema/bounds audit and frozen tolerance sample replay."""
import sys,json,time,uuid
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p6_common_v24 import ROOT,read,stat,sha,once,need,now
from scripts.p2_audit_v27 import tolerance_audit,DATA
LOG=ROOT/'logs/P2P3/label_schema_v27a';OUT=ROOT/'results/P2P3/label_schema_v27a'

def run():
 import numpy as np,zipfile
 from src.grasp_prerequisite_data_v27 import mapped_npz
 if (OUT/'result.json').exists():print('SKIP completed label schema audit');return
 start=time.monotonic();receipts={r['path']:r for r in [json.loads(s) for s in (ROOT/'logs/P2P3/prerequisites_v26/member_receipts.jsonl').read_text(encoding='utf-8').splitlines()]};ids=read(ROOT/'logs/P2P3/prerequisites_v26/scene_object_inventory.json')['object_ids'];sources={p.relative_to(ROOT).as_posix():stat(p) for p in [Path(__file__),ROOT/'scripts/p2_audit_v27.py',ROOT/'src/grasp_prerequisite_data_v27.py']}
 once(LOG/'freeze.json',dict(at=now(),sources=sources,objects=ids,scenes=list(range(100)),maximum_seconds=1800,score_bounds=[-1,1.2],tolerance_bounds=[0,.0500001]))
 def check(path):
  rel=path.relative_to(DATA).as_posix();r=receipts.get(rel);need(r is not None and stat(path)=={k:r[k] for k in ['sha256','bytes','mtime_ns']},'Label member not committed/changed '+rel);need(time.monotonic()-start<1800,'Label audit time budget')
 point_counts={};objects=[]
 for obj in ids:
  p=DATA/f'grasp_label/{obj:03d}_labels.npz';t=DATA/f'tolerance/{obj:03d}_tolerance.npy';check(p);check(t)
  pts=mapped_npz(str(p),'points');off=mapped_npz(str(p),'offsets');score=mapped_npz(str(p),'scores');coll=mapped_npz(str(p),'collision');tol=np.load(t,mmap_mode='r');n=len(pts);point_counts[obj]=n
  need(pts.shape==(n,3) and off.shape==(n,300,12,4,3) and score.shape==coll.shape==tol.shape==(n,300,12,4),'Label shape mismatch')
  need(coll.dtype==np.bool_ and all(a.dtype==np.float32 for a in [pts,off,score,tol]),'Dense label dtype mismatch');need(np.isfinite(pts).all(),'Invalid object anchor points')
  for i in range(0,n,32):
   s=score[i:i+32];o=off[i:i+32];tt=tol[i:i+32];need(np.isfinite(s).all() and np.min(s)>=-1 and np.max(s)<=1.2,'Invalid friction label');need(np.isfinite(o).all(),'Invalid grasp offset');need(np.isfinite(tt).all() and np.min(tt)>=0 and np.max(tt)<=.0500001,'Invalid tolerance')
  objects.append(dict(object_id=obj,points=n,all_numeric_elements_checked=True))
 scenes=[]
 for scene in range(100):
  p=DATA/f'collision_label/scene_{scene:04d}/collision_labels.npz';check(p);order=list(map(int,(DATA/f'scenes/scene_{scene:04d}/object_id_list.txt').read_text().split()))
  with zipfile.ZipFile(p) as z:need(set(z.namelist())=={f'arr_{i}.npy' for i in range(len(order))},'Scene collision member layout differs')
  for i,obj in enumerate(order):
   arr=mapped_npz(str(p),f'arr_{i}');need(arr.dtype==np.bool_ and arr.shape==(point_counts[obj],300,12,4),'Scene collision tensor mismatch')
  scenes.append(dict(scene=scene,objects=len(order),object_order=order))
 tolerance=tolerance_audit(ids)
 for p,s in sources.items():need(stat(ROOT/p)==s,'Audit source changed')
 once(OUT/'result.json',dict(at=now(),status='passed',objects=objects,scenes=scenes,tolerance=tolerance,seconds=time.monotonic()-start,training_updates=0,model_forwards=0));print(json.dumps(dict(status='passed',objects=len(ids),scenes=100,seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':run()

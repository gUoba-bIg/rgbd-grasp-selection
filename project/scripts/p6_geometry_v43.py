"""Authorized four-frame CPU reference rendering and zero-rerender review."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[_k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
import sys
sys.dont_write_bytecode=True
import argparse
import json
import time
import uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src import p6_geometry_reference_v43 as core
from scripts import p6_correspondence_v40 as common
from scripts.pipeline import project_lock

LOG=ROOT/'logs/P6/mixture_v43/geometry'
OUT=ROOT/'results/P6/mixture_v43/geometry'
AUTH=ROOT/'logs/P6/mixture_v43/authorization.json'
SOURCES=['src/p6_geometry_reference_v43.py','scripts/p6_geometry_v43.py','scripts/p6_correspondence_v40.py','src/correspondence_v40.py','scripts/pipeline.py']
read,once,replace_json,append,now=common.read,common.once,common.replace_json,common.append,common.now


def runtime():
    import importlib.metadata as metadata
    return dict(python=sys.version.split()[0],executable=str(Path(sys.executable).resolve()),
        packages={k:metadata.version(k) for k in ('numpy','scipy','Pillow','open3d')},threads=2,GPU=False)


def ledger():
    starts={}; returns={}; keys=set(); path=LOG/'render_calls.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            e=json.loads(line)
            if e['event']=='started':
                core.need(e['id'] not in starts and e['key'] not in keys,'Duplicate rendering identity')
                starts[e['id']]=e; keys.add(e['key'])
            else:
                core.need(e['event']=='returned' and e['id'] in starts and e['id'] not in returns,'Unpaired rendering return')
                core.need(e['key']==starts[e['id']]['key'],'Changed rendering return key'); returns[e['id']]=e
    return dict(started=len(starts),returned=len(returns),unknown=[e for i,e in starts.items() if i not in returns],
        returned_by_key={e['key']:e for e in returns.values()})


class Resources:
    def __init__(self,stage):
        self.stage=stage; self.start=time.perf_counter(); self.last=-1e9
        prior=read(LOG/'usage.json') if (LOG/'usage.json').exists() else {}
        self.seconds=float(prior.get('seconds',0)); self.rss=int(prior.get('peak_rss_bytes',0))
        self.files=set(p for folder in (LOG,OUT) for p in folder.rglob('*') if p.is_file())
        self.check(True)
    def register(self,path): self.files.add(path)
    def check(self,force=False):
        elapsed=time.perf_counter()-self.start
        if not force and elapsed-self.last<2: return
        self.last=elapsed; self.rss=max(self.rss,common.peak_rss()); size=sum(p.stat().st_size for p in self.files if p.exists())
        l=ledger()
        value=dict(seconds=self.seconds+elapsed,peak_rss_bytes=self.rss,output_bytes=size,stage=self.stage,at=now(),
            real_render_started=l['started'],real_render_returned=l['returned'],unknown_calls=l['unknown'],neural_calls=0,
            note='Root external guard monitors blocking CPU calls; internal checks at least per completed operation, sampled otherwise every2s.')
        replace_json(LOG/'usage.json',value); self.register(LOG/'usage.json')
        core.need(value['seconds']<=core.LIMITS['seconds'] and self.rss<=core.LIMITS['rss_bytes'] and size<=core.LIMITS['output_bytes'],'Geometry time/RSS/output limit exceeded')


def verify_authorization():
    a=read(AUTH); core.need(a.get('frozen') is True,'Root geometry authorization missing')
    for name in SOURCES: core.need(name in a['protected'],'Source absent from root freeze: '+name)
    common.verify_identities(a['protected'])
    inv=core.inventory()
    for name,info in inv['protected'].items():
        core.need(name in a['protected'] and a['protected'][name]['sha256']==info['sha256'],'Reference asset absent/different from root freeze: '+name)
    tests=read(LOG/'component_tests.json')
    core.need(tests['passed'] and tests['runtime']==runtime(),'Geometry tests/runtime identity mismatch')
    for name,digest in tests['source_sha256'].items(): core.need(core.sha(ROOT/name)==digest,'Geometry tested source changed')
    return inv


def load_artifact(key):
    p=OUT/(key+'.npz'); m=read(OUT/(key+'.json'))
    core.need(core.sha(p)==m['npz_sha256'],'Geometry artifact content changed')
    with np.load(p,allow_pickle=False) as data: arrays={k:data[k].copy() for k in data.files}
    return arrays,m


def run(inv,resources):
    l=ledger(); core.need(not l['unknown'],'Unknown prior rendering retained; no automatic rerender')
    frames=[]
    for scene,frame in core.REFERENCE_KEYS:
        key=f'scene_{scene:04d}_{frame}'; p=OUT/(key+'.npz'); mp=OUT/(key+'.json')
        l=ledger()
        if key in l['returned_by_key']:
            core.need(p.exists() and mp.exists(),'Returned rendering artifacts missing')
            _,meta=load_artifact(key); core.need(core.sha(mp)==l['returned_by_key'][key]['metadata_sha256'],'Returned render metadata changed')
        else:
            core.need(l['started']<4 and not p.exists() and not mp.exists(),'Rendering budget exceeded or uncommitted artifact exists')
            cid=uuid.uuid4().hex; append(LOG/'render_calls.jsonl',dict(event='started',id=cid,key=key,at=now())); resources.register(LOG/'render_calls.jsonl')
            arrays,meta=core.render_reference(scene,frame)
            metrics,gates=core.reference_metrics(arrays)
            with p.open('xb') as f: np.savez_compressed(f,**arrays)
            meta.update(metrics=metrics,quality_gates=gates,geometry_trusted=all(gates.values()),npz_sha256=core.sha(p),runtime=runtime(),
                primary_truth='Known synthetic source-ID; this mesh reference is auxiliary.',no_prediction_access=True)
            once(mp,meta); resources.register(p); resources.register(mp)
            append(LOG/'render_calls.jsonl',dict(event='returned',id=cid,key=key,at=now(),npz_sha256=meta['npz_sha256'],metadata_sha256=core.sha(mp)))
        frames.append(dict(scene=scene,frame=frame,artifact=core.relative(p),metadata=core.relative(mp),metrics=meta['metrics'],quality_gates=meta['quality_gates'],geometry_trusted=meta['geometry_trusted']))
        resources.check(True)
    l=ledger(); core.need(l['returned']==4 and not l['unknown'],'Incomplete geometry rendering ledger')
    result=dict(version='v43',component='geometry_reference',engineering_computation_completed=True,independent_review_pending=True,
        geometry_trusted=all(r['geometry_trusted'] for r in frames),trusted_frame_count=sum(r['geometry_trusted'] for r in frames),frames=frames,
        quality_thresholds=core.QUALITY_THRESHOLDS,real_render_calls=4,new_neural_calls=0,input_hashes=inv['protected'],
        source_hashes={k:core.sha(ROOT/k) for k in SOURCES},runtime=runtime(),P6_full_passed=False,
        scope='Four exposed training frames only; mesh/pose reference is not absolute sensor truth. Untrusted mesh labels must not supervise the probability model; exact synthetic source-ID validation may continue.',
        candidate_support_checked=False,candidate_support_scope='This component supplies a pure evaluation helper. Actual fixed candidate family/support is checked by the probability component; rendering alone does not establish candidate coverage.')
    target=OUT/'result.json'
    if target.exists(): core.need(read(target)==result,'Geometry result differs on reentry')
    else: once(target,result)
    resources.register(target)


def review(inv,resources):
    result=read(OUT/'result.json'); l=ledger()
    core.need(l['returned']==4 and not l['unknown'],'Geometry review ledger incomplete')
    common.verify_identities(result['input_hashes'])
    reviews=[]; reviewed_outputs={core.relative(OUT/'result.json'):core.identity(OUT/'result.json')}
    for r in result['frames']:
        key=f"scene_{r['scene']:04d}_{r['frame']}"; arrays,meta=load_artifact(key)
        core.need(core.sha(OUT/(key+'.json'))==l['returned_by_key'][key]['metadata_sha256'],'Render-return metadata mismatch')
        audit=core.audit_cached(arrays,meta)
        common.close_numeric(audit['metrics'],r['metrics']); common.close_numeric(audit['quality_gates'],r['quality_gates'])
        core.need(audit['geometry_trusted']==r['geometry_trusted'],'Geometry quality conclusion mismatch')
        reviews.append(dict(scene=r['scene'],frame=r['frame'],**audit))
        for extension in ('.npz','.json'): reviewed_outputs[core.relative(OUT/(key+extension))]=core.identity(OUT/(key+extension))
        resources.check(True)
    core.need(result['geometry_trusted']==all(r['geometry_trusted'] for r in reviews),'Overall reference trust mismatch')
    record=dict(passed=True,engineering_passed=True,geometry_trusted=result['geometry_trusted'],P6_full_passed=False,
        frames=reviews,reviewed_outputs=reviewed_outputs,new_render_calls=0,real_render_calls_total=4,unknown_calls=[],at=now())
    target=LOG/'review.json'
    if target.exists():
        prior=read(target); record['at']=prior['at']; core.need(prior==record,'Geometry review differs')
    else: once(target,record)
    resources.register(target)


def toy():
    start=time.perf_counter()
    def square(a,z):
        return dict(vertices=np.array([[-a,-a,z],[a,-a,z],[a,a,z],[-a,a,z]]),triangles=np.array([[0,1,2],[0,2,3]]),pose=np.column_stack((np.eye(3),np.zeros(3))))
    meshes=[dict(object_id=1,**square(.2,1.)),dict(object_id=2,**square(.04,.5))]
    K=np.array([[100.,0,31.5],[0,100.,23.5],[0,0,1.]])
    arrays,metadata=core.cast_scene(meshes,K,(64,48),(32,24))
    rays=core.camera_rays(K,(64,48),(32,24)); d=rays[...,3:]
    far=(np.abs(d[...,0])<.2)&(np.abs(d[...,1])<.2)
    near=(np.abs(d[...,0]*.5)<.04)&(np.abs(d[...,1]*.5)<.04)
    expected=np.where(near,.5,np.where(far,1.,0.)); ids=np.where(near,2,np.where(far,1,0))
    core.need(np.allclose(arrays['reference_depth_m'],expected,atol=1e-6) and np.array_equal(arrays['reference_object_id'],ids),'Synthetic first-visible surface/depth convention failed')
    core.need(near.any() and (far&~near).any() and (~far&~near).any(),'Toy lacks occlusion/far/missing cases')
    arrays.update(observed_depth_m=arrays['reference_depth_m'].copy(),label_object_id=arrays['reference_object_id'].copy(),K=K,original_size=np.array([64,48]),factor_depth=np.array(1000.))
    check=core.audit_cached(arrays,metadata); core.need(check['passed'] and check['geometry_trusted'],'Perfect synthetic reference failed audit')
    arrays['observed_depth_m']=np.where(arrays['reference_valid'],arrays['reference_depth_m']+.1,0)
    _,gates=core.reference_metrics(arrays); core.need(not gates['interior_depth_median'] and not gates['interior_depth_p95'],'Bad depth reference incorrectly trusted')
    support=core.candidate_support(np.array([[0,0,1],[0,0,2.]]),np.array([[[0,0,1]],[[0,0,1.]]]),np.array([[True],[False]]),np.array([True,True]),.01)
    core.need(support['eligible_count']==2 and support['supported_count']==1 and support['missing_target_or_no_valid_candidate']==1,'Missing candidate removed from support denominator')
    result=dict(passed=True,synthetic_render_calls=1,real_render_calls=0,real_asset_mesh_reads=0,real_image_decodes=0,new_fit_calls=0,
        checks=['finite triangle units/pose','first-hit inter-object occlusion','t_hit optical-axis depth not Euclidean distance',
                'local triangle ids and barycentric/ray anchors','missing means unknown not free space','reference metrics exact and deliberately bad depth rejected','finite candidate missing retains denominator'],
        runtime=runtime(),source_sha256={k:core.sha(ROOT/k) for k in SOURCES},seconds=time.perf_counter()-start,peak_rss_bytes=common.peak_rss(),at=now())
    p=LOG/'component_tests.json'
    if p.exists(): os.replace(p,p.with_name('component_tests.previous.'+uuid.uuid4().hex+'.json'))
    once(p,result)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['inventory','toy','run','review']); args=parser.parse_args()
    core.need(Path.cwd().resolve()==ROOT,'Use explicit project working directory')
    LOG.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); success=False; resources=None; answer={}
    try:
        with project_lock(ROOT):
            if args.action=='inventory': answer=core.inventory(); print(json.dumps(answer,ensure_ascii=False)); success=True
            elif args.action=='toy': answer=toy(); print(json.dumps(dict(passed=True,synthetic_render_calls=1,real_render_calls=0))); success=True
            else:
                inv=verify_authorization(); resources=Resources(args.action)
                globals()[args.action](inv,resources); resources.check(True); success=True
                print(json.dumps(dict(action=args.action,completed=True,real_render_calls=ledger()['returned'])))
    finally:
        if args.action in ('inventory','toy'):
            p=LOG/'preparation_usage.json'; usage=read(p) if p.exists() else dict(seconds=0.,attempts=[])
            seconds=time.perf_counter()-started; usage['seconds']+=seconds
            usage['attempts'].append(dict(action=args.action,seconds=seconds,passed=success,synthetic_render_calls=answer.get('synthetic_render_calls',0),real_render_calls=0,at=now()))
            replace_json(p,usage)
        elif resources is not None and not success: resources.check(True)


if __name__=='__main__': main()

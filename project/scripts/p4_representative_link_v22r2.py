"""Fixed 10-scene x 2-view P4 predicted-instance to grasp interface acceptance."""
import os,sys,json,hashlib,argparse,time,uuid
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scripts import generate_multiview_grasps_v14r1 as generator
from scripts.pipeline import project_lock
CONFIG=ROOT/'configs/p4_representative_link_v22r2.json'
BASE=ROOT/'results/P4/full_baseline_v22r2'
OUT=BASE/'representative_link'
LOG=ROOT/'logs/P4/full_baseline_v22r2/representative_link'
def need(v,m):
    if not v:raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stat(p):
    s=Path(p).stat();return dict(sha256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def now():return datetime.now().astimezone().isoformat()
def once(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():need(read(p)==x,'Immutable report mismatch: '+str(p));return
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def event(kind,**x):
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'calls.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(kind=kind,at=now(),**x))+'\n');f.flush();os.fsync(f.fileno())
def name(r):return f"scene_{r['scene']:04d}_{r['frame']}"
def prepare():
    need(not CONFIG.exists(),'Representative protocol already exists')
    parent=read(ROOT/'configs/p4_full_baseline_v22r2.json');old=read(ROOT/'configs/grasp_multiview_v14r1.json')
    selected=sorted([r for r in parent['development_records'] if r['frame'] in ['0000','0128']],key=lambda r:(r['scene'],r['frame']))
    need(len(selected)==20 and len({r['scene'] for r in selected})==10,'Representative coverage differs')
    protected={p:stat(ROOT/p) for p in old['generation_shared_inputs']}
    for p,h in old['generation_shared_inputs'].items():need(protected[p]['sha256']==h,'Frozen grasp model/numerical source changed')
    for p in ['scripts/p4_representative_link_v22r2.py','scripts/generate_multiview_grasps_v14r1.py','scripts/multiview_common_v14r1.py','scripts/pipeline.py','configs/p4_full_baseline_v22r2.json','configs/p4_complete_evaluation_v22r2.json','configs/grasp_multiview_v14r1.json']:
        protected[p]=stat(ROOT/p)
    for r in selected:
        for kind in ['rgb','depth','meta']:
            p=(Path(parent['data_root'])/r['files'][kind]).as_posix();protected[p]=stat(ROOT/p);need(protected[p]['sha256']==r['sha256'][kind],'Representative raw input changed')
    c=dict(schema=22,created_at=now(),scope='Complete P4 interface acceptance over every development scene and two predetermined views; engineering linkage, not official grasp AP or independent generalization',records=selected,data_root=parent['data_root'],protected=protected,
        generation_protocol=old,source_sha256=sha(Path(__file__)),candidate_policy='Frozen official checkpoint full depth-valid cloud, z in (0,2]m, deterministic sample20000 without replacement; no ground-truth label/workspace filtering; no mask prefilter or re-ranking; preserve all17 columns and order',
        association='actual SA1->SA2->selected seed->sampled valid index->original pixel->predicted RGBD instance; sidecar instance ID is not object-class or whole-gripper ownership. Background remains 0 and is retained.',
        mask_resize='exact rational pixel-center nearest from480x640: original row y maps (2*y+1)//3, original col x maps x//2; agrees with v22r2 AP policy',
        gates=['20/20 fixed frames completed','all candidates preserved and finite valid rotations/geometry','all actual seed/pixel/depth correspondences independently verified','each of10scenes has at least one associated candidate','no ground-truth segmentation used by candidate generator','all sources unchanged'],
        resources=dict(max_forward_intents=24,max_samples=24,per_frame_seconds=300,total_seconds=1200,gpu_bytes=4*1024**3,rss_bytes=8*1024**3,output_bytes=1024**3),
        limitations=['Official pretrained grasp training exposure unknown; engineering-only outputs, not leakage-safe formal main-experiment candidates','No new official geometry labels or grasp AP; every new candidate official success/collision outcome unmeasured','No claim of RGBD advantage or risk-gating benefit','Development subset20train/10dev; no additional held-out, calibration or test access'])
    once(CONFIG,c);once(LOG/'freeze.json',dict(protocol_sha256=sha(CONFIG),source_sha256=sha(Path(__file__)),records=20));print('Representative 20-frame interface protocol frozen; no inference.',flush=True)
def binding(c):
    need(sha(CONFIG)==read(LOG/'freeze.json')['protocol_sha256'] and sha(Path(__file__))==c['source_sha256'],'Representative protocol/source changed')
    need(read(BASE/'training_and_export.json')['preliminary_baseline_usable'],'Selected baseline failed adoption gates')
    need(read(BASE/'evaluation/readonly.json')['status']=='passed','Complete segmentation evaluation/readonly pending')
    expected=dict(c['protected'])
    for p,v in expected.items():need(stat(ROOT/p)==v,'Frozen representative input changed: '+p)
    for p in [BASE/'training_and_export.json',BASE/'selection.json',BASE/'evaluation/metrics.json',BASE/'evaluation/readonly.json']:
        expected[p.relative_to(ROOT).as_posix()]=stat(p)
    for r in c['records']:
        for suffix in ['.npz','.json']:
            p=BASE/'exports/rgbd'/(name(r)+suffix);expected[p.relative_to(ROOT).as_posix()]=stat(p)
    once(LOG/'input_binding.json',expected);return expected
def depth_input(c,r):
    root=ROOT/c['data_root'];m=loadmat(root/r['files']['meta'],variable_names=['intrinsic_matrix','factor_depth'])
    K=np.asarray(m['intrinsic_matrix'],np.float64);factor=float(np.asarray(m['factor_depth']).item())
    with Image.open(root/r['files']['depth']) as im:depth=np.array(im)
    need(depth.shape==(720,1280) and np.isfinite(K).all() and K.shape==(3,3) and factor>0,'Depth/calibration shape invalid')
    return depth,K,factor
def associate(c,r,arrays,bundle,independent=False):
    cand=bundle['candidates'];generator.validate_seed_chain(bundle,arrays)
    ids=arrays['sampled_valid_indices']
    need(ids.dtype==np.int64 and ids.shape==(20000,) and len(np.unique(ids))==20000 and np.all((ids>=0)&(ids<len(arrays['valid_points']))),'Sample identity/bounds differ')
    need(np.array_equal(arrays['sampled_points'],arrays['valid_points'][ids]) and np.array_equal(arrays['sampled_pixel_indices'],arrays['valid_pixel_indices'][ids]),'Sample/valid/raw-pixel chain differs')
    seeds=bundle['sa1_indices'][bundle['sa2_indices']][bundle['selected_seed_indices']]
    if independent:seeds=np.asarray([int(bundle['sa1_indices'][int(bundle['sa2_indices'][int(i)])]) for i in bundle['selected_seed_indices']],np.int64)
    pixels=arrays['sampled_pixel_indices'][seeds]
    cache=BASE/'exports/rgbd'/(name(r)+'.npz');record=read(cache.with_suffix('.json'));need(sha(cache)==record['cache_sha256'],'Instance export digest differs')
    with np.load(cache,allow_pickle=False) as a:mask=a['instances']
    need(mask.shape==(480,640) and mask.dtype==np.int32,'Prediction grid differs')
    if independent:labels=np.asarray([mask[((int(p)//1280)*2+1)//3,(int(p)%1280)//2] for p in pixels],np.int32)
    else:labels=np.repeat(mask,2,axis=1)[(2*np.arange(720)+1)//3].reshape(-1)[pixels].astype(np.int32)
    depth,K,factor=depth_input(c,r);y=pixels//1280;x=pixels%1280;z=depth.reshape(-1)[pixels].astype(np.float64)/factor
    direct=np.column_stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z])
    need(np.array_equal(direct.astype(np.float32),cand[:,13:16]),'Seed depth and candidate translation differ')
    if independent:
        idx=np.arange(depth.size);zz=depth.reshape(-1).astype(np.float64)/factor;cloud=np.column_stack([(idx%1280-K[0,2])*zz/K[0,0],(idx//1280-K[1,2])*zz/K[1,1],zz]);valid=np.isfinite(cloud).all(axis=1)&(zz>0)&(zz<=2.)
        need(np.array_equal(idx[valid],arrays['valid_pixel_indices']) and np.array_equal(cloud[valid].astype(np.float32),arrays['valid_points']),'Independent full raw cloud reconstruction differs')
        inverse=np.stack([x,y,np.ones(len(x))],axis=1)@np.linalg.inv(K).T*z[:,None]
        need(np.allclose(inverse,direct,rtol=0,atol=1e-12),'Independent inverse-camera backprojection differs')
    projected=cand[:,13:16].astype(np.float64)@K.T;error=float(np.max(np.abs(projected[:,:2]/projected[:,2,None]-np.column_stack([x,y])),initial=0))
    need(error<=.001,'Seed reprojection error exceeds .001px')
    return dict(original_candidate_indices=np.arange(len(cand),dtype=np.int64),seed_instance_ids=labels,seed_pixel_indices=pixels,seed_sampled_indices=seeds,seed_valid_indices=arrays['sampled_valid_indices'][seeds]),dict(scene=r['scene'],frame=r['frame'],candidates=len(cand),associated=int(np.count_nonzero(labels)),background=int(np.count_nonzero(labels==0)),max_reprojection_error_pixels=error,all_candidates_preserved=True,ground_truth_used_for_candidate_generation=False,new_official_scores=0,unmeasured_official_outcomes=len(cand))
def guard(c,start):
    need(time.monotonic()-start<c['resources']['per_frame_seconds'],'Representative per-frame wall cap')
    previous=sum(read(p).get('seconds',0) for p in LOG.glob('invocation_*.json'));need(previous+time.monotonic()-start<c['resources']['total_seconds'],'Representative cumulative wall cap')
    total=sum(read(p).get('seconds',0) for folder in ['logs/P4/full_baseline_v22','logs/P4/full_baseline_v22r1','logs/P4/full_baseline_v22r2','logs/P4/full_baseline_v22r2/evaluation','logs/P4/full_baseline_v22r2/representative_link'] for p in (ROOT/folder).glob('invocation_*.json'))
    need(total+time.monotonic()-start<14400,'Whole P4 wall cap')
    need(generator.common.peak_memory_bytes()<c['resources']['rss_bytes'],'Representative RSS cap')
    need(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<c['resources']['output_bytes'],'Representative output cap')
def frame(c,index):
    base=binding(c);r=c['records'][index];directory=OUT/name(r);directory.mkdir(parents=True,exist_ok=True);start=time.monotonic();uid=uuid.uuid4().hex
    if (directory/'complete.json').exists():
        need(all(sha(directory/p)==h for p,h in read(directory/'complete.json')['files'].items()),'Existing frame changed');print('Verified existing frame; skipped.');return
    try:
        depth,K,factor=depth_input(c,r);points=generator.reconstruct_points(depth,K,factor)
        input_file=directory/'input.npz'
        if input_file.exists():
            with np.load(input_file,allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
            need(np.array_equal(arrays['valid_points'],points['valid_points']) and np.array_equal(arrays['valid_pixel_indices'],points['valid_pixel_indices']),'Input recovery differs')
        else:
            event('sampling_intent',id=uid,frame=name(r));arrays,info=generator.sample_full_cloud(points['valid_points'],points['valid_pixel_indices']);need(info['input_status']=='ready','Insufficient depth input')
            np.savez_compressed(input_file,**arrays);once(directory/'sampling.json',info);event('sampling_returned',id=uid,frame=name(r),sha256=sha(input_file))
        bundlefile=directory/'forward_bundle.npz'
        if bundlefile.exists():
            with np.load(bundlefile,allow_pickle=False) as b:bundle={k:b[k] for k in b.files}
            need(sha(bundlefile)==read(directory/'forward_return.json')['sha256'],'Forward recovery digest differs')
        else:
            events=[json.loads(x) for x in (LOG/'calls.jsonl').read_text().splitlines()];need(sum(e['kind']=='forward_intent' for e in events)<c['resources']['max_forward_intents'],'Representative forward cap')
            cumulative=sum(sum(json.loads(x)['kind']=='forward_intent' for x in (ROOT/p/'calls.jsonl').read_text().splitlines()) for p in ['logs/P4/full_baseline_v22','logs/P4/full_baseline_v22r1','logs/P4/full_baseline_v22r2'])+sum(e['kind']=='forward_intent' for e in events)
            need(cumulative<35000,'Whole P4 forward cap');guard(c,start)
            event('forward_intent',id=uid,frame=name(r),input_sha256=sha(input_file));bundle,info=generator.forward_once(ROOT,arrays,c['generation_protocol'],sha)
            np.savez_compressed(bundlefile,**bundle);once(directory/'forward_return.json',dict(sha256=sha(bundlefile),information=info));event('forward_returned',id=uid,frame=name(r),sha256=sha(bundlefile))
        sidecar,report=associate(c,r,arrays,bundle);np.savez_compressed(directory/'association.npz',**sidecar);once(directory/'report.json',report);guard(c,start)
        need(all(stat(ROOT/p)==v for p,v in base.items()),'Representative original input modified')
        once(directory/'complete.json',dict(status='complete',files={p.name:sha(p) for p in directory.iterdir() if p.is_file()}));once(LOG/f'invocation_{uid}.json',dict(status='passed',seconds=time.monotonic()-start,frame=name(r)));print(json.dumps(report),flush=True)
    except BaseException as e:
        once(LOG/f'invocation_{uid}.json',dict(status='failed_preserved',seconds=time.monotonic()-start,frame=name(r),error=repr(e)));raise
def verify(c):
    base=binding(c);before={p.relative_to(ROOT).as_posix():stat(p) for p in OUT.rglob('*') if p.is_file()};rows=[]
    for r in c['records']:
        d=OUT/name(r);complete=read(d/'complete.json');need(all(sha(d/p)==h for p,h in complete['files'].items()),'Candidate frame changed')
        with np.load(d/'input.npz') as a:arrays={k:a[k] for k in a.files}
        with np.load(d/'forward_bundle.npz') as b:bundle={k:b[k] for k in b.files}
        ann,row=associate(c,r,arrays,bundle,True)
        with np.load(d/'association.npz') as actual:need(set(actual.files)==set(ann) and all(np.array_equal(actual[k],v) for k,v in ann.items()),'Independent sidecar differs')
        need(row==read(d/'report.json'),'Independent linkage report differs');rows.append(row)
    scenes=sorted({r['scene'] for r in rows});usable=all(sum(r['associated'] for r in rows if r['scene']==s)>0 for s in scenes)
    need(all(stat(ROOT/p)==v for p,v in {**base,**before}.items()),'Read-only source/candidate alteration')
    events=[json.loads(x) for x in (LOG/'calls.jsonl').read_text().splitlines()];intents={e['id'] for e in events if e['kind']=='forward_intent'};returned={e['id'] for e in events if e['kind']=='forward_returned'}
    result=dict(status='accepted_representative_engineering_linkage' if usable else 'valid_negative_unusable_linkage',frames=len(rows),scenes=len(scenes),candidates=sum(r['candidates'] for r in rows),associated=sum(r['associated'] for r in rows),background=sum(r['background'] for r in rows),all_candidates_preserved=True,all_frame_reports_independently_verified=True,all_originals_unchanged=True,each_scene_has_associated_candidate=usable,known_model_returns=len(returned),unknown_model_outcomes=len(intents-returned),exact_model_call_count=len(intents) if not intents-returned else None,new_official_scores=0,new_training_updates=0,rows=rows,limitations=c['limitations'])
    once(OUT/'result.json',result);once(OUT/'readonly.json',dict(status='passed',at=now(),verified_frames=20,verified_candidates=result['candidates'],zero_new_model_calls=True,zero_new_sampling=True,zero_new_scores=True,all_originals_unchanged=True,result_sha256=sha(OUT/'result.json')));print(json.dumps({k:v for k,v in result.items() if k not in ['rows','limitations']}),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','frame','verify']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:
        c=read(CONFIG)
        with project_lock(ROOT):
            if a.mode=='verify':verify(c)
            else:need(a.index is not None and 0<=a.index<20,'Invalid selected frame');frame(c,a.index)
if __name__=='__main__':main()
